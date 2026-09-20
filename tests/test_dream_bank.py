"""dream_bank (ADR 0018): routed sets, the anchor penalty, calibration, and
the payload round trip. Runs where torch is (skips in the client env)."""

from __future__ import annotations

import unittest

from rlstack.policy.adapters.dream_bank import (
    BASE, DREAMER, Route, check_route, memory_route, set_routes,
)
from rlstack.policy.adapters.rollout import Request
from rlstack.registry import ADAPTER_TYPES

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from test_batched_replay import PATH, SITES, a_model
    from rlstack.policy.adapters import dream_bank_torch as bank, lora_torch
    from rlstack.policy.adapters.replay import ReplayRows, row_plan

needs_torch = unittest.skipUnless(torch is not None, "torch required")


class RouteVocabularyTest(unittest.TestCase):
    def test_sets_are_the_dreamer_then_the_memories_in_order(self):
        self.assertEqual(set_routes(2), ("dreamer", "memory:00", "memory:01"))
        self.assertEqual(memory_route(7), "memory:07")

    def test_a_route_names_a_set_or_the_base(self):
        check_route(BASE, 0)
        check_route("memory:01", 2)
        with self.assertRaises(ValueError):
            check_route("memory:02", 2)

    def test_the_directive_records_its_route_and_defaults_to_the_dreamer(self):
        adapter = ADAPTER_TYPES.get("dream_bank").instance
        request = Request(token_ids=(1, 2), directives=(Route("memory:03"),))
        self.assertEqual(adapter.record_directive(adapter.directive_for(request), request),
                         {"route": "memory:03"})
        bare = Request(token_ids=(1,))
        self.assertEqual(adapter.record_directive(adapter.directive_for(bare), bare),
                         {"route": DREAMER})


@needs_torch
class DreamBankTorchTest(unittest.TestCase):
    def setUp(self):
        self.model = a_model().requires_grad_(False)
        self.state = bank.build(SITES, {"r": 4, "memories": 2, "seed": 3, "lam": 0.5})
        for route in set_routes(2):                      # B = 0 would hide every set
            generator = torch.Generator().manual_seed(hash(route) % 1000)
            self.state.sets[route].b[PATH].data = torch.randn(6, 4, generator=generator) / 4
        bank.install(self.model, self.state)
        self.x = torch.randn(3, 5, 8)

    def rows(self, routes):
        return ReplayRows(slots=({PATH: self.state},),
                          index=torch.zeros(len(routes), dtype=torch.long),
                          facts=tuple(({"route": r},) if r is not None else ({},)
                                      for r in routes))

    def forward(self, routes):
        with row_plan(self.model).route(self.rows(routes)):
            return self.model.block.proj(self.x)

    def one_set(self, route, x):
        return self.model.block.proj.inner(x) + lora_torch._whole_batch_delta(
            x, self.state.sets[route], PATH)

    def test_each_row_gets_its_own_set_and_a_base_row_gets_none(self):
        out = self.forward(["memory:01", BASE, DREAMER])
        self.assertTrue(torch.allclose(out[0], self.one_set("memory:01", self.x[0:1])[0]))
        self.assertTrue(torch.allclose(out[1], self.model.block.proj.inner(self.x[1:2])[0]))
        self.assertTrue(torch.allclose(out[2], self.one_set(DREAMER, self.x[2:3])[0]))

    def test_a_row_with_no_route_fact_is_the_dreamer(self):
        uniform = self.forward([None, None, None])
        self.assertTrue(torch.allclose(uniform, self.one_set(DREAMER, self.x)))

    def test_an_unknown_route_is_refused(self):
        with self.assertRaises(ValueError):
            self.forward(["memory:07", DREAMER, DREAMER])

    def test_the_penalty_equals_the_dense_quadratic_form_at_full_rank(self):
        d_in = 8
        c = torch.randn(d_in, d_in); c = c @ c.T + torch.eye(d_in)
        values, vectors = torch.linalg.eigh(c)
        anchor = bank.Anchor(u={PATH: vectors}, e={PATH: values}, delta={PATH: 0.0})
        lora = self.state.sets["memory:00"]
        dense = lora.b[PATH] @ lora.a[PATH]
        expected = 0.5 * torch.trace(dense @ c @ dense.T)
        got = bank.set_penalty(lora, anchor, (PATH,))
        self.assertTrue(torch.allclose(got, expected, atol=1e-5), (got, expected))

    def test_the_penalty_gradient_matches_finite_differences(self):
        d_in = 8
        c = torch.randn(d_in, d_in, dtype=torch.float64); c = c @ c.T
        values, vectors = torch.linalg.eigh(c)
        k = 3
        anchor = bank.Anchor(u={PATH: vectors[:, -k:]}, e={PATH: values[-k:]},
                             delta={PATH: float(values[:-k].mean())})
        lora = self.state.sets["memory:00"]
        lora.a[PATH].data = lora.a[PATH].data.double()
        lora.b[PATH].data = lora.b[PATH].data.double()
        loss = bank.set_penalty(lora, anchor, (PATH,))
        (grad,) = torch.autograd.grad(loss, lora.b[PATH])
        eps = 1e-6
        b = lora.b[PATH]
        with torch.no_grad():
            b[0, 0] += eps
            up = bank.set_penalty(lora, anchor, (PATH,))
            b[0, 0] -= 2 * eps
            down = bank.set_penalty(lora, anchor, (PATH,))
            b[0, 0] += eps
        self.assertAlmostEqual(float(grad[0, 0]), float((up - down) / (2 * eps)), places=5)

    def test_provide_prices_only_the_memories_and_watches_both_norms(self):
        provided = bank.provide(self.state)
        self.assertEqual(float(provided["anchor_penalty"]), 0.0)   # no anchor yet
        self.state.anchor = bank.Anchor(u={PATH: torch.eye(8)}, e={PATH: torch.ones(8)},
                                        delta={PATH: 0.0})
        priced = bank.provide(self.state)
        expected = 0.5 * sum(bank.set_penalty(self.state.sets[r], self.state.anchor, (PATH,))
                             for r in ("memory:00", "memory:01"))
        self.assertTrue(torch.allclose(priced["anchor_penalty"], expected))
        self.assertGreater(float(priced["dreamer_delta_norm"]), 0.0)
        self.assertGreater(float(priced["memory_delta_norm"]), 0.0)

    def test_calibrate_accumulates_and_factorizes_the_seen_covariance(self):
        state = bank.build(SITES, {"r": 2, "memories": 0, "mode": "calibrate", "anchor_rank": 3})
        model = a_model().requires_grad_(False)
        bank.install(model, state)
        x = torch.randn(4, 7, 8)
        with row_plan(model).route(ReplayRows(slots=({PATH: state},),
                                              index=torch.zeros(4, dtype=torch.long))):
            model.block.proj(x)
        flat = x.reshape(-1, 8)
        self.assertEqual(state.count, flat.shape[0])
        self.assertTrue(torch.allclose(state.sums[PATH], flat.T @ flat, atol=1e-4))
        anchor = bank.factorize(state)
        self.assertEqual(tuple(anchor.u[PATH].shape), (8, 3))
        self.assertTrue(torch.all(anchor.e[PATH][:-1] >= anchor.e[PATH][1:]))   # descending
        self.assertGreater(anchor.captured[PATH], 0.0)
        payload = bank.emit(state)
        fresh = bank.build(SITES, {"r": 4, "memories": 2, "seed": 9})
        with self.assertRaises(ValueError):          # rank differs: refused, not coerced
            bank.load(fresh, payload)
        stream = bank.build(SITES, {"r": 2, "memories": 2, "seed": 9})
        bank.load(stream, payload)                    # the warm start: anchor arrives
        self.assertIsNotNone(stream.anchor)
        self.assertEqual(tuple(stream.anchor.u[PATH].shape), (8, 3))

    def test_emit_load_round_trips_every_set_and_the_anchor(self):
        self.state.anchor = bank.Anchor(u={PATH: torch.randn(8, 3)}, e={PATH: torch.rand(3)},
                                        delta={PATH: 0.25}, captured={PATH: 0.9}, count=11)
        payload = bank.emit(self.state)
        other = bank.build(SITES, {"r": 4, "memories": 2, "seed": 77})
        bank.load(other, payload)
        for route in set_routes(2):
            self.assertTrue(torch.equal(other.sets[route].a[PATH], self.state.sets[route].a[PATH]))
            self.assertTrue(torch.equal(other.sets[route].b[PATH], self.state.sets[route].b[PATH]))
        self.assertEqual(other.anchor.delta[PATH], 0.25)
        self.assertEqual(other.anchor.count, 11)
        self.assertTrue(torch.allclose(other.anchor.u[PATH].float(),
                                       self.state.anchor.u[PATH].to(torch.bfloat16).float()))
        meta, fragments = bank.split_sets(payload)
        self.assertEqual(sorted(fragments), sorted(set_routes(2)))
        self.assertNotIn("anchor", fragments)
        self.assertIn(f"{lora_torch.PEFT_PREFIX}{PATH}.lora_A.weight", fragments[DREAMER])

    def test_with_no_memories_the_dreamer_is_a_lora_in_bytes(self):
        plain = lora_torch.build(SITES, {"r": 4, "seed": bank._set_seed(3, DREAMER)})
        state = bank.build(SITES, {"r": 4, "memories": 0, "seed": 3})
        self.assertTrue(torch.equal(plain.a[PATH], state.sets[DREAMER].a[PATH]))
        _, fragments = bank.split_sets(bank.emit(state))
        from safetensors.torch import load as st_load
        self.assertTrue(torch.equal(st_load(lora_torch.emit(plain))[f"{lora_torch.PEFT_PREFIX}{PATH}.lora_A.weight"],
                                    fragments[DREAMER][f"{lora_torch.PEFT_PREFIX}{PATH}.lora_A.weight"]))

    def test_param_groups_are_one_per_route(self):
        """ADR 0019: the dreamer and EACH memory is its own named group."""
        groups = bank.param_groups(self.state)
        self.assertEqual(sorted(groups), ["dreamer", "memory:00", "memory:01"])
        self.assertTrue(all(len(group) == 2 for group in groups.values()))
