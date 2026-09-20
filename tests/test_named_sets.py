"""ADR 0019, the learner's side: named sets, library sets, stacked routes, the
no-grad forward and the per-step learning-rate scales.

Three layers. The route GRAMMAR and the FAKE learner run everywhere (stdlib):
every new verb is deterministic, crosses both doors, and leaves a run that
never says `load_set` byte-identical. The TORCH layer (skips where torch is
not) proves the numbers on a one-site model and a toy LM: a stacked row gets
the sum of its parts' deltas, the gradient reaches the dreamer and never the
library, `load_set(None)` is the set's own init with no moments, emit_set ->
load_set is exact, `forward` is the loss path's per-document NLL, and an
lr scale touches one group for one step.
"""

from __future__ import annotations

import tempfile
import unittest

import rlstack  # noqa: F401  (registers the adapter types and the losses)
from common import arith_store
from rlstack import (
    FakeEngine, FakeLearner, Host, HostService, LocalTransport, Regime,
    RemoteLearner,
)
from rlstack.data.flatten import TokenBatch
from rlstack.policy.adapters.dream_bank import (
    BASE, DREAMER, check_route, check_trains_at_most_one, is_library,
    library_name, library_names_of, library_route, parts_of, route_of,
)
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.fakes import FAKE_SET
from rlstack.runner.interfaces import (
    EntryInstall, OptimSettings, Parameterization, check_lr_scales, lr_scale_of,
)
from rlstack.runner.remote import LEARNER_VERBS, LearnerService
from rlstack.spec.canonical import content_hash

try:
    import torch
except ImportError:                                  # the client environment
    torch = None

if torch is not None:
    from test_batched_replay import PATH, SITES, a_model
    from rlstack.policy.adapters import dream_bank_torch as bank, lora_torch
    from rlstack.policy.adapters.replay import ReplayRows, row_plan
    from rlstack.runner.learners.torch_learner import TorchLearner

needs_torch = unittest.skipUnless(torch is not None, "torch required")

WIDTH, VOCAB = 8, 23
SITE = SiteMeta(name="proj", path="proj", has_weight=True,
                shape=(WIDTH, WIDTH), is_boundary=False)


def a_bank(memories: int = 2, lr: float = 1e-2, overrides=None,
           adapter_type: str = "dream_bank") -> Parameterization:
    """One trainable dream_bank entry `pi` with `memories` lanes, under sft."""
    init = {"r": 2, "seed": 5}
    if adapter_type == "dream_bank":
        init["memories"] = memories
    return Parameterization(
        base="toy", loss="sft",
        entries=(EntryInstall(name="pi", adapter_type=adapter_type, init=init,
                              trainable=True, sites=(SITE,)),),
        optim=OptimSettings(name="adamw", lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0, overrides=overrides or {}))


def a_batch(docs, routes=None) -> TokenBatch:
    """Documents packed flat; every token after a document's first is
    trainable; `routes` stamps each document's `route` turn fact."""
    flat = [t for doc in docs for t in doc]
    starts, mask, at = [], [], 0
    for doc in docs:
        starts.append(at)
        mask += [0] + [1] * (len(doc) - 1)
        at += len(doc)
    extras = () if routes is None else tuple(
        ({},) if route is None else ({"route": route},) for route in routes)
    return TokenBatch(token_ids=tuple(flat), loss_mask=tuple(mask),
                      behavior_logprobs=(0.0,) * len(flat),
                      segment_ids=(0,) * len(flat), doc_starts=tuple(starts),
                      doc_turn_extras=extras)


DOCS = [[3, 1, 4, 1, 5], [9, 2, 6], [6, 5, 3, 5]]


# ---------------------------------------------------------------------------
# the route grammar
# ---------------------------------------------------------------------------

class RouteGrammarTest(unittest.TestCase):
    def test_a_route_has_one_part_or_two(self):
        self.assertEqual(parts_of("dreamer"), ("dreamer",))
        self.assertEqual(parts_of("lib:topics/t07.m3+dreamer"),
                         ("lib:topics/t07.m3", "dreamer"))
        self.assertEqual(parts_of("memory:01"), ("memory:01",))

    def test_a_library_part_carries_its_store_name(self):
        self.assertTrue(is_library("lib:a/b"))
        self.assertFalse(is_library("memory:00"))
        self.assertEqual(library_name(library_route("a/b-1.x")), "a/b-1.x")
        with self.assertRaises(ValueError):
            library_name("dreamer")

    def test_the_accepted_routes(self):
        for route in ("dreamer", "memory:01", BASE, "lib:m", "lib:a/b_c.d-e",
                      "lib:m+dreamer", "dreamer+lib:m", "lib:m+memory:00",
                      "lib:m+lib:n"):
            check_route(route, 2)

    def test_the_refused_routes(self):
        for route, why in (
                ("memory:02", "names no set"),
                ("lib:", "not an adapter name"),
                ("lib:/abs", "not an adapter name"),
                ("lib:a/../b", "not an adapter name"),
                ("lib:a b", "not an adapter name"),
                ("lib:a+lib:b+dreamer", "exactly two"),
                ("lib:m+lib:m", "on itself"),
                ("lib:m+base", "no delta"),
                ("lib:m+", "names no set")):
            with self.assertRaises(ValueError, msg=route) as caught:
                check_route(route, 2)
            self.assertIn(why, str(caught.exception), route)

    def test_a_training_row_trains_at_most_one_part(self):
        """The grammar serves `dreamer+memory:00`; a TRAINING row may not
        carry it — a row's gradient belongs to one set."""
        check_route("dreamer+memory:00", 2)
        for route in ("dreamer", BASE, "lib:m+dreamer", "lib:m+lib:n"):
            check_trains_at_most_one(route)
        with self.assertRaises(ValueError) as caught:
            check_trains_at_most_one("dreamer+memory:00")
        self.assertIn("at most one part", str(caught.exception))
        self.assertEqual(library_names_of("lib:a/b+dreamer"), ("a/b",))
        self.assertEqual(library_names_of("lib:a+lib:b"), ("a", "b"))
        self.assertEqual(library_names_of("memory:00"), ())

    def test_a_row_with_no_route_fact_is_the_dreamers(self):
        self.assertEqual(route_of(({}, {"route": "lib:m+dreamer"})), "lib:m+dreamer")
        self.assertEqual(route_of(({},)), DREAMER)


class LrScaleRuleTest(unittest.TestCase):
    def test_specific_wins_over_general(self):
        scales = {"pi": 0.5, "memory:00": 0.25, "pi.memory:01": 0.125}
        self.assertEqual(lr_scale_of(scales, "pi", "memory:00"), 0.25)
        self.assertEqual(lr_scale_of(scales, "pi", "memory:01"), 0.125)
        self.assertEqual(lr_scale_of(scales, "pi", "dreamer"), 0.5)
        self.assertEqual(lr_scale_of(scales, "pi", ""), 0.5)
        self.assertEqual(lr_scale_of({}, "pi", "dreamer"), 1.0)

    def test_a_key_that_reaches_no_group_is_refused(self):
        check_lr_scales({"memory:00": 0.5, "pi": 1.0, "pi.dreamer": 2.0},
                        {"pi": ["dreamer", "memory:00"]})
        with self.assertRaises(ValueError) as caught:
            check_lr_scales({"memory:07": 0.5}, {"pi": ["dreamer", "memory:00"]})
        self.assertIn("memory:07", str(caught.exception))


# ---------------------------------------------------------------------------
# the fake learner
# ---------------------------------------------------------------------------

class FakeNamedSetsTest(unittest.TestCase):
    def setUp(self):
        self.learner = FakeLearner()
        self.learner.install("t", a_bank())

    def test_a_run_that_never_says_load_set_is_byte_identical(self):
        """The digest folds exactly what it did before ADR 0019: the default
        step is `{state, step}`, and the named-set verbs never touch it."""
        other = FakeLearner()
        other.install("t", a_bank())
        for learner, scales in ((self.learner, None), (other, {})):
            learner.forward_backward("t", a_batch(DOCS))
            learner.optim_step("t", scales) if scales is not None else learner.optim_step("t")
        self.assertEqual(self.learner.emit("t"), other.emit("t"))
        before = self.learner.emit("t")
        self.learner.load_set("t", "pi", "memory:00", None)
        self.learner.load_set("t", "pi", "lib:m", b"anything")
        self.learner.forward("t", a_batch(DOCS))
        self.learner.emit_set("t", "pi", "memory:00")
        self.assertEqual(self.learner.emit("t"), before)
        init = self.learner._tenants["t"].init
        fresh = FakeLearner()
        fresh.install("t", a_bank())
        fresh.optim_step("t")
        self.assertEqual(fresh._tenants["t"].state,
                         content_hash({"state": init, "step": 1}))

    def test_emit_set_then_load_set_round_trips_exactly(self):
        self.learner.load_set("t", "pi", "memory:00", None)
        self.learner.forward_backward("t", a_batch(DOCS, ["memory:00"] * 3))
        self.learner.optim_step("t")
        payload = self.learner.emit_set("t", "pi", "memory:00")
        self.assertTrue(payload.startswith(FAKE_SET))
        self.learner.load_set("t", "pi", "memory:01", payload)
        self.assertEqual(self.learner.emit_set("t", "pi", "memory:01"), payload)

    def test_load_set_none_is_the_sets_own_init_and_resets_its_moments(self):
        self.learner.load_set("t", "pi", "memory:00", None)
        init = self.learner.emit_set("t", "pi", "memory:00")
        self.learner.forward_backward("t", a_batch(DOCS, ["memory:00"] * 3))
        self.learner.optim_step("t")
        self.assertNotEqual(self.learner.emit_set("t", "pi", "memory:00"), init)
        self.learner.forward_backward("t", a_batch(DOCS, ["memory:00"] * 3))
        self.learner.load_set("t", "pi", "memory:00", None)   # mid-accumulation
        held = self.learner._tenants["t"].sets[("pi", "memory:00")]
        self.assertEqual((held.steps, held.pending), (0, None))
        self.assertEqual(self.learner.emit_set("t", "pi", "memory:00"), init)
        self.learner.load_set("t", "pi", "memory:01", None)
        self.assertNotEqual(self.learner.emit_set("t", "pi", "memory:01"), init)

    def test_forward_is_deterministic_per_document_and_trains_nothing(self):
        self.learner.load_set("t", "pi", "memory:00", None)
        batch = a_batch(DOCS, ["memory:00", "memory:00", BASE])
        before = self.learner.emit_set("t", "pi", "memory:00")
        first = self.learner.forward("t", batch)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(isinstance(nll, float) for nll in first))
        self.assertEqual(self.learner.forward("t", batch), first)
        self.assertEqual(self.learner.emit_set("t", "pi", "memory:00"), before)
        self.learner.forward_backward("t", batch)
        self.learner.optim_step("t")
        after = self.learner.forward("t", batch)
        self.assertNotEqual(after[0], first[0])       # its set moved
        self.assertEqual(after[2], first[2])          # the base never does

    def test_a_stacked_forward_reads_the_library_and_a_missing_one_is_refused(self):
        batch = a_batch(DOCS[:1], ["lib:m+dreamer"])
        with self.assertRaises(ValueError) as caught:
            self.learner.forward("t", batch)
        self.assertIn("lib:m", str(caught.exception))
        with self.assertRaises(ValueError):
            self.learner.forward_backward("t", batch)
        self.learner.load_set("t", "pi", "lib:m", b"one")
        one = self.learner.forward("t", batch)
        self.learner.load_set("t", "pi", "lib:m", b"two")
        self.assertNotEqual(self.learner.forward("t", batch), one)
        self.assertNotEqual(self.learner.forward("t", a_batch(DOCS[:1], ["dreamer"])), one)

    def test_a_library_is_loaded_with_bytes_and_only_a_library_is_dropped(self):
        with self.assertRaises(ValueError):
            self.learner.load_set("t", "pi", "lib:m", None)
        self.learner.load_set("t", "pi", "lib:m", b"one")
        self.learner.drop_set("t", "pi", "lib:m")
        self.learner.drop_set("t", "pi", "lib:m")              # already gone
        with self.assertRaises(ValueError):
            self.learner.forward("t", a_batch(DOCS[:1], ["lib:m"]))
        with self.assertRaises(ValueError):
            self.learner.drop_set("t", "pi", "memory:00")

    def test_the_set_verbs_take_one_set_of_a_known_entry(self):
        for route in (BASE, "lib:m+dreamer", "memory:02", "nonsense"):
            with self.assertRaises(ValueError, msg=route):
                self.learner.load_set("t", "pi", route, None)
        with self.assertRaises(KeyError):
            self.learner.emit_set("t", "nobody", "dreamer")

    def test_lr_scales_are_recorded_and_scale_their_own_lane_only(self):
        def fit(scales):
            learner = FakeLearner()
            learner.install("t", a_bank())
            for lane in ("memory:00", "memory:01"):
                learner.load_set("t", "pi", lane, None)
            learner.forward_backward("t", a_batch(DOCS[:2], ["memory:00", "memory:01"]))
            learner.optim_step("t", scales)
            return learner
        plain, scaled = fit(None), fit({"memory:00": 0.5})
        self.assertEqual(plain.lr_scales_of("t"), [{}])
        self.assertEqual(scaled.lr_scales_of("t"), [{"memory:00": 0.5}])
        self.assertNotEqual(plain.emit_set("t", "pi", "memory:00"),
                            scaled.emit_set("t", "pi", "memory:00"))
        self.assertEqual(plain.emit_set("t", "pi", "memory:01"),
                         scaled.emit_set("t", "pi", "memory:01"))
        self.assertEqual(fit({"memory:00": 1.0}).emit_set("t", "pi", "memory:00"),
                         plain.emit_set("t", "pi", "memory:00"))

    def test_k_lanes_in_one_tenant_are_k_tenants_exactly(self):
        """The ADR's promise, as the fakes prove it: a lane's result depends
        on its start, its own documents in order, its steps and its scales —
        not on who shares the tenant, the batch, or which lane it sits in."""
        start = FAKE_SET + b"0" * 64
        docs = {"memory:00": [[3, 1, 4], [1, 5, 9, 2]], "memory:01": [[6, 5], [3, 5, 8]]}
        scales = {"memory:00": 0.75, "memory:01": 0.25}
        shared = FakeLearner()
        shared.install("t", a_bank())
        for lane in docs:
            shared.load_set("t", "pi", lane, start)
        for step in range(2):                     # the lanes' documents interleaved
            shared.forward_backward("t", a_batch(
                [docs["memory:00"][step], docs["memory:01"][step]],
                ["memory:00", "memory:01"]))
            shared.optim_step("t", scales)
        for lane in docs:
            alone = FakeLearner()
            alone.install("solo", a_bank(memories=1))
            alone.load_set("solo", "pi", "memory:00", start)
            for step in range(2):
                alone.forward_backward("solo", a_batch([docs[lane][step]], ["memory:00"]))
                alone.optim_step("solo", {"memory:00": scales[lane]})
            self.assertEqual(shared.emit_set("t", "pi", lane),
                             alone.emit_set("solo", "pi", "memory:00"), lane)

    def test_a_lane_with_no_documents_does_not_step(self):
        for lane in ("memory:00", "memory:01"):
            self.learner.load_set("t", "pi", lane, None)
        idle = self.learner.emit_set("t", "pi", "memory:01")
        self.learner.forward_backward("t", a_batch(DOCS, ["memory:00"] * 3))
        self.learner.optim_step("t")
        self.assertEqual(self.learner.emit_set("t", "pi", "memory:01"), idle)

    def test_an_untouched_set_emits_what_the_tenant_trained(self):
        before = self.learner.emit_set("t", "pi", DREAMER)
        self.learner.forward_backward("t", a_batch(DOCS))
        self.learner.optim_step("t")
        self.assertNotEqual(self.learner.emit_set("t", "pi", DREAMER), before)


class NewVerbsCrossTheDoorsTest(unittest.TestCase):
    """The new verbs cross the resident door and the host door exactly as
    `load` / `emit` / `forward_backward` do: what the wire carries equals
    what the object does."""

    def drive(self, learner):
        learner.install("t", a_bank())
        learner.load_set("t", "pi", "memory:00", None)
        learner.load_set("t", "pi", "lib:topics/m.3", b"\x00\xffraw bytes")
        batch = a_batch(DOCS, ["lib:topics/m.3+memory:00", "memory:00", BASE])
        learner.forward_backward("t", batch)
        learner.optim_step("t", {"memory:00": 0.5})
        learner.optim_step("t")
        payload = learner.emit_set("t", "pi", "memory:00")
        learner.load_set("t", "pi", "memory:01", payload)
        nll = learner.forward("t", batch)
        learner.drop_set("t", "pi", "lib:topics/m.3")
        return payload, nll, learner.emit_set("t", "pi", "memory:01"), learner.emit("t")

    def test_the_wire_names_every_new_verb(self):
        for verb in ("forward", "load_set", "drop_set", "emit_set"):
            self.assertIn(verb, LEARNER_VERBS)

    def test_through_the_resident_door(self):
        served = FakeLearner()
        proxy = RemoteLearner(LocalTransport(LearnerService(served)))
        direct = self.drive(FakeLearner())
        self.assertEqual(self.drive(proxy), direct)
        self.assertIsInstance(direct[0], bytes)
        self.assertIsInstance(direct[1], tuple)
        self.assertEqual(served.lr_scales_of("t"), [{"memory:00": 0.5}, {}])

    def test_through_a_host_door_admitted(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, _, _ = arith_store(tmp.name)
        host = Host("alt", engines=(FakeEngine(base="toy"),),
                    learner=FakeLearner(), store=store,
                    regimes=(Regime("serve", "inference", "toy", 1),
                             Regime("train", "training", "toy", 1)))
        routed = RemoteLearner(LocalTransport(HostService(host)), admitted=True)
        self.assertEqual(self.drive(routed), self.drive(FakeLearner()))
        self.assertEqual(host.arbiter.in_flight(), 0)

    def test_a_frame_from_before_lr_scales_is_a_plain_step(self):
        served, direct = FakeLearner(), FakeLearner()
        for learner in (served, direct):
            learner.install("t", a_bank())
        LearnerService(served).answer("optim_step", {"tenant": "t"})
        direct.optim_step("t")
        self.assertEqual(served.emit("t"), direct.emit("t"))


# ---------------------------------------------------------------------------
# torch: the sites
# ---------------------------------------------------------------------------

@needs_torch
class StackedSiteTest(unittest.TestCase):
    def setUp(self):
        self.model = a_model().requires_grad_(False)
        self.state = bank.build(SITES, {"r": 4, "memories": 2, "seed": 3, "lam": 0.5})
        for index, route in enumerate(bank.set_routes(2)):   # B = 0 would hide every set
            generator = torch.Generator().manual_seed(index)
            self.state.sets[route].b[PATH].data = torch.randn(6, 4, generator=generator) / 4
        bank.install(self.model, self.state)
        self.library = self.a_lora(seed=11, r=4)
        bank.load_set(self.state, "lib:m", lora_torch.emit(self.library), None)
        self.x = torch.randn(3, 5, 8)

    def a_lora(self, seed, r):
        lora = lora_torch.build(SITES, {"r": r, "seed": seed})
        generator = torch.Generator().manual_seed(seed)
        lora.b[PATH].data = torch.randn(6, r, generator=generator) / r
        return lora

    def forward(self, routes, x=None):
        rows = ReplayRows(slots=({PATH: self.state},),
                          index=torch.zeros(len(routes), dtype=torch.long),
                          facts=tuple(({"route": r},) for r in routes))
        with row_plan(self.model).route(rows):
            return self.model.block.proj(self.x if x is None else x)

    def delta(self, lora, x):
        return lora_torch._whole_batch_delta(x, lora, PATH)

    def inner(self, x):
        return self.model.block.proj.inner(x)

    def test_a_stacked_row_gets_the_sum_of_its_parts_deltas(self):
        want = (self.inner(self.x) + self.delta(self.library, self.x)
                + self.delta(self.state.sets[DREAMER], self.x))
        self.assertTrue(torch.allclose(self.forward(["lib:m+dreamer"] * 3), want, atol=1e-6))
        self.assertTrue(torch.allclose(self.forward(["dreamer+lib:m"] * 3), want, atol=1e-6))

    def test_a_mixed_batch_gives_each_row_its_own_parts(self):
        out = self.forward(["lib:m+memory:01", "lib:m", DREAMER])
        x = self.x
        want0 = (self.inner(x[0:1]) + self.delta(self.library, x[0:1])
                 + self.delta(self.state.sets["memory:01"], x[0:1]))
        want1 = self.inner(x[1:2]) + self.delta(self.library, x[1:2])
        want2 = self.inner(x[2:3]) + self.delta(self.state.sets[DREAMER], x[2:3])
        for got, want in zip(out, (want0, want1, want2)):
            self.assertTrue(torch.allclose(got, want[0], atol=1e-6))
        mixed = self.forward(["lib:m+dreamer", BASE, "memory:00"])
        self.assertTrue(torch.allclose(mixed[1], self.inner(x[1:2])[0]))

    def test_existing_routes_keep_their_exact_numbers(self):
        """The single-route fast path is LoRA's own expression, bit for bit."""
        for route in (DREAMER, "memory:01"):
            want = self.inner(self.x) + self.delta(self.state.sets[route], self.x).to(self.x.dtype)
            self.assertTrue(torch.equal(self.forward([route] * 3), want))

    def test_the_gradient_reaches_the_dreamer_and_never_the_library(self):
        x = self.x.clone().requires_grad_(True)
        self.forward(["lib:m+dreamer", "lib:m+dreamer", "lib:m"], x).pow(2).sum().backward()
        dreamer, library = self.state.sets[DREAMER], self.state.libraries["lib:m"]
        self.assertGreater(float(dreamer.a[PATH].grad.abs().sum()), 0.0)
        self.assertGreater(float(dreamer.b[PATH].grad.abs().sum()), 0.0)
        for parameter in library.parameters():
            self.assertFalse(parameter.requires_grad)
            self.assertIsNone(parameter.grad)
        self.assertIsNone(self.state.sets["memory:00"].a[PATH].grad)

    def test_the_gradient_flows_through_the_library_to_the_layers_below(self):
        """Detached FACTORS, not a detached delta: d(out)/dx carries the
        library's B A, or a dreamer delta in an earlier layer would train
        against a model the row never ran under."""
        x = self.x[:1].clone().requires_grad_(True)
        self.forward(["lib:m"], x).sum().backward()
        a, b = self.library.a[PATH], self.library.b[PATH]
        weight = self.model.block.proj.inner.weight
        want = (weight + b @ a).sum(0).expand_as(x)
        self.assertTrue(torch.allclose(x.grad, want, atol=1e-5))

    def test_a_library_nobody_loaded_is_refused_at_the_forward(self):
        with self.assertRaises(ValueError) as caught:
            self.forward(["lib:absent+dreamer"] * 3)
        self.assertIn("lib:absent", str(caught.exception))
        self.assertIn("lib:m", str(caught.exception))

    def test_a_training_row_under_two_trainable_sets_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.forward(["dreamer+memory:00"] * 3)
        self.assertIn("at most one part", str(caught.exception))

    def test_loading_a_library_again_replaces_it(self):
        """A resumed Trainer reloads its libraries: the same bytes are the
        same set, other bytes are the new set."""
        bank.load_set(self.state, "lib:m", lora_torch.emit(self.library), None)
        self.assertEqual(sorted(self.state.libraries), ["lib:m"])
        self.assertEqual(bank.emit_set(self.state, "lib:m"), lora_torch.emit(self.library))
        other = self.a_lora(seed=21, r=4)
        bank.load_set(self.state, "lib:m", lora_torch.emit(other), None)
        want = self.inner(self.x) + self.delta(other, self.x)
        self.assertTrue(torch.allclose(self.forward(["lib:m"] * 3), want, atol=1e-6))

    def test_a_library_is_in_no_group_norm_penalty_or_payload(self):
        owned = {id(p) for p in self.state.parameters()}
        grouped = {id(p) for group in bank.param_groups(self.state).values() for p in group}
        held = {id(p) for p in self.state.libraries["lib:m"].parameters()}
        self.assertEqual(owned, grouped)
        self.assertFalse(held & owned)
        bare = bank.build(SITES, {"r": 4, "memories": 2, "seed": 3, "lam": 0.5})
        for route in bank.set_routes(2):
            bare.sets[route].b[PATH].data = self.state.sets[route].b[PATH].data.clone()
        self.assertEqual(bank.emit(self.state), bank.emit(bare))
        for name, value in bank.provide(bare).items():
            self.assertTrue(torch.equal(bank.provide(self.state)[name], value), name)
        meta, fragments = bank.split_sets(bank.emit(self.state))
        self.assertEqual(sorted(fragments), sorted(bank.set_routes(2)))

    def test_a_library_may_be_any_rank_and_is_dropped_by_name(self):
        wide = self.a_lora(seed=12, r=7)
        bank.load_set(self.state, "lib:wide", lora_torch.emit(wide), None)
        want = self.inner(self.x) + self.delta(wide, self.x) + self.delta(
            self.state.sets[DREAMER], self.x)
        self.assertTrue(torch.allclose(self.forward(["lib:wide+dreamer"] * 3), want, atol=1e-6))
        bank.drop_set(self.state, "lib:wide")
        bank.drop_set(self.state, "lib:wide")                 # already gone
        self.assertEqual(sorted(self.state.libraries), ["lib:m"])
        with self.assertRaises(ValueError):
            bank.drop_set(self.state, DREAMER)

    def test_one_param_group_per_route_in_set_order(self):
        groups = bank.param_groups(self.state)
        self.assertEqual(tuple(sorted(groups)), bank.set_routes(2))
        for route, group in groups.items():
            self.assertEqual([id(p) for p in group],
                             [id(p) for p in self.state.sets[route].parameters()])
        self.assertEqual(sorted(bank.param_groups(
            bank.build(SITES, {"r": 4, "memories": 0}))), [DREAMER])


@needs_torch
class NamedPayloadTest(unittest.TestCase):
    def setUp(self):
        self.state = bank.build(SITES, {"r": 4, "memories": 2, "seed": 3})
        groups = bank.param_groups(self.state)
        self.optimizer = torch.optim.AdamW(
            [{"params": groups[name]} for name in sorted(groups)], lr=1e-2)

    def step(self):
        for parameter in self.state.parameters():
            parameter.grad = torch.ones_like(parameter)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def test_a_named_payload_is_lora_emit_of_one_set(self):
        self.step()
        payload = bank.emit_set(self.state, "memory:01")
        self.assertEqual(payload, lora_torch.emit(self.state.sets["memory:01"]))
        plain = lora_torch.build(SITES, {"r": 4, "seed": 0})
        lora_torch.load(plain, payload)                       # any LoRA reader reads it
        self.assertTrue(torch.equal(plain.b[PATH], self.state.sets["memory:01"].b[PATH]))

    def test_emit_set_then_load_set_is_exact(self):
        self.step()
        payload = bank.emit_set(self.state, "memory:00")
        a_before = self.state.sets["memory:00"].a[PATH]
        bank.load_set(self.state, "memory:01", payload, self.optimizer)
        self.assertEqual(bank.emit_set(self.state, "memory:01"), payload)
        for mine, theirs in zip(self.state.sets["memory:01"].parameters(),
                                self.state.sets["memory:00"].parameters()):
            self.assertTrue(torch.equal(mine, theirs))
        self.assertIs(self.state.sets["memory:00"].a[PATH], a_before)

    def test_load_set_none_is_the_build_init_with_no_moments(self):
        fresh = bank.build(SITES, {"r": 4, "memories": 2, "seed": 3})
        held = self.state.sets["memory:00"].a[PATH]
        self.step()
        self.assertFalse(torch.equal(held, fresh.sets["memory:00"].a[PATH]))
        held.grad = torch.ones_like(held)                     # mid-accumulation
        bank.load_set(self.state, "memory:00", None, self.optimizer)
        self.assertIs(self.state.sets["memory:00"].a[PATH], held)   # in place
        self.assertEqual(bank.emit_set(self.state, "memory:00"),
                         bank.emit_set(fresh, "memory:00"))
        for parameter in self.state.sets["memory:00"].parameters():
            self.assertNotIn(parameter, self.optimizer.state)
            self.assertIsNone(parameter.grad)
        for route in (DREAMER, "memory:01"):                  # the others keep theirs
            for parameter in self.state.sets[route].parameters():
                self.assertEqual(int(self.optimizer.state[parameter]["step"]), 1)
        self.step()                                           # and it steps again, from 1
        self.assertEqual(int(self.optimizer.state[held]["step"]), 1)
        self.assertEqual(int(self.optimizer.state[self.state.sets[DREAMER].a[PATH]]["step"]), 2)

    def test_the_optimizer_blob_round_trips_after_a_reset(self):
        """dream_bank's resume under the one-group-per-route layout: a
        state_dict saved with one set's moments absent loads back."""
        self.step()
        bank.load_set(self.state, "memory:00", None, self.optimizer)
        saved = self.optimizer.state_dict()
        self.assertEqual(len(saved["param_groups"]), 3)
        again = bank.build(SITES, {"r": 4, "memories": 2, "seed": 3})
        groups = bank.param_groups(again)
        optimizer = torch.optim.AdamW(
            [{"params": groups[name]} for name in sorted(groups)], lr=1e-2)
        optimizer.load_state_dict(saved)
        self.assertNotIn(again.sets["memory:00"].a[PATH], optimizer.state)
        self.assertEqual(int(optimizer.state[again.sets["memory:01"].a[PATH]]["step"]), 1)

    def test_a_payload_of_another_rank_or_other_paths_is_refused_by_name(self):
        other_rank = lora_torch.emit(lora_torch.build(SITES, {"r": 3, "seed": 1}))
        with self.assertRaises(ValueError) as caught:
            bank.load_set(self.state, "memory:00", other_rank, self.optimizer)
        self.assertIn("rank [3]", str(caught.exception))
        self.assertIn("rank 4", str(caught.exception))
        elsewhere = (SiteMeta(name="other", path="block.other", has_weight=True,
                              shape=(8, 6), is_boundary=False),)
        other_paths = lora_torch.emit(lora_torch.build(elsewhere, {"r": 4, "seed": 1}))
        for route in ("memory:00", "lib:m"):
            with self.assertRaises(ValueError) as caught:
                bank.load_set(self.state, route, other_paths, self.optimizer)
            self.assertIn("block.other", str(caught.exception))
            self.assertIn(PATH, str(caught.exception))
        misshapen = (SiteMeta(name=PATH, path=PATH, has_weight=True,
                              shape=(9, 6), is_boundary=False),)
        with self.assertRaises(ValueError) as caught:
            bank.load_set(self.state, "lib:m", lora_torch.emit(
                lora_torch.build(misshapen, {"r": 4, "seed": 1})), None)
        self.assertIn("do not fit", str(caught.exception))

    def test_the_set_verbs_take_one_set(self):
        for route in (BASE, "lib:m+dreamer", "memory:02"):
            with self.assertRaises(ValueError, msg=route):
                bank.emit_set(self.state, route)
        with self.assertRaises(ValueError):
            bank.load_set(self.state, "lib:m", None, None)    # a library has no init


# ---------------------------------------------------------------------------
# torch: the learner
# ---------------------------------------------------------------------------

if torch is not None:
    class _Logits:
        def __init__(self, logits):
            self.logits = logits

    class _SitedLM(torch.nn.Module):
        """A causal toy LM with ONE adapter site (`proj`): each position sees
        the running mean of its prefix, so right-padding cannot leak and a
        delta at `proj` moves every later logit."""

        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(VOCAB, WIDTH)
            self.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)
            self.head = torch.nn.Linear(WIDTH, VOCAB, bias=False)

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            h = self.embed(input_ids)
            counts = torch.arange(1, h.shape[1] + 1, dtype=h.dtype)[None, :, None]
            h = h.cumsum(1) / counts
            return _Logits(self.head(h + self.proj(h)))


@needs_torch
class TorchLearnerVerbsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.learner = self.a_learner()
        self.params = self.learner._tenants["t"].params["pi"]
        for index, route in enumerate(bank.set_routes(2)):
            generator = torch.Generator().manual_seed(index)
            self.params.sets[route].b["proj"].data = torch.randn(
                WIDTH, 2, generator=generator) / 2

    def a_learner(self, **kwargs):
        learner = TorchLearner(device="cpu", dtype=torch.float32, grad_clip=1e9)
        torch.manual_seed(0)
        learner._model = _SitedLM().requires_grad_(False).eval()
        learner._base = "toy"
        learner.install("t", a_bank(**kwargs))
        return learner

    def test_forward_is_the_loss_paths_per_document_nll(self):
        routes = ["memory:00", "memory:01", None]    # None: the dreamer's row
        nll = self.learner.forward("t", a_batch(DOCS, routes))
        self.assertEqual(len(nll), 3)
        for doc, route, got in zip(DOCS, routes, nll):
            loss = self.learner.forward_backward("t", a_batch([doc], [route])).loss
            self.assertAlmostEqual(got, loss, places=5)
        tokens = [len(doc) - 1 for doc in DOCS]
        whole = self.learner.forward_backward("t", a_batch(DOCS, routes)).loss
        self.assertAlmostEqual(
            sum(n * t for n, t in zip(nll, tokens)) / sum(tokens), whole, places=5)

    def test_a_microbatch_of_one_token_documents_scores_zero_and_trains_nothing(self):
        """A dream cut at blank lines holds one-token trajectories (a `---`
        line), and a fork at batch 1 steps one alone: no position has a
        prefix, so there is no target, no graph and no gradient."""
        before = self.learner.emit("t")
        lonely = a_batch([[7], [3]], ["memory:00", "memory:01"])
        self.assertEqual(self.learner.forward("t", lonely), (0.0, 0.0))
        self.assertEqual(self.learner.forward_backward("t", lonely).loss, 0.0)
        self.assertTrue(all(p.grad is None for p in self.params.parameters()))
        self.learner.optim_step("t")
        self.assertEqual(self.learner.emit("t").adapters, before.adapters)

    def test_forward_accumulates_nothing(self):
        self.learner.forward("t", a_batch(DOCS, ["memory:00"] * 3))
        self.assertTrue(all(p.grad is None for p in self.params.parameters()))
        before = self.learner.emit("t")
        self.learner.forward("t", a_batch(DOCS))
        self.assertEqual(self.learner.emit("t").adapters, before.adapters)

    def test_forward_routes_each_row_by_its_own_fact(self):
        stacked = a_batch(DOCS[:1] * 3, ["memory:00", BASE, "lib:m+memory:00"])
        with self.assertRaises(ValueError):
            self.learner.forward("t", stacked)
        library = lora_torch.build((SITE,), {"r": 3, "seed": 9})
        library.b["proj"].data = torch.randn(WIDTH, 3) / 3
        self.learner.load_set("t", "pi", "lib:m", lora_torch.emit(library))
        under_memory, under_base, under_stack = self.learner.forward("t", stacked)
        self.assertEqual(len({round(under_memory, 6), round(under_base, 6),
                              round(under_stack, 6)}), 3)
        alone = self.learner.forward("t", a_batch(DOCS[:1], ["memory:00"]))[0]
        self.assertAlmostEqual(under_memory, alone, places=5)

    def test_a_stacked_row_trains_its_trainable_part_only(self):
        library = lora_torch.build((SITE,), {"r": 3, "seed": 9})
        library.b["proj"].data = torch.randn(WIDTH, 3) / 3
        payload = lora_torch.emit(library)
        self.learner.load_set("t", "pi", "lib:m", payload)
        untouched = self.learner.emit_set("t", "pi", "memory:01")
        before = self.learner.emit_set("t", "pi", DREAMER)
        self.learner.forward_backward("t", a_batch(DOCS, ["lib:m+dreamer"] * 3))
        self.learner.optim_step("t")
        self.assertNotEqual(self.learner.emit_set("t", "pi", DREAMER), before)
        self.assertEqual(self.learner.emit_set("t", "pi", "memory:01"), untouched)
        self.assertEqual(self.learner.emit_set("t", "pi", "lib:m"), payload)

    def test_lr_scales_scale_one_group_for_one_step_only(self):
        """Adam's first step moves every element by lr (bias-corrected
        m/sqrt(v) is the gradient's sign), so the step size IS the lr."""
        learner = self.a_learner(lr=1e-2)
        params = learner._tenants["t"].params["pi"]
        optimizer = learner._tenants["t"].optimizers["pi"]

        def step(scales):
            before = {route: params.sets[route].a["proj"].detach().clone()
                      for route in bank.set_routes(2)}
            for parameter in params.parameters():
                parameter.grad = torch.ones_like(parameter)
            learner.optim_step("t", scales)
            return {route: float((params.sets[route].a["proj"] - before[route]).abs().max())
                    for route in before}
        moved = step({"memory:00": 0.5})
        self.assertAlmostEqual(moved["memory:00"], 0.5e-2, places=6)
        self.assertAlmostEqual(moved["memory:01"], 1e-2, places=6)
        self.assertAlmostEqual(moved[DREAMER], 1e-2, places=6)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-2] * 3)
        again = step(None)                        # the scale did not outlive its step
        self.assertAlmostEqual(again["memory:00"], 1e-2, places=5)
        twice = step({"pi.memory:00": 0.5})       # from the BASE, never compounding
        self.assertAlmostEqual(twice["memory:00"], 0.5e-2, places=5)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-2] * 3)

    def test_a_lane_s_gradient_norm_never_scales_another_lane_s_step(self):
        """dream_bank clips BY GROUP (AdapterType.clips_by_group): a joint norm
        would let one lane's large gradient shrink its neighbour's clipped
        gradient, and Adam's second step would then differ with the company a
        lane keeps — K lanes would stop equalling K tenants on metal."""
        def neighbour_after(loud: float):
            learner = TorchLearner(device="cpu", dtype=torch.float32, grad_clip=1.0)
            torch.manual_seed(0)
            learner._model = _SitedLM().requires_grad_(False).eval()
            learner._base = "toy"
            learner.install("t", a_bank(lr=1e-2))
            params = learner._tenants["t"].params["pi"]
            for step, noise in enumerate((1.0, loud)):
                for route in bank.set_routes(2):
                    size = noise if route == "memory:00" else 0.5 + step
                    for parameter in params.sets[route].parameters():
                        parameter.grad = torch.full_like(parameter, size)
                learner.optim_step("t")
            return params.sets["memory:01"].a["proj"].detach().clone()
        self.assertTrue(torch.equal(neighbour_after(loud=1e-3), neighbour_after(loud=1e3)))

    def test_an_lr_scale_for_no_group_is_refused_before_anything_steps(self):
        before = self.learner.emit("t")
        with self.assertRaises(ValueError) as caught:
            self.learner.optim_step("t", {"memory:09": 0.5})
        self.assertIn("memory:09", str(caught.exception))
        self.assertEqual(self.learner.emit("t").adapters, before.adapters)

    def test_the_overrides_reach_the_dreamer_and_every_memory_group(self):
        learner = self.a_learner(lr=1e-3, overrides={
            "pi.dreamer": {"lr": 5e-4}, "pi.memory": {"lr": 2e-3},
            "pi.memory:01": {"lr": 4e-3}})
        tenant = learner._tenants["t"]
        self.assertEqual(tenant.base_lrs["pi"],
                         (("dreamer", 5e-4), ("memory:00", 2e-3), ("memory:01", 4e-3)))
        self.assertEqual([group["lr"] for group in tenant.optimizers["pi"].param_groups],
                         [5e-4, 2e-3, 4e-3])

    def test_load_set_through_the_learner_resets_that_sets_moments(self):
        self.learner.forward_backward("t", a_batch(DOCS, ["memory:00", "memory:01", None]))
        self.learner.optim_step("t")
        optimizer = self.learner._tenants["t"].optimizers["pi"]
        trained = self.learner.emit_set("t", "pi", "memory:00")
        self.learner.load_set("t", "pi", "memory:00", None)
        for parameter in self.params.sets["memory:00"].parameters():
            self.assertNotIn(parameter, optimizer.state)
        self.assertIn(self.params.sets["memory:01"].a["proj"], optimizer.state)
        self.assertNotEqual(self.learner.emit_set("t", "pi", "memory:00"), trained)
        self.learner.load_set("t", "pi", "memory:00", trained)
        self.assertEqual(self.learner.emit_set("t", "pi", "memory:00"), trained)
        self.learner.forward_backward("t", a_batch(DOCS, ["memory:00"] * 3))
        self.learner.optim_step("t")                          # and it trains on
        self.assertNotEqual(self.learner.emit_set("t", "pi", "memory:00"), trained)

    def test_the_whole_entry_still_round_trips_with_its_moments(self):
        """dream_bank's resume under the new optimizer layout: emit -> a fresh
        tenant's load -> the same next step."""
        batch = a_batch(DOCS, ["memory:00", "memory:01", None])
        self.learner.forward_backward("t", batch)
        self.learner.optim_step("t")
        saved = self.learner.emit("t")
        resumed = self.a_learner()
        resumed.load("t", saved.adapters, saved.optim)
        for learner in (self.learner, resumed):
            learner.forward_backward("t", batch)
            learner.optim_step("t")
        self.assertEqual(resumed.emit("t").adapters, self.learner.emit("t").adapters)

    def test_an_adapter_type_with_no_sets_refuses_by_name(self):
        learner = self.a_learner(adapter_type="lora")
        with self.assertRaises(ValueError) as caught:
            learner.load_set("t", "pi", "memory:00", None)
        self.assertIn("holds no sets by route", str(caught.exception))
        with self.assertRaises(KeyError):
            learner.emit_set("t", "nobody", DREAMER)


@needs_torch
class ChorusTest(unittest.TestCase):
    """On a sharded learner every rank holds a whole copy of the sets and
    takes its half of every forward: the verbs that change a set or run the
    base are ANNOUNCED, and the follower's table knows each one."""

    def a_learner(self):
        from test_learner_sleep import Heard
        from rlstack.runner.learners.fsdp_torch import FsdpTorchLearner

        return FsdpTorchLearner(Heard())

    def test_the_new_verbs_are_announced_before_they_run(self):
        learner = self.a_learner()
        for verb, args in (("forward", ("t", a_batch(DOCS))),
                           ("load_set", ("t", "pi", "memory:00", None)),
                           ("drop_set", ("t", "pi", "lib:m")),
                           ("optim_step", ("t", {"memory:00": 0.5}))):
            with self.assertRaises(KeyError):                 # nobody installed `t`
                getattr(learner, verb)(*args)
        self.assertEqual(learner.ranks.said,
                         ["forward", "load_set", "drop_set", "optim_step"])

    def test_the_follower_table_holds_every_announced_verb(self):
        from rlstack.runner.learners.ranks import RankCommand

        learner = self.a_learner()
        for verb, args in (("forward", ("t", a_batch(DOCS))),
                           ("load_set", ("t", "pi", "memory:00", None)),
                           ("drop_set", ("t", "pi", "lib:m")),
                           ("optim_step", ("t", {"memory:00": 0.5}))):
            with self.assertRaises(KeyError):                 # reached the verb itself
                learner.follow(RankCommand(verb, args))
        self.assertEqual(learner.ranks.said, [])              # a follower never announces


if __name__ == "__main__":
    unittest.main()
