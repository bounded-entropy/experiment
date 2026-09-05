"""The norm-scaled steer: the steer family with `alpha` (nsteer).

The injection at a token is alpha * ||h_t|| along a learned UNIT direction —
the paper's calibration, per token, off the live residual — so a fixed add
means the same thing at every layer, model and token. Only the direction is
learned; version 0 is a seeded random direction at the full fraction, never
the identity. Both compute halves scale the same way: the torch site (replay)
and the engine hook (rollout), which reads alpha off the fused file's
metadata that the learner's emit wrote.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None

from rlstack import PolicySpec, nsteer, steer
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.steer import STEER_RECORD, NSteer, Steer, SteerWindow
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES
from rlstack.runner.fakes import FakeEngine
from rlstack.spec.validate import check_sites_reachable_on, site_space, validate
from common import arith_spec
from test_steer import SCHEMA

if torch is not None:
    from rlstack.policy.adapters import steer_torch

needs_torch = unittest.skipUnless(
    torch is not None, "torch is trainer metal: this suite runs in the image")
RESIDUAL = frozenset({Mechanism.RESIDUAL})


class DeclarationTest(unittest.TestCase):
    def test_registered_as_the_steer_family_under_its_own_name(self) -> None:
        adapter_type = ADAPTER_TYPES.get("nsteer").instance
        self.assertIsInstance(adapter_type, NSteer)
        self.assertIsInstance(adapter_type, Steer)             # one family
        self.assertEqual(adapter_type.serving, Mechanism.RESIDUAL)
        self.assertEqual(adapter_type.records, (STEER_RECORD,))
        self.assertIs(adapter_type.directive, SteerWindow)

    def test_the_sugar_declares_the_fraction_and_a_direction_to_start_from(self) -> None:
        spec = nsteer("resid_pre.10", d=5120)
        self.assertEqual(spec.adapter_type, "nsteer")
        self.assertEqual(spec.init, {"d": 5120, "alpha": 0.1, "tie": False,
                                     "init_std": 1.0})
        with self.assertRaisesRegex(ValueError, "alpha > 0"):
            nsteer("resid_pre.10", d=8, alpha=0.0)
        with self.assertRaisesRegex(ValueError, "init_std > 0"):
            nsteer("resid_pre.10", d=8, init_std=0.0)   # no zero to start from

    def test_another_fraction_is_another_identity(self) -> None:
        self.assertNotEqual(nsteer("resid_pre.10", d=8, alpha=0.1).init,
                            nsteer("resid_pre.10", d=8, alpha=0.2).init)


class GateTest(unittest.TestCase):
    def policy(self, **init) -> PolicySpec:
        return PolicySpec(base="Qwen/Qwen3-0.6B",
                          bank={"dir": nsteer("resid_pre.1-2", d=64, **init)})

    def test_a_norm_scaled_bank_validates_and_reaches_through_the_plugin(self) -> None:
        spec = arith_spec("cas://t", policy=self.policy())
        self.assertEqual(validate(spec, SCHEMA), [])
        space = site_space(spec, SCHEMA)
        served = FakeEngine(plugins=RESIDUAL).reachability(space)
        self.assertEqual(served["resid_pre.1"], Mechanism.RESIDUAL)
        self.assertEqual(check_sites_reachable_on(spec, SCHEMA, "main", served), [])
        bare = FakeEngine().reachability(space)
        self.assertEqual({i.code for i in check_sites_reachable_on(
            spec, SCHEMA, "main", bare)}, {"site-unreachable"})

    def test_it_lives_where_a_steer_lives(self) -> None:
        (resid,) = SCHEMA.resolve("resid_pre.0")
        (proj,) = SCHEMA.resolve("layers.0.mlp.up_proj")
        self.assertTrue(NSteer().site_ok(resid))
        self.assertFalse(NSteer().site_ok(proj))


@needs_torch
class ScaledAddTest(unittest.TestCase):
    """Both halves add alpha * ||h_t|| along the unit direction, per token."""

    def sites(self):
        return tuple(SCHEMA.resolve("resid_pre.1"))

    def test_the_state_starts_at_a_direction_and_emits_its_fraction(self) -> None:
        state = steer_torch.build(self.sites(), {"d": 64, "alpha": 0.25,
                                                 "init_std": 1.0, "seed": 3})
        self.assertEqual(state.alpha, 0.25)
        (vector,) = state.parameters()
        self.assertGreater(float(vector.norm()), 0.0)          # never the identity
        payload = steer_torch.emit(state)
        self.assertEqual(steer_torch.payload_alpha(payload), 0.25)
        self.assertIsNone(steer_torch.payload_alpha(
            steer_torch.emit(steer_torch.build(self.sites(), {"d": 64}))))
        self.assertEqual(steer_torch.merge_alpha({"dir": payload}), 0.25)
        with self.assertRaisesRegex(ValueError, "direction to start from"):
            steer_torch.build(self.sites(), {"d": 64, "alpha": 0.25, "init_std": 0.0})

    def test_the_replay_site_scales_by_each_tokens_own_norm(self) -> None:
        from rlstack.policy.adapters.replay import ReplayRows

        state = steer_torch.build(self.sites(), {"d": 8, "alpha": 0.5,
                                                 "init_std": 1.0, "seed": 1})
        (path,) = state.paths
        torch.manual_seed(0)
        out = torch.randn(2, 3, 8) * torch.tensor([[[1.0], [4.0], [9.0]],
                                                    [[2.0], [2.0], [2.0]]])
        rows = ReplayRows(slots=({path: state},), index=torch.zeros(2, dtype=torch.long),
                          facts=None)
        delta = steer_torch._rows_delta(rows, path, out)
        scale = steer_torch._rows_scale(rows, path, out)
        added = (scale * delta)[:, :, :]                        # [rows, tokens, d]
        unit = state.vectors[path] / state.vectors[path].norm()
        for r in range(2):
            for t in range(3):
                expect = 0.5 * float(out[r, t].norm()) * unit
                self.assertTrue(torch.allclose(added[r, t], expect, atol=1e-5),
                                (r, t))

    def test_the_engine_hook_scales_the_same_way(self) -> None:
        import tempfile
        from pathlib import Path

        from safetensors.torch import save_file

        from rlstack_engine.steer import SteerPlugin
        from rlstack.policy.adapters.steer import STEER_END, STEER_FILE, STEER_START
        from test_steer_plugin import view_of

        plugin = SteerPlugin(max_slots=2, device="cpu", dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "steer.safetensors"
            direction = torch.tensor([3.0, 4.0, 0.0, 0.0])      # norm 5 -> unit (.6, .8, 0, 0)
            save_file({"model.layers.0": direction}, str(file),
                      metadata={steer_torch.ALPHA_KEY: "0.1"})
            extra = {STEER_FILE: str(file), STEER_START: 0, STEER_END: None}
            slots = [plugin.slot_of(extra)]
            view = view_of((0, 2), (2,), slots, (extra,), 0)   # one request, two prefill tokens
            routing = plugin.routing(view)
            hidden = torch.tensor([[0.0, 0.0, 10.0, 0.0],       # norm 10
                                   [0.0, 0.0, 0.0, 20.0]])      # norm 20
            before = hidden.clone()
            plugin.add(routing, "model.layers.0", hidden)
            added = hidden - before
            self.assertTrue(torch.allclose(added[0], torch.tensor([0.6, 0.8, 0.0, 0.0])))
            self.assertTrue(torch.allclose(added[1], torch.tensor([1.2, 1.6, 0.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
