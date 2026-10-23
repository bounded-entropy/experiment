"""Ordinary bank routes select independent adapter state in sealed replay."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import rlstack
from rlstack.data.flatten import TokenBatch
from rlstack.policy.siteschema import SiteMeta, fake_qwen_schema
from rlstack.runner.interfaces import EntryInstall, OptimSettings, Parameterization
from rlstack.runner.remote import decode_parameterization, encode_parameterization, spec_from_json
from rlstack.spec.canonical import canonical_json
from rlstack.spec.specs import AdapterSpec, HostSpec, Plans, PolicySpec, Schedule, Topology, learner
from rlstack.spec.validate import check_adapter_routes_are_supported, check_schedule_is_sane, check_sites_do_not_overlap

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from rlstack.policy.adapters.replay import row_plan
    from rlstack.runner.learners.torch_learner import TorchLearner

SITE = SiteMeta(name="proj", path="proj", has_weight=True, shape=(4, 4), is_boundary=False)
OPTIM = OptimSettings("adamw", 0.01, (0.9, 0.95), 0.1, {})


def installation(adapter_type="lora"):
    site = SITE if adapter_type == "lora" else SiteMeta(
        name="boundary", path="boundary", has_weight=False, shape=None, is_boundary=True)
    init = {"r": 2} if adapter_type == "lora" else {"d": 4}
    return Parameterization("route-toy", "sft", tuple(
        EntryInstall(name, adapter_type, {**init, "seed": seed}, True, (site,), name)
        for name, seed in (("first", 7), ("second", 11))), OPTIM)


def batch(routes):
    return TokenBatch(token_ids=(1, 2, 3) * len(routes),
        loss_mask=(0, 1, 1) * len(routes), behavior_logprobs=(0., 0., 0.) * len(routes),
        segment_ids=(0, 0, 0) * len(routes), doc_starts=tuple(3 * i for i in range(len(routes))),
        doc_turn_extras=tuple(({} if route is None else {"route": route},) for route in routes))


class RoutedRecordsTest(unittest.TestCase):
    def test_wire_roundtrip_and_absent_fields_keep_old_canonical_shape(self):
        old = AdapterSpec("lora", "proj", {"r": 2})
        self.assertNotIn("route", json.loads(canonical_json(old)))
        self.assertNotIn("lr_scales", json.loads(canonical_json(Schedule())))
        configured = replace(old, route="task/0")
        self.assertEqual(spec_from_json(canonical_json(configured)), configured)
        schedule = Schedule(lr_scales=({"pi": 1.0}, {"pi": 0.5}))
        self.assertEqual(spec_from_json(canonical_json(schedule)), schedule)
        self.assertEqual(decode_parameterization(encode_parameterization(installation())), installation())
        legacy = replace(installation(), entries=(replace(installation().entries[0], route=None),))
        row = encode_parameterization(legacy)
        self.assertNotIn("route", row["entries"][0])
        self.assertEqual(decode_parameterization(row), legacy)

    def test_only_routes_that_can_cooccur_conflict(self):
        schema = fake_qwen_schema(1, base="toy")
        entry = AdapterSpec("lora", "layers.0.self_attn.q_proj", {"r": 2}, route="one")
        for other, expected in (("two", 0), ("one", 1), (None, 1)):
            spec = SimpleNamespace(policy=PolicySpec("toy", {"a": entry, "b": replace(entry, route=other)}))
            self.assertEqual(len(check_sites_do_not_overlap(spec, schema)), expected)

    def test_routes_refuse_missing_serving_selector_and_provided_owner_merge(self):
        from rlstack.spec.specs import pool
        spec = SimpleNamespace(policy=PolicySpec("toy", {"a": AdapterSpec("lora", "proj", route="a")}),
                               topology=Topology((HostSpec((learner(),)),)))
        self.assertEqual(check_adapter_routes_are_supported(spec, None), [])
        spec.topology = Topology((HostSpec((pool("main"),)),))
        self.assertEqual(check_adapter_routes_are_supported(spec, None)[0].code, "routed-serving-unsupported")
        spec.topology = Topology((HostSpec((learner(),)),))
        spec.policy = PolicySpec("toy", {"v": AdapterSpec("value_head", "final_hidden", route="a")})
        self.assertEqual(check_adapter_routes_are_supported(spec, None)[0].code, "routed-provides-unsupported")

    def test_invalid_route_and_lr_scale_are_refused(self):
        spec = SimpleNamespace(policy=PolicySpec("toy", {"a": AdapterSpec("lora", "proj", route="")}),
                               topology=Topology((HostSpec((learner(),)),)))
        self.assertEqual(check_adapter_routes_are_supported(spec, None)[0].code, "bad-adapter-route")
        for value in (-1, float("nan"), float("inf")):
            spec.algo = SimpleNamespace(schedule=Schedule(lr_scales=({"first": value},)))
            self.assertEqual(check_schedule_is_sane(spec, None)[0].code, "bad-schedule")


if torch is not None:
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(7, 4)
            self.proj = torch.nn.Linear(4, 4, bias=False)
            self.boundary = torch.nn.Identity()
            self.head = torch.nn.Linear(4, 7, bias=False)

        def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
            hidden = self.boundary(self.proj(self.embed(input_ids)))
            return SimpleNamespace(logits=self.head(hidden))


@unittest.skipIf(torch is None, "CPU torch required for actual routed updates")
class RoutedLearnerTest(unittest.TestCase):
    def make(self, adapter_type="lora"):
        with torch.random.fork_rng():
            torch.manual_seed(41)
            learner = TorchLearner("cpu", torch.float32, grad_clip=0.02, checkpoint_activations=False)
            learner._model = TinyModel().requires_grad_(False).eval()
        learner._base = "route-toy"
        learner.install("tenant", installation(adapter_type))
        return learner, learner._tenants["tenant"]

    def backward(self, learner, routes, scales=None):
        state = learner._tenants["tenant"]
        with row_plan(learner._model).route(learner._rows_of(state, batch(routes), len(routes))):
            logits = learner._model(torch.tensor([[1, 2, 3]] * len(routes))).logits
            factors = torch.tensor(scales or [1.] * len(routes))[:, None, None]
            (logits * factors).sum().backward()

    def test_mixed_rows_match_separate_models_and_independent_clipping(self):
        mixed, state = self.make()
        first, first_state = self.make()
        second, second_state = self.make()
        self.backward(mixed, ["first", "second"], [100., 0.1])
        self.backward(first, ["first"], [100.])
        self.backward(second, ["second"], [0.1])
        for name, other in (("first", first_state), ("second", second_state)):
            for got, want in zip(state.params[name].parameters(), other.params[name].parameters()):
                torch.testing.assert_close(got.grad, want.grad)
        mixed.optim_step("tenant", {"first": 0.5, "second": 2.0})
        first.optim_step("tenant", {"first": 0.5})
        second.optim_step("tenant", {"second": 2.0})
        for name, other in (("first", first_state), ("second", second_state)):
            for got, want in zip(state.params[name].parameters(), other.params[name].parameters()):
                torch.testing.assert_close(got, want)
            for got, want in zip(state.optimizers[name].state.values(),
                                 other.optimizers[name].state.values()):
                for moment in ("exp_avg", "exp_avg_sq", "step"):
                    torch.testing.assert_close(got[moment], want[moment])

    def test_absent_route_keeps_weights_moments_and_step_count(self):
        learner, state = self.make()
        self.backward(learner, ["first", "second"])
        learner.optim_step("tenant")
        before = [p.detach().clone() for p in state.params["second"].parameters()]
        moments = copy.deepcopy(state.optimizers["second"].state_dict())
        self.backward(learner, ["first"])
        self.assertTrue(all(p.grad is None for p in state.params["second"].parameters()))
        with patch.object(state.adapter_types["first"], "after_step") as stepped:
            learner.optim_step("tenant")
        self.assertEqual(stepped.call_count, 1)
        for got, want in zip(state.params["second"].parameters(), before):
            self.assertTrue(torch.equal(got, want))
        after = state.optimizers["second"].state_dict()
        for key, values in moments["state"].items():
            for name, value in values.items():
                self.assertTrue(torch.equal(after["state"][key][name], value))

    def test_steer_uses_the_same_generic_row_route(self):
        learner, state = self.make("steer")
        self.backward(learner, ["second"])
        self.assertTrue(all(p.grad is None for p in state.params["first"].parameters()))
        self.assertTrue(all(p.grad is not None for p in state.params["second"].parameters()))
        self.assertTrue(state.adapter_types["second"].supports_mixed_routes)

    def test_unknown_and_inconsistent_document_routes_raise(self):
        learner, state = self.make()
        with self.assertRaisesRegex(ValueError, "unknown adapter route"):
            learner._rows_of(state, batch(["missing"]), 1)
        disagreeing = replace(batch(["first"]), doc_turn_extras=(({"route": "first"}, {"route": "second"}),))
        with self.assertRaisesRegex(ValueError, "one adapter route"):
            learner._rows_of(state, disagreeing, 1)

    def test_default_slot_joins_each_route_without_inactive_states(self):
        learner, state = self.make()
        state.slot["shared"] = object()
        rows = learner._rows_of(state, batch(["second", "first", "second"]), 3)
        self.assertEqual(rows.index.tolist(), [0, 1, 0])
        self.assertIs(rows.slots[0]["proj"], state.params["second"])
        self.assertIs(rows.slots[1]["proj"], state.params["first"])
        self.assertIs(rows.slots[0]["shared"], state.slot["shared"])
        self.assertEqual(len(learner._rows_of(state, batch(["second"]), 1).slots), 1)


if __name__ == "__main__":
    unittest.main()
