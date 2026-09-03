"""The steer adapter type (ADR 0004): declaration, gate, window, and — where
torch is present — the replay lowering's numerics.

The stdlib half runs everywhere: the adapter type's declaration, the window
rule (coordinates, offset, refusal), Phase 0 accepting a steer bank on a
build that reaches the residual boundaries and refusing it on one that does
not, the directive's round trip to the seal through the fake bus, and a
steer run's resume equivalence. The torch half SKIPS locally and RUNS in the
image: zero is the identity bit for bit, the add lands inside the recorded
window and nowhere else, two tenants at one boundary each get their own,
missing or disagreeing records refuse, and install/uninstall leave the tree
as they found it.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from common import arith_spec, arith_store
from test_resume import CrashingStore, SimulatedCrash, snapshot
from rlstack import (
    Bundle, EnginePoolClient, FakeEngine, FakeLearner, LocalStore, Mechanism,
    Message, PolicySpec, Role, SamplingSpec, SteerWindow, fake_qwen_schema,
    lora, run_experiment, steer,
)
from rlstack.policy.adapters.rollout import (
    BuildDemands, Request, check_demand_fits,
)
from rlstack.policy.adapters.steer import (
    STEER_RECORD, Steer, resolve_window,
)
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES
from rlstack.spec.validate import check_sites_reachable_on, site_space, validate

try:
    import torch
except ImportError:                                  # the client environment
    torch = None

if torch is not None:
    from rlstack.policy.adapters import lora_torch, steer_torch
    from rlstack.policy.adapters.replay import ReplayRows, row_plan

needs_torch = unittest.skipUnless(
    torch is not None, "torch is trainer metal: this suite runs in the image")

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
RESIDUAL = frozenset({Mechanism.RESIDUAL})


def go(coro):
    return asyncio.run(coro)


def steer_policy(site: str = "resid_pre.1-2", **init) -> PolicySpec:
    return PolicySpec(base="Qwen/Qwen3-0.6B",
                      bank={"nudge": steer(site, d=64, **init)})


# ---------------------------------------------------------------------------
# the declaration and the gate
# ---------------------------------------------------------------------------

class DeclarationTest(unittest.TestCase):
    def test_registered_on_the_residual_lever_with_its_record(self) -> None:
        adapter_type = ADAPTER_TYPES.get("steer").instance
        self.assertIsInstance(adapter_type, Steer)
        self.assertEqual(adapter_type.serving, Mechanism.RESIDUAL)
        self.assertEqual(adapter_type.records, (STEER_RECORD,))
        self.assertIs(adapter_type.directive, SteerWindow)

    def test_it_lives_at_unweighted_boundaries_only(self) -> None:
        (resid,) = SCHEMA.resolve("resid_pre.0")
        (proj,) = SCHEMA.resolve("layers.0.mlp.up_proj")
        self.assertTrue(Steer().site_ok(resid))
        self.assertFalse(Steer().site_ok(proj))

    def test_the_sugar_declares_the_init(self) -> None:
        spec = steer("resid_pre.*", d=1024, tie=True, init_std=0.02)
        self.assertEqual(spec.adapter_type, "steer")
        self.assertEqual(spec.init, {"d": 1024, "tie": True, "init_std": 0.02})


class GateTest(unittest.TestCase):
    def test_a_steer_bank_validates_and_a_range_is_one_entry(self) -> None:
        spec = arith_spec("cas://t", policy=steer_policy("resid_pre.1-2"))
        self.assertEqual(validate(spec, SCHEMA), [])
        space = site_space(spec, SCHEMA)
        self.assertEqual([m.name for m in SCHEMA.resolve("resid_pre.1-2")],
                         ["resid_pre.1", "resid_pre.2"])
        self.assertEqual(len(space), len(SCHEMA.sites))    # exports nothing

    def test_reachable_on_a_build_with_the_plugin_only(self) -> None:
        spec = arith_spec("cas://t", policy=steer_policy("resid_pre.1-2"))
        space = site_space(spec, SCHEMA)
        served = FakeEngine(plugins=RESIDUAL).reachability(space)
        bare = FakeEngine().reachability(space)
        self.assertEqual(served["resid_pre.1"], Mechanism.RESIDUAL)
        self.assertEqual(served["final_hidden"], Mechanism.RESIDUAL)
        self.assertEqual(served["logits"], Mechanism.LOGITS)
        self.assertEqual(bare["resid_pre.1"], Mechanism.NONE)
        self.assertEqual(check_sites_reachable_on(spec, SCHEMA, "main", served), [])
        refused = check_sites_reachable_on(spec, SCHEMA, "main", bare)
        self.assertEqual({i.code for i in refused}, {"site-unreachable"})

    def test_a_steer_beside_a_lora_is_two_levers_one_bank(self) -> None:
        spec = arith_spec("cas://t", policy=PolicySpec(
            base="Qwen/Qwen3-0.6B",
            bank={"pi": lora("layers.0-3.self_attn.*", r=8),
                  "nudge": steer("resid_pre.0-3", d=64)}))
        self.assertEqual(validate(spec, SCHEMA), [])


# ---------------------------------------------------------------------------
# the window
# ---------------------------------------------------------------------------

class WindowTest(unittest.TestCase):
    def test_no_directive_is_every_position(self) -> None:
        self.assertEqual(resolve_window(None, Request((1, 2, 3))), (0, None))

    def test_the_window_is_offset_by_what_sits_in_front(self) -> None:
        request = Request((1, 2, 3), occupied=4)
        self.assertEqual(resolve_window(SteerWindow(2, 5), request), (6, 9))
        self.assertEqual(resolve_window(SteerWindow(start=3), request), (7, None))

    def test_a_window_that_cannot_fit_is_refused_not_clamped(self) -> None:
        for bad in (SteerWindow(start=-1), SteerWindow(3, 2)):
            with self.assertRaises(ValueError):
                resolve_window(bad, Request((1,)))

    def test_the_record_is_the_resolved_window(self) -> None:
        request = Request((1, 2), occupied=2, directives=(SteerWindow(1, 2),))
        fact = Steer().record_directive(Steer().directive_for(request), request)
        self.assertEqual(fact, {STEER_RECORD: [3, 4]})
        self.assertEqual(Steer().record_directive(None, Request((1,))),
                         {STEER_RECORD: [0, None]})


class DemandsTest(unittest.TestCase):
    """A demand never overrides a build fact (ADR 0004, Q6): arguments,
    environment, and an earlier adapter type's paid demands are all facts."""

    STEER = BuildDemands(engine_args={"worker_cls": "w", "enforce_eager": True},
                         env={"VLLM_USE_V2_MODEL_RUNNER": "0"})

    def test_a_build_that_already_agrees_pays_silently(self) -> None:
        check_demand_fits("steer", self.STEER, {"enforce_eager": True},
                          {"VLLM_USE_V2_MODEL_RUNNER": "0"})
        check_demand_fits("steer", self.STEER, {}, {})

    def test_a_graph_captured_build_refuses(self) -> None:
        with self.assertRaises(NotImplementedError) as refused:
            check_demand_fits("steer", self.STEER, {"enforce_eager": False}, {})
        self.assertIn("enforce_eager", str(refused.exception))

    def test_a_process_pinned_to_the_other_runner_refuses(self) -> None:
        with self.assertRaises(NotImplementedError) as refused:
            check_demand_fits("steer", self.STEER, {},
                              {"VLLM_USE_V2_MODEL_RUNNER": "1"})
        self.assertIn("VLLM_USE_V2_MODEL_RUNNER", str(refused.exception))

    def test_two_plugins_claiming_the_worker_refuse(self) -> None:
        with self.assertRaises(NotImplementedError):
            check_demand_fits("other", BuildDemands(engine_args={"worker_cls": "x"}),
                              {"worker_cls": "w"}, {})


class SealTest(unittest.TestCase):
    """The window round-trips to the seal through the fake bus."""

    BUNDLE = Bundle("bundle:s", {"nudge": 0}, payloads={"nudge": b"\x00"},
                    adapter_types={"nudge": "steer"})

    def client(self) -> EnginePoolClient:
        engine = FakeEngine(plugins=RESIDUAL)
        engine.add_bundle(self.BUNDLE)
        return EnginePoolClient({"main": (engine, self.BUNDLE)}, SamplingSpec(),
                                episode_seed=5)

    def test_completion_only_seals_as_the_prompt_length(self) -> None:
        prompt = (Message(Role.USER, "2+3"),)
        turn = go(self.client().sample(prompt,
                                       directives=(SteerWindow(start=3),)))
        self.assertEqual(turn.turn_extras[STEER_RECORD], [3, None])

    def test_the_default_seals_too(self) -> None:
        turn = go(self.client().sample((Message(Role.USER, "2+3"),)))
        self.assertEqual(turn.turn_extras[STEER_RECORD], [0, None])


class SteerResumeTest(unittest.TestCase):
    """A steer bank's run dir is byte-identical across a crash and resume —
    the same claim test_resume.py makes for the lora bank."""

    def run_it(self, root: str, store) -> str:
        _, train, _ = arith_store(root)
        report = run_experiment(arith_spec(train, policy=steer_policy()), SCHEMA,
                                store, FakeEngine(plugins=RESIDUAL),
                                FakeLearner())
        return report.run_id

    def test_crash_then_resume_is_byte_identical(self) -> None:
        straight = tempfile.TemporaryDirectory()
        self.addCleanup(straight.cleanup)
        run_id = self.run_it(straight.name, LocalStore(straight.name))
        reference = snapshot(LocalStore(straight.name), run_id)

        crashed = tempfile.TemporaryDirectory()
        self.addCleanup(crashed.cleanup)
        with self.assertRaises(SimulatedCrash):
            self.run_it(crashed.name,
                        CrashingStore(crashed.name, "append_ledger", 2))
        resumed = self.run_it(crashed.name, LocalStore(crashed.name))
        self.assertEqual(resumed, run_id)
        self.assertEqual(snapshot(LocalStore(crashed.name), run_id), reference)


# ---------------------------------------------------------------------------
# the replay lowering's numerics — torch, so the image
# ---------------------------------------------------------------------------

WIDTH = 8
PATHS = ("block.a", "block.b")
SITES = tuple(SiteMeta(name=f"resid_pre.{i}", path=path, has_weight=False,
                       shape=None, is_boundary=True)
              for i, path in enumerate(PATHS))
PROJ = (SiteMeta(name="block.proj", path="block.proj", has_weight=True,
                 shape=(WIDTH, WIDTH), is_boundary=False),)


def a_model():
    """Two boundary modules and one Linear a LoRA can wrap."""
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.a = torch.nn.Linear(WIDTH, WIDTH, bias=False)
    model.block.b = torch.nn.Linear(WIDTH, WIDTH, bias=False)
    model.block.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)
    return model


def a_state(seed: int, std: float = 0.5, tie: bool = False):
    return steer_torch.build(SITES, {"d": WIDTH, "seed": seed,
                                     "init_std": std, "tie": tie})


def facts(*windows):
    """One turn per row, each recording its window."""
    return tuple(({STEER_RECORD: list(window)},) for window in windows)


def rows_for(slots, index, recorded):
    return ReplayRows(slots=tuple(slots), index=torch.tensor(index),
                      facts=recorded)


@needs_torch
class ReplayNumericsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = a_model()
        self.x = torch.randn(3, 5, WIDTH)
        self.base_a = self.model.block.a(self.x).detach().clone()

    def test_zero_is_the_identity_bit_for_bit(self) -> None:
        zero = a_state(1, std=0.0)
        steer_torch.install(self.model, zero)
        plan = rows_for([{PATHS[0]: zero, PATHS[1]: zero}], [0, 0, 0],
                        facts((0, None), (0, None), (0, None)))
        with row_plan(self.model).route(plan):
            out = self.model.block.a(self.x)
        self.assertTrue(torch.equal(out, self.base_a))

    def test_the_add_lands_inside_the_recorded_window_only(self) -> None:
        state = a_state(2)
        steer_torch.install(self.model, state)
        plan = rows_for([{PATHS[0]: state, PATHS[1]: state}], [0, 0, 0],
                        facts((0, None), (1, 3), (4, None)))
        with row_plan(self.model).route(plan):
            out = self.model.block.a(self.x)
        v = state.vectors[PATHS[0]].detach()
        expected = self.base_a.clone()
        expected[0] += v
        expected[1, 1:3] += v
        expected[2, 4:] += v
        self.assertTrue(torch.allclose(out, expected))

    def test_two_tenants_at_one_boundary_each_get_their_own(self) -> None:
        first, second = a_state(3), a_state(4)
        delta = lora_torch.build(PROJ, {"r": 2, "seed": 9})
        steer_torch.install(self.model, first)
        steer_torch.install(self.model, second)
        lora_torch.install(self.model, delta)
        plan = rows_for([{PATHS[0]: first, PATHS[1]: first},
                         {PATHS[0]: second, PATHS[1]: second},
                         {"block.proj": delta}],                # lora-only tenant
                        [0, 1, 2], facts((0, None), (2, None), (0, None)))
        with row_plan(self.model).route(plan):
            out = self.model.block.a(self.x)
        expected = self.base_a.clone()
        expected[0] += first.vectors[PATHS[0]].detach()
        expected[1, 2:] += second.vectors[PATHS[0]].detach()
        self.assertTrue(torch.allclose(out, expected))
        self.assertTrue(torch.equal(out[2], self.base_a[2]))   # transparent

    def test_a_missing_or_disagreeing_record_refuses(self) -> None:
        state = a_state(5)
        steer_torch.install(self.model, state)
        slot = {PATHS[0]: state, PATHS[1]: state}
        for recorded in (None, facts((0, None), (0, None), (0, None))[:2] + (
                ({STEER_RECORD: [0, None]}, {STEER_RECORD: [1, None]}),)):
            plan = rows_for([slot], [0, 0, 0], recorded)
            with row_plan(self.model).route(plan):
                with self.assertRaises(ValueError):
                    self.model.block.a(self.x)

    def test_uninstall_restores_the_tree(self) -> None:
        state = a_state(6)
        a, b = self.model.block.a, self.model.block.b
        steer_torch.install(self.model, state)
        self.assertIsInstance(self.model.block.a, steer_torch.SteerSite)
        steer_torch.uninstall(self.model, state)
        self.assertIs(self.model.block.a, a)
        self.assertIs(self.model.block.b, b)


class _Block(torch.nn.Module if torch is not None else object):
    """A decoder-layer stand-in: a module WITH children, so a steer at its
    output stands on the path to its projection."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)

    def forward(self, x):
        return self.proj(x)


LAYER = (SiteMeta(name="resid_pre.0", path="layer", has_weight=False,
                  shape=None, is_boundary=True),)
LAYER_PROJ = (SiteMeta(name="layer.proj", path="layer.proj", has_weight=True,
                       shape=(WIDTH, WIDTH), is_boundary=False),)


@needs_torch
class WrapperOnTheWayTest(unittest.TestCase):
    """Found on metal: a steer wrapping model.layers.8 hid the layer's
    children from the walk to model.layers.8.self_attn.q_proj. A wrapper on
    the way is walked through, in either install order."""

    def compose(self, steer_first: bool):
        torch.manual_seed(0)
        model = torch.nn.Module()
        model.layer = _Block()
        x = torch.randn(1, 3, WIDTH)
        base = model.layer(x).detach().clone()
        vec = steer_torch.build(LAYER, {"d": WIDTH, "seed": 1, "init_std": 0.5})
        delta = lora_torch.build(LAYER_PROJ, {"r": 2, "seed": 2})
        delta.b["layer.proj"].data = torch.randn(WIDTH, 2) / 2
        order = [(steer_torch, vec), (lora_torch, delta)]
        for lowering, state in (order if steer_first else order[::-1]):
            lowering.install(model, state)
        plan = rows_for([{"layer": vec, "layer.proj": delta}], [0],
                        facts((0, None)))
        with row_plan(model).route(plan):
            out = model.layer(x)
        lora_only = (x @ delta.a["layer.proj"].T) @ delta.b["layer.proj"].T
        expected = base + lora_only + vec.vectors["layer"].detach()
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))
        for lowering, state in order:
            lowering.uninstall(model, state)
        self.assertIsInstance(model.layer, _Block)
        self.assertIsInstance(model.layer.proj, torch.nn.Linear)

    def test_a_lora_installs_inside_a_steered_layer(self) -> None:
        self.compose(steer_first=True)

    def test_a_steer_wraps_a_layer_holding_a_lora(self) -> None:
        self.compose(steer_first=False)


@needs_torch
class StateTest(unittest.TestCase):
    def test_tied_is_one_parameter_under_every_path(self) -> None:
        tied = a_state(7, tie=True)
        self.assertEqual(len(tied.parameters()), 1)
        self.assertIs(tied.vectors[PATHS[0]], tied.vectors[PATHS[1]])
        self.assertEqual(len(a_state(7).parameters()), 2)

    def test_emit_load_round_trip_and_the_engine_side_merge(self) -> None:
        state, fresh = a_state(8), a_state(8, std=0.0)
        payload = steer_torch.emit(state)
        steer_torch.load(fresh, payload)
        for path in PATHS:
            self.assertTrue(torch.equal(fresh.vectors[path], state.vectors[path]))
        merged = steer_torch.merge_vectors({"nudge": payload})
        self.assertEqual(set(merged), set(PATHS))
        with self.assertRaises(ValueError):
            steer_torch.merge_vectors({"a": payload, "b": payload})


if __name__ == "__main__":
    unittest.main()
