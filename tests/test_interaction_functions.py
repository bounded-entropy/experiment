"""Plain interactions: declared code, sealed dependencies and crash recovery."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace

import boundary_fixtures
from common import arith_spec, arith_store, generation_spec
from test_resume import CrashingStore, SimulatedCrash, snapshot
from rlstack import (
    Bundle, Continue, FakeEngine, FakeLearner, HistoricalDerive, HistoricalSample, LocalStore,
    GenSpec, GroupPlan, Message, Plans, Role, RunPlan, Sample, SamplingSpec,
    Task, WavePlan, decode, encode, fake_qwen_schema, run_experiment,
)
from rlstack.code import require_declared_function
from rlstack.data.plan import PlanError, Replay, WaveRef
from rlstack.runner.assemble import check_generation_plan
from rlstack.runner.checkpointing import EVERY_UPDATE
from rlstack.runner.remote import spec_from_json
from rlstack.runner.traffic import EnginePoolClient, run_episode

SAMPLE = "boundary_fixtures:single_turn"
CONTINUE = "boundary_fixtures:continuation"
FUNCTIONS = (SAMPLE, CONTINUE)
CODE = ("boundary_fixtures",)
SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def plan(*leaves) -> RunPlan:
    return RunPlan(tuple(WavePlan((GroupPlan(str(index), (leaf,)),))
                         for index, leaf in enumerate(leaves, 1)))


class PlanFunctionTest(unittest.TestCase):
    def test_new_leaf_codec_roundtrip_and_explicit_dependencies(self) -> None:
        original = plan(Sample("t", SAMPLE),
                        Continue("self://rollouts/1#0", CONTINUE,
                                 dependencies=("student/1", "student/2")))
        self.assertEqual(decode(encode(original)), original)
        rows = [json.loads(row) for row in encode(original).splitlines()]
        self.assertNotIn("dependencies", rows[0]["wave"][0]["leaves"][0])
        self.assertEqual(rows[1]["wave"][0]["leaves"][0]["dependencies"],
                         ["student/1", "student/2"])
        check_generation_plan(original, FUNCTIONS)

    def test_old_leaf_rows_roundtrip_but_have_no_executor(self) -> None:
        original = plan(HistoricalSample("t", "old_environment"),
                        HistoricalDerive("self://rollouts/1#0", "maker", "old_environment"))
        encoded = encode(original)
        self.assertEqual(encode(decode(encoded)), encoded)
        self.assertIsInstance(decode(encoded).wave(1).groups[0].leaves[0], HistoricalSample)
        self.assertIsInstance(decode(encoded).wave(2).groups[0].leaves[0], HistoricalDerive)
        for leaf in original.waves:
            with self.assertRaisesRegex(PlanError, "read-only"):
                check_generation_plan(RunPlan((leaf,)), FUNCTIONS)

    def test_legacy_gen_specs_refuse_execution_instead_of_remapping_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "read-only"):
            spec_from_json({"__type__": "GenSpec", "envs": ["single_turn"],
                            "tasks": [], "makers": []})

    def test_only_declared_functions_may_execute(self) -> None:
        with self.assertRaisesRegex(PlanError, "not declared"):
            check_generation_plan(plan(Sample("t", SAMPLE)), ())
        # A prior import never grants permission to a different run.
        with self.assertRaisesRegex(ValueError, "not covered"):
            require_declared_function(SAMPLE, ())

    def test_continue_sources_one_earlier_sealed_trajectory(self) -> None:
        for source in ("self://rollouts/1#0", "self://rollouts/2#0",
                       "self://rollouts/0#0", "self://rollouts/1"):
            with self.subTest(source=source), self.assertRaises(PlanError):
                check_generation_plan(plan(Continue(source, CONTINUE)), FUNCTIONS)
        check_generation_plan(plan(Sample("t", SAMPLE),
                                   Continue("self://rollouts/1#0", CONTINUE)), FUNCTIONS)

    def test_replay_and_wave_refs_cannot_enter_a_generation_plan(self) -> None:
        with self.assertRaises(PlanError):
            check_generation_plan(plan(Replay("cas://rows#0")), FUNCTIONS)
        with self.assertRaises(PlanError):
            check_generation_plan(RunPlan((WaveRef("cas://rows"),)), FUNCTIONS)


class InteractionRecoveryTest(unittest.TestCase):
    def setup_run(self, root: str, *, invalid: bool = False):
        store, train, _ = arith_store(root)
        rollout = plan(Sample("arith-0000", SAMPLE),
                       Continue("self://rollouts/1#0", CONTINUE),
                       Continue("self://rollouts/2#0", CONTINUE))
        if invalid:
            rollout = plan(Sample("arith-0000", SAMPLE),
                           HistoricalSample("arith-0000", "single_turn"))
        spec = generation_spec(train,
            gen=GenSpec(FUNCTIONS, (train,)),
            plans=Plans(rollout=store.cas_put(encode(rollout))))
        return store, spec

    def test_continue_recovery_skips_sealed_work_and_preserves_every_byte(self) -> None:
        with tempfile.TemporaryDirectory() as straight, tempfile.TemporaryDirectory() as killed:
            baseline, spec = self.setup_run(straight)
            expected = run_experiment(spec, SCHEMA, baseline, FakeEngine(), None,
                                      checkpointing=EVERY_UPDATE)
            local, same_spec = self.setup_run(killed)
            self.assertEqual(spec, same_spec)
            with self.assertRaises(SimulatedCrash):
                run_experiment(spec, SCHEMA,
                    CrashingStore(killed, "write_rollout", after=1, post=False),
                    FakeEngine(), None, checkpointing=EVERY_UPDATE)
            fresh_engine = FakeEngine()
            recovered = run_experiment(spec, SCHEMA, LocalStore(killed), fresh_engine,
                                       None, checkpointing=EVERY_UPDATE, resume=True)
            self.assertEqual(recovered.completed, 3)
            self.assertEqual(fresh_engine.served, 2)
            self.assertEqual(snapshot(LocalStore(killed), recovered.run_id),
                             snapshot(baseline, expected.run_id))
            source = local.peek_rollout(recovered.run_id, 1)[0]
            derived = local.peek_rollout(recovered.run_id, 2)[0]
            self.assertEqual(derived["task"]["id"], source["task"]["id"] + "/next")
            self.assertIn("previous", derived["task"]["meta"])

    def test_only_the_final_continuation_wave_feeds_the_normal_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store, generation = self.setup_run(root)
            train = RunPlan((WaveRef("self://rollouts/3"),))
            spec = arith_spec(generation.gen.tasks[0], gen=generation.gen,
                             plans=Plans(rollout=generation.plans.rollout,
                                         train=store.cas_put(encode(train))))
            learner = FakeLearner()
            report = run_experiment(spec, SCHEMA, store, FakeEngine(), learner,
                                    checkpointing=EVERY_UPDATE)
            self.assertEqual(report.completed, 1)
            rows = store.peek_wave(report.run_id, 1)
            self.assertEqual(rows[0]["task"]["id"], "arith-0000/next/next")
            self.assertEqual(len(store.peek_ledger(report.run_id)), 1)

    def test_later_legacy_leaf_is_refused_before_the_first_sample(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            store, spec = self.setup_run(root, invalid=True)
            engine = FakeEngine()
            with self.assertRaisesRegex(PlanError, "read-only"):
                run_experiment(spec, SCHEMA, store, engine, None,
                               checkpointing=EVERY_UPDATE)
            self.assertEqual(engine.served, 0)

    def test_runner_seals_the_function_result(self) -> None:
        engine = FakeEngine()
        bundle = Bundle("bundle:function", {})
        engine.add_bundle(bundle)
        client = EnginePoolClient({"main": (engine, bundle)}, SamplingSpec(), 7)
        task = Task("t", "What is 2+2?", {"answer": 4})
        trajectory = asyncio.run(run_episode(boundary_fixtures.single_turn, task, client))
        self.assertIsInstance(trajectory.messages, tuple)
        self.assertIs(trajectory.turns[0].message, trajectory.messages[-1])


class PerCallSamplingTest(unittest.TestCase):
    def test_override_is_explicit_and_does_not_mutate_later_calls_or_seed_order(self) -> None:
        class SeenSampling(FakeEngine):
            def __init__(self):
                super().__init__()
                self.sampling_seen = []

            async def sample_tokens(self, messages, sampling, *args, **kwargs):
                self.sampling_seen.append(sampling)
                async for event in super().sample_tokens(messages, sampling, *args, **kwargs):
                    yield event

        engine = SeenSampling()
        bundle = Bundle("bundle:sampling", {})
        engine.add_bundle(bundle)
        defaults = SamplingSpec(temperature=0.8, top_p=0.9, max_tokens=2)
        greedy = SamplingSpec(temperature=0.0, top_p=1.0, max_tokens=64)
        trace = []
        client = EnginePoolClient({"main": (engine, bundle)}, defaults, 7, calls=trace)
        messages = (Message(Role.USER, "What is 30+40?"),)
        first = asyncio.run(client.sample(messages, sampling=greedy, max_tokens=16, stop=("\n",)))
        second = asyncio.run(client.sample(messages))
        self.assertEqual(engine.sampling_seen, [replace(greedy, max_tokens=16), defaults])
        self.assertNotEqual(first.seed, second.seed)
        self.assertEqual(first.turn_extras["sampling"],
                         {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16, "stop": ["\n"]})
        self.assertEqual(trace[0]["sampling"], first.turn_extras["sampling"])
        self.assertEqual(trace[0]["seed"], first.seed)
        self.assertEqual(second.turn_extras["sampling"]["temperature"], 0.8)
        # The override replaces defaults, preserving the old environment's
        # independent greedy budget; max_tokens remains a per-call ceiling.
        self.assertEqual(engine.sampling_seen[0].max_tokens, 16)


if __name__ == "__main__":
    unittest.main()
