"""Supervised curricula compile to normal plans, losses and checkpoint recovery."""
from dataclasses import replace
import json
import tempfile
import unittest

from rlstack import FakeLearner, LocalStore, Message, Role, Task, Trajectory, Turn, lora
from rlstack.data.plan import decode
from rlstack.data.trajectory import trajectory_to_row
from rlstack.policy.siteschema import fake_qwen_schema
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.loop import run_experiment
from rlstack.training.supervision import SftCurriculum, SftStage, curriculum_spec, supervised_spec


def row(index):
    message = Message(Role.ASSISTANT, "abc")
    turn = Turn(message, (3, 4, 5), (0., 0., 0.), "stop", None, "fixed", {}, 7)
    return trajectory_to_row(Trajectory(Task(f"doc/{index}", "", {}), (message,), (turn,), {}))


class SupervisionPlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = LocalStore(self.tmp.name)
        self.adapter = lora("layers.0.self_attn.q_proj", 2)

    def compile(self):
        return supervised_spec("fake/base", self.adapter,
            ((row(0), row(1), row(2)), (row(3),)),
            (SftStage(2, 2, 3e-4), SftStage(1, 1, 1e-4, "constant")),
            17, self.store.cas_put)

    def test_independent_entries_shared_steps_and_shorter_student_absence(self):
        spec = self.compile()
        self.assertIsNone(spec.plans.rollout)
        self.assertEqual(spec.algo.loss, "independent_sft")
        self.assertEqual(spec.algo.post, ("sft_document_weights",))
        self.assertEqual({a.adapter_type for a in spec.policy.bank.values()}, {"lora"})
        self.assertEqual({a.init["seed"] for a in spec.policy.bank.values()}, {17})
        plan = decode(self.store.cas_get(spec.plans.train))
        self.assertEqual(len(plan.waves), 4)
        self.assertEqual([len(wave.groups) for wave in plan.waves], [2, 1, 1, 1])
        self.assertAlmostEqual(spec.algo.schedule.lr_scales[0]["student0"], 3.)
        self.assertEqual(spec.algo.schedule.lr_scales[0]["student1"], 1.)
        self.assertAlmostEqual(spec.algo.schedule.lr_scales[-1]["student0"], .75)
        self.assertEqual([len(wave.groups[0].leaves) for wave in plan.waves], [2, 1, 2, 1])
        self.assertTrue(all(leaf.role == "student/0" for wave in plan.waves
                            for leaf in wave.groups[0].leaves))

    def test_curriculum_changes_corpus_and_records_before_and_each_stage_probe(self):
        curriculum = SftCurriculum(((row(0),), (row(1), row(2))),
                                  (SftStage(1, 1, 1e-4), SftStage(1, 2, 2e-4)))
        spec = curriculum_spec("fake/base", self.adapter, (curriculum,), 17,
                               self.store.cas_put, probe_rows=((row(8),),))
        self.assertEqual([probe.update for probe in spec.algo.schedule.probes], [0, 1, 2])
        probes = [json.loads(self.store.cas_get(probe.rows)) for probe in spec.algo.schedule.probes]
        self.assertTrue(all(rows["turns"][0]["turn_extras"]["route"] == "student/0"
                            for rows in probes))

    def test_run_uses_normal_trainer_ledger_and_cadence(self):
        spec = self.compile()
        report = run_experiment(spec, fake_qwen_schema(1, base="fake/base"), self.store,
                                {}, FakeLearner(), checkpointing=Checkpointing(every=3))
        handle = self.store.open_run(report.run_id)
        self.assertEqual(report.completed, 4)
        self.assertEqual([line["update"] for line in handle.read_checkpoints()], [0, 3, 4])
        self.assertEqual(len(handle.read_ledger()), 4)
        self.assertFalse(any("fit" in name for name in spec.policy.bank))

    def test_probes_preserve_full_vectors_routes_and_active_versions(self):
        spec = supervised_spec("fake/base", self.adapter,
            ((row(0), row(1), row(2)), (row(3),)),
            (SftStage(2, 2, 3e-4), SftStage(1, 1, 1e-4)), 17,
            self.store.cas_put, probe_rows=((row(8), row(9)), (row(10),)))
        report = run_experiment(spec, fake_qwen_schema(1, base="fake/base"), self.store,
                                {}, FakeLearner(), checkpointing=Checkpointing(every=3))
        handle = self.store.open_run(report.run_id)
        self.assertEqual(handle.checkpoint_tail()["versions"], {"student0": 4, "student1": 1})
        initial = handle.read_postdata_part(0, "probe")
        self.assertEqual(len(initial["nll"]), 3)
        self.assertEqual(initial["routes"], ["student/0", "student/0", "student/1"])
        self.assertEqual(initial["scored_tokens"], [3, 3, 3])
        final = handle.read_postdata_part(4, "probe")
        self.assertEqual(final["versions"], {"student0": 4, "student1": 1})
        self.assertEqual(len(final["nll_sum"]), 2)

    def test_crash_resume_with_independent_ages_is_byte_identical(self):
        from test_resume import CrashingStore, SimulatedCrash, snapshot
        spec = self.compile()
        schema = fake_qwen_schema(1, base="fake/base")
        straight = run_experiment(spec, schema, self.store, {}, FakeLearner(),
                                  checkpointing=Checkpointing(every=3))
        expected = snapshot(self.store, straight.run_id)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        crashing = CrashingStore(tmp.name, "append_ledger", 2, False)
        resumed_spec = supervised_spec("fake/base", self.adapter,
            ((row(0), row(1), row(2)), (row(3),)),
            (SftStage(2, 2, 3e-4), SftStage(1, 1, 1e-4, "constant")),
            17, crashing.cas_put)
        with self.assertRaises(SimulatedCrash):
            run_experiment(resumed_spec, schema, crashing, {}, FakeLearner(),
                           checkpointing=Checkpointing(every=3))
        resumed_store = LocalStore(tmp.name)
        report = run_experiment(resumed_spec, schema, resumed_store, {}, FakeLearner(),
                                checkpointing=Checkpointing(every=3))
        self.assertEqual(report.run_id, straight.run_id)
        self.assertEqual(snapshot(resumed_store, report.run_id), expected)

    def test_explicit_initial_payload_uses_existing_cas_warm_start(self):
        spec = supervised_spec("fake/base", self.adapter, ((row(0),),),
            SftStage(1, 1, 1e-4), 17, self.store.cas_put, start_payloads=(b"weights",))
        self.assertTrue(spec.init.policy.startswith("cas://"))
        self.assertEqual(spec.init.optim, "fresh")
        from rlstack.runner.loop import cas_warm_start
        self.assertEqual(cas_warm_start(spec.init, self.store, {"student0"}).adapters,
                         {"student0": b"weights"})
