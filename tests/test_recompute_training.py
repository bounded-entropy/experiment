"""Disposable supervision uses the normal Trainer without weight archives."""
import asyncio
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rlstack import FakeLearner, LocalStore, lora
from rlstack.data.stores.base import run_done, run_progress
from rlstack.policy.siteschema import fake_qwen_schema
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.loop import run_experiment
from rlstack.runner.refs import RefReader
from rlstack.runner.roles import Trainer
from rlstack.runner.training_client import TrainingClient
from rlstack.training.supervision import SftStage, supervised_spec
from test_supervision import row


class NoArchives(FakeLearner):
    def emit_initial(self, *args):
        raise AssertionError("disposable supervision must not archive initialization")

    def emit_adapters(self, *args):
        raise AssertionError("disposable supervision must not archive adapters")

    def emit_optim(self, *args):
        raise AssertionError("disposable supervision must not archive Adam")


class RecomputeTrainingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = LocalStore(self.tmp.name)
        self.adapter = lora("layers.0.self_attn.q_proj", 2)
        self.schema = fake_qwen_schema(1, base="fake/base")
        self.spec = supervised_spec("fake/base", self.adapter,
            ((row(0), row(1), row(2)),), SftStage(1, 1, 1e-4), 17, self.store.cas_put,
            probe_rows=((row(9),),))
        self.policy = Checkpointing(every=1, state="recompute")

    def run_spec(self, learner):
        return run_experiment(self.spec, self.schema, self.store, {}, learner,
                              subdir="disposable", checkpointing=self.policy)

    def test_full_observations_without_any_adapter_or_optimizer_archive(self):
        learner = NoArchives()
        result = self.run_spec(learner)
        reference = f"disposable/{result.run_id}"
        handle = self.store.open_run(reference)
        self.assertEqual(len(handle.read_ledger()), 3)
        self.assertEqual(handle.read_checkpoints(), [])
        self.assertTrue(run_done(self.store, reference))
        self.assertIsNone(run_progress(self.store, reference).checkpointed)
        self.assertEqual(handle.read_postdata_part(0, "probe")["row_ids"], ["doc/9"])
        self.assertEqual(handle.read_postdata_part(3, "probe")["versions"], {"student0": 3})
        home = Path(self.tmp.name, "runs", reference)
        self.assertFalse((home / "adapters").exists())
        self.assertFalse((home / "optim").exists())

    def test_completed_measurements_are_reused_without_reinstalling_a_policy(self):
        self.run_spec(NoArchives())
        class NoCompute(NoArchives):
            def install(self, *args):
                raise AssertionError("completed observations need no live policy")
        self.assertEqual(self.run_spec(NoCompute()).completed, 3)

    def test_same_no_archive_policy_crosses_the_existing_learner_wire(self):
        from rlstack.runner.remote import LearnerService, LocalTransport, RemoteLearner
        served = NoArchives()
        proxy = RemoteLearner(LocalTransport(LearnerService(served)))
        result = self.run_spec(proxy)
        self.assertEqual(result.completed, 3)
        self.assertEqual(self.store.peek_checkpoints(f"disposable/{result.run_id}"), [])

    def test_incomplete_measurements_recompute_from_initialization_to_same_bytes(self):
        commit = Trainer.commit
        async def die(trainer, update, *args):
            await commit(trainer, update, *args)
            if update == 2:
                raise RuntimeError("injected incomplete disposable run")
        with patch.object(Trainer, "commit", die):
            with self.assertRaisesRegex(RuntimeError, "injected incomplete"):
                self.run_spec(NoArchives())
        result = self.run_spec(NoArchives())
        def files(root):
            return {str(path.relative_to(root)): path.read_bytes()
                    for path in root.rglob("*") if path.is_file()}
        actual = files(Path(self.tmp.name, "runs", "disposable", result.run_id))
        with tempfile.TemporaryDirectory() as tmp:
            other = LocalStore(tmp)
            # The same content-addressed plans, initial state and probe data.
            for source in Path(self.tmp.name, "cas").rglob("*"):
                if source.is_file():
                    target = Path(tmp, source.relative_to(self.tmp.name))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            report = run_experiment(self.spec, self.schema, other, {}, NoArchives(),
                                    subdir="disposable", checkpointing=self.policy)
            expected = files(Path(tmp, "runs", "disposable", report.run_id))
        self.assertEqual(actual, expected)

    def test_recompute_mode_refuses_named_weights_and_store_delivery(self):
        with self.assertRaisesRegex(ValueError, "recompute"):
            Checkpointing(every=1, delivery="store", state="recompute")
        named = replace(self.spec, policy=replace(self.spec.policy, bank={
            "student0": replace(self.spec.policy.bank["student0"], out="weights/final")}))
        with self.assertRaisesRegex(ValueError, "without named"):
            run_experiment(named, self.schema, self.store, {}, NoArchives(), checkpointing=self.policy)
        self.assertEqual(Checkpointing.from_row(self.policy.row()), self.policy)
        self.assertEqual(Checkpointing(every=3).row(), {"every": 3, "delivery": "wire"})

    def test_child_client_returns_versions_and_probes_but_no_snapshot(self):
        run = self.store.open_run("parent", {"run_id": "parent"}, subdir="test")
        learner, arbiter = NoArchives(), Arbiter()
        arbiter.attach(learner, label="learner")
        client = TrainingClient(learner, arbiter, tenant="parent", refs=RefReader(self.store, run),
                                schema=self.schema, seed=17, microbatch_tokens=128)
        async def work():
            result = await client.run(self.spec, checkpoint_every=1, save_state=False)
            self.assertEqual(result.versions, {"student0": 3})
            self.assertEqual((await client.probes(result.run_ref, 3))["row_ids"], ["doc/9"])
            with self.assertRaisesRegex(ValueError, "no checkpoint"):
                await client.snapshot(result.run_ref)
            self.assertNotIn(result.run_id, learner._tenants)
        asyncio.run(work())

    def test_identical_children_of_concurrent_parents_have_distinct_tenancies(self):
        learner, arbiter = NoArchives(), Arbiter()
        arbiter.attach(learner, label="learner")
        clients = [TrainingClient(learner, arbiter, tenant=name,
            refs=RefReader(self.store, self.store.open_run(name, {"run_id": name}, subdir="parents")),
            schema=self.schema, seed=17, microbatch_tokens=128) for name in ("first", "second")]
        async def work():
            results = await asyncio.gather(*(client.run(self.spec, checkpoint_every=1, save_state=False)
                                             for client in clients))
            self.assertNotEqual(results[0].run_id, results[1].run_id)
            self.assertEqual([result.versions for result in results], [{"student0": 3}] * 2)
            self.assertEqual([self.store.peek_manifest(result.run_ref)["parent_run"] for result in results],
                             ["parents/first", "parents/second"])
            self.assertEqual(learner._tenants, {})
        asyncio.run(work())
