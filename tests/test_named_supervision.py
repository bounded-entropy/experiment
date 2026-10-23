"""Named checkpoint outputs connect ordinary sealed-data runs through the store."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

from common import arith_store, sealed
from rlstack.data.trajectory import trajectory_to_row
from rlstack.policy.siteschema import fake_qwen_schema
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.fakes import FakeLearner
from rlstack.runner.loop import experiment_identity, named_warm_start, run_experiment_async
from rlstack.runner.roles.base import StopRequest
from rlstack.spec.specs import AdapterSpec, WarmStart
from rlstack.training.supervision import SftStage, supervised_spec

BASE = "Qwen/Qwen3-0.6B"
SCHEMA = fake_qwen_schema(4, base=BASE)
FOLDER = "family/supervision"


class NamedSupervisionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.store, _, _ = arith_store(root.name)
        self.learner, self.arbiter = FakeLearner(), Arbiter()
        self.arbiter.attach(self.learner, label="learner")

    def spec(self, text="4", name="source"):
        row = trajectory_to_row(sealed("fact/" + text, text))
        spec = supervised_spec(BASE, AdapterSpec("lora", "layers.0.self_attn.q_proj", {"r": 2}),
            [[row]], SftStage(2, 1, 1e-4, "constant"), 7, self.store.cas_put)
        return replace(spec, policy=replace(spec.policy, bank={
            entry: replace(adapter, out=name) for entry, adapter in spec.policy.bank.items()}))

    async def run_spec(self, spec, **kwargs):
        return await run_experiment_async(spec, SCHEMA, self.store, {}, self.learner,
            arbiter=self.arbiter, subdir=FOLDER, checkpointing=Checkpointing(every=2), **kwargs)

    async def test_dependent_waits_without_install_or_admission_then_loads_named_bytes(self):
        producer = self.spec()
        child = replace(self.spec("5", "child"), init=WarmStart(
            f"named://{FOLDER}#source", map={"adapter": "student0"}))
        child_id = experiment_identity(child, SCHEMA)
        pending = asyncio.create_task(self.run_spec(child))
        try:
            for _ in range(100):
                if self.store.named_writer(FOLDER, "child") == f"{FOLDER}/{child_id}":
                    break
                await asyncio.sleep(0.001)
            self.assertNotIn(child_id, self.learner._tenants)
            self.assertEqual(self.arbiter.in_flight(), 0)
            self.assertFalse(pending.done())
            self.assertEqual(self.store.named_writer(FOLDER, "child"), f"{FOLDER}/{child_id}")
            source = await self.run_spec(producer)
            completed = await asyncio.wait_for(pending, 3)
        finally:
            if not pending.done():
                pending.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await pending
        self.assertEqual(completed.completed, 2)
        run = self.store.open_run(completed.run_id, subdir=FOLDER)
        self.assertEqual(run.read_blob("adapters", "student0", 0), self.store.read_named(FOLDER, "source"))
        meta = self.store.named_meta(FOLDER, "child")
        self.assertEqual(meta["parent"], f"named://{FOLDER}#source")
        self.assertEqual(meta["loss"], "independent_sft")
        self.assertEqual(self.store.named_meta(FOLDER, "source")["run_ref"], f"{FOLDER}/{source.run_id}")

    async def test_named_resume_uses_checkpoint_even_if_source_disappears(self):
        await self.run_spec(self.spec())
        child = replace(self.spec("5", "child"), init=WarmStart(
            f"named://{FOLDER}#source", map={"adapter": "student0"}))
        report = await self.run_spec(child)
        with patch.object(self.store, "read_named", side_effect=AssertionError("must restore checkpoint")):
            resumed = await self.run_spec(child, resume=True)
        self.assertEqual(resumed.completed, report.completed)
        self.assertEqual(resumed.resumed_from, 2)

    async def test_failed_final_checkpoint_replays_and_preserves_named_provenance(self):
        spec = self.spec()
        original = self.store._append_line
        failed = False
        def crash(key, line):
            nonlocal failed
            if key.endswith("checkpoints.jsonl") and '"update":2' in line and not failed:
                failed = True
                raise RuntimeError("after named export, before final checkpoint")
            return original(key, line)
        with patch.object(self.store, "_append_line", crash):
            with self.assertRaisesRegex(RuntimeError, "before final checkpoint"):
                await self.run_spec(spec)
        first = self.store.read_named(FOLDER, "source")
        meta = self.store.named_meta(FOLDER, "source")
        report = await self.run_spec(spec, resume=True)
        self.assertEqual(report.completed, 2)
        self.assertEqual(self.store.read_named(FOLDER, "source"), first)
        self.assertEqual(self.store.named_meta(FOLDER, "source"), meta)
        self.assertEqual(self.store.peek_checkpoints(f"{FOLDER}/{report.run_id}")[-1]["update"], 2)

    async def test_waiting_named_start_honors_stop_before_allocating_adapter(self):
        child = replace(self.spec("5", "child"), init=WarmStart(
            f"named://{FOLDER}#unknown", map={"adapter": "student0"}))
        stop = StopRequest()
        task = asyncio.create_task(self.run_spec(child, stop=stop))
        await asyncio.sleep(0)
        stop.request("test before producer exists")
        report = await asyncio.wait_for(task, 3)
        self.assertTrue(report.stopped)
        self.assertEqual(report.completed, 0)
        self.assertTrue(stop.drained)
        self.assertEqual(self.learner._tenants, {})

    async def test_named_sources_validate_base_and_explicit_map(self):
        self.store.write_named(FOLDER, "source", b"payload", {"base": BASE})
        init = WarmStart(f"named://{FOLDER}#source", map={"adapter": "student0"})
        self.assertEqual(named_warm_start(init, self.store, {"student0"}, BASE).adapters,
                         {"student0": b"payload"})
        with self.assertRaisesRegex(ValueError, "belongs to base"):
            named_warm_start(init, self.store, {"student0"}, "another")
        with self.assertRaisesRegex(ValueError, "maps source"):
            named_warm_start(replace(init, map={}), self.store, {"student0"}, BASE)
        with self.assertRaisesRegex(ValueError, "no optimizer moments"):
            WarmStart(init.policy, optim="load")
        with self.assertRaisesRegex(ValueError, "named://"):
            WarmStart("named://source")

    async def test_named_outputs_refuse_missing_writer_duplicate_and_invalid_names(self):
        from rlstack.spec.validate import check_named_outputs_have_one_trainer_writer
        spec = self.spec()
        self.assertEqual(check_named_outputs_have_one_trainer_writer(spec, SCHEMA), [])
        self.assertEqual(check_named_outputs_have_one_trainer_writer(replace(spec, algo=None), SCHEMA)[0].code,
                         "output-without-trainer")
        entry = spec.policy.bank["student0"]
        duplicate = replace(spec, policy=replace(spec.policy, bank={"a": entry, "b": entry}))
        self.assertEqual(check_named_outputs_have_one_trainer_writer(duplicate, SCHEMA)[0].code,
                         "duplicate-output-name")
        invalid = replace(spec, policy=replace(spec.policy, bank={"a": replace(entry, out="bad//name")}))
        self.assertEqual(check_named_outputs_have_one_trainer_writer(invalid, SCHEMA)[0].code, "bad-output-name")


if __name__ == "__main__":
    unittest.main()
