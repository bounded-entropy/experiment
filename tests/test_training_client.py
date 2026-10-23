"""Normal child training and read-only measurement share resident custody safely."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import tempfile
import threading
import unittest
from unittest.mock import patch

from common import arith_spec, arith_store, sealed
from rlstack.client import GradientRow
from rlstack.data.stores.base import NamedAdapterConflict
from rlstack.data.trajectory import trajectory_to_row
from rlstack.policy.siteschema import fake_qwen_schema, resolve
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.fakes import FakeLearner
from rlstack.runner.gradient_measurement import gradient_row_record
from rlstack.runner.loop import experiment_identity, parameterization_of
from rlstack.runner.meters import HostJournal
from rlstack.runner.refs import RefReader
from rlstack.runner.training_client import TrainingClient
from rlstack.spec.specs import AdapterSpec
from rlstack.training.supervision import SftStage, supervised_spec
from test_resume import snapshot

BASE = "Qwen/Qwen3-0.6B"
SCHEMA = fake_qwen_schema(4, base=BASE)
ADAPTER = AdapterSpec("lora", "layers.0.self_attn.q_proj", {"r": 2})
REFERENCES = (GradientRow("support/Q0", (1, 2, 3), (0, 0, 1)),
              GradientRow("support/Q1", (4, 5, 6, 7), (0, 1, 0, 1)))
CANDIDATES = ((GradientRow("latent/0/task/7", (8, 9, 10), (0, 1, 1)),),)


class RecordingLearner(FakeLearner):
    def __init__(self):
        super().__init__()
        self.installs, self.uninstalls, self.loads, self.measurements = [], [], [], []
        self.steps_seen = 0

    def install(self, tenant, parameterization):
        self.installs.append((tenant, parameterization))
        return super().install(tenant, parameterization)

    def uninstall(self, tenant):
        self.uninstalls.append(tenant)
        return super().uninstall(tenant)

    def load(self, tenant, adapters, optim):
        self.loads.append((tenant, dict(adapters), optim))
        return super().load(tenant, adapters, optim)

    def optim_step(self, tenant, lr_scales=None):
        self.steps_seen += 1
        return super().optim_step(tenant, lr_scales)

    def gradient_cosines(self, tenant, entry, reference_batches, candidate_batches):
        self.measurements.append((tenant, entry, reference_batches, candidate_batches))
        return super().gradient_cosines(tenant, entry, reference_batches, candidate_batches)


class TrainingClientTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.store, self.train, _ = arith_store(root.name)
        self.learner = RecordingLearner()
        self.arbiter = Arbiter()
        self.arbiter.attach(self.learner, label="training")
        parent_spec = arith_spec(self.train)
        self.learner.install("parent", parameterization_of(parent_spec, {
            name: resolve(SCHEMA.sites, adapter.site)
            for name, adapter in parent_spec.policy.bank.items()}))
        self.parent_before = self.learner.emit("parent")
        self.parent = self.store.open_run("parent", {"run_id": "parent"}, subdir="family")
        self.client = TrainingClient(self.learner, self.arbiter, tenant="parent",
            refs=RefReader(self.store, self.parent), schema=SCHEMA, seed=31,
            microbatch_tokens=6, journal=HostJournal(self.store, "host"))

    async def asyncTearDown(self):
        await self.client.close()

    def child_spec(self, store=None):
        store = store or self.store
        datasets = [[trajectory_to_row(sealed("fact/A", "4"))],
                    [trajectory_to_row(sealed("fact/B", "5"))]]
        return supervised_spec(BASE, ADAPTER, datasets,
            (SftStage(3, 1, 1e-4, "constant"), SftStage(1, 1, 2e-4, "constant")),
            31, store.cas_put, microbatch_tokens=64)

    async def measure(self, payload=None, adapter=ADAPTER):
        return await self.client.gradient_cosines(payload, REFERENCES, CANDIDATES, adapter=adapter)

    def assert_parent_unchanged(self):
        self.assertEqual(self.learner.emit("parent"), self.parent_before)
        self.assertEqual(self.arbiter.in_flight(), 0)

    async def test_measurement_is_lazy_isolated_and_reuses_unchanged_initialization(self):
        self.assertEqual(set(self.learner._tenants), {"parent"})
        with patch.object(self.learner, "tokenize", side_effect=AssertionError("must preserve exact IDs")):
            first, second = await self.measure(), await self.measure()
        self.assertEqual(first, second)
        installations = [p for tenant, p in self.learner.installs if tenant == "parent:measurement"]
        self.assertEqual(len(installations), 1)
        self.assertEqual(installations[0].entries[0].adapter_type, "lora")
        self.assertEqual(installations[0].entries[0].init["seed"], 31)
        self.assertIsNone(installations[0].entries[0].route)
        self.assertEqual(self.learner.steps_seen, 0)
        self.assert_parent_unchanged()
        await self.client.close()
        await self.client.close()
        self.assertEqual(set(self.learner._tenants), {"parent"})

    async def test_payload_and_adapter_change_reset_only_measurement_tenant(self):
        await self.measure()
        await self.measure(b"fake-delta:adapter:loaded")
        await self.measure(b"fake-delta:adapter:loaded")
        await self.measure(b"fake-delta:adapter:changed")
        await self.measure(adapter=replace(ADAPTER, init={"r": 4}))
        self.assertEqual(sum(tenant == "parent:measurement" for tenant, _ in self.learner.installs), 4)
        self.assertEqual(self.learner.uninstalls.count("parent:measurement"), 3)
        self.assertEqual([call[1] for call in self.learner.loads],
                         [{"adapter": b"fake-delta:adapter:loaded"}, {"adapter": b"fake-delta:adapter:changed"}])
        self.assertEqual(self.learner.steps_seen, 0)
        self.assert_parent_unchanged()

    async def test_measurement_records_identifiers_masks_and_committed_parent_update(self):
        self.client.update = 7
        await self.measure()
        tenant, entry, refs, candidates = self.learner.measurements[-1]
        self.assertEqual((tenant, entry), ("parent:measurement", "adapter"))
        self.assertEqual(tuple(token for batch in refs for token in batch.token_ids), (1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(tuple(value for batch in refs for value in batch.loss_mask), (0, 0, 1, 0, 1, 0, 1))
        self.assertTrue(all(batch.documents_in_update == 2 for batch in refs))
        self.assertEqual(candidates[0][0].loss_mask, (0, 1, 1))
        event, = self.store.read_host_log("host")
        self.assertEqual((event["run_id"], event["update"]), ("parent", 7))
        self.assertEqual(event["references"], [gradient_row_record(row) for row in REFERENCES])
        self.assertEqual(event["candidates"][0][0]["row_id"], "latent/0/task/7")
        self.assertEqual(event["student_initialization"]["adapter"]["adapter_type"], "lora")
        self.assertIn("t", event)
        self.assertEqual(len(event["references"][0]["token_mask_sha256"]), 64)

    async def test_measurement_failure_uninstalls_partial_initialization(self):
        with patch.object(self.learner, "load", side_effect=ValueError("bad payload")):
            with self.assertRaisesRegex(ValueError, "bad payload"):
                await self.measure(b"invalid")
        self.assertIsNone(self.client.measurement_key)
        self.assertEqual(set(self.learner._tenants), {"parent"})
        self.assert_parent_unchanged()

    async def test_cancelled_measurement_keeps_lock_and_admission_until_worker_returns(self):
        entered, release = threading.Event(), threading.Event()
        original = self.learner.gradient_cosines
        def blocked(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test failed to release worker")
            return original(*args)
        with patch.object(self.learner, "gradient_cosines", blocked):
            task = asyncio.create_task(self.measure())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertTrue(self.client.lock.locked())
                self.assertEqual(self.arbiter.in_flight(), 1)
                closed = asyncio.create_task(self.client.close())
                await asyncio.sleep(0)
                self.assertFalse(closed.done())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await closed
        self.assertEqual(set(self.learner._tenants), {"parent"})
        self.assert_parent_unchanged()

    async def test_run_uses_normal_routed_updates_checkpoints_and_cleans_live_child(self):
        spec = self.child_spec()
        result = await self.client.run(spec, checkpoint_every=2)
        self.assertEqual(result.completed, 3)
        self.assertEqual(result.versions, {"student0": 3, "student1": 1})
        self.assertEqual(result.run_ref, f"family/children/parent/{result.run_id}")
        checkpoints = self.store.peek_checkpoints(result.run_ref)
        self.assertEqual([row["update"] for row in checkpoints], [0, 2, 3])
        saved = await self.client.snapshot(result.run_ref)
        self.assertEqual(saved.versions, result.versions)
        self.assertEqual(set(saved.adapters), {"student0", "student1"})
        self.assertEqual(set(saved.optim), {"student0", "student1"})
        self.assertTrue(saved.optim["student1"].startswith(b"fake-optim:student1:1:"))
        self.assertEqual(set(self.learner._tenants), {"parent"})
        self.assertEqual(self.learner.steps_seen, 3)
        self.assert_parent_unchanged()
        event = [row for row in self.store.read_host_log("host") if row["event"] == "child-training"][-1]
        self.assertEqual(event["checkpoint_every"], 2)
        self.assertEqual(event["child_run_ref"], result.run_ref)
        # Reattaching a completed child reads its checkpoint and performs no update.
        self.assertEqual(await self.client.run(spec, checkpoint_every=2), result)
        self.assertEqual(self.learner.steps_seen, 3)

    async def test_failed_child_releases_tenant_and_resumes_checkpoint_bytes(self):
        spec = self.child_spec()
        original = self.learner.optim_step
        count = 0
        def fail_second(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("injected child failure")
            return original(*args, **kwargs)
        with patch.object(self.learner, "optim_step", fail_second):
            with self.assertRaisesRegex(RuntimeError, "injected child failure"):
                await self.client.run(spec, checkpoint_every=2)
        self.assertEqual(set(self.learner._tenants), {"parent"})
        resumed = await self.client.run(spec, checkpoint_every=2)
        with tempfile.TemporaryDirectory() as root:
            store, _, _ = arith_store(root)
            parent = store.open_run("parent", {"run_id": "parent"}, subdir="family")
            learner = RecordingLearner()
            learner.install("parent", self.learner.installs[0][1])
            arbiter = Arbiter()
            arbiter.attach(learner, label="training")
            client = TrainingClient(learner, arbiter, tenant="parent", refs=RefReader(store, parent),
                schema=SCHEMA, seed=31, microbatch_tokens=6)
            straight = await client.run(self.child_spec(store), checkpoint_every=2)
            self.assertEqual(resumed.run_id, straight.run_id)
            self.assertEqual(snapshot(self.store, resumed.run_ref), snapshot(store, straight.run_ref))
            await client.close()
        self.assert_parent_unchanged()

    async def test_cancelled_child_drains_before_uninstall_and_can_resume(self):
        entered, release = threading.Event(), threading.Event()
        original = self.learner.accumulate
        def blocked(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test failed to release child")
            return original(*args)
        spec = self.child_spec()
        child_id = experiment_identity(replace(spec, parent_run="family/parent"), SCHEMA)
        with patch.object(self.learner, "accumulate", blocked):
            task = asyncio.create_task(self.client.run(spec, checkpoint_every=3))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                task.cancel()
                await asyncio.sleep(0)
                self.assertFalse(task.done())
                self.assertTrue(self.client.lock.locked())
                self.assertEqual(self.arbiter.in_flight(), 1)
                self.assertIn(child_id, self.learner._tenants)
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(set(self.learner._tenants), {"parent"})
        run_ref = f"family/children/parent/{child_id}"
        self.assertEqual(self.store.peek_checkpoints(run_ref)[-1]["update"], 1)
        self.assertEqual((await self.client.run(spec, checkpoint_every=3)).completed, 3)
        self.assert_parent_unchanged()

    async def test_child_validation_precedes_installation(self):
        spec = self.child_spec()
        before = len(self.learner.installs)
        with self.assertRaisesRegex(ValueError, "resident base"):
            await self.client.run(replace(spec, policy=replace(spec.policy, base="another")), checkpoint_every=1)
        with self.assertRaisesRegex(ValueError, "cannot generate"):
            await self.client.run(replace(spec, gen=arith_spec(self.train).gen), checkpoint_every=1)
        self.assertEqual(len(self.learner.installs), before)

    async def test_content_rows_and_named_outputs_use_parent_folder(self):
        rows = [trajectory_to_row(sealed("fact/A", "4")), trajectory_to_row(sealed("fact/B", "5"))]
        payload = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
        uri = await self.client.put(payload)
        self.assertEqual(uri, "cas://" + hashlib.sha256(payload).hexdigest())
        self.assertEqual(await self.client.rows(uri + "#1:2"), rows[1:2])
        with self.assertRaisesRegex(ValueError, "invalid sealed row span"):
            await self.client.rows(uri + "#2:1")
        with self.assertRaisesRegex(ValueError, "content-addressed"):
            await self.client.rows("store://another/waves/1")
        await self.client.publish("answer/A", b"named bytes", {"child_run_ref": "child/A"})
        await self.client.publish("answer/A", b"named bytes", {"child_run_ref": "child/A"})
        self.assertEqual(self.store.read_named("family", "answer/A"), b"named bytes")
        with self.assertRaises(NamedAdapterConflict):
            await self.client.publish("answer/A", b"different bytes", {"child_run_ref": "child/A"})
        with self.assertRaisesRegex(NamedAdapterConflict, "scientific provenance"):
            await self.client.publish("answer/A", b"named bytes", {"child_run_ref": "child/B"})


if __name__ == "__main__":
    unittest.main()
