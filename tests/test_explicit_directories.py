"""Known run addresses never discover the store, including across containers."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from common import arith_spec, arith_store
from rlstack import FakeEngine, FakeLearner, Host, LocalStore, ModalVolumeStore, WarmStart, fake_qwen_schema
from rlstack.data.stores.base import StoreError
from rlstack.observe.cache import CachedReadStore
from rlstack.observe.locate import Root
from rlstack.observe.ui import api, ui_app
from rlstack.observe.views import runs_data
from rlstack.runner.desk import Desk
from rlstack.runner.loop import run_experiment, warm_start_source, sealed_payloads
from rlstack.runner.refs import RefReader
from rlstack.runner.remote import HostService, LocalTransport, RemoteHost


class DirectVolume:
    """A remote committed view that makes any global enumeration a failure."""
    def __init__(self, files=None):
        self.files = files or {}
        self.reads = []

    def iterdir(self, *args, **kwargs):
        raise AssertionError("direct addressing must not enumerate the volume")

    def read_file(self, key):
        self.reads.append(key)
        if key not in self.files:
            raise FileNotFoundError(key)
        return iter((self.files[key],))

    def commit(self):
        pass


class ExplicitDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = LocalStore(self.tmp.name)

    def test_new_remote_run_and_wrong_resume_do_not_enumerate_any_history(self):
        volume = DirectVolume()
        store = ModalVolumeStore(self.tmp.name, volume=volume)
        self.addCleanup(store._volume_thread.shutdown)
        for number in range(8):
            rid = f"new-{number}"
            run = store.open_run(rid, {"run_id": rid}, subdir="campaign/new")
            self.assertEqual(run.run_dir, f"runs/campaign/new/{rid}")
        with self.assertRaisesRegex(StoreError, "runs/wrong/new-0"):
            store.open_run("new-0", {"run_id": "new-0"}, subdir="wrong", create=False)
        self.assertFalse(store.path_of("runs/wrong").exists())
        self.assertTrue(all(key.endswith("/manifest.json") for key in volume.reads))

    def test_remote_parent_and_replay_read_only_the_named_paths(self):
        home = "runs/archive/parent"
        volume = DirectVolume({
            f"{home}/manifest.json": b'{"run_id":"parent"}',
            f"{home}/ledger.jsonl": b'{"update":1,"versions":{"pi":1}}\n',
            f"{home}/adapters/pi@1.bin": b"sealed-parent",
        })
        store = ModalVolumeStore(self.tmp.name, volume=volume)
        self.addCleanup(store._volume_thread.shutdown)
        parent, version = warm_start_source(WarmStart("store://archive/parent@1"), store)
        self.assertEqual(parent.run_id, "parent")
        self.assertEqual(sealed_payloads(parent, version, {}, {"pi"}), {"pi": b"sealed-parent"})
        self.assertTrue(all(key.startswith(home + "/") for key in volume.reads))
        with self.assertRaises(FileNotFoundError):
            warm_start_source(WarmStart("store://parent@1"), store)

        local_parent = self.store.open_run("source", {"run_id": "source"}, subdir="teachers")
        local_parent.write_wave(1, [{"group": "g", "turns": []}])
        consumer = self.store.open_run("consumer", {"run_id": "consumer"})
        with patch.object(self.store, "_run_directories", side_effect=AssertionError("discovery")):
            rows = RefReader(self.store, consumer).rows("store://teachers/source/waves/1")
        self.assertEqual(rows, [{"group": "g", "turns": []}])

    def test_browsing_does_not_change_the_meaning_of_a_bare_id(self):
        self.store.open_run("same", {"run_id": "same", "value": 1}, subdir="a")
        self.store.open_run("same", {"run_id": "same", "value": 2}, subdir="b")
        self.assertEqual(self.store.list_runs(), ["a/same", "b/same"])
        rows = runs_data([self.store])
        self.assertEqual({row["run_ref"] for row in rows}, {"a/same", "b/same"})
        self.assertIsNone(self.store.peek_manifest("same"))
        with patch.object(self.store, "_run_directories", side_effect=AssertionError("discovery")):
            self.assertEqual(self.store.peek_manifest("a/same")["value"], 1)
            self.assertEqual(self.store.peek_manifest("b/same")["value"], 2)
            body, status = api([Root("", CachedReadStore(self.store))],
                               ["api", "run", "a/same", "page"], "", None)
        self.assertEqual(status, "200 OK")
        self.assertEqual(body["run"]["run_ref"], "a/same")
        from test_ui import call
        with patch.object(self.store, "_run_directories", side_effect=AssertionError("discovery")):
            status, _, raw = call(ui_app([self.store]), "/api/run/same/page?root=&subdir=a")
        self.assertEqual(status, "200 OK")
        self.assertEqual(json.loads(raw)["row"]["run_ref"], "a/same")


    def test_sync_runner_carries_subdir_and_preserves_root_resume_bytes(self):
        store, train, _ = arith_store(self.tmp.name)
        schema = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
        report = run_experiment(arith_spec(train), schema, store, FakeEngine(),
                                FakeLearner(), subdir="sync/nested")
        ref = f"sync/nested/{report.run_id}"
        ledger = store._read(f"runs/{ref}/ledger.jsonl")
        self.assertIsNone(store.peek_manifest(report.run_id))
        again = run_experiment(arith_spec(train), schema, store, FakeEngine(),
                              FakeLearner(), subdir="sync/nested", resume=True)
        self.assertEqual(again.run_id, report.run_id)
        self.assertEqual(store._read(f"runs/{ref}/ledger.jsonl"), ledger)
        with self.assertRaisesRegex(StoreError, "runs/wrong/"):
            run_experiment(arith_spec(train), schema, store, FakeEngine(),
                           FakeLearner(), subdir="wrong", resume=True)

    def test_remote_resume_refuses_wrong_directory_before_accepting(self):
        store, train, _ = arith_store(self.tmp.name)
        schema = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
        host = Host("explicit", engines=(FakeEngine(),), learner=FakeLearner(),
                    store=store, schema_for=lambda base: schema)
        remote = RemoteHost(LocalTransport(HostService(host)))

        async def drive():
            first = await remote.adopt(arith_spec(train), subdir="right")
            await host._adoptions[first["run_id"]]
            wrong = await remote.adopt(arith_spec(train), subdir="wrong", resume=True)
            right = await remote.adopt(arith_spec(train), subdir="right", resume=True)
            await host._adoptions[right["run_id"]]
            return first, wrong, right

        first, wrong, right = asyncio.run(drive())
        self.assertFalse(wrong["accepted"], wrong)
        self.assertIn("runs/wrong/", wrong["error"])
        self.assertEqual(right["run_ref"], first["run_ref"])
        self.assertFalse(store.path_of("runs/wrong").exists())


class RecoveryGenerationTest(unittest.TestCase):
    def test_registration_skips_old_parked_frames_until_explicit_resubmission(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalStore(directory)
            frame = {"spec": {}, "subdir": "old"}
            store.append_fleet_event({"event": "place", "run_id": "r", "frame": frame})
            desk = Desk(store, host_for=lambda address: None, recovery_generation="new-campaign")
            desk.park("r", "historical paused work")
            move = AsyncMock(return_value={"rerouted": True})
            with patch.object(desk, "reroute", move), patch.object(desk, "finished", return_value=False):
                asyncio.run(desk.serve("metal", {"name": "new-metal", "gpu": "L4", "devices": 1,
                                                  "vram_gb": 24, "idle_s": 300}))
                move.assert_not_awaited()
                desk.journal_intent("wrong-frame-key", frame)
                self.assertEqual(asyncio.run(desk.retry_parked()), {})
                from rlstack.runner.desk import submit_key
                desk.journal_intent(submit_key(frame), frame)
                self.assertEqual(asyncio.run(desk.retry_parked()), {"r": "rerouted"})
            rebuilt = Desk.from_journal(store, host_for=lambda address: None,
                                        recovery_generation="new-campaign")
            self.assertEqual(rebuilt.recovery_keys(), desk.recovery_keys())
            other = Desk.from_journal(store, host_for=lambda address: None,
                                      recovery_generation="next-campaign")
            self.assertEqual(other.recovery_keys(), set())
