"""ModalVolumeStore: the commit discipline over a recorded volume.

The property under test: staged writes persist EXACTLY at the protocol's
durable points — run creation, the ledger line (the commit point, sealing
everything the update staged), and eval output — never per blob or per wave.
"""

from __future__ import annotations

import tempfile
import unittest
import json
import concurrent.futures
import threading
from types import SimpleNamespace
from unittest.mock import patch

from rlstack import ModalVolumeStore


class RecordingVolume:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1

    def iterdir(self, path, *, recursive=True):
        return iter(())

    def read_file(self, key):
        raise FileNotFoundError(key)


class ModalVolumeStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.volume = RecordingVolume()
        self.store = ModalVolumeStore(tmp.name, volume=self.volume)

    def test_waiting_writers_share_a_commit_that_has_not_started(self):
        """One commit covers every writer already waiting behind a volume read."""
        release_read = threading.Event()
        self.store._volume_thread.submit(release_read.wait)
        waiting = threading.Condition()
        count = 0
        errors = []
        original_result = concurrent.futures.Future.result

        def result(future, timeout=None):
            nonlocal count
            if threading.current_thread().name.startswith("persist-"):
                with waiting:
                    count += 1
                    waiting.notify_all()
            return original_result(future, timeout=timeout)

        def persist(name):
            try:
                self.store._write(f"staged/{name}", name.encode())
                self.store._persist()
            except Exception as error:
                errors.append(error)

        writers = [threading.Thread(target=persist, args=(name,), name=f"persist-{name}")
                   for name in ("a", "b")]
        with patch.object(concurrent.futures.Future, "result", result):
            for writer in writers:
                writer.start()
            try:
                with waiting:
                    self.assertTrue(waiting.wait_for(lambda: count == 2, timeout=2))
                self.assertEqual(self.volume.commits, 0)
            finally:
                release_read.set()
                for writer in writers:
                    writer.join(timeout=2)
        self.assertFalse(any(writer.is_alive() for writer in writers))
        self.assertEqual(errors, [])
        self.assertEqual(self.volume.commits, 1)

    def test_writer_arriving_during_a_commit_waits_for_the_next_commit(self):
        """A commit already in progress cannot acknowledge a later writer."""
        started = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        second_waiting = threading.Event()
        original_result = concurrent.futures.Future.result

        def result(future, timeout=None):
            if threading.current_thread().name == "second-writer":
                second_waiting.set()
            return original_result(future, timeout=timeout)

        instrument = patch.object(concurrent.futures.Future, "result", result)
        instrument.start()
        self.addCleanup(instrument.stop)

        class HeldVolume(RecordingVolume):
            def commit(self):
                index = self.commits
                super().commit()
                started[index].set()
                if not release[index].wait(timeout=3):
                    raise TimeoutError("test did not release the commit")

        self.volume = HeldVolume()
        self.store._volume = self.volume
        errors = []

        def persist():
            try:
                self.store._persist()
            except Exception as error:
                errors.append(error)

        first = threading.Thread(target=persist)
        second = threading.Thread(target=persist, name="second-writer")
        first.start()
        try:
            self.assertTrue(started[0].wait(timeout=2))
            second.start()
            self.assertTrue(second_waiting.wait(timeout=2))
            release[0].set()
            self.assertTrue(started[1].wait(timeout=2))
            first.join(timeout=2)
            self.assertFalse(first.is_alive())
            self.assertTrue(second.is_alive())
        finally:
            for event in release:
                event.set()
            first.join(timeout=2)
            if second.ident is not None:
                second.join(timeout=2)
        self.assertEqual(errors, [])
        self.assertEqual(self.volume.commits, 2)

    def test_adoption_refreshes_a_stale_or_missing_ledger_before_append(self):
        """Both mounted-old and read-through-only histories retain every commit."""
        from pathlib import Path

        home = 'runs/parent'
        committed = b'{"update":1,"versions":{"pi":1}}\n{"update":2,"versions":{"pi":2}}\n'
        files = {f'{home}/manifest.json': b'{"run_id":"parent"}',
                 f'{home}/ledger.jsonl': committed,
                 f'{home}/adapters/pi@2.bin': b'sealed adapter'}

        class CommittedVolume(RecordingVolume):
            def iterdir(self, path, *, recursive=True):
                if path == '/':
                    return iter((SimpleNamespace(path='runs'),))
                return iter(SimpleNamespace(path=key) for key in files)

            def read_file(self, key):
                return iter((files[key],))

        for stale in (True, False):
            with self.subTest(stale=stale), tempfile.TemporaryDirectory() as folder:
                store = ModalVolumeStore(folder, volume=CommittedVolume())
                if stale:
                    for key, data in files.items():
                        p = Path(folder) / key
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_bytes(data if not key.endswith('ledger.jsonl') else committed.splitlines(keepends=True)[0])
                run = store.open_run('parent', manifest={'run_id': 'parent'})
                self.assertEqual(run.ledger_tail()['update'], 2)
                self.assertEqual(run.read_blob('adapters', 'pi', 2), b'sealed adapter')
                run.append_ledger({'update': 3, 'versions': {'pi': 3}})
                self.assertEqual([e['update'] for e in run.read_ledger()], [1, 2, 3])

    def test_adoption_refuses_an_unavailable_committed_ledger(self):
        """A failed observation cannot authorize resuming an older checkpoint."""
        run = self.store.open_run('parent', manifest={'run_id': 'parent'})
        run.append_ledger({'update': 1, 'versions': {}})
        before = self.store.path_of(run.ledger_key).read_bytes()

        class UnavailableVolume(RecordingVolume):
            def iterdir(self, path, *, recursive=True):
                raise TimeoutError('committed view unavailable')

        self.store._volume = UnavailableVolume()
        with self.assertRaises(TimeoutError):
            self.store.open_run('parent', manifest={'run_id': 'parent'})
        self.assertEqual(self.store.path_of(run.ledger_key).read_bytes(), before)

    def test_warm_start_finds_a_committed_parent_absent_from_the_mount(self):
        from rlstack.runner.loop import warm_start_source, sealed_payloads
        from rlstack import WarmStart

        home = "runs/elsewhere/parent"
        files = {f"{home}/manifest.json": b'{"run_id":"parent"}',
                 f"{home}/ledger.jsonl": json.dumps(
                     {"update": 1, "versions": {"pi": 1}}).encode() + b"\n",
                 f"{home}/adapters/pi@1.bin": b"trained adapter"}

        class CommittedVolume(RecordingVolume):
            def iterdir(self, path, *, recursive=True):
                if path == "/":
                    return iter((SimpleNamespace(path="runs"),))
                return iter(SimpleNamespace(path=key) for key in files)

            def read_file(self, key):
                if key not in files:
                    raise FileNotFoundError(key)
                return iter((files[key],))

        self.store._volume = CommittedVolume()
        parent, version = warm_start_source(WarmStart("store://elsewhere/parent@1"), self.store)
        self.assertEqual(sealed_payloads(parent, version, {}, {"pi"}),
                         {"pi": b"trained adapter"})
        self.assertFalse(self.store.path_of(home).exists())
        self.assertEqual(self.store._volume.commits, 0)

    def test_staged_work_persists_only_at_the_durable_points(self) -> None:
        run = self.store.open_run("r1", manifest={"run_id": "r1"})
        created = self.volume.commits          # manifest + ledger creation
        self.assertGreaterEqual(created, 1)

        # an update's staged work: no commits until the ledger line
        run.write_wave(1, [{"task": "t0"}])
        run.write_postdata(1, {"reward": [1.0]})
        run.write_blob("adapters", "pi", 1, b"delta")
        run.write_blob("optim", "pi", 1, b"moments")
        self.assertEqual(self.volume.commits, created)

        run.append_ledger({"update": 1, "versions": {"pi": 1}})
        self.assertEqual(self.volume.commits, created + 1)   # THE commit point

        self.store.append_measurement_point(                 # observation
            "r1", "heldout", {"update": 1, "means": {}})     # outside the run
        self.assertEqual(self.volume.commits, created + 2)

    def test_without_a_volume_it_is_just_a_local_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ModalVolumeStore(tmp)                    # volume=None
            run = store.open_run("r1", manifest={"run_id": "r1"})
            run.append_ledger({"update": 1, "versions": {}})
            self.assertEqual(run.ledger_tail()["update"], 1)

    def test_retention_does_not_fetch_checkpoints_it_already_deleted(self):
        """An old optimizer stays absent without one remote miss per update."""
        from rlstack.data.stores.retention import DEFAULT_RETENTION

        reads = []

        class CountingVolume(RecordingVolume):
            def read_file(self, key):
                reads.append(key)
                raise FileNotFoundError(key)

        self.store._volume = CountingVolume()
        run = self.store.open_run("retained", manifest={"run_id": "retained"})
        for version in (1, 2):
            run.write_blob("optim", "pi", version, b"moments")
            run.append_ledger({"update": version, "versions": {"pi": version}})
        run.sweep(DEFAULT_RETENTION)
        reads.clear()
        run.sweep(DEFAULT_RETENTION)
        self.assertEqual(reads, [])
        # Reopening cannot reintroduce the repeated remote misses either.
        reopened = self.store.open_run("retained")
        reads.clear()
        reopened.sweep(DEFAULT_RETENTION)
        self.assertEqual(reads, [])


if __name__ == "__main__":
    unittest.main()
