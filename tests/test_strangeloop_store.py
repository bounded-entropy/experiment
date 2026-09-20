"""Scratch storage: publication, stale readers, single-owner recovery and bytes."""

from __future__ import annotations

import json
import multiprocessing
import http.client
import os
import subprocess
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from rlstack.data.stores import LocalStore, StoreAddress, StoreError, open_store
from rlstack.data.stores.strangeloop import READ_ATTEMPTS
import rlstack.data.stores.strangeloop as strangeloop_store
from rlstack.data.stores.strangeloop import (
    BlobCache, HashedReads, ScratchRejected, DEFAULT_API_BASE, ScratchClient, ScratchCredentials, ScratchEntry,
    StrangeLoopLocalStore, StrangeLoopStore, resolve_scratch_credentials, scratch_key, scratch_path,
)
from rlstack.observe.cache import CachedReadStore
from rlstack.observe.locate import store_for


def read_cached_in_child(base, directory, uri, ready, start, results):
    """A real spawned process with its own store and HTTP client."""
    client = ScratchClient("sl-scratch-account", token="test-token", api_base=base)
    client._note = lambda message: None
    store = StrangeLoopStore(client, hashed_reads=HashedReads(Path(directory), 1_000_000))
    ready.put(True)
    if not start.wait(10):
        raise RuntimeError("reader start timed out")
    results.put(store.cas_get(uri))


def leave_cache_insertion_midway(directory, digest, ready):
    """Killed by the test while holding both locks and an incomplete file."""
    cache = BlobCache(Path(directory), 100, lambda message: None)
    with cache.claim(digest), cache._lock("maintenance"):
        path = cache.path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_name(path.name + ".dead.tmp").write_bytes(b"partial")
        ready.set()
        threading.Event().wait(30)


class MemoryScratch(ScratchClient):
    """One committed tree; deliberately separate from a mount's snapshot."""

    def __init__(self):
        self.volume = "sl-scratch-account"
        self.prefix = "rlstack"
        self.credentials = ScratchCredentials("test-secret", DEFAULT_API_BASE, "test")
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.read_error: Exception | None = None
        self.read_failures: int | None = None  # None: read_error is persistent
        self.write_error: Exception | None = None
        self.write_failures: int | None = None  # None: write_error is persistent
        self.drop_writes = False                # a failing write leaves nothing behind
        self.mismatch = False

    def read(self, key):
        self.calls.append(("read", key))
        self._maybe_fail_read()
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key]

    def write(self, key, data):
        self.calls.append(("write", key))
        failing = self.write_error is not None and (self.write_failures is None or self.write_failures > 0)
        if failing and self.write_failures is not None:
            self.write_failures -= 1
        if not (failing and self.drop_writes):
            self.files[key] = b"wrong bytes" if self.mismatch else data
        if failing:
            raise self.write_error  # a lost acknowledgement, or the API's refusal

    def delete(self, key):
        self.calls.append(("delete", key))
        del self.files[key]

    def _maybe_fail_read(self):
        if self.read_error:
            if self.read_failures is None:
                raise self.read_error
            if self.read_failures > 0:
                self.read_failures -= 1
                raise self.read_error

    def stat(self, key):
        self.calls.append(("stat", key))
        self._maybe_fail_read()
        if key in self.files:
            return ScratchEntry(key, "file", len(self.files[key]))
        if any(path.startswith(key.rstrip("/") + "/") for path in self.files):
            return ScratchEntry(key.rstrip("/"), "dir", 0)
        raise FileNotFoundError(key)

    def list(self, prefix, *, recursive=False):
        self.calls.append(("recursive" if recursive else "list", prefix))
        if self.read_error:
            raise self.read_error
        prefix = prefix.rstrip("/") + "/"
        rows = {}
        for key, data in self.files.items():
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            if recursive or "/" not in rest:
                rows[key] = ScratchEntry(key, "file", len(data))
            else:
                top = prefix + rest.split("/")[0]
                rows[top] = ScratchEntry(top, "dir", 0)
        return [rows[key] for key in sorted(rows)]


class MountedScratch:
    """Publish changed mount files only; stale untouched files stay stale."""

    def __init__(self, root, scratch):
        self.root = Path(root)
        self.scratch = scratch
        self.last = self.snapshot()
        self.stamps = self.file_stamps()
        self.commits = 0
        self.fail = False
        self.noop = False
        self.publications = []

    def snapshot(self):
        return {path.relative_to(self.root).as_posix(): path.read_bytes()
                for path in self.root.rglob("*") if path.is_file()}

    def file_stamps(self):
        return {path.relative_to(self.root).as_posix(): (path.stat().st_ino, path.stat().st_mtime_ns)
                for path in self.root.rglob("*") if path.is_file()}

    def sync(self, command, *, check, timeout, capture_output):
        assert command[0] == "sync" and check and timeout > 0 and capture_output
        self.commits += 1
        if self.fail:
            raise subprocess.TimeoutExpired(command, timeout)
        if self.noop:
            return
        current = self.snapshot()
        stamps = self.file_stamps()
        for key, data in current.items():
            if self.last.get(key) != data or self.stamps.get(key) != stamps[key]:
                self.scratch.files[scratch_key(key)] = data
                self.publications.append((scratch_key(key), data))
        for key in self.last.keys() - current.keys():
            self.scratch.files.pop(scratch_key(key), None)
        self.last = current
        self.stamps = stamps


class ScratchStoreTests(unittest.TestCase):
    def setUp(self):
        self.scratch = MemoryScratch()
        self.store = StrangeLoopStore(self.scratch)
        self._pause = strangeloop_store.READBACK_PAUSE_S
        strangeloop_store.READBACK_PAUSE_S = 0.0
        self.addCleanup(setattr, strangeloop_store, "READBACK_PAUSE_S", self._pause)

    def test_transient_readback_failure_is_retried_and_the_write_never_replayed(self):
        # The write landed; its verifying read fails twice on the wire
        # (a response shorter than its declared byte count) and then answers.
        self.scratch.read_error = StoreError("scratch response ended before its declared byte count")
        self.scratch.read_failures = 2
        self.store._write("file", b"bytes")
        self.assertEqual(self.scratch.calls, [("write", "file"), ("read", "file"),
                                              ("read", "file"), ("read", "file")])
        self.assertEqual(self.scratch.files["file"], b"bytes")
        # A readback that fails every attempt still stops the owner, and
        # the write itself is attempted exactly once.
        self.scratch.calls.clear()
        self.scratch.read_failures = None
        with self.assertRaisesRegex(StoreError, "declared byte count"):
            self.store._write("other", b"more")
        self.assertEqual([c for c in self.scratch.calls if c[0] == "write"], [("write", "other")])
        self.assertEqual(len([c for c in self.scratch.calls if c[0] == "read"]),
                         strangeloop_store.READBACK_ATTEMPTS)
        with self.assertRaisesRegex(StoreError, "stopped"):
            self.store._write("dependent", b"never")

    def test_the_fleet_journal_is_written_off_the_caller_s_thread_in_order(self):
        # the desk appends from its event loop; a slow write must not hold it
        gate = threading.Event()
        original = self.scratch.write
        def slow_write(key, data):
            gate.wait(2.0)
            original(key, data)
        self.scratch.write = slow_write
        started = time.monotonic()
        for i in range(3):
            self.store.append_fleet_event({"event": "tick", "n": i})
        self.assertLess(time.monotonic() - started, 0.5)          # enqueued, not written
        self.assertNotIn("fleet/log.jsonl", self.scratch.files)
        gate.set()
        rows = self.store.read_fleet_log()                          # drains first
        self.assertEqual([r["n"] for r in rows], [0, 1, 2])
        self.assertEqual(self.scratch.files["fleet/log.jsonl"].count(b"\n"), 3)

    def test_a_journal_write_that_fails_stops_the_owner_for_the_next_append(self):
        self.scratch.write_error = StoreError("scratch response ended before its declared byte count")
        self.scratch.drop_writes = True
        self.store.append_fleet_event({"event": "first"})          # returns; the writer fails behind it
        self.store._drain_journal()
        with self.assertRaisesRegex(StoreError, "stopped"):
            self.store.append_fleet_event({"event": "second"})
        with self.assertRaisesRegex(StoreError, "stopped"):
            self.store._persist()

    def test_acknowledgement_requires_matching_readback(self):
        self.store._write("file", b"bytes")
        self.assertEqual(self.scratch.calls[-2:], [("write", "file"), ("read", "file")])
        self.scratch.mismatch = True
        with self.assertRaises(StoreError):
            self.store._write("bad", b"expected")
        before = list(self.scratch.calls)
        with self.assertRaisesRegex(StoreError, "stopped"):
            self.store._write("dependent", b"never")
        self.assertEqual(self.scratch.calls, before)

    def test_lost_write_acknowledgement_is_read_back_never_replayed(self):
        # the write landed but its acknowledgement was lost: the readback
        # shows the offered bytes, so the publication stands — one write
        self.scratch.write_error = TimeoutError("ack lost")
        self.store._append_line("journal", "one")
        self.assertEqual(self.scratch.files["journal"], b"one\n")
        self.assertEqual([c for c in self.scratch.calls if c[0] == "write"], [("write", "journal")])
        # the acknowledgement lost AND the bytes not there: never replayed,
        # the owner stops (the write may still land later)
        self.scratch.calls.clear()
        self.scratch.write_error = TimeoutError("ack lost")
        self.scratch.drop_writes = True
        self.addCleanup(setattr, self.scratch, "drop_writes", False)
        with self.assertRaises(TimeoutError):
            self.store._write("other", b"more")
        self.assertEqual([c for c in self.scratch.calls if c[0] == "write"], [("write", "other")])
        with self.assertRaisesRegex(StoreError, "stopped"):
            self.store._write("dependent", b"never")

    def test_a_rejected_write_is_tried_again_and_a_write_rejected_every_time_stops_the_owner(self):
        from rlstack.data.stores.strangeloop import ScratchRejected, WRITE_ATTEMPTS
        self.scratch.files.clear(); self.scratch.calls.clear()
        # the API answered 500 once: nothing in flight, so the write is tried again
        self.scratch.write_error = ScratchRejected("scratch PUT rejected (HTTP 500)")
        self.scratch.write_failures = 1
        self.scratch.drop_writes = True
        with patch("rlstack.data.stores.strangeloop.time.sleep"):
            self.store._write("file", b"bytes")
        self.assertEqual(self.scratch.files["file"], b"bytes")
        self.assertEqual(len([c for c in self.scratch.calls if c[0] == "write"]), 2)
        # rejected every time: the owner stops after WRITE_ATTEMPTS tries
        self.scratch.calls.clear()
        self.scratch.write_error = ScratchRejected("scratch PUT rejected (HTTP 500)")
        self.scratch.write_failures = None
        with patch("rlstack.data.stores.strangeloop.time.sleep"):
            with self.assertRaises(ScratchRejected):
                self.store._write("other", b"more")
        self.assertEqual(len([c for c in self.scratch.calls if c[0] == "write"]), WRITE_ATTEMPTS)
        with self.assertRaisesRegex(StoreError, "stopped"):
            self.store._write("dependent", b"never")
        # a stopped owner refuses every mutation but still reads
        self.scratch.write_error = None
        with self.assertRaisesRegex(StoreError, "stopped"):
            self.store._append_line("journal", "one")
        with self.assertRaises(StoreError):
            self.store._persist()
        self.assertNotIn("journal", self.scratch.files)

    def test_failed_read_never_authorizes_empty_append(self):
        self.scratch.files["journal"] = b"history\n"
        self.scratch.read_error = TimeoutError("cannot observe")
        with self.assertRaises(TimeoutError):
            self.store._append_line("journal", "next")
        with self.assertRaises(TimeoutError):
            self.store._exists("journal")
        self.assertEqual(self.scratch.files["journal"], b"history\n")
        self.assertNotIn(("write", "journal"), self.scratch.calls)

    def test_appends_from_threads_of_one_owner_keep_every_line(self):
        with ThreadPoolExecutor(max_workers=8) as writers:
            list(writers.map(lambda n: self.store._append_line("journal", str(n)), range(40)))
        self.assertEqual(sorted(map(int, self.store._read("journal").splitlines())), list(range(40)))

    def test_observer_and_spawned_reader_are_read_only(self):
        self.store._write("panels.json", b"{}")
        with patch.object(ScratchClient, "from_locator", return_value=self.scratch):
            readers = [open_store(self.store.address()), store_for(self.store.describe())]
        for reader in readers:
            self.assertEqual(reader._read("panels.json"), b"{}")
            for operation in (lambda: reader._write("bad", b"no"),
                              lambda: reader._append_line("journal", "no"),
                              lambda: reader._delete("panels.json"),
                              reader._persist,
                              lambda: reader._sweep_partial("runs")):
                with self.assertRaises(StoreError):
                    operation()
        self.assertNotIn("test-secret", json.dumps(self.store.address().row()))
        self.assertEqual(StoreAddress.from_row(self.store.address().row()), self.store.address())

    def test_equal_length_replacement_and_deletion_outrank_observer_cache(self):
        reader = CachedReadStore(self.store)
        self.scratch.files["panels.json"] = b"old"
        self.assertEqual(reader._read("panels.json"), b"old")
        self.scratch.files["panels.json"] = b"new"
        self.assertEqual(reader._read("panels.json"), b"new")
        del self.scratch.files["panels.json"]
        with self.assertRaises(FileNotFoundError):
            reader._read("panels.json")
        self.scratch.files["panels.json"] = b"now"
        self.assertEqual(reader._read("panels.json"), b"now")

    def test_empty_authoritative_listing_does_not_blink_back_old_runs(self):
        reader = CachedReadStore(self.store)
        self.scratch.files["runs/r/manifest.json"] = b"{}"
        with patch("rlstack.observe.cache.time.time", return_value=0):
            self.assertEqual(reader.list_runs(), ["r"])
            self.assertEqual(reader._list("runs/"), ["runs/r/manifest.json"])
        self.scratch.files.clear()
        with patch("rlstack.observe.cache.time.time", return_value=3):
            self.assertEqual(reader.list_runs(), [])
            self.assertEqual(reader._list("runs/"), [])

    def test_exact_run_paths_and_shallow_browse(self):
        self.store.open_run("r", {"run_id": "r"}, subdir="a")
        self.scratch.calls.clear()
        with self.assertRaises(StoreError):
            self.store.open_run("r", {"run_id": "r"}, subdir="b", create=False)
        self.assertFalse(any(method in ("list", "recursive") for method, _ in self.scratch.calls))
        self.assertEqual(self.store.run_children(), [{"name": "a", "kind": "folder", "path": "a"}])
        self.assertEqual(self.store.run_children("a"), [{"name": "r", "kind": "run", "path": "a/r"}])
        self.assertEqual(self.store.run_children("a/r"), [])
        self.assertEqual(self.store.list_runs(), ["a/r"])
        self.assertNotIn(("recursive", "runs"), self.scratch.calls)

    def test_transient_stat_failure_after_delete_is_retried(self):
        self.scratch.files["file"] = b"old"
        real_delete = self.scratch.delete

        def delete_then_flaky_stat(key):
            # The delete lands; the two stats that verify it fail on the wire.
            real_delete(key)
            self.scratch.read_error = StoreError("scratch response ended before its declared byte count")
            self.scratch.read_failures = 2
        with patch.object(self.scratch, "delete", side_effect=delete_then_flaky_stat):
            self.store._delete("file")
        self.scratch.read_error = None
        self.assertNotIn("file", self.scratch.files)
        self.assertEqual([c for c in self.scratch.calls if c[0] == "delete"], [("delete", "file")])
        self.store._write("next", b"allowed")

    def test_delete_failure_stops_owner(self):
        self.scratch.files["file"] = b"old"
        with patch.object(self.scratch, "delete", side_effect=TimeoutError("unknown")):
            with self.assertRaises(TimeoutError):
                self.store._delete("file")
        with self.assertRaises(StoreError):
            self.store._write("next", b"not allowed")


class HashedScratchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.scratch = MemoryScratch()
        self.notes = []
        self.scratch._note = self.notes.append
        self.settings = HashedReads(self.directory / "cache", 1000, True)
        self.mount = self.directory / "mount"
        self.store = StrangeLoopStore(self.scratch, hashed_reads=self.settings, mount_root=self.mount)
        self.pauses = patch.multiple(strangeloop_store, READBACK_PAUSE_S=0, WRITE_PAUSE_S=0)
        self.pauses.start()
        self.addCleanup(self.pauses.stop)

    def blob(self, data=b"adapter"):
        digest = self.store.fingerprint(data)
        key = f"cas/{digest}/blob"
        self.scratch.files[key] = data
        return key, digest

    def mounted(self, key, data):
        path = self.mount / scratch_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def test_api_fills_shared_cache_and_fresh_store_needs_no_payload_read(self):
        key, digest = self.blob()
        self.assertEqual(self.store.cas_get(f"cas://{digest}/label"), b"adapter")
        other = StrangeLoopStore(self.scratch, hashed_reads=self.settings)
        self.assertEqual(other.cas_get(f"cas://{digest}"), b"adapter")
        self.assertEqual(self.scratch.calls, [("read", key)])
        self.assertTrue(any("cache:" in note for note in self.notes))

    def test_visible_remote_mount_is_enough_and_never_refreshes(self):
        key, digest = self.blob()
        self.mounted(key, b"adapter")
        self.scratch.files.clear()
        self.assertEqual(self.store.cas_get(f"cas://{digest}"), b"adapter")
        self.assertEqual(self.scratch.calls, [])
        self.assertEqual(self.store._blob_cache.path(digest).read_bytes(), b"adapter")

    def test_mount_uses_encoded_keys_and_no_cache_is_required(self):
        data = b"adapter"
        key = "names/some:adapter"
        self.mounted(key, data)
        store = StrangeLoopStore(self.scratch, hashed_reads=HashedReads(mount_reads=True),
                                 mount_root=self.mount)
        self.assertEqual(store._read_hashed(key, store.fingerprint(data)), data)
        self.assertEqual(self.scratch.calls, [])

    def test_corrupt_cache_and_mount_fall_through_without_rewriting_mount(self):
        key, digest = self.blob()
        self.mounted(key, b"corrupt")
        cache_path = self.store._blob_cache.path(digest)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(b"corrupt")
        self.assertEqual(self.store.cas_get(f"cas://{digest}"), b"adapter")
        self.assertEqual(self.scratch.calls, [("read", key)])
        self.assertEqual(cache_path.read_bytes(), b"adapter")
        self.assertEqual((self.mount / key).read_bytes(), b"corrupt")

    def test_authoritative_corruption_raises_and_is_never_cached_or_repaired(self):
        key, digest = self.blob()
        self.scratch.files[key] = b"corrupt"
        with self.assertRaisesRegex(StoreError, f"expected {digest}, actual"):
            self.store.cas_get(f"cas://{digest}")
        self.assertFalse(self.store._blob_cache.path(digest).exists())
        self.assertEqual(self.scratch.calls, [("read", key)])

    def test_absence_is_not_cached(self):
        key, digest = self.blob()
        del self.scratch.files[key]
        with self.assertRaises(FileNotFoundError):
            self.store.cas_get(f"cas://{digest}")
        self.scratch.files[key] = b"adapter"
        self.assertEqual(self.store.cas_get(f"cas://{digest}"), b"adapter")

    def test_mutable_reads_and_stats_ignore_mount_and_cache(self):
        self.mounted("ledger", b"old")
        self.scratch.files["ledger"] = b"new"
        self.assertEqual(self.store._read("ledger"), b"new")
        self.scratch.files["ledger"] = b"updated"
        self.assertEqual(self.store._read("ledger"), b"updated")
        self.assertEqual(self.store._size("ledger"), 7)
        self.assertEqual(self.scratch.calls, [("read", "ledger"), ("read", "ledger"), ("stat", "ledger")])

    def test_default_settings_use_http_even_when_a_mount_is_supplied(self):
        key, digest = self.blob()
        self.mounted(key, b"adapter")
        store = StrangeLoopStore(self.scratch, hashed_reads=HashedReads(), mount_root=self.mount)
        store.cas_get(f"cas://{digest}")
        self.assertEqual(self.scratch.calls, [("read", key)])

    def test_named_metadata_seals_payload_before_cache_can_serve_it(self):
        payload = b"adapter"
        digest = self.store.fingerprint(payload)
        self.store._blob_cache.put(digest, payload)
        self.assertIsNone(self.store.read_named("family", "m"))
        self.scratch.files["runs/family/names/m.json"] = json.dumps({"sha256": digest}).encode()
        self.assertEqual(self.store.read_named("family", "m"), payload)
        before = list(self.scratch.calls)
        self.assertEqual(self.store.read_named("family", "m"), payload)
        self.assertEqual(self.scratch.calls, before)

    def test_write_named_confirms_payload_by_size_and_meta_in_full(self):
        self.store.write_named("family", "m", b"adapter", {"r": 2})
        payload, meta = "runs/family/names/m.bin", "runs/family/names/m.json"
        self.assertEqual(self.scratch.calls, [("read", meta), ("write", payload), ("stat", payload),
                                               ("write", meta), ("read", meta)])
        self.assertEqual(self.store.read_named("family", "m"), b"adapter")

    def test_hashed_accepted_write_checks_size_without_downloading(self):
        uri = self.store.cas_put(b"adapter")
        key = "cas/" + uri.removeprefix("cas://") + "/blob"
        self.assertEqual(self.scratch.calls, [("stat", key), ("write", key), ("stat", key)])

    def test_hashed_lost_ack_is_observed_but_never_replayed(self):
        self.scratch.write_error = TimeoutError("lost")
        self.store.cas_put(b"adapter")
        self.assertEqual([op for op, key in self.scratch.calls], ["stat", "write", "stat"])

    def test_hashed_rejection_retries_but_permanent_rejection_stops_owner(self):
        self.scratch.write_error = ScratchRejected("refused")
        self.scratch.drop_writes = True
        self.scratch.write_failures = 1
        self.store.cas_put(b"adapter")
        self.assertEqual([op for op, key in self.scratch.calls], ["stat", "write", "write", "stat"])
        self.scratch.write_failures = None
        with self.assertRaises(ScratchRejected):
            self.store.cas_put(b"another")
        with self.assertRaisesRegex(StoreError, "stopped"):
            self.store._write("later", b"no")

    def test_missing_uncertain_payload_stops_owner_and_does_not_seal(self):
        self.scratch.write_error = TimeoutError("lost")
        self.scratch.drop_writes = True
        with self.assertRaises(TimeoutError):
            self.store.write_named("family", "m", b"adapter", {})
        self.assertEqual(len([op for op, key in self.scratch.calls if op == "write"]), 1)
        self.assertNotIn("runs/family/names/m.json", self.scratch.files)
        with self.assertRaisesRegex(StoreError, "stopped"):
            self.store._write("later", b"no")

    def test_wrong_size_never_seals_and_same_size_corruption_is_found_at_read(self):
        self.scratch.mismatch = True
        with self.assertRaisesRegex(StoreError, "length"):
            self.store.write_named("family", "m", b"adapter", {})
        self.assertNotIn("runs/family/names/m.json", self.scratch.files)
        store = StrangeLoopStore(self.scratch, hashed_reads=HashedReads())
        uri = store.cas_put(b"x" * len(b"wrong bytes"))
        with self.assertRaisesRegex(StoreError, "hash mismatch"):
            store.cas_get(uri)

    def test_stat_transport_errors_retry_only_observation(self):
        self.scratch.read_error = TimeoutError("stat interrupted")
        self.scratch.read_failures = 2
        payload = b"adapter"
        self.store._write_hashed("hashed", payload, self.store.fingerprint(payload))
        self.assertEqual(self.scratch.calls, [("write", "hashed")] + [("stat", "hashed")] * 3)

    def test_directory_stat_cannot_confirm_even_an_empty_payload(self):
        self.scratch.write = lambda key, data: None
        self.scratch.files["hashed/child"] = b"child"
        with self.assertRaisesRegex(StoreError, "length"):
            self.store._write_hashed("hashed", b"", self.store.fingerprint(b""))

    def test_unavailable_cache_and_mount_do_not_prevent_verified_api_reads(self):
        key, digest = self.blob()
        with patch.object(BlobCache, "_lock", side_effect=OSError("no cache lock")):
            self.assertEqual(self.store.cas_get(f"cas://{digest}"), b"adapter")
        with patch.object(BlobCache, "__init__", side_effect=OSError("disk full")):
            store = StrangeLoopStore(self.scratch, hashed_reads=self.settings)
            self.assertEqual(store.cas_get(f"cas://{digest}"), b"adapter")
        self.mounted(key, b"adapter")
        (self.mount / key).unlink()
        (self.mount / key).mkdir()
        self.assertEqual(self.store.cas_get(f"cas://{digest}"), b"adapter")

    def test_eviction_is_bounded_and_an_open_reader_survives(self):
        settings = HashedReads(self.directory / "tiny", 8)
        store = StrangeLoopStore(self.scratch, hashed_reads=settings)
        keys = [self.blob(data) for data in (b"aaaa", b"bbbb", b"cccc")]
        for key, digest in keys[:2]:
            store.cas_get(f"cas://{digest}")
        cache = store._blob_cache
        with cache.path(keys[0][1]).open("rb") as reader:
            os.utime(cache.path(keys[0][1]), (1, 1))
            store.cas_get(f"cas://{keys[2][1]}")
            self.assertEqual(reader.read(), b"aaaa")
        self.assertFalse(cache.path(keys[0][1]).exists())
        self.assertEqual(sum(p.stat().st_size for p in cache.root.glob("*/*")
                             if len(p.name) == 64), 8)

    def test_oversized_payload_is_served_without_retention(self):
        store = StrangeLoopStore(self.scratch, hashed_reads=HashedReads(self.directory / "tiny", 2))
        key, digest = self.blob()
        for _ in range(2):
            self.assertEqual(store.cas_get(f"cas://{digest}"), b"adapter")
        self.assertFalse(store._blob_cache.path(digest).exists())
        self.assertEqual(self.scratch.calls, [("read", key)] * 2)

    def test_same_hash_threads_share_fetch_and_distinct_hashes_fetch_concurrently(self):
        key, digest = self.blob()
        entered, release = threading.Event(), threading.Event()
        original = self.scratch.read
        def slow_read(key):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("read blocked")
            return original(key)
        self.scratch.read = slow_read
        with ThreadPoolExecutor(2) as workers:
            first = workers.submit(self.store.cas_get, f"cas://{digest}")
            self.assertTrue(entered.wait(5))
            second = workers.submit(self.store.cas_get, f"cas://{digest}")
            release.set()
            self.assertEqual(first.result(5), second.result(5))
        self.assertEqual(self.scratch.calls, [("read", key)])
        keys = [self.blob(data) for data in (b"one", b"two")]
        self.assertNotEqual(keys[0][1][:2], keys[1][1][:2])
        barrier = threading.Barrier(2)
        def simultaneous_read(key):
            barrier.wait(5)
            return original(key)
        self.scratch.read = simultaneous_read
        with ThreadPoolExecutor(2) as workers:
            self.assertEqual(list(workers.map(self.store.cas_get, [f"cas://{d}" for k, d in keys])),
                             [b"one", b"two"])

    def test_dead_process_releases_locks_and_orphan_is_cleaned(self):
        context = multiprocessing.get_context("spawn")
        directory = self.directory / "crashed"
        digest = self.store.fingerprint(b"complete")
        ready = context.Event()
        process = context.Process(target=leave_cache_insertion_midway, args=(str(directory), digest, ready))
        process.start()
        try:
            self.assertTrue(ready.wait(10))
        finally:
            process.terminate()
            process.join(10)
        self.assertFalse(process.is_alive())
        cache = BlobCache(directory, 100, self.notes.append)
        self.assertEqual(list(directory.glob("*/*.tmp")), [])
        with cache.claim(digest):
            cache.put(digest, b"complete")
        self.assertEqual(cache.get(digest), b"complete")

    def test_resident_uses_inherited_cache_and_mount_settings_but_refuses_mutation(self):
        key, digest = self.blob()
        self.mounted(key, b"adapter")
        address = StoreAddress("strangeloop", str(self.mount), self.store.describe())
        with patch.dict(os.environ, self.settings.environment()), \
                patch.object(ScratchClient, "from_locator", return_value=self.scratch):
            reader = open_store(StoreAddress.from_row(address.row()))
        self.assertEqual(reader.cas_get(f"cas://{digest}"), b"adapter")
        self.assertEqual(reader._blob_cache.root, self.settings.blob_cache_dir)
        self.assertEqual(self.scratch.calls, [])
        with self.assertRaisesRegex(StoreError, "read-only"):
            reader.cas_put(b"different")

    def test_cache_validation_resolves_provider_mount_symlinks(self):
        # Modal's public mount paths can be links to the real mounted folders.
        original = Path.resolve
        def resolved(path, *args, **kwargs):
            for root in (Path("/scratch"), Path("/persist")):
                if path.is_relative_to(root):
                    return original(self.directory) / "provider" / root.name / path.relative_to(root)
            return original(path, *args, **kwargs)
        with patch.object(Path, "resolve", resolved):
            for directory in (Path("/scratch/cache"), Path("/persist/cache"),
                              self.directory / "provider/scratch/cache"):
                with self.subTest(directory=directory), self.assertRaises(ValueError):
                    HashedReads(directory, 100)

    def test_cache_configuration_rejects_shared_mounts_and_missing_bounds(self):
        for directory, bound in ((Path("/scratch/cache"), 100), (Path("/persist/cache"), 100),
                                 (Path("relative"), 100), (None, 100), (self.directory, 0),
                                 (None, -1)):
            with self.subTest(directory=directory, bound=bound), self.assertRaises(ValueError):
                HashedReads(directory, bound)
        with self.assertRaisesRegex(ValueError, "mounted store"):
            StrangeLoopStore(self.scratch, hashed_reads=HashedReads(self.mount / "cache", 100),
                             mount_root=self.mount)


class MountedScratchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "rlstack"
        self.scratch = MemoryScratch()
        self.mount = MountedScratch(self.root, self.scratch)
        self.sync = patch("rlstack.data.stores.strangeloop.subprocess.run", self.mount.sync)
        self.sync.start()
        self.addCleanup(self.sync.stop)
        self.store = StrangeLoopLocalStore(self.root, self.scratch, mountpoint=self.temp.name)

    def test_requires_real_publication_not_a_successful_noop(self):
        with self.assertRaisesRegex(StoreError, "verify_publication"):
            self.store._write("before", b"no")
        self.mount.noop = True
        with self.assertRaises(FileNotFoundError):
            self.store.verify_publication()
        with self.assertRaisesRegex(StoreError, "stopped"):
            self.store._write("after", b"no")

    def test_each_write_publishes_before_returning_and_probe_cleans_up(self):
        self.store.verify_publication()
        self.assertEqual(self.scratch.files, {})
        before = self.mount.commits
        self.store._write("cas/blob", b"payload")
        self.assertEqual(self.scratch.files["cas/blob"], b"payload")
        self.assertEqual(self.mount.commits, before + 1)
        self.store._append_line("journal", "line")
        self.assertEqual(self.scratch.files["journal"], b"line\n")
        self.assertEqual(self.mount.commits, before + 2)

    def test_hashed_mounted_payload_is_committed_and_stat_confirmed_before_seal(self):
        self.store.verify_publication()
        self.scratch.calls.clear()
        self.store.write_named("family", "m", b"adapter", {"r": 2})
        payload, meta = "runs/family/names/m.bin", "runs/family/names/m.json"
        self.assertEqual(self.scratch.calls, [("read", meta), ("stat", payload), ("read", meta)])
        published = [key for key, data in self.mount.publications]
        self.assertLess(published.index(payload), published.index(meta))
        fresh = StrangeLoopStore(self.scratch, hashed_reads=HashedReads())
        self.assertEqual(fresh.read_named("family", "m"), b"adapter")

    def test_hashed_sync_failure_does_not_seal_or_acknowledge(self):
        self.store.verify_publication()
        self.mount.fail = True
        with self.assertRaises(subprocess.TimeoutExpired):
            self.store.write_named("family", "m", b"adapter", {})
        self.assertNotIn("runs/family/names/m.json", self.scratch.files)
        with self.assertRaisesRegex(StoreError, "stopped"):
            self.store._write("later", b"no")

    def test_hashed_noop_sync_does_not_claim_publication_from_own_mount(self):
        self.store.verify_publication()
        self.mount.noop = True
        with patch.object(strangeloop_store, "READBACK_PAUSE_S", 0):
            with self.assertRaises(FileNotFoundError):
                self.store.write_named("family", "m", b"adapter", {})
        self.assertTrue(self.store.path_of("runs/family/names/m.bin").exists())
        self.assertNotIn("runs/family/names/m.json", self.scratch.files)

    def test_mounted_host_journal_uses_same_encoding_as_api(self):
        self.store.verify_publication()
        host = "sl-qwen4b:0.main-tp1.c1"
        self.store.append_host_event(host, {"event": "probe"})
        self.assertEqual(self.store.read_host_log(host), [{"event": "probe"}])
        self.assertTrue((self.root / "hosts/sl-qwen4b@3a0.main-tp1.c1/log.jsonl").is_file())
        self.assertFalse((self.root / f"hosts/{host}/log.jsonl").exists())

    def test_sync_failure_leaves_ledger_unacknowledged_and_stops_later_writes(self):
        self.store.verify_publication()
        self.store._write("journal", b"old\n")
        self.mount.fail = True
        with self.assertRaises(subprocess.TimeoutExpired):
            self.store._append_line("journal", "next")
        self.assertEqual(self.scratch.files["journal"], b"old\n")
        commits = self.mount.commits
        with self.assertRaisesRegex(StoreError, "journal.*mount sync: TimeoutExpired"):
            self.store._write("dependent", b"no")
        self.assertEqual(self.mount.commits, commits)

    def test_mounted_write_failure_survives_a_later_operation(self):
        self.store.verify_publication()
        commits = self.mount.commits
        with patch.object(LocalStore, "_write", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                self.store.append_host_event("host", {"event": "traffic"})
        with self.assertRaisesRegex(StoreError, "hosts/host/log.jsonl.*local write: OSError: disk unavailable"):
            self.store._write("rollout", b"must not publish")
        self.assertEqual(self.mount.commits, commits)
        self.assertNotIn("rollout", self.scratch.files)

    def test_mounted_readback_failure_survives_a_later_operation(self):
        self.store.verify_publication()
        with patch.object(self.store, "_published", side_effect=TimeoutError("readback unavailable")):
            with self.assertRaisesRegex(TimeoutError, "readback unavailable"):
                self.store.append_host_event("host", {"event": "traffic"})
        commits = self.mount.commits
        with self.assertRaisesRegex(StoreError, "hosts/host/log.jsonl.*publication readback: TimeoutError: readback unavailable"):
            self.store._write("rollout", b"must not publish")
        self.assertEqual(self.mount.commits, commits)
        self.assertNotIn("rollout", self.scratch.files)

    def test_existing_stale_files_and_missing_mount_reads_are_authoritative(self):
        self.store.verify_publication()
        self.store._write("ledger", b"old")
        self.scratch.files["ledger"] = b"new"
        self.scratch.files["remote-only"] = b"remote"
        self.assertEqual(self.store._read("ledger"), b"new")
        self.assertEqual(self.store._read("remote-only"), b"remote")
        self.assertEqual(self.store.path_of("ledger").read_bytes(), b"old")
        self.assertFalse(self.store.path_of("remote-only").exists())

    def test_resumed_stale_append_keeps_the_committed_prefix(self):
        self.store.verify_publication()
        run = self.store.open_run("r", {"run_id": "r"}, subdir="family")
        run.append_ledger({"update": 1, "versions": {}})
        run.append_checkpoint(1, {})
        # another writer's commit AND its checkpoint landed remotely
        self.scratch.files[run.ledger_key] += b'{"update":2,"versions":{}}\n'
        self.scratch.files[run.checkpoints_key] += b'{"update":2,"versions":{}}\n'
        replacement = StrangeLoopLocalStore(self.root, self.scratch, mountpoint=self.temp.name)
        replacement.verify_publication()
        resumed = replacement.open_run("r", {"run_id": "r"}, subdir="family", create=False)
        resumed.append_ledger({"update": 3, "versions": {}})
        self.assertEqual([row["update"] for row in resumed.read_ledger()], [1, 2, 3])
        self.assertEqual(self.root.joinpath(run.ledger_key).read_bytes(), self.scratch.files[run.ledger_key])

    def test_deleting_remote_only_and_stale_files_does_not_resurrect_them(self):
        self.store.verify_publication()
        self.store._write("stale", b"old")
        self.scratch.files["stale"] = b"new"
        self.scratch.files["remote-only"] = b"elsewhere"
        self.store._delete("stale")
        self.store._delete("remote-only")
        self.store._write("unrelated", b"later sync")
        self.assertEqual(self.scratch.files, {"unrelated": b"later sync"})
        self.assertFalse(self.store.path_of("stale").exists())

    def test_cleanup_finds_unsealed_and_partial_files_missing_from_mount(self):
        self.store.verify_publication()
        run = self.store.open_run("r", {"run_id": "r"})
        run.append_ledger({"update": 1, "versions": {}})
        self.scratch.files["runs/r/waves/000002.jsonl.gz"] = b"uncommitted"
        self.scratch.files["runs/r/junk.tmp"] = b"partial"
        self.store.open_run("r", {"run_id": "r"}, create=False)
        self.assertNotIn("runs/r/waves/000002.jsonl.gz", self.scratch.files)
        self.assertNotIn("runs/r/junk.tmp", self.scratch.files)

    def test_torn_committed_tail_is_repaired_before_resumed_append(self):
        self.store.verify_publication()
        run = self.store.open_run("r", {"run_id": "r"})
        run.append_ledger({"update": 1, "versions": {}})
        run.append_checkpoint(1, {})
        self.scratch.files[run.ledger_key] += b'{"update":2'
        resumed = self.store.open_run("r", {"run_id": "r"}, create=False)
        resumed.append_ledger({"update": 2, "versions": {}})
        self.assertEqual([row["update"] for row in resumed.read_ledger()], [1, 2])
        self.assertEqual(self.store.path_of(run.ledger_key).read_bytes(),
                         self.scratch.files[run.ledger_key])

    def test_distinct_threads_publish_prerequisites_before_ledger(self):
        self.store.verify_publication()
        def write_run(name):
            run = self.store.open_run(name, {"run_id": name})
            run.write_blob("adapters", "pi", 1, name.encode())
            run.append_ledger({"update": 1, "versions": {"pi": 1}})
        with ThreadPoolExecutor(max_workers=2) as writers:
            list(writers.map(write_run, ("a", "b")))
        for name in ("a", "b"):
            checkpoint = next(index for index, (key, _) in enumerate(self.mount.publications)
                              if key == f"runs/{name}/adapters/pi@1.bin")
            ledger = next(index for index, (key, data) in enumerate(self.mount.publications)
                          if key == f"runs/{name}/ledger.jsonl" and data)
            self.assertLess(checkpoint, ledger)
            self.assertEqual(self.scratch.files[f"runs/{name}/adapters/pi@1.bin"], name.encode())

    def test_resident_reopens_as_api_reader_without_mount_publication(self):
        self.store.verify_publication()
        self.store._write("blob", b"old")
        self.scratch.files["blob"] = b"new"
        count = self.mount.commits
        with patch.object(ScratchClient, "from_locator", return_value=self.scratch):
            reader = open_store(self.store.address())
        self.assertEqual(reader._read("blob"), b"new")
        with self.assertRaises(StoreError):
            reader._write("blob", b"bad")
        self.assertEqual(self.mount.commits, count)

    def test_mismatched_root_is_rejected(self):
        with self.assertRaises(StoreError):
            StrangeLoopLocalStore(Path(self.temp.name) / "other", self.scratch,
                                  mountpoint=self.temp.name)

    def test_all_backends_preserve_run_bytes_across_recovery(self):
        self.store.verify_publication()
        with tempfile.TemporaryDirectory() as local_root:
            local = LocalStore(local_root)
            remote = StrangeLoopStore(MemoryScratch())
            for store in (local, remote, self.store):
                run = store.open_run("r", {"run_id": "r"}, subdir="family")
                run.write_wave(1, [{"unicode": "λ", "tokens": [1, 2, 3]}])
                run.write_postdata(1, {"reward": [1.0]})
                run.write_blob("adapters", "pi", 1, b"adapter\x00\xff")
                run.write_blob("optim", "pi", 1, b"moments")
                run.append_ledger({"update": 1, "versions": {"pi": 1}})
                run.write_wave(2, [{"uncommitted": True}])
                run = store.open_run("r", {"run_id": "r"}, subdir="family", create=False)
                run.write_wave(2, [{"unicode": "雪"}])
                run.write_blob("adapters", "pi", 2, b"adapter2")
                run.append_ledger({"update": 2, "versions": {"pi": 2}})
            expected = {key: local._read(key) for key in local._list("runs/")}
            for store in (remote, self.store):
                actual = {key: store._read(key) for key in store._list("runs/")}
                self.assertEqual(actual, expected)


class ScratchHTTPTests(unittest.TestCase):
    def test_dropped_read_connection_retries_and_returns_authoritative_bytes(self):
        self.client.write("ledger", b"committed")
        original = self.client._opener.open
        calls = 0

        def open_once_dropped(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise http.client.RemoteDisconnected("peer closed before headers")
            return original(*args, **kwargs)

        with patch.object(self.client._opener, "open", side_effect=open_once_dropped):
            self.assertEqual(self.client.read("ledger"), b"committed")
        self.assertEqual(calls, 3)      # the dropped read, the health question, the read again
        self.assertIn(("GET", "/auth/me", ""), self.seen[-2:])

    def test_a_read_against_a_dead_api_waits_out_the_ceiling_and_writes_are_not_replayed(self):
        # every connection drops, the health question included: the API is
        # DOWN. A read waits it out to the ceiling; a write with a lost
        # acknowledgement is never replayed, whatever the API's state.
        clock = FakeClock()
        self.client._clock, self.client._sleep = clock.now, clock.sleep
        self.client.outage_ceiling_s = 120.0
        self.client._note = lambda message: None
        for method, action, expected in (
            ("GET", lambda: self.client.read("ledger"), "gave up after 120 s"),
            ("PUT", lambda: self.client.write("ledger", b"next"), "connection failed"),
            ("DELETE", lambda: self.client.delete("ledger"), "connection failed"),
        ):
            with self.subTest(method=method), patch.object(
                self.client._opener, "open", side_effect=http.client.RemoteDisconnected("closed")
            ) as opened:
                slept_before = sum(clock.slept)
                with self.assertRaisesRegex(StoreError, expected):
                    action()
                urls = [call.args[0].full_url for call in opened.call_args_list]
                if method == "GET":
                    self.assertGreaterEqual(sum(clock.slept) - slept_before, 120.0)
                    self.assertTrue(any(url.endswith("/auth/me") for url in urls))
                    self.assertGreater(len([u for u in urls if "/scratch/files/raw" in u]), 2)
                else:
                    self.assertEqual(opened.call_count, 1)
                    self.assertEqual(sum(clock.slept), slept_before)

    @classmethod
    def setUpClass(cls):
        cls.files = {}
        cls.status = 200
        cls.user_id = "account"
        cls.seen = []
        cls.truncate = False
        cls.error_keys = set()
        cls.error_lists = set()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.serve()
            def do_PUT(self):
                self.serve()
            def do_DELETE(self):
                self.serve()
            def serve(self):
                parsed = urllib.parse.urlsplit(self.path)
                route = parsed.path.removeprefix("/api/v1")
                key = urllib.parse.parse_qs(parsed.query).get("path", [""])[0]
                cls.seen.append((self.command, route, key))
                status = cls.status
                if key in cls.error_keys:
                    status = 500
                if route == "/scratch/files" and key in cls.error_lists:
                    status = 500
                if self.headers.get("Authorization") != "Bearer test-token":
                    status = 401
                result = {"error": "failure"}
                raw = None
                if status == 200:
                    if route == "/auth/me":
                        result = {"user_id": cls.user_id, "clusters": ["modal"]}
                    elif route == "/scratch/files/raw" and self.command == "PUT":
                        cls.files[key] = self.rfile.read(int(self.headers["Content-Length"]))
                        result = {"path": key}
                    elif route == "/scratch/files/raw":
                        if key in cls.files:
                            raw = cls.files[key]
                        else:
                            status = 404
                    elif route == "/scratch/files/stat":
                        if key in cls.files:
                            result = {"path": key, "type": "file", "size": len(cls.files[key])}
                        elif any(path.startswith(key.rstrip("/") + "/") for path in cls.files):
                            result = {"path": key, "type": "dir", "size": None}
                        else:
                            status = 404
                    elif self.command == "DELETE":
                        cls.files.pop(key, None)
                        result = {"path": key}
                    else:
                        result = {"entries": [{"path": path, "type": "file", "size": len(data)}
                                               for path, data in cls.files.items()
                                               if path.startswith(key.rstrip("/") + "/")]}
                body = raw if raw is not None else json.dumps(result).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body[:-1] if cls.truncate else body)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:" + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        type(self).files.clear()
        type(self).seen.clear()
        type(self).status = 200
        type(self).user_id = "account"
        type(self).truncate = False
        type(self).error_keys = set()
        type(self).error_lists = set()
        self.client = ScratchClient("sl-scratch-account", token="test-token", api_base=self.base)
        # pauses are virtual, and an API that answers 500 to everything is
        # waited out for one virtual minute rather than eight real hours
        self.clock = FakeClock()
        self.client._clock, self.client._sleep = self.clock.now, self.clock.sleep
        self.client.outage_ceiling_s = 60.0
        self.client._note = lambda message: None

    def test_two_spawned_readers_download_payload_once(self):
        data = b"independent processes" * 1000
        store = StrangeLoopStore(self.client, hashed_reads=HashedReads())
        uri = store.cas_put(data)
        self.seen.clear()
        context = multiprocessing.get_context("spawn")
        ready, results, start = context.Queue(), context.Queue(), context.Event()
        with tempfile.TemporaryDirectory() as directory:
            processes = [context.Process(target=read_cached_in_child,
                         args=(self.base, directory, uri, ready, start, results)) for _ in range(2)]
            try:
                for process in processes:
                    process.start()
                for _ in processes:
                    self.assertTrue(ready.get(timeout=10))
                start.set()
                for _ in processes:
                    self.assertEqual(results.get(timeout=10), data)
                for process in processes:
                    process.join(10)
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(10)
                ready.close()
                results.close()
        downloads = [row for row in self.seen if row[0:2] == ("GET", "/scratch/files/raw")]
        self.assertEqual(len(downloads), 1)

    def test_binary_roundtrip_list_stat_delete_and_streamed_download(self):
        payload = bytes(range(256)) * 8193
        self.client.write("folder/file &雪.bin", payload)
        self.assertEqual(self.client.read("folder/file &雪.bin"), payload)
        self.assertEqual(self.client.stat("folder/file &雪.bin").size, len(payload))
        self.assertEqual([row.path for row in self.client.list("folder")], ["folder/file &雪.bin"])
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "nested/blob"
            self.client.download("folder/file &雪.bin", destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])
        self.client.delete("folder/file &雪.bin")
        with self.assertRaises(FileNotFoundError):
            self.client.read("folder/file &雪.bin")

    def test_host_names_and_literal_escape_names_remain_distinct(self):
        keys = ["hosts/sl-qwen4b:0.main-tp1.c1/log.jsonl", "hosts/sl-qwen4b@3a0.main-tp1.c1/log.jsonl", "hosts/雪/log.jsonl"]
        for index, key in enumerate(keys):
            self.client.write(key, str(index).encode())
            self.assertEqual(self.client.read(key), str(index).encode())
            self.assertEqual(scratch_key(scratch_path(key)), key)
        self.assertEqual(sorted(row.path for row in self.client.list("hosts", recursive=True)), sorted(keys))
        self.assertEqual(len(type(self).files), 3)
        self.assertTrue(all(":" not in key and "雪" not in key for key in type(self).files))
        for invalid in ("hosts/@2f", "hosts/@", "hosts/@ff", "hosts/@2e@2e/file"):
            with self.assertRaises(StoreError):
                scratch_key(invalid)

    def test_truncated_download_keeps_existing_destination_and_discards_partial(self):
        self.client.write("blob", b"complete")
        type(self).truncate = True
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "blob"
            destination.write_bytes(b"previous")
            with self.assertRaisesRegex(StoreError, "byte count"):
                self.client.download("blob", destination)
            self.assertEqual(destination.read_bytes(), b"previous")
            self.assertEqual(sorted(path.name for path in Path(folder).iterdir()), ["blob"])
        with self.assertRaises(StoreError):
            self.client.read("blob")

    def test_only_not_found_is_absence(self):
        with self.assertRaises(FileNotFoundError):
            self.client.stat("missing")
        for status in (401, 403, 409, 429, 500, 504):
            with self.subTest(status=status):
                type(self).status = status
                with self.assertRaises(StoreError):
                    self.client.stat("missing")
                with self.assertRaises(StoreError):
                    self.client.list("missing")

    def test_wrong_account_rejected_before_any_file_request(self):
        type(self).user_id = "other"
        with self.assertRaisesRegex(StoreError, "account differs"):
            ScratchClient("sl-scratch-account", token="test-token", api_base=self.base)
        self.assertTrue(all(route == "/auth/me" for _, route, _ in self.seen))

    def test_missing_parent_404_resolves_live_api_child_500(self):
        type(self).error_keys = {"rlstack/fleet/log.jsonl", "rlstack/fleet"}
        for read in (self.client.read, self.client.stat):
            with self.assertRaises(FileNotFoundError):
                read("fleet/log.jsonl")
        self.assertIn(("GET", "/scratch/files/stat", "rlstack"), self.seen)

    def test_child_500_with_existing_parent_is_not_absence(self):
        self.client.write("fleet/sibling", b"exists")
        type(self).error_keys = {"rlstack/fleet/log.jsonl"}
        for read in (self.client.read, self.client.stat):
            with self.assertRaises(StoreError):
                read("fleet/log.jsonl")

    def test_listing_500_is_empty_only_when_metadata_proves_absence(self):
        self.client.write("cas/input", b"exists")
        type(self).error_lists = {"rlstack/runs", "rlstack/cas"}
        self.assertEqual(self.client.list("runs"), [])
        with self.assertRaises(StoreError):
            self.client.list("cas")

    def test_locator_roundtrip_keeps_deployment_but_not_credentials(self):
        self.assertNotIn("test-token", self.client.locator)
        with patch.dict(os.environ, {"SL_API_TOKEN": "test-token"}):
            reopened = ScratchClient.from_locator(self.client.locator)
        self.assertEqual(reopened.locator, self.client.locator)
        self.assertNotIn("test-token", repr(self.client.credentials))

    def test_path_traversal_and_credential_locators_are_rejected(self):
        for key in ("", "../other", "/scratch/elsewhere", "a/../b", "a//b"):
            with self.subTest(key=key), self.assertRaises(StoreError):
                self.client.write(key, b"bad")
        for locator in ("strangeloop://secret@sl-scratch-account/rlstack",
                        "strangeloop://sl-scratch-account/../escape",
                        "strangeloop://sl-scratch-account/rlstack?token=secret"):
            with self.subTest(locator=locator), self.assertRaises(StoreError):
                ScratchClient.from_locator(locator)

    def test_default_locator_does_not_follow_another_api_environment(self):
        with patch.object(ScratchClient, "__init__", return_value=None) as constructor:
            with patch.dict(os.environ, {"SL_API_BASE": self.base}):
                ScratchClient.from_locator("strangeloop://sl-scratch-account/rlstack")
        constructor.assert_called_once_with("sl-scratch-account", "rlstack", api_base=DEFAULT_API_BASE)

    def test_profile_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                "STRANGELOOP_CONFIG_DIR": folder, "SL_PROFILE": "worker.dev",
                "SL_API_TOKEN": "environment", "SL_API_BASE": self.base}, clear=True):
            Path(folder, "config.toml").write_text('[profile.worker.dev]\ntoken = "stored"\n')
            self.assertEqual(resolve_scratch_credentials().token, "environment")
            self.assertEqual(resolve_scratch_credentials(token="explicit").token, "explicit")
            del os.environ["SL_API_TOKEN"]
            self.assertEqual(resolve_scratch_credentials().token, "stored")
            self.assertEqual(resolve_scratch_credentials().api_base, self.base + "/api/v1")

    def test_the_token_file_sits_between_the_environment_and_the_profile(self):
        # a pod has no profile and its environment cannot be renewed: the
        # file the desk rewrites is what its processes resolve from
        from rlstack.data.stores.strangeloop import credential_file
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                "STRANGELOOP_CONFIG_DIR": folder, "SL_API_TOKEN_FILE": str(Path(folder, "token.json")),
                "SL_API_BASE": self.base}, clear=True):
            Path(folder, "config.toml").write_text('[profile.default]\ntoken = "stored"\n')
            self.assertEqual(resolve_scratch_credentials().token, "stored")     # no file yet: the profile
            Path(folder, "token.json").write_bytes(credential_file(
                ScratchCredentials("filed", self.base + "/api/v1", "default")))
            self.assertEqual(resolve_scratch_credentials().token, "filed")
            self.assertEqual(resolve_scratch_credentials(token="explicit").token, "explicit")
            with patch.dict(os.environ, {"SL_API_TOKEN": "environment"}):
                self.assertEqual(resolve_scratch_credentials().token, "environment")
            Path(folder, "token.json").write_bytes(credential_file(
                ScratchCredentials("renewed", self.base + "/api/v1", "default")))
            self.assertEqual(resolve_scratch_credentials().token, "renewed")    # re-read every time
            Path(folder, "config.toml").unlink()
            Path(folder, "token.json").write_text("not json")
            with self.assertRaisesRegex(StoreError, "SL_API_TOKEN_FILE"):
                resolve_scratch_credentials()


class FakeClock:
    """A clock the client sleeps against: every sleep advances it."""

    def __init__(self) -> None:
        self.t, self.slept = 1000.0, []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


class FakeResponse:
    """What an opener returns: readable, and a context manager."""

    ok = True

    def read(self, *args) -> bytes:
        return b"{}"

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


class ScratchClientOutageTests(unittest.TestCase):
    """The outage rule of `_open` (2026-09-17: twenty minutes of the API
    answering 500 and timing out ended four 7B runs and the desk's owner).
    An API that is UP keeps the older rules — a read owns READ_ATTEMPTS
    tries, a write one — and an API that is DOWN, or a token it refuses, is
    waited out to the ceiling."""

    def make_client(self, ceiling=3600.0):
        # the constructor verifies the account over the wire; the rule
        # lives in _open, which needs the timeout, the opener and the clock
        import urllib.request
        from rlstack.data.stores.strangeloop import ScratchClient, _NoRedirect
        client = ScratchClient.__new__(ScratchClient)
        client.timeout = 5.0
        client.credentials = ScratchCredentials("old", "https://api.example", "default")
        client._source = {"token": None, "api_base": None, "profile": None}
        client.outage_ceiling_s = ceiling
        client.clock = FakeClock()
        client._clock, client._sleep = client.clock.now, client.clock.sleep
        client._opener = urllib.request.build_opener(_NoRedirect())
        client.notes = []
        client._note = client.notes.append
        return client

    def http_error(self, code):
        import io
        import urllib.error
        return urllib.error.HTTPError("https://api.example/x", code, "boom", {}, io.BytesIO(b""))

    def fake_opener(self, answers, health):
        """`answers` are consumed by the request itself, in order; `health`
        answers the client's `/auth/me` question, its last entry repeating.
        Each entry is an exception to raise or a response to return."""
        calls = []

        def fake_open(request, timeout):
            if request.full_url.endswith("/auth/me"):
                calls.append(("health", request.get_header("Authorization")))
                answer = health.pop(0) if len(health) > 1 else health[0]
            else:
                calls.append((request.get_method(), request.get_header("Authorization"), timeout))
                answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        return fake_open, calls

    def request(self, method="GET", data=None):
        import urllib.request
        request = urllib.request.Request("https://api.example/scratch/files/raw?path=x",
                                         method=method, data=data)
        request.add_header("Authorization", "Bearer old")
        return request

    def ok(self):
        return FakeResponse()

    def test_a_get_answered_500_twice_by_a_live_api_then_200_succeeds(self):
        client = self.make_client()
        fake, calls = self.fake_opener([self.http_error(500), self.http_error(502), self.ok()],
                                       [self.ok()])
        with patch.object(client._opener, "open", side_effect=fake):
            self.assertTrue(client._open(self.request()).ok)
        self.assertEqual([c[0] for c in calls], ["GET", "health", "GET", "health", "GET"])
        self.assertEqual(client.clock.slept, [1.0, 2.0])          # the older backoff
        self.assertEqual(client.notes, [])                        # no outage was declared

    def test_a_get_answered_500_by_a_live_api_owns_bounded_tries_each_with_the_full_timeout(self):
        import urllib.error
        client = self.make_client()
        fake, calls = self.fake_opener([self.http_error(500)] * READ_ATTEMPTS, [self.ok()])
        with patch.object(client._opener, "open", side_effect=fake):
            with self.assertRaises(urllib.error.HTTPError):
                client._open(self.request())
        gets = [c for c in calls if c[0] == "GET"]
        self.assertEqual(len(gets), READ_ATTEMPTS)
        self.assertTrue(all(c[2] == 5.0 for c in gets))

    def test_a_write_refused_by_a_live_api_is_retried_until_it_lands(self):
        # a refused write is never in flight; a live API refuses journal
        # appends with 500 on its bad minutes, so the health question is not asked
        client = self.make_client()
        fake, calls = self.fake_opener([self.http_error(500), self.http_error(500), self.ok()], [self.ok()])
        with patch.object(client._opener, "open", side_effect=fake):
            self.assertTrue(client._open(self.request("PUT", b"x")).ok)
        self.assertEqual([c[0] for c in calls], ["PUT", "PUT", "PUT"])
        self.assertEqual(client.clock.slept, [1.0, 2.0])
        self.assertIn("HTTP 500 refused the write", client.notes[0])
        client = self.make_client(ceiling=5.0)
        fake, calls = self.fake_opener([self.http_error(503)] * 10, [self.ok()])
        with patch.object(client._opener, "open", side_effect=fake):
            with self.assertRaisesRegex(StoreError, "gave up after 5 s: HTTP 503 refused the write"):
                client._open(self.request("PUT", b"x"))

    def test_a_rate_limited_request_is_waited_out_read_or_write(self):
        """429 took nothing of the request, so nothing is in flight: come back
        slower. Twelve writers from one laptop met it within seconds
        (2026-09-19), and a pod that met it would have lost its run."""
        for method, body in (("GET", None), ("PUT", b"x")):
            with self.subTest(method):
                client = self.make_client()
                fake, calls = self.fake_opener([self.http_error(429), self.http_error(429), self.ok()], [self.ok()])
                with patch.object(client._opener, "open", side_effect=fake):
                    request = self.request(method, body) if body else self.request()
                    self.assertTrue(client._open(request).ok)
                self.assertEqual([c[0] for c in calls], [method] * 3)      # no health question
                self.assertEqual(client.clock.slept, [1.0, 2.0])
                self.assertIn("rate-limiting", client.notes[0])

    def test_a_get_while_the_api_is_down_waits_and_succeeds_when_it_returns(self):
        client = self.make_client()
        fake, calls = self.fake_opener(
            [self.http_error(503), TimeoutError(), self.ok()],
            [self.http_error(503), http.client.RemoteDisconnected("closed"), self.ok()])
        with patch.object(client._opener, "open", side_effect=fake):
            self.assertTrue(client._open(self.request()).ok)
        self.assertEqual([c[0] for c in calls], ["GET", "health", "GET", "health", "GET"])
        self.assertEqual(client.clock.slept, [1.0, 2.0])
        self.assertEqual(len(client.notes), 2)
        self.assertIn("waiting: HTTP 503 while the API is down", client.notes[0])
        self.assertIn("recovered after 3 s", client.notes[1])

    def test_a_get_while_the_api_stays_down_gives_up_at_the_ceiling_with_capped_pauses(self):
        from rlstack.data.stores.strangeloop import OUTAGE_PAUSE_MAX_S
        client = self.make_client(ceiling=900.0)
        fake, calls = self.fake_opener([TimeoutError()] * 1000, [TimeoutError()])
        with patch.object(client._opener, "open", side_effect=fake):
            with self.assertRaisesRegex(StoreError, "gave up after 900 s: TimeoutError while the API is down"):
                client._open(self.request())
        self.assertGreaterEqual(sum(client.clock.slept), 900.0)
        self.assertEqual(max(client.clock.slept), OUTAGE_PAUSE_MAX_S)
        self.assertEqual(len(client.notes), 2)                    # said once, then once more after ten minutes
        self.assertTrue(all("waiting" in note for note in client.notes))

    def test_a_refused_write_waits_out_an_outage_and_a_lost_acknowledgement_is_never_replayed(self):
        client = self.make_client()
        fake, calls = self.fake_opener([self.http_error(503), self.ok()], [self.http_error(502), self.ok()])
        with patch.object(client._opener, "open", side_effect=fake):
            self.assertTrue(client._open(self.request("PUT", b"x")).ok)
        self.assertEqual([c[0] for c in calls], ["PUT", "PUT"])         # refused: retried, no question asked
        self.assertEqual(client.clock.slept, [1.0])
        client = self.make_client()
        fake, calls = self.fake_opener([http.client.RemoteDisconnected("mid-body")], [self.ok()])
        with patch.object(client._opener, "open", side_effect=fake):
            with self.assertRaisesRegex(StoreError, "PUT connection failed: RemoteDisconnected"):
                client._open(self.request("PUT", b"x"))
        self.assertEqual([c[0] for c in calls], ["PUT"])          # no question asked, nothing replayed
        self.assertEqual(client.clock.slept, [])

    def test_a_401_with_a_renewed_token_is_retried_at_once_with_the_new_bearer(self):
        client = self.make_client()
        fake, calls = self.fake_opener([self.http_error(401), self.ok()], [self.ok()])
        renewed = ScratchCredentials("new", "https://api.example", "default")
        with patch.object(client._opener, "open", side_effect=fake), \
                patch.object(strangeloop_store, "resolve_scratch_credentials", return_value=renewed):
            self.assertTrue(client._open(self.request()).ok)
        self.assertEqual(calls, [("GET", "Bearer old", 5.0), ("GET", "Bearer new", 5.0)])
        self.assertEqual(client.clock.slept, [])
        self.assertEqual(client.credentials.token, "new")

    def test_a_401_with_the_same_token_waits_for_a_login(self):
        client = self.make_client()
        fake, calls = self.fake_opener([self.http_error(401), self.http_error(401), self.ok()], [self.ok()])
        same = ScratchCredentials("old", "https://api.example", "default")
        renewed = ScratchCredentials("new", "https://api.example", "default")
        with patch.object(client._opener, "open", side_effect=fake), \
                patch.object(strangeloop_store, "resolve_scratch_credentials", side_effect=[same, renewed]):
            self.assertTrue(client._open(self.request()).ok)
        self.assertEqual([c[1] for c in calls], ["Bearer old", "Bearer old", "Bearer new"])
        self.assertEqual(client.clock.slept, [1.0])
        self.assertIn("credentials refused (HTTP 401)", client.notes[0])

    def test_credentials_given_outright_are_never_renewed(self):
        import urllib.error
        client = self.make_client()
        client._source = {"token": "old", "api_base": None, "profile": None}
        fake, calls = self.fake_opener([self.http_error(401)] * 3, [self.ok()])
        with patch.object(client._opener, "open", side_effect=fake), \
                patch.object(strangeloop_store, "resolve_scratch_credentials") as resolve:
            client.outage_ceiling_s = 2.0
            with self.assertRaisesRegex(StoreError, "gave up after 2 s: credentials refused"):
                client._open(self.request())
        resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
