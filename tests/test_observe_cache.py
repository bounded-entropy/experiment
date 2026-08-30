"""The observer's read cache and the API memo.

The claims: a CachedReadStore answers byte-identically to the store it
wraps; immutable keys are read from the backend ONCE; an append-only journal
re-reads only when its size changed and always shows a fresh append; the
observer's store refuses every write; and the WSGI app serves one computed
payload per poll tick (the memo), never recomputing for each viewer.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest

from rlstack import LocalStore
from rlstack.data.stores.base import StoreError
from rlstack.observe.cache import CachedReadStore
from rlstack.observe.ui import ui_app


class CountingStore(LocalStore):
    """A LocalStore that counts its backend byte verbs."""

    def __init__(self, root) -> None:
        super().__init__(root)
        self.reads: list[str] = []
        self.lists: list[str] = []

    def _read(self, key: str) -> bytes:
        self.reads.append(key)
        return super()._read(key)

    def _list(self, prefix: str) -> list[str]:
        self.lists.append(prefix)
        return super()._list(prefix)


def seeded_store():
    tmp = tempfile.TemporaryDirectory()
    store = CountingStore(tmp.name)
    run = store.open_run("r1", manifest={"run_id": "r1"})
    run.append_ledger({"update": 1, "train": {"loss": 0.5},
                       "versions": {"pi": 1}})
    store.append_host_event("h", {"event": "stats", "t": 1.0})
    return tmp, store


class CachedReadStoreTest(unittest.TestCase):
    def test_answers_are_byte_identical(self) -> None:
        tmp, store = seeded_store()
        self.addCleanup(tmp.cleanup)
        cached = CachedReadStore(store)
        self.assertEqual(cached.peek_manifest("r1"), store.peek_manifest("r1"))
        self.assertEqual(cached.peek_ledger("r1"), store.peek_ledger("r1"))
        self.assertEqual(cached.read_host_log("h"), store.read_host_log("h"))
        self.assertEqual(cached.list_runs(), store.list_runs())

    def test_immutable_keys_hit_the_backend_once(self) -> None:
        tmp, store = seeded_store()
        self.addCleanup(tmp.cleanup)
        cached = CachedReadStore(store)
        cached.peek_manifest("r1")
        before = len(store.reads)
        cached.peek_manifest("r1")
        cached.peek_manifest("r1")
        self.assertEqual(len(store.reads), before)

    def test_a_journal_rereads_only_when_it_grew(self) -> None:
        tmp, store = seeded_store()
        self.addCleanup(tmp.cleanup)
        cached = CachedReadStore(store)
        self.assertEqual(len(cached.peek_ledger("r1")), 1)
        before = len(store.reads)
        self.assertEqual(len(cached.peek_ledger("r1")), 1)   # size unchanged
        self.assertEqual(len(store.reads), before)
        run = store.open_run("r1")       # the attach reads on its own account
        run.append_ledger({"update": 2, "train": {"loss": 0.4},
                           "versions": {"pi": 2}})
        before = len(store.reads)
        rows = cached.peek_ledger("r1")                       # size grew
        self.assertEqual([r["update"] for r in rows], [1, 2])
        self.assertEqual(len(store.reads), before + 1)

    def test_listings_hold_one_tick(self) -> None:
        tmp, store = seeded_store()
        self.addCleanup(tmp.cleanup)
        cached = CachedReadStore(store)
        cached.list_runs()
        before = len(store.lists)
        cached.list_runs()
        self.assertEqual(len(store.lists), before)

    def test_held_bytes_outrank_the_reload_blink(self) -> None:
        """A mount reload makes files transiently absent; nothing deletes a
        journal — so held bytes and the last real listing keep answering."""
        tmp, store = seeded_store()
        self.addCleanup(tmp.cleanup)
        cached = CachedReadStore(store)
        rows = cached.peek_ledger("r1")
        listed = cached.list_runs()
        self.assertEqual(listed, ["r1"])
        ledger = store.path_of("runs/r1/ledger.jsonl")
        manifest = store.path_of("runs/r1/manifest.json")
        blink = ledger.read_bytes(), manifest.read_bytes()
        ledger.unlink()                                # the blink
        manifest.unlink()
        self.assertEqual(cached.peek_ledger("r1"), rows)
        cached._lists = {k: (0.0, keys)                # the TTL has passed...
                         for k, (_, keys) in cached._lists.items()}
        self.assertEqual(cached.list_runs(), ["r1"])   # ...held listing stands
        ledger.write_bytes(blink[0])                   # the mount comes back
        manifest.write_bytes(blink[1])
        self.assertEqual(cached.peek_ledger("r1"), rows)

    def test_the_observers_store_never_writes(self) -> None:
        tmp, store = seeded_store()
        self.addCleanup(tmp.cleanup)
        cached = CachedReadStore(store)
        with self.assertRaises(StoreError):
            cached._write("runs/r1/manifest.json", b"{}")
        with self.assertRaises(StoreError):
            cached._append_line("runs/r1/ledger.jsonl", "{}")
        with self.assertRaises(StoreError):
            cached._delete("runs/r1/manifest.json")


class MemoTest(unittest.TestCase):
    def test_one_computation_per_poll_tick(self) -> None:
        """N viewers on one tick cost one refresh and one payload — and the
        memo serves bytes equal to the computed answer."""
        tmp, store = seeded_store()
        self.addCleanup(tmp.cleanup)
        refreshes = []
        app = ui_app([store], refresh=lambda: refreshes.append(1))

        def call(path):
            got = {}
            def start_response(status, headers):
                got["status"] = status
            body = b"".join(app({"PATH_INFO": path, "QUERY_STRING": ""},
                                start_response))
            return got["status"], body

        first = call("/api/runs")
        again = call("/api/runs")
        self.assertEqual(first, again)
        self.assertEqual(len(refreshes), 1)
        self.assertEqual(first[0], "200 OK")
        json.loads(first[1])                     # real JSON, both times


if __name__ == "__main__":
    unittest.main()
