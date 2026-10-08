"""A bank grows by write-once versions that list only sealed named adapters."""
from __future__ import annotations

import tempfile
import unittest

from rlstack.data.banks import BankMemory, bank_at, draw, latest_version, publish, read_version
from rlstack.data.stores.base import StoreError
from rlstack.data.stores.local import LocalStore

SUBDIR = "bank-test"


class BankTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = LocalStore(temporary.name)

    def seal(self, name: str, split: str = "train") -> BankMemory:
        self.store.write_named(SUBDIR, name, name.encode() * 4, {"by": "test"})
        return BankMemory(name, self.store.named_meta(SUBDIR, name)["sha256"], split, {"group": name})

    def test_versions_fold_in_order_and_the_latest_is_the_default(self):
        self.assertEqual(latest_version(self.store, SUBDIR, "b"), 0)
        self.assertEqual(bank_at(self.store, SUBDIR, "b").memories, ())
        v1 = publish(self.store, SUBDIR, "b", [self.seal("m/1"), self.seal("m/2", "dev")], created="t1",
                     about={"commit": "abc"})
        v2 = publish(self.store, SUBDIR, "b", [self.seal("m/3")], created="t2")
        self.assertEqual((v1.version, v1.total, v2.version, v2.total), (1, 2, 2, 3))
        self.assertEqual(bank_at(self.store, SUBDIR, "b").names(), ["m/1", "m/2", "m/3"])
        self.assertEqual(bank_at(self.store, SUBDIR, "b", 1).names("train"), ["m/1"])
        self.assertEqual(read_version(self.store, SUBDIR, "b", 1).about, {"commit": "abc"})
        self.assertEqual(bank_at(self.store, SUBDIR, "b").memory("m/2").record, {"group": "m/2"})
        # a fresh store object (another process) reads the same thing
        again = LocalStore(self.store.root)
        self.assertEqual(bank_at(again, SUBDIR, "b").version, 2)

    def test_only_sealed_memories_once_each(self):
        with self.assertRaisesRegex(StoreError, "not a present named adapter"):
            publish(self.store, SUBDIR, "b", [BankMemory("ghost", "0" * 64, "train")], created="t")
        m = self.seal("m/1")
        with self.assertRaisesRegex(StoreError, "sealed as"):
            publish(self.store, SUBDIR, "b", [BankMemory("m/1", "f" * 64, "train")], created="t")
        publish(self.store, SUBDIR, "b", [m], created="t")
        with self.assertRaisesRegex(StoreError, "already in bank"):
            publish(self.store, SUBDIR, "b", [m], created="t")
        with self.assertRaises(StoreError):
            publish(self.store, SUBDIR, "b", [], created="t")

    def test_a_retired_memory_leaves_the_draws_and_never_comes_back(self):
        publish(self.store, SUBDIR, "b", [self.seal(f"m/{k}") for k in range(3)], created="t1")
        with self.assertRaisesRegex(StoreError, "reason"):
            publish(self.store, SUBDIR, "b", [], created="t", retired={"m/1": " "})
        with self.assertRaisesRegex(StoreError, "not in use"):
            publish(self.store, SUBDIR, "b", [], created="t", retired={"m/9": "x"})
        v2 = publish(self.store, SUBDIR, "b", [self.seal("m/3")], created="t2",
                     retired={"m/1": "fact superseded"})
        self.assertEqual(v2.total, 3)
        view = bank_at(self.store, SUBDIR, "b")
        self.assertEqual(view.names(), ["m/0", "m/2", "m/3"])
        self.assertEqual(view.retired, {"m/1": (2, "fact superseded")})
        self.assertNotIn("m/1", draw(view, 3, seed="s"))
        self.assertEqual(bank_at(self.store, SUBDIR, "b", 1).names(), ["m/0", "m/1", "m/2"])
        with self.assertRaisesRegex(StoreError, "already in bank"):
            publish(self.store, SUBDIR, "b", [self.seal("m/1")], created="t3")
        with self.assertRaisesRegex(StoreError, "not in use"):
            publish(self.store, SUBDIR, "b", [], created="t3", retired={"m/1": "again"})
        self.assertEqual(read_version(self.store, SUBDIR, "b", 1).retired, {})

    def test_a_version_is_written_once_and_versions_are_dense(self):
        self.store.write_bank_version(SUBDIR, "b", 1, {"x": 1})
        self.store.write_bank_version(SUBDIR, "b", 1, {"x": 1})       # a retry: no-op
        with self.assertRaisesRegex(StoreError, "written once"):
            self.store.write_bank_version(SUBDIR, "b", 1, {"x": 2})
        with self.assertRaisesRegex(StoreError, "dense"):
            self.store.write_bank_version(SUBDIR, "b", 3, {"x": 3})
        with self.assertRaises(StoreError):
            self.store.write_bank_version(SUBDIR, "a/b", 1, {})
        self.assertEqual(self.store.bank_versions(SUBDIR, "b"), [1])

    def test_a_bank_is_not_a_run(self):
        publish(self.store, SUBDIR, "b", [self.seal("m/1")], created="t")
        self.assertEqual(self.store.list_runs(), [])

    def test_draw_is_seeded_by_seed_and_version(self):
        publish(self.store, SUBDIR, "b", [self.seal(f"m/{k}") for k in range(10)], created="t")
        view = bank_at(self.store, SUBDIR, "b")
        a = draw(view, 4, seed="run:u1")
        self.assertEqual(a, draw(view, 4, seed="run:u1"))
        self.assertEqual(len(set(a)), 4)
        self.assertNotEqual(a, draw(view, 4, seed="run:u2"))
        with self.assertRaises(ValueError):
            draw(view, 11, seed="x")
        self.assertEqual(draw(view, 0, seed="x", split="dev"), [])


if __name__ == "__main__":
    unittest.main()
