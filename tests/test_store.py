"""The store (rlstack.data.store): layout, resume, crash recovery, versions."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rlstack.data.stores import (
    LedgerError, LocalStore, ManifestMismatch, RunHandle, Store, StoreError, bump,
)

def rpath(run: RunHandle, *parts: str):
    """On-disk path of a run-relative key (LocalStore only, for assertions)."""
    return run.store.path_of("/".join(("runs", run.run_id, *parts)))


MANIFEST: dict[str, Any] = {
    "run_id": "run-abc",
    "spec_hash": "3fa9c2",
    "base": "Qwen/Qwen3-8B",
    "bank": {"attn": {"kind": "lora", "r": 16}},
    "backend": "local",
}


class StoreTestCase(unittest.TestCase):
    """Common fixture: a store in a throwaway directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.store = LocalStore(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def open(self, manifest: dict[str, Any] | None = MANIFEST) -> RunHandle:
        return self.store.open_run("run-abc", manifest)

    def assert_no_tmp_files(self) -> None:
        leftovers = sorted(str(p) for p in self.root.rglob("*.tmp"))
        self.assertEqual(leftovers, [], f"atomic writes left temporaries: {leftovers}")


class OpenRunTest(StoreTestCase):
    def test_create_then_attach(self) -> None:
        run = self.open()
        self.assertEqual(run.manifest, MANIFEST)
        self.assertTrue((self.root / "runs" / "run-abc" / "manifest.json").exists())
        for section in ("rollouts", "adapters", "optim", "eval"):
            self.assertTrue(rpath(run, section).parent.is_dir(), section)
        again = self.store.open_run("run-abc")  # attach without offering a manifest
        self.assertEqual(again.manifest, MANIFEST)
        self.assertEqual(self.store.list_runs(), ["run-abc"])

    def test_create_requires_manifest(self) -> None:
        with self.assertRaises(StoreError):
            self.store.open_run("nope")

    def test_manifest_mismatch(self) -> None:
        self.open()
        with self.assertRaises(ManifestMismatch):
            self.store.open_run("run-abc", {**MANIFEST, "base": "Qwen/Qwen3-1.7B"})

    def test_manifest_compared_canonically(self) -> None:
        self.open()
        reordered = dict(reversed(list(MANIFEST.items())))
        self.assertEqual(self.store.open_run("run-abc", reordered).manifest, MANIFEST)

    def test_manifest_is_a_copy(self) -> None:
        run = self.open()
        run.manifest["base"] = "mutated"
        self.assertEqual(run.manifest["base"], MANIFEST["base"])

    def test_manifest_on_disk_is_canonical_json(self) -> None:
        run = self.open()
        raw = rpath(run, "manifest.json").read_text(encoding="utf-8")
        self.assertEqual(raw, json.dumps(MANIFEST, sort_keys=True, separators=(",", ":")))
        self.assert_no_tmp_files()


class LedgerTest(StoreTestCase):
    def test_tail_is_none_on_fresh_run(self) -> None:
        self.assertIsNone(self.open().ledger_tail())

    def test_append_and_tail(self) -> None:
        run = self.open()
        run.append_ledger({"update": 0, "versions": {"attn": 1}})
        run.append_ledger({"update": 1, "versions": {"attn": 2}})
        tail = run.ledger_tail()
        assert tail is not None
        self.assertEqual(tail["update"], 1)
        self.assertEqual(tail["versions"], {"attn": 2})
        self.assertEqual([e["update"] for e in run.read_ledger()], [0, 1])

    def test_monotonicity_enforced(self) -> None:
        run = self.open()
        run.append_ledger({"update": 5})
        with self.assertRaises(LedgerError):
            run.append_ledger({"update": 5})
        with self.assertRaises(LedgerError):
            run.append_ledger({"update": 4})
        run.append_ledger({"update": 6})
        self.assertEqual([e["update"] for e in run.read_ledger()], [5, 6])

    def test_entry_must_carry_int_update(self) -> None:
        run = self.open()
        with self.assertRaises(LedgerError):
            run.append_ledger({"metrics": {"loss": 1.0}})
        with self.assertRaises(LedgerError):
            run.append_ledger({"update": "3"})
        with self.assertRaises(LedgerError):
            run.append_ledger({"update": True})

    def test_append_only_on_disk(self) -> None:
        run = self.open()
        run.append_ledger({"update": 0})
        first = rpath(run, "ledger.jsonl").read_text(encoding="utf-8")
        run.append_ledger({"update": 1})
        self.assertTrue(rpath(run, "ledger.jsonl").read_text(encoding="utf-8").startswith(first))

    def test_torn_final_line_is_uncommitted(self) -> None:
        run = self.open()
        run.append_ledger({"update": 0})
        run.append_ledger({"update": 1})
        with open(rpath(run, "ledger.jsonl"), "a", encoding="utf-8") as handle:
            handle.write('{"update": 2, "versi')  # kill -9 mid-append
        tail = run.ledger_tail()
        assert tail is not None
        self.assertEqual(tail["update"], 1)
        run.append_ledger({"update": 2})
        self.assertEqual([e["update"] for e in run.read_ledger()], [0, 1, 2])


class RolloutsTest(StoreTestCase):
    def test_roundtrip_and_listing(self) -> None:
        run = self.open()
        rows = [{"traj": 0, "token_ids": [1, 2, 3]}, {"traj": 1, "token_ids": []}]
        run.write_rollouts(3, rows)
        self.assertTrue(rpath(run, "rollouts", "000003.jsonl.gz").exists())
        self.assertEqual(run.read_rollouts(3), rows)
        run.write_rollouts(10, [])
        self.assertEqual(run.list_updates(), [3, 10])
        self.assert_no_tmp_files()

    def test_gzip_fidelity_unicode_and_floats(self) -> None:
        run = self.open()
        rows = [
            {
                "text": "café ↔ 日本語 \U0001f600 \"quoted\" \\ backslash \n newline",
                "logprobs": [-0.1, -1e-17, 1.7976931348623157e308, 5e-324, 0.30000000000000004],
                "big_int": 2**62,
                "nested": {"π": [1.0, -0.0, 3.141592653589793]},
                "flag": True,
                "none": None,
            }
        ]
        run.write_rollouts(1, rows)
        back = run.read_rollouts(1)
        self.assertEqual(back, rows)
        self.assertEqual(back[0]["logprobs"][3], 5e-324)
        self.assertEqual(repr(back[0]["nested"]["π"][1]), "-0.0")
        with gzip.open(rpath(run, "rollouts", "000001.jsonl.gz"), "rb") as handle:
            raw = handle.read().decode("utf-8")
        self.assertIn("café", raw)  # UTF-8 on disk, not \uXXXX escapes
        self.assertEqual(raw.count("\n"), 1)

    def test_missing_update(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.open().read_rollouts(99)

    def test_committed_rollouts_are_append_only(self) -> None:
        run = self.open()
        run.write_rollouts(1, [{"a": 1}])
        run.append_ledger({"update": 1})
        with self.assertRaises(StoreError):
            run.write_rollouts(1, [{"a": 2}])
        self.assertEqual(run.read_rollouts(1), [{"a": 1}])


class BlobTest(StoreTestCase):
    def test_adapters_and_optim_roundtrip(self) -> None:
        run = self.open()
        run.write_blob("adapters", "attn", 2, b"\x00delta-bytes")
        run.write_blob("optim", "attn", 2, b"\x01moments")  # decision #14: lockstep
        self.assertEqual(run.read_blob("adapters", "attn", 2), b"\x00delta-bytes")
        self.assertEqual(run.read_blob("optim", "attn", 2), b"\x01moments")
        self.assertTrue(rpath(run, "adapters", "attn@2.bin").exists())
        self.assertTrue(rpath(run, "optim", "attn@2.bin").exists())
        self.assert_no_tmp_files()

    def test_unknown_section_rejected(self) -> None:
        run = self.open()
        with self.assertRaises(ValueError):
            run.write_blob("weights", "attn", 1, b"x")
        with self.assertRaises(ValueError):
            run.read_blob("weights", "attn", 1)

    def test_missing_blob(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.open().read_blob("adapters", "attn", 7)


class EvalTest(StoreTestCase):
    def test_write_and_read_back(self) -> None:
        run = self.open()
        run.write_eval(12, "summary.json", '{"means": {}}')
        self.assertEqual(run.read_eval(12, "summary.json"), '{"means": {}}')
        self.assertTrue(rpath(run, "eval", "12", "summary.json").exists())


class CasTest(StoreTestCase):
    def test_roundtrip(self) -> None:
        data = "tasks\n日本語\n".encode("utf-8")
        uri = self.store.cas_put(data)
        self.assertEqual(uri, f"cas://{self.store.fingerprint(data)}")
        self.assertEqual(self.store.cas_get(uri), data)
        self.assert_no_tmp_files()

    def test_dedup(self) -> None:
        data = b"same bytes"
        first = self.store.cas_put(data)
        second = self.store.cas_put(data)
        self.assertEqual(first, second)
        self.assertEqual(len(list((self.root / "cas").iterdir())), 1)
        self.store.cas_put(b"other bytes")
        self.assertEqual(len(list((self.root / "cas").iterdir())), 2)

    def test_fingerprint_is_sha256(self) -> None:
        import hashlib

        self.assertEqual(self.store.fingerprint(b"abc"), hashlib.sha256(b"abc").hexdigest())

    def test_bad_uri_and_missing_object(self) -> None:
        with self.assertRaises(ValueError):
            self.store.cas_get("store://run/x")
        with self.assertRaises(FileNotFoundError):
            self.store.cas_get("cas://" + "0" * 64)


class CrashRecoveryTest(StoreTestCase):
    def commit(self, run: RunHandle, update: int, version: int) -> None:
        """Normal order: rollouts, then blobs, then the ledger entry that seals them."""
        run.write_rollouts(update, [{"update": update}])
        run.write_blob("adapters", "attn", version, b"delta")
        run.write_blob("optim", "attn", version, b"moments")
        run.append_ledger({"update": update, "versions": {"attn": version}})

    def test_rollouts_without_ledger_are_dropped(self) -> None:
        run = self.open()
        self.commit(run, 1, 1)
        self.commit(run, 2, 2)
        run.write_rollouts(3, [{"update": 3}])  # crash before append_ledger
        self.assertEqual(run.list_updates(), [1, 2, 3])

        reopened = self.open()
        self.assertEqual(reopened.list_updates(), [1, 2])
        self.assertEqual(reopened.read_rollouts(2), [{"update": 2}])
        tail = reopened.ledger_tail()
        assert tail is not None
        self.assertEqual(tail["update"], 2)

    def test_blob_beyond_committed_version_is_dropped(self) -> None:
        run = self.open()
        self.commit(run, 1, 1)
        self.commit(run, 2, 2)
        run.write_blob("adapters", "attn", 3, b"uncommitted delta")
        run.write_blob("optim", "attn", 3, b"uncommitted moments")

        reopened = self.open()
        self.assertEqual(reopened.read_blob("adapters", "attn", 2), b"delta")
        self.assertEqual(reopened.read_blob("optim", "attn", 2), b"moments")
        for section in ("adapters", "optim"):
            with self.assertRaises(FileNotFoundError):
                reopened.read_blob(section, "attn", 3)
            self.assertFalse(rpath(reopened, section, "attn@3.bin").exists())

    def test_recovery_is_per_delta(self) -> None:
        run = self.open()
        run.write_blob("adapters", "attn", 4, b"a")
        run.write_blob("adapters", "head", 1, b"h")
        run.append_ledger({"update": 0, "versions": {"attn": 4, "head": 1}})
        run.write_blob("adapters", "head", 2, b"uncommitted")  # crash mid optim_step

        reopened = self.open()
        self.assertEqual(reopened.read_blob("adapters", "attn", 4), b"a")
        self.assertEqual(reopened.read_blob("adapters", "head", 1), b"h")
        with self.assertRaises(FileNotFoundError):
            reopened.read_blob("adapters", "head", 2)

    def test_everything_dropped_when_ledger_is_empty(self) -> None:
        run = self.open()
        run.write_rollouts(0, [{"update": 0}])
        self.assertEqual(self.open().list_updates(), [])

    def test_blobs_kept_when_ledger_records_no_versions(self) -> None:
        run = self.open()
        run.write_blob("adapters", "attn", 9, b"delta")
        run.append_ledger({"update": 0})
        self.assertEqual(self.open().read_blob("adapters", "attn", 9), b"delta")

    def test_torn_ledger_tail_truncated_on_attach(self) -> None:
        run = self.open()
        self.commit(run, 1, 1)
        with open(rpath(run, "ledger.jsonl"), "a", encoding="utf-8") as handle:
            handle.write('{"update": 2, "vers')  # kill -9 mid-append

        reopened = self.open()
        self.assertTrue(rpath(reopened, "ledger.jsonl").read_bytes().endswith(b"\n"))
        self.assertEqual([e["update"] for e in reopened.read_ledger()], [1])
        reopened.append_ledger({"update": 2, "versions": {"attn": 2}})
        self.assertEqual([e["update"] for e in reopened.read_ledger()], [1, 2])

    def test_stray_temporaries_swept_on_attach(self) -> None:
        run = self.open()
        stray = rpath(run, "rollouts", "000007.jsonl.gz.999.tmp")
        stray.parent.mkdir(parents=True, exist_ok=True)
        stray.write_bytes(b"partial")
        self.open()
        self.assert_no_tmp_files()

    def test_resume_continues_after_recovery(self) -> None:
        run = self.open()
        self.commit(run, 1, 1)
        run.write_rollouts(2, [{"update": 2, "torn": True}])  # crash

        resumed = self.open()
        resumed.write_rollouts(2, [{"update": 2}])
        resumed.append_ledger({"update": 2, "versions": {"attn": 2}})
        self.assertEqual(resumed.read_rollouts(2), [{"update": 2}])
        self.assertEqual(resumed.list_updates(), [1, 2])
        self.assert_no_tmp_files()


class BumpTest(unittest.TestCase):
    def test_named_deltas_advance(self) -> None:
        out = bump({"attn": 4, "mlp": 4, "head": 2}, ["attn", "head"])
        self.assertEqual(out, {"attn": 5, "mlp": 4, "head": 3})

    def test_does_not_mutate_input(self) -> None:
        version = {"pi": 1}
        self.assertEqual(bump(version, ["pi"]), {"pi": 2})
        self.assertEqual(version, {"pi": 1})

    def test_unknown_name_starts_at_one(self) -> None:
        self.assertEqual(bump({}, ["critic"]), {"critic": 1})

    def test_no_names_is_identity(self) -> None:
        self.assertEqual(bump({"pi": 3}, []), {"pi": 3})


if __name__ == "__main__":
    unittest.main()
