"""Runs file into SUBDIRS: runs/<subdir>/<run_id>, indicated at submit.

The claims: a subdir is FILING, never identity (same run_id wherever filed,
and it never hashes); a run's home is fixed at birth (attach finds it
anywhere, a different subdir asked later is ignored — resume, not a move);
every peek and every handle resolves through the one seam (run_prefix); bad
segments are refused; the indication threads submit -> adopt -> desk; and
the observer's rows say where each run lives.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from common import arith_spec, arith_store
from rlstack import (
    FakeEngine, FakeLearner, Host, LocalStore, fake_qwen_schema,
)
from rlstack.data.stores.base import StoreError, check_subdir
from rlstack.observe.views import runs_data
from rlstack.runner.remote import HostService, LocalTransport, RemoteHost
from rlstack.spec.canonical import canonical_json

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def go(coro):
    return asyncio.run(coro)


class SubdirStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)

    def test_a_run_spawns_where_it_says_and_every_peek_finds_it(self) -> None:
        run = self.store.open_run("r1", manifest={"run_id": "r1"},
                                  subdir="ablations/plora")
        run.append_ledger({"update": 1})
        run.write_plan("train", b"x")
        self.assertTrue(self.store.path_of(
            "runs/ablations/plora/r1/manifest.json").exists())
        self.assertEqual(self.store.run_prefix("r1"), "runs/ablations/plora/r1")
        self.assertEqual(self.store.peek_manifest("r1"), {"run_id": "r1"})
        self.assertEqual(len(self.store.peek_ledger("r1")), 1)
        self.assertEqual(self.store.peek_plan("r1", "train"), b"x")
        self.assertEqual(self.store.list_runs(), ["r1"])
        self.assertEqual(self.store.run_subdirs(), {"r1": "ablations/plora"})

    def test_the_home_is_fixed_at_birth(self) -> None:
        """Attaching with a DIFFERENT subdir finds the run where it lives:
        resubmission is resume, not a move."""
        self.store.open_run("r1", manifest={"run_id": "r1"}, subdir="a")
        again = self.store.open_run("r1", manifest={"run_id": "r1"},
                                    subdir="b")
        self.assertEqual(again._key(), "runs/a/r1")
        self.assertFalse(self.store.path_of("runs/b").exists())

    def test_top_level_stays_the_degenerate_case(self) -> None:
        self.store.open_run("r1", manifest={"run_id": "r1"})
        self.assertEqual(self.store.run_prefix("r1"), "runs/r1")
        self.assertEqual(self.store.run_subdirs(), {"r1": ""})

    def test_bad_segments_are_refused_and_slashes_normalize(self) -> None:
        for bad in ("", "/", "..", "a/../b", "a b", "sweep:1"):
            with self.assertRaises(StoreError, msg=bad):
                check_subdir(bad)
        self.assertEqual(check_subdir("/a/b/"), "a/b")
        self.assertEqual(check_subdir("a//b"), "a/b")


class SubdirThreadingTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)

    def host(self) -> Host:
        return Host("filed-host", engines=(FakeEngine(),),
                    learner=FakeLearner(), store=self.store,
                    schema_for=lambda base: fake_qwen_schema(4, base=base))

    def test_submit_files_the_run_and_identity_ignores_it(self) -> None:
        spec = arith_spec(self.train)
        report = go(self.host().submit(spec, SCHEMA, subdir="sweeps/arith"))
        self.assertTrue(self.store.path_of(
            f"runs/sweeps/arith/{report.run_id}/ledger.jsonl").exists())

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        other_store, other_train, _ = arith_store(tmp.name)
        top = Host("top-host", engines=(FakeEngine(),), learner=FakeLearner(),
                   store=other_store)
        plain = go(top.submit(arith_spec(other_train), SCHEMA))
        self.assertEqual(report.run_id, plain.run_id)   # filing never hashes
        self.assertEqual(
            self.store.path_of(
                f"runs/sweeps/arith/{report.run_id}/ledger.jsonl").read_bytes(),
            other_store.path_of(
                f"runs/{plain.run_id}/ledger.jsonl").read_bytes())

    def test_the_adopt_frame_carries_the_filing(self) -> None:
        host = self.host()
        remote = RemoteHost(LocalTransport(HostService(host)))
        spec = arith_spec(self.train)

        async def drive():
            reply = await remote.adopt(spec, subdir="desk/filed")
            await host._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(reply["accepted"], reply)
        self.assertTrue(self.store.path_of(
            f"runs/desk/filed/{reply['run_id']}/manifest.json").exists())

    def test_the_observer_says_where_each_run_lives(self) -> None:
        go(self.host().submit(arith_spec(self.train), SCHEMA,
                              subdir="sweeps/arith"))
        (row,) = runs_data([self.store])
        self.assertEqual(row["subdir"], "sweeps/arith")


if __name__ == "__main__":
    unittest.main()
