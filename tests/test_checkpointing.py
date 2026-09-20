"""Checkpointing (ADR 0014): the checkpoint is the commit, the wire carries the
policy.

The claim under test, in one line: a run under `Checkpointing(every=k)`
killed at ANY point and resumed equals the same run straight, byte for byte —
for every k, at every crash point of the two-step protocol — and it writes
blobs ⌈n/k⌉ times, not n. The rest of the file is the edges: the rollout
rewind, a drained stop, `store` delivery, a pre-0014 directory, and the miss
that must park a run rather than lie to it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import arith_spec, arith_store
from rlstack import (
    FakeEngine, FakeLearner, LocalStore, fake_qwen_schema, run_experiment,
)
from rlstack.data.stores.base import CHECKPOINTS
from rlstack.runner.checkpointing import EVERY_UPDATE, Checkpointing
from rlstack.runner.roles import Trainer
from rlstack.runner.roles.base import StopRequest
from rlstack.runner.loop import run_experiment_async
from rlstack.runner.restore import BundleUnavailable
from test_resume import CrashingStore, SimulatedCrash, snapshot

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
CADENCES = (1, 2, 3, 7)      # 7 > the plan's 4 updates: only the extent checkpoints


def straight(case: unittest.TestCase, cadence: Checkpointing) -> tuple[LocalStore, str]:
    tmp = tempfile.TemporaryDirectory()
    case.addCleanup(tmp.cleanup)
    store, train, _ = arith_store(tmp.name)
    report = run_experiment(arith_spec(train), SCHEMA, store, FakeEngine(),
                            FakeLearner(), checkpointing=cadence)
    return store, report.run_id


def record(store: LocalStore, run_id: str) -> list[int]:
    """The checkpoint record's updates, read off the file."""
    raw = store.path_of(f"runs/{run_id}/{CHECKPOINTS}").read_text()
    return [json.loads(line)["update"] for line in raw.split("\n") if line]


def blob_versions(store: LocalStore, run_id: str, section: str) -> list[int]:
    root = store.path_of(f"runs/{run_id}/{section}")
    return sorted(int(p.name.split("@")[1][:-4]) for p in root.glob("pi@*.bin"))


class TheDeclarationTest(unittest.TestCase):
    def test_no_default_anywhere(self) -> None:
        with self.assertRaises(ValueError):
            Checkpointing.from_row(None)
        with self.assertRaises(ValueError):
            Checkpointing.from_row({"delivery": "wire"})
        with self.assertRaises(TypeError):
            run_experiment(None, None, None, None, None)   # keyword-only, required

    def test_every_is_a_positive_int(self) -> None:
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                Checkpointing(every=bad)

    def test_store_delivery_is_bound_to_every_update(self) -> None:
        Checkpointing(every=1, delivery="store")
        with self.assertRaises(ValueError):
            Checkpointing(every=2, delivery="store")
        with self.assertRaises(ValueError):
            Checkpointing(every=1, delivery="carrier-pigeon")

    def test_due_at_the_cadence_and_at_the_extent(self) -> None:
        every3 = Checkpointing(every=3)
        self.assertEqual([u for u in range(1, 8) if every3.due(u, 7)], [3, 6, 7])
        self.assertEqual([u for u in range(1, 5) if Checkpointing(7).due(u, 4)], [4])

    def test_the_row_round_trips(self) -> None:
        for cadence in (EVERY_UPDATE, Checkpointing(4), Checkpointing(1, "store")):
            self.assertEqual(Checkpointing.from_row(cadence.row()), cadence)


class BlobsAtTheCadenceTest(unittest.TestCase):
    """⌈n/k⌉ checkpoints, not n; the ledger is one line per update whatever k."""

    def test_the_record_and_the_blobs_follow_the_cadence(self) -> None:
        expected = {1: [0, 1, 2, 3, 4], 2: [0, 2, 4], 3: [0, 3, 4], 7: [0, 4]}
        for every in CADENCES:
            with self.subTest(every=every):
                store, run_id = straight(self, Checkpointing(every))
                self.assertEqual(record(store, run_id), expected[every])
                self.assertEqual(blob_versions(store, run_id, "adapters"), expected[every])
                # retention keeps only the tail's moments, at every cadence
                self.assertEqual(blob_versions(store, run_id, "optim"), [4])
                ledger = store.peek_ledger(run_id)
                self.assertEqual([e["update"] for e in ledger], [1, 2, 3, 4])

    def test_the_ledger_does_not_depend_on_the_cadence(self) -> None:
        """The commit record is the same bytes under every k: cadence changes
        what is RESTORABLE, never what was COMPUTED."""
        ledgers = set()
        for every in CADENCES:
            store, run_id = straight(self, Checkpointing(every))
            ledgers.add(store.path_of(f"runs/{run_id}/ledger.jsonl").read_bytes())
        self.assertEqual(len(ledgers), 1)

    def test_progress_is_done_only_once_the_extent_is_sealed(self) -> None:
        from rlstack.data.stores.base import run_progress

        store, run_id = straight(self, Checkpointing(3))
        progress = run_progress(store, run_id)
        self.assertEqual((progress.completed, progress.checkpointed), (4, 4))
        self.assertTrue(progress.done)
        # a ledger that reached the plan with the final checkpoint missing is
        # still work: attach would rewind it
        Path(store.path_of(f"runs/{run_id}/{CHECKPOINTS}")).write_text(
            '{"update": 0, "versions": {"pi": 0}}\n{"update": 3, "versions": {"pi": 3}}\n')
        self.assertFalse(run_progress(store, run_id).done)


class RewindEquivalenceTest(unittest.TestCase):
    """Kill the run anywhere under every=k, reattach, get the straight run's
    bytes. The crash points walk the two-step protocol — before the ledger
    line, after it and before the checkpoint's blobs, between the two blobs,
    after the checkpoint line and before its sweep — at every cadence."""

    def crash_points(self, every: int) -> list[tuple[str, int, bool]]:
        points = [
            ("write_wave", 2, False),       # entering update 3's wave
            ("write_postdata", 2, False),   # wave written, pipeline output not
            ("append_ledger", 2, False),    # update 3 staged, never committed
            ("append_ledger", 2, True),     # update 3 committed, nothing after it
            ("append_ledger", 3, True),     # update 4 committed; its checkpoint pending
            ("append_ledger", 1, True),     # update 2 committed
        ]
        # the blobs: one write for pi@0, then two per checkpoint — crash with
        # the first checkpoint's adapters written and its optim not
        points.append(("write_blob", 1 + 0, False))
        points.append(("write_blob", 1 + 1, False))
        # the checkpoint line: after checkpoint 0, and after the first real one
        points.append(("append_checkpoint", 1, True))
        points.append(("append_checkpoint", 0, True))
        return points

    def test_crash_anywhere_at_any_cadence_then_resume_is_byte_identical(self) -> None:
        for every in CADENCES:
            cadence = Checkpointing(every)
            reference_store, run_id = straight(self, cadence)
            reference = snapshot(reference_store, run_id)
            for method, after, post in self.crash_points(every):
                with self.subTest(every=every, crash=f"{method}@{after}{'+post' if post else ''}"):
                    tmp = tempfile.TemporaryDirectory()
                    self.addCleanup(tmp.cleanup)
                    _, train, _ = arith_store(tmp.name)
                    crashing = CrashingStore(tmp.name, method, after, post)
                    with self.assertRaises(SimulatedCrash):
                        run_experiment(arith_spec(train), SCHEMA, crashing,
                                       FakeEngine(), FakeLearner(), checkpointing=cadence)
                    resumed = run_experiment(arith_spec(train), SCHEMA, LocalStore(tmp.name),
                                             FakeEngine(), FakeLearner(), checkpointing=cadence)
                    self.assertEqual(resumed.run_id, run_id)
                    self.assertEqual(resumed.completed, 4)
                    self.assertEqual(snapshot(LocalStore(tmp.name), run_id), reference)

    def test_the_rewind_discards_rollouts_pinned_past_the_checkpoint(self) -> None:
        """every=7: nothing is checkpointed before the extent, so a crash after
        three commits rewinds to checkpoint 0 — and the rollouts sampled at
        v1..v3 must go with the ledger lines, or the Scorer would pin a bundle
        no store can rebuild. The regenerated run is the straight run."""
        cadence = Checkpointing(7)
        reference_store, run_id = straight(self, cadence)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, train, _ = arith_store(tmp.name)
        crashing = CrashingStore(tmp.name, "append_ledger", 2, True)
        with self.assertRaises(SimulatedCrash):
            run_experiment(arith_spec(train), SCHEMA, crashing, FakeEngine(),
                           FakeLearner(), checkpointing=cadence)
        crashed = LocalStore(tmp.name)
        self.assertEqual([e["update"] for e in crashed.peek_ledger(run_id)], [1, 2, 3])
        pinned_before = {
            index: crashed.peek_rollout(run_id, index)[0]["turns"][0]["policy_version"]["pi"]
            for index in range(1, 5) if crashed.peek_rollout(run_id, index) is not None}
        self.assertTrue(any(v > 0 for v in pinned_before.values()))

        run = crashed.open_run(run_id)              # THE REWIND
        self.assertEqual(run.read_ledger(), [])
        self.assertEqual(run.checkpoint_tail()["update"], 0)
        for index, version in pinned_before.items():
            self.assertEqual(crashed.peek_rollout(run_id, index) is None, version > 0,
                             f"rollout {index} at v{version}")
        resumed = run_experiment(arith_spec(train), SCHEMA, LocalStore(tmp.name),
                                 FakeEngine(), FakeLearner(), checkpointing=cadence)
        self.assertEqual(resumed.completed, 4)
        self.assertEqual(snapshot(LocalStore(tmp.name), run_id),
                         snapshot(reference_store, run_id))

    def test_a_pre_0014_directory_is_a_run_where_every_line_was_a_checkpoint(self) -> None:
        """No checkpoint record at all: the ledger is the record. Such a run
        attaches, resumes from its ledger tail, and reads done."""
        from rlstack.data.stores.base import run_progress

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        # every=1 and a crash right after checkpoint 2's line: two committed
        # updates, each with its blobs — exactly what the old discipline left
        crashing = CrashingStore(tmp.name, "append_checkpoint", 2, True)
        with self.assertRaises(SimulatedCrash):
            run_experiment(arith_spec(train), SCHEMA, crashing, FakeEngine(),
                           FakeLearner(), checkpointing=EVERY_UPDATE)
        (run_dir,) = [p for p in Path(store.path_of("runs")).iterdir() if p.is_dir()]
        (run_dir / CHECKPOINTS).unlink()           # the old discipline's directory
        legacy = LocalStore(tmp.name)
        self.assertEqual([e["update"] for e in legacy.peek_checkpoints(run_dir.name)], [1, 2])
        resumed = run_experiment(arith_spec(train), SCHEMA, legacy, FakeEngine(),
                                 FakeLearner(), checkpointing=Checkpointing(2))
        self.assertEqual(resumed.resumed_from, 2)
        self.assertEqual(resumed.completed, 4)
        self.assertTrue(run_progress(LocalStore(tmp.name), run_dir.name).done)
        # the new record starts where the new discipline started: at the
        # first checkpoint it wrote, and reads as the record from there on
        self.assertEqual(record(LocalStore(tmp.name), run_dir.name), [4])


class DrainedStopTest(unittest.TestCase):
    """A deliberate stop loses nothing (Q6): the Trainer reads the request at
    the update boundary, seals the last commit with a checkpoint, and returns
    — and the resumed run is the straight run."""

    def test_a_stop_after_update_two_checkpoints_update_two(self) -> None:
        cadence = Checkpointing(7)                  # no checkpoint before the extent
        reference_store, run_id = straight(self, cadence)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        stop = StopRequest()
        original = Trainer.commit

        async def commit_then_stop(trainer, update, *args, **kwargs):
            await original(trainer, update, *args, **kwargs)
            if update == 2:
                stop.request("test: stop after two")

        with patch.object(Trainer, "commit", commit_then_stop):
            report = asyncio.run(run_experiment_async(
                arith_spec(train), SCHEMA, store, FakeEngine(), FakeLearner(),
                checkpointing=cadence, stop=stop))
        self.assertEqual(report.completed, 2)
        self.assertEqual(record(store, run_id), [0, 2])
        self.assertEqual(blob_versions(store, run_id, "adapters"), [0, 2])
        self.assertEqual(blob_versions(store, run_id, "optim"), [2])

        resumed = run_experiment(arith_spec(train), SCHEMA, LocalStore(tmp.name),
                                 FakeEngine(), FakeLearner(), checkpointing=cadence)
        self.assertEqual(resumed.resumed_from, 2)
        self.assertEqual(resumed.completed, 4)
        # the drain ADDED one durable point the straight run never had —
        # adapters@2 and its line in the record — and changed nothing else:
        # every file both directories hold is the same bytes
        theirs, ours = snapshot(reference_store, run_id), snapshot(LocalStore(tmp.name), run_id)
        self.assertEqual(set(ours) - set(theirs), {"adapters/pi@2.bin"})
        self.assertEqual(set(theirs) - set(ours), set())
        differing = {path for path in theirs if theirs[path] != ours[path]}
        self.assertEqual(differing, {CHECKPOINTS})
        self.assertEqual(record(LocalStore(tmp.name), run_id), [0, 2, 4])

    def test_a_stop_at_a_checkpoint_writes_no_second_one(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        stop = StopRequest()
        original = Trainer.commit

        async def commit_then_stop(trainer, update, *args, **kwargs):
            await original(trainer, update, *args, **kwargs)
            if update == 2:
                stop.request()

        with patch.object(Trainer, "commit", commit_then_stop):
            report = asyncio.run(run_experiment_async(
                arith_spec(train), SCHEMA, store, FakeEngine(), FakeLearner(),
                checkpointing=Checkpointing(2), stop=stop))
        self.assertEqual(record(store, report.run_id), [0, 2])


class DeliveryTest(unittest.TestCase):
    def test_wire_pushes_every_commit_to_the_serving_pool(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        engine = FakeEngine()
        report = run_experiment(arith_spec(train), SCHEMA, store, engine,
                                FakeLearner(), checkpointing=Checkpointing(3))
        ledger = store.peek_ledger(report.run_id)
        # the initial bundle (first route) then one push per committed update,
        # blobs or no blobs: the wire carries the policy between checkpoints
        self.assertEqual(engine.bundle_log[1:], [e["bundle_id"] for e in ledger])

    def test_store_delivery_pushes_nothing(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        engine = FakeEngine()
        delivered = []
        original = Trainer.deliver

        async def spy(trainer, bundle):
            delivered.append((dict(trainer.pools), len(engine.bundle_log)))
            await original(trainer, bundle)

        with patch.object(Trainer, "deliver", spy):
            run_experiment(arith_spec(train), SCHEMA, store, engine, FakeLearner(),
                           checkpointing=Checkpointing(1, "store"))
        # the step ran at every update with NO pools to push to: every
        # install on the engine came from the Generator's own route, from
        # the blobs, after the ledger line — and the final version, which no
        # rollout consumes, was never installed at all
        self.assertEqual([pools for pools, _ in delivered], [{}] * 4)
        self.assertEqual(len(engine.bundle_log), 4)

    def test_store_delivery_hands_the_trainer_no_pool(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        seen = []
        original = Trainer.__init__

        def spy(self, *args, **kwargs):
            original(self, *args, **kwargs)
            seen.append(dict(self.pools))

        with patch.object(Trainer, "__init__", spy):
            run_experiment(arith_spec(train), SCHEMA, store, FakeEngine(), FakeLearner(),
                           checkpointing=Checkpointing(1, "store"))
            run_experiment(arith_spec(train), SCHEMA, LocalStore(tmp.name), FakeEngine(),
                           FakeLearner(), checkpointing=Checkpointing(1, "wire"))
        self.assertEqual(sorted(seen[0]), [])
        self.assertEqual(sorted(seen[1]), ["main"])

    def test_a_miss_on_an_uncheckpointed_version_parks_rather_than_lies(self) -> None:
        """An engine that lost a version no blob backs: the route cannot
        rebuild it, and says so by name — BundleUnavailable, the
        infrastructure death the desk parks and resumes from the checkpoint."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)

        class Forgetful(FakeEngine):
            """Forgets the second version it was handed — the one every=7
            never checkpoints — the moment it is asked about it."""

            def knows_bundle(self, bundle_id: str) -> bool:
                if len(self.bundle_log) >= 3 and bundle_id == self.bundle_log[2]:
                    self._known.discard(bundle_id)      # really gone
                    return False
                return super().knows_bundle(bundle_id)

        with self.assertRaises(BundleUnavailable) as caught:
            run_experiment(arith_spec(train), SCHEMA, store, Forgetful(), FakeLearner(),
                           checkpointing=Checkpointing(7))
        self.assertIn("not checkpointed", str(caught.exception))

    def test_a_miss_on_a_checkpointed_version_is_restored_from_the_store(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)

        class Forgetful(FakeEngine):
            def knows_bundle(self, bundle_id: str) -> bool:
                self._known.discard(bundle_id)   # forgets everything, always
                return False

        engine = Forgetful()
        report = run_experiment(arith_spec(train), SCHEMA, store, engine, FakeLearner(),
                                checkpointing=EVERY_UPDATE)
        self.assertEqual(report.completed, 4)
        # every route restored what the engine claimed to have lost
        self.assertGreater(len(engine.bundle_log), 5)


if __name__ == "__main__":
    unittest.main()
