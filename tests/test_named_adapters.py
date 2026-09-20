"""Named adapters in the store (ADR 0019): written once, promised at the
writer's birth, and answerable — present, promised, orphaned, unknown.

The properties under test are the ones consumers lean on: a name never changes
bytes once it has them; a name is present exactly when its meta is; a missing
name says whether it is still coming (its writer is still work) or never
coming (its writer is done, stopped or failed); and every write a consumer on
other metal waits on is persisted when it lands.
"""

from __future__ import annotations

import json
import tempfile
import unittest

from rlstack import LocalStore, ModalVolumeStore
from rlstack.data.stores.base import (
    NamedAdapterConflict, StoreError, check_name, named_key, run_ended, run_progress,
    standing_disposition,
)

EXP = "dreams/g1"


def a_run(store, run_id: str, *, plan: str, planned: int, done: int,
          checkpointed: int | None = None, subdir: str = EXP) -> str:
    """One run directory with a `plan`-kind plan of `planned` lines, `done`
    ledger lines, and a checkpoint at `checkpointed` (default: at `done`).
    Returns its run reference — what a promise names as the writer."""
    run = store.open_run(run_id, manifest={"run_id": run_id}, subdir=subdir)
    run.write_plan(plan, b"".join(b'{"job": %d}\n' % i for i in range(planned)))
    for update in range(1, done + 1):
        run.append_ledger({"update": update})
    sealed = done if checkpointed is None else checkpointed
    if sealed:
        run.append_checkpoint(sealed, {})
    return f"{subdir}/{run_id}"


class NameGrammarTest(unittest.TestCase):
    def test_plain_and_foldered_names_are_names(self) -> None:
        for name in ("m0", "mem/topic-03/007", "a.b_c-d", "v1.2/x"):
            self.assertEqual(check_name(name), name)

    def test_what_a_name_may_not_be(self) -> None:
        for name in ("", "/abs", "a/../b", "..", "a//b", "a/", "a+b", "lib:a",
                     "a b", "exp/manifest"):
            with self.assertRaises(StoreError, msg=name):
                check_name(name)

    def test_names_live_under_the_subdir_beside_its_runs(self) -> None:
        self.assertEqual(named_key(EXP, "mem/007", ".bin"),
                         "runs/dreams/g1/names/mem/007.bin")


class WriteOnceTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)

    def test_a_written_name_reads_back_with_its_meta_and_fingerprint(self) -> None:
        self.store.write_named(EXP, "m0", b"payload", {"r": 4, "base": "qwen"})
        self.assertEqual(self.store.read_named(EXP, "m0"), b"payload")
        self.assertEqual(self.store.named_meta(EXP, "m0"),
                         {"r": 4, "base": "qwen",
                          "sha256": self.store.fingerprint(b"payload")})
        self.assertEqual(self.store.named_state(EXP, "m0"), "present")

    def test_a_present_payload_is_read_from_the_store_once(self) -> None:
        """Write-once makes the cache always right; absence is never cached."""
        self.assertIsNone(self.store.read_named(EXP, "mem/007"))
        self.store.write_named(EXP, "mem/007", b"payload", {})
        reads = []
        inner = self.store._read
        self.store._read = lambda key: (reads.append(key), inner(key))[1]
        first, second = self.store.read_named(EXP, "mem/007"), self.store.read_named(EXP, "mem/007")
        self.assertEqual((first, second), (b"payload", b"payload"))
        self.assertEqual(sum(key.endswith(".bin") for key in reads), 1)

    def test_an_absent_name_reads_as_none(self) -> None:
        self.assertIsNone(self.store.read_named(EXP, "m0"))
        self.assertIsNone(self.store.named_meta(EXP, "m0"))

    def test_an_identical_rewrite_is_a_no_op_and_keeps_the_first_meta(self) -> None:
        self.store.write_named(EXP, "m0", b"payload", {"attempt": 1})
        self.store.write_named(EXP, "m0", b"payload", {"attempt": 2})
        self.assertEqual(self.store.named_meta(EXP, "m0")["attempt"], 1)

    def test_different_bytes_are_a_conflict_and_the_name_keeps_its_bytes(self) -> None:
        self.store.write_named(EXP, "m0", b"payload", {})
        with self.assertRaises(NamedAdapterConflict):
            self.store.write_named(EXP, "m0", b"other", {})
        self.assertEqual(self.store.read_named(EXP, "m0"), b"payload")

    def test_a_payload_with_no_meta_is_debris_not_a_name(self) -> None:
        """A writer killed between the payload and its seal left no name: the
        rerun of that fit writes whatever it computes this time."""
        self.store._write(named_key(EXP, "m0", ".bin"), b"torn attempt")
        self.assertIsNone(self.store.read_named(EXP, "m0"))
        self.assertEqual(self.store.named_state(EXP, "m0"), "unknown")
        self.store.write_named(EXP, "m0", b"second attempt", {})
        self.assertEqual(self.store.read_named(EXP, "m0"), b"second attempt")

    def test_the_same_name_under_two_subdirs_is_two_adapters(self) -> None:
        self.store.write_named(EXP, "m0", b"one", {})
        self.store.write_named("dreams/g2", "m0", b"two", {})
        self.assertEqual(self.store.read_named(EXP, "m0"), b"one")
        self.assertEqual(self.store.read_named("dreams/g2", "m0"), b"two")

    def test_an_illegal_name_is_refused_at_every_verb(self) -> None:
        for verb in (lambda: self.store.write_named(EXP, "../x", b"", {}),
                     lambda: self.store.read_named(EXP, "/x"),
                     lambda: self.store.named_state(EXP, "a+b"),
                     lambda: self.store.promise_named(EXP, ["ok", "a//b"], "exp/r")):
            with self.assertRaises(StoreError):
                verb()

    def test_names_are_not_runs(self) -> None:
        a_run(self.store, "fit-1", plan="fit", planned=1, done=0)
        self.store.write_named(EXP, "mem/007", b"payload", {})
        self.assertEqual(self.store.list_runs(), [f"{EXP}/fit-1"])


class NamedStateTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)

    def test_no_bytes_and_no_promise_is_unknown(self) -> None:
        self.assertEqual(self.store.named_state(EXP, "m0"), "unknown")

    def test_a_promise_by_an_unfinished_writer_is_promised(self) -> None:
        writer = a_run(self.store, "fit-1", plan="fit", planned=3, done=1)
        self.store.promise_named(EXP, ["m0", "m1"], writer)
        self.assertEqual(self.store.named_writer(EXP, "m1"), writer)
        self.assertEqual(self.store.named_state(EXP, "m0"), "promised")

    def test_a_promise_by_a_writer_not_yet_born_is_promised(self) -> None:
        self.store.promise_named(EXP, ["m0"], f"{EXP}/fit-unborn")
        self.assertEqual(self.store.named_state(EXP, "m0"), "promised")

    def test_a_done_writer_that_left_no_bytes_orphans_the_name(self) -> None:
        writer = a_run(self.store, "fit-1", plan="fit", planned=2, done=2)
        self.store.promise_named(EXP, ["m0", "m1"], writer)
        self.store.write_named(EXP, "m0", b"payload", {})
        self.assertEqual(self.store.named_state(EXP, "m0"), "present")
        self.assertEqual(self.store.named_state(EXP, "m1"), "orphaned")

    def test_a_writer_whose_last_job_is_not_checkpointed_is_still_work(self) -> None:
        """Done is the extent's done (ADR 0014): a ledger at the plan's end
        whose checkpoint never landed rewinds on attach, so it may yet write."""
        writer = a_run(self.store, "fit-1", plan="fit", planned=2, done=2, checkpointed=1)
        self.store.promise_named(EXP, ["m0"], writer)
        self.assertFalse(run_progress(self.store, writer).done)
        self.assertEqual(self.store.named_state(EXP, "m0"), "promised")

    def test_a_fit_plan_is_a_run_s_extent(self) -> None:
        writer = a_run(self.store, "fit-1", plan="fit", planned=3, done=2)
        progress = run_progress(self.store, writer)
        self.assertEqual((progress.extent, progress.completed, progress.planned),
                         ("fit", 2, 3))

    def test_a_training_writer_counts_too(self) -> None:
        writer = a_run(self.store, "train-1", plan="train", planned=2, done=2)
        self.store.promise_named(EXP, ["m0"], writer)
        self.assertEqual(self.store.named_state(EXP, "m0"), "orphaned")

    def test_a_stopped_or_failed_writer_orphans_what_it_never_wrote(self) -> None:
        for word in ("stopped", "failed"):
            with self.subTest(word):
                writer = a_run(self.store, f"fit-{word}", plan="fit", planned=3, done=1)
                self.store.promise_named(EXP, [f"m-{word}"], writer)
                self.store.append_fleet_event({"event": word, "run_id": f"fit-{word}"})
                self.assertEqual(standing_disposition(self.store, writer), word)
                self.assertTrue(run_ended(self.store, writer))
                self.assertEqual(self.store.named_state(EXP, f"m-{word}"), "orphaned")

    def test_a_parked_writer_is_still_coming(self) -> None:
        writer = a_run(self.store, "fit-1", plan="fit", planned=3, done=1)
        self.store.promise_named(EXP, ["m0"], writer)
        self.store.append_fleet_event({"event": "parked", "run_id": "fit-1"})
        self.assertEqual(self.store.named_state(EXP, "m0"), "promised")

    def test_a_resubmitted_writer_is_work_again(self) -> None:
        writer = a_run(self.store, "fit-1", plan="fit", planned=3, done=1)
        self.store.promise_named(EXP, ["m0"], writer)
        self.store.append_fleet_event({"event": "failed", "run_id": "fit-1"})
        self.assertEqual(self.store.named_state(EXP, "m0"), "orphaned")
        self.store.append_fleet_event({"event": "place", "run_id": "fit-1",
                                       "delivered": True, "accepted": True})
        self.assertEqual(self.store.named_state(EXP, "m0"), "promised")

    def test_the_latest_promise_takes_an_orphan_over(self) -> None:
        dead = a_run(self.store, "fit-1", plan="fit", planned=1, done=1)
        self.store.promise_named(EXP, ["m0"], dead)
        self.assertEqual(self.store.named_state(EXP, "m0"), "orphaned")
        again = a_run(self.store, "fit-2", plan="fit", planned=1, done=0)
        self.store.promise_named(EXP, ["m0"], again)
        self.assertEqual(self.store.named_writer(EXP, "m0"), again)
        self.assertEqual(self.store.named_state(EXP, "m0"), "promised")

    def test_a_writer_promises_every_name_in_one_write(self) -> None:
        """The first fit runs on metal spent half an hour writing 200 promise
        files at ten seconds each before their first step (2026-09-19)."""
        writer = a_run(self.store, "fit-1", plan="fit", planned=200, done=0)
        before = set(self.store._list(f"runs/{EXP}/names/"))
        self.store.promise_named(EXP, [f"mem/{i:04d}" for i in range(200)], writer)
        self.assertEqual(len(set(self.store._list(f"runs/{EXP}/names/")) - before), 1)
        self.assertEqual(self.store.named_state(EXP, "mem/0199"), "promised")
        self.assertEqual(self.store.named_state(EXP, "mem/0200"), "unknown")

    def test_a_name_two_writers_promised_is_coming_while_either_is_work(self) -> None:
        ended = a_run(self.store, "fit-old", plan="fit", planned=1, done=1)
        live = a_run(self.store, "fit-new", plan="fit", planned=1, done=0)
        self.store.promise_named(EXP, ["m0"], ended)
        self.assertEqual(self.store.named_state(EXP, "m0"), "orphaned")
        self.store.promise_named(EXP, ["m0"], live)
        self.assertEqual(self.store.named_writer(EXP, "m0"), live)
        self.assertEqual(self.store.named_state(EXP, "m0"), "promised")

    def test_a_promise_left_one_file_per_name_is_still_a_promise(self) -> None:
        """What the runs in flight when the layout changed had written."""
        from rlstack.data.stores.base import NAMED_PROMISE, named_key
        writer = a_run(self.store, "fit-1", plan="fit", planned=2, done=0)
        self.store._write(named_key(EXP, "m0", NAMED_PROMISE),
                          json.dumps({"writer": writer}).encode("utf-8"))
        self.assertEqual(self.store.named_state(EXP, "m0"), "promised")

    def test_the_promise_folder_is_not_a_name(self) -> None:
        with self.assertRaises(StoreError):
            self.store.write_named(EXP, "_promises/abc", b"x", {})

    def test_bytes_outrank_every_fact_about_the_writer(self) -> None:
        writer = a_run(self.store, "fit-1", plan="fit", planned=1, done=1)
        self.store.promise_named(EXP, ["m0"], writer)
        self.store.append_fleet_event({"event": "failed", "run_id": "fit-1"})
        self.store.write_named(EXP, "m0", b"payload", {})
        self.assertEqual(self.store.named_state(EXP, "m0"), "present")


class RecordingVolume:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1

    def iterdir(self, path, *, recursive=True):
        return iter(())

    def read_file(self, key):
        raise FileNotFoundError(key)


class NamedWritesPersistTest(unittest.TestCase):
    """On a staging backend a name and a promise are committed AS THEY LAND:
    the consumer waiting on them is another container."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.volume = RecordingVolume()
        self.store = ModalVolumeStore(tmp.name, volume=self.volume)

    def test_a_written_name_is_committed_once(self) -> None:
        self.store.write_named(EXP, "m0", b"payload", {})
        self.assertEqual(self.volume.commits, 1)
        self.store.write_named(EXP, "m0", b"payload", {})        # the no-op
        self.assertEqual(self.volume.commits, 1)

    def test_a_batch_of_promises_is_one_commit_and_a_restatement_none(self) -> None:
        self.store.promise_named(EXP, ["m0", "m1", "m2"], f"{EXP}/fit-1")
        self.assertEqual(self.volume.commits, 1)
        self.store.promise_named(EXP, ["m0", "m1", "m2"], f"{EXP}/fit-1")
        self.assertEqual(self.volume.commits, 1)

    def test_a_name_committed_by_another_container_is_present_here(self) -> None:
        """The mount is a boot-time snapshot; presence reads through it to the
        volume's committed view, like every cross-container blob."""
        committed = {named_key(EXP, "m0", ".json"): json.dumps({"sha256": self.store.fingerprint(b"payload")}).encode(),
                     named_key(EXP, "m0", ".bin"): b"payload"}
        self.volume.read_file = lambda key: iter([committed[key]]) \
            if key in committed else (_ for _ in ()).throw(FileNotFoundError(key))
        self.assertEqual(self.store.named_state(EXP, "m0"), "present")
        self.assertEqual(self.store.read_named(EXP, "m0"), b"payload")


if __name__ == "__main__":
    unittest.main()
