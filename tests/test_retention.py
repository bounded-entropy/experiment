"""Retention: what a run's store may forget, and the proof it still resumes.

Three layers, in the order the design has them. The POLICY is a pure function
of the ledger, so it is tested with no store in the room at all. The SWEEP is
the store's one deletion of COMMITTED bytes, and the tests hold it to the two
things that make it safe: it can address nothing but a blob version, and it
checks the whole batch before it deletes anything. Then the property the rest
exists to protect — a swept run resumes, restores its tenant from the tail, and
finishes.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from typing import Any

from common import arith_spec, arith_store
from rlstack import (
    FakeEngine, FakeLearner, KeepRestorable, LocalStore, ModalVolumeStore,
    RetentionPolicy, StoreError, fake_qwen_schema, run_experiment,
)
from rlstack.runner.loop import experiment_identity

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def ledger(*versions: dict[str, int]) -> list[dict[str, Any]]:
    """A commit record as the trainer writes it: update u sealed version map u."""
    return [{"update": u, "versions": dict(v)}
            for u, v in enumerate(versions, start=1)]


# ---------------------------------------------------------------------------
# the policy: a pure function of the ledger
# ---------------------------------------------------------------------------

class KeepRestorableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = KeepRestorable()
        self.ledger = ledger({"pi": 1}, {"pi": 2}, {"pi": 3}, {"pi": 4})

    def test_the_tail_is_never_named(self) -> None:
        """The version the ledger tail commits is the run's live state."""
        named = self.policy.expendable(self.ledger)
        self.assertNotIn(("optim", "pi", 4), named)
        self.assertEqual(named, (("optim", "pi", 1), ("optim", "pi", 2),
                                 ("optim", "pi", 3)))

    def test_no_adapter_is_ever_named(self) -> None:
        """Adapter blobs have readers that pin HISTORICAL versions — eval's
        bundle_for, restore_bundle_on, a WarmStart naming @v — so deleting one
        breaks restore for that version forever."""
        sections = {section for section, _, _ in
                    self.policy.expendable(self.ledger)}
        self.assertEqual(sections, {"optim"})

    def test_the_same_ledger_gives_the_same_answer(self) -> None:
        """No store, no clock, no state: two calls and two instances agree, so
        a sweep that never ran costs only the sweep that follows it."""
        self.assertEqual(self.policy.expendable(self.ledger),
                         self.policy.expendable(self.ledger))
        self.assertEqual(self.policy.expendable(self.ledger),
                         KeepRestorable().expendable(list(self.ledger)))

    def test_an_empty_ledger_names_nothing(self) -> None:
        """A run that has committed nothing has nothing stale."""
        self.assertEqual(self.policy.expendable([]), ())

    def test_a_delta_the_tail_does_not_name_is_kept(self) -> None:
        """The tail is the only evidence of what is live; absent evidence
        keeps bytes."""
        entries = ledger({"pi": 1, "aux": 1}, {"pi": 2, "aux": 2}, {"pi": 3})
        named = self.policy.expendable(entries)
        self.assertEqual(named, (("optim", "pi", 1), ("optim", "pi", 2)))

    def test_a_frozen_delta_at_version_zero_is_never_named(self) -> None:
        """A frozen delta does not advance, so version 0 is also its tail."""
        entries = ledger({"pi": 1, "frozen": 0}, {"pi": 2, "frozen": 0})
        self.assertEqual(self.policy.expendable(entries), (("optim", "pi", 1),))

    def test_two_trainable_deltas_are_named_independently(self) -> None:
        entries = ledger({"pi": 1, "value": 1}, {"pi": 2, "value": 2})
        self.assertEqual(self.policy.expendable(entries),
                         (("optim", "pi", 1), ("optim", "value", 1)))


# ---------------------------------------------------------------------------
# the sweep: the store's deletion of committed bytes
# ---------------------------------------------------------------------------

class NamesWhatItIsTold(RetentionPolicy):
    """A policy under test's control — the ABC is the whole contract, so a
    test writes its own the same way an experiment would."""

    def __init__(self, *triples: tuple[str, str, int]) -> None:
        self.triples = triples

    def expendable(self, ledger):
        return self.triples


class SweepTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        self.run = self.store.open_run("r1", manifest={"run_id": "r1"})
        for update in range(1, 5):
            self.run.write_wave(update, [{"update": update}])
            self.run.write_blob("adapters", "pi", update, b"delta" * update)
            self.run.write_blob("optim", "pi", update, b"moments" * update)
            self.run.append_ledger({"update": update, "versions": {"pi": update}})

    def optim_versions(self) -> list[int]:
        return [int(key.rsplit("@", 1)[1][: -len(".bin")])
                for key in self.store._list("runs/r1/optim")]

    def digest(self) -> dict[str, str]:
        """Every key outside optim/, by content — what a sweep may not move."""
        return {key: hashlib.sha256(self.store._read(key)).hexdigest()
                for key in self.store._list("runs/r1")
                if "/optim/" not in key}

    def test_it_frees_what_the_policy_named_and_reports_the_bytes(self) -> None:
        before = self.digest()
        swept = self.run.sweep(KeepRestorable())

        self.assertEqual(swept.blobs, (("optim", "pi", 1), ("optim", "pi", 2),
                                       ("optim", "pi", 3)))
        self.assertEqual(swept.freed, len(b"moments") * (1 + 2 + 3))
        self.assertEqual(self.optim_versions(), [4])
        self.assertEqual(self.digest(), before,
                         "a sweep touched something outside optim/")

    def test_sweeping_twice_frees_nothing_the_second_time(self) -> None:
        first = self.run.sweep(KeepRestorable())
        second = self.run.sweep(KeepRestorable())

        self.assertEqual(len(first), 3)
        self.assertEqual(second.blobs, ())
        self.assertEqual(second.freed, 0)
        self.assertEqual(self.optim_versions(), [4])

    def test_a_blob_already_gone_is_skipped_not_an_error(self) -> None:
        self.store._delete("runs/r1/optim/pi@2.bin")
        swept = self.run.sweep(KeepRestorable())
        self.assertEqual(swept.blobs, (("optim", "pi", 1), ("optim", "pi", 3)))

    def test_a_policy_naming_the_tail_frees_nothing_at_all(self) -> None:
        """The floor under every policy — and it holds for the WHOLE batch,
        so a policy that names one live version does not get to delete the
        stale ones on its way to being refused."""
        policy = NamesWhatItIsTold(("optim", "pi", 1), ("optim", "pi", 4))
        with self.assertRaises(StoreError):
            self.run.sweep(policy)
        self.assertEqual(self.optim_versions(), [1, 2, 3, 4])

    def test_a_policy_can_name_nothing_but_a_blob_version(self) -> None:
        """Every deletion is addressed through _blob_key, so the ledger, the
        manifest and the sealed waves are not expressible from here."""
        before = self.digest()
        for triple in (("waves", "000001", 1), ("ledger.jsonl", "pi", 1),
                       ("eval", "summary", 1)):
            with self.assertRaises(ValueError):
                self.run.sweep(NamesWhatItIsTold(triple))
        self.assertEqual(self.digest(), before)
        self.assertEqual(self.optim_versions(), [1, 2, 3, 4])

    def test_the_ledger_still_reads_and_the_waves_still_open(self) -> None:
        self.run.sweep(KeepRestorable())
        self.assertEqual([e["update"] for e in self.run.read_ledger()],
                         [1, 2, 3, 4])
        self.assertEqual(self.run.read_wave(3), [{"update": 3}])
        for version in range(1, 5):
            self.run.read_blob("adapters", "pi", version)   # raises if missing


class VolumeSweepTest(unittest.TestCase):
    """A deletion on a mounted Volume stages exactly like a write, so a sweep
    that freed bytes must commit or the bytes come back."""

    class RecordingVolume:
        def __init__(self) -> None:
            self.commits = 0

        def commit(self) -> None:
            self.commits += 1

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.volume = self.RecordingVolume()
        self.store = ModalVolumeStore(tmp.name, volume=self.volume)
        self.run = self.store.open_run("r1", manifest={"run_id": "r1"})
        for update in (1, 2):
            self.run.write_blob("optim", "pi", update, b"moments")
            self.run.append_ledger({"update": update, "versions": {"pi": update}})

    def test_a_sweep_that_freed_bytes_commits_them(self) -> None:
        before = self.volume.commits
        swept = self.run.sweep(KeepRestorable())
        self.assertEqual(len(swept), 1)
        self.assertEqual(self.volume.commits, before + 1)

    def test_a_sweep_that_freed_nothing_commits_nothing(self) -> None:
        self.run.sweep(KeepRestorable())
        before = self.volume.commits
        self.run.sweep(KeepRestorable())
        self.assertEqual(self.volume.commits, before)


# ---------------------------------------------------------------------------
# the property all of it exists to protect: a swept run resumes
# ---------------------------------------------------------------------------

class Interrupted(RuntimeError):
    """kill -9 in the middle of an update's gradient, as an exception."""


class StopsAtUpdate(FakeLearner):
    """A learner that dies entering the Nth update's optimizer step: the
    updates before it are committed, that one never seals."""

    def __init__(self, at: int) -> None:
        super().__init__()
        self.at = at
        self.calls = 0

    def optim_step(self, tenant: str) -> None:
        self.calls += 1
        if self.calls >= self.at:
            raise Interrupted(f"update {self.at}")
        super().optim_step(tenant)


class SweptRunTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.store, self.train, _ = arith_store(self.root)
        self.spec = arith_spec(self.train)

    def optim_versions(self, run_id: str) -> list[int]:
        return sorted(int(key.rsplit("@", 1)[1][: -len(".bin")])
                      for key in self.store._list(f"runs/{run_id}/optim"))

    def test_a_finished_run_keeps_every_adapter_and_one_optim(self) -> None:
        """The trainer swept as it went: four updates, four deltas, one set
        of moments."""
        report = run_experiment(self.spec, SCHEMA, self.store, FakeEngine(),
                                FakeLearner())
        run = self.store.open_run(report.run_id)
        for version in range(1, 5):
            run.read_blob("adapters", "pi", version)     # raises if missing
        self.assertEqual(self.optim_versions(report.run_id), [4])

    def test_a_swept_run_restores_its_tenant_from_the_tail(self) -> None:
        """Attach a finished, swept run with FRESH fakes: the learner comes
        back from the tail's blobs alone, and `steps` proves the OPTIM blob
        was read (a FakeLearner recovers its step count from nowhere else)."""
        report = run_experiment(self.spec, SCHEMA, self.store, FakeEngine(),
                                FakeLearner())
        self.store.open_run(report.run_id).sweep(KeepRestorable())

        learner = FakeLearner()
        again = run_experiment(self.spec, SCHEMA, self.store, FakeEngine(),
                               learner)
        self.assertEqual(again.run_id, report.run_id)
        self.assertEqual(again.resumed_from, 4)
        self.assertEqual(learner._tenant(report.run_id).steps, 4)

    def test_an_interrupted_swept_run_resumes_and_finishes(self) -> None:
        """The important one. Commit two updates, sweep, then resume on fresh
        metal and run to the end: the ledger is whole and the moments the
        third update trained from were the tail's."""
        with self.assertRaises(Interrupted):
            run_experiment(self.spec, SCHEMA, self.store, FakeEngine(),
                           StopsAtUpdate(3))
        run_id = experiment_identity(self.spec, SCHEMA)
        run = self.store.open_run(run_id)
        self.assertEqual(int(run.ledger_tail()["update"]), 2)

        swept = run.sweep(KeepRestorable())
        self.assertEqual(swept.blobs, ())            # the trainer already had
        self.assertEqual(self.optim_versions(run_id), [2])

        learner = FakeLearner()
        report = run_experiment(self.spec, SCHEMA, self.store, FakeEngine(),
                                learner)
        self.assertEqual(report.resumed_from, 2)
        self.assertEqual([e["update"] for e in run.read_ledger()], [1, 2, 3, 4])
        self.assertEqual(learner._tenant(run_id).steps, 4)
        self.assertEqual(self.optim_versions(run_id), [4])

    def test_sweeping_moves_no_byte_of_identity(self) -> None:
        """Retention changes what is RECOVERABLE, never what was COMPUTED: the
        run_id is a function of (spec, code, data) and none of them is here."""
        report = run_experiment(self.spec, SCHEMA, self.store, FakeEngine(),
                                FakeLearner())
        manifest = self.store._read(f"runs/{report.run_id}/manifest.json")

        self.store.open_run(report.run_id).sweep(KeepRestorable())

        self.assertEqual(experiment_identity(self.spec, SCHEMA), report.run_id)
        self.assertEqual(self.store._read(f"runs/{report.run_id}/manifest.json"),
                         manifest)


if __name__ == "__main__":
    unittest.main()
