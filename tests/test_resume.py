"""Resume equivalence: kill the run anywhere, reattach, get the same bytes.

The strongest claim the architecture makes: a run is a pure function of
(spec, code, data). We simulate a crash at each interesting point of the commit
protocol, resume with FRESH engine/learner objects (resume must owe nothing to
in-memory state), and require the resulting run directory to be byte-identical
to an uninterrupted run's.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

from common import arith_spec, arith_store
from rlstack import (
    FakeEngine, FakeLearner, FinishEvent, LocalStore, fake_qwen_schema,
    run_experiment,
)

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


class SimulatedCrash(RuntimeError):
    """kill -9, as an exception."""


class CrashingHandle:
    """Proxies a RunHandle; the named method crashes after `after` calls —
    before executing (post=False) or just after succeeding (post=True)."""

    def __init__(self, inner: Any, method: str, after: int, post: bool) -> None:
        self._inner = inner
        self._method = method
        self._after = after
        self._post = post
        self._calls = 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._inner, name)
        if name != self._method or not callable(attr):
            return attr

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._calls += 1
            crashing = self._calls > self._after
            if crashing and not self._post:
                raise SimulatedCrash(f"{name} call {self._calls}")
            result = attr(*args, **kwargs)
            if crashing and self._post:
                raise SimulatedCrash(f"{name} call {self._calls} (post)")
            return result

        return wrapped


class CrashingStore(LocalStore):
    def __init__(self, root: Any, method: str, after: int, post: bool = False) -> None:
        super().__init__(root)
        self._crash = (method, after, post)

    def open_run(self, run_id: str, manifest: dict | None = None,
                 subdir: str | None = None) -> Any:
        return CrashingHandle(super().open_run(run_id, manifest), *self._crash)


def snapshot(store: LocalStore, run_id: str) -> dict[str, str]:
    """{relative path: sha256} over the whole run directory."""
    root = store.path_of(f"runs/{run_id}")
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


class ResumeEquivalenceTest(unittest.TestCase):
    """One straight run vs crash-at-X + resume, for each X in the protocol."""

    # (method, calls that succeed, post) — write_blob is called twice per
    # update (adapters + optim), the others once.
    CRASH_POINTS = [
        ("write_wave", 2, False),   # crash entering update 3's wave write
        ("write_postdata", 2, False),   # wave written, pipeline output not
        ("write_blob", 5, False),       # update 3: adapters@3 written, optim@3 not
        ("append_ledger", 2, False),    # update 3 fully staged, never committed
        ("append_ledger", 3, True),     # update 3 committed; crash before sync/next
    ]

    def straight(self) -> tuple[Store, str]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        report = run_experiment(arith_spec(train), SCHEMA, store,
                                FakeEngine(), FakeLearner())
        return store, report.run_id

    def test_crash_anywhere_then_resume_is_byte_identical(self) -> None:
        reference_store, run_id = self.straight()
        reference = snapshot(reference_store, run_id)
        self.assertGreaterEqual(len(reference), 4 * 3 + 2)  # sanity: files exist

        for method, after, post in self.CRASH_POINTS:
            with self.subTest(crash=f"{method}@{after}{'+post' if post else ''}"):
                tmp = tempfile.TemporaryDirectory()
                self.addCleanup(tmp.cleanup)
                crashing, train, _ = arith_store(tmp.name)
                crashing = CrashingStore(tmp.name, method, after, post)

                with self.assertRaises(SimulatedCrash):
                    run_experiment(arith_spec(train), SCHEMA, crashing,
                                   FakeEngine(), FakeLearner())

                # fresh store handle, fresh fakes: nothing survives but disk
                resumed = run_experiment(arith_spec(train), SCHEMA,
                                         LocalStore(tmp.name), FakeEngine(),
                                         FakeLearner())
                self.assertEqual(resumed.run_id, run_id)
                self.assertIsNotNone(resumed.resumed_from)
                self.assertEqual(snapshot(LocalStore(tmp.name), run_id), reference)


class DeterminismTest(unittest.TestCase):
    def test_two_straight_runs_are_byte_identical_including_eval(self) -> None:
        snapshots = []
        for _ in range(2):
            tmp = tempfile.TemporaryDirectory()
            self.addCleanup(tmp.cleanup)
            store, train, heldout = arith_store(tmp.name)
            report = run_experiment(arith_spec(train, heldout), SCHEMA, store,
                                    FakeEngine(), FakeLearner())
            snapshots.append(snapshot(store, report.run_id))
        self.assertEqual(snapshots[0], snapshots[1])


class InterleavingEngine(FakeEngine):
    """A FakeEngine that hands control back to the event loop a seed-dependent
    number of times, so concurrent episodes finish in an order that has nothing
    to do with the order they were launched in.

    It changes no content: the fakes' determinism contract makes a completion a
    pure function of its seed, and this only decides who gets there first.
    """

    def __init__(self) -> None:
        super().__init__()
        self.launched: list[int] = []
        self.finished: list[int] = []

    async def sample_tokens(self, messages, sampling, stop, bundle_id, seed):
        self.launched.append(seed)
        for _ in range(seed % 7):
            await asyncio.sleep(0)
        async for event in super().sample_tokens(messages, sampling, stop,
                                                 bundle_id, seed):
            if isinstance(event, FinishEvent):
                self.finished.append(seed)
            yield event
            await asyncio.sleep(0)


class CompletionOrderTest(unittest.TestCase):
    """The bytes may not know which episode finished first.

    Both fans that sample — the Generator's wave and the Evaluator's held-out
    set — launch every (task, sample) episode at once under a max_inflight
    semaphore, so the width of the fan and the speed of each episode decide
    the completion order. Neither may reach the run directory: seeds are
    derived per (task, sample) before anything is scheduled, and every
    reduction runs in task order (#53). So the same run at max_inflight=1, at
    64, and with the engine finishing episodes out of order is the same bytes,
    eval/ included.
    """

    def go(self, engine: FakeEngine, max_inflight: int) -> dict[str, str]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, heldout = arith_store(tmp.name)
        report = run_experiment(arith_spec(train, heldout), SCHEMA, store,
                                engine, FakeLearner(), max_inflight)
        return snapshot(store, report.run_id)

    def test_run_bytes_do_not_depend_on_the_completion_order(self) -> None:
        """Completion order must not reach the bytes (#53) — the whole run
        dir, under a scrambling engine. (The measurement half of this claim
        lives in test_measure.py now: measurement left the run dir, #70.)"""
        reference = self.go(FakeEngine(), 64)
        scrambler = InterleavingEngine()
        narrow = InterleavingEngine()
        for label, run in (("serial", self.go(FakeEngine(), 1)),
                           ("scrambled", self.go(scrambler, 64)),
                           ("scrambled, narrow", self.go(narrow, 3))):
            with self.subTest(schedule=label):
                self.assertEqual(run, reference)

        # the pin is only worth something if the scrambling really scrambled
        self.assertNotEqual(scrambler.finished, scrambler.launched)


class AnnotationLeavesTheRunAloneTest(unittest.TestCase):
    """#58's half of this file's claim: an annotation is FLAVORTEXT beside
    runs/, so naming, tagging and noting a run changes not one byte of its
    run directory — and crash recovery, which sweeps everything no ledger
    line committed, never sees it."""

    def test_annotating_a_run_changes_no_byte_of_it(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, heldout = arith_store(tmp.name)
        report = run_experiment(arith_spec(train, heldout), SCHEMA, store,
                                FakeEngine(), FakeLearner())
        before = snapshot(store, report.run_id)
        self.assertTrue(before)

        store.annotate_run(report.run_id, name="the good one",
                           tags=["opd", "l3-5"], note="kl fell to 0.28")
        self.assertEqual(snapshot(store, report.run_id), before)
        self.assertTrue(store.path_of("annotations.jsonl").exists())
        self.assertNotIn("annotations.jsonl", before)   # never inside the run

        store.open_run(report.run_id)                   # attach: sweeps unsealed
        self.assertEqual(snapshot(store, report.run_id), before)
        self.assertEqual(store.read_annotations()[report.run_id]["tags"],
                         ["opd", "l3-5"])


if __name__ == "__main__":
    unittest.main()
