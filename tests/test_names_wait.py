"""The consumer's half of named adapters (ADR 0019): which names a wave needs,
the wait until they exist, and the two roles that hand residents the bytes.

The properties under test: a role string's `lib:<name>` parts are read whether
alone or stacked; the wait returns on bytes, refuses on an orphan, and keeps
waiting — audibly — on a promise or on nothing; the Generator hands every
library set to its pool before it samples and the Trainer to its learner
before it steps, dropping what the next wave does not name; and a run whose
plans mention no library takes none of these steps.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import tempfile
import threading
import unittest

from common import TRAIN, arith_spec, arith_task_bytes, cas_uri, task_ids
from rlstack import (
    FakeEngine, FakeLearner, GroupPlan, LocalStore, Plans, PolicySpec, Replay, RunPlan,
    RunSignals, Sample, Task, WavePlan, WaveRef, encode, fake_qwen_schema,
)
from rlstack.data.plan import EVAL, PlanError
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.loop import run_experiment_async
from rlstack.runner.meters import HostJournal
from rlstack.runner.names import (
    ENDED_CHECK_EVERY_S, NOTE_EVERY_S, NamedAdapterOrphaned, NamesWait, check_plan_names,
    lib_names, names_ready, names_subdir, row_lib_names, wave_lib_names,
)

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
EXP = "dreams/g1"
ENV = "boundary_fixtures:single_turn"


def go(coro):
    return asyncio.run(coro)


def fast_signals() -> RunSignals:
    """A blackboard whose beat is short enough that a test never feels it."""
    return RunSignals(poll_seconds=0.005)


def a_fit_run(store, run_id: str, *, planned: int, done: int) -> str:
    """A writer: a fit run with `done` of `planned` jobs committed and sealed."""
    run = store.open_run(run_id, manifest={"run_id": run_id}, subdir=EXP)
    run.write_plan("fit", b"".join(b'{"job": %d}\n' % i for i in range(planned)))
    for update in range(1, done + 1):
        run.append_ledger({"update": update})
    if done:
        run.append_checkpoint(done, {})
    return f"{EXP}/{run_id}"




class NamesReadyTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)

    def test_present_names_return_at_once_and_no_names_ask_nothing(self) -> None:
        self.store.write_named(EXP, "m0", b"payload", {})
        go(names_ready(self.store, EXP, ["m0"], fast_signals()))
        go(names_ready(None, EXP, [], None))        # nothing needed: nothing touched

    def test_an_orphan_refuses_naming_the_adapter_and_its_writer(self) -> None:
        writer = a_fit_run(self.store, "fit-1", planned=1, done=1)
        self.store.promise_named(EXP, ["m0"], writer)
        with self.assertRaises(NamedAdapterOrphaned) as refused:
            go(names_ready(self.store, EXP, ["m0"], fast_signals()))
        self.assertIn("'m0'", str(refused.exception))
        self.assertIn(writer, str(refused.exception))

    def test_a_promised_and_an_unknown_name_are_waited_for_until_written(self) -> None:
        writer = a_fit_run(self.store, "fit-1", planned=2, done=0)
        self.store.promise_named(EXP, ["promised"], writer)
        order: list[str] = []

        async def scenario():
            signals = fast_signals()

            async def the_fit_run():
                for name in ("promised", "unknown"):
                    await signals.wait_once()            # the consumer is waiting by now
                    order.append(f"wrote {name}")
                    self.store.write_named(EXP, name, name.encode(), {})
                    await signals.notify()

            fitting = asyncio.get_running_loop().create_task(the_fit_run())
            await names_ready(self.store, EXP, ["promised", "unknown"], signals)
            order.append("ready")
            await fitting

        go(scenario())
        self.assertEqual(order, ["wrote promised", "wrote unknown", "ready"])

    def test_a_stop_is_the_second_way_out(self) -> None:
        asked: list[bool] = []

        def stopped() -> bool:
            asked.append(True)
            return len(asked) > 2                      # a stop arrives on the third beat

        go(names_ready(self.store, EXP, ["never"], fast_signals(), stopped=stopped))
        self.assertEqual(len(asked), 3)
        self.assertEqual(self.store.named_state(EXP, "never"), "unknown")


class NamesWaitPacingTest(unittest.TestCase):
    """The predicate, beat by beat, on a clock the test owns."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        self.now = [1000.0]
        self.journal = HostJournal(self.store, "metal-0")

    def waiting(self, names) -> NamesWait:
        return NamesWait(self.store, EXP, names, self.journal, lambda: self.now[0])

    def test_a_writer_that_fails_mid_wait_is_refused_at_the_next_ended_check(self) -> None:
        writer = a_fit_run(self.store, "fit-1", planned=2, done=0)
        self.store.promise_named(EXP, ["m0"], writer)
        waiting = self.waiting(["m0"])
        self.assertFalse(waiting.all_present())        # first beat: asked, still work
        self.store.append_fleet_event({"event": "failed", "run_id": "fit-1"})
        self.now[0] += ENDED_CHECK_EVERY_S / 2
        self.assertFalse(waiting.all_present())        # between checks: presence only
        self.now[0] += ENDED_CHECK_EVERY_S / 2
        with self.assertRaises(NamedAdapterOrphaned):
            waiting.all_present()

    def test_a_long_wait_says_so_every_ten_minutes_and_not_between(self) -> None:
        writer = a_fit_run(self.store, "fit-1", planned=2, done=0)
        self.store.promise_named(EXP, ["m0"], writer)
        waiting = self.waiting(["m0", "m1"])
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            self.assertFalse(waiting.all_present())
            self.now[0] += NOTE_EVERY_S - 1
            self.assertFalse(waiting.all_present())
            self.assertEqual(said.getvalue(), "")
            self.now[0] += 1
            self.assertFalse(waiting.all_present())
            self.assertFalse(waiting.all_present())    # the same minute: said once
        self.assertEqual(said.getvalue().count("[names]"), 1)
        self.assertIn("'m0': 'promised'", said.getvalue())
        self.assertIn("'m1': 'unknown'", said.getvalue())
        notes = [e for e in self.store.read_host_log("metal-0") if e["event"] == "names-wait"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["missing"], {"m0": "promised", "m1": "unknown"})
        self.assertEqual(notes[0]["waited_s"], NOTE_EVERY_S)

    def test_a_name_that_arrives_leaves_the_wait(self) -> None:
        waiting = self.waiting(["m0", "m1"])
        self.assertFalse(waiting.all_present())
        self.store.write_named(EXP, "m0", b"payload", {})
        self.assertFalse(waiting.all_present())
        self.assertEqual(waiting.missing, ("m1",))
        self.store.write_named(EXP, "m1", b"payload", {})
        self.assertTrue(waiting.all_present())


# ---------------------------------------------------------------------------
# the roles: a whole run on fakes, with residents that record what they are handed
# ---------------------------------------------------------------------------









def tree_bytes(store: LocalStore) -> dict[str, bytes]:
    return {key: store._read(key) for key in store._list("runs/")}


if __name__ == "__main__":
    unittest.main()
