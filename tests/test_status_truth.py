"""Status is reconciled against the LEDGER, and fleet views read a window.

The claims: a run whose commits reached its plan is `done` whatever the
journal's tail says (a crashed container loses its detach events); an attach
after a detach reopens the story (resubmission is resume, not a second
death); the hosts view counts each run once, by its last word; and the
fleet-page series window bounds series, never identity facts.
"""

from __future__ import annotations

import tempfile
import unittest

from common import arith_plan_blobs
from rlstack import LocalStore
from rlstack.observe.host_series import lanes_overlapping, windowed
from rlstack.observe.views import hosts_data, runs_data


def store_with(journal: list[dict], committed: int = 0) -> LocalStore:
    """A store holding one journaled run ("r1" on host "h") and, when asked,
    a ledger with `committed` updates against the 4-wave arith train plan."""
    tmp = tempfile.mkdtemp()
    store = LocalStore(tmp)
    for event in journal:
        store.append_host_event("h", event)
    if committed:
        run = store.open_run("r1", manifest={"run_id": "r1"})
        run.write_plan("train", arith_plan_blobs()["train"])
        for update in range(1, committed + 1):
            run.append_ledger({"update": update})
    return store


class StatusTruthTest(unittest.TestCase):
    def test_the_ledger_outranks_a_lost_detach(self) -> None:
        """attach, detach(failed), attach, container dies — but the ledger
        reached the plan: the run is DONE, on both views."""
        store = store_with([
            {"event": "attach", "t": 1.0, "run_id": "r1"},
            {"event": "detach", "t": 2.0, "run_id": "r1", "status": "failed"},
            {"event": "attach", "t": 3.0, "run_id": "r1"},
        ], committed=4)
        (row,) = runs_data([store])
        self.assertEqual(row["status"], "done")
        self.assertEqual((row["committed"], row["target"]), (4, 4))
        (host,) = hosts_data([store])
        self.assertEqual((host["done"], host["failed"], host["running"]),
                         (1, 0, []))

    def test_a_reattach_reopens_the_story(self) -> None:
        """The same journal with an unfinished ledger: the second attach means
        RUNNING — the first attempt's death is not the run's status."""
        store = store_with([
            {"event": "attach", "t": 1.0, "run_id": "r1"},
            {"event": "detach", "t": 2.0, "run_id": "r1", "status": "failed"},
            {"event": "attach", "t": 3.0, "run_id": "r1"},
        ], committed=2)
        (row,) = runs_data([store])
        self.assertEqual(row["status"], "running")
        (host,) = hosts_data([store])
        self.assertEqual(host["running"], ["r1"])

    def test_a_true_failure_still_reads_failed(self) -> None:
        store = store_with([
            {"event": "attach", "t": 1.0, "run_id": "r1"},
            {"event": "detach", "t": 2.0, "run_id": "r1", "status": "failed"},
        ], committed=2)
        (row,) = runs_data([store])
        self.assertEqual(row["status"], "failed")

    def test_a_dead_hosts_journal_cannot_shadow_a_live_reattach(self) -> None:
        """The run hopped hosts: died on one venue, resumed on another. The
        NEWEST event speaks for the run whichever host journal holds it —
        journals are walked per host, so without the time gate the dead
        host's detach(failed), iterated after the live host's re-attach,
        would call the healthy second attempt by the first attempt's death
        (observed live: six running arms shown failed by a dead venue)."""
        for old_host, live_host in (("z-dead-venue", "a-live-venue"),
                                    ("a-dead-venue", "z-live-venue")):
            with self.subTest(order=(old_host, live_host)):
                tmp = tempfile.mkdtemp()
                store = LocalStore(tmp)
                store.append_host_event(old_host, {
                    "event": "attach", "t": 1.0, "run_id": "r1"})
                store.append_host_event(old_host, {
                    "event": "detach", "t": 2.0, "run_id": "r1",
                    "status": "failed"})
                store.append_host_event(live_host, {
                    "event": "attach", "t": 3.0, "run_id": "r1"})
                run = store.open_run("r1", manifest={"run_id": "r1"})
                run.write_plan("train", arith_plan_blobs()["train"])
                run.append_ledger({"update": 1})
                (row,) = runs_data([store])
                self.assertEqual(row["status"], "running")
                self.assertEqual(sorted(row["hosts"]),
                                 sorted([old_host, live_host]))


class WindowTest(unittest.TestCase):
    def test_windowed_bounds_series_and_lanes_keep_open_residencies(self) -> None:
        events = [{"t": 10.0, "event": "stats"}, {"t": 99.0, "event": "stats"}]
        self.assertEqual(len(windowed(events, 50.0)), 1)
        self.assertEqual(len(windowed(events, None)), 2)
        lanes = [{"attached": 1.0, "detached": 2.0},     # closed before: hidden
                 {"attached": 1.0, "detached": 60.0},    # closed inside: shown
                 {"attached": 1.0, "detached": None}]    # still open: shown
        self.assertEqual(len(lanes_overlapping(lanes, 50.0)), 2)
        self.assertEqual(len(lanes_overlapping(lanes, None)), 3)


if __name__ == "__main__":
    unittest.main()
