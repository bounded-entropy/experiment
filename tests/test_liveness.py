"""Liveness, stalling, and the overlay's query grammar.

The claims: a heartbeat presumes down only after several of the host's OWN
cadences (and presumes nothing from a too-thin journal); a desk probe
overrides the presumption and names itself as the source; STALLED is a
running run on a pulse-less host; the query grammar reads regex terms as
AND within a clause and "|" as OR across clauses; and the overlay serves
one series per matching run off the ledgers alone.
"""

from __future__ import annotations

import tempfile
import unittest

from common import arith_plan_blobs
from rlstack import LocalStore
from rlstack.observe.liveness import (
    heartbeat, liveness_by_host, stall_runs,
)
from rlstack.observe.select import match_expr, metric_names, overlay


def beats(times: list[float]) -> list[dict]:
    return [{"event": "stats", "t": t} for t in times]


class HeartbeatTest(unittest.TestCase):
    def test_quiet_past_the_cadence_is_presumed_down(self) -> None:
        steady = beats([float(t) for t in range(0, 300, 30)])   # 30s cadence
        alive = heartbeat(steady, now=330.0)
        self.assertTrue(alive["live"])
        dead = heartbeat(steady, now=270.0 + 30 * 40)
        self.assertFalse(dead["live"])
        self.assertAlmostEqual(dead["cadence"], 30.0)

    def test_a_thin_journal_presumes_nothing(self) -> None:
        self.assertIsNone(heartbeat(beats([5.0]), now=1e9)["live"])
        self.assertIsNone(heartbeat([], now=1e9)["live"])

    def test_the_desk_probe_outranks_the_presumption(self) -> None:
        quiet = beats([0.0, 30.0])
        rows = liveness_by_host([("h", quiet)], now=1e6,
                                desk=lambda: {"h": True})
        self.assertTrue(rows["h"]["live"])
        self.assertEqual(rows["h"]["source"], "desk")
        rows = liveness_by_host([("h", quiet)], now=1e6, desk=None)
        self.assertFalse(rows["h"]["live"])
        self.assertEqual(rows["h"]["source"], "heartbeat")

    def test_a_dead_desk_is_a_journal_only_day(self) -> None:
        def broken():
            raise ConnectionError("desk gone")
        rows = liveness_by_host([("h", beats([0.0, 30.0]))], now=1e6,
                                desk=broken)
        self.assertEqual(rows["h"]["source"], "heartbeat")


class StallTest(unittest.TestCase):
    def test_running_on_a_dead_host_is_stalled(self) -> None:
        rows = [{"status": "running", "hosts": ["a", "b"]},
                {"status": "running", "hosts": ["b"]},
                {"status": "done", "hosts": ["a"]}]
        pulses = {"a": {"live": False}, "b": {"live": True}}
        stall_runs(rows, pulses)
        self.assertEqual([r["status"] for r in rows],
                         ["stalled", "running", "done"])
        self.assertEqual(rows[0]["stalled_hosts"], ["a"])

    def test_an_unknown_pulse_stalls_nothing(self) -> None:
        rows = [{"status": "running", "hosts": ["a"]}]
        stall_runs(rows, {"a": {"live": None}})
        self.assertEqual(rows[0]["status"], "running")


class GrammarTest(unittest.TestCase):
    ROW = {"run_id": "abc123", "name": "plora-k4-p0.3-l32-s12",
           "note": "sweep", "tags": ["plora", "k=4", "prior=0.3", "seed=12"]}

    def test_and_or_and_regex(self) -> None:
        self.assertTrue(match_expr(self.ROW, "plora k=4"))
        self.assertFalse(match_expr(self.ROW, "plora k=8"))
        self.assertTrue(match_expr(self.ROW, "k=8 | prior=0.3"))
        self.assertTrue(match_expr(self.ROW, r"plora-k(4|8)-p0\.3"))
        self.assertTrue(match_expr(self.ROW, ""))
        # an uncompilable term falls back to substring, not to an error
        self.assertFalse(match_expr(self.ROW, "((("))


class OverlayTest(unittest.TestCase):
    def test_one_series_per_matching_run_off_the_ledgers(self) -> None:
        store = LocalStore(tempfile.mkdtemp())
        for rid, rewards in (("r1", [0.1, 0.5]), ("r2", [0.9])):
            run = store.open_run(rid, manifest={"run_id": rid})
            run.write_plan("train", arith_plan_blobs(False)["train"])
            for update, reward in enumerate(rewards, start=1):
                run.append_ledger({"update": update,
                                   "train": {"loss": 0.0},
                                   "post": {"reward": reward}})
            store.append_host_event("h", {"event": "attach", "t": 1.0,
                                          "run_id": rid})
        store.annotate_run("r1", name="alpha", tags=["good"])
        data = overlay([store], "reward", "good")
        self.assertEqual([s["run_id"] for s in data["series"]], ["r1"])
        self.assertEqual(data["series"][0]["points"], [[1, 0.1], [2, 0.5]])
        both = overlay([store], "reward", "")
        self.assertEqual(len(both["series"]), 2)
        self.assertIn("reward", metric_names([store]))


if __name__ == "__main__":
    unittest.main()
