"""A store ref to a run that is still running says "not yet" (refs.py).

Found on the venue: ADR 0005's student arms replay a teacher run's rollouts
by `store://<teacher>/rollouts/<r>#<i>`, and the reader refused any rollout
not yet sealed as a plan error — so an arm could only start once the whole
teacher set existed, on the same metal, in series. A consumer now paces on
its source the way a Trainer paces on its own Generator: None while the
source run exists and is unfinished, a plan error by name when the run is
missing or reached its extent without that row.
"""

from __future__ import annotations

import tempfile
import unittest

from rlstack import GroupPlan, LocalStore, RunPlan, Sample, WavePlan, encode
from rlstack.data.plan import PlanError
from rlstack.runner.refs import RefReader

SOURCE, CONSUMER = "teacher00", "student00"


def a_row(index: int) -> dict:
    return {"group": f"g{index}", "turns": []}


class StoreRefPacingTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        wave = WavePlan((GroupPlan("t", (Sample("t", "single_turn"),)),))
        self.source = self.store.open_run(SOURCE, {"run_id": SOURCE, "spec": "{}"})
        self.source.write_plan("rollout", encode(RunPlan((wave, wave, wave))))
        self.source.write_rollout(1, [a_row(1)])
        consumer = self.store.open_run(CONSUMER, {"run_id": CONSUMER, "spec": "{}"})
        self.reader = RefReader(self.store, consumer)

    def test_a_sealed_rollout_reads(self) -> None:
        self.assertEqual(self.reader.row(f"store://{SOURCE}/rollouts/1#0"),
                         a_row(1))

    def test_an_unsealed_rollout_of_a_running_source_is_not_yet(self) -> None:
        self.assertIsNone(self.reader.rows(f"store://{SOURCE}/rollouts/2"))
        self.assertIsNone(self.reader.row(f"store://{SOURCE}/rollouts/2#0"))
        # and it arrives: the same reader, asked again, reads it
        self.source.write_rollout(2, [a_row(2)])
        self.assertEqual(self.reader.row(f"store://{SOURCE}/rollouts/2#0"),
                         a_row(2))

    def test_a_missing_run_is_a_plan_error_by_name(self) -> None:
        with self.assertRaisesRegex(PlanError, "no run 'nobody' exists"):
            self.reader.rows("store://nobody/rollouts/1")

    def test_a_finished_source_without_the_row_never_will_have_it(self) -> None:
        self.source.write_rollout(2, [a_row(2)])
        self.source.write_rollout(3, [a_row(3)])        # the extent: 3 of 3
        with self.assertRaisesRegex(PlanError, "reached its extent"):
            self.reader.rows(f"store://{SOURCE}/rollouts/4")
