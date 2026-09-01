"""The pure placement vocabulary: demands, units, sizing, the join rule.

These are the fleet plane's pure functions — no desk, no metal, no store.
The claims: a spec's gpu_config reads as capability demands with the anchor
on the learner (the learner is never remote, said ONCE, in campaign.py); a
sleep group is one placement unit while concurrent members split; a GB hint
becomes a fraction of the one device it lands on, and more than a device
holds is the acquire rung; and coverage is capability equality — the ONE
join rule, desk and metal alike.
"""

from __future__ import annotations

import unittest

from common import arith_spec, arith_store
from rlstack import (
    Demand, FleetError, GpuConfig, GpuGroup, Metal, Regime, demands_of,
    fraction_for_gb, gpus, learner, pool,
)
from rlstack.runner.desk import covers, placement_units

import tempfile

BASE = "Qwen/Qwen3-0.6B"


def spec_with(groups):
    tmp = tempfile.TemporaryDirectory()
    store, train, _ = arith_store(tmp.name)
    made = arith_spec(train, gpu_config=GpuConfig(groups=groups))
    tmp.cleanup()
    return made


class DemandsTest(unittest.TestCase):
    def test_demands_read_capability_off_the_spec(self) -> None:
        demands = demands_of(spec_with((
            GpuGroup(gpus(n=1), (pool("main"),)),
            GpuGroup(gpus(n=1), (learner(),)))))
        self.assertEqual([d.capability for d in demands],
                         ["inference", "training"])
        self.assertEqual([d.pool for d in demands], ["main", None])

    def test_the_anchor_is_the_learner_and_only_the_learner(self) -> None:
        demands = demands_of(spec_with((
            GpuGroup(gpus(n=1), (pool("main"),)),
            GpuGroup(gpus(n=1), (learner(),)))))
        self.assertEqual([d.anchor for d in demands], [False, True])

    def test_a_sleep_group_is_one_placement_unit(self) -> None:
        demands = demands_of(spec_with((
            GpuGroup(gpus(n=1), (pool("main"), learner()), sharing="sleep"),)))
        units = placement_units(demands)
        self.assertEqual(len(units), 1)
        self.assertEqual(len(units[0]), 2)

    def test_concurrent_members_place_one_by_one(self) -> None:
        demands = demands_of(spec_with((
            GpuGroup(gpus(n=1), (pool("main"),)),
            GpuGroup(gpus(n=1), (learner(),)))))
        self.assertEqual(len(placement_units(demands)), 2)

    def test_one_concurrent_group_is_one_unit(self) -> None:
        """A GpuGroup co-locates whatever the sharing: pool and learner in
        ONE concurrent group land on ONE host (the stress-matrix shape) so
        the tenancy's pool is local — no wire, no self-dial."""
        demands = demands_of(spec_with((
            GpuGroup(gpus(n=1), (pool("main"), learner())),)))
        units = placement_units(demands)
        self.assertEqual(len(units), 1)
        self.assertEqual(len(units[0]), 2)


class SizingTest(unittest.TestCase):
    def test_a_gb_hint_becomes_a_fraction_of_one_device(self) -> None:
        self.assertEqual(fraction_for_gb(12.0, Metal("node-a", "L4", 4)), 0.5)
        self.assertAlmostEqual(
            fraction_for_gb(20.0, Metal("node-h", "H100", 8, vram_gb=80.0)),
            0.25)

    def test_more_than_one_device_holds_is_the_acquire_rung(self) -> None:
        with self.assertRaises(FleetError):
            fraction_for_gb(25.0, Metal("node-a", "L4", 4))


class CoversTest(unittest.TestCase):
    REGIMES = (Regime("main-tp2", "inference", BASE, 2),)

    def demand(self, **overrides) -> Demand:
        base = dict(pool="main", capability="inference", base=BASE, shape=2,
                    memory=0.5, group=0, sharing="concurrent")
        base.update(overrides)
        return Demand(**base)

    def test_coverage_is_capability_equality(self) -> None:
        self.assertTrue(covers(self.REGIMES, self.demand()))
        self.assertFalse(covers(self.REGIMES, self.demand(shape=1)))
        self.assertFalse(covers(self.REGIMES, self.demand(base="other/model")))
        self.assertFalse(covers(self.REGIMES,
                                self.demand(capability="training")))


if __name__ == "__main__":
    unittest.main()
