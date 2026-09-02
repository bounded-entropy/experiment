"""The pure placement vocabulary: demands, units, sizing, the join rule.

These are the fleet plane's pure functions — no desk, no metal, no store.
The claims: a spec's gpu_config reads as capability demands with the anchor
on the learner (the learner is never remote, said ONCE, in campaign.py) and
its GB passed through as declared (ADR 0001); ONE HostSpec is ONE placement
unit; a unit's per-device GB is its largest member's, a whole device sized
per metal; a GB figure becomes a fraction of the one device it lands on, and
more than a device holds is the acquire rung; and coverage is capability
equality — the ONE join rule, desk and metal alike.
"""

from __future__ import annotations

import unittest

from common import arith_spec, arith_store
from rlstack import (
    Demand, DeskError, GpuConfig, HostSpec, Metal, Regime, demands_of,
    fraction_for_gb, learner, pool,
)
from rlstack.runner.desk import covers, placement_units, unit_gb

import tempfile

BASE = "Qwen/Qwen3-0.6B"


def spec_with(hosts):
    tmp = tempfile.TemporaryDirectory()
    store, train, _ = arith_store(tmp.name)
    made = arith_spec(train, gpu_config=GpuConfig(hosts=hosts))
    tmp.cleanup()
    return made


class DemandsTest(unittest.TestCase):
    def test_demands_read_capability_off_the_spec(self) -> None:
        demands = demands_of(spec_with((
            HostSpec((pool("main"),)), HostSpec((learner(),)))))
        self.assertEqual([d.capability for d in demands],
                         ["inference", "training"])
        self.assertEqual([d.pool for d in demands], ["main", None])

    def test_the_anchor_is_the_learner_and_only_the_learner(self) -> None:
        demands = demands_of(spec_with((
            HostSpec((pool("main"),)), HostSpec((learner(),)))))
        self.assertEqual([d.anchor for d in demands], [False, True])

    def test_vram_gb_passes_through_as_declared(self) -> None:
        """GB in, GB out — total across shards, None for a whole device per
        shard: the campaign layer converts nothing, because the crossing to a
        fraction belongs to the metal that knows its card (ADR 0001)."""
        demands = demands_of(spec_with((
            HostSpec((pool("main", tp=2, vram_gb=30),)),
            HostSpec((learner(),)))))
        self.assertEqual([d.vram_gb for d in demands], [30, None])
        self.assertEqual([d.per_device_gb() for d in demands], [15.0, None])

    def test_a_multi_member_host_is_one_placement_unit(self) -> None:
        demands = demands_of(spec_with((
            HostSpec((pool("main"), learner())),)))
        units = placement_units(demands)
        self.assertEqual(len(units), 1)
        self.assertEqual(len(units[0]), 2)

    def test_separate_hosts_place_one_by_one(self) -> None:
        demands = demands_of(spec_with((
            HostSpec((pool("main"),)), HostSpec((learner(),)))))
        self.assertEqual(len(placement_units(demands)), 2)


class SizingTest(unittest.TestCase):
    def test_a_gb_figure_becomes_a_fraction_of_one_device(self) -> None:
        self.assertEqual(fraction_for_gb(12.0, Metal("node-a", "L4", 4)), 0.5)
        self.assertAlmostEqual(
            fraction_for_gb(20.0, Metal("node-h", "H100", 8, vram_gb=80.0)),
            0.25)

    def test_more_than_one_device_holds_is_the_acquire_rung(self) -> None:
        with self.assertRaises(DeskError) as caught:
            fraction_for_gb(25.0, Metal("node-a", "L4", 4))
        self.assertIn("acquire rung", str(caught.exception))

    def test_a_unit_is_sized_by_its_largest_member_per_metal(self) -> None:
        """Members of one HostSpec alternate, so the partition holds the
        LARGEST per-device need; a whole-device member (None) is the whole
        of whichever card the unit is being sized against."""
        l4, h100 = Metal("a", "L4", 1, 24.0), Metal("h", "H100", 1, 80.0)
        sized = demands_of(spec_with((
            HostSpec((pool("main", tp=2, vram_gb=30), learner(vram_gb=20))),)))
        self.assertEqual(unit_gb(sized, l4), 20.0)      # max(15, 20)
        whole = demands_of(spec_with((
            HostSpec((pool("main", vram_gb=10), learner())),)))
        self.assertEqual(unit_gb(whole, l4), 24.0)
        self.assertEqual(unit_gb(whole, h100), 80.0)


class CoversTest(unittest.TestCase):
    REGIMES = (Regime("main-tp2", "inference", BASE, 2),)

    def demand(self, **overrides) -> Demand:
        base = dict(pool="main", capability="inference", base=BASE, shape=2,
                    vram_gb=30.0, group=0)
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
