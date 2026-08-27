"""Built-in advantages (rlstack.training.advantages)."""

from __future__ import annotations

import unittest

from common import sealed
from rlstack import ADVANTAGES, Group, Wave, grpo_group_norm, zscore


class ZscoreTest(unittest.TestCase):
    def test_population_std(self) -> None:
        self.assertEqual(zscore([0.0, 2.0]), [-1.0, 1.0])  # population, not sample

    def test_all_equal_is_zeros(self) -> None:
        self.assertEqual(zscore([3.0, 3.0, 3.0]), [0.0, 0.0, 0.0])

    def test_singleton_is_zero(self) -> None:
        self.assertEqual(zscore([5.0]), [0.0])

    def test_known_three_point_case(self) -> None:
        out = zscore([1.0, 2.0, 3.0])
        expected = 1.0 / (2.0 / 3.0) ** 0.5  # population std = sqrt(2/3)
        self.assertAlmostEqual(out[0], -expected, places=12)
        self.assertAlmostEqual(out[1], 0.0, places=12)
        self.assertAlmostEqual(out[2], expected, places=12)


class GrpoGroupNormTest(unittest.TestCase):
    """The registered builtin: zscore within each group, concatenated in wave order."""

    def test_groups_are_independent(self) -> None:
        wave = Wave([
            Group("t0", [sealed("t0", {"reward": 1.0}), sealed("t0", {"reward": 0.0})]),
            Group("t1", [sealed("t1", {"reward": 3.0}), sealed("t1", {"reward": 3.0})]),
        ])
        self.assertEqual(grpo_group_norm(wave), [1.0, -1.0, 0.0, 0.0])

    def test_alignment_is_wave_trajectory_order(self) -> None:
        wave = Wave([
            Group("a", [sealed("a", {"reward": 0.0}), sealed("a", {"reward": 2.0})]),
            Group("b", [sealed("b", {"reward": 14.0}), sealed("b", {"reward": 10.0})]),
        ])
        adv = grpo_group_norm(wave)
        self.assertEqual(len(adv), len(wave))
        self.assertEqual(adv, [-1.0, 1.0, 1.0, -1.0])

    def test_is_the_registered_compute_half(self) -> None:
        adef = ADVANTAGES.get("grpo_group_norm")
        self.assertIs(adef.fn, grpo_group_norm)
        self.assertEqual(adef.consumes, ("reward",))


if __name__ == "__main__":
    unittest.main()
