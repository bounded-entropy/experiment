"""The seed tree (rlstack.runner.seeds)."""

from __future__ import annotations

import unittest

from rlstack import derive


class DeriveTest(unittest.TestCase):
    def test_deterministic(self) -> None:
        self.assertEqual(derive(17, "rollout", 3, "task-9", 0),
                         derive(17, "rollout", 3, "task-9", 0))

    def test_every_path_component_matters(self) -> None:
        base = derive(17, "rollout", 3, "task-9", 0)
        for other in (derive(18, "rollout", 3, "task-9", 0),
                      derive(17, "eval", 3, "task-9", 0),
                      derive(17, "rollout", 4, "task-9", 0),
                      derive(17, "rollout", 3, "task-8", 0),
                      derive(17, "rollout", 3, "task-9", 1)):
            self.assertNotEqual(base, other)

    def test_str_and_int_components_do_not_collide(self) -> None:
        self.assertNotEqual(derive(0, 1), derive(0, "1"))

    def test_63_bit_non_negative(self) -> None:
        for seed in (derive(0), derive(2 ** 40, "x"), derive(-5, "y")):
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 2 ** 63)


if __name__ == "__main__":
    unittest.main()
