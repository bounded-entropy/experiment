"""Reward saturation and code diversity describe different properties."""

import asyncio
import dataclasses
import unittest

from common import sealed
from rlstack import Group, Task
from rlstack.training.post.program_diagnostics import ProgramDiagnostics


def group_of(*programs):
    task = Task("p", "write a function", {"domain": "mbpp"})
    trajectories = [dataclasses.replace(sealed("p", content=program), task=task)
                    for program in programs]
    return Group("p", trajectories)


class ProgramDiagnosticsTest(unittest.TestCase):
    def test_solved_group_can_have_only_one_distinct_program(self):
        group = group_of("def f(x):\n return x", "def f(x):\n # comment\n return x")
        columns = asyncio.run(ProgramDiagnostics().process(group, {"reward": [1.0, 1.0]}, None))
        self.assertEqual(columns["all_pass_group"], [1.0, 1.0])
        self.assertEqual(columns["mixed_reward_group"], [0.0, 0.0])
        self.assertEqual(columns["distinct_program_fraction"], [0.5, 0.5])

    def test_diverse_failures_are_not_counted_as_success_diversity(self):
        group = group_of("def f(x):\n return x", "def f(x):\n return 2*x")
        columns = asyncio.run(ProgramDiagnostics().process(group, {"reward": [0.0, 0.0]}, None))
        self.assertEqual(columns["all_fail_group"], [1.0, 1.0])
        self.assertEqual(columns["distinct_program_fraction"], [1.0, 1.0])
        self.assertEqual(columns["distinct_correct_fraction"], [0.0, 0.0])

    def test_mixed_rewards_preserve_a_group_relative_signal(self):
        columns = asyncio.run(ProgramDiagnostics().process(
            group_of("x=1", "x=2"), {"reward": [1.0, 0.0]}, None))
        self.assertEqual(columns["mixed_reward_group"], [1.0, 1.0])
        self.assertEqual(columns["distinct_correct_fraction"], [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
