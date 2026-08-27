"""The postprocessing pipeline (rlstack.training.post, rlstack.runner.post)."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from common import sealed
from rlstack import (
    POST, Bundle, FakeEngine, Group, PostProcessor, SamplingSpec, Wave,
    postprocessor, run_pipeline, zscore,
)

BUNDLE = Bundle(bundle_id="bundle:post0000", policy_version={"pi": 0})


def pools():
    engine = FakeEngine()
    engine.add_bundle(BUNDLE)
    return {"main": (engine, BUNDLE)}


def go(coro):
    return asyncio.run(coro)


@postprocessor("post_bad_columns")
class _BadColumns(PostProcessor):
    produces = ("promised",)

    async def process(self, group: Any, data: Any, llm: Any):
        return {"something_else": [0.0] * len(group)}


@postprocessor("post_bad_length")
class _BadLength(PostProcessor):
    produces = ("short",)

    async def process(self, group: Any, data: Any, llm: Any):
        return {"short": [0.0]}  # wrong length for any group > 1


@postprocessor("post_sampling_judge")
class _SamplingJudge(PostProcessor):
    """A judge: a postprocessor that SAMPLES (through any pool it likes)."""

    produces = ("judge",)

    async def process(self, group: Any, data: Any, llm: Any):
        scores = []
        for traj in group.trajectories:
            turn = await llm.sample(traj.messages)   # llm.pool(name) also works
            scores.append(float(len(turn.message.content)))
        return {"judge": scores}


class ZscoreTest(unittest.TestCase):
    def test_population_std(self) -> None:
        self.assertEqual(zscore([0.0, 2.0]), [-1.0, 1.0])

    def test_all_equal_is_zeros(self) -> None:
        self.assertEqual(zscore([3.0, 3.0, 3.0]), [0.0, 0.0, 0.0])

    def test_singleton_is_zero(self) -> None:
        self.assertEqual(zscore([5.0]), [0.0])

    def test_known_three_point_case(self) -> None:
        out = zscore([1.0, 2.0, 3.0])
        expected = 1.0 / (2.0 / 3.0) ** 0.5  # population std = sqrt(2/3)
        self.assertAlmostEqual(out[0], -expected, places=12)
        self.assertAlmostEqual(out[2], expected, places=12)


class PipelineTest(unittest.TestCase):
    def wave(self) -> Wave:
        # group a: one right ("4"), one wrong answer; group b: both wrong
        return Wave([
            Group("a", [sealed("a", content="4"), sealed("a", content="7")]),
            Group("b", [sealed("b", content="9"), sealed("b", content="8")]),
        ])

    def run_pipe(self, pipeline: tuple[str, ...]):
        return go(run_pipeline(pipeline, self.wave(), pools(), SamplingSpec(),
                               master=17, update=1))

    def test_verifier_then_advantage_chains_per_group(self) -> None:
        columns = self.run_pipe(("verifier", "grpo_advantage"))
        self.assertEqual(columns["reward"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(columns["advantage"], [1.0, -1.0, 0.0, 0.0])

    def test_columns_align_to_wave_order(self) -> None:
        columns = self.run_pipe(("verifier",))
        self.assertEqual(len(columns["reward"]), 4)

    def test_deterministic(self) -> None:
        self.assertEqual(self.run_pipe(("verifier", "grpo_advantage")),
                         self.run_pipe(("verifier", "grpo_advantage")))

    def test_a_judge_samples_deterministically(self) -> None:
        first = self.run_pipe(("post_sampling_judge",))
        second = self.run_pipe(("post_sampling_judge",))
        self.assertEqual(first, second)
        self.assertEqual(len(first["judge"]), 4)

    def test_output_columns_must_match_produces(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.run_pipe(("post_bad_columns",))
        self.assertIn("produces", str(caught.exception))

    def test_output_length_must_match_group(self) -> None:
        with self.assertRaises(ValueError):
            self.run_pipe(("post_bad_length",))


class RegistrationTest(unittest.TestCase):
    def test_builtins_declare_their_wiring(self) -> None:
        verifier = POST.get("verifier")
        self.assertEqual(verifier.produces, ("reward",))
        self.assertEqual(verifier.consumes, ())
        grpo = POST.get("grpo_advantage")
        self.assertEqual(grpo.consumes, ("reward",))
        self.assertEqual(grpo.produces, ("advantage",))


if __name__ == "__main__":
    unittest.main()
