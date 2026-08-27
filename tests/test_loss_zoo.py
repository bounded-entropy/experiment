"""The loss zoo validates: every registered loss, paired with the pipeline
that produces what it requires, passes Phase 0 — and a broken pairing fails
with unsatisfied-requires. Numerics run on real metal (deploy/stress_l4.py);
this layer proves the DECLARATIONS wire up without torch."""

from __future__ import annotations

import tempfile
import unittest

from rlstack import (
    AlgoSpec, OptimSpec, Schedule, TrajectorySource, fake_qwen_schema, validate,
)
from tests.common import arith_spec, arith_store

# loss -> the AlgoSpec.post pipeline it pairs with in the stress matrix
ZOO = {
    "grpo": ("verifier", "grpo_advantage"),
    "ppo": ("verifier", "center_reward"),
    "gspo": ("verifier", "grpo_advantage"),
    "sdft": ("verifier",),
    "self_anchor": ("verifier",),
    "sft": (),
    "opd": (),
}

OFFLINE = {"sft", "opd"}   # consume sealed store data; no gen, no Generator


def algo(loss: str, post: tuple[str, ...], lag: int = 0) -> AlgoSpec:
    return AlgoSpec(loss=loss, post=post, optim=OptimSpec("adamw", lr=1e-5),
                    schedule=Schedule(group_size=2, trajectories_per_wave=4,
                                      n_updates=4, microbatch_tokens=64,
                                      max_policy_lag=lag))


class LossZooTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store, self.train, _ = arith_store(self.tmp.name)
        self.schema = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")

    def spec_for(self, loss: str):
        overrides = {"algo": algo(loss, ZOO[loss],
                                  lag=2 if loss == "self_anchor" else 0)}
        if loss in OFFLINE:
            overrides["gen"] = None
            overrides["trajectories"] = TrajectorySource("store://parent/waves")
        return arith_spec(self.train, **overrides)

    def test_every_zoo_pairing_validates(self) -> None:
        for loss, post in ZOO.items():
            with self.subTest(loss=loss):
                issues = validate(self.spec_for(loss), self.schema)
                self.assertEqual(issues, [], f"{loss} with post={post}")

    def test_requires_still_bites_without_its_pipeline(self) -> None:
        """sdft names "reward"; an empty pipeline must fail Phase 0."""
        spec = arith_spec(self.train, algo=algo("sdft", ()))
        codes = {issue.code for issue in validate(spec, self.schema)}
        self.assertIn("unsatisfied-requires", codes)

    def test_advantage_owners_collide_loudly(self) -> None:
        """grpo_advantage and center_reward both produce "advantage" — one
        pipeline may carry only one owner per column."""
        spec = arith_spec(self.train, algo=algo(
            "ppo", ("verifier", "grpo_advantage", "center_reward")))
        codes = {issue.code for issue in validate(spec, self.schema)}
        self.assertIn("post-collision", codes)


if __name__ == "__main__":
    unittest.main()
