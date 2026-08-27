"""A flat scalar — the degenerate vector (useful as a baseline/stub)."""

from __future__ import annotations

from rlstack.inference.environments.base import SampleClient
from rlstack.inference.rewards.base import Reward, reward
from rlstack.inference.rollout import Rollout


@reward("constant")
class Constant(Reward):
    components = ("reward",)

    async def score(self, rollout: Rollout, llm: SampleClient) -> tuple[float, ...]:
        return (1.0,)
