"""The mean-baseline estimator as a postprocessor: subtract each group's mean
reward, nothing else. The group IS the baseline scope; ppo's pairing — same
scope as grpo_advantage, but no variance rescaling."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor


@postprocessor("center_reward")
class CenterReward(PostProcessor):
    consumes = ("reward",)
    produces = ("advantage",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        rewards = data["reward"]
        mean = math.fsum(rewards) / len(rewards)
        return {"advantage": [r - mean for r in rewards]}
