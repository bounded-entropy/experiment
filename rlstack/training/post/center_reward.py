"""The mean-baseline estimator as a postprocessor: subtract each group's mean
reward, nothing else. The value-free PPO pairing — same baseline scope as
grpo_advantage but no variance rescaling."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rlstack.client import SampleClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor


@postprocessor("center_reward")
class CenterReward(PostProcessor):
    consumes = ("reward",)
    produces = ("advantage",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      llm: SampleClient) -> Mapping[str, Sequence[float]]:
        rewards = data["reward"]
        mean = math.fsum(rewards) / len(rewards)
        return {"advantage": [r - mean for r in rewards]}
