"""The reward as a postprocessor, checked rather than sampled: the last number
in each trajectory's final turn against the task's known answer. No pool
traffic, so it costs nothing and consumes no randomness."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor

_NUMBER = re.compile(r"-?\d+")


@postprocessor("verifier")
class Verifier(PostProcessor):
    produces = ("reward",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        rewards = []
        for traj in group.trajectories:
            numbers = _NUMBER.findall(traj.turns[-1].message.content)
            correct = bool(numbers) and numbers[-1] == str(traj.task.meta["answer"])
            rewards.append(float(correct))
        return {"reward": rewards}
