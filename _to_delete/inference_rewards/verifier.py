"""Pure check, no sampling: the last number in the final turn against the
task's known answer."""

from __future__ import annotations

import re

from rlstack.inference.environments.base import SampleClient
from rlstack.inference.rewards.base import Reward, reward
from rlstack.inference.rollout import Rollout

_NUMBER = re.compile(r"-?\d+")


@reward("verifier")
class Verifier(Reward):
    components = ("reward",)

    async def score(self, rollout: Rollout, llm: SampleClient) -> tuple[float, ...]:
        numbers = _NUMBER.findall(rollout.turns[-1].message.content)
        correct = bool(numbers) and numbers[-1] == str(rollout.task.meta["answer"])
        return (float(correct),)
