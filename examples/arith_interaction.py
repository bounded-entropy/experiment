"""The fake arithmetic demo owns its interaction and reward."""

import re

from rlstack.data.trajectory import Message, Role
from rlstack.inference.rollout import Rollout
from rlstack.training.post.base import PostProcessor, postprocessor


async def answer(client, task):
    prompt = Message(Role.USER, task.prompt)
    turn = await client.sample([prompt])
    return Rollout(task=task, messages=[prompt, turn.message], turns=[turn])


@postprocessor("demo_arithmetic_reward")
class ArithmeticReward(PostProcessor):
    produces = ("reward",)

    async def process(self, group, data, client):
        rewards = []
        for trajectory in group.trajectories:
            numbers = re.findall(r"-?\d+", trajectory.turns[-1].message.content)
            rewards.append(float(bool(numbers) and
                                 numbers[-1] == str(trajectory.task.meta["answer"])))
        return {"reward": rewards}
