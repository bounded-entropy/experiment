"""Test-owned interactions for framework service and recovery qualification."""

import re

from rlstack.client import PoolClient
from rlstack.data.trajectory import Message, Role, Task, Trajectory
from rlstack.inference.rollout import Rollout
from rlstack.registry import loss
from rlstack.spec.specs import SamplingSpec
from rlstack.training.post.base import PostProcessor, postprocessor


async def single_turn(client: PoolClient, task: Task) -> Rollout:
    prompt = Message(Role.USER, task.prompt)
    turn = await client.sample([prompt])
    return Rollout(task=task, messages=[prompt, turn.message], turns=[turn])


async def continuation(client: PoolClient, source: Trajectory) -> Rollout:
    task = Task(source.task.id + "/next", source.task.prompt,
                dict(source.task.meta, previous=source.text))
    return await single_turn(client, task)


async def tool_use(client: PoolClient, task: Task) -> Rollout:
    """The examples only need a named interaction to demonstrate declaration."""
    return await single_turn(client, task)



@postprocessor("fixture_reward")
class ArithmeticReward(PostProcessor):
    """A test's reward oracle; no research scorer is imported by the framework."""

    produces = ("reward",)

    async def process(self, group, data, client):
        rewards = []
        for trajectory in group.trajectories:
            numbers = re.findall(r"-?\d+", trajectory.turns[-1].message.content)
            correct = bool(numbers) and numbers[-1] == str(trajectory.task.meta["answer"])
            rewards.append(float(correct))
        return {"reward": rewards}



@postprocessor("fixture_judge")
class SampledAgreement(PostProcessor):
    """Exercise a second pool through the same recorded sampling services."""

    produces = ("reward",)
    pools = ("judge",)
    sampling = SamplingSpec(temperature=0.0, top_p=1.0, max_tokens=16)

    async def process(self, group, data, client):
        judge = client.pool("judge")
        rewards = []
        for trajectory in group.trajectories:
            sampled = await judge.sample((Message(Role.USER, trajectory.task.prompt),))
            rewards.append(float(sampled.message.content == trajectory.turns[-1].message.content))
        return {"reward": rewards}


@loss("fixture_reward_loss", requires=("reward",))
def reward_input(out, batch):
    """Declaration fixture for direct flow-graph dependencies, never optimized."""
    raise NotImplementedError("declaration-only test fixture")
