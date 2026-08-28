"""One sample call, one turn, done: the smallest complete environment."""

from __future__ import annotations

from rlstack.data.trajectory import Message, Role, Task
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.rollout import Rollout


@environment("math_single_turn")
class MathSingleTurn(Environment):
    async def run(self, llm: PoolClient, task: Task) -> Rollout:
        prompt = Message(Role.USER, task.prompt)
        turn = await llm.sample([prompt])
        return Rollout(task=task, messages=[prompt, turn.message], turns=[turn])
