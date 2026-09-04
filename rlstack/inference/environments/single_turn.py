"""One sample call, one turn, done: the smallest complete environment.

`math_single_turn` is this same body under a name that says math. A prompt
corpus that is not math (ADR 0005's `concept_prompts`) runs the identical
episode, and a word used with a different meaning is a finding
(ARCHITECTURE.md) — so the shape gets an honest name here. The old name
stays registered: runs that hashed it keep their identity (I3).
"""

from __future__ import annotations

from rlstack.data.trajectory import Message, Role, Task
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.rollout import Rollout


@environment("single_turn")
class SingleTurn(Environment):
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        prompt = Message(Role.USER, task.prompt)
        turn = await client.sample([prompt])
        return Rollout(task=task, messages=[prompt, turn.message], turns=[turn])
