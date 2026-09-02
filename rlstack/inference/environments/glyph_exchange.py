"""One Glyph Exchange attempt: sample once, keep whatever the model wrote.

Single-turn like stamp_office and registered separately — the two DSLs are
two interaction protocols, and each must be able to grow a turn without
renaming the other's runs. Grading is glyph_grade's, not the environment's."""

from __future__ import annotations

from rlstack.data.trajectory import Message, Role, Task
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.rollout import Rollout


@environment("glyph_exchange")
class GlyphExchange(Environment):
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        prompt = Message(Role.USER, task.prompt)
        turn = await client.sample([prompt])
        return Rollout(task=task, messages=[prompt, turn.message], turns=[turn])
