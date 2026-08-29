"""One competition-math attempt: sample once, keep whatever the model wrote.

The same interaction shape as math_single_turn, registered separately because
an environment IS an interaction protocol and this campaign's protocol will
grow (a second turn that asks for the final answer when the first is
unparseable is the obvious next step, and it must not silently change the
identity of every math run in the repo).

WHAT THIS OWNS: the turns. One user message, one assistant completion, no stop
strings — the prompt was chat-formatted at task-build time, so the model's own
end-of-turn token ends the completion, and a stop string here could only cut it
earlier and wrongly.

WHAT IT DELIBERATELY DOES NOT OWN: the answer. Whether a completion contains a
right answer is postprocessing (boxed_verifier), and whether it ran out of room
before saying one is readable at the seal from the turn's finish reason. A
reasoning model spends most of its tokens in a think block, so the ANSWER is
what post reads and the LENGTH is what the campaign has to budget — the
environment stays out of both.
"""

from __future__ import annotations

from rlstack.data.trajectory import Message, Role, Task
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.rollout import Rollout


@environment("dapo_math")
class DapoMath(Environment):
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        prompt = Message(Role.USER, task.prompt)
        turn = await client.sample([prompt])
        return Rollout(task=task, messages=[prompt, turn.message], turns=[turn])
