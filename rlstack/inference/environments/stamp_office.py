"""One Stamp Office attempt: sample once, keep whatever the model wrote.

The same interaction shape as dapo_math, registered separately for the same
reason: an environment IS an interaction protocol, and this campaign's
protocol may grow (a second turn showing the office's error message is the
obvious next step) without silently changing the identity of every math run.

WHAT THIS OWNS: the turns — one user message (rulebook + request, chat-
formatted at task-build time), one assistant completion, no stop strings.
WHAT IT DOES NOT OWN: the grading. Whether the calls were legal and the
document landed in the right drawer is postprocessing (stamp_grade)."""

from __future__ import annotations

from rlstack.data.trajectory import Message, Role, Task
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.rollout import Rollout


@environment("stamp_office")
class StampOffice(Environment):
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        prompt = Message(Role.USER, task.prompt)
        turn = await client.sample([prompt])
        return Rollout(task=task, messages=[prompt, turn.message], turns=[turn])
