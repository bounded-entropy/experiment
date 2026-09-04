"""The conditioned teacher: sample under a hint the student will never see,
and seal the hint OUT of the message stream (ADR 0005).

One sample call against `main`, with `[hint_for(task), prompt]` in the
context and no directive — so the pool answers as the model conditioned on
the task's system block, and the trajectory that comes out holds the prompt
and the completion alone. That asymmetry IS the experiment: a student
trained on these rows sees exactly what it will be asked at eval, and
everything the teacher knew that the student does not is what the
intervention has to reproduce.

The hint is kept as PROVENANCE in `env_extras` — the environment's open
notebook — and never in `messages`, so `flatten` cannot tokenize it into the
student's document. Nothing here is a steer: the teacher run's bank is empty
(the bare base under the hint is the conditioned teacher), so its turns
record no window, and at replay a row with no record is steered at every
position — the adapter type's own default (ADR 0004 Q2, ADR 0005 Q3).
"""

from __future__ import annotations

from rlstack.data.trajectory import Message, Role, Task, hint_for
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.rollout import Rollout


@environment("conditioned_teacher")
class ConditionedTeacher(Environment):
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        hint = hint_for(task)
        prompt = Message(Role.USER, task.prompt)
        turn = await client.sample([hint, prompt])
        return Rollout(task=task, messages=[prompt, turn.message],
                       turns=[turn], env_extras={"hint": hint.content})
