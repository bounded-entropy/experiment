"""Answer one question under the set the task names — greedy, short, stopped
at the line's end (ADR 0018).

The task's `meta["route"]` is the dream_bank set (a memory, the dreamer, or
`base`) the answer is sampled under; the `Route` directive carries it and
the engine records it, so the sealed row says which memory answered. The
prompt is plain text the task rendered (a base model, no chat template);
the completion is the answer line. Grading is a postprocessor's
(`squad_answers`), never this file's."""

from __future__ import annotations

from rlstack.data.trajectory import Message, Role, Task
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.rollout import Rollout
from rlstack.policy.adapters.dream_bank import Route
from rlstack.spec.specs import SamplingSpec


@environment("answer")
class Answer(Environment):
    sampling = SamplingSpec(temperature=0.0, top_p=1.0, max_tokens=16)

    async def run(self, client: PoolClient, task: Task) -> Rollout:
        prompt = Message(Role.USER, task.prompt)
        turn = await client.sample([prompt], stop=("\n",),
                                   directives=(Route(str(task.meta["route"])),))
        return Rollout(task=task, messages=[prompt, turn.message], turns=[turn])
