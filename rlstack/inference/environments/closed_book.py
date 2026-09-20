"""Answer one question CLOSED BOOK under the route the task names — SEAL's
evaluation episode (research-log post 0008, gate G1; ADR 0019).

Greedy, 64 new tokens, and NO stop string: the prompt is whatever the task
rendered (SEAL's base-model prompt ending in "Answer:\\n", or the same content
through a chat template with thinking off), a base model's answer may open
with a newline, and a chat model ends its own turn. The answer that gets
graded is the completion's first line, which is the postprocessor's rule
(`squad_answers.answered`), never this file's.

The task's `meta["route"]` is the set the answer is sampled under: `base`, a
set of the run's own bank, or `lib:<name>` — a fitted adapter from the
library, which the Generator waits for by name (ADR 0019)."""

from __future__ import annotations

from rlstack.data.trajectory import Message, Role, Task
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.rollout import Rollout
from rlstack.policy.adapters.dream_bank import Route
from rlstack.spec.specs import SamplingSpec


@environment("closed_book")
class ClosedBook(Environment):
    sampling = SamplingSpec(temperature=0.0, top_p=1.0, max_tokens=64)

    async def run(self, client: PoolClient, task: Task) -> Rollout:
        prompt = Message(Role.USER, task.prompt)
        turn = await client.sample([prompt], directives=(Route(str(task.meta["route"])),))
        return Rollout(task=task, messages=[prompt, turn.message], turns=[turn])
