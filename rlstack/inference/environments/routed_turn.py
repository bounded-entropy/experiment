"""One sample call under the route the task names, with the RUN's sampling
(research-log post 0008; ADR 0019).

`single_turn` sends no directive, so a dream_bank serves it under the default
set — the dreamer ALONE. A menu task is sampled by the dreamer STACKED on a
library memory, and which memory is the task's fact: `meta["route"]` is
`lib:<name>+dreamer`, carried by the `Route` directive and recorded at the
seal, so the replayed row trains under the same stack. A Sample leaf's role
never reaches the engine (it is stamped at realize, for the train wave); the
route a rollout is SAMPLED under is the environment's to say, as `answer`
and `closed_book` say theirs. Unlike those two this environment declares no
sampling of its own: thinking then dreaming wants the run's temperature and
token budget."""

from __future__ import annotations

from rlstack.data.trajectory import Message, Role, Task
from rlstack.inference.environments.base import Environment, PoolClient, environment
from rlstack.inference.rollout import Rollout
from rlstack.policy.adapters.dream_bank import Route


@environment("routed_turn")
class RoutedTurn(Environment):
    async def run(self, client: PoolClient, task: Task) -> Rollout:
        prompt = Message(Role.USER, task.prompt)
        turn = await client.sample([prompt], directives=(Route(str(task.meta["route"])),))
        return Rollout(task=task, messages=[prompt, turn.message], turns=[turn])
