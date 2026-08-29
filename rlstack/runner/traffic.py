"""The client side of pool traffic: how an episode talks to pools.

EnginePoolClient drives one pool for one episode — sample assembles a token
stream into Turns, score is one prefill pass — and `pool(name)` hands a
sibling for any other declared pool under one shared seed sequence.
run_episode is the seal point; collect_wave assembles one wave of episodes
deterministically given (master, update), assigning group keys.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping, Sequence

from rlstack.data.stores.base import Store
from rlstack.data.tasks import load_tasks
from rlstack.data.trajectory import Group, Message, Role, Task, Trajectory, Turn, Wave
from rlstack.registry import ENVS
from rlstack.policy.compile import Bundle
from rlstack.runner.interfaces import Engine, FinishEvent
from rlstack.runner.seeds import derive
from rlstack.spec.specs import SamplingSpec

# One named pool: the engine and the bundle its requests pin.
Routes = Mapping[str, tuple[Engine, Bundle]]


class EnginePoolClient:
    """The concrete PoolClient: one instance per episode per pool, all
    siblings sharing ONE seed sequence, so multi-pool traffic is deterministic
    regardless of which pools an episode touches."""

    def __init__(self, routes: Routes, sampling: SamplingSpec, episode_seed: int,
                 pool_name: str = "main", _counter: list[int] | None = None) -> None:
        if pool_name not in routes:
            raise KeyError(
                f"unknown engine pool {pool_name!r}; pools: {sorted(routes)}")
        self._routes = routes
        self._sampling = sampling
        self._episode_seed = episode_seed
        self._pool_name = pool_name
        self._engine, self._bundle = routes[pool_name]
        self._counter = _counter if _counter is not None else [0]

    def pool(self, name: str) -> "EnginePoolClient":
        """A sibling client for another pool, sharing this episode's seeds."""
        return EnginePoolClient(self._routes, self._sampling, self._episode_seed,
                                  name, self._counter)

    async def score(self, messages: Sequence[Message],
                    token_ids: Sequence[int]) -> tuple[float, ...]:
        """Score given tokens under this pool's pinned bundle. Deterministic:
        consumes NO seed from the episode's sequence (scoring draws nothing),
        so adding a scoring processor never shifts sampling seeds."""
        return await self._engine.score_tokens(
            messages, tuple(token_ids), self._bundle.bundle_id)

    async def sample(self, messages: Sequence[Message],
                     stop: tuple[str, ...] = ()) -> Turn:
        seed = derive(self._episode_seed, "call", self._counter[0])
        self._counter[0] += 1

        token_ids: list[int] = []
        logprobs: list[float] = []
        text_parts: list[str] = []
        columns: dict[str, list] = {}
        finish: FinishEvent | None = None

        stream = self._engine.sample_tokens(
            messages, self._sampling, stop, self._bundle.bundle_id, seed)
        async for event in stream:
            if isinstance(event, FinishEvent):
                finish = event
                break
            token_ids.append(event.token_id)
            logprobs.append(event.logprob)
            text_parts.append(event.text_delta)
            for name, value in event.extras.items():
                # a column appearing mid-stream backfills earlier positions
                columns.setdefault(name, [None] * (len(token_ids) - 1)).append(value)
            for name in columns:
                if len(columns[name]) < len(token_ids):
                    columns[name].append(None)
        if finish is None:
            raise RuntimeError("engine stream ended without a FinishEvent")

        return Turn(
            message=Message(Role.ASSISTANT, "".join(text_parts)),
            token_ids=tuple(token_ids),
            behavior_logprobs=tuple(logprobs),
            finish=finish.finish,
            stop_hit=finish.stop_hit,
            bundle_id=self._bundle.bundle_id,
            policy_version=dict(self._bundle.policy_version),
            seed=seed,
            token_extras={k: tuple(v) for k, v in columns.items()},
            turn_extras=dict(finish.turn_extras),
        )


def load_task_sets(store: Store, uris: Sequence[str]) -> dict[str, Task]:
    """Every declared task set, keyed by task id — what a plan's leaves name.

    Ids are global across a run's sets: a leaf carries an id alone, so two sets
    sharing one id would make a leaf ambiguous, and the collision is refused
    here rather than silently resolved by set order.
    """
    tasks: dict[str, Task] = {}
    for uri in uris:
        for task in load_tasks(store, uri):
            if task.id in tasks:
                raise ValueError(
                    f"task id {task.id!r} appears in two declared task sets; "
                    f"a plan's leaf names an id alone, so ids must be unique")
            tasks[task.id] = task
    return tasks


async def run_episode(env_name: str, task: Task,
                      client: EnginePoolClient) -> Trajectory:
    """One episode across the membrane: the Environment produces the Rollout,
    the seal turns it into a Trajectory (I1)."""
    rollout = await ENVS.get(env_name).instance.run(client, task)
    return rollout.seal()
