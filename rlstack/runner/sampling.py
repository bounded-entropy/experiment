"""The runner's sampling side: everything between an engine's token stream
and a sealed wave.

Bottom-up: EngineSampleClient is the concrete SampleClient (rlstack/client.py
protocol) — it drives one engine pool for one episode, assembling TokenEvents
into Turns, and `pool(name)` hands environments and postprocessors a sibling
for any other pool under ONE shared seed sequence. `run_episode` is the seal
point: the Environment produces the Rollout, the seal turns it into a
Trajectory (I1). `collect_wave` schedules one wave of episodes —
`trajectories_per_wave / group_size` tasks, group keys ASSIGNED here, one
Group per task — deterministic given (master, update). `load_tasks` reads a
cas:// task file. Scoring is none of this module's business: it happens later,
in the post pipeline (runner/post.py).
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Mapping, Sequence

from rlstack.data.stores.base import Store
from rlstack.data.trajectory import Group, Message, Role, Task, Trajectory, Turn, Wave
from rlstack.registry import ENVS
from rlstack.policy.compile import Bundle
from rlstack.runner.interfaces import Engine, FinishEvent
from rlstack.runner.seeds import derive
from rlstack.spec.specs import SamplingSpec

# One named pool: the engine and the bundle its requests pin.
Routes = Mapping[str, tuple[Engine, Bundle]]


class EngineSampleClient:
    """One instance per episode per pool; all siblings share ONE seed
    sequence, so multi-pool traffic is deterministic regardless of which
    pools an episode touches."""

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

    def pool(self, name: str) -> "EngineSampleClient":
        """A sibling client for another pool, sharing this episode's seeds."""
        return EngineSampleClient(self._routes, self._sampling, self._episode_seed,
                                  name, self._counter)

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


def load_tasks(store: Store, uri: str) -> list[Task]:
    """Materialize a cas://-addressed jsonl of {id, prompt, meta} rows."""
    rows = [json.loads(line) for line in
            store.cas_get(uri).decode("utf-8").splitlines() if line]
    return [Task(id=r["id"], prompt=r["prompt"], meta=r.get("meta", {})) for r in rows]


async def run_episode(env_name: str, task: Task,
                      client: EngineSampleClient) -> Trajectory:
    """One episode across the membrane: the Environment produces the Rollout,
    the seal turns it into a Trajectory (I1)."""
    rollout = await ENVS.get(env_name).instance.run(client, task)
    return rollout.seal()


def choose_tasks(tasks: Sequence[Task], n_groups: int, master: int,
                 phase: str, update: int) -> list[Task]:
    """Deterministic without-replacement draw of this wave's tasks."""
    if n_groups > len(tasks):
        raise ValueError(
            f"wave needs {n_groups} distinct tasks but the set has {len(tasks)}")
    rng = random.Random(derive(master, phase, update, "tasks"))
    return rng.sample(list(tasks), n_groups)


async def collect_wave(
    update: int,
    *,
    env_name: str,
    sampling: SamplingSpec,
    tasks: Sequence[Task],
    group_size: int,
    trajectories_per_wave: int,
    routes: Routes,
    master: int,
    phase: str = "rollout",
    max_inflight: int = 64,
) -> Wave:
    """One wave of sealed groups, deterministic given (master, update).

    Group keys are ASSIGNED here — one group per chosen task, keyed by the task
    id. The Group primitive doesn't require that: a TTT-style wave of many
    groups over one task just assembles differently at this spot.
    """
    if trajectories_per_wave % group_size:
        raise ValueError(
            f"trajectories_per_wave={trajectories_per_wave} is not a multiple "
            f"of group_size={group_size}")
    chosen = choose_tasks(tasks, trajectories_per_wave // group_size,
                          master, phase, update)

    limiter = asyncio.Semaphore(max_inflight)

    async def one(task: Task, sample_index: int) -> Trajectory:
        async with limiter:
            episode_seed = derive(master, phase, update, task.id, sample_index)
            client = EngineSampleClient(routes, sampling, episode_seed)
            return await run_episode(env_name, task, client)

    jobs = [one(task, s) for task in chosen for s in range(group_size)]
    trajectories = await asyncio.gather(*jobs)
    return Wave([
        Group(task.id, trajectories[i * group_size:(i + 1) * group_size])
        for i, task in enumerate(chosen)
    ])
