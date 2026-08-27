"""Wave collection: environments drive the engines; each finished Rollout
seals into a Trajectory the moment its episode ends.

An episode is now exactly the environment's business — everything necessary
for the policy to complete a rollout and seal it. Scoring happens later, in
the postprocessing pipeline (runner/post.py). `collect_wave` schedules one
wave: `trajectories_per_wave / group_size` tasks (chosen deterministically per
update), `group_size` episodes each, one Group per task.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Sequence

from rlstack.data.stores.base import Store
from rlstack.data.trajectory import Group, Task, Trajectory, Wave
from rlstack.registry import ENVS
from rlstack.runner.client import EngineSampleClient, Pools
from rlstack.runner.seeds import derive
from rlstack.spec.specs import SamplingSpec


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
    pools: Pools,
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
            f"trajectories_per_wave={trajectories_per_wave} is not a multiple of "
            f"group_size={group_size}")
    chosen = choose_tasks(tasks, trajectories_per_wave // group_size,
                          master, phase, update)

    limiter = asyncio.Semaphore(max_inflight)

    async def one(task: Task, sample_index: int) -> Trajectory:
        async with limiter:
            episode_seed = derive(master, phase, update, task.id, sample_index)
            client = EngineSampleClient(pools, sampling, episode_seed)
            return await run_episode(env_name, task, client)

    jobs = [one(task, s) for task in chosen for s in range(group_size)]
    trajectories = await asyncio.gather(*jobs)
    return Wave([
        Group(task.id, trajectories[i * group_size:(i + 1) * group_size])
        for i, task in enumerate(chosen)
    ])
