"""Wave collection: envs and rewards run against the engine; trajectories seal.

The SampleClient is what an env sees — it pins the current bundle, derives a
seed per sample call, and assembles the engine's token stream into a Turn.
`collect_wave` schedules one wave: `rollouts_per_wave / group_size` tasks
(chosen deterministically per update), `group_size` rollouts each, so group
identity = task identity, which is what grpo_group_norm assumes. Each
trajectory seals the moment its own rewards finish.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Sequence
from typing import Any

from rlstack.data.store import Store
from rlstack.data.trajectory import Group, Message, Role, Task, Trajectory, Turn, Wave
from rlstack.policy.compile import Bundle
from rlstack.registry import ENVS, REWARDS
from rlstack.runner.interfaces import Engine, FinishEvent, TokenEvent
from rlstack.runner.seeds import derive
from rlstack.spec.specs import SamplingSpec


def load_tasks(store: Store, uri: str) -> list[Task]:
    """Materialize a cas://-addressed jsonl of {id, prompt, meta} rows."""
    rows = [json.loads(line) for line in
            store.cas_get(uri).decode("utf-8").splitlines() if line]
    return [Task(id=r["id"], prompt=r["prompt"], meta=r.get("meta", {})) for r in rows]


class SampleClient:
    """What an env (or a sampling reward) sees: `await llm.sample(msgs) -> Turn`.

    Fills sampling defaults, pins the wave's bundle, derives one seed per call
    from the rollout's seed, and folds the token stream into a Turn — text from
    the deltas, per-token extras into columns, finish from the terminal event.
    """

    def __init__(self, engine: Engine, sampling: SamplingSpec, bundle: Bundle,
                 rollout_seed: int) -> None:
        self._engine = engine
        self._sampling = sampling
        self._bundle = bundle
        self._rollout_seed = rollout_seed
        self._calls = 0

    async def sample(self, messages: Sequence[Message],
                     stop: tuple[str, ...] = ()) -> Turn:
        seed = derive(self._rollout_seed, "call", self._calls)
        self._calls += 1

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


async def run_episode(env_name: str, reward_names: Sequence[str],
                      task: Task, client: SampleClient) -> Trajectory:
    """One rollout: env produces the trajectory, rewards fill components, seal."""
    traj = await ENVS.get(env_name).fn(client, task)
    for name in reward_names:
        components = await REWARDS.get(name).fn(traj, client)
        for key, value in components.items():
            if key in traj.reward_components:
                raise ValueError(
                    f"reward {name!r} wrote component {key!r}, already written "
                    f"— components must have one owner (I4)")
            traj.reward_components[key] = float(value)
    return traj.seal()


def choose_tasks(tasks: Sequence[Task], n_groups: int, master: int,
                 phase: str, update: int) -> list[Task]:
    """Deterministic without-replacement draw of this wave's tasks. Distinct
    tasks per wave keep group identity = task identity within the wave."""
    if n_groups > len(tasks):
        raise ValueError(
            f"wave needs {n_groups} distinct tasks but the set has {len(tasks)}")
    rng = random.Random(derive(master, phase, update, "tasks"))
    return rng.sample(list(tasks), n_groups)


async def collect_wave(
    update: int,
    *,
    env_name: str,
    reward_names: Sequence[str],
    sampling: SamplingSpec,
    tasks: Sequence[Task],
    group_size: int,
    rollouts_per_wave: int,
    engine: Engine,
    bundle: Bundle,
    master: int,
    phase: str = "rollout",
    max_inflight: int = 64,
) -> Wave:
    """One wave of sealed groups, deterministic given (master, update).

    Group keys are ASSIGNED here — one group per chosen task, keyed by the task
    id. The Group primitive doesn't require that: a TTT-style wave of many
    groups over one task just assembles differently at this spot.
    """
    if rollouts_per_wave % group_size:
        raise ValueError(
            f"rollouts_per_wave={rollouts_per_wave} is not a multiple of "
            f"group_size={group_size}")
    chosen = choose_tasks(tasks, rollouts_per_wave // group_size,
                          master, phase, update)

    limiter = asyncio.Semaphore(max_inflight)

    async def one(task: Task, sample_index: int) -> Trajectory:
        async with limiter:
            rollout_seed = derive(master, phase, update, task.id, sample_index)
            client = SampleClient(engine, sampling, bundle, rollout_seed)
            return await run_episode(env_name, reward_names, task, client)

    jobs = [one(task, s) for task in chosen for s in range(group_size)]
    trajectories = await asyncio.gather(*jobs)
    return Wave([
        Group(task.id, trajectories[i * group_size:(i + 1) * group_size])
        for i, task in enumerate(chosen)
    ])
