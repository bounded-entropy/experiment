"""Assembly: a plan's wave becomes real, on whichever side of the bridge it is.

Two verbs, one per leaf constructor, because a leaf is the whole taxonomy
(data/plan.py):

    sample_wave   MAKE the trajectories a WavePlan of Sample leaves names —
                  the Generator's work, pool traffic, seeds drawn here.
    realize       TAKE the trajectories a WavePlan of Replay leaves names —
                  the Trainer's work, sealed rows read through refs, and the
                  ONE place "not yet" is answered.

What used to be three feeds (live / replay / static) is these two verbs and a
ref grammar: a wave of Sample leaves is what "live" meant, a wave of Replay
leaves into another run is "replay", into a cas file is "static", and a wave
holding both — a fresh rollout beside its anchor — was previously inexpressible.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rlstack.data.plan import GroupPlan, PlanError, Replay, Sample, WavePlan, WaveRef, Waves
from rlstack.data.trajectory import Group, Task, Trajectory, Wave
from rlstack.runner.refs import RefReader
from rlstack.runner.seeds import derive
from rlstack.runner.traffic import EnginePoolClient, Routes, run_episode
from rlstack.spec.specs import SamplingSpec


# ---------------------------------------------------------------------------
# make: the Generator's side
# ---------------------------------------------------------------------------

async def sample_wave(plan: WavePlan, *, index: int, tasks: Mapping[str, Task],
                      sampling: SamplingSpec, routes: Routes, master: int,
                      phase: str = "rollout",
                      max_inflight: int = 64) -> Wave:
    """Run every Sample leaf of one planned wave; return the sealed Wave.

    Deterministic given (master, phase, index) and the PLAN: an episode's seed
    is derived from its group key and its position in that group, so the same
    plan replays the same draws no matter how the leaves interleave in flight.
    Episodes are scheduled flat and reassembled in PLAN order, so completion
    order cannot reach the bytes (#53).
    """
    limiter = asyncio.Semaphore(max_inflight)
    jobs: list[asyncio.Future] = []
    shape: list[int] = []

    async def one(group_key: str, position: int, leaf: Sample) -> Trajectory:
        async with limiter:
            if leaf.task_id not in tasks:
                raise PlanError(
                    f"group {group_key!r} samples task {leaf.task_id!r}, which "
                    f"no declared task set contains")
            seed = derive(master, phase, index, group_key, position)
            client = EnginePoolClient(routes, sampling, seed)
            return await run_episode(leaf.env, tasks[leaf.task_id], client)

    for group in plan.groups:
        shape.append(len(group.leaves))
        for position, leaf in enumerate(group.leaves):
            if not isinstance(leaf, Sample):
                raise PlanError(
                    f"group {group.key!r}: a rollout plan MAKES trajectories, "
                    f"so its leaves are Sample; got {type(leaf).__name__}")
            jobs.append(one(group.key, position, leaf))

    done = await asyncio.gather(*jobs)
    groups, cursor = [], 0
    for group, width in zip(plan.groups, shape):
        groups.append(Group(group.key, done[cursor:cursor + width]))
        cursor += width
    return Wave(groups)


# ---------------------------------------------------------------------------
# take: the Trainer's side
# ---------------------------------------------------------------------------

def realize(entry: Waves, reader: RefReader) -> list[dict[str, Any]] | None:
    """One planned wave as sealed rows — or None while a leaf is still pending.

    None is the trainer's whole await condition, and only a `self://` ref can
    produce it: everything else was sealed before this run began and the submit
    gate proved it resolvable. The rows come back tagged with THIS plan's group
    keys, never the keys they were sealed under, because a group is assigned at
    assembly (data/trajectory.py) — which is what lets one run regroup another
    run's trajectories without copying them.
    """
    if isinstance(entry, WaveRef):
        return reader.rows(entry.ref)          # the sealed wave IS its groups
    rows: list[dict[str, Any]] = []
    for group in entry.groups:
        for row in _group_rows(group, reader):
            if row is None:
                return None                    # not yet: leave the wave unbuilt
            rows.append(dict(row, group=group.key))
    return rows


def _group_rows(group: GroupPlan, reader: RefReader):
    """One group's rows in leaf order, each None while it is unsealed."""
    for leaf in group.leaves:
        if not isinstance(leaf, Replay):
            raise PlanError(
                f"group {group.key!r}: a train plan TAKES trajectories, so its "
                f"leaves are Replay or a WaveRef; got {type(leaf).__name__} — "
                f"sampling belongs to the rollout plan")
        yield reader.row(leaf.ref)


def pending_refs(entry: Waves) -> tuple[str, ...]:
    """Every ref a wave waits on — what an observer shows while it is unbuilt."""
    if isinstance(entry, WaveRef):
        return (entry.ref,)
    return tuple(leaf.ref for group in entry.groups for leaf in group.leaves
                 if isinstance(leaf, Replay))


def rollouts_needed(waves: Sequence[Waves]) -> dict[int, int]:
    """rollout index -> the FIRST update that consumes it.

    The generator's lag rule reads this: a rollout is due when the update that
    first needs it is due, which is how one plan paces two daemons without
    either calling the other.
    """
    from rlstack.runner.refs import SELF, parse

    first: dict[int, int] = {}
    for update, entry in enumerate(waves, start=1):
        for ref in pending_refs(entry):
            location = parse(ref).location
            if location.startswith(SELF):
                index = int(location[len(SELF):])
                first.setdefault(index, update)
    return first
