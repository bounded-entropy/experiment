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
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import copy

from rlstack.data.plan import (
    TRAIN, Continue, GroupPlan, HistoricalDerive, HistoricalSample, PlanError,
    Replay, RunPlan, Sample, WavePlan, WaveRef, Waves,
    route_of_role,
)
from rlstack.data.trajectory import Group, Task, Trajectory, Wave, trajectory_from_row
from rlstack.code import require_declared_function
from rlstack.runner.refs import SELF, RefReader, parse
from rlstack.runner.seeds import derive
from rlstack.runner.traffic import EnginePoolClient, Routes, run_episode
from rlstack.spec.specs import SamplingSpec


# ---------------------------------------------------------------------------
# make: the Generator's side
# ---------------------------------------------------------------------------

async def sample_wave(plan: WavePlan, *, index: int, tasks: Mapping[str, Task],
                      sampling: SamplingSpec, routes: Routes, master: int,
                      functions: Sequence[str], code: tuple[str, ...],
                      phase: str = "rollout",
                      max_inflight: int = 64,
                      reader: RefReader | None = None,
                      trace: list[dict] | None = None) -> Wave:
    """Run every Sample and Continue leaf of one planned wave; return the
    sealed Wave.

    Deterministic given (master, phase, index) and the PLAN: an episode's seed
    is derived from its group key and its position in that group, so the same
    plan replays the same draws no matter how the leaves interleave in flight.
    Episodes are scheduled flat and reassembled in PLAN order, so completion
    order cannot reach the bytes (#53).

    Continue resolves one sealed source before invoking the experiment's
    function. Self sources name earlier rollout waves. The same per-leaf seed
    is reused after a crash; already sealed waves are not invoked again.

    `trace`, when handed a list, receives one row per episode in PLAN order:
    when it was admitted and ended, and every sample call it made (the
    generator journals it; it never reaches the wave's bytes).
    """
    check_generation_plan(RunPlan((plan,)), functions, first_index=index)
    resolved = {name: require_declared_function(name, code) for name in functions}
    limiter = asyncio.Semaphore(max_inflight)
    jobs: list[asyncio.Future] = []
    shape: list[int] = []

    def sealed_source(group_key: str, leaf: Continue) -> Trajectory:
        if reader is None:
            raise PlanError(
                f"group {group_key!r} derives from {leaf.source!r} but this "
                f"caller resolves no refs — Continue leaves need the run's "
                f"reader")
        row = reader.row(leaf.source)
        if row is None:
            raise PlanError(
                f"group {group_key!r} derives from {leaf.source!r}, which is "
                f"not sealed yet — a Continue sources an EARLIER wave (the "
                f"generator makes waves in order)")
        return trajectory_from_row(row)

    async def one(group_key: str, position: int, leaf: Sample | Continue,
                  row: dict | None) -> Trajectory:
        async with limiter:
            if row is not None:
                row["start"] = time.time()
            if isinstance(leaf, Continue):
                source = sealed_source(group_key, leaf)
            else:
                if leaf.task_id not in tasks:
                    raise PlanError(
                        f"group {group_key!r} samples task {leaf.task_id!r}, "
                        f"which no declared task set contains")
                source = tasks[leaf.task_id]
            seed = derive(master, phase, index, group_key, position)
            client = EnginePoolClient(routes, sampling, seed,
                                      calls=row["calls"] if row is not None else None,
                                      episode=episode_id(phase, index, group_key, position))
            trajectory = await run_episode(resolved[leaf.function], source, client)
            if row is not None:
                row["end"] = time.time()
            return trajectory

    for group in plan.groups:
        shape.append(len(group.leaves))
        for position, leaf in enumerate(group.leaves):
            if not isinstance(leaf, (Sample, Continue)):
                raise PlanError(
                    f"group {group.key!r}: a rollout plan MAKES trajectories, "
                    f"so its leaves are Sample or Continue; got "
                    f"{type(leaf).__name__}")
            row = None
            if trace is not None:
                row = {"group": group.key, "position": position, "calls": []}
                trace.append(row)
            jobs.append(one(group.key, position, leaf, row))

    done = await asyncio.gather(*jobs)
    groups, cursor = [], 0
    for group, width in zip(plan.groups, shape):
        groups.append(Group(group.key, done[cursor:cursor + width]))
        cursor += width
    return Wave(groups)


def check_generation_plan(plan: RunPlan, functions: Sequence[str], *,
                          first_index: int = 1) -> None:
    """Only declared functions execute; self continuations source earlier waves.

    Historical leaves decode for inspection but cannot enter a new run.
    Checking the whole plan before setup avoids discovering a bad later leaf
    after earlier waves have already sampled.
    """
    for index, wave in enumerate(plan.waves, first_index):
        if not isinstance(wave, WavePlan):
            raise PlanError("a rollout plan makes trajectories and cannot contain WaveRef")
        for leaf in wave.leaves():
            if isinstance(leaf, (HistoricalSample, HistoricalDerive)):
                raise PlanError("legacy environment/maker plans are read-only; migrate to Sample/Continue functions")
            if not isinstance(leaf, (Sample, Continue)):
                raise PlanError("a rollout plan holds only Sample or Continue leaves")
            if leaf.function not in functions:
                raise PlanError(f"function {leaf.function!r} is not declared in gen.functions")
            if isinstance(leaf, Continue):
                source = parse(leaf.source)
                if source.index is None:
                    raise PlanError("Continue requires one sealed source trajectory (#<row>)")
                if source.location.startswith(SELF):
                    before = source.location[len(SELF):]
                    if not before.isdigit() or not 1 <= int(before) < index:
                        raise PlanError(f"Continue {leaf.source!r} must source an earlier rollout wave than {index}")


def episode_id(phase: str, index: int, group_key: str, position: int) -> str:
    """AN EPISODE'S ID: where it sits in the run — the same coordinates its
    seed is derived from, spelled readably. The default affinity key of
    every call it makes (ADR 0022), so one episode's turns share a replica's
    prefix cache."""
    return f"{phase}/{index}/{group_key}/{position}"


# ---------------------------------------------------------------------------
# take: the Trainer's side
# ---------------------------------------------------------------------------

def realize(entry: Waves, reader: RefReader) -> list[dict[str, Any]] | None:
    """One planned wave as sealed rows — or None while a leaf is still pending.

    None is the trainer's whole await condition. A `self://` ref produces it
    while this run's Generator is still sealing; a `store://` ref produces it
    while the run it names is still running (refs.py) — a consumer paces on
    its source either way. A cas ref never does. The rows come back tagged
    with THIS plan's group
    keys, never the keys they were sealed under, because a group is assigned at
    assembly (data/trajectory.py) — which is what lets one run regroup another
    run's trajectories without copying them.
    """
    if isinstance(entry, WaveRef):
        return reader.rows(entry.ref)          # the sealed wave IS its groups
    rows: list[dict[str, Any]] = []
    for group in entry.groups:
        for leaf, row in _group_rows(group, reader):
            if row is None:
                return None                    # not yet: leave the wave unbuilt
            rows.append(dict(stamped(row, leaf.role), group=group.key))
    return rows


def _group_rows(group: GroupPlan, reader: RefReader):
    """One group's (leaf, row) pairs in leaf order, the row None while it is
    unsealed."""
    for leaf in group.leaves:
        if not isinstance(leaf, Replay):
            raise PlanError(
                f"group {group.key!r}: a train plan TAKES trajectories, so its "
                f"leaves are Replay or a WaveRef; got {type(leaf).__name__} — "
                f"sampling belongs to the rollout plan")
        yield leaf, reader.row(leaf.ref)


def stamped(row: dict[str, Any], role: str) -> dict[str, Any]:
    """THE LEAF'S ROLE, AS THE ROW'S FACTS (ADR 0018): a `train` leaf leaves
    the sealed row untouched, byte for byte; any other role is written into
    every turn's facts as `role`, and — when the role names a set — as
    `route`, which is what a replay lowering routes by. A copy, never the
    cached row: the same sealed rollout may be replayed under several roles
    in one plan, and the reader's cache holds the record as sealed.

    Realized on the Scorer and the Trainer alike, so the rows they see agree
    byte for byte, and `waves/<u>` records exactly what trained."""
    if role == TRAIN:
        return row
    stamped_row = copy.deepcopy(row)
    route = route_of_role(role)
    for turn in stamped_row["turns"]:
        turn["turn_extras"]["role"] = role
        if route is not None:
            turn["turn_extras"]["route"] = route
    return stamped_row


def pending_refs(entry: Waves) -> tuple[str, ...]:
    """Every ref a wave waits on — what an observer shows while it is unbuilt."""
    if isinstance(entry, WaveRef):
        return (entry.ref,)
    return tuple(leaf.ref for group in entry.groups for leaf in group.leaves
                 if isinstance(leaf, Replay))


def rollouts_needed(waves: Sequence[Waves]) -> dict[int, int]:
    """rollout index -> the FIRST update that consumes it.

    The generator's lag rule reads this: a rollout is due when the update that
    first needs it is due, which is how one plan paces two runners without
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
