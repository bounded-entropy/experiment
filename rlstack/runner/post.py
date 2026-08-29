"""EXECUTION of the post pipeline: per group, after the seal, before the loss.

The processors themselves are DECLARED in training/post/; this module only runs
a declared pipeline in order. Each processor sees the columns its predecessors
produced for that group and a PoolClient, and its output is validated against
its declaration — exactly the `produces` names, one float per trajectory —
before being concatenated into wave-order columns. Deterministic: each (group,
processor) draws its own seed from the tree, so resume recomputes byte-
identical postdata.

ONE runner, TWO callers. A pipeline is split by `split_pipeline` into the
pooled half the Scorer runs and the pool-less half the Trainer runs, and each
half comes through here unchanged — same order, same seed paths, same
validation. `given` is the seam: wave-order columns the OTHER caller already
produced, sliced back per group so a processor cannot tell which daemon
produced what it consumes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from rlstack.data.trajectory import Group, Wave
from rlstack.registry import POST
from rlstack.runner.traffic import EnginePoolClient, Routes
from rlstack.runner.seeds import derive
from rlstack.spec.specs import SamplingSpec


def _token_vector(processor: str, column: str, traj, value) -> list[float]:
    """A token_level column carries one float per GENERATED token of its
    trajectory, in sealed order — the channel for per-token teacher signals
    (#38: the loss is pure math; post produces everything it operates on)."""
    generated = sum(len(turn.token_ids) for turn in traj.turns)
    if len(value) != generated:
        raise ValueError(
            f"postprocessor {processor!r} token_level column {column!r} has "
            f"{len(value)} floats for a trajectory with {generated} generated "
            f"tokens")
    return [float(v) for v in value]


def group_slice(given: Mapping[str, Sequence], wave: Wave,
                index: int) -> dict[str, list]:
    """One group's share of wave-order columns another runner produced.

    Wave order IS the groups concatenated in order, so a group's rows are the
    contiguous window at its offset — the same arithmetic `broadcast` and the
    observer walk the columns with, stated once here for the seam.
    """
    start = sum(len(group) for group in wave.groups[:index])
    width = len(wave.groups[index])
    return {name: list(values[start:start + width])
            for name, values in given.items()}


async def run_pipeline(
    pipeline: Sequence[str],
    wave: Wave,
    routes: Routes,
    sampling: SamplingSpec,
    master: int,
    update: int,
    phase: str = "post",
    given: Mapping[str, Sequence] | None = None,
) -> dict[str, list[float]]:
    """The pipeline over every group; columns aligned to wave order.

    `given` seeds each group's `data` with columns produced elsewhere (the
    Scorer's part), so a processor consumes one exactly as it consumes a
    neighbour's — and the return value stays THIS call's own produces, never
    the given columns echoed back. What makes that identical to running the
    whole pipeline in one process is the contract `consumes` already states: a
    processor reads its declared inputs and nothing else, and every declared
    input is produced earlier (checked at Phase 0), so no processor can observe
    which half of the split it is in.
    """

    async def one_group(index: int, group: Group) -> dict[str, list[float]]:
        data: dict[str, list[float]] = (
            group_slice(given, wave, index) if given else {})
        produced: dict[str, None] = {}
        for name in pipeline:
            pdef = POST.get(name)
            # a processor's declared sampling (a judge's own budget) wins
            # over the run's generation sampling; None inherits
            client = EnginePoolClient(
                routes, pdef.instance.sampling or sampling,
                derive(master, phase, update, group.key, name))
            out = await pdef.instance.process(group, data, client)
            if set(out) != set(pdef.produces):
                raise ValueError(
                    f"postprocessor {name!r} declared produces={pdef.produces} "
                    f"but returned columns {sorted(out)}")
            for column, values in out.items():
                if len(values) != len(group):
                    raise ValueError(
                        f"postprocessor {name!r} column {column!r} has "
                        f"{len(values)} values for {len(group)} trajectories")
                if column in pdef.token_level:
                    data[column] = [_token_vector(name, column, traj, value)
                                    for traj, value
                                    in zip(group.trajectories, values)]
                else:
                    data[column] = [float(v) for v in values]
                produced[column] = None
        return {column: data[column] for column in produced}

    per_group = await asyncio.gather(
        *[one_group(index, group) for index, group in enumerate(wave.groups)])

    columns: dict[str, list[float]] = {}
    for data in per_group:
        for name, values in data.items():
            columns.setdefault(name, []).extend(values)
    return columns
