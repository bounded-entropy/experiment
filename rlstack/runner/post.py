"""EXECUTION of the post pipeline: per group, after the seal, before the loss.
The processors themselves are DECLARED in training/post/ (one class per file);
this module only runs a declared pipeline in order.

For each group, the declared processors run in pipeline order; each sees the
columns its predecessors produced for that group and a SampleClient (judges
sample; `llm.pool(name)` reaches any engine pool). Outputs are validated
against the declaration — exactly the `produces` names, one float per
trajectory — and concatenated into wave-order columns, which the loop stores
as postdata and broadcasts into the TokenBatch.

Deterministic: each (group, processor) gets its own seed from the tree, so
resume recomputes byte-identical postdata.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from rlstack.data.trajectory import Group, Wave
from rlstack.registry import POST
from rlstack.runner.sampling import EngineSampleClient, Pools
from rlstack.runner.seeds import derive
from rlstack.spec.specs import SamplingSpec


async def run_pipeline(
    pipeline: Sequence[str],
    wave: Wave,
    pools: Pools,
    sampling: SamplingSpec,
    master: int,
    update: int,
    phase: str = "post",
) -> dict[str, list[float]]:
    """The pipeline over every group; columns aligned to wave order."""

    async def one_group(group: Group) -> dict[str, list[float]]:
        data: dict[str, list[float]] = {}
        for name in pipeline:
            pdef = POST.get(name)
            # a processor's declared sampling (a judge's own budget) wins
            # over the run's generation sampling; None inherits
            client = EngineSampleClient(
                pools, pdef.instance.sampling or sampling,
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
                data[column] = [float(v) for v in values]
        return data

    per_group = await asyncio.gather(*[one_group(g) for g in wave.groups])

    columns: dict[str, list[float]] = {}
    for data in per_group:
        for name, values in data.items():
            columns.setdefault(name, []).extend(values)
    return columns
