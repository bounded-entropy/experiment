"""The eval driver: firewalled measurement (SPEC.md EvalSpec).

Held-out tasks run through the env under a pinned committed bundle, then the
EVAL post pipeline scores the resulting groups — the same postprocessing
machinery training uses, over measurement data. Output goes only to the run's
eval/ section, which crash recovery never consults: eval is measurement, not
run state.
"""

from __future__ import annotations

import asyncio
import json
import math

from rlstack.data.stores.base import RunHandle, Store
from rlstack.data.trajectory import Group, Wave
from rlstack.runner.client import EngineSampleClient, Pools
from rlstack.runner.post import run_pipeline
from rlstack.runner.seeds import derive
from rlstack.runner.waves import load_tasks, run_episode
from rlstack.spec.specs import ExperimentSpec, SamplingSpec


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                   for r in rows)


class EvalDriver:
    """Every held-out task, n_samples times, one group per task, then the
    eval post pipeline over the groups."""

    def __init__(self, spec: ExperimentSpec, run: RunHandle, store: Store,
                 max_inflight: int = 64) -> None:
        assert spec.eval is not None
        self.env_name = spec.eval.env or (spec.gen.env if spec.gen else None)
        if self.env_name is None:
            raise ValueError("eval on a run without gen must set eval.env")
        self.pipeline = spec.eval.post
        self.every = spec.eval.every
        self.n_samples = spec.eval.n_samples
        self.sampling = spec.gen.sampling if spec.gen else SamplingSpec()
        self.master = spec.seeds.master
        self.tasks = load_tasks(store, spec.eval.tasks)
        self.run = run
        self.max_inflight = max_inflight

    async def run_eval(self, update: int, pools: Pools) -> None:
        limiter = asyncio.Semaphore(self.max_inflight)

        async def one(task, sample_index: int):
            async with limiter:
                seed = derive(self.master, "eval", update, task.id, sample_index)
                client = EngineSampleClient(pools, self.sampling, seed)
                return await run_episode(self.env_name, task, client)

        groups = []
        for task in self.tasks:
            episodes = await asyncio.gather(
                *[one(task, s) for s in range(self.n_samples)])
            groups.append(Group(task.id, episodes))
        wave = Wave(groups)

        columns = await run_pipeline(self.pipeline, wave, pools, self.sampling,
                                     self.master, update, phase="eval-post")

        rows = []
        index = 0
        for group in wave.groups:
            for sample_index in range(len(group)):
                rows.append({
                    "task": group.key, "sample": sample_index,
                    "columns": {name: columns[name][index] for name in columns},
                })
                index += 1
        summary = {
            "update": update,
            "episodes": len(wave),
            "means": {name: math.fsum(values) / len(values)
                      for name, values in sorted(columns.items()) if values},
        }
        self.run.write_eval(update, "results.jsonl", _jsonl(rows))
        self.run.write_eval(update, "summary.json",
                            json.dumps(summary, sort_keys=True,
                                       separators=(",", ":")))
