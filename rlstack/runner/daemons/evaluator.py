"""The Evaluator: firewalled measurement, subscribed to ledger commits.

Its condition is a modulus over the commit bus: for each due update, await that
ledger entry, pin the bundle it names, run the held-out episodes and the eval
pipeline, write eval/<u>. The firewall is the rule — output goes only to eval/,
which crash recovery never consults, so an eval already written is skipped on
resume and one lost to a crash simply backfills. Pinning never depends on what
some earlier process happened to register: content addressing lets the store
re-serve ANY committed version exactly.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Callable

from rlstack.data.plan import RunPlan, WavePlan
from rlstack.data.stores.base import RunHandle
from rlstack.runner.assemble import sample_wave
from rlstack.data.trajectory import Task, Wave
from rlstack.policy.compile import Bundle, compile_bundle
from rlstack.registry import ADAPTER_TYPES
from rlstack.runner.traffic import Routes
from rlstack.runner.interfaces import Engine
from rlstack.runner.arbiter import GpuArbiter
from rlstack.runner.post import run_pipeline
from rlstack.runner.daemons.base import Daemon
from rlstack.runner.signals import RunSignals
from rlstack.spec.specs import ExperimentSpec, SamplingSpec


class Evaluator(Daemon):
    def __init__(self, signals: RunSignals, arbiter: GpuArbiter, run: RunHandle, *,
                 spec: ExperimentSpec, plan: RunPlan,
                 tasks: Mapping[str, Task], engine: Engine,
                 post_residents: tuple[Engine, ...],
                 routes_at: Callable[[Bundle], Routes],
                 max_inflight: int) -> None:
        super().__init__(signals, arbiter, run)
        self.post_residents = post_residents
        assert spec.eval is not None
        self.engine = engine
        bank = spec.policy.bank
        self.servable = sorted(n for n, a in bank.items()
                               if ADAPTER_TYPES.get(a.adapter_type).instance.serving is not None)
        self.adapter_types = {n: bank[n].adapter_type for n in self.servable}
        self.plan = plan
        self.pipeline = spec.eval.post
        self.every = spec.eval.every
        self.sampling = spec.gen.sampling if spec.gen else SamplingSpec()
        self.master = spec.seeds.master
        self.tasks = tasks
        self.routes_at = routes_at
        self.max_inflight = max_inflight

    # ---- the acquisition condition (override to change the schedule) --------

    def committed_entry(self, update: int) -> dict | None:
        """The ledger entry for `update`, once the trainer has sealed it."""
        for entry in self.run.read_ledger():
            if int(entry["update"]) == update:
                return entry
        return None

    def due_updates(self) -> list[int]:
        """One update per wave of the eval plan, `every` updates apart.

        The plan's LENGTH is how many measurements this run takes — the same
        rule the train plan sets for how many updates it has, so neither is a
        count anyone types twice.
        """
        return [k * self.every for k in range(1, len(self.plan) + 1)]

    def wave_for(self, update: int) -> WavePlan:
        """The eval wave measured at `update` — the k-th, for the k-th point."""
        entry = self.plan.wave(update // self.every)
        if not isinstance(entry, WavePlan):
            raise TypeError(
                f"eval wave for update {update} is a WaveRef: eval MAKES its "
                f"trajectories, so it names them leaf by leaf")
        return entry

    # ---- pinning ------------------------------------------------------------

    def bundle_for(self, entry: dict) -> Bundle:
        """Recompile the committed bundle the entry names, out of the store.

        Content addressing makes it exact — same blobs, same versions, same
        bundle_id — and add_bundle is additive-idempotent, so re-adding costs
        nothing. (v0 limit: a servable-but-frozen delta has no blob to read.)"""
        versions = {name: int(v) for name, v in entry["versions"].items()}
        payloads = {name: self.run.read_blob("adapters", name, versions[name])
                    for name in self.servable}
        bundle = compile_bundle(payloads, versions, self.servable, self.adapter_types)
        if bundle.bundle_id != entry["bundle_id"]:
            raise RuntimeError(
                f"recompiled bundle {bundle.bundle_id} != committed "
                f"{entry['bundle_id']} for update {entry['update']}")
        return bundle

    # ---- the daemon ---------------------------------------------------------

    async def run_forever(self) -> None:
        for update in self.due_updates():
            entry = await self.signals.wait_for(
                lambda: self.committed_entry(update))
            if self.run.has_eval(update):
                continue                       # written before a crash
            pinned = self.bundle_for(entry)
            self.engine.add_bundle(pinned)
            async with self.arbiter.admit_all(
                    (self.engine, *self.post_residents)):
                await self._evaluate(update, self.routes_at(pinned))
            await self.signals.notify()

    async def _evaluate(self, update: int, routes: Routes) -> None:
        wave = await sample_wave(
            self.wave_for(update), index=update, tasks=self.tasks,
            sampling=self.sampling, routes=routes, master=self.master,
            phase="eval", max_inflight=self.max_inflight)
        columns = await run_pipeline(self.pipeline, wave, routes, self.sampling,
                                     self.master, update, phase="eval-post")
        rows, summary = self.reduce_in_task_order(update, wave, columns)
        self.run.write_eval(update, "results.jsonl", _jsonl(rows))
        self.run.write_eval(update, "summary.json",
                            json.dumps(summary, sort_keys=True,
                                       separators=(",", ":")))

    def reduce_in_task_order(
            self, update: int, wave: Wave,
            columns: dict[str, list]) -> tuple[list[dict], dict]:
        """The eval's two files, folded in WAVE order — which is task order.

        Every accumulation over a concurrently sampled eval runs over the
        wave, never over completions: float addition is not associative, so a
        mean summed in completion order would depend on the scheduler, and
        eval/ is inside the byte-identity contract resume-equivalence holds
        the run to."""
        rows, index = [], 0
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
        return rows, summary


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                   for r in rows)
