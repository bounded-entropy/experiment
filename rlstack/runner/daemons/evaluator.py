"""The Evaluator: firewalled measurement, subscribed to ledger commits.

Its condition is a modulus over the commit bus: for each due update (every
N-th), await that ledger entry, pin the bundle it names, run the held-out
episodes and the EVAL post pipeline, write eval/<u>. Output goes only to the
eval/ section, which crash recovery never consults; an eval already written
is skipped on resume, and one lost to a crash is backfilled — the store holds
everything needed to re-serve ANY committed version (blobs + versions →
compile_bundle → the same content-addressed id), so pinning never depends on
what some earlier process happened to register.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Callable

from rlstack.data.stores.base import RunHandle, Store
from rlstack.data.trajectory import Group, Wave
from rlstack.policy.compile import Bundle, compile_bundle
from rlstack.registry import ADAPTERS
from rlstack.runner.sampling import EngineSampleClient, Routes
from rlstack.runner.interfaces import Engine
from rlstack.runner.arbiter import GpuArbiter
from rlstack.runner.post import run_pipeline
from rlstack.runner.daemons.base import Daemon
from rlstack.runner.seeds import derive
from rlstack.runner.signals import RunSignals
from rlstack.runner.sampling import load_tasks, run_episode
from rlstack.spec.specs import ExperimentSpec, SamplingSpec


class Evaluator(Daemon):
    def __init__(self, signals: RunSignals, arbiter: GpuArbiter, run: RunHandle, *,
                 spec: ExperimentSpec, store: Store, engine: Engine,
                 post_residents: tuple[Engine, ...],
                 routes_at: Callable[[Bundle], Routes],
                 max_inflight: int) -> None:
        super().__init__(signals, arbiter, run)
        self.post_residents = post_residents
        assert spec.eval is not None
        self.engine = engine
        bank = spec.policy.bank
        self.servable = sorted(n for n, a in bank.items()
                               if ADAPTERS.get(a.kind).instance.serving is not None)
        self.kinds = {n: bank[n].kind for n in self.servable}
        self.env_name = spec.eval.env or (spec.gen.env if spec.gen else None)
        if self.env_name is None:
            raise ValueError("eval on a run without gen must set eval.env")
        self.pipeline = spec.eval.post
        self.every = spec.eval.every
        self.n_samples = spec.eval.n_samples
        self.n_updates = spec.algo.schedule.n_updates
        self.sampling = spec.gen.sampling if spec.gen else SamplingSpec()
        self.master = spec.seeds.master
        self.tasks = load_tasks(store, spec.eval.tasks)
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
        """Every N-th update of the run."""
        return [u for u in range(1, self.n_updates + 1)
                if u % self.every == 0]

    # ---- pinning ------------------------------------------------------------

    def bundle_for(self, entry: dict) -> Bundle:
        """Recompile the committed bundle the entry names, from the store.

        Content addressing makes this exact: same blobs, same versions, same
        bundle_id — and registration is additive-idempotent, so re-adding is
        free. (v0 limit: a servable-but-frozen delta has no blob to read.)"""
        versions = {name: int(v) for name, v in entry["versions"].items()}
        payloads = {name: self.run.read_blob("adapters", name, versions[name])
                    for name in self.servable}
        bundle = compile_bundle(payloads, versions, self.servable, self.kinds)
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
        limiter = asyncio.Semaphore(self.max_inflight)

        async def one(task, sample_index: int):
            async with limiter:
                seed = derive(self.master, "eval", update, task.id, sample_index)
                client = EngineSampleClient(routes, self.sampling, seed)
                return await run_episode(self.env_name, task, client)

        groups = []
        for task in self.tasks:
            episodes = await asyncio.gather(
                *[one(task, s) for s in range(self.n_samples)])
            groups.append(Group(task.id, episodes))
        wave = Wave(groups)

        columns = await run_pipeline(self.pipeline, wave, routes, self.sampling,
                                     self.master, update, phase="eval-post")

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
        self.run.write_eval(update, "results.jsonl", _jsonl(rows))
        self.run.write_eval(update, "summary.json",
                            json.dumps(summary, sort_keys=True,
                                       separators=(",", ":")))


def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                   for r in rows)
