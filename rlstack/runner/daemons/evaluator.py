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

import asyncio
import json
import math
from typing import Callable

from rlstack.data.stores.base import RunHandle, Store
from rlstack.data.trajectory import Group, Task, Trajectory, Wave
from rlstack.policy.compile import Bundle, restore_bundle
from rlstack.registry import ADAPTER_TYPES
from rlstack.runner.traffic import EnginePoolClient, Routes
from rlstack.runner.interfaces import Engine
from rlstack.runner.arbiter import GpuArbiter
from rlstack.runner.post import run_pipeline
from rlstack.runner.daemons.base import Daemon
from rlstack.runner.seeds import derive
from rlstack.runner.signals import RunSignals
from rlstack.runner.traffic import load_tasks, run_episode
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
                               if ADAPTER_TYPES.get(a.adapter_type).instance.serving is not None)
        self.adapter_types = {n: bank[n].adapter_type for n in self.servable}
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
        """The committed bundle this entry names, restored from the store.

        Eval pins a version the run may have moved far past, so this is the
        canonical restore caller: the id in the ledger line plus the blobs at
        its version map, with the content-addressed id as the proof
        (policy/compile.py). The rebuild is exact, so re-registering it is
        additive and idempotent.
        """
        versions = {name: int(v) for name, v in entry["versions"].items()}
        return restore_bundle(versions, entry["bundle_id"], self.run.read_blob,
                              self.servable, self.adapter_types)

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
        wave = Wave(await self.sample_heldout(update, routes))
        columns = await run_pipeline(self.pipeline, wave, routes, self.sampling,
                                     self.master, update, phase="eval-post")
        rows, summary = self.reduce_in_task_order(update, wave, columns)
        self.run.write_eval(update, "results.jsonl", _jsonl(rows))
        self.run.write_eval(update, "summary.json",
                            json.dumps(summary, sort_keys=True,
                                       separators=(",", ":")))

    async def sample_heldout(self, update: int,
                             routes: Routes) -> list[Group]:
        """Every held-out (task, sample) episode at once, bounded by
        max_inflight — and grouped back in TASK order.

        Held-out tasks are independent, so they are launched together and the
        engine batches whatever arrives, under the same bound the Generator's
        wave runs under.

        Concurrency may not reach the bytes. Each episode's seed is derived
        from (task.id, sample_index) — fixed before it is scheduled, so WHICH
        episodes ran together cannot change what any of them sampled — and
        `gather` returns in ARGUMENT order, so the wave this builds is a
        function of self.tasks alone, never of who finished first."""
        limiter = asyncio.Semaphore(self.max_inflight)

        async def one(task: Task, sample_index: int) -> Trajectory:
            async with limiter:
                seed = derive(self.master, "eval", update, task.id, sample_index)
                client = EnginePoolClient(routes, self.sampling, seed)
                return await run_episode(self.env_name, task, client)

        jobs = [one(task, sample_index)
                for task in self.tasks
                for sample_index in range(self.n_samples)]
        episodes = await asyncio.gather(*jobs)
        n = self.n_samples
        return [Group(task.id, episodes[i * n:(i + 1) * n])
                for i, task in enumerate(self.tasks)]

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
