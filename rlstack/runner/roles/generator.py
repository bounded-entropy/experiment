"""The Generator: makes the rollout plan's waves, bounded by the lag buffer.

Its condition (`may_generate`) is the whole async-RL policy in one rule: a
rollout may be sampled once the update that FIRST consumes it is within the lag
buffer of the last commit. Which update that is comes from the train plan, so
one plan paces both runners without either calling the other (#59). WHICH
policy version actually serves each wave stays opportunistic within the buffer
and is recorded per turn — never prescribed, never re-derived.

A WAVE THAT SAMPLES UNDER A LIBRARY SET WAITS FOR IT (ADR 0019): a leaf role
or a task route naming `lib:<name>` makes the wave wait until that named
adapter is present in the store, and the bytes are handed to the pool before
the first request — the engine never reads a store. A plan that mentions no
library name takes none of these steps, and this runner is the runner it was.

The buffer comes from the CALLER (needs_of, ADR 0006 Part B): the lag when a
Trainer consumes these rollouts, None when nothing does. None is UNPACED — the
buffer bounds staleness against a moving policy, and in a generation-only run
the policy never moves.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping

from rlstack.data.plan import RunPlan, WavePlan
from rlstack.data.stores.base import RunHandle
from rlstack.data.trajectory import Task, wave_to_rows
from rlstack.policy.compile import Bundle
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.assemble import sample_wave
from rlstack.runner.roles.base import Runner
from rlstack.runner.interfaces import Engine
from rlstack.runner.meters import HostJournal
from rlstack.runner.names import (
    check_plan_names, names_ready, names_subdir, wave_lib_names,
)
from rlstack.runner.signals import RunSignals, store_work
from rlstack.runner.traffic import Routes
from rlstack.spec.specs import ExperimentSpec


class Generator(Runner):
    def __init__(self, signals: RunSignals, arbiter: Arbiter, run: RunHandle, *,
                 spec: ExperimentSpec, plan: RunPlan, due_at: Mapping[int, int],
                 tasks: Mapping[str, Task], engine: Engine,
                 routes_at: Callable[[Bundle], Routes],
                 initial_bundle: Bundle, max_inflight: int,
                 buffer: int | None,
                 refs=None, journal: HostJournal | None = None) -> None:
        super().__init__(signals, arbiter, run)
        check_plan_names(plan)
        # where a names wait says it is waiting, beside the log (ADR 0019)
        self.journal = journal
        self.engine = engine
        self.gen = spec.gen
        self.master = spec.seeds.master
        self.buffer = buffer
        self.plan = plan
        self.due_at = due_at
        self.tasks = tasks
        self.routes_at = routes_at
        self.initial_bundle = initial_bundle
        self.max_inflight = max_inflight
        # the ref reader Derive leaves mint through (assemble.sample_wave);
        # a plan without them never touches it
        self.refs = refs

    # ---- the acquisition condition (override to change the alternation) -----

    def consumed_by(self, index: int) -> int:
        """The update that first trains on rollout `index` — or, for a rollout
        no update names, the first update consuming any LATER rollout: a
        Derive chain's intermediate waves are un-referenced by construction
        (training reads only the loop-final wave), and an intermediate is due
        exactly when the wave it feeds is due. Only a rollout past EVERY
        reference is dead weight, paced as its own update."""
        if index in self.due_at:
            return self.due_at[index]
        later = [update for j, update in self.due_at.items() if j > index]
        return min(later) if later else index

    def may_generate(self, index: int) -> bool:
        """Rollout r waits for commit u-1-B, where u is the update that first
        consumes it: at most B waves in flight beyond the last committed.

        With NO buffer nothing consumes these rollouts (there is no Trainer to
        advance a ledger this would wait on), so the run is unpaced and every
        rollout may be sampled the moment the one before it is sealed.
        """
        if self.buffer is None:
            return True
        return self.committed() >= self.consumed_by(index) - 1 - self.buffer

    def already_sealed(self, index: int) -> bool:
        """Resume skips what is already on disk. A rollout is sealed or absent
        — never half — because write_rollout is atomic."""
        try:
            self.run.read_rollout(index)
        except FileNotFoundError:
            return False
        return True

    # ---- the runner ---------------------------------------------------------

    async def run_forever(self) -> None:
        for index in range(1, len(self.plan) + 1):
            if self.already_sealed(index):
                continue
            entry = self.plan.wave(index)
            if not isinstance(entry, WavePlan):
                raise TypeError(
                    f"rollout {index} is a WaveRef: a rollout plan MAKES "
                    f"trajectories, so it names them leaf by leaf")
            await self.signals.wait_for(lambda: self.may_generate(index))
            libraries = wave_lib_names(entry, self.tasks)
            if libraries:
                await names_ready(self.run.store, names_subdir(self.run),
                                  libraries, self.signals, journal=self.journal)
            async with self.arbiter.admit(self.engine):
                # routes_at restores the bundle on the pool by ASKING it —
                # sync wire verbs — so it runs off the loop (check_off_loop)
                routes = await asyncio.to_thread(self.routes_at,
                                                 self.newest_bundle())
                if libraries:
                    await self.hand_libraries(routes, libraries)
                wave = await sample_wave(
                    entry, index=index, tasks=self.tasks,
                    sampling=self.gen.sampling, routes=routes,
                    master=self.master, max_inflight=self.max_inflight,
                    reader=self.refs)
            self.run.write_rollout(index, wave_to_rows(wave))
            await self.signals.notify()

    async def hand_libraries(self, routes: Routes, names: tuple[str, ...]) -> None:
        """THE WAVE'S LIBRARY SETS, ON EVERY POOL IT ROUTES TO, as bytes (ADR
        0019, Q2): this runner reads the store and the engine is handed a
        payload, exactly as a bundle arrives. Once per wave and not once per
        run, for the reason `routes_at` states — residency is not durable (the
        engine's library is LRU-bounded, a restarted container starts empty)
        and `add_library` is idempotent, so restating is the restore. Asked
        from a thread: the pool may be another container."""
        subdir = names_subdir(self.run)
        engines = {id(engine): engine
                   for _, (engine, _) in sorted(routes.items())}
        for name in names:
            payload = await store_work(self.run.store.read_named, subdir, name)
            for engine in engines.values():
                await asyncio.to_thread(engine.add_library, name, payload)

    def newest_bundle(self) -> Bundle:
        """The freshest committed policy, as the store tells it — a pinning
        stub (id + versions); routes_at loads its payloads from storage."""
        tail = self.run.ledger_tail()
        if tail is None:
            return self.initial_bundle
        return Bundle.pin(tail["bundle_id"], tail["versions"])
