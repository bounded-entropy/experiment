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

import time
from collections.abc import Awaitable, Callable, Mapping

from rlstack.data.plan import RunPlan, WavePlan
from rlstack.data.stores.base import RunHandle
from rlstack.data.trajectory import Task, wave_to_rows
from rlstack.policy.compile import Bundle
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.assemble import sample_wave
from rlstack.runner.roles.base import Runner
from rlstack.runner.meters import HostJournal
from rlstack.runner.names import (
    check_plan_names, names_ready, names_subdir, wave_lib_names,
)
from rlstack.runner.pool import Pool, admitted
from rlstack.runner.signals import RunSignals, store_work
from rlstack.runner.traffic import Routes
from rlstack.spec.specs import ExperimentSpec


class Generator(Runner):
    def __init__(self, signals: RunSignals, arbiter: Arbiter, run: RunHandle, *,
                 spec: ExperimentSpec, plan: RunPlan, due_at: Mapping[int, int],
                 tasks: Mapping[str, Task], pool: Pool,
                 routes_at: Callable[[Bundle], Awaitable[Routes]],
                 initial_bundle: Bundle, max_inflight: int,
                 buffer: int | None,
                 refs=None, journal: HostJournal | None = None) -> None:
        super().__init__(signals, arbiter, run)
        check_plan_names(plan)
        # where a names wait says it is waiting, beside the log (ADR 0019)
        self.journal = journal
        # the policy pool — every replica of `main` (ADR 0022): what a wave
        # is admitted onto, and whose window each rollout row journals
        self.pool = pool
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
            asked = time.time()
            trace: list[dict] | None = [] if self.journal is not None else None
            async with admitted(self.arbiter, [self.pool]):
                was_admitted = time.time()
                # routes_at restores the bundle on every live replica by
                # ASKING it — sync wire verbs, each from a thread
                routes = await self.routes_at(self.newest_bundle())
                if libraries:
                    await self.hand_libraries(routes, libraries)
                routed = time.time()
                wave = await sample_wave(
                    entry, index=index, tasks=self.tasks,
                    sampling=self.gen.sampling, routes=routes,
                    master=self.master, max_inflight=self.inflight_bound(),
                    reader=self.refs, trace=trace)
            sampled = time.time()
            self.run.write_rollout(index, wave_to_rows(wave))
            if trace is not None:
                await store_work(self.journal.append, rollout_row(
                    self.run.run_id, index, trace, asked=asked,
                    admitted=was_admitted, routed=routed, sampled=sampled,
                    sealed=time.time(), max_inflight=self.inflight_bound(),
                    pools={name: pool.window() for name, (pool, _)
                           in sorted(routes.items()) if isinstance(pool, Pool)}))
            await self.signals.notify()

    def inflight_bound(self) -> int:
        """EPISODES IN FLIGHT, PER REPLICA (ADR 0022): `max_inflight` bounds
        what one engine is asked to batch, so a pool of N replicas runs N
        times as many episodes at once — a bound that stayed per run would
        leave every added replica idle. Scheduling, outside identity (I5):
        completion order never reaches the bytes (#53)."""
        return self.max_inflight * len(self.pool.replicas)

    async def hand_libraries(self, routes: Routes, names: tuple[str, ...]) -> None:
        """THE WAVE'S LIBRARY SETS, ON EVERY POOL IT ROUTES TO, as bytes (ADR
        0019, Q2): this runner reads the store and the engine is handed a
        payload, exactly as a bundle arrives. Once per wave and not once per
        run, for the reason `routes_at` states — residency is not durable (the
        engine's library is LRU-bounded, a restarted container starts empty)
        and `add_library` is idempotent, so restating is the restore. Every
        LIVE REPLICA of each pool gets them, and a replica revived during the
        wave is handed them before it is readmitted (ADR 0022)."""
        subdir = names_subdir(self.run)
        payloads = {name: await store_work(self.run.store.read_named, subdir, name)
                    for name in names}
        pools = {id(pool): pool for _, (pool, _) in sorted(routes.items())}
        for pool in pools.values():
            await pool.hand_libraries(payloads)

    def newest_bundle(self) -> Bundle:
        """The freshest committed policy, as the store tells it — a pinning
        stub (id + versions); routes_at loads its payloads from storage."""
        tail = self.run.ledger_tail()
        if tail is None:
            return self.initial_bundle
        return Bundle.pin(tail["bundle_id"], tail["versions"])


def rollout_row(run_id: str, index: int, episodes: list[dict], *, asked: float,
                admitted: float, routed: float, sampled: float, sealed: float,
                max_inflight: int, pools: Mapping[str, dict] | None = None) -> dict:
    """One rollout wave as its journal event: the wall clock of its phases —
    waiting for the engine (`admit`: the arbiter's switch, a wake included),
    restoring its bundle and libraries (`route`), sampling (`sample`) and
    sealing (`seal`) — and, per episode in plan order, when it ran and every
    sample call it made (pool, route, start, seconds, tokens, finish).
    Achieved concurrency and the tail are read off these intervals; the
    cadence is every wave, the size bounded by the wave's calls. `pools` is
    each routed pool's window over the wave (ADR 0022): per replica its
    requests, tokens/s, affinity hits and keys, and the key table's moves."""
    return {"event": "rollout", "t": sealed, "run_id": run_id, "rollout": index,
            "seconds": sealed - asked, "cadence": "every rollout wave",
            "max_inflight": max_inflight,
            "phases": {"admit": admitted - asked, "route": routed - admitted,
                       "sample": sampled - routed, "seal": sealed - sampled},
            "tokens": sum(call["tokens"] for row in episodes for call in row["calls"]),
            "pools": dict(pools or {}),
            "episodes": episodes}
