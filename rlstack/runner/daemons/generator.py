"""The Generator: free-runs at the newest committed policy, bounded by the lag
buffer.

Its condition (`may_generate`) is the whole async-RL policy in one line: wave
w may be sampled once the trainer has committed update w-1-B, where B =
Schedule.max_policy_lag. B=0 is strict alternation (wave w waits for version
w-1 — on-policy); B=1 lets generation run one wave ahead of training. WHICH
version actually serves each wave is opportunistic within that bound and is
recorded per turn — never prescribed, never re-derived.
"""

from __future__ import annotations

from typing import Callable, Sequence

from rlstack.data.stores.base import RunHandle
from rlstack.data.trajectory import Task, wave_to_rows
from rlstack.policy.compile import Bundle
from rlstack.runner.sampling import Routes
from rlstack.runner.arbiter import GpuArbiter
from rlstack.runner.interfaces import Engine
from rlstack.runner.daemons.base import Daemon
from rlstack.runner.signals import RunSignals
from rlstack.runner.sampling import collect_wave
from rlstack.spec.specs import ExperimentSpec


class Generator(Daemon):
    def __init__(self, signals: RunSignals, arbiter: GpuArbiter, run: RunHandle, *,
                 spec: ExperimentSpec, tasks: Sequence[Task], engine: Engine,
                 routes_at: Callable[[Bundle], Routes],
                 initial_bundle: Bundle, max_inflight: int) -> None:
        super().__init__(signals, arbiter, run)
        self.engine = engine
        self.gen = spec.gen
        self.schedule = spec.algo.schedule
        self.master = spec.seeds.master
        self.buffer = spec.algo.schedule.max_policy_lag
        self.tasks = tasks
        self.routes_at = routes_at
        self.initial_bundle = initial_bundle
        self.max_inflight = max_inflight

    # ---- the acquisition condition (override to change the alternation) -----

    def may_generate(self, wave_index: int) -> bool:
        """Wave w waits for commit w-1-B: at most B waves in flight beyond
        the last committed update."""
        return self.committed() >= wave_index - 1 - self.buffer

    def newest_bundle(self) -> Bundle:
        """The freshest committed policy, as the store tells it — a pinning
        stub (id + versions); the engine already holds the payloads."""
        tail = self.run.ledger_tail()
        if tail is None:
            return self.initial_bundle
        return Bundle.pin(tail["bundle_id"], tail["versions"])

    # ---- the daemon ---------------------------------------------------------

    async def run_forever(self) -> None:
        for wave_index in range(self.committed() + 1,
                                self.schedule.n_updates + 1):
            await self.signals.wait_for(lambda: self.may_generate(wave_index))
            async with self.arbiter.admit(self.engine):
                wave = await collect_wave(
                    wave_index,
                    env_name=self.gen.env,
                    sampling=self.gen.sampling,
                    tasks=self.tasks,
                    group_size=self.schedule.group_size,
                    trajectories_per_wave=self.schedule.trajectories_per_wave,
                    routes=self.routes_at(self.newest_bundle()),
                    master=self.master,
                    max_inflight=self.max_inflight,
                )
            self.run.write_wave(wave_index, wave_to_rows(wave))
            await self.signals.notify()
