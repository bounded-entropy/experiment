"""The Generator: makes the rollout plan's waves, bounded by the lag buffer.

Its condition (`may_generate`) is the whole async-RL policy in one rule: a
rollout may be sampled once the update that FIRST consumes it is within the lag
buffer of the last commit. Which update that is comes from the train plan, so
one plan paces both daemons without either calling the other (#59). WHICH
policy version actually serves each wave stays opportunistic within the buffer
and is recorded per turn — never prescribed, never re-derived.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from rlstack.data.plan import RunPlan, WavePlan
from rlstack.data.stores.base import RunHandle
from rlstack.data.trajectory import Task, wave_to_rows
from rlstack.policy.compile import Bundle
from rlstack.runner.arbiter import GpuArbiter
from rlstack.runner.assemble import sample_wave
from rlstack.runner.daemons.base import Daemon
from rlstack.runner.interfaces import Engine
from rlstack.runner.signals import RunSignals
from rlstack.runner.traffic import Routes
from rlstack.spec.specs import ExperimentSpec


class Generator(Daemon):
    def __init__(self, signals: RunSignals, arbiter: GpuArbiter, run: RunHandle, *,
                 spec: ExperimentSpec, plan: RunPlan, due_at: Mapping[int, int],
                 tasks: Mapping[str, Task], engine: Engine,
                 routes_at: Callable[[Bundle], Routes],
                 initial_bundle: Bundle, max_inflight: int,
                 refs=None) -> None:
        super().__init__(signals, arbiter, run)
        self.engine = engine
        self.gen = spec.gen
        self.master = spec.seeds.master
        self.buffer = spec.algo.schedule.max_policy_lag
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
        consumes it: at most B waves in flight beyond the last committed."""
        return self.committed() >= self.consumed_by(index) - 1 - self.buffer

    def already_sealed(self, index: int) -> bool:
        """Resume skips what is already on disk. A rollout is sealed or absent
        — never half — because write_rollout is atomic."""
        try:
            self.run.read_rollout(index)
        except FileNotFoundError:
            return False
        return True

    # ---- the daemon ---------------------------------------------------------

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
            async with self.arbiter.admit(self.engine):
                wave = await sample_wave(
                    entry, index=index, tasks=self.tasks,
                    sampling=self.gen.sampling,
                    routes=self.routes_at(self.newest_bundle()),
                    master=self.master, max_inflight=self.max_inflight,
                    reader=self.refs)
            self.run.write_rollout(index, wave_to_rows(wave))
            await self.signals.notify()

    def newest_bundle(self) -> Bundle:
        """The freshest committed policy, as the store tells it — a pinning
        stub (id + versions); the engine already holds the payloads."""
        tail = self.run.ledger_tail()
        if tail is None:
            return self.initial_bundle
        return Bundle.pin(tail["bundle_id"], tail["versions"])
