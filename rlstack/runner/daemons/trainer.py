"""The Trainer: awaits data, trains, commits — the only writer of the ledger.

One update's choreography and the commit protocol that makes kill -9 safe at
any point:

    await rollouts/<u> (the feed) → POST PIPELINE → write postdata →
    flatten/broadcast/pack → forward_backward × microbatches × epochs →
    optim_step → bump → write blobs → register bundle → APPEND LEDGER
    (the commit point) → notify

Everything before the ledger line is unsealed: crash recovery discards it on
attach and it regenerates. The bundle is registered on the engine BEFORE the
ledger line so any role that reads the commit (generator, evaluator) can pin
its bundle_id immediately. Under sleep colocation the post pipeline holds the
ENGINE (judges sample) and the backward holds the LEARNER — the lease's
sticky resident makes the common no-judge case cost one switch per update.
"""

from __future__ import annotations

import math
from typing import Callable

from rlstack.data.flatten import broadcast, flatten, pack
from rlstack.data.stores.base import RunHandle, bump
from rlstack.data.trajectory import wave_from_rows
from rlstack.policy.compile import Bundle, compile_bundle
from rlstack.registry import ADAPTERS
from rlstack.runner.client import Pools
from rlstack.runner.interfaces import Engine, Learner, TrainStats
from rlstack.runner.lease import ENGINE, LEARNER, Lease
from rlstack.runner.post import run_pipeline
from rlstack.runner.daemons.base import Daemon
from rlstack.runner.signals import RunSignals
from rlstack.runner.sources import WaveFeed
from rlstack.spec.specs import ExperimentSpec, SamplingSpec


class Trainer(Daemon):
    def __init__(self, signals: RunSignals, lease: Lease, run: RunHandle, *,
                 spec: ExperimentSpec, feed: WaveFeed, engine: Engine,
                 learner: Learner, pools_at: Callable[[Bundle], Pools],
                 initial_bundle: Bundle,
                 initial_version: dict[str, int]) -> None:
        super().__init__(signals, lease, run)
        self.spec = spec
        self.schedule = spec.algo.schedule
        self.sampling = spec.gen.sampling if spec.gen else SamplingSpec()
        self.feed = feed
        self.engine = engine
        self.learner = learner
        self.pools_at = pools_at
        self.bundle = initial_bundle
        self.version = dict(initial_version)
        bank = spec.policy.bank
        self.trainable = sorted(n for n, a in bank.items() if a.trainable)
        self.servable = sorted(n for n, a in bank.items()
                               if ADAPTERS.get(a.kind).instance.serving is not None)
        self.kinds = {n: bank[n].kind for n in self.servable}

    # ---- the acquisition condition (override to change the alternation) -----

    def next_rows(self, update: int) -> list[dict] | None:
        """Data for update u, from this run's own rollouts/ (the feed makes
        storage-backed data appear there; the Generator makes live data)."""
        return self.feed.obtain(update)

    # ---- the daemon ---------------------------------------------------------

    async def run_forever(self) -> None:
        for update in range(self.committed() + 1, self.schedule.n_updates + 1):
            rows = await self.signals.wait_for(lambda: self.next_rows(update))
            wave = wave_from_rows(rows)

            async with self.lease.held(ENGINE):    # postprocessors may sample
                postdata = await run_pipeline(
                    self.spec.algo.post, wave, self.pools_at(self.bundle),
                    self.sampling, self.spec.seeds.master, update)
            self.run.write_postdata(update, postdata)

            tokenize = self.engine.tokenize
            flats = [flatten(t, tokenize) for t in wave.trajectories]
            docs = list(zip(flats, broadcast(postdata, flats)))

            stats: list[TrainStats] = []
            async with self.lease.held(LEARNER):
                for _ in range(self.schedule.epochs_per_wave):
                    for batch in pack(docs, self.schedule.microbatch_tokens):
                        stats.append(self.learner.forward_backward(batch))
                self.learner.optim_step()
                emitted = self.learner.emit()

            self.version = bump(self.version, self.trainable)
            self.bundle = compile_bundle(emitted.adapters, self.version,
                                         self.servable, self.kinds)
            for name in self.trainable:
                self.run.write_blob("adapters", name, self.version[name],
                                    emitted.adapters[name])
                self.run.write_blob("optim", name, self.version[name],
                                    emitted.optim[name])
            self.engine.add_bundle(self.bundle)    # registered BEFORE the commit
            self.run.append_ledger({
                "update": update,
                "versions": dict(self.version),
                "bundle_id": self.bundle.bundle_id,
                "wave": {"rollouts": len(wave), "groups": len(wave.groups)},
                "post": _column_means(postdata),
                "train": _train_summary(stats),
            })
            await self.signals.notify()


def _column_means(columns) -> dict[str, float]:
    return {name: math.fsum(values) / len(values)
            for name, values in sorted(columns.items()) if values}


def _train_summary(stats: list[TrainStats]) -> dict:
    n = len(stats)
    return {
        "microbatches": n,
        "tokens": sum(s.tokens for s in stats),
        "loss": math.fsum(s.loss for s in stats) / n,
        "mean_ratio": math.fsum(s.mean_ratio for s in stats) / n,
        "logprob_gap": max(s.logprob_gap for s in stats),
        "grad_norm": max(s.grad_norm for s in stats),
    }
