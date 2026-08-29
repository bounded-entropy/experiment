"""The Trainer: awaits data, trains, commits — the ledger's only writer.

One update, and the commit protocol that makes kill -9 safe at any point:

    await the plan's wave → await the SCORER's part → INLINE POST →
    write the merged postdata → flatten/broadcast/pack →
    forward_backward × microbatches × epochs → optim_step → bump →
    write blobs → register bundle → APPEND LEDGER (the commit point) → notify

Everything before the ledger line is unsealed: attach discards it and it
regenerates. The bundle is registered on the engine BEFORE the ledger line, so
any daemon reading the commit can pin its bundle_id immediately.

THE POST PHASE HAS TWO HALVES since the Scorer exists (#65). The processors
that address a pool are the Scorer's, run beside that pool and arriving here as
a postdata PART; the pool-less ones are arithmetic over columns and run here,
over the part as `given`. The Trainer still writes the one merged
postdata/<u>.json, so everything downstream — flatten, the ledger's column
means, the observer, the wave browser — reads exactly what it always did. A
pipeline with no pooled half awaits nothing and plans no Scorer, and this
daemon is byte-for-byte the daemon it was.

The gradient admits the LEARNER; the inline half admits nothing, because
occupying metal is what put a processor on the other side of the split.

Those boundaries are also the update's four measured phases — collect / post /
train / seal — lapped into an UpdateClock and journaled to the HOST after the
commit. Durations are wall clock, so they live in the host journal and nowhere
near a run directory. `post` now measures the WAIT for the part plus the inline
half, which is the number that says whether the Scorer is keeping up: a scorer
running far enough ahead makes it collapse to the arithmetic alone.
"""

from __future__ import annotations

import math
from typing import Callable

from rlstack.data.flatten import broadcast, flatten, pack
from rlstack.data.plan import RunPlan
from rlstack.runner.assemble import realize
from rlstack.runner.refs import RefReader
from rlstack.data.stores.base import RunHandle, bump
from rlstack.data.stores.retention import DEFAULT_RETENTION, RetentionPolicy
from rlstack.data.trajectory import wave_from_rows
from rlstack.policy.compile import Bundle, compile_bundle
from rlstack.registry import ADAPTER_TYPES
from rlstack.runner.traffic import Routes
from rlstack.runner.arbiter import GpuArbiter
from rlstack.runner.interfaces import Engine, Learner, TrainStats
from rlstack.runner.meters import HostJournal, UpdateClock
from rlstack.runner.post import run_pipeline
from rlstack.runner.daemons.base import Daemon
from rlstack.runner.daemons.scorer import SCORER
from rlstack.runner.signals import RunSignals
from rlstack.spec.flow import split_pipeline
from rlstack.spec.specs import ExperimentSpec, SamplingSpec


class Trainer(Daemon):
    def __init__(self, signals: RunSignals, arbiter: GpuArbiter, run: RunHandle, *,
                 spec: ExperimentSpec, plan: RunPlan, refs: RefReader,
                 engine: Engine,
                 learner: Learner, tenant: str,
                 post_residents: tuple[Engine, ...],
                 routes_at: Callable[[Bundle], Routes],
                 initial_bundle: Bundle,
                 initial_version: dict[str, int],
                 journal: HostJournal | None = None,
                 retention: RetentionPolicy = DEFAULT_RETENTION) -> None:
        super().__init__(signals, arbiter, run)
        self.tenant = tenant
        self.journal = journal
        self.retention = retention
        self.post_residents = post_residents
        self.spec = spec
        split = split_pipeline(spec.algo.post)
        self.pooled, self.inline = split.pooled, split.inline
        self.schedule = spec.algo.schedule
        self.sampling = spec.gen.sampling if spec.gen else SamplingSpec()
        self.plan = plan
        self.refs = refs
        self.engine = engine
        self.learner = learner
        self.routes_at = routes_at
        self.bundle = initial_bundle
        self.version = dict(initial_version)
        bank = spec.policy.bank
        self.trainable = sorted(n for n, a in bank.items() if a.trainable)
        self.servable = sorted(n for n, a in bank.items()
                               if ADAPTER_TYPES.get(a.adapter_type).instance.serving is not None)
        self.adapter_types = {n: bank[n].adapter_type for n in self.servable}

    # ---- the acquisition condition (override to change the alternation) -----

    def next_rows(self, update: int) -> list[dict] | None:
        """Update u's rows as its plan names them, or None while a leaf is
        still unsealed — the ONE await this daemon has (#59).

        Realized rows are written into this run's own waves/ before they are
        used, so a wave is self-contained however far its leaves reached.
        """
        try:
            return self.run.read_wave(update)          # already realized
        except FileNotFoundError:
            pass
        rows = realize(self.plan.wave(update), self.refs)
        if rows is None:
            return None
        self.run.write_wave(update, rows)
        return rows

    def scored(self, update: int) -> bool:
        """The Scorer's part for this update exists — the SECOND await, and it
        exists only when the pipeline has a pooled half.

        A predicate over the store, like every other daemon condition: the part
        file IS the handshake, and this daemon never learns that a Scorer is
        running, only that a file appeared.
        """
        return self.run.read_postdata_part(update, SCORER) is not None

    async def await_scored(self, update: int) -> dict[str, list]:
        """The pooled half's columns, in wave order — {} when there is no
        pooled half, which is when the Trainer is exactly what it was before
        the Scorer existed."""
        if not self.pooled:
            return {}
        await self.signals.wait_for(lambda: self.scored(update))
        return self.run.read_postdata_part(update, SCORER) or {}

    # ---- the daemon ---------------------------------------------------------

    async def run_forever(self) -> None:
        self.sweep_stale()          # converge what an interrupted process left
        for update in range(self.committed() + 1, len(self.plan) + 1):
            clock = UpdateClock()
            rows = await self.signals.wait_for(lambda: self.next_rows(update))
            wave = wave_from_rows(rows)
            clock.collected()

            # the pooled half, awaited on the store; then the inline half over
            # it; then the ONE merged file every reader downstream still reads
            given = await self.await_scored(update)
            async with self.arbiter.admit_all(self.post_residents):
                produced = await run_pipeline(
                    self.inline, wave, self.routes_at(self.bundle),
                    self.sampling, self.spec.seeds.master, update, given=given)
            postdata = {**given, **produced}
            self.run.write_postdata(update, postdata)
            clock.posted()

            tokenize = self.engine.tokenize
            flats = [flatten(t, tokenize) for t in wave.trajectories]
            docs = list(zip(flats, broadcast(postdata, flats)))

            # ONE WAVE IS ONE GRADIENT UPDATE: every microbatch accumulates
            # into a single step, and the ledger line below is written only
            # after all of them — so a checkpoint is never half a wave (#59).
            stats: list[TrainStats] = []
            async with self.arbiter.admit(self.learner):
                for batch in pack(docs, self.schedule.microbatch_tokens):
                    stats.append(
                        self.learner.forward_backward(self.tenant, batch))
                self.learner.optim_step(self.tenant)
                emitted = self.learner.emit(self.tenant)
            clock.trained()

            self.version = bump(self.version, self.trainable)
            self.bundle = compile_bundle(emitted.adapters, self.version,
                                         self.servable, self.adapter_types)
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
                "wave": {"trajectories": len(wave), "groups": len(wave.groups)},
                "post": _column_means(postdata),
                "train": _train_summary(stats),
            })
            self.sweep_stale()
            clock.sealed()
            self.journal_update(clock, update)
            await self.signals.notify()

    # ---- retention (at the commit, because that is when a version goes stale)

    def sweep_stale(self) -> None:
        """Free what the ledger has moved past — every optimizer blob below the
        tail, and nothing else the default policy can name.

        AT THE COMMIT, because that is the moment the previous update's moments
        stopped having a reader, and the moment the whole design already
        serializes on — no daemon, no clock, no second authority. AND WHEN THIS
        DAEMON STARTS, because a process killed between a commit and its sweep
        leaves one stale blob nobody would ever come back for: the run's LAST
        commit has no later commit to sweep after it. Sweeping on attach makes
        the store CONVERGE — after any commit and after any attach it holds
        every adapter and exactly one optim per delta — which is what keeps a
        run directory a pure function of (spec, code, data) even though bytes
        now leave it.

        NEVER FATAL. The update is committed; the ledger says so, and a failed
        unlink leaves bytes on a volume, which is the cheapest failure in the
        system. It also heals: the policy is a pure function of the ledger, so
        every later commit names the same blobs again, and `python -m rlstack
        sweep` is the backstop for the last one. What was freed is deliberately
        recorded NOWHERE — it is a fact about a directory, not about the
        experiment, and a run directory holding it would no longer be a pure
        function of (spec, code, data).
        """
        try:
            self.run.sweep(self.retention)
        except Exception:      # any backend failure; see the rule above
            pass

    # ---- the emission (observability; never a run-directory byte) -----------

    def journal_update(self, clock: UpdateClock, update: int) -> None:
        """One `update` event on the HOST's journal: how long this update took
        and where the time went. Wall clock may never enter a run directory —
        resume-equivalence is byte-identical run dirs — so the ledger line
        above carries the update's FACTS and this line carries its DURATION,
        in the one place durations are allowed. A run on no host emits
        nothing."""
        if self.journal is not None:
            self.journal.append(clock.row(self.tenant, update))


def _column_means(columns) -> dict[str, float]:
    """Scalar columns mean over trajectories; token_level columns over all
    their tokens — one ledger scalar either way."""
    out = {}
    for name, values in sorted(columns.items()):
        flat = [v for value in values
                for v in (value if isinstance(value, (list, tuple)) else [value])]
        if flat:
            out[name] = math.fsum(flat) / len(flat)
    return out


def _train_summary(stats: list[TrainStats]) -> dict:
    """One update's training facts. The bank's PROVIDED tensors are folded in
    first and the standing rails written over them, so the rails always own
    their own names; every other provided name lands here beside them, mean of
    the microbatch means — the same convention `loss` already uses, and exact
    for a quantity that is a function of the parameters alone (a latent KL is
    identical in every microbatch of one update)."""
    n = len(stats)
    return {
        **_provided_means(stats),
        "microbatches": n,
        "tokens": sum(s.tokens for s in stats),
        "loss": math.fsum(s.loss for s in stats) / n,
        "mean_ratio": math.fsum(s.mean_ratio for s in stats) / n,
        "logprob_gap": max(s.logprob_gap for s in stats),
        "grad_norm": max(s.grad_norm for s in stats),
    }


def _provided_means(stats: list[TrainStats]) -> dict[str, float]:
    """Each declared provided name, meaned across the update's microbatches."""
    names = sorted({name for s in stats for name in s.provided})
    return {name: math.fsum(s.provided[name] for s in stats if name in s.provided)
                  / sum(1 for s in stats if name in s.provided)
            for name in names}
