"""The Scorer: score traffic, off the gradient's critical path.

The fourth runner, and the only one that exists because of a MEASUREMENT: a
teacher column computed inside the Trainer's post phase makes every gradient
wait on a fan of sequential 32B prefills (#47). So the POOLED half of the post
pipeline moves here — beside the pools it addresses — and the two runners meet
where every pair of runners meets, at the store. The Scorer writes
`postdata/<u>.scorer.json`; the Trainer awaits exactly that file and merges it.
Nobody calls anybody.

The four beats, with this runner's answers:

    await   wave u's rows — the ones the Trainer will train on, read from
            waves/<u> when it has written them and otherwise REALIZED
            read-only, which is what lets this runner work an update ahead.
    admit   the engines behind its processors' declared pools.
    work    the pooled subset of the pipeline, through the same `run_pipeline`
            at the same seed paths, so a column is the same float whichever
            runner computed it.
    write   the part, then notify.

THE VERSION-PINNING RULE: policy-pool traffic is scored under the bundle THE
WAVE RECORDED, never the newest committed one. Everything else about this
runner is a rearrangement of work; that rule is the one place where running
beside the Trainer instead of inside it changes what has to be asked.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Callable

from rlstack.data.plan import RunPlan
from rlstack.data.stores.base import RunHandle
from rlstack.data.trajectory import Wave, wave_from_rows
from rlstack.policy.compile import Bundle
from rlstack.registry import POST
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.assemble import realize
from rlstack.runner.roles.base import Runner
from rlstack.runner.interfaces import Engine
from rlstack.runner.post import run_pipeline
from rlstack.runner.refs import RefReader
from rlstack.runner.signals import RunSignals
from rlstack.runner.traffic import Routes
from rlstack.spec.flow import split_pipeline
from rlstack.spec.specs import ExperimentSpec, SamplingSpec
from rlstack.spec.validate import serving_pools

# Who wrote the part. One scorer owns the whole pooled half in v1, so this is a
# constant rather than a per-processor name: the Trainer awaits ONE file, and
# the day a second scorer exists is the day the parts need distinct producers.
SCORER = "scorer"


class Scorer(Runner):
    def __init__(self, signals: RunSignals, arbiter: Arbiter, run: RunHandle,
                 *, spec: ExperimentSpec, plan: RunPlan, refs: RefReader,
                 residents: tuple[Engine, ...],
                 routes_at: Callable[[Bundle], Routes],
                 initial_bundle: Bundle) -> None:
        super().__init__(signals, arbiter, run)
        self.pipeline = split_pipeline(spec.algo.post).pooled
        self.pools = sorted({pool for name in self.pipeline
                             for pool in POST.get(name).pools})
        # the pools that serve the POLICY (ADR 0014): scoring through one of
        # them is scoring the policy, and takes the wave's recorded version
        self.serving = frozenset(serving_pools(spec))
        # the bank this run's bundles version: a turn pinned to another
        # bank (a teacher run's, a supervised placeholder's) is not a vote
        self.bank = frozenset(spec.policy.bank)
        self.residents = residents
        self.sampling = spec.gen.sampling if spec.gen else SamplingSpec()
        self.master = spec.seeds.master
        self.plan = plan
        self.refs = refs
        self.routes_at = routes_at
        self.initial_bundle = initial_bundle

    # ---- the acquisition condition (override to change the alternation) -----

    def next_rows(self, update: int) -> list[dict] | None:
        """Update u's rows as its plan names them, or None while a leaf is
        still unsealed — this runner's ONE await, mirroring the Trainer's.

        READ-ONLY where the Trainer's is not. The Trainer WRITES the wave it
        realizes and one writer per artifact is the rule, so this reads
        waves/<u> when the Trainer has already put it there and otherwise
        realizes the same rows without writing them. `realize` is a pure
        function of the plan and the sealed rollouts, so the two agree byte for
        byte — and realizing rather than waiting is precisely what lets the
        Scorer get ahead. HOW FAR ahead is not this runner's policy: realize
        answers None until the Generator has sealed the leaves, so the Scorer
        inherits the lag buffer that paces the Generator (#59) and needs no
        bound of its own.
        """
        try:
            return self.run.read_wave(update)
        except FileNotFoundError:
            return realize(self.plan.wave(update), self.refs)

    def already_scored(self, update: int) -> bool:
        """This update's part is already on disk, so resume skips it. A part is
        written atomically and swept if its update never committed: it is whole
        or absent, never half."""
        return self.run.read_postdata_part(update, SCORER) is not None

    # ---- pinning ------------------------------------------------------------

    def pinned_bundle(self, update: int, wave: Wave) -> Bundle:
        """THE VERSION-PINNING RULE: the policy version THIS WAVE was sampled
        under, as its own turns recorded it (I6) — and, for a wave whose
        sampled turns do not agree, the policy AS OF THE PREVIOUS UPDATE.

        Scoring inline, the Trainer always held the current bundle and the
        question never came up. A runner that may be an update ahead — or, after
        a crash, an update behind — holds no such thing, so it asks the DATA:
        every turn pins the bundle id and version map it was submitted under,
        and a wave sampled under one bundle is scored under it; `routes_at`
        restores that version out of the store if the pool no longer holds it
        (residency is not durable; the content-addressed id is the proof).
        Scoring the newest bundle instead would make a column depend on when
        the Scorer got round to it, which is the one thing a run directory may
        never depend on.

        A wave may hold turns pinned to DIFFERENT bundles of this bank, or
        none at all (ADR 0018): an arrival's text rows are supervised records with no
        policy behind them, and the dreams replayed beside them for the
        dreamer's step were sampled many updates ago. Such a wave is scored
        under the bundle the ledger committed at update u−1 — the policy just
        before this update, which is exactly what a probe "before the update"
        means, and a function of the update index alone, so a resumed Scorer
        recomputes the same column. Under a lag buffer of zero this is also
        the bundle a freshly sampled wave records, so no wave whose turns
        agree changes hands. `next_rows` waits for that commit.

        A pipeline that never addresses the policy pool needs no pin at all,
        and demanding one would make another run's replayed trajectories —
        whose versions this store does not hold — unscoreable by a teacher that
        never looks at the policy. So the rule applies where it bites and
        nowhere else.
        """
        if not self.serving.intersection(self.pools):
            return self.initial_bundle
        pinned = self.votes(wave)
        if len(pinned) == 1:
            bundle_id, versions = next(iter(pinned.items()))
            return Bundle.pin(bundle_id, {name: int(v) for name, v in versions.items()})
        return self.committed_bundle(update - 1)

    def votes(self, wave: Wave) -> dict[str, Mapping[str, int]]:
        """The bundles this wave's turns pin, counting only turns pinned to
        THIS run's bank: a row another run sealed (a base-dreams teacher, a
        supervised text) carries no policy of ours behind it and does not
        say which of ours to score under."""
        return {turn.bundle_id: turn.policy_version
                for traj in wave.trajectories for turn in traj.turns
                if turn.bundle_id.startswith("bundle:")
                and frozenset(turn.policy_version) == self.bank}

    def committed_bundle(self, update: int) -> Bundle:
        """The bundle the ledger committed at `update` (0 is the initial
        bundle), as a pinning stub; the caller has waited for the commit."""
        if update <= 0:
            return self.initial_bundle
        for line in self.run.read_ledger():
            if int(line["update"]) == update:
                return Bundle.pin(line["bundle_id"],
                                  {name: int(v) for name, v in line["versions"].items()})
        raise RuntimeError(f"update {update} is not on the ledger yet")

    def needs_previous_commit(self, wave: Wave) -> bool:
        """Does pinning this wave read the ledger at u−1? (Its turns do not
        agree on one bundle.)"""
        if not self.serving.intersection(self.pools):
            return False
        return len(self.votes(wave)) != 1

    # ---- the runner ---------------------------------------------------------

    async def run_forever(self) -> None:
        for update in range(self.committed() + 1, len(self.plan) + 1):
            if self.already_scored(update):
                continue                      # written before a crash
            rows = await self.signals.wait_for(lambda: self.next_rows(update))
            wave = wave_from_rows(rows)
            if self.needs_previous_commit(wave):
                # the pin is the ledger's bundle at u-1 (ADR 0018): wait for it
                await self.signals.wait_for(lambda: self.committed() >= update - 1)
            async with self.arbiter.admit_all(self.residents):
                routes = await asyncio.to_thread(     # sync asks, off the loop
                    self.routes_at, self.pinned_bundle(update, wave))
                columns = await run_pipeline(
                    self.pipeline, wave, routes,
                    self.sampling, self.master, update)
            self.run.write_postdata_part(update, SCORER, columns)
            await self.signals.notify()
