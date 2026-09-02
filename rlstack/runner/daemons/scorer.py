"""The Scorer: score traffic, off the gradient's critical path.

The fourth daemon, and the only one that exists because of a MEASUREMENT: a
teacher column computed inside the Trainer's post phase makes every gradient
wait on a fan of sequential 32B prefills (#47). So the POOLED half of the post
pipeline moves here — beside the pools it addresses — and the two daemons meet
where every pair of daemons meets, at the store. The Scorer writes
`postdata/<u>.scorer.json`; the Trainer awaits exactly that file and merges it.
Nobody calls anybody.

The four beats, with this daemon's answers:

    await   wave u's rows — the ones the Trainer will train on, read from
            waves/<u> when it has written them and otherwise REALIZED
            read-only, which is what lets this daemon work an update ahead.
    admit   the engines behind its processors' declared pools.
    work    the pooled subset of the pipeline, through the same `run_pipeline`
            at the same seed paths, so a column is the same float whichever
            daemon computed it.
    write   the part, then notify.

THE VERSION-PINNING RULE: policy-pool traffic is scored under the bundle THE
WAVE RECORDED, never the newest committed one. Everything else about this
daemon is a rearrangement of work; that rule is the one place where running
beside the Trainer instead of inside it changes what has to be asked.
"""

from __future__ import annotations

from typing import Callable

from rlstack.data.plan import RunPlan
from rlstack.data.stores.base import RunHandle
from rlstack.data.trajectory import Wave, wave_from_rows
from rlstack.policy.compile import Bundle
from rlstack.registry import POST
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.assemble import realize
from rlstack.runner.daemons.base import Daemon
from rlstack.runner.interfaces import Engine
from rlstack.runner.post import run_pipeline
from rlstack.runner.refs import RefReader
from rlstack.runner.signals import RunSignals
from rlstack.runner.traffic import Routes
from rlstack.spec.flow import split_pipeline
from rlstack.spec.specs import ExperimentSpec, SamplingSpec

# Who wrote the part. One scorer owns the whole pooled half in v1, so this is a
# constant rather than a per-processor name: the Trainer awaits ONE file, and
# the day a second scorer exists is the day the parts need distinct producers.
SCORER = "scorer"


class Scorer(Daemon):
    def __init__(self, signals: RunSignals, arbiter: Arbiter, run: RunHandle,
                 *, spec: ExperimentSpec, plan: RunPlan, refs: RefReader,
                 residents: tuple[Engine, ...],
                 routes_at: Callable[[Bundle], Routes],
                 initial_bundle: Bundle) -> None:
        super().__init__(signals, arbiter, run)
        self.pipeline = split_pipeline(spec.algo.post).pooled
        self.pools = sorted({pool for name in self.pipeline
                             for pool in POST.get(name).pools})
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
        still unsealed — this daemon's ONE await, mirroring the Trainer's.

        READ-ONLY where the Trainer's is not. The Trainer WRITES the wave it
        realizes and one writer per artifact is the rule, so this reads
        waves/<u> when the Trainer has already put it there and otherwise
        realizes the same rows without writing them. `realize` is a pure
        function of the plan and the sealed rollouts, so the two agree byte for
        byte — and realizing rather than waiting is precisely what lets the
        Scorer get ahead. HOW FAR ahead is not this daemon's policy: realize
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
        under, as its own turns recorded it (I6).

        Scoring inline, the Trainer always held the current bundle and the
        question never came up. A daemon that may be an update ahead — or, after
        a crash, an update behind — holds no such thing, so it asks the DATA:
        every turn pins the bundle id and version map it was submitted under,
        one wave is sampled under one bundle, and `routes_at` restores that
        version out of the store if the pool no longer holds it (residency is
        not durable; the content-addressed id is the proof). Scoring the newest
        bundle instead would make a column depend on when the Scorer got round
        to it, which is the one thing a run directory may never depend on.

        A wave whose turns pin two bundles is refused rather than scored under
        whichever came first — a plan may mix leaves from several rollouts, and
        silently blending two policies into one column is exactly the wrong
        number this design exists to prevent.

        A pipeline that never addresses the policy pool needs no pin at all,
        and demanding one would make another run's replayed trajectories —
        whose versions this store does not hold — unscoreable by a teacher that
        never looks at the policy. So the rule applies where it bites and
        nowhere else.
        """
        if "main" not in self.pools:
            return self.initial_bundle
        pinned = {turn.bundle_id: turn.policy_version
                  for traj in wave.trajectories for turn in traj.turns}
        if len(pinned) != 1:
            raise ValueError(
                f"wave {update} pins {len(pinned)} bundle(s) "
                f"{sorted(pinned)}: a pipeline scoring the policy pool is "
                f"scored under the version the wave recorded, so its turns "
                f"must agree on one")
        bundle_id, versions = next(iter(pinned.items()))
        return Bundle.pin(bundle_id, {name: int(v) for name, v in versions.items()})

    # ---- the daemon ---------------------------------------------------------

    async def run_forever(self) -> None:
        for update in range(self.committed() + 1, len(self.plan) + 1):
            if self.already_scored(update):
                continue                      # written before a crash
            rows = await self.signals.wait_for(lambda: self.next_rows(update))
            wave = wave_from_rows(rows)
            async with self.arbiter.admit_all(self.residents):
                columns = await run_pipeline(
                    self.pipeline, wave,
                    self.routes_at(self.pinned_bundle(update, wave)),
                    self.sampling, self.master, update)
            self.run.write_postdata_part(update, SCORER, columns)
            await self.signals.notify()
