"""Where training data comes from: one base, one file per option.

    base.py    — WaveFeed: `obtain(update) -> rows | None`, the whole contract
    live.py    — LiveFeed: the Generator role writes rollouts/; this reads them
    replay.py  — ReplayFeed: another run's sealed rollouts (store://<run_id>)
    static.py  — StaticFeed: a fixed trajectory dataset (cas://<sha>)

The trainer neither knows nor cares which it has (I1: training consumes
sealed waves from its own run, full stop). `feed_for` dispatches on
RolloutSource.source; whether a Generator role EXISTS is the same dispatch,
made in plan_roles."""

from __future__ import annotations

from rlstack.data.stores.base import RunHandle, Store
from rlstack.runner.sources.base import WaveFeed
from rlstack.runner.sources.live import LiveFeed
from rlstack.runner.sources.replay import ReplayFeed
from rlstack.runner.sources.static import StaticFeed
from rlstack.spec.specs import ExperimentSpec


def feed_for(spec: ExperimentSpec, store: Store, run: RunHandle) -> WaveFeed:
    """The feed RolloutSource names. Validate has already vetted the spec."""
    source = spec.rollouts.source
    if source == "live":
        return LiveFeed(run)
    if source.startswith("store://"):
        return ReplayFeed(store, source, run)
    return StaticFeed(store, source, run,
                      rollouts_per_wave=spec.algo.schedule.rollouts_per_wave)
