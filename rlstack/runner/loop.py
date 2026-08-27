"""The runner: Phase 0 (identity), Phase 1 (idempotent setup), Phase 2 (daemons).

Phase 2 is a blackboard, not a choreography: plan_daemons derives one daemon
per GPU responsibility from the spec — a Generator iff trajectories are live, the
Trainer always, an Evaluator iff eval is declared — and they run concurrently,
synchronized ONLY through the store (signals.py) and throttled onto shared
metal by leases (lease.py). Nobody calls anybody:

    Generator   awaits commit w-1-B      → writes waves/<w>
    Trainer     awaits waves/<u>      → post, train, blobs, LEDGER (commit)
    Evaluator   awaits commits (mod N)   → writes eval/<u>

The ledger is the commit bus, waves/ is the data bus, and crash recovery
(data/stores/) plus the seed tree make the whole thing attachable: resume =
re-run Phases 0-1, then the daemons pick up from the ledger tail.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from rlstack.data.stores.base import Store
from rlstack.policy.compile import Bundle, compile_bundle
from rlstack.policy.siteschema import SiteSchema, resolve
from rlstack.registry import ADAPTERS, code_hashes
from rlstack.runner.client import Pools
from rlstack.runner.interfaces import Engine, Learner
from rlstack.runner.lease import LeaseMap, leases_for
from rlstack.runner.daemons import Daemon, Evaluator, Generator, Trainer
from rlstack.runner.signals import RunSignals
from rlstack.runner.sources import feed_for
from rlstack.runner.waves import load_tasks
from rlstack.spec.canonical import canonical_json, run_id
from rlstack.spec.specs import ExperimentSpec, WarmStart
from rlstack.spec.validate import (
    SpecError, check_sites_reachable_on, site_space, validate_or_raise,
)


@dataclass(frozen=True)
class RunReport:
    """What run_experiment hands back."""

    run_id: str
    updates_completed: int
    resumed_from: int | None


def run_experiment(spec: ExperimentSpec, schema: SiteSchema, store: Store,
                   engines: Engine | Mapping[str, Engine], learner: Learner,
                   max_inflight: int = 64) -> RunReport:
    """Submit and drive one experiment to completion. Safe to call again on the
    same spec: identical identity attaches and continues (or no-ops if done).

    `engines` is one Engine (used as the "main" policy pool) or {pool: Engine}.
    """
    return asyncio.run(
        run_experiment_async(spec, schema, store, engines, learner, max_inflight))


async def run_experiment_async(
        spec: ExperimentSpec, schema: SiteSchema, store: Store,
        engines: Engine | Mapping[str, Engine], learner: Learner,
        max_inflight: int = 64) -> RunReport:
    """The async form of run_experiment — the multi-tenant entry.

    The multi-tenancy invariant (Engine protocol) is only expressible when
    several experiments share ONE event loop around one resident engine:
    gather() any number of these on the same Engine and their bundles coexist,
    each request pinning its own. The sync wrapper is the one-experiment
    convenience; the Phase-C resident daemon drives this form directly.
    """
    if spec.algo is None:
        raise NotImplementedError("B1 runs training specs: algo required")
    engine_map: dict[str, Engine] = (
        dict(engines) if isinstance(engines, Mapping) else {"main": engines})
    if "main" not in engine_map:
        raise ValueError(f"no 'main' engine pool; pools: {sorted(engine_map)}")

    # ---- Phase 0: identity — computed, never typed (I3) ----------------------
    validate_or_raise(spec, schema)
    # reachability is a build fact, not a spec fact: ask the serving pool's
    # engine for its inventory and hold every served adapter against it
    space = site_space(spec, schema)
    reachability_issues = check_sites_reachable_on(
        spec, schema, "main", engine_map["main"].reachability(space))
    if reachability_issues:
        raise SpecError(reachability_issues)
    hashes = code_hashes(spec)
    # both forms are content-addressed: live → the task file, else the source
    data_fingerprint = spec.gen.tasks if spec.gen is not None else spec.trajectories.source
    rid = run_id(spec, hashes, data_fingerprint)
    run = store.open_run(rid, manifest={
        "run_id": rid,
        "spec": canonical_json(spec),
        "code": hashes,
        "data": data_fingerprint,
        "schema": schema.fingerprint(),   # same base name, different schema → loud
        "parent": spec.init.policy if spec.init is not None else None,
    })

    # ---- Phase 1: idempotent setup ------------------------------------------
    bank = spec.policy.bank
    trainable = sorted(name for name, a in bank.items() if a.trainable)
    servable = sorted(name for name, a in bank.items()
                      if ADAPTERS.get(a.kind).instance.serving is not None)
    kinds = {name: bank[name].kind for name in servable}
    resolved = {name: resolve(space, a.site) for name, a in bank.items()}
    learner.install(spec, resolved)

    policy_version = {name: 0 for name in bank}
    resumed_from: int | None = None
    tail = run.ledger_tail()
    if tail is not None:
        policy_version = {name: int(v) for name, v in tail["versions"].items()}
        learner.load(
            adapters={name: run.read_blob("adapters", name, policy_version[name])
                      for name in trainable},
            optim={name: run.read_blob("optim", name, policy_version[name])
                   for name in trainable},
        )
        resumed_from = int(tail["update"])
    elif spec.init is not None:
        _warm_start(spec.init, bank_names=set(bank), trainable=trainable,
                    store=store, learner=learner)

    emitted = learner.emit()
    bundle = compile_bundle(emitted.adapters, policy_version, servable, kinds)
    engine_map["main"].add_bundle(bundle)

    # non-policy pools serve their own base; register a base bundle once each
    base_bundles = {name: Bundle(f"bundle:base:{name}", {}, {})
                    for name in engine_map if name != "main"}
    for name, base_bundle in base_bundles.items():
        engine_map[name].add_bundle(base_bundle)

    def pools_at(current: Bundle) -> Pools:
        return {name: (eng, current if name == "main" else base_bundles[name])
                for name, eng in engine_map.items()}

    # ---- Phase 2: the blackboard --------------------------------------------
    daemons = plan_daemons(spec, run=run, store=store, engine_map=engine_map,
                           learner=learner, pools_at=pools_at,
                           initial_bundle=bundle,
                           initial_version=policy_version,
                           max_inflight=max_inflight)
    try:
        async with asyncio.TaskGroup() as group:
            for daemon in daemons:
                group.create_task(daemon.run_forever())
    except ExceptionGroup as failures:
        raise failures.exceptions[0] from None

    return RunReport(run_id=rid, updates_completed=spec.algo.schedule.n_updates,
                     resumed_from=resumed_from)


def plan_daemons(spec: ExperimentSpec, *, run, store, engine_map, learner,
                 pools_at, initial_bundle, initial_version,
                 max_inflight) -> list[Daemon]:
    """The spec already declares the daemons; this reads them off.

    live trajectories → a Generator writes the data bus; eval declared → an
    Evaluator watches the commit bus; the Trainer always. Leases come from
    GpuConfig (sleep-sharing groups share an ExclusiveLease); each daemon's
    acquisition condition is its own overridable method.
    """
    signals = RunSignals()
    leases: LeaseMap = leases_for(spec)
    daemons: list[Daemon] = [
        Trainer(signals, leases.for_learner(), run,
                spec=spec, feed=feed_for(spec, store, run),
                engine=engine_map["main"], learner=learner, pools_at=pools_at,
                initial_bundle=initial_bundle, initial_version=initial_version),
    ]
    if spec.trajectories.source == "live":
        daemons.append(Generator(
            signals, leases.for_pool("main"), run,
            spec=spec, tasks=load_tasks(store, spec.gen.tasks),
            pools_at=pools_at, initial_bundle=initial_bundle,
            max_inflight=max_inflight))
    if spec.eval is not None:
        daemons.append(Evaluator(
            signals, leases.for_pool(spec.eval.pool), run,
            spec=spec, store=store, engine=engine_map["main"],
            pools_at=pools_at, max_inflight=max_inflight))
    return daemons


def _warm_start(init: WarmStart, *, bank_names: set[str], trainable: list[str],
                store: Store, learner: Learner) -> None:
    """Load another run's sealed deltas (renamed via init.map) into this learner."""
    if not init.policy.startswith("store://"):
        raise NotImplementedError("B1 warm-starts from store:// runs only")
    address = init.policy[len("store://"):]
    parent_id, _, version_text = address.partition("@")
    version = int(version_text)
    parent = store.open_run(parent_id)

    source_name = {this: source for source, this in init.map.items()}
    adapters: dict[str, bytes] = {}
    optim: dict[str, bytes] = {}
    for name in bank_names:
        source = source_name.get(name, name)
        try:
            adapters[name] = parent.read_blob("adapters", source, version)
        except FileNotFoundError:
            continue  # no sealed state for this delta: it starts fresh
        if init.optim == "load" and name in trainable:
            optim[name] = parent.read_blob("optim", source, version)
    learner.load(adapters, optim if init.optim == "load" else None)
