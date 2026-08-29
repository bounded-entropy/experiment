"""The runner: Phase 0 (identity), Phase 1 (idempotent setup), Phase 2 (daemons).

Phase 2 is a blackboard, not a choreography: plan_daemons derives one daemon
per GPU responsibility from the spec — a Generator iff trajectories are live,
the Trainer always, an Evaluator iff eval is declared — and they run
concurrently, synchronized ONLY through the store (signals.py) and admitted
onto shared metal by the arbiter. Nobody calls anybody: the ledger is the
commit bus and waves/ the data bus.

Resume is re-running Phases 0-1 — attach discards everything the ledger never
committed, and the daemons pick up from the ledger tail.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from rlstack.data.stores.base import RunHandle, Store
from rlstack.policy.compile import Bundle, compile_bundle
from rlstack.policy.siteschema import SiteSchema, resolve
from rlstack.registry import ADAPTER_TYPES, POST, code_hashes
from rlstack.runner.daemons import Daemon, Evaluator, Generator, Trainer
from rlstack.runner.interfaces import Engine, Learner
from rlstack.runner.arbiter import GpuArbiter
from rlstack.runner.meters import HostJournal
from rlstack.data.plan import RunPlan, decode
from rlstack.runner.assemble import rollouts_needed
from rlstack.runner.refs import RefReader
from rlstack.runner.restore import restore_bundle_on, restore_tenant
from rlstack.runner.traffic import Routes, load_task_sets
from rlstack.runner.signals import RunSignals
from rlstack.spec.canonical import canonical_json, run_id
from rlstack.spec.flow import flow_graph
from rlstack.spec.specs import ExperimentSpec, Plans, PoolMember, WarmStart
from rlstack.runner.remote import RemotePool
from rlstack.spec.validate import (
    SpecError, check_members_match_their_shape, check_pools_serve_their_base,
    check_sites_reachable_on, site_space, traffic_pools, validate_or_raise,
)


@dataclass(frozen=True)
class RunReport:
    """What run_experiment hands back."""

    run_id: str
    updates_completed: int
    resumed_from: int | None


def run_experiment(spec: ExperimentSpec, schema: SiteSchema, store: Store,
                   engines: Engine | Mapping[str, Engine], learner: Learner,
                   max_inflight: int = 64,
                   arbiter: GpuArbiter | None = None) -> RunReport:
    """Submit and drive one experiment to completion. Safe to call again on the
    same spec: identical identity attaches and continues (or no-ops if done).

    `engines` is one Engine (used as the "main" policy pool) or {pool: Engine}.
    """
    return asyncio.run(
        run_experiment_async(spec, schema, store, engines, learner, max_inflight,
                             arbiter))


def data_fingerprint(spec: ExperimentSpec) -> str:
    """What the run READ, as one content-addressed string.

    The plans and the task sets are both cas uris, so their shas already are
    their contents: a plan that draws different tasks, or a task file whose
    rows changed, is a different experiment without anyone saying so (I3).
    """
    parts = [spec.plans.train, spec.plans.rollout or "-", spec.plans.eval or "-"]
    if spec.gen is not None:
        parts.extend(spec.gen.tasks)
    return "|".join(parts)


def experiment_identity(spec: ExperimentSpec, schema: SiteSchema) -> str:
    """Phase 0's identity, importable: validate, hash the referenced code,
    fingerprint the data, derive the run_id — computed, never typed (I3).
    The host journals under this id before the run opens."""
    validate_or_raise(spec, schema)
    hashes = code_hashes(spec)
    return run_id(spec, hashes, data_fingerprint(spec))


async def run_experiment_async(
        spec: ExperimentSpec, schema: SiteSchema, store: Store,
        engines: Engine | Mapping[str, Engine], learner: Learner,
        max_inflight: int = 64,
        arbiter: GpuArbiter | None = None,
        journal: HostJournal | None = None) -> RunReport:
    """The async form of run_experiment — the multi-tenant entry.

    The multi-tenancy invariant (I8) is only expressible when several
    experiments share ONE event loop around one resident engine: gather() any
    number of these on the same Engine and their bundles coexist, each request
    pinning its own — and on the same Learner, whose tenants coexist the same
    way. Pass the metal-owner's shared `arbiter` so colocated tenants alternate
    under ONE admission authority; None builds a private one, the
    single-experiment convenience.

    `journal` is the host's write door for this tenant's update timings
    (Host.submit supplies it): observability that never touches the run
    directory, and None — a run on no host — simply emits none.
    """
    if spec.algo is None:
        raise NotImplementedError(
            "this loop runs TRAINING specs, so algo is required: with no algo "
            "there is no Trainer, and the Trainer is the ledger's only writer "
            "— no update would ever commit. Generation-only runs need a "
            "committing Sealer daemon of their own")
    engine_map: dict[str, Engine] = (
        dict(engines) if isinstance(engines, Mapping) else {"main": engines})
    if "main" not in engine_map:
        raise ValueError(f"no 'main' engine pool; pools: {sorted(engine_map)}")

    # ---- Phase 0: identity — computed, never typed (I3) ----------------------
    validate_or_raise(spec, schema)
    # the engine map must cover every pool the spec's traffic can route to
    # (gen, eval, and each pipeline processor's declared judge/teacher pools)
    unmapped = sorted(traffic_pools(spec) - set(engine_map))
    if unmapped:
        raise ValueError(
            f"spec routes traffic to pools {unmapped} but the engine map "
            f"provides only {sorted(engine_map)}")
    # reachability is a build fact, not a spec fact: ask the serving pool's
    # engine for its inventory and hold every served adapter against it
    space = site_space(spec, schema)
    binding_issues = (
        check_sites_reachable_on(
            spec, schema, "main", engine_map["main"].reachability(space))
        + check_pools_serve_their_base(spec, engine_map)
        + check_members_match_their_shape(spec, engine_map, learner))
    if binding_issues:
        raise SpecError(binding_issues)
    hashes = code_hashes(spec)
    fingerprint = data_fingerprint(spec)
    rid = run_id(spec, hashes, fingerprint)
    run = store.open_run(rid, manifest={
        "run_id": rid,
        "spec": canonical_json(spec),
        "code": hashes,
        "data": fingerprint,
        "schema": schema.fingerprint(),   # same base name, different schema → loud
        "parent": spec.init.policy if spec.init is not None else None,
    })
    # the run describes its own observability: a UI reads THIS, never the
    # registries (spec/flow.py — same walk the submit gate validated with)
    run.write_dictionary(flow_graph(spec).to_json())
    # ...and its own shape: the plans it actually ran, copied in verbatim (I11)
    plans = load_plans(spec.plans, store)
    for kind, plan in plans.items():
        run.write_plan(kind, store.cas_get(getattr(spec.plans, kind)))

    # ---- Phase 1: idempotent setup ------------------------------------------
    bank = spec.policy.bank
    trainable = sorted(name for name, a in bank.items() if a.trainable)
    servable = sorted(name for name, a in bank.items()
                      if ADAPTER_TYPES.get(a.adapter_type).instance.serving is not None)
    adapter_types = {name: bank[name].adapter_type for name in servable}
    resolved = {name: resolve(space, a.site) for name, a in bank.items()}
    learner.install(rid, spec, resolved)
    if arbiter is None:
        arbiter = GpuArbiter()
    attach_residents(spec, engine_map, learner, arbiter)

    policy_version = {name: 0 for name in bank}
    resumed_from: int | None = None
    tail = run.ledger_tail()
    if tail is not None:
        policy_version = {name: int(v) for name, v in tail["versions"].items()}
        restore_tenant(learner, rid, policy_version, run.read_blob, trainable)
        resumed_from = int(tail["update"])
    elif spec.init is not None:
        _warm_start(spec.init, tenant=rid, bank_names=set(bank),
                    trainable=trainable, store=store, learner=learner)

    emitted = learner.emit(rid)
    write_frozen_blobs(run, emitted.adapters, policy_version,
                       set(servable) - set(trainable))
    bundle = compile_bundle(emitted.adapters, policy_version, servable,
                            adapter_types)
    engine_map["main"].add_bundle(bundle)

    # non-policy pools serve their own base; register a base bundle once each
    base_bundles = {name: Bundle(f"bundle:base:{name}", {}, {})
                    for name in engine_map if name != "main"}
    for name, base_bundle in base_bundles.items():
        engine_map[name].add_bundle(base_bundle)

    def routes_at(current: Bundle) -> Routes:
        """Where this wave's traffic goes — and the one place that makes sure
        the policy pool can still serve the version it is about to pin.

        Residency is not durable (a bounded pool evicts, a restarted container
        starts empty), so the route is established by ASKING and restoring on a
        miss. It runs once per wave, never per request.
        """
        restore_bundle_on(engine_map["main"], current, run.read_blob,
                          servable, adapter_types)
        return {name: (eng, current if name == "main" else base_bundles[name])
                for name, eng in engine_map.items()}

    # ---- Phase 2: the blackboard --------------------------------------------
    daemons = plan_daemons(spec, run=run, store=store, engine_map=engine_map,
                           learner=learner, routes_at=routes_at, plans=plans,
                           initial_bundle=bundle,
                           initial_version=policy_version,
                           max_inflight=max_inflight,
                           arbiter=arbiter, tenant=rid, journal=journal)
    try:
        async with asyncio.TaskGroup() as group:
            for daemon in daemons:
                group.create_task(daemon.run_forever())
    except ExceptionGroup as failures:
        raise failures.exceptions[0] from None

    return RunReport(run_id=rid, updates_completed=len(plans["train"]),
                     resumed_from=resumed_from)


def load_plans(declared: Plans, store: Store) -> dict[str, RunPlan]:
    """Resolve the declared plan uris. `train` is mandatory — it is the run's
    length; the other two are absent when the run makes or measures nothing."""
    out = {"train": decode(store.cas_get(declared.train))}
    for kind in ("rollout", "eval"):
        uri = getattr(declared, kind)
        if uri is not None:
            out[kind] = decode(store.cas_get(uri))
    return out


def write_frozen_blobs(run, adapters: Mapping[str, bytes],
                       policy_version: Mapping[str, int],
                       frozen_servable: set[str]) -> None:
    """Persist the servable deltas that never advance, once, at their version.

    The trainer writes a blob per TRAINABLE entry per update, so a frozen
    servable delta — served on every request, changed by nothing — would have no
    blob at all, and restore could not rebuild the bundles that carry it. It is
    written here instead: emitted once at Phase 1, at the version it will hold
    for the run's life. Without this the store is complete only for the deltas
    that happen to move, and a restore is total only by luck.
    """
    for name in sorted(frozen_servable):
        if not run.has_blob("adapters", name, policy_version[name]):
            run.write_blob("adapters", name, policy_version[name],
                           adapters[name])


def plan_daemons(spec: ExperimentSpec, *, run, store, engine_map, learner,
                 routes_at, plans, initial_bundle, initial_version,
                 max_inflight, arbiter, tenant,
                 journal: HostJournal | None = None) -> list[Daemon]:
    """The spec already declares the daemons; this reads them off.

    a rollout plan → a Generator makes its waves; an eval plan → an Evaluator
    watches the commit bus; the Trainer always. Each daemon admits
    the RESIDENTS its work occupies: the trainer's post phase the engines of
    its pipeline's declared pools, its train phase the learner, the generator
    and evaluator their serving pool's engine (plus the eval pipeline's).

    Only the Trainer takes the host journal: an update is the unit of progress
    the other daemons orbit, so its phase timings are the run's own clock.
    """
    signals = RunSignals()
    tasks = load_task_sets(store, spec.gen.tasks) if spec.gen is not None else {}
    daemons: list[Daemon] = [
        Trainer(signals, arbiter, run,
                spec=spec, plan=plans["train"], refs=RefReader(store, run),
                engine=engine_map["main"], learner=learner, tenant=tenant,
                post_residents=pipeline_residents(spec.algo.post, engine_map),
                routes_at=routes_at, journal=journal,
                initial_bundle=initial_bundle, initial_version=initial_version),
    ]
    if "rollout" in plans:
        daemons.append(Generator(
            signals, arbiter, run,
            spec=spec, plan=plans["rollout"],
            due_at=rollouts_needed(plans["train"].waves), tasks=tasks,
            engine=engine_map["main"], routes_at=routes_at,
            initial_bundle=initial_bundle, max_inflight=max_inflight))
    if spec.eval is not None:
        daemons.append(Evaluator(
            signals, arbiter, run,
            spec=spec, plan=plans["eval"], tasks=tasks,
            engine=engine_map[spec.eval.pool],
            post_residents=pipeline_residents(spec.eval.post, engine_map),
            routes_at=routes_at, max_inflight=max_inflight))
    return daemons


def pipeline_residents(pipeline, engine_map) -> tuple[Engine, ...]:
    """The distinct engine objects behind a post pipeline's declared pools —
    what its judges will occupy, so what its runner must admit."""
    pools = sorted({name for proc in pipeline if proc in POST
                    for name in POST.get(proc).pools})
    distinct = {id(engine_map[name]): engine_map[name] for name in pools}
    return tuple(distinct.values())


def attach_residents(spec: ExperimentSpec, engine_map, learner,
                     arbiter: GpuArbiter) -> None:
    """Register this experiment's metal with the (possibly shared) arbiter.

    Object-keyed and idempotent: two pools backed by one engine are ONE
    resident; a second experiment attaching the same engine is a no-op.
    Exclusive groups come from GpuGroup.sharing="sleep" — alternation exists
    only there. Declared fractions are reported into the arbiter's load, which
    the host's fit check refuses against; nothing polices the metal itself.

    A REMOTE pool attaches as a zero-footprint free resident: its metal is
    another host's partition, so local admission is bookkeeping — the real
    admission happens host-side, in HostService, at the serving partition —
    and its declared fraction is a carve hint that never counts here. A remote
    pool in a sleep group is refused: alternation is an intra-partition fact,
    so a sleep group's members must all live on one host.

    A sleep demand on metal the host ALREADY holds in an alternation group
    (a multi-regime host attached it at birth) is satisfied, not conflicting:
    the demand defers to the metal's own group. Only a sleep demand on
    always-resident metal still raises — that metal cannot alternate.
    """
    pool_group: dict[str, str | None] = {}
    pool_fraction: dict[str, float | None] = {}
    learner_group: str | None = None
    learner_fraction: float | None = None
    for gi, gpu_group in enumerate(spec.gpu_config.groups):
        sleeping = gpu_group.sharing == "sleep"
        for member in gpu_group.members:
            if isinstance(member, PoolMember):
                pool_group[member.name] = f"sleep:{gi}" if sleeping else None
                pool_fraction[member.name] = member.fraction
            else:
                learner_group = f"sleep:{gi}" if sleeping else None
                learner_fraction = member.fraction

    def deferred(obj: object, group: str | None) -> str | None:
        """The metal's own alternation group satisfies (and overrides) a
        sleep demand — the physical owner declared it at birth."""
        if arbiter.is_attached(obj) and arbiter.attached_group(obj) is not None:
            return None
        return group

    for name in sorted(engine_map):
        if isinstance(engine_map[name], RemotePool):
            if pool_group.get(name) is not None:
                raise ValueError(
                    f"pool {name!r} is served by another host but declared "
                    f"in a sleep group — alternation is an intra-partition "
                    f"fact; a sleep group's members must all live on one host")
            arbiter.attach(engine_map[name], label=f"remote:{name}")
            continue
        arbiter.attach(engine_map[name], label=f"engine:{name}",
                       group=deferred(engine_map[name], pool_group.get(name)),
                       fraction=pool_fraction.get(name))
    arbiter.attach(learner, label="learner",
                   group=deferred(learner, learner_group),
                   fraction=learner_fraction)


def _warm_start(init: WarmStart, *, tenant: str, bank_names: set[str],
                trainable: list[str], store: Store, learner: Learner) -> None:
    """Load another run's sealed deltas (renamed via init.map) into this learner."""
    if not init.policy.startswith("store://"):
        raise NotImplementedError(
            f"warm start reads another run's SEALED deltas out of a run "
            f"store, so init.policy must be a 'store://<run_id>@<version>' "
            f"address; got {init.policy!r}")
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
            optim[name] = _parent_moments(parent, source, version)
    learner.load(tenant, adapters, optim if init.optim == "load" else None)


def _parent_moments(parent: RunHandle, source: str, version: int) -> bytes:
    """The parent's Adam moments at the version this warm start names — or a
    refusal that says where they went.

    Retention keeps only the LEDGER TAIL's moments (data/stores/retention.py),
    because that is the only version `restore_tenant` can read. So
    `optim="load"` against a MID-RUN version of a swept parent finds nothing,
    and it must say so rather than surface as a bare missing file: warm start
    from the parent's tail (what `extend` does), or take its deltas with fresh
    moments.
    """
    try:
        return parent.read_blob("optim", source, version)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"no optimizer moments for {source}@{version} in run "
            f"{parent.run_id}: retention keeps only the ledger tail's moments, "
            f"so optim='load' works from the parent's LAST committed version "
            f"(its tail is {(parent.ledger_tail() or {}).get('versions')}) — "
            f"or use optim='fresh' to take the deltas alone") from None
