"""The runner: Phase 0 (identity), Phase 1 (idempotent setup), Phase 2 (daemons).

A RUN IS A SET OF DAEMON NEEDS, each naming its daemon, the plan it consumes
and the residents it admits (ADR 0006 Part B). `needs_of` reads the
experiment's needs off its spec — a Trainer iff there is an algo, a Scorer iff
the post pipeline addresses a pool, a Generator iff there is a rollout plan —
and everything after that is generic: Phase 1 sets up what the needs admit and
Phase 2 is one TaskGroup over them, run by `plan_daemons`.

Phase 2 is a blackboard, not a choreography: the daemons run concurrently,
synchronized ONLY through the store (signals.py) and admitted onto shared metal
by the arbiter. Nobody calls anybody: the ledger is the commit bus, waves/ the
data bus, and postdata parts the scoring bus.

Resume is re-running Phases 0-1 — attach discards everything the ledger never
committed, and the daemons pick up from the ledger tail (a run whose extent is
its rollouts has no ledger: its Generator picks up from what is sealed).
"""

from __future__ import annotations

import asyncio
import json
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rlstack.data.stores.base import RunHandle, Store
from rlstack.policy.compile import Bundle, compile_bundle
from rlstack.policy.siteschema import SiteSchema, resolve
from rlstack.registry import ADAPTER_TYPES, POST, code_hashes
from rlstack.runner.daemons import Daemon, Generator, Scorer, Trainer
from rlstack.runner.interfaces import (
    Emitted, Engine, EntryInstall, Learner, OptimSettings, Parameterization,
)
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.meters import HostJournal
from rlstack.data.plan import RunPlan, decode
from rlstack.runner.assemble import rollouts_needed
from rlstack.runner.refs import RefReader
from rlstack.runner.restore import restore_bundle_on, restore_tenant
from rlstack.runner.traffic import Routes, load_task_sets
from rlstack.runner.signals import RunSignals, store_work
from rlstack.spec.canonical import canonical_json, run_id
from rlstack.spec.flow import flow_graph, split_pipeline
from rlstack.spec.specs import ExperimentSpec, Plans, PoolMember, WarmStart
from rlstack.runner.remote import RemoteLearner, RemotePool
from rlstack.spec.validate import (
    SpecError, check_members_match_their_shape, check_pools_serve_their_base,
    check_sites_reachable_on, site_space, traffic_pools, validate_or_raise,
)


@dataclass(frozen=True)
class RunReport:
    """What run_experiment hands back: how far the run's EXTENT got.

    `completed` counts the extent's waves — updates committed for a run that
    trains, rollouts sealed for one that only generates — and `extent` names
    which, so a reader never has to guess what the number counts (ADR 0006
    Part B, Q7). Observability, never identity.
    """

    run_id: str
    completed: int
    extent: str
    resumed_from: int | None


@dataclass(frozen=True)
class DaemonNeed:
    """ONE DAEMON A RUN NEEDS, with the resources it occupies.

    A run is a set of these; the experiment is the set `needs_of` reads off an
    ExperimentSpec (ADR 0006 Part B). Stating a need as a VALUE rather than
    building the daemon straight away is what lets Phase 1 ask what this run
    requires — does anything here train? — before Phase 2 runs any of it.
    """

    daemon: type[Daemon]
    plan: str                          # "train" | "rollout": what it consumes
    pools: tuple[str, ...] = ()        # pool names whose engines it admits
    learner: bool = False              # admits the learner
    buffer: int | None = None          # Generator only: None = unpaced


def run_experiment(spec: ExperimentSpec, schema: SiteSchema, store: Store,
                   engines: Engine | Mapping[str, Engine],
                   learner: Learner | None,
                   max_inflight: int = 64,
                   arbiter: Arbiter | None = None,
                   subdir: str | None = None, resume: bool = False) -> RunReport:
    """Submit and drive one experiment to completion. Safe to call again on the
    same spec: identical identity attaches and continues (or no-ops if done).

    `engines` is one Engine (used as the "main" policy pool) or {pool: Engine}.
    """
    return asyncio.run(
        run_experiment_async(spec, schema, store, engines, learner, max_inflight,
                             arbiter, subdir=subdir, resume=resume))


def data_fingerprint(spec: ExperimentSpec) -> str:
    """What the run READ, as one content-addressed string.

    The plans and the task sets are both cas uris, so their shas already are
    their contents: a plan that draws different tasks, or a task file whose
    rows changed, is a different experiment without anyone saying so (I3).
    A plan the run does not declare renders as `-`, so a generation-only run
    (no train plan) fingerprints as one, and every run that has both is
    fingerprinted exactly as it always was.
    """
    parts = [spec.plans.train or "-", spec.plans.rollout or "-"]
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
        engines: Engine | Mapping[str, Engine], learner: Learner | None,
        max_inflight: int = 64,
        arbiter: Arbiter | None = None,
        journal: HostJournal | None = None,
        subdir: str | None = None, resume: bool = False) -> RunReport:
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

    `learner` is None for a run whose needs admit none — a generation-only run
    is a Generator and nothing else (ADR 0006 Part B).
    """
    needs = needs_of(spec)
    if any(need.learner for need in needs) and learner is None:
        raise ValueError(
            "this run needs a Trainer (its spec declares an algo) and a "
            "Trainer trains on a learner, but none was handed to the run: "
            "either hand one, or drop the algo — a spec with no algo is a "
            "generation-only run, which needs no learner and writes no ledger")
    engine_map: dict[str, Engine] = (
        dict(engines) if isinstance(engines, Mapping) else {"main": engines})
    if "main" not in engine_map:
        raise ValueError(f"no 'main' engine pool; pools: {sorted(engine_map)}")

    # ---- Phase 0: identity — computed, never typed (I3) ----------------------
    validate_or_raise(spec, schema)
    # the engine map must cover every pool the spec's traffic can route to
    # (gen, and each pipeline processor's declared judge/teacher pools)
    unmapped = sorted(traffic_pools(spec) - set(engine_map))
    if unmapped:
        raise ValueError(
            f"spec routes traffic to pools {unmapped} but the engine map "
            f"provides only {sorted(engine_map)}")
    # reachability is a build fact, not a spec fact: ask the serving pool's
    # engine for its inventory and hold every served adapter against it
    space = site_space(spec, schema)
    # a build fact asked over the wire — from a thread, so this loop keeps
    # dispatching while the serving pool answers (check_off_loop)
    inventory = await asyncio.to_thread(engine_map["main"].reachability, space)
    binding_issues = (
        check_sites_reachable_on(spec, schema, "main", inventory)
        + check_pools_serve_their_base(spec, engine_map)
        + check_members_match_their_shape(spec, engine_map, learner))
    if binding_issues:
        raise SpecError(binding_issues)
    hashes = code_hashes(spec)
    fingerprint = data_fingerprint(spec)
    rid = run_id(spec, hashes, fingerprint)
    # The caller supplies the exact directory. A requested resume cannot
    # create a missing run or find a run in another folder.
    run = await attach_run(store, rid, subdir=subdir, resume=resume, manifest={
        "run_id": rid,
        "spec": canonical_json(spec),
        "code": hashes,
        "data": fingerprint,
        "schema": schema.fingerprint(),   # same base name, different schema → loud
        "parent": spec.init.policy if spec.init is not None else None,
    })
    # the run describes its own observability: a UI reads THIS, never the
    # registries (spec/flow.py — same walk the submit gate validated with)
    await store_work(run.write_dictionary, flow_graph(spec).to_json())
    # ...and its own shape: the plans it actually ran, copied in verbatim (I11)
    plans = await store_work(load_plans, spec.plans, store)
    for kind, plan in plans.items():
        data = await store_work(store.cas_get, getattr(spec.plans, kind))
        await store_work(run.write_plan, kind, data)

    # ---- Phase 1: idempotent setup ------------------------------------------
    # The policy's initial bundle is EVERY run's — the main pool must serve the
    # bank whether or not anything trains it — and it is built by one function
    # either way: the learner's install when a need admits a learner, the
    # adapter type's own init function when none does (ADR 0006 Part B, Q6).
    bank = spec.policy.bank
    trains = any(need.learner for need in needs)
    trainable = sorted(name for name, a in bank.items() if a.trainable)
    servable = sorted(name for name, a in bank.items()
                      if ADAPTER_TYPES.get(a.adapter_type).instance.serving is not None)
    adapter_types = {name: bank[name].adapter_type for name in servable}
    resolved = {name: resolve(space, a.site) for name, a in bank.items()}
    if trains:
        # a Learner verb is synchronous by protocol (ADR 0002 Q6) and WAITS
        # on a resident or another host; the wait leaves this loop
        await asyncio.to_thread(learner.install, rid,
                                parameterization_of(spec, resolved))
    if arbiter is None:
        arbiter = Arbiter()
    attach_residents(spec, engine_map, learner if trains else None, arbiter)

    policy_version = {name: 0 for name in bank}
    resumed_from: int | None = None
    if trains:
        tail = await store_work(run.ledger_tail)
        if tail is not None:
            policy_version = {name: int(v) for name, v in tail["versions"].items()}
            await asyncio.to_thread(restore_tenant, learner, rid,
                                    policy_version, run.read_blob, trainable)
            resumed_from = int(tail["update"])
        elif spec.init is not None:
            await asyncio.to_thread(
                _warm_start, spec.init, tenant=rid, bank_names=set(bank),
                trainable=trainable, store=store, learner=learner)
        adapters = (await asyncio.to_thread(learner.emit, rid)).adapters
    else:
        adapters = await store_work(
            initial_adapters, bank_entries(spec, resolved), spec.init, store)

    # every servable delta's blob at the version the run STARTS from: the
    # frozen ones for life, the trainable ones at version 0 (the Trainer
    # writes v@1 onward) — a pool that evicts the initial bundle before the
    # first wave restores it from these, and could not otherwise (observed
    # live: nine tenants on an eight-bundle pool, and the on-policy arms
    # died at their first route on `no adapters blob v@0`)
    await store_work(write_initial_blobs, run, adapters, policy_version, set(servable),
                     fresh=(tail is None) if trains else True)
    bundle = compile_bundle(adapters, policy_version, servable, adapter_types)
    await asyncio.to_thread(engine_map["main"].add_bundle, bundle)

    # non-policy pools serve their own base; register a base bundle once each
    base_bundles = {name: Bundle(f"bundle:base:{name}", {}, {})
                    for name in engine_map if name != "main"}
    for name, base_bundle in base_bundles.items():
        await asyncio.to_thread(engine_map[name].add_bundle, base_bundle)

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
    daemons = plan_daemons(spec, needs=needs, run=run, store=store,
                           engine_map=engine_map,
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

    extent = spec.plans.extent
    return RunReport(run_id=rid, completed=len(plans[extent]), extent=extent,
                     resumed_from=resumed_from)


def bank_entries(spec: ExperimentSpec,
                 resolved: Mapping[str, tuple]) -> tuple[EntryInstall, ...]:
    """THE BANK AS INSTALLS, in bank order: each entry with its adapter type BY
    KEY, its resolved sites, and its init carrying the per-entry seed already
    derived. The master seed never crosses — the derivation lives here so a
    learner receives a seed and not the tree it came from (ADR 0002, Q2).

    One derivation, both Phase-1 paths: `parameterization_of` hands these to a
    learner's install, and a run with NO learner hands the same records to the
    adapter types' init function — which is why the two build the same v0.
    """
    entries = []
    for name, adapter in spec.policy.bank.items():
        init = dict(adapter.init)
        init.setdefault("seed", init_seed(spec.seeds.master, name))
        entries.append(EntryInstall(
            name=name, adapter_type=adapter.adapter_type, init=init,
            trainable=adapter.trainable, sites=tuple(resolved[name])))
    return tuple(entries)


def parameterization_of(spec: ExperimentSpec,
                        resolved: Mapping[str, tuple]) -> Parameterization:
    """THE place a spec becomes an install (ADR 0002, Q2): the base, the loss
    by registry key, the bank as entries, and the optimizer settings. Only a
    run that TRAINS builds one — the loss and the optimizer are the algo's."""
    optim = spec.algo.optim
    return Parameterization(
        base=spec.policy.base, loss=spec.algo.loss,
        entries=bank_entries(spec, resolved),
        optim=OptimSettings(name=optim.name, lr=optim.lr,
                            betas=tuple(optim.betas),
                            weight_decay=optim.weight_decay,
                            overrides={k: dict(v)
                                       for k, v in optim.overrides.items()}))


async def attach_run(store: Store, rid: str, *, manifest: dict,
                     subdir: str | None, resume: bool = False) -> RunHandle:
    """Attach off the host loop; cancellation waits for its store writes.

    Remote discovery and recovery can take longer than a heartbeat lease.
    A stopped tenancy must nevertheless retain custody until this worker
    has finished, so a successor cannot race its cleanup or ledger writes.
    """
    if resume:
        return await store_work(store.open_run, rid, manifest=manifest,
                                subdir=subdir, create=False)
    return await store_work(store.open_run, rid, manifest=manifest, subdir=subdir)


def initial_adapters(entries: Sequence[EntryInstall], init: WarmStart | None,
                     store: Store) -> dict[str, bytes]:
    """VERSION 0 FOR A RUN WITH NO LEARNER: each bank entry's payload from its
    adapter type's own init function (ADR 0006 Part B, Q6) — the same bytes an
    install would have emitted, so the bundle is the same content-addressed
    bundle — or, where a WarmStart names a parent that sealed this delta, that
    parent's payload instead.

    Nothing in such a run trains, so these payloads are its policy for life:
    they are written as frozen blobs and compiled into what the main pool
    serves.
    """
    names = {entry.name for entry in entries}
    if init is None:
        sealed = {}
    elif init.policy.startswith("cas://"):
        sealed = cas_warm_start(init, store, names).adapters
    else:
        sealed = sealed_payloads(*warm_start_source(init, store), init.map, names)
    return {
        entry.name: sealed.get(entry.name) or ADAPTER_TYPES.get(
            entry.adapter_type).instance.initial_payload(entry.sites, entry.init)
        for entry in entries}


def init_seed(master: int, entry: str) -> int:
    """One bank entry's init seed off the master: sha256("<master>:init:
    <entry>")'s first eight bytes. Bytes-stable — every existing run's
    deltas were initialized from exactly this number."""
    digest = hashlib.sha256(f"{master}:init:{entry}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def load_plans(declared: Plans, store: Store) -> dict[str, RunPlan]:
    """Resolve the declared plan uris, by kind. A plan the spec leaves None is
    absent from the map: no train plan is a run that never trains, no rollout
    plan is a run that never samples, and the gate refuses the spec that
    declares neither."""
    out = {}
    for kind in ("train", "rollout"):
        uri = getattr(declared, kind)
        if uri is not None:
            out[kind] = decode(store.cas_get(uri))
    return out


def write_initial_blobs(run, adapters: Mapping[str, bytes],
                        policy_version: Mapping[str, int],
                        servable: set[str], *, fresh: bool = True) -> None:
    """Persist every servable delta at the version the run starts from, once.

    The trainer writes a blob per TRAINABLE entry per update — from version 1.
    Version 0 is the init the learner emitted at Phase 1, and a frozen servable
    delta never has any other version; neither would be on the store without
    this, and restore could not rebuild the bundles that carry them. A bounded
    pool evicts the initial bundle as soon as its neighbours publish theirs,
    and the first wave's route then restores it from these blobs. On a resume
    the versions come off the ledger tail and their blobs already exist, so
    `has_blob` keeps this from ever rewriting a committed version.
    """
    for name in sorted(servable):
        version = policy_version[name]
        if run.has_blob("adapters", name, version):
            if not fresh:
                continue            # a committed version: the ledger's bytes
            stored = run.read_blob("adapters", name, version)
            if stored == adapters[name]:
                continue
            # A FRESH START THAT DISAGREES WITH AN EARLIER ATTEMPT'S VERSION
            # 0: the bundle this attempt serves is compiled from ITS bytes,
            # so those are what a restore must find — the earlier blob would
            # recompile to a different id and every restore would refuse
            # (found live 2026-09-05, an on-policy arm dead at wave 5). This
            # should never happen: version 0 is a seeded draw, and two
            # attempts differing is a byte-identity fault to chase.
            import hashlib
            print(f"[loop] adapters {name}@{version} on the store "
                  f"({hashlib.sha256(stored).hexdigest()[:12]}) is not this "
                  f"attempt's emit ({hashlib.sha256(adapters[name]).hexdigest()[:12]}): "
                  f"overwritten so restore compiles to the served bundle — "
                  f"resume-equivalence is broken here, investigate", flush=True)
        run.write_blob("adapters", name, version, adapters[name])


def needs_of(spec: ExperimentSpec) -> tuple[DaemonNeed, ...]:
    """THE ONE PLACE A SPEC BECOMES DAEMONS (ADR 0006 Part B).

    A Trainer iff there is an algo — it is the ledger's only writer, so a run
    without one commits nothing and its extent is its rollouts; a Scorer iff
    the post pipeline has a POOLED half; a Generator iff there is a rollout
    plan. Each need names the RESIDENTS its daemon's work occupies: the
    Trainer the learner and its inline half's pools, the Scorer the pools of
    its processors, the Generator its serving pool.

    The Generator's buffer is the lag when a Trainer consumes what it makes
    and None — UNPACED — when nothing does (Q1): the lag buffer bounds
    staleness against a MOVING policy, and with no Trainer the policy never
    moves. Throughput stays bounded by max_inflight and the engine.

    Measurement is NOT a need: it is not part of the run (a Measurement
    follows the ledger from outside — runner/measure.py — and writes its own
    area, never the run dir).
    """
    needs: list[DaemonNeed] = []
    if spec.algo is not None:
        split = split_pipeline(spec.algo.post)
        needs.append(DaemonNeed(
            daemon=Trainer, plan="train", learner=True,
            # empty by the split rule — a processor that occupies metal is the
            # Scorer's — but read off the pipeline rather than written as (),
            # so the Trainer stays right if the rule ever moves
            pools=pipeline_pools(split.inline)))
        if split.pooled:
            needs.append(DaemonNeed(daemon=Scorer, plan="train",
                                    pools=pipeline_pools(split.pooled)))
    if spec.plans.rollout is not None:
        needs.append(DaemonNeed(
            daemon=Generator, plan="rollout", pools=("main",),
            buffer=(spec.algo.schedule.max_policy_lag
                    if spec.algo is not None else None)))
    return tuple(needs)


def plan_daemons(spec: ExperimentSpec, *, run, store, engine_map, learner,
                 routes_at, plans, initial_bundle, initial_version,
                 max_inflight, arbiter, tenant,
                 needs: Sequence[DaemonNeed] | None = None,
                 journal: HostJournal | None = None) -> list[Daemon]:
    """THE EXECUTOR OF NEEDS: one daemon per need, built with the plan it
    consumes and the engines behind the pools it admits (`needs_of` reads the
    needs off the spec; unasked, this asks it).

    ONE Scorer owns the WHOLE pooled half of the TRAIN pipeline, in this
    process, beside whatever engines the routes map holds. One v1 limit
    stated rather than hidden: a scorer standing on its own host is placement
    work that belongs with host adoption, not here.

    Only the Trainer takes the host journal: an update is the unit of progress
    the other daemons orbit, so its phase timings are the run's own clock.
    """
    signals = RunSignals()
    tasks = load_task_sets(store, spec.gen.tasks) if spec.gen is not None else {}
    # a rollout is due when the update that first consumes it is (#59); with
    # no train plan nothing consumes anything, and the Generator is unpaced
    due_at = rollouts_needed(plans["train"].waves) if "train" in plans else {}
    daemons: list[Daemon] = []
    # three needs, three daemons — the closed set needs_of produces; each is
    # built with what only it takes (the Trainer its learner and journal, the
    # Generator its tasks and pacing)
    for need in (needs_of(spec) if needs is None else needs):
        residents = pool_residents(need.pools, engine_map)
        if need.daemon is Trainer:
            daemons.append(Trainer(
                signals, arbiter, run,
                spec=spec, plan=plans[need.plan], refs=RefReader(store, run),
                engine=engine_map["main"], learner=learner, tenant=tenant,
                post_residents=residents,
                routes_at=routes_at, journal=journal,
                initial_bundle=initial_bundle, initial_version=initial_version))
        elif need.daemon is Scorer:
            daemons.append(Scorer(
                signals, arbiter, run,
                spec=spec, plan=plans[need.plan], refs=RefReader(store, run),
                residents=residents,
                routes_at=routes_at, initial_bundle=initial_bundle))
        else:
            daemons.append(Generator(
                signals, arbiter, run,
                spec=spec, plan=plans[need.plan],
                due_at=due_at, tasks=tasks, buffer=need.buffer,
                engine=engine_map["main"], routes_at=routes_at,
                initial_bundle=initial_bundle, max_inflight=max_inflight,
                refs=RefReader(store, run)))
    return daemons


def pipeline_pools(pipeline) -> tuple[str, ...]:
    """The pools a post pipeline's processors declare (PostDef.pools) — what
    its judges address, so what the daemon running it must admit."""
    return tuple(sorted({name for proc in pipeline if proc in POST
                         for name in POST.get(proc).pools}))


def pool_residents(pools: Sequence[str], engine_map) -> tuple[Engine, ...]:
    """The distinct engine objects behind a need's pool names: two names over
    one engine are ONE resident, and that is what gets admitted."""
    distinct = {id(engine_map[name]): engine_map[name] for name in pools}
    return tuple(distinct.values())


def attach_residents(spec: ExperimentSpec, engine_map, learner: Learner | None,
                     arbiter: Arbiter) -> None:
    """Register this experiment's metal with the (possibly shared) arbiter —
    the residents its needs admit, and nothing else: a run with no Trainer
    admits no learner, so `learner` is None and none is attached.

    Object-keyed and idempotent: two pools backed by one engine are ONE
    resident; a second experiment attaching the same engine is a no-op.
    Exclusive groups come from a multi-member HostSpec — its members
    ALTERNATE on one partition, and alternation exists only there (ADR 0001).
    Nothing about memory is declared here: a spec's `vram_gb` is a carve
    size the metal converts at build, and the arbiter's declared load is the
    host's own.

    A REMOTE pool attaches as a zero-footprint free resident: its metal is
    another host's partition, so local admission is bookkeeping — the real
    admission, alternation included, happens host-side at the serving
    partition's own arbiter, whatever HostSpec the pool came from. A ROUTED
    LEARNER is the same rule on the training side (ADR 0006 Part A): the
    Trainer runs where the run is anchored, and each of its learner verbs is
    admitted per frame at the host that wears the learner.

    An alternation demand on metal the host ALREADY holds in an alternation
    group (a multi-regime host attached it at birth) is satisfied, not
    conflicting: the demand defers to the metal's own group. Only an
    alternation demand on always-resident metal still raises — that metal
    cannot alternate.
    """
    pool_group: dict[str, str | None] = {}
    learner_group: str | None = None
    for hi, host in enumerate(spec.topology.hosts):
        alternating = len(host.members) > 1
        for member in host.members:
            if isinstance(member, PoolMember):
                pool_group[member.name] = f"alternate:{hi}" if alternating else None
            else:
                learner_group = f"alternate:{hi}" if alternating else None

    def deferred(obj: object, group: str | None) -> str | None:
        """The metal's own alternation group satisfies (and overrides) an
        alternation demand — the physical owner declared it at birth."""
        if arbiter.is_attached(obj) and arbiter.attached_group(obj) is not None:
            return None
        return group

    # A RemotePool the host attached AT BIRTH is its own resident's door
    # (ADR 0002: every engine is a process behind a proxy) and takes the
    # local path below; one nobody attached is served by ANOTHER host.
    # Decided for every pool BEFORE any attach, so two names over one remote
    # do not read the first's attachment as the metal's own.
    remote_pools = {name for name, engine in engine_map.items()
                    if isinstance(engine, RemotePool)
                    and not arbiter.is_attached(engine)}
    for name in sorted(engine_map):
        if name in remote_pools:
            arbiter.attach(engine_map[name], label=f"remote:{name}")
            continue
        arbiter.attach(engine_map[name], label=f"engine:{name}",
                       group=deferred(engine_map[name], pool_group.get(name)))
    if learner is None:
        return
    if isinstance(learner, RemoteLearner) and learner.admitted:
        arbiter.attach(learner, label="remote:learner")
    else:
        arbiter.attach(learner, label="learner",
                       group=deferred(learner, learner_group))


def cas_warm_start(init: WarmStart, store: Store, bank_names: set[str],
                   trainable: Sequence[str] = ()) -> Emitted:
    """Restore CAS state using the existing Emitted wire codec and bank map.

    A materialized adapter can initialize an ordinary run without inventing a
    parent run or a ledger. CAS contents are immutable; the init URI hashes
    into the child's identity. Missing requested optimizer moments refuse a
    load, just as a store warm start whose moments were swept does.
    """
    from rlstack.runner.remote import decode_emitted

    source = decode_emitted(json.loads(store.cas_get(init.policy)))
    source_name = {this: original for original, this in init.map.items()}
    adapters = {name: source.adapters[source_name.get(name, name)]
                for name in sorted(bank_names)
                if source_name.get(name, name) in source.adapters}
    optim = {}
    if init.optim == "load":
        for name in trainable:
            if name not in adapters:
                continue
            original = source_name.get(name, name)
            if original not in source.optim:
                raise FileNotFoundError(
                    f"CAS warm start {init.policy!r} has no optimizer moments "
                    f"for {original!r}; use optim='fresh' or include its moments")
            optim[name] = source.optim[original]
    return Emitted(adapters=adapters, optim=optim)


def warm_start_source(init: WarmStart, store: Store) -> tuple[RunHandle, int]:
    """Read the named parent without attaching or sweeping its in-flight work."""
    if not init.policy.startswith("store://"):
        raise NotImplementedError(
            f"warm start reads another run's SEALED deltas out of a run "
            f"store, so init.policy must be a 'store://<run_id>@<version>' "
            f"address; got {init.policy!r}")
    address = init.policy[len("store://"):]
    parent_id, _, version_text = address.partition("@")
    manifest = store.peek_manifest(parent_id)
    if manifest is None:
        raise FileNotFoundError(f"warm start parent {parent_id!r} does not exist")
    return (RunHandle(store, parent_id.rsplit("/", 1)[-1], json.dumps(manifest),
                      store.run_prefix(parent_id)),
            int(version_text))


def sealed_payloads(parent: RunHandle, version: int,
                    rename: Mapping[str, str],
                    bank_names: set[str]) -> dict[str, bytes]:
    """The parent's sealed delta payloads for the names in THIS bank, renamed
    through the warm start's map (source name -> this bank's name). A name the
    parent never sealed is simply absent: that delta starts fresh."""
    source_name = {this: source for source, this in rename.items()}
    payloads: dict[str, bytes] = {}
    for name in sorted(bank_names):
        try:
            payloads[name] = parent.read_blob(
                "adapters", source_name.get(name, name), version)
        except FileNotFoundError:
            continue  # no sealed state for this delta: it starts fresh
    return payloads


def _warm_start(init: WarmStart, *, tenant: str, bank_names: set[str],
                trainable: list[str], store: Store, learner: Learner) -> None:
    """Load another run's sealed deltas (renamed via init.map) into this learner."""
    if init.policy.startswith("cas://"):
        state = cas_warm_start(init, store, bank_names, trainable)
        learner.load(tenant, state.adapters,
                     state.optim if init.optim == "load" else None)
        return
    parent, version = warm_start_source(init, store)
    adapters = sealed_payloads(parent, version, init.map, bank_names)
    source_name = {this: source for source, this in init.map.items()}
    # the moments are read only where they are asked for: optim="fresh" takes
    # the deltas alone, and retention may have swept a mid-run version's
    optim = (None if init.optim != "load" else
             {name: _parent_moments(parent, source_name.get(name, name), version)
              for name in adapters if name in trainable})
    learner.load(tenant, adapters, optim)


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
