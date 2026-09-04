"""Host: an ATOMIC, PURPOSED PARTITION of GPU memory — the fleet's unit.

Not a GPU, not a node, not a container: a host is some slice of some GPUs, born
with its Partition and its Regimes, attesting the metal it was handed against
them and never growing or reshaping afterward (I12). One regime is a dedicated
host; several make it ALTERNATE them on its own arbiter group — one host
wearing masks, never two hosts coordinating. It owns its engines, at most ONE
multi-tenant learner, its arbiter, its journal, and the one traffic meter both
of the former count into. A run it holds may reach ANOTHER host's engines and,
since ADR 0006 Part A, another host's learner: the Trainer sits where the run
is ANCHORED, and every verb it issues is admitted at the host that owns the
metal it occupies.

`await host.submit(spec, schema, store)` is how an experiment reaches metal:
BIND each declared pool onto an owned engine serving that base at that shape,
FIT (a learner member finds a learner — this host's own, or one routed to
another host's door: custody, never memory arithmetic), ATTEST (roster in
memory, journal to
hosts/<name>/log.jsonl — observability only, which correctness never reads),
and RUN under the shared arbiter against the experiment's OWN run store. That
last is the rule submit exists to enforce: one experiment, one store, for life
(I10) — run_id is global but existence is store-scoped, so the same spec
against two stores forks history silently.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

from rlstack.data.stores.base import Store
from rlstack.policy.siteschema import SiteSchema
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.interfaces import Engine, Learner
from rlstack.runner.loop import (
    RunReport, experiment_identity, run_experiment_async,
)
from rlstack.runner.meters import HostJournal, TrafficMeter

if TYPE_CHECKING:
    from rlstack.runner.residents import Resident
from rlstack.runner.remote import spec_from_json
from rlstack.spec.specs import ExperimentSpec, LearnerMember, PoolMember


class HostError(RuntimeError):
    """A submission this host cannot serve (no matching engine, no learner,
    a solo host already occupied)."""


@dataclass(frozen=True)
class Partition:
    """The irreducible carved share of metal a host is born onto: the NAME of
    the Metal it was carved from, the KIND of GPU it is made of ("L4",
    "H100"), the device indices, and the memory FRACTION it owns on each.
    Since ADR 0002 every resident wearing this partition is its own PROCESS,
    so both numbers are enforced where the substrate can enforce them: the
    devices become that process's CUDA_VISIBLE_DEVICES, and the fraction is
    vLLM's gpu_memory_utilization for an engine and torch's per-process
    allocator cap for a learner (runner/residents.py). Memory partitions
    honestly; SMs still time-share across partition boundaries — a stated
    cost, visible in latency, not hidden by this record.
    `memory` is the SUBSTRATE'S unit, never the spec's (ADR 0001): a spec
    declares GB (`vram_gb`, total across shards), the desk deduces in GB, and
    the metal DERIVES this fraction once at build — `fraction_for_gb` against
    the card it measured — because 0.5 of an L4 and 0.5 of an H100 are
    different amounts of memory and only the metal knows which it holds.
    `gpu` is descriptive, never decisive: the carve carries the kind down from
    the Metal it drew on, so a reader of this row can tell the two halves
    apart. `metal` is a registered Metal's NAME — a provider fact; the spec
    names no metal at all."""

    metal: str
    devices: tuple[int, ...]
    memory: float = 1.0
    gpu: str = ""

    def row(self) -> dict:
        """The partition as a JSON row — ONE home for the shape the host-up
        event journals, status() reports over the wire, and the observer's
        hosts view reads back."""
        return {"metal": self.metal, "gpu": self.gpu,
                "devices": list(self.devices), "memory": self.memory}


@dataclass(frozen=True)
class Regime:
    """One capability a host can wear: inference (an engine built tp=shape)
    or training (a learner built fsdp=shape) over one base. The fleet
    matches joins against these; the host attests its metal against them at
    birth and never changes them after."""

    name: str
    capability: str                 # "inference" | "training"
    base: str | None
    shape: int = 1

    def __post_init__(self) -> None:
        if self.capability not in ("inference", "training"):
            raise ValueError(
                f"Regime.capability must be 'inference' or 'training', "
                f"got {self.capability!r}")


LEARNER_ROUTE = "learner"
"""The key a learner's ADDRESS rides under in a route table — the same name
the desk gives the learner demand (`Demand.name`), so a route table reads as
the placement it came from."""


@dataclass(frozen=True)
class Routed:
    """What placement's routes resolve into at this host's door: the pools
    ANOTHER host serves, and — since ADR 0006 Part A — the learner another
    host wears. Both are live proxies behind the protocols the runner drives
    locally, so nothing downstream can tell where they live; `learner` None
    means this run trains on the host's own learner, or on none at all."""

    pools: dict[str, Engine] = field(default_factory=dict)
    learner: Learner | None = None


@dataclass
class Tenancy:
    """One experiment on this host, as the roster tracks it."""

    run_id: str
    pools: dict[str, str]           # pool name -> base the bound engine serves
    store: str = ""                 # locator of the run's OWN store
    status: str = "running"         # running | done | failed
    updates_completed: int | None = None
    attached_at: float = field(default_factory=time.time)


class Host:
    def __init__(self, name: str, *, engines: Sequence[Engine],
                 learner: Learner | None, store: Store,
                 arbiter: Arbiter | None = None,
                 partition: Partition | None = None,
                 regimes: tuple[Regime, ...] = (),
                 solo: bool = False,
                 transport_for: Callable[[str], "Transport"] | None = None,
                 schema_for: Callable[[str], SiteSchema] | None = None,
                 sampler=None,
                 residents: Sequence["Resident"] = ()) -> None:
        self.name = name
        self.engines = tuple(engines)
        self.learner = learner
        # The processes behind `engines` and `learner` when this host was
        # carved by a metal (ADR 0002): one per regime, each holding the door
        # its proxy speaks through. Empty for a hand-built host whose engines
        # and learner are plain objects in this process — the same Host,
        # because the daemons cannot tell and were never meant to.
        self.residents = tuple(residents)
        self.store = store
        self.arbiter = arbiter or Arbiter()
        self.partition = partition
        self.regimes = regimes
        # A birth fact like the partition and the regimes (I12): this host's
        # purpose is ONE experiment at a time. Nothing about it is discovered
        # or negotiated later — a deploy that means it says so at construction.
        self.solo = solo
        # Two more birth facts, both for ADOPTION — an experiment arriving
        # over the wire instead of in-process. `transport_for` turns an ADDRESS
        # from an adopt frame into a TRANSPORT to the host serving it:
        # address formats are venue (I5), so the deploy that knows them hands
        # the resolver in, and THIS host wraps the transport with the pool's
        # own capability facts (base, tp) read off the spec — the venue knows
        # where, the spec knows what. `schema_for` compiles the base's
        # SiteSchema HERE — an adopted spec never ships a schema, because the
        # schema must describe the checkpoint THIS metal serves.
        self.transport_for = transport_for
        self.schema_for = schema_for
        self._adoptions: dict[str, asyncio.Task] = {}
        self.sampler = sampler or sample_gpu
        self.meter = TrafficMeter()
        self.roster: dict[str, Tenancy] = {}
        self.attest_name()
        self.attest_regimes()
        self._attach_regimes()
        self.wire_meter()
        store.append_host_event(name, {
            "event": "host-up", "t": time.time(),
            "engines": [engine.base or "*" for engine in self.engines],
            "partition": partition.row() if partition is not None else None,
            "regimes": [{"name": r.name, "capability": r.capability,
                         "base": r.base, "shape": r.shape} for r in regimes],
            "solo": self.solo,
            # the processes this host was born with, label + pid: what lets
            # the hosts view show which residents are living without a probe
            "residents": [{"label": r.label, "pid": r.pid()}
                          for r in self.residents],
            "store": store.describe()})

    # ---- birth facts, one named method per rule -----------------------------

    def attest_name(self) -> None:
        """A host's name is a JOURNAL PATH SEGMENT — hosts/<name>/log.jsonl —
        so it may contain no "/": a name that did would journal one directory
        deeper than Store.list_hosts() looks, leaving the host perfectly alive
        and completely invisible to the observer, its runs and its gpu samples
        with it. Refused at birth, where the name is still just a string."""
        if "/" in self.name:
            raise HostError(
                f"host name {self.name!r} contains '/': the name is a journal "
                f"path segment (hosts/<name>/log.jsonl), so a '/' would hide "
                f"the host from the observer entirely")

    def attest_regimes(self) -> None:
        """A host IS its regimes: each inference regime must be backed by an
        owned engine BUILT at exactly (base, tp=shape); each training regime
        by the learner built at fsdp=shape. Capability is a birth fact —
        attested here, never mutated after — so a deploy handing the wrong
        metal dies at construction, not mid-run."""
        for regime in self.regimes:
            if regime.capability == "inference":
                if self.engine_for(regime.base, regime.shape) is None:
                    raise HostError(
                        f"host {self.name!r} declares regime {regime.name!r} "
                        f"({regime.base!r}, tp={regime.shape}) but no owned "
                        f"engine is built for it; engines: "
                        f"{[(e.base, e.tp) for e in self.engines]}")
            elif self.learner is None or self.learner.fsdp != regime.shape:
                built = None if self.learner is None else self.learner.fsdp
                raise HostError(
                    f"host {self.name!r} declares training regime "
                    f"{regime.name!r} at fsdp={regime.shape} but the learner "
                    f"handed is built fsdp={built}")

    def occupied(self) -> bool:
        """Is an experiment RUNNING on this host right now? THE one reading of
        "has a tenant" — the roster keeps finished tenancies for the observer,
        and a host that finished a run is free again. Both the solo refusal and
        the fleet's join rung ask here, so the rule has one home."""
        return any(t.status == "running" for t in self.roster.values())

    def check_solo(self, run_id: str) -> None:
        """A SOLO host serves one experiment at a time, and says no to the
        second at SUBMIT — after the cheap spec-shaped refusals, before the
        roster and the journal, which is where a tenancy actually begins.

        The multi-tenancy invariant (I8) says tenants cannot disturb each
        other's RESULTS; it never promised they cannot disturb each other's
        THROUGHPUT, and a partition small enough that one tenant fills it is
        exactly where that matters. Resubmitting the SAME run_id is a resume,
        not a second tenancy.
        """
        if not self.solo:
            return
        others = sorted(rid for rid, t in self.roster.items()
                        if rid != run_id and t.status == "running")
        if others:
            raise HostError(
                f"host {self.name!r} was born solo: it serves ONE experiment "
                f"at a time and {others[0]} is still running. Submit elsewhere, "
                f"or carve a second host — the partition is the unit, not the "
                f"queue")

    def _attach_regimes(self) -> None:
        """A regime-host's residents attach AT BIRTH: the partition is the
        footprint, so a joining tenant's declared size never counts here (a
        size is a carve hint, meaningless on a join — the weights already
        live). Several regimes share ONE exclusive group: the alternation IS
        the host, switched by its own arbiter."""
        group = f"host:{self.name}" if len(self.regimes) > 1 else None
        for regime in self.regimes:
            obj = (self.learner if regime.capability == "training"
                   else self.engine_for(regime.base, regime.shape))
            resident = self.resident_for(regime)
            sleeps = resident is not None and resident.hello.get("sleeps")
            # fraction 0.0, not None: the arbiter's declared load stays the
            # host's own (nothing a tenant declares is adopted into it). The
            # alternation hooks are the resident's door verbs, wired for
            # every resident whose hello says it can hand the device back —
            # engine or learner alike (ADR 0002, Q8)
            self.arbiter.attach(obj, label=f"{self.name}:{regime.name}",
                                group=group, fraction=0.0,
                                wake=resident.wake if sleeps else None,
                                evict=resident.sleep if sleeps else None)

    def resident_for(self, regime: Regime) -> "Resident | None":
        """The process wearing `regime`, if this host was carved (label is
        `<host>:<regime>`); None on a hand-built host."""
        for resident in self.residents:
            if resident.regime.name == regime.name:
                return resident
        return None

    def wire_meter(self) -> None:
        """ONE meter per host: every engine it owns counts its tokens into
        this meter and its arbiter counts admission into the same one, so a
        `traffic` event describes the PARTITION — which is what a shared
        engine's load is a property of. Wired by assignment at birth because
        engines and arbiters are built by the deploy that owns the metal and
        handed to the host afterwards."""
        for engine in self.engines:
            engine.meter = self.meter
        self.arbiter.meter = self.meter

    def engine_for(self, base: str | None, tp: int) -> Engine | None:
        """THE shape-matched lookup — exact base first, fake-metal wildcard
        as fallback, tp always exact (a build fact has no wildcard). One home
        for the rule: binding, attestation, and the wire's HostService all
        resolve through this."""
        exact = [e for e in self.engines if e.base == base and e.tp == tp]
        wildcard = [e for e in self.engines if e.base is None and e.tp == tp]
        return (exact or wildcard or [None])[0]

    # ---- the submission steps, one named method per rule --------------------

    def bind_pools(self, spec: ExperimentSpec,
                   remotes: frozenset[str] = frozenset()) -> dict[str, Engine]:
        """Each declared pool gets an owned engine serving its base at its
        declared tp (exact base preferred; base None is fake metal's
        wildcard). No match is a HostError — this host does not carry that
        model at that shape. Pools named in `remotes` are served by ANOTHER
        host and skip local binding."""
        binding: dict[str, Engine] = {}
        for host in spec.topology.hosts:
            for member in host.members:
                if not isinstance(member, PoolMember):
                    continue
                if member.name in remotes:
                    continue
                wanted = member.base or spec.policy.base
                engine = self.engine_for(wanted, member.tp)
                if engine is None:
                    raise HostError(
                        f"host {self.name!r} has no engine serving {wanted!r} "
                        f"at tp={member.tp} for pool {member.name!r}; it "
                        f"serves {[(e.base or '*', e.tp) for e in self.engines]}")
                binding[member.name] = engine
        return binding

    def check_fit(self, spec: ExperimentSpec, binding: dict[str, Engine],
                  remotes: frozenset[str] = frozenset(),
                  learner: Learner | None = None) -> None:
        """FIT is custody, not arithmetic: this host can HOLD the submission
        when every declared pool is bound (bind_pools' job, already done) and
        a learner member finds a learner — this host's own, or the one ROUTED
        to another host's door (ADR 0006 Part A), which is accepted here for
        the same reason a routed pool is: the metal is another partition's,
        and the run reaches it over the wire. Memory is NOT summed at this
        door — since ADR 0001 a spec declares GB and a partition owns a
        fraction, the crossing happens once at the metal's build, and the
        residents enforce it per process (vLLM's reservation, torch's cap); a
        JOIN is always memory-free because the weights already live. A remote
        pool is another partition's footprint and never counts here."""
        for host in spec.topology.hosts:
            for member in host.members:
                if isinstance(member, PoolMember):
                    continue
                if learner is None and self.learner is None:
                    raise HostError(
                        f"host {self.name!r} has no training regime and no "
                        f"learner was routed to it; it cannot serve this "
                        f"spec's learner member")

    # ---- submit -------------------------------------------------------------

    async def submit(self, spec: ExperimentSpec, schema: SiteSchema,
                     store: Store | None = None,
                     max_inflight: int = 64,
                     remotes: Mapping[str, Engine] | None = None,
                     subdir: str | None = None,
                     learner: Learner | None = None) -> RunReport:
        """Run one experiment on this host's metal: bind, fit, solo, attest, run.
        `remotes` maps pool names served by OTHER hosts to their RemotePools
        and `learner` is a learner worn by another host (None: this host's
        own) — the run is ANCHORED here and reaches every other partition over
        the wire, so this host binds, fits and journals only what it serves."""
        run_store = store if store is not None else self.store
        remote_pools = dict(remotes or {})
        learner = learner if learner is not None else self.learner
        binding = self.bind_pools(spec, remotes=frozenset(remote_pools))
        self.check_fit(spec, binding, remotes=frozenset(remote_pools),
                       learner=learner)
        binding |= remote_pools
        rid = experiment_identity(spec, schema)
        self.check_solo(rid)
        self.roster[rid] = Tenancy(rid, pools={
            name: (engine.base or "*") for name, engine in sorted(binding.items())},
            store=run_store.describe())
        self.store.append_host_event(self.name, {
            "event": "attach", "t": time.time(), "run_id": rid,
            "pools": sorted(binding), "remotes": sorted(remote_pools),
            "plan": spec.plans.train,     # the shape, by reference (#59)
            "store": run_store.describe(),
            # filing rides the attach so an observer knows it from the run's
            # FIRST breath: the run directory's manifest lands moments after
            # this line, and a reader snapshotting between the two would
            # otherwise file the newborn at the store's top for one refresh
            "subdir": subdir or ""})
        try:
            report = await run_experiment_async(
                spec, schema, run_store, binding, learner,
                max_inflight, arbiter=self.arbiter, subdir=subdir,
                # the tenant knows its own phases but not its metal: this is
                # the door through which its update timings reach THIS host's
                # journal, and the only reason the runner learns a host name
                journal=HostJournal(self.store, self.name))
        except BaseException:
            self.roster[rid].status = "failed"
            self.store.append_host_event(self.name, {
                "event": "detach", "t": time.time(), "run_id": rid,
                "status": "failed"})
            raise
        finally:
            self.release_tenant(learner, rid)
        self.roster[rid].status = "done"
        self.roster[rid].updates_completed = report.updates_completed
        self.store.append_host_event(self.name, {
            "event": "detach", "t": time.time(), "run_id": rid,
            "status": "done", "updates_completed": report.updates_completed})
        return report

    def release_tenant(self, learner: Learner | None, run_id: str) -> None:
        """THE TENANCY'S END AT THE LEARNER, whichever host wears it: the run
        is over — done or failed — so its parameterization leaves the training
        metal instead of accumulating there (ADR 0006 Part A). It matters now
        that a learner outlives the runner: a standing learner joined by runs
        anchored elsewhere would otherwise carry every dead tenant of every
        host that ever used it. Resume is unaffected — a re-submission
        installs and restores from the store's blobs, as every attach does.

        A learner that cannot be reached is a learner with nothing to
        release: its process holds the tenant, so a dead one has already
        forgotten it, and a run being torn down must not fail its teardown
        over a frame to a host that is going away."""
        if learner is None:
            return
        try:
            learner.uninstall(run_id)
        except Exception as unreachable:      # noqa: BLE001 — see the docstring
            print(f"[host {self.name}] uninstall {run_id}: {unreachable}")

    # ---- adopt: the submission door -----------------------------------------

    async def adopt(self, spec_row: Mapping,
                    routes: Mapping[str, str] | None = None,
                    code: Mapping[str, str] | None = None,
                    subdir: str | None = None) -> dict:
        """Take an experiment IN OVER THE WIRE and run it as one more tenancy
        on this host's own loop — submit, without the submitter in-process.

        The frame is the spec's canonical JSON plus `routes` (pool name ->
        ADDRESS, only for demanded pools this host does not serve — placement's
        answer, authored by the fleet, never by hand). The schema is derived
        HERE from this host's own `schema_for`, so identity is computed where
        the code that will run lives (I3), and a client cannot ship a schema
        the metal disagrees with.

        The reply is ACCEPTANCE, never completion: {run_id, state} out at
        once, the run continuing as a background task — the ledger is the
        result channel and peeks are the door, exactly as for any run. What
        acceptance means is only that this host can HOLD the experiment
        (bind, fit, solo — the fail-fast custody checks); the submit gate
        still runs at Phase 0 inside the task, and a gate refusal lands in
        the roster and journal as a failed tenancy. Adopting a run_id this
        host already finished is a RESUME, the same way resubmitting one is.
        """
        try:
            spec = self.decode_adoption(spec_row)
            self.check_code_agreement(spec, code)
            schema = self.derive_schema(spec)
            routed = self.resolve_routes(spec, routes or {})
            binding = self.bind_pools(spec, remotes=frozenset(routed.pools))
            self.check_fit(spec, binding, remotes=frozenset(routed.pools),
                           learner=routed.learner)
            rid = experiment_identity(spec, schema)
            self.check_solo(rid)
        except (HostError, TypeError, ValueError) as refusal:
            return {"accepted": False, "error": str(refusal)}
        live = self._adoptions.get(rid)
        if live is not None and not live.done():
            return {"accepted": True, "run_id": rid, "state": "running"}
        # ACCEPTANCE IS VISIBLE THE MOMENT IT IS GIVEN: the tenancy enters the
        # roster HERE, synchronously, not when the background task gets its
        # first tick — otherwise two adopts in one breath both pass check_solo
        # against an empty roster, and the door's own rule races itself.
        # submit() re-writes this row (and journals it); the eager copy is
        # in-memory custody only.
        self.roster[rid] = Tenancy(rid, pools={
            name: (engine.base or "*")
            for name, engine in sorted((binding | routed.pools).items())},
            store=self.store.describe())
        task = asyncio.create_task(
            self.submit(spec, schema, remotes=routed.pools, subdir=subdir,
                        learner=routed.learner))
        # a failed run already journals and rosters its failure (submit's own
        # except path) — but the EXCEPTION ITSELF would otherwise vanish into
        # a retrieved future, and a silent adoption death is undiagnosable
        # from any journal. Print the traceback where the host's stdout goes;
        # a CANCELLED task (stop's doing) is left alone.
        def adoption_ended(done, rid=rid, host=self.name):
            if done.cancelled():
                return
            failure = done.exception()
            if failure is not None:
                import traceback
                told = "".join(traceback.format_exception(failure))
                print(f"[host {host}] adoption {rid} DIED:\n{told[-4000:]}")
        task.add_done_callback(adoption_ended)
        self._adoptions[rid] = task
        return {"accepted": True, "run_id": rid, "state": "adopted"}

    async def stop(self, run_id: str) -> dict:
        """A tenancy told to die — adopt's per-run inverse, and the verb a
        reroute rides: the adoption task is cancelled and AWAITED, so the
        reply means the death is COMPLETE. The daemons unwind structurally
        (they run under one TaskGroup, so cancelling the adoption cancels
        them all), the roster row reads failed, the journal's detach is
        written (submit's own except path does both), and the run_id is free
        to adopt again — here or elsewhere. Stopping mid-update is safe by
        resume-equivalence: the uncommitted update is redone on resume, and
        nothing else exists outside the store. A run this host is not
        running answers stopped: False instead of raising — already dead is
        the goal state, not an error."""
        task = self._adoptions.get(run_id)
        if task is None or task.done():
            told = self.roster.get(run_id)
            return {"stopped": False, "run_id": run_id,
                    "state": told.status if told is not None else "unknown"}
        task.cancel()
        # swallow the task's CancelledError, propagate our own (gather keeps
        # the two apart; a real failure was already rostered and journaled by
        # submit's except path)
        await asyncio.gather(task, return_exceptions=True)
        told = self.roster.get(run_id)
        if told is not None and told.status == "running":
            # cancelled before submit's try block ever ran: the eager roster
            # row is still "running", so submit could not write its own
            # bookkeeping — the journal must not lose a detach
            told.status = "failed"
            self.store.append_host_event(self.name, {
                "event": "detach", "t": time.time(), "run_id": run_id,
                "status": "failed"})
        return {"stopped": True, "run_id": run_id,
                "state": told.status if told is not None else "unknown"}

    def check_code_agreement(self, spec: ExperimentSpec,
                             claimed: Mapping[str, str] | None) -> None:
        """LOUD where version skew was silent: the client ships the source
        hashes of every registered name its spec references, and this host
        diffs them against its own registries — per name, so the refusal says
        WHICH loss or processor the container's image predates.

        The quiet failure this kills: edit a loss body locally, submit to a
        standing host running the old image, and the host would happily run
        ITS body under YOUR name — a different experiment than you meant,
        detectable only by noticing an unexpected run_id. Code never crosses
        the wire (identity is computed where the code runs, I3); agreement
        about WHICH code does. A frame carrying no hashes skips the check —
        the caller chose not to claim anything."""
        if not claimed:
            return
        from rlstack.registry import code_hashes

        mine = code_hashes(spec)
        stale = sorted(name for name, theirs in claimed.items()
                       if name in mine and mine[name] != theirs)
        if stale:
            raise HostError(
                f"code skew: this host's image runs different source for "
                f"{', '.join(stale)} than the submitting checkout — the "
                f"container predates your edit (or you predate its). Redeploy "
                f"the host image, then resubmit; running anyway would be a "
                f"different experiment than you meant")

    def decode_adoption(self, spec_row: Mapping) -> ExperimentSpec:
        """The frame's spec, decoded and TYPED: an adopt frame carries exactly
        one ExperimentSpec, and anything else is refused before it can reach
        identity."""
        spec = spec_from_json(spec_row)
        if not isinstance(spec, ExperimentSpec):
            raise TypeError(
                f"adopt expects an ExperimentSpec's canonical form, decoded "
                f"{type(spec).__name__}")
        return spec

    def derive_schema(self, spec: ExperimentSpec) -> SiteSchema:
        """This host's own schema for the spec's base — or the honest refusal:
        a host born without `schema_for` cannot compute an adopted run's
        identity, and says so at the door rather than serving a wrong one."""
        if self.schema_for is None:
            raise HostError(
                f"host {self.name!r} was born with no schema_for: it cannot "
                f"derive {spec.policy.base!r}'s site schema, so it cannot "
                f"adopt (pass schema_for=hf_schema on metal, or the fake "
                f"schema in tests)")
        return self.schema_for(spec.policy.base)

    def resolve_routes(self, spec: ExperimentSpec,
                       routes: Mapping[str, str]) -> Routed:
        """Every route resolved into a live proxy, ONCE, at the door: the
        venue's resolver turns the ADDRESS into a transport, and the member's
        declared capability (base, tp for a pool; fsdp for the learner — read
        off the spec, the only side that knows it) wraps it into the
        RemotePool or RemoteLearner the runner will drive. A host born without
        a resolver refuses routed adoption rather than guessing what an
        address means; a route naming no declared member is a placement bug
        and refused the same way.

        The `learner` key is the learner member's address (ADR 0006 Part A) —
        the run is anchored HERE and its learner lives THERE, so every learner
        verb crosses that host's door and is admitted at its arbiter."""
        if not routes:
            return Routed()
        if self.transport_for is None:
            raise HostError(
                f"host {self.name!r} was born with no transport_for: it "
                f"cannot resolve addresses {sorted(routes)} (pass "
                f"transport_for= at construction — the deploy that owns the "
                f"venue knows the address format)")
        from rlstack.runner.remote import RemoteLearner, RemotePool

        members = {member.name: member
                   for host in spec.topology.hosts
                   for member in host.members
                   if isinstance(member, PoolMember)}
        self.check_learner_route_is_unambiguous(members, routes)
        pools: dict[str, Engine] = {}
        learner: Learner | None = None
        for name, address in sorted(routes.items()):
            if name == LEARNER_ROUTE:      # never a pool: checked just above
                learner = RemoteLearner(self.transport_for(address),
                                        fsdp=self.learner_member(spec).fsdp,
                                        admitted=True)
                continue
            member = members.get(name)
            if member is None:
                raise HostError(
                    f"route {name!r} names no pool this spec declares "
                    f"({sorted(members)}) — routes are placement's answer to "
                    f"the spec's own demands")
            pools[name] = RemotePool(self.transport_for(address),
                                     base=member.base or spec.policy.base,
                                     tp=member.tp)
        return Routed(pools=pools, learner=learner)

    def check_learner_route_is_unambiguous(self, members: Mapping,
                                           routes: Mapping[str, str]) -> None:
        """A spec may not declare a pool named `learner` AND route one: the
        learner's address rides under that key (the desk's own name for the
        learner demand), so the two would be one entry and the wrong metal
        would answer. Refused where it is still a name, not a proxy."""
        if LEARNER_ROUTE in routes and LEARNER_ROUTE in members:
            raise HostError(
                f"this spec declares a pool named {LEARNER_ROUTE!r} and the "
                f"placement routes a learner: one route key cannot address "
                f"both — rename the pool")

    def learner_member(self, spec: ExperimentSpec) -> LearnerMember:
        """The spec's ONE learner member — the build fact a routed learner is
        attested against (its `fsdp`). A route to a learner the spec never
        declared is a placement bug, refused here."""
        for host in spec.topology.hosts:
            for member in host.members:
                if not isinstance(member, PoolMember):
                    return member
        raise HostError(
            f"a learner was routed to host {self.name!r} but this spec "
            f"declares no learner member — routes are placement's answer to "
            f"the spec's own demands")

    # ---- observability ------------------------------------------------------

    async def run_stats(self, every: float = 30.0) -> None:
        """Journal one GPU sample and one traffic window every `every` seconds
        until cancelled — the CLI's `gpu` view (utilization, memory, downtime)
        is computed from the samples; a gap in them IS the downtime. Run it
        alongside submissions: create_task(host.run_stats()), cancel when done.

        ONE cadence, two events: the traffic window is drained on the same
        tick rather than by a second timer, so load and utilization are read
        against the same clock. The gpu sample is skipped where no NVIDIA
        runtime answers; the traffic window never is — a host that served
        nothing this window says so with zeros."""
        while True:
            sample = await asyncio.to_thread(self.sampler)
            now = time.time()
            if sample is not None:
                self.store.append_host_event(self.name, {
                    "event": "stats", "t": now, **sample})
            self.store.append_host_event(self.name, {
                "event": "traffic", "t": now, **self.meter.drain(now).row()})
            await asyncio.sleep(every)

    def status(self) -> dict:
        """The metal as this host sees it: the partition it was born onto
        (the Metal's name, kind of GPU, devices, fraction — as a row, so a
        status crosses the wire unchanged), state (arbiter residency),
        declared load, the WORK AT THE DOOR (in flight now, and admitted
        since birth — what the desk's idleness test reads, ADR 0003), and
        the tenant roster."""
        return {
            "host": self.name,
            "engines": [engine.base or "*" for engine in self.engines],
            "partition": self.partition.row() if self.partition else None,
            "regimes": [regime.name for regime in self.regimes],
            "solo": self.solo,
            "declared_load": round(self.arbiter.declared_load(), 3),
            "residency": self.arbiter.residency(),
            "in_flight": self.arbiter.in_flight(),
            "admitted": self.arbiter.admitted(),
            "residents": [r.row() for r in self.residents],
            "tenants": {rid: {"status": t.status, "pools": t.pools,
                              "updates_completed": t.updates_completed}
                        for rid, t in sorted(self.roster.items())},
        }


def sample_gpu() -> dict | None:
    """One nvidia-smi sample: per-GPU utilization %% and memory MiB. None
    when no NVIDIA runtime is visible (fakes, laptops, CPU CI) — sampling
    quietly does nothing off the metal."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        try:
            util, used, total = [int(part.strip()) for part in line.split(",")]
        except ValueError:
            continue
        gpus.append({"util": util, "mem_used": used, "mem_total": total})
    return {"gpus": gpus} if gpus else None
