"""The Fleet: the inventory of Metal and hosts, and the placement ladder over them.

An experiment declares capability DEMANDS — what, never where: capability, base
and shard shape read straight off its gpu_config, with the declared fraction as a
carve hint. Placement climbs three rungs, one currency and one decider each
(I12): JOIN a host that already serves the capability (automatic; the target
host's own arbiter decides, and fractions are ignored because the weights
already live there), CARVE a new host out of RESIDUAL metal (automatic BECAUSE
journaled, residual-only so a living host is never shrunk or reshaped), or
ACQUIRE — new metal costs money, so place() names what to buy and submit()
refuses to run it.

A sleep group places as ONE unit onto one multi-regime host; concurrent members
place per member, because colocation is only a hint and is semantics-neutral
(I5). submit() then runs the experiment where the learner landed — the learner
is never remote — and reaches every other partition through RemotePools.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from rlstack.data.stores.base import Store
from rlstack.policy.siteschema import SiteSchema
from rlstack.runner.host import Host, Partition, Regime
from rlstack.runner.interfaces import Engine, Learner
from rlstack.runner.loop import RunReport, experiment_identity
from rlstack.runner.remote import (
    HostService, LocalTransport, RemoteHost, RemotePool,
)
from rlstack.spec.specs import ExperimentSpec, PoolMember


class FleetError(RuntimeError):
    """A placement the fleet may not decide alone (acquire is a human's) or
    a plan it cannot execute."""


@dataclass(frozen=True)
class Metal:
    """Owned metal: one GpuSet the fleet may carve. Registering Metal IS the
    acquire rung executed — the rung that costs money, so the one a human
    does. `gpu` is the KIND (what was bought) and `vram_gb` one device's VRAM
    in GB (what a model is measured against — L4=24, A100=40 or 80, H100=80);
    a carve stamps the kind onto the Partition it births."""

    name: str
    gpu: str = "L4"
    devices: int = 1
    vram_gb: float = 24.0


def fraction_for_gb(gb: float, metal: Metal) -> float:
    """THE meeting point of the unit a human sizes models in (GB of VRAM) and
    the unit a partition owns (a fraction of ONE device): the carve hint `gb`
    becomes gb / metal.vram_gb on the target metal. Partition.memory stays a
    fraction because both substrates take one, so a GB figure is converted
    HERE, against the metal it will live on, and never stored. More GB than
    one device holds is not a smaller fraction, it is bigger metal — the
    acquire rung — so it raises instead of clamping."""
    fraction = gb / metal.vram_gb
    if fraction > 1.0 + 1e-9:
        raise FleetError(
            f"{gb:g} GB is more than one {metal.gpu} device holds "
            f"({metal.vram_gb:g} GB on {metal.name!r}) — VRAM per shard past "
            f"one device is the acquire rung (bigger metal), not a fraction")
    return fraction


@dataclass(frozen=True)
class Demand:
    """One member's capability demand, read off the spec: WHAT is needed
    (capability, base, shard shape), never WHERE. `memory` is the carve hint —
    it sizes a new partition at rung two and is ignored on a join. `pool`
    None is the learner member."""

    pool: str | None
    capability: str                 # "inference" | "training"
    base: str
    shape: int
    memory: float
    group: int
    sharing: str


@dataclass(frozen=True)
class Join:
    """Rung one: the capability already exists — attach to its host."""

    host: str
    demand: Demand


@dataclass(frozen=True)
class Carve:
    """Rung two: partition a new host out of residual metal. One regime per
    demand in the unit; several regimes mean the carved host alternates."""

    metal: str
    devices: tuple[int, ...]
    memory: float
    regimes: tuple[Regime, ...]
    demands: tuple[Demand, ...]


@dataclass(frozen=True)
class Acquire:
    """Rung three: nothing fits — new metal, a human's call."""

    gpu: str
    devices: int
    demands: tuple[Demand, ...]


@dataclass(frozen=True)
class Plan:
    """A placement, as data: joins and carves execute automatically;
    one acquire step makes the whole plan a human's."""

    steps: tuple[Join | Carve | Acquire, ...]

    @property
    def acquires(self) -> tuple[Acquire, ...]:
        return tuple(s for s in self.steps if isinstance(s, Acquire))

    @property
    def needs_human(self) -> bool:
        return bool(self.acquires)


def demands_of(spec: ExperimentSpec) -> tuple[Demand, ...]:
    """The spec's gpu_config as capability demands. A member with no
    declared fraction demands a WHOLE device per shard when carved — the
    safe default; sub-device partitions are opt-in via fractions."""
    out: list[Demand] = []
    for gi, group in enumerate(spec.gpu_config.groups):
        for member in group.members:
            if isinstance(member, PoolMember):
                out.append(Demand(
                    pool=member.name, capability="inference",
                    base=member.base or spec.policy.base, shape=member.tp,
                    memory=member.fraction if member.fraction is not None else 1.0,
                    group=gi, sharing=group.sharing))
            else:
                out.append(Demand(
                    pool=None, capability="training", base=spec.policy.base,
                    shape=member.fsdp,
                    memory=member.fraction if member.fraction is not None else 1.0,
                    group=gi, sharing=group.sharing))
    return tuple(out)


def placement_units(demands: Sequence[Demand]) -> tuple[tuple[Demand, ...], ...]:
    """Sleep groups place as ONE unit (their members alternate on one
    partition, so they must land together — one host wearing masks);
    concurrent members place one by one (per-capability hosts)."""
    units: list[tuple[Demand, ...]] = []
    seen_sleep: set[int] = set()
    for demand in demands:
        if demand.sharing == "sleep":
            if demand.group in seen_sleep:
                continue
            seen_sleep.add(demand.group)
            units.append(tuple(d for d in demands if d.group == demand.group))
        else:
            units.append((demand,))
    return tuple(units)


def regime_of(demand: Demand) -> Regime:
    """The regime a carve births for one demand — named by capability, so a
    carved host's name reads as what it serves."""
    if demand.capability == "inference":
        return Regime(name=f"{demand.pool}-tp{demand.shape}",
                      capability="inference",
                      base=demand.base, shape=demand.shape)
    return Regime(name=f"learner-fsdp{demand.shape}", capability="training",
                  base=demand.base, shape=demand.shape)


class Fleet:
    """The inventory and the ladder. Factories make a carve's metal (an
    engine per inference regime, a learner per training regime) — fakes in
    tests, vLLM/torch builders on real metal; the Host constructor attests
    the result against the regimes either way.

    A factory is handed BOTH birth facts of the host it is building: the
    Regime (what capability) and the Partition (how much of what metal). The
    fraction is the whole point of a sub-GPU host and only the carve knows it,
    so the contract that realizes a partition is paid it rather than left to
    re-derive it off the plan."""

    def __init__(self, metal: Sequence[Metal], *, store: Store,
                 engine_factory: Callable[[Regime, Partition], Engine],
                 learner_factory: Callable[[Regime, Partition], Learner],
                 hosts: Sequence[Host] = ()) -> None:
        self.metal = {m.name: m for m in metal}
        self.store = store
        self.engine_factory = engine_factory
        self.learner_factory = learner_factory
        self.hosts: dict[str, Host] = {}
        self.carves = 0                 # carve ordinal, for unique host names
        for host in hosts:
            self.register(host)

    def register(self, host: Host) -> None:
        """A host's name is its identity in the fleet: registration REFUSES a
        name already taken instead of replacing the host that holds it.

        Replacing is never right, even when names are unique by construction
        (carve_name). The replaced host keeps its metal — engines resident,
        tenants bound, arbiter admitting — while dropping out of the dict
        residual() sums over, so its fraction silently returns to the residual
        and the next carve is sized against memory that is already gone."""
        if host.name in self.hosts:
            raise FleetError(
                f"host {host.name!r} is already registered with this fleet; "
                f"a host is never replaced — its metal outlives the dict "
                f"entry, and the residual would count its partition free")
        self.hosts[host.name] = host

    # ---- the inventory ------------------------------------------------------

    def residual(self, metal_name: str) -> list[float]:
        """Free memory per device — capacity no partition owns. The residual
        is NOBODY'S, which is exactly why carving from it needs no
        approval."""
        free = [1.0] * self.metal[metal_name].devices
        for host in self.hosts.values():
            part = host.partition
            if part is None or part.metal != metal_name:
                continue
            for device in part.devices:
                free[device] -= part.memory
        return free

    # ---- the ladder, one named method per rung ------------------------------

    def find_join(self, unit: tuple[Demand, ...]) -> Host | None:
        """Rung one: ONE host whose regimes cover every demand in the unit.
        Coverage is capability equality — same capability, base, and shape; a
        fraction never enters (the weights already live there).

        A SOLO host that is already running an experiment is skipped rather
        than offered and then refused: soloness is a birth fact, so it belongs
        to placement, and the ladder falls through to carve exactly as it does
        for a host that lacks the capability at all.
        """
        for name in sorted(self.hosts):
            host = self.hosts[name]
            if host.solo and host.occupied():
                continue
            if all(self._covers(host, demand) for demand in unit):
                return host
        return None

    @staticmethod
    def _covers(host: Host, demand: Demand) -> bool:
        return any(regime.capability == demand.capability
                   and regime.base == demand.base
                   and regime.shape == demand.shape
                   for regime in host.regimes)

    def plan_carve(self, unit: tuple[Demand, ...],
                   booked: dict[str, dict[int, float]]) -> Carve | None:
        """Rung two: first-fit over residual, in registered-metal order —
        `shape` devices each with the unit's memory free. `booked` carries
        this plan's earlier carves so one plan never double-books a device.
        A sleep unit's members alternate, so it needs max(shape) devices at
        max(memory) — one partition, worn in turns."""
        need_devices = max(demand.shape for demand in unit)
        need_memory = max(demand.memory for demand in unit)
        for metal_name in sorted(self.metal):
            free = self.residual(metal_name)
            for device, used in booked.get(metal_name, {}).items():
                free[device] -= used
            chosen = [i for i, f in enumerate(free)
                      if f >= need_memory - 1e-9][:need_devices]
            if len(chosen) == need_devices:
                return Carve(metal=metal_name, devices=tuple(chosen),
                             memory=need_memory,
                             regimes=tuple(regime_of(d) for d in unit),
                             demands=unit)
        return None

    def place(self, spec: ExperimentSpec) -> Plan:
        """The ladder, per placement unit: JOIN if some host already serves
        the whole unit; else CARVE from residual; else ACQUIRE (a human).
        Pure planning — nothing is built or journaled until apply()."""
        steps: list[Join | Carve | Acquire] = []
        booked: dict[str, dict[int, float]] = {}
        for unit in placement_units(demands_of(spec)):
            host = self.find_join(unit)
            if host is not None:
                steps.extend(Join(host.name, demand) for demand in unit)
                continue
            carve = self.plan_carve(unit, booked)
            if carve is not None:
                slate = booked.setdefault(carve.metal, {})
                for device in carve.devices:
                    slate[device] = slate.get(device, 0.0) + carve.memory
                steps.append(carve)
                continue
            gpu = (next(iter(self.metal.values())).gpu if self.metal
                   else "GPU")
            steps.append(Acquire(gpu=gpu,
                                 devices=max(d.shape for d in unit),
                                 demands=unit))
        return Plan(tuple(steps))

    def apply(self, plan: Plan) -> dict[str | None, Host]:
        """Execute the automatic rungs: joins resolve to their hosts, carves
        BUILD hosts (the factories make the metal, the Host constructor
        attests it and journals host-up, the fleet journals the carve).
        Refuses a plan with acquire steps — that rung is a human's.
        Returns demand.pool -> serving host (None key: the learner's)."""
        if plan.needs_human:
            raise FleetError(
                "the plan needs new metal — a human's call: "
                + "; ".join(f"{a.devices}x {a.gpu} for "
                            f"{sorted(d.pool or 'learner' for d in a.demands)}"
                            for a in plan.acquires)
                + ". Register Metal(...) with the fleet and resubmit.")
        placement: dict[str | None, Host] = {}
        for step in plan.steps:
            if isinstance(step, Join):
                placement[step.demand.pool] = self.hosts[step.host]
            else:
                host = self.carve(step)
                for demand in step.demands:
                    placement[demand.pool] = host
        return placement

    def carve_name(self, step: Carve) -> str:
        """The name a carved host is born with: the metal it came from, the
        devices it owns, the regimes it wears, and a per-fleet CARVE ORDINAL.

        The ordinal is not decoration. A regime's name carries capability and
        shape but not base, so two carves that differ only by base would otherwise
        produce one name — and the base cannot go in the name either, because
        a base is "Qwen/Qwen3-0.6B" and a host name is a journal path segment
        that may hold no "/" (Host attests it). No separator here is "/" for
        that same reason."""
        self.carves += 1
        devices = "-".join(str(d) for d in step.devices)
        regimes = "+".join(regime.name for regime in step.regimes)
        return f"{step.metal}:{devices}.{regimes}.c{self.carves}"

    def carve(self, step: Carve) -> Host:
        """Rung two executed, under the three conditions that make it
        automatic: residual-only by construction (plan_carve drew from
        residual), never mutating an existing host (a NEW Host is born with
        its capability), journaled — legibility by record, not by approval.
        The born partition is STAMPED with the metal's GPU kind: the fraction
        says how much, the kind says of what."""
        metal = self.metal[step.metal]
        partition = Partition(step.metal, step.devices, step.memory, metal.gpu)
        engines: list[Engine] = []
        carved_learner: Learner | None = None
        for regime in step.regimes:
            if regime.capability == "inference":
                engines.append(self.engine_factory(regime, partition))
            else:
                carved_learner = self.learner_factory(regime, partition)
        name = self.carve_name(step)
        host = Host(name, engines=tuple(engines), learner=carved_learner,
                    store=self.store, partition=partition,
                    regimes=step.regimes)
        self.register(host)
        self.store.append_fleet_event({
            "event": "carve", "t": time.time(), "host": name,
            "metal": step.metal, "gpu": metal.gpu,
            "devices": list(step.devices), "memory": step.memory,
            "regimes": [{"name": r.name, "capability": r.capability,
                         "base": r.base, "shape": r.shape}
                        for r in step.regimes]})
        return host

    # ---- submit -------------------------------------------------------------

    async def submit(self, spec: ExperimentSpec, schema: SiteSchema,
                     store: Store | None = None,
                     max_inflight: int = 64) -> RunReport:
        """place → apply → run. The runner goes to the learner's host; every
        pool that landed elsewhere is reached through a RemotePool over a
        LocalTransport — every host in this process, which the Modal-cls
        transport replaces behind the same two verbs. The placement is
        journaled under the run's identity before the run opens."""
        plan = self.place(spec)
        placement = self.apply(plan)          # raises FleetError on acquire
        rid = experiment_identity(spec, schema)
        self.store.append_fleet_event({
            "event": "place", "t": time.time(), "run_id": rid,
            "steps": [_step_row(step) for step in plan.steps]})
        learner_host = placement.get(None)
        if learner_host is None:
            raise FleetError(
                "the spec declares no learner member — the fleet runs "
                "training specs (generation-only runs are an open thread)")
        remotes: dict[str, Engine] = {}
        for demand in demands_of(spec):
            if demand.pool is None:
                continue
            serving = placement[demand.pool]
            if serving is learner_host:
                continue
            remotes[demand.pool] = RemotePool(
                LocalTransport(HostService(serving)),
                base=demand.base, tp=demand.shape)
        return await learner_host.submit(spec, schema, store=store,
                                         max_inflight=max_inflight,
                                         remotes=remotes)


def _step_row(step: Join | Carve | Acquire) -> dict:
    """One plan step as a journal row."""
    if isinstance(step, Join):
        return {"rung": "join", "host": step.host,
                "pool": step.demand.pool or "learner"}
    if isinstance(step, Carve):
        return {"rung": "carve", "metal": step.metal,
                "devices": list(step.devices), "memory": step.memory,
                "regimes": [r.name for r in step.regimes]}
    return {"rung": "acquire", "gpu": step.gpu, "devices": step.devices,
            "pools": sorted(d.pool or "learner" for d in step.demands)}


# ---------------------------------------------------------------------------
# the standing fleet: one desk, many partitions, none of them in-process
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Listing:
    """One standing host as the desk knows it: the BIRTH FACTS placement
    matches on (regimes, solo), the ADDRESS other hosts dial its pools at,
    and the RemoteHost the desk itself adopts through. A listing is a
    description, never the host — the metal lives in the host's own
    container, which is the entire reason the desk can be a CPU process."""

    name: str
    regimes: tuple[Regime, ...]
    address: str
    solo: bool
    host: "RemoteHost"

    def occupied(self) -> bool:
        """Asked over the wire, at placement time only: the roster is the
        host's, and the desk holds no copy that could go stale."""
        return any(t.get("status") == "running"
                   for t in self.host.status().get("tenants", {}).values())

    def alive(self) -> bool:
        """Does the container behind this listing still answer? A listing is
        a description, so the only way to know is to ask — at placement time,
        never cached: a host that died between placements must not be offered,
        and one that came back must not stay buried."""
        try:
            self.host.status()
            return True
        except Exception:
            return False


class FleetService:
    """The standing fleet: placement as a SERVICE, and the fleet journal's
    ONE WRITER — the Trainer/ledger pattern applied to the fleet plane.

    In-process, Fleet holds live Hosts and may carve because the factories
    are beside the metal. Standing, none of that is true: hosts are GPU
    containers elsewhere, so the desk holds LISTINGS (deploy-registered at
    each host's boot), the join rung matches against them, and "carve" is a
    venue action — boot a container wearing the regimes, list it — which is
    why a placement nothing covers comes back as a BOOT instruction, the
    standing world's acquire. Truth stays in the store: every listing and
    every placement is journaled, `from_journal` rebuilds the desk after a
    kill, and the desk's memory is only the single writer's cache. Multi-user
    concurrency is exactly this single-writer property: two campaigns
    submitting at once serialize through one desk instead of double-reading
    one residual.

    Serves over the same Transport contract as HostService: `submit` on the
    async path (it ends in an adopt at the learner's host), `status` on the
    sync one.
    """

    def __init__(self, store: Store,
                 connect: Callable[[str], "RemoteHost"]) -> None:
        self.store = store
        self.connect = connect          # address -> RemoteHost: the desk's dialer
        self.listings: dict[str, Listing] = {}

    @classmethod
    def from_journal(cls, store: Store,
                     connect: Callable[[str], "RemoteHost"]) -> "FleetService":
        """The desk, rebuilt from its own record: every `list` event redials.
        Kill -9 the desk and nothing was lost but a process — the same
        recovery shape as attach, on the fleet plane."""
        desk = cls(store, connect)
        for event in store.read_fleet_log():
            if event.get("event") == "list":
                desk.listings[event["host"]] = _listing_from(event, connect)
            elif event.get("event") == "delist":
                desk.listings.pop(event["host"], None)
        return desk

    def list_host(self, name: str, regimes: Sequence[Regime], address: str,
                  solo: bool = False) -> None:
        """A host enters the standing fleet: the deploy that booted it lists
        it here, once, and the desk journals the listing so a rebuilt desk
        knows it too. Refuses a taken name for Fleet.register's reason."""
        if name in self.listings:
            raise FleetError(
                f"host {name!r} is already listed with this desk; a listing "
                f"is never replaced — delist first if the container is gone")
        self.listings[name] = Listing(name=name, regimes=tuple(regimes),
                                      address=address, solo=solo,
                                      host=self.connect(address))
        self.store.append_fleet_event({
            "event": "list", "t": time.time(), "host": name,
            "address": address, "solo": solo,
            "regimes": [{"name": r.name, "capability": r.capability,
                         "base": r.base, "shape": r.shape} for r in regimes]})

    def delist(self, name: str) -> None:
        """A host leaves the standing fleet — its container is gone, or the
        deploy is retiring it. Journaled like the listing was, so a rebuilt
        desk knows the departure too; the metal itself was never the desk's
        to touch."""
        if name not in self.listings:
            raise FleetError(f"host {name!r} is not listed with this desk")
        del self.listings[name]
        self.store.append_fleet_event({
            "event": "delist", "t": time.time(), "host": name})

    # ---- placement over listings (the join rung; carve is a venue action) ---

    def find_listing(self, unit: tuple[Demand, ...]) -> Listing | None:
        """Rung one over listings: sorted-name order, coverage by capability
        equality (Fleet.find_join's rule, matched against descriptions),
        solo-and-occupied skipped — and so is a listing whose container no
        longer ANSWERS: placement must never offer a host it cannot reach,
        and a dead listing is a fact discovered here, reported by the boot
        refusal, and cured by a delist or a reboot."""
        for name in sorted(self.listings):
            listing = self.listings[name]
            if not all(_covers_regimes(listing.regimes, d) for d in unit):
                continue
            if not listing.alive():
                continue
            if listing.solo and listing.occupied():
                continue
            return listing
        return None

    def place_listings(self, spec: ExperimentSpec
                       ) -> tuple[dict[str | None, Listing], list[dict]]:
        """Every placement unit onto a listing, or the boot list: what a
        placement nothing serves needs BOOTED — regimes to wear, memory to
        own — which the deploy executes and lists, the standing carve."""
        placement: dict[str | None, Listing] = {}
        boot: list[dict] = []
        for unit in placement_units(demands_of(spec)):
            listing = self.find_listing(unit)
            if listing is None:
                boot.append({
                    "regimes": [regime_of(d).name for d in unit],
                    "capabilities": sorted({d.capability for d in unit}),
                    "base": unit[0].base,
                    "memory": max(d.memory for d in unit)})
                continue
            for demand in unit:
                placement[demand.pool] = listing
        return placement, boot

    # ---- submit: place, journal, adopt --------------------------------------

    async def submit(self, spec_row: Mapping) -> dict:
        """One frame in, one placement out: decode, place over the listings,
        journal, and ADOPT at the learner's listing with every other pool's
        address threaded as routes. The reply is the host's own adopt reply
        plus where everything landed; the run itself is the adopted host's
        business, and the ledger is the result channel there as everywhere."""
        from rlstack.runner.remote import spec_from_json

        spec = spec_from_json(spec_row)
        if not isinstance(spec, ExperimentSpec):
            return {"accepted": False,
                    "error": f"submit expects an ExperimentSpec's canonical "
                             f"form, decoded {type(spec).__name__}"}
        placement, boot = self.place_listings(spec)
        if boot:
            return {"accepted": False, "boot": boot,
                    "error": "no listed host serves these units — boot hosts "
                             "wearing the named regimes and list them (the "
                             "standing carve is a venue action)"}
        learner_listing = placement.get(None)
        if learner_listing is None:
            return {"accepted": False,
                    "error": "the spec declares no learner member — the desk "
                             "places training specs"}
        routes = {pool: listing.address
                  for pool, listing in placement.items()
                  if pool is not None and listing is not learner_listing}
        try:
            reply = await learner_listing.host.adopt(spec_row, routes)
        except Exception as down:
            # alive() passed and the container died between the probe and the
            # knock: the reply says so instead of the desk falling over, and
            # the cure is a delist or a reboot, both venue actions
            return {"accepted": False, "host": learner_listing.name,
                    "error": f"host {learner_listing.name!r} did not answer "
                             f"the adopt: {down}"}
        self.store.append_fleet_event({
            "event": "place", "t": time.time(),
            "run_id": reply.get("run_id"),
            "host": learner_listing.name,
            "pools": {pool or "learner": listing.name
                      for pool, listing in placement.items()},
            "accepted": bool(reply.get("accepted"))})
        return {**reply, "host": learner_listing.name,
                "pools": {pool or "learner": listing.name
                          for pool, listing in placement.items()}}

    def status(self) -> dict:
        """The desk's inventory, no wire calls: what is listed and what it
        wears. Occupancy is asked per placement, never cached here."""
        return {"listings": {name: {
            "address": listing.address, "solo": listing.solo,
            "regimes": [r.name for r in listing.regimes]}
            for name, listing in sorted(self.listings.items())}}

    # ---- the Transport surface (HostService's contract, fleet-addressed) ----

    async def serve(self, verb: str, payload: dict) -> dict:
        if verb == "submit":
            return await self.submit(payload["spec"])
        raise ValueError(f"unknown fleet verb {verb!r}")

    def answer(self, verb: str, payload: dict) -> dict:
        if verb == "status":
            return self.status()
        raise ValueError(f"unknown admission-free fleet verb {verb!r}")


def _covers_regimes(regimes: Sequence[Regime], demand: Demand) -> bool:
    """Coverage is capability equality — Fleet._covers, over a description."""
    return any(regime.capability == demand.capability
               and regime.base == demand.base
               and regime.shape == demand.shape
               for regime in regimes)


def _listing_from(event: Mapping,
                  connect: Callable[[str], "RemoteHost"]) -> Listing:
    """A journal `list` event back as a Listing — from_journal's one row."""
    return Listing(
        name=event["host"],
        regimes=tuple(Regime(r["name"], r["capability"], r["base"], r["shape"])
                      for r in event["regimes"]),
        address=event["address"], solo=bool(event.get("solo", False)),
        host=connect(event["address"]))
