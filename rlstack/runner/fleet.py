"""The Fleet: metal inventory + the join / carve / acquire ladder (#43).

Hosts are atomic purposed partitions (runner/host.py); the fleet is what
knows all of them, plus the registered Metal they were carved from and the
RESIDUAL — capacity no partition owns. An experiment declares capability
demands (what, never where: base + shard shape per member, read straight off
its gpu_config); placement climbs a three-rung ladder, one currency and one
decider per rung:

    JOIN      a host already serves the demanded capability. Automatic — the
              target host's own arbiter is the decider (admission stays with
              the metal). Declared fractions are IGNORED: the weights
              already live there; a join's currencies are adapter slots and
              contention, not memory.
    CARVE     nothing serves it, but residual metal fits: partition a new
              host into existence. Automatic, because the residual is
              nobody's — under three conditions that keep the fleet legible:
              residual-only (an existing host is NEVER shrunk or reshaped;
              capability is a birth fact), journaled (fleet/log.jsonl), and
              the declared fraction finally means something (it sizes the
              new partition — the carve hint).
    ACQUIRE   nothing fits: new metal costs money, so a human registers
              Metal. place() returns the plan naming what to buy; submit()
              refuses to run it.

Placement units: a sleep group places as ONE unit onto ONE host (alternation
is an intra-partition fact), carved as a single multi-regime host when no
host covers it. A concurrent group places PER MEMBER — per-capability hosts,
so a later experiment wanting just the tp-4 teacher contacts that host and
nothing else. Concurrent grouping was only ever a colocation hint, and
colocation is semantics-neutral (I5).

submit() then runs the experiment where the learner landed (the learner is
never remote — the runner goes to it) and reaches every other partition
through RemotePools over the wire.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rlstack.data.stores.base import Store
from rlstack.policy.siteschema import SiteSchema
from rlstack.runner.host import Host, Partition, Regime
from rlstack.runner.interfaces import Engine, Learner
from rlstack.runner.loop import RunReport, experiment_identity
from rlstack.runner.remote import HostService, LocalTransport, RemotePool
from rlstack.spec.specs import ExperimentSpec, PoolMember


class FleetError(RuntimeError):
    """A placement the fleet may not decide alone (acquire is a human's) or
    a plan it cannot execute."""


@dataclass(frozen=True)
class Metal:
    """Owned metal: one GpuSet the fleet may carve. Registering Metal IS the
    acquire rung executed — the one that costs money, so the one a human
    does; everything below it is automatic."""

    name: str
    gpu: str = "L4"
    devices: int = 1


@dataclass(frozen=True)
class Demand:
    """One member's capability demand, read off the spec: WHAT is needed
    (kind, base, shard shape), never WHERE. `memory` is the carve hint —
    it sizes a new partition at rung two and is ignored on a join. `pool`
    None is the learner member."""

    pool: str | None
    kind: str                       # "inference" | "training"
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
                    pool=member.name, kind="inference",
                    base=member.base or spec.policy.base, shape=member.tp,
                    memory=member.fraction if member.fraction is not None else 1.0,
                    group=gi, sharing=group.sharing))
            else:
                out.append(Demand(
                    pool=None, kind="training", base=spec.policy.base,
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
    if demand.kind == "inference":
        return Regime(name=f"{demand.pool}-tp{demand.shape}", kind="inference",
                      base=demand.base, shape=demand.shape)
    return Regime(name=f"learner-fsdp{demand.shape}", kind="training",
                  base=demand.base, shape=demand.shape)


class Fleet:
    """The inventory and the ladder. Factories make a carve's metal (an
    engine per inference regime, a learner per training regime) — fakes in
    tests, vLLM/torch builders on real metal; the Host constructor attests
    the result against the regimes either way."""

    def __init__(self, metal: Sequence[Metal], *, store: Store,
                 engine_factory: Callable[[Regime], Engine],
                 learner_factory: Callable[[Regime], Learner],
                 hosts: Sequence[Host] = ()) -> None:
        self.metal = {m.name: m for m in metal}
        self.store = store
        self.engine_factory = engine_factory
        self.learner_factory = learner_factory
        self.hosts: dict[str, Host] = {h.name: h for h in hosts}

    # ---- the inventory ------------------------------------------------------

    def residual(self, metal_name: str) -> list[float]:
        """Free memory per device: what carving may draw from. The residual
        is NOBODY'S — which is exactly why carving from it needs no
        approval (#43)."""
        free = [1.0] * self.metal[metal_name].devices
        for host in self.hosts.values():
            part = host.partition
            if part is None or part.gpuset != metal_name:
                continue
            for device in part.devices:
                free[device] -= part.memory
        return free

    # ---- the ladder, one named method per rung ------------------------------

    def find_join(self, unit: tuple[Demand, ...]) -> Host | None:
        """Rung one: ONE host whose regimes cover every demand in the unit.
        Coverage is capability equality — same kind, base, and shape; a
        fraction never enters (the weights already live there)."""
        for name in sorted(self.hosts):
            host = self.hosts[name]
            if all(self._covers(host, demand) for demand in unit):
                return host
        return None

    @staticmethod
    def _covers(host: Host, demand: Demand) -> bool:
        return any(regime.kind == demand.kind and regime.base == demand.base
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

    def carve(self, step: Carve) -> Host:
        """Rung two executed. Residual-only by construction (plan_carve drew
        from residual), never mutates an existing host (a NEW Host is born
        with its capability), journaled (legibility by record, not by
        approval — #43)."""
        engines: list[Engine] = []
        carved_learner: Learner | None = None
        for regime in step.regimes:
            if regime.kind == "inference":
                engines.append(self.engine_factory(regime))
            else:
                carved_learner = self.learner_factory(regime)
        name = (f"{step.metal}:{'-'.join(str(d) for d in step.devices)}/"
                + "+".join(regime.name for regime in step.regimes))
        host = Host(name, engines=tuple(engines), learner=carved_learner,
                    store=self.store,
                    partition=Partition(step.metal, step.devices, step.memory),
                    regimes=step.regimes)
        self.hosts[name] = host
        self.store.append_fleet_event({
            "event": "carve", "t": time.time(), "host": name,
            "metal": step.metal, "devices": list(step.devices),
            "memory": step.memory,
            "regimes": [{"name": r.name, "kind": r.kind, "base": r.base,
                         "shape": r.shape} for r in step.regimes]})
        return host

    # ---- submit -------------------------------------------------------------

    async def submit(self, spec: ExperimentSpec, schema: SiteSchema,
                     store: Store | None = None,
                     max_inflight: int = 64) -> RunReport:
        """place → apply → run. The runner goes to the learner's host; every
        pool that landed elsewhere is reached through a RemotePool over a
        LocalTransport (v0: all hosts share this process; the Modal-cls
        transport slots in behind the same two verbs). The placement is
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
