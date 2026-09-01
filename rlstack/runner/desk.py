"""The Desk: a demand allocator, and the metal plane it commands.

The desk is WORKLOAD-BLIND. Its whole vocabulary is Demands (what capability,
what base, what shard shape — never where, never why), listings, metal, and
addresses; a workload reaches it as demand rows plus an OPAQUE FRAME the desk
relays to the anchor demand's host without decoding. What an experiment is —
specs, identity, plans, code claims — lives with the host that adopts it and
in the campaign layer (runner/campaign.py), where specs are turned INTO
demands. The acid test of the boundary: this module imports no spec class.

Placement climbs the same rungs it always has (I12): JOIN a listing that
already serves the capability (coverage is capability equality; solo-and-
occupied skipped, dead listings skipped), CARVE a new host out of a
registered metal's residual (the desk deduces, the metal's own books
enforce), or answer with BOOT instructions — new metal costs money, so
acquiring stays a human's. A sleep group places as ONE unit onto one
multi-regime host; concurrent members place per member (I5: colocation is a
hint, never semantics).

Truth stays in the store: every listing, metal registration, placement and
delisting is journaled, `from_journal` rebuilds the desk after a kill, and
the desk's memory is only the single writer's cache — "the fleet" names the
aggregate this journal records, and the desk is that journal's one writer.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from rlstack.data.stores.base import Store
from rlstack.policy.siteschema import SiteSchema
from rlstack.runner.host import Host, Partition, Regime
from rlstack.runner.interfaces import Engine, Learner
from rlstack.runner.remote import (
    HostService, LocalTransport, RemoteHost, RemotePool, Transport,
)


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
    # the ANCHOR is where a delivered frame lands — the one demand whose host
    # receives the workload. The desk never knows WHY (for an experiment it is
    # the learner, because the learner is never remote — but that rule lives
    # with whoever built the demands, not here).
    anchor: bool = False


def demand_rows(demands: Sequence[Demand]) -> list[dict]:
    """Demands as wire rows — the desk's whole input vocabulary."""
    return [{"pool": d.pool, "capability": d.capability, "base": d.base,
             "shape": d.shape, "memory": d.memory, "group": d.group,
             "sharing": d.sharing, "anchor": d.anchor} for d in demands]


def demands_from(rows: Sequence[Mapping]) -> tuple[Demand, ...]:
    """Wire rows back as Demands — demand_rows' typed inverse."""
    return tuple(Demand(
        pool=row["pool"], capability=row["capability"], base=row["base"],
        shape=int(row["shape"]), memory=float(row["memory"]),
        group=int(row["group"]), sharing=row["sharing"],
        anchor=bool(row.get("anchor", False))) for row in rows)


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
    # The VIEW half of capacity (the metal's books are the ENFORCEMENT half):
    # the partition row this host was born onto and the registered Metal it
    # lives on. Descriptions like everything else here — the desk deduces
    # from them, plans over them, and shows them; only the metal's own
    # booking refuses over them. Empty for hosts listed before the row rode
    # the frame.
    partition: Mapping | None = None
    metal: str = ""

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


class Desk:
    """Placement as a SERVICE, and the fleet journal's ONE WRITER — the
    Trainer/ledger pattern applied to the fleet plane. Workload-blind: its
    input is Demands, its output is addresses, and a delivered frame passes
    through it unread (the module docstring's boundary).

    Hosts are containers elsewhere, so the desk holds LISTINGS (phone-home
    registered at each host's boot), the join rung matches against them, and
    the standing CARVE is a command sent to a registered metal's own
    container: the desk
    DEDUCES from the metal's residual and the metal ENFORCES with a booking
    (MetalService), so a placement nothing covers comes back as a BOOT
    instruction only when no registered metal can hold it — the standing
    world's acquire. Truth stays in the store: every listing, every metal,
    and every placement is journaled, `from_journal` rebuilds the desk after
    a kill, and the desk's memory is only the single writer's cache.
    Multi-user concurrency is exactly this single-writer property: two
    campaigns submitting at once serialize through one desk instead of
    double-reading one residual.

    Serves over the same Transport contract as HostService: `submit` on the
    async path (it ends in an adopt at the learner's host), `status` on the
    sync one.
    """

    def __init__(self, store: Store,
                 connect: Callable[[str], "RemoteHost"],
                 connect_metal: Callable[[str], "RemoteMetal"] | None = None,
                 ) -> None:
        self.store = store
        self.connect = connect          # address -> RemoteHost: the desk's dialer
        # address -> RemoteMetal: the dialer for the METAL PLANE — the verbs
        # that create and free hosts (carve/decarve) and the residual the
        # desk deduces from. Venue like `connect`; a desk born without it
        # still places over listings and answers misses with boot
        # instructions, it just cannot command a carve.
        self.connect_metal = connect_metal
        self.metal: dict[str, Metal] = {}
        self.metal_remotes: dict[str, "RemoteMetal"] = {}
        self.listings: dict[str, Listing] = {}

    @classmethod
    def from_journal(cls, store: Store,
                     connect: Callable[[str], "RemoteHost"],
                     connect_metal: Callable[[str], "RemoteMetal"] | None = None,
                     ) -> "Desk":
        """The desk, rebuilt from its own record: every `list` event redials,
        every addressed `metal` event redials the metal plane. Kill -9 the
        desk and nothing was lost but a process — the same recovery shape as
        attach, on the fleet plane."""
        desk = cls(store, connect, connect_metal)
        for event in store.read_fleet_log():
            if event.get("event") == "list":
                desk.listings[event["host"]] = _listing_from(event, connect)
            elif event.get("event") == "delist":
                desk.listings.pop(event["host"], None)
            elif event.get("event") == "metal":
                desk.metal[event["name"]] = Metal(
                    name=event["name"], gpu=event.get("gpu", "L4"),
                    devices=int(event.get("devices", 1)),
                    vram_gb=float(event.get("vram_gb", 24.0)))
                address = event.get("address")
                if address and connect_metal is not None:
                    desk.metal_remotes[event["name"]] = connect_metal(address)
        return desk

    def register_metal(self, metal: Metal, address: str | None = None) -> None:
        """The acquire rung, recorded at the desk: what the fleet OWNS and may
        carve against. Journaled like a listing, so a rebuilt desk knows its
        inventory too. `address` is where that metal's own container answers
        the metal plane (carve/decarve/residual) — with it and a dialer the
        desk can command the standing carve; without it the row is inventory
        only and misses still answer with boot instructions."""
        if metal.name in self.metal:
            raise FleetError(f"metal {metal.name!r} is already registered")
        self.metal[metal.name] = metal
        if address and self.connect_metal is not None:
            self.metal_remotes[metal.name] = self.connect_metal(address)
        self.store.append_fleet_event({
            "event": "metal", "t": time.time(), "name": metal.name,
            "gpu": metal.gpu, "devices": metal.devices,
            "vram_gb": metal.vram_gb, "address": address})

    def list_host(self, name: str, regimes: Sequence[Regime], address: str,
                  solo: bool = False, partition: Mapping | None = None,
                  metal: str = "") -> None:
        """A host enters the standing fleet: the deploy that booted it lists
        it here, once, and the desk journals the listing so a rebuilt desk
        knows it too. `partition` and `metal` are the capacity VIEW — the row
        the host was born onto and the registered Metal it lives on — carried
        so the desk can deduce and the reaper can free; enforcement stays at
        the metal's own books. Refuses a taken name — a listing
        is never replaced."""
        if name in self.listings:
            raise FleetError(
                f"host {name!r} is already listed with this desk; a listing "
                f"is never replaced — delist first if the container is gone")
        self.listings[name] = Listing(name=name, regimes=tuple(regimes),
                                      address=address, solo=solo,
                                      host=self.connect(address),
                                      partition=dict(partition) if partition
                                      else None,
                                      metal=metal)
        self.store.append_fleet_event({
            "event": "list", "t": time.time(), "host": name,
            "address": address, "solo": solo,
            "partition": dict(partition) if partition else None,
            "metal": metal,
            "regimes": [{"name": r.name, "capability": r.capability,
                         "base": r.base, "shape": r.shape} for r in regimes]})

    def delist(self, name: str, reason: str = "") -> None:
        """A host leaves the standing fleet — its container is gone, the
        deploy is retiring it, or the reaper concluded it (the reason says
        which). Journaled like the listing was, so a rebuilt desk knows the
        departure too; the metal itself was never the desk's to touch."""
        if name not in self.listings:
            raise FleetError(f"host {name!r} is not listed with this desk")
        del self.listings[name]
        self.store.append_fleet_event({
            "event": "delist", "t": time.time(), "host": name,
            "reason": reason})

    # ---- placement over listings (the join rung; carve is a venue action) ---

    def find_listing(self, unit: tuple[Demand, ...],
                     avoid: frozenset[str] = frozenset()) -> Listing | None:
        """Rung one over listings: sorted-name order, coverage by capability
        equality (covers(), the one join rule, matched against descriptions),
        solo-and-occupied skipped — and so is a listing whose container no
        longer ANSWERS: placement must never offer a host it cannot reach,
        and a dead listing is a fact discovered here, reported by the boot
        refusal, and cured by a delist or a reboot. Names in `avoid` are off
        the table — a reroute excluding the listing being torn down."""
        for name in sorted(self.listings):
            if name in avoid:
                continue
            listing = self.listings[name]
            if not all(covers(listing.regimes, d) for d in unit):
                continue
            if not listing.alive():
                continue
            if listing.solo and listing.occupied():
                continue
            return listing
        return None

    async def place_listings(self, demands: Sequence[Demand],
                             avoid: frozenset[str] = frozenset(),
                             ) -> tuple[dict[str | None, Listing], list[dict]]:
        """Every placement unit onto a listing, a fresh carve, or the boot
        list: what no listed host serves is CARVED on a registered metal that
        can hold it, and only what no metal can hold lands in `boot` — the
        standing acquire, a human's. A unit carved for a placement whose
        LATER unit then missed stays listed: metal born is metal listed, and
        the next submit's join rung finds it. `avoid` passes to the join rung
        — a carve can never land on an avoided listing, because a carve is
        always a NEW name."""
        placement: dict[str | None, Listing] = {}
        boot: list[dict] = []
        for unit in placement_units(demands):
            listing = (self.find_listing(unit, avoid)
                       or await self.provision_unit(unit))
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

    async def provision_unit(self, unit: tuple[Demand, ...]) -> Listing | None:
        """The standing CARVE, desk-issued: nothing listed serves this unit,
        so the desk asks each registered metal whether it can hold it (the
        residual — the DEDUCTION) and COMMANDS the first that can (carve).
        The metal ENFORCES: it books the fraction synchronously at its own
        door, so a deduction gone stale between the ask and the command costs
        a refusal, never a double-book — and a refusal or a silent metal
        falls through to the next, then to the boot instructions. What comes
        back is journaled and listed HERE: the desk stays the fleet journal's
        one writer, which is exactly why the metal writes nothing."""
        if not self.metal_remotes:
            return None
        need_devices = max(demand.shape for demand in unit)
        need_memory = max(demand.memory for demand in unit)
        request = {
            "regimes": [{"name": regime_of(d).name,
                         "capability": d.capability,
                         "base": d.base, "shape": d.shape} for d in unit],
            "base": unit[0].base,
            "memory": need_memory,
        }
        for metal_name in sorted(self.metal_remotes):
            remote = self.metal_remotes[metal_name]
            try:
                free = remote.residual()
            except Exception:
                continue                # a silent metal is the reaper's, not ours
            if sum(1 for f in free if f >= need_memory - 1e-9) < need_devices:
                continue
            try:
                born = await remote.carve(request)
            except Exception:
                continue
            if not born.get("carved"):
                continue                # raced: booked away between ask and command
            corpse = self.listings.get(born["host"])
            if corpse is not None and corpse.metal == metal_name:
                # carve names EMBED the metal, so only this metal can re-mint
                # one — and a metal that re-minted a listed name has RECYCLED
                # (its counter reset), which means the old container and every
                # host on it are gone. The standing listing is a corpse by
                # construction (probing would lie: the newborn answers at the
                # same name-derived address); reap it in place. A collision
                # from any other source still hits list_host's refusal.
                self.delist(born["host"], reason="superseded by a new carve")
            self.store.append_fleet_event({
                "event": "provision", "t": time.time(), "host": born["host"],
                "metal": metal_name, "request": request,
                "partition": born.get("partition")})
            self.list_host(
                born["host"],
                tuple(Regime(r["name"], r["capability"], r["base"], r["shape"])
                      for r in born["regimes"]),
                born["address"], solo=bool(born.get("solo", False)),
                partition=born.get("partition"), metal=metal_name)
            return self.listings[born["host"]]
        return None

    # ---- place and submit: demands in, addresses (and one delivery) out -----

    async def place(self, demands: Sequence[Demand]) -> dict:
        """Demands in, addresses out — the PURE CLIENT's door: an evaluator,
        a scorer, anything that wants a pool without being a workload the
        fleet tracks. Same ladder as a delivery (join, carve, boot), same
        journal row; the client wraps the addresses itself and is thereafter
        just admitted traffic at each host."""
        placement, boot = await self.place_listings(demands)
        if boot:
            return {"placed": False, "boot": boot}
        pools = {demand.pool or "learner": placement[demand.pool].address
                 for demand in demands}
        self.store.append_fleet_event({
            "event": "place", "t": time.time(), "delivered": False,
            "pools": {pool or "learner": listing.name
                      for pool, listing in placement.items()}})
        return {"placed": True, "pools": pools}

    async def submit(self, rows: Sequence[Mapping], frame: Mapping) -> dict:
        """Demand rows plus one OPAQUE FRAME: place, then DELIVER the frame to
        the anchor demand's host with every other pool's address threaded as
        routes. The desk reads the frame's envelope (spec/code/subdir are the
        adopt wire's argument names) and never its contents — what the spec
        means is the anchor host's business, and the ledger is the result
        channel there as everywhere. Routes come off the DEMAND rows, which
        is what makes the blind relay possible at all — and the delivery is
        ARCHIVED (rows and frame ride the journaled placement, still unread),
        which is what makes a later reroute a replay."""
        demands = demands_from(rows)
        anchored = [d for d in demands if d.anchor]
        if len(anchored) != 1:
            return {"accepted": False,
                    "error": f"a delivery needs exactly one anchor demand "
                             f"(where the frame lands); got {len(anchored)}"}
        placement, boot = await self.place_listings(demands)
        if boot:
            return {"accepted": False, "boot": boot,
                    "error": "no listed host serves these units and no "
                             "registered metal can hold them — boot or "
                             "register metal wearing the named regimes (the "
                             "standing acquire is a human's)"}
        return await self.deliver(demands, placement, rows, frame)

    async def deliver(self, demands: Sequence[Demand],
                      placement: Mapping[str | None, Listing],
                      rows: Sequence[Mapping], frame: Mapping) -> dict:
        """The delivery half of a submission — submit's and reroute's ONE
        copy: the frame lands at the anchor demand's host with every other
        pool's address threaded as routes (read off the demand rows alone,
        the blind relay), and the placement is journaled WITH the rows and
        the frame — the archive a reroute replays without the desk ever
        having decoded it."""
        anchor = next(d for d in demands if d.anchor)
        anchor_listing = placement[anchor.pool]
        routes = {d.pool: placement[d.pool].address for d in demands
                  if not d.anchor and d.pool is not None
                  and placement[d.pool] is not anchor_listing}
        try:
            reply = await anchor_listing.host.adopt(
                frame.get("spec"), routes, frame.get("code"),
                frame.get("subdir"))
        except Exception as down:
            # alive() passed and the container died between the probe and the
            # knock: the reply says so instead of the desk falling over, and
            # the cure is a delist or a reboot, both venue actions
            return {"accepted": False, "host": anchor_listing.name,
                    "error": f"host {anchor_listing.name!r} did not answer "
                             f"the delivery: {down}"}
        pools = {pool or "learner": listing.name
                 for pool, listing in placement.items()}
        self.store.append_fleet_event({
            "event": "place", "t": time.time(), "delivered": True,
            "run_id": reply.get("run_id"),
            "host": anchor_listing.name, "pools": pools,
            "accepted": bool(reply.get("accepted")),
            "demands": [dict(row) for row in rows], "frame": dict(frame)})
        return {**reply, "host": anchor_listing.name, "pools": pools}

    def placements(self) -> dict[str, dict]:
        """The desk's CURRENT-BINDING table, derived on demand: the latest
        delivered placement per run_id — which listing each pool landed on,
        and (for deliveries since the archive existed) the demand rows and
        opaque frame that made it. Journal archaeology promoted to a read:
        dependents joins against it, reroute replays from it, an observer
        may render it. The frame stays as unread here as it was in flight."""
        table: dict[str, dict] = {}
        for event in self.store.read_fleet_log():
            if event.get("event") == "place" and event.get("run_id"):
                row = {"pools": event.get("pools", {}),
                       "host": event.get("host")}
                if event.get("demands") is not None:
                    row["demands"] = event["demands"]
                if event.get("frame") is not None:
                    row["frame"] = event["frame"]
                table[event["run_id"]] = row
        return table

    def dependents(self, name: str) -> list[str]:
        """Running runs whose LATEST journaled placement routes through
        listing `name` — the guard decommission refuses over. Occupancy
        alone would miss half of them: a serve host's roster is empty (a
        tenancy lives at its anchor), but every placement journaled the
        pools it landed on, and the rosters say which runs still run."""
        placed = {rid: row["pools"]
                  for rid, row in self.placements().items()}
        running: set[str] = set()
        for listing in self.listings.values():
            try:
                tenants = listing.host.status().get("tenants", {})
            except Exception:
                continue                 # a silent host holds nothing running
            running.update(rid for rid, told in tenants.items()
                           if told.get("status") == "running")
        return sorted(rid for rid, pools in placed.items()
                      if rid in running and name in pools.values())

    async def stop_anchored(self, run_id: str) -> dict:
        """Stop a tenancy WHEREVER it runs: probe the listings' rosters for
        the one carrying `run_id` running — tenancies live only at their
        anchor, so at most one listing answers — and tell that host to stop
        it (cancellation awaited host-side). A run nobody carries answers
        stopped: False; already dead is the goal state, not an error."""
        for name in sorted(self.listings):
            listing = self.listings[name]
            try:
                tenants = listing.host.status().get("tenants", {})
            except Exception:
                continue
            if tenants.get(run_id, {}).get("status") == "running":
                return await listing.host.stop(run_id)
        return {"stopped": False, "run_id": run_id, "state": "unlisted"}

    async def reroute(self, run_id: str, avoiding: str = "",
                      park: bool = False) -> dict:
        """RESTART-IS-REDIAL: move a delivered workload by replaying the
        desk's own archived delivery — place the archived demand rows again
        with `avoiding` off the table, stop the old tenancy at whichever
        listing's roster carries it, and deliver the archived frame to the
        new placement. Priced by resume-equivalence: the store is the run,
        so the move costs at most one uncommitted update, and nothing is
        copied because there is nothing to copy.

        PLACE-FIRST: a healthy run is refused a move rather than stopped
        with nowhere to go. `park` — decommission's mode, when the host is
        dying regardless — inverts that: the tenancy is stopped anyway and
        journaled PARKED with the boot instructions, the run waiting whole
        in the store for metal a human must add; resubmitting it through
        its campaign revives it. A delivery from before the archive carried
        rows and frame cannot be replayed — refused (or parked) with the
        same cure named."""
        archived = self.placements().get(run_id, {})
        rows, frame = archived.get("demands"), archived.get("frame")
        if rows is None or frame is None:
            if not park:
                return {"rerouted": False, "run_id": run_id,
                        "error": f"run {run_id} has no archived delivery at "
                                 f"this desk (placed before the archive, or "
                                 f"a pure client) — resubmit it through its "
                                 f"campaign instead"}
            stopped = await self.stop_anchored(run_id)
            self.store.append_fleet_event({
                "event": "parked", "t": time.time(), "run_id": run_id,
                "reason": "no archived delivery to replay — resubmit "
                          "through its campaign"})
            return {"rerouted": False, "parked": True, "run_id": run_id,
                    "stopped": stopped}
        demands = demands_from(rows)
        avoid = frozenset({avoiding}) if avoiding else frozenset()
        placement, boot = await self.place_listings(demands, avoid)
        if boot and not park:
            return {"rerouted": False, "run_id": run_id, "boot": boot,
                    "error": "nowhere to go: nothing else serves these "
                             "units and no registered metal can hold them "
                             "— the run keeps running where it is"}
        stopped = await self.stop_anchored(run_id)
        if boot:
            self.store.append_fleet_event({
                "event": "parked", "t": time.time(), "run_id": run_id,
                "boot": boot, "avoiding": avoiding})
            return {"rerouted": False, "parked": True, "run_id": run_id,
                    "boot": boot, "stopped": stopped}
        reply = await self.deliver(demands, placement, rows, frame)
        if not reply.get("accepted"):
            self.store.append_fleet_event({
                "event": "parked", "t": time.time(), "run_id": run_id,
                "reason": f"redelivery refused: {reply.get('error')}"})
            return {"rerouted": False, "parked": True, "run_id": run_id,
                    **reply}
        return {"rerouted": True, **reply}

    async def decommission(self, name: str, force: bool = False,
                           reroute: bool = False) -> dict:
        """CARVE'S INVERSE, client-asked: tear the host down at its metal
        (decarve — the engine shut down, the fraction back to residual) and
        delist it, one verb. The refusal is the point: a host that running
        work lives on OR routes through is NAMED rather than yanked, and
        `force` says you mean it. With `reroute` the running work is MOVED
        first: each dependent replayed onto a fresh placement with this
        listing off the table (restart-is-redial), and one nothing else
        covers is stopped and journaled PARKED — the host is coming down
        either way, and a parked run waits whole in the store for metal a
        human adds. A hand-listed host (no metal on its listing) only
        delists — its metal was never the desk's to touch; a silent metal
        delists too, reap's reasoning on demand: the fraction freed itself
        when the container died. The freed metal is reallocated by nothing
        more than existing rules — residual grew, so the next placement's
        carve may land there."""
        listing = self.listings.get(name)
        if listing is None:
            raise FleetError(f"host {name!r} is not listed with this desk")
        holding = self.dependents(name)
        moved: dict[str, dict] = {}
        if holding and reroute:
            for rid in holding:
                moved[rid] = await self.reroute(rid, avoiding=name, park=True)
            holding = self.dependents(name)   # what a replay could not clear
        if holding and not force:
            return {"decommissioned": False, "host": name,
                    "running": holding, "rerouted": moved,
                    "error": f"host {name!r} carries or serves running work "
                             f"({', '.join(holding)}) — decommission with "
                             f"force to tear it down anyway"}
        decarved = False
        if listing.metal and listing.metal in self.metal_remotes:
            try:
                reply = await self.metal_remotes[listing.metal].decarve(name)
                decarved = bool(reply.get("decarved"))
            except Exception:
                pass
        self.delist(name, reason="decommissioned")
        return {"decommissioned": True, "host": name, "decarved": decarved,
                "running": holding, "rerouted": moved}

    async def reap(self, probes: int = 3, wait: float = 0.0) -> dict:
        """Every listing probed, the silent ones retried, the still-silent
        ones REAPED — the desk's answer to a host that died without saying
        delist (a crashed process can never announce its crash; someone else
        must ask and hear nothing).

        The retries ARE the restart attempt: on a lazy venue the knock itself
        boots a stopped-but-still-deployed container, so a probe that fails,
        waits, and probes again gives the reboot its window — `recovered` is
        that verdict. A listing silent through every retry is concluded:
        DECARVED at its metal when the metal still answers (a living
        container frees the fraction back to residual; a dead one already
        did, physically), then DELISTED with the reason journaled, so a
        rebuilt desk agrees the host is gone. Verdicts per listing:
        alive | recovered | reaped."""
        verdicts: dict[str, str] = {}
        for name in sorted(self.listings):
            listing = self.listings[name]
            if listing.alive():
                verdicts[name] = "alive"
                continue
            recovered = False
            for _ in range(probes):
                if wait:
                    await asyncio.sleep(wait)
                if listing.alive():
                    recovered = True
                    break
            if recovered:
                verdicts[name] = "recovered"
                continue
            if listing.metal and listing.metal in self.metal_remotes:
                try:
                    await self.metal_remotes[listing.metal].decarve(name)
                except Exception:
                    pass            # the metal is as dead as the host: the
                                    # fraction freed itself when the container did
            self.delist(name, reason="reaped")
            verdicts[name] = "reaped"
        return verdicts

    def status(self) -> dict:
        """The desk's inventory, no wire calls: what is listed and what it
        wears, and the registered metal. Occupancy and residual are asked per
        placement, never cached here."""
        return {"listings": {name: {
            "address": listing.address, "solo": listing.solo,
            "regimes": [r.name for r in listing.regimes],
            "partition": listing.partition, "metal": listing.metal}
            for name, listing in sorted(self.listings.items())},
            "metal": {name: {"gpu": m.gpu, "devices": m.devices,
                             "vram_gb": m.vram_gb,
                             "plane": name in self.metal_remotes}
                      for name, m in sorted(self.metal.items())}}

    # ---- the Transport surface (HostService's contract, fleet-addressed) ----

    async def serve(self, verb: str, payload: dict) -> dict:
        if verb == "submit":
            return await self.submit(payload["demands"], payload["frame"])
        if verb == "place":
            return await self.place(demands_from(payload["demands"]))
        if verb == "list":
            # the phone-home half of the deploy contract: when the desk is its
            # own container, the deploy that booted a host reaches list_host
            # over the wire — same rows the journal speaks
            self.list_host(
                payload["host"],
                tuple(Regime(r["name"], r["capability"], r["base"], r["shape"])
                      for r in payload["regimes"]),
                payload["address"], solo=bool(payload.get("solo", False)),
                partition=payload.get("partition"),
                metal=payload.get("metal", ""))
            return {"listed": payload["host"]}
        if verb == "delist":
            self.delist(payload["host"], reason=payload.get("reason", ""))
            return {"delisted": payload["host"]}
        if verb == "decommission":
            return await self.decommission(
                payload["host"], force=bool(payload.get("force", False)),
                reroute=bool(payload.get("reroute", False)))
        if verb == "reroute":
            return await self.reroute(
                payload["run_id"], avoiding=payload.get("avoiding", ""),
                park=bool(payload.get("park", False)))
        if verb == "metal":
            # the metal container phones home its OWN existence, address
            # included — after this the desk can deduce and command against it
            self.register_metal(
                Metal(name=payload["name"], gpu=payload.get("gpu", "L4"),
                      devices=int(payload.get("devices", 1)),
                      vram_gb=float(payload.get("vram_gb", 24.0))),
                address=payload.get("address"))
            return {"registered": payload["name"]}
        if verb == "reap":
            return await self.reap(probes=int(payload.get("probes", 3)),
                                   wait=float(payload.get("wait", 0.0)))
        raise ValueError(f"unknown fleet verb {verb!r}")

    def answer(self, verb: str, payload: dict) -> dict:
        if verb == "status":
            return self.status()
        if verb == "liveness":
            return self.liveness()
        if verb == "placements":
            return {"placements": self.placements()}
        raise ValueError(f"unknown admission-free fleet verb {verb!r}")

    def liveness(self) -> dict:
        """Every listing PROBED, now: {host: answered}. The desk is the one
        place that can ask a container instead of presuming from a journal,
        and an observer given a desk shows probes where it has them."""
        return {name: listing.alive()
                for name, listing in sorted(self.listings.items())}


def covers(regimes: Sequence[Regime], demand: Demand) -> bool:
    """THE join rule, the only copy: coverage is capability equality."""
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
        host=connect(event["address"]),
        partition=event.get("partition"), metal=event.get("metal", ""))


# ---------------------------------------------------------------------------
# the metal plane: the container that owns a device, answering the desk
# ---------------------------------------------------------------------------

class MetalService:
    """The metal-side end of the standing carve: ONE registered Metal's
    devices, the factories that realize partitions on them, and the BOOKING
    rule that makes a desk-issued carve safe.

    The desk DEDUCES, the metal ENFORCES: `residual` is the deduction feed
    (per-device free = 1 - built - booked, this container's own books), and
    `carve` is the command — it books its fraction SYNCHRONOUSLY, before the
    build's first await, so two carves in one breath see each other and the
    loser refuses instead of double-booking the window where metal is
    promised but not yet built. A failed build releases its booking; nothing
    half-born is ever routed.

    Hosts born here — and hosts the venue built at boot and handed in via
    `adopt_born` — live in ONE table, so the residual is honest about both;
    the venue routes host frames through `service_for(address)`. The metal
    writes NOTHING to the fleet journal (the desk is that journal's one
    writer — a carve reply carries exactly what the desk journals and
    lists); host-plane events (host-up, attach, traffic) journal exactly as
    always, because they are each host's own.
    """

    def __init__(self, metal: Metal, *, store: Store,
                 engine_factory: Callable[[Regime, Partition], Engine],
                 learner_factory: Callable[[Regime, Partition], Learner],
                 address_of: Callable[[str], str],
                 schema_for: Callable[[str], SiteSchema] | None = None,
                 dial: Callable[[str], Transport] | None = None,
                 release: Callable[[Host], None] | None = None) -> None:
        self.metal = metal
        self.store = store
        self.engine_factory = engine_factory
        self.learner_factory = learner_factory
        # address formats are venue (I5): the venue mints a born host's
        # address, and the venue unmakes what the factories made (`release`,
        # decarve's teardown — an engine shutdown on real metal, nothing on
        # fakes). schema_for/dial are the adoption birth facts every host
        # born here is handed, same as a hand-built one.
        self.address_of = address_of
        self.schema_for = schema_for
        self.dial = dial
        self.release = release
        self.hosts: dict[str, Host] = {}
        self.addresses: dict[str, str] = {}         # host name -> address
        self.services: dict[str, HostService] = {}  # address -> service
        self.pending: list[tuple[str, tuple[int, ...], float]] = []
        self.carves = 0

    # ---- the books ----------------------------------------------------------

    def adopt_born(self, host: Host, address: str) -> None:
        """A host the venue built at boot enters this metal's books: counted
        into residual and routed at its address exactly like a carve's child,
        so hand-built standing hosts and desk-carved ones share one table and
        one truth. Refuses a host born without a partition — a host that owns
        no stated fraction cannot be accounted, and unaccounted metal is the
        double-book this class exists to kill."""
        if host.partition is None:
            raise FleetError(
                f"host {host.name!r} has no partition: a metal's books count "
                f"fractions, so every host on them must own one")
        if host.name in self.hosts:
            raise FleetError(f"host {host.name!r} is already on this metal")
        self.hosts[host.name] = host
        self.addresses[host.name] = address
        self.services[address] = HostService(host)

    def residual(self) -> list[float]:
        """Free memory per device, counting BUILT partitions and PENDING
        bookings — capacity nobody owns and nobody has been promised. The
        number the desk reads to deduce, and the number choose_devices
        refuses over."""
        free = [1.0] * self.metal.devices
        for host in self.hosts.values():
            for device in host.partition.devices:
                free[device] -= host.partition.memory
        for _, devices, memory in self.pending:
            for device in devices:
                free[device] -= memory
        return free

    def choose_devices(self, count: int, memory: float) -> tuple[int, ...] | None:
        """First-fit against this metal's own books: `count` devices each
        with `memory` free — plan_carve's rule, where the truth lives."""
        free = self.residual()
        chosen = [i for i, f in enumerate(free) if f >= memory - 1e-9][:count]
        return tuple(chosen) if len(chosen) == count else None

    def carve_name(self, devices: tuple[int, ...],
                   regimes: tuple[Regime, ...]) -> str:
        """Carve names, on this metal's own ordinal — minted on
        the loop, BEFORE any build thread runs, so concurrent carves never
        race the counter."""
        self.carves += 1
        joined = "-".join(str(d) for d in devices)
        worn = "+".join(regime.name for regime in regimes)
        return f"{self.metal.name}:{joined}.{worn}.c{self.carves}"

    # ---- the two commands ---------------------------------------------------

    async def carve(self, request: Mapping) -> dict:
        """The desk's command executed: decode the regimes, choose and BOOK
        the devices synchronously, then build (factories may block for
        minutes — an engine boot — so they run in a worker thread), attest
        (the Host constructor's own job), route, and reply with the listing
        facts. The booking is released on every exit: on success the born
        partition has replaced it on the books in the same tick; on failure
        the fraction is free again and the refusal says what the residual is
        NOW, so the desk's next deduction is current."""
        regimes = tuple(
            Regime(r["name"], r["capability"], r["base"],
                   int(r.get("shape", 1)))
            for r in request["regimes"])
        memory = float(request["memory"])
        count = max(regime.shape for regime in regimes)
        devices = self.choose_devices(count, memory)
        if devices is None:
            return {"carved": False, "residual": self.residual(),
                    "error": f"metal {self.metal.name!r} cannot hold "
                             f"{count} device(s) at {memory:g}: residual is "
                             f"{self.residual()}"}
        name = self.carve_name(devices, regimes)
        booking = (name, devices, memory)
        self.pending.append(booking)
        try:
            host = await asyncio.to_thread(self.build, name, regimes,
                                           devices, memory)
            # booked -> built in one tick: no await between the thread's
            # return and these lines, so residual never blinks
            self.hosts[name] = host
            address = self.address_of(name)
            self.addresses[name] = address
            self.services[address] = HostService(host)
        except Exception as failure:
            return {"carved": False, "residual": self.residual(),
                    "error": f"the build failed and the booking is released: "
                             f"{failure}"}
        finally:
            self.pending.remove(booking)
        return {"carved": True, "host": name, "address": address,
                "solo": host.solo,
                "partition": host.partition.row(),
                "regimes": [{"name": r.name, "capability": r.capability,
                             "base": r.base, "shape": r.shape}
                            for r in regimes]}

    def build(self, name: str, regimes: tuple[Regime, ...],
              devices: tuple[int, ...], memory: float) -> Host:
        """(worker thread) The factories realize the partition and the Host
        constructor attests the result against
        books this container owns."""
        partition = Partition(self.metal.name, devices, memory, self.metal.gpu)
        engines: list[Engine] = []
        carved_learner: Learner | None = None
        for regime in regimes:
            if regime.capability == "inference":
                engines.append(self.engine_factory(regime, partition))
            else:
                carved_learner = self.learner_factory(regime, partition)
        return Host(name, engines=tuple(engines), learner=carved_learner,
                    store=self.store, partition=partition, regimes=regimes,
                    schema_for=self.schema_for, dial=self.dial)

    def decarve(self, name: str) -> dict:
        """The inverse, for the reaper and the deliberate retirement: the
        venue's `release` unmakes what its factories made, the host leaves
        the books, and its fraction is residual again. The desk journals the
        departure (its delist) — this side only frees."""
        host = self.hosts.pop(name, None)
        if host is None:
            return {"decarved": False,
                    "error": f"no host {name!r} on metal {self.metal.name!r}"}
        address = self.addresses.pop(name)
        self.services.pop(address, None)
        if self.release is not None:
            self.release(host)
        return {"decarved": True, "host": name,
                "partition": host.partition.row()}

    # ---- routing and the Transport surface ----------------------------------

    def service_for(self, address: str) -> HostService:
        """The venue's router: host frames arrive addressed, and an address
        nothing answers (decarved, or never carved) refuses by name instead
        of KeyError-ing inside a frame handler."""
        service = self.services.get(address)
        if service is None:
            raise FleetError(
                f"no host answers at {address!r} on metal "
                f"{self.metal.name!r} (decarved, or never carved); serving "
                f"{sorted(self.services)}")
        return service

    def describe(self) -> dict:
        """The registration row plus the books — what a phone-home ships and
        what an observer renders."""
        return {"name": self.metal.name, "gpu": self.metal.gpu,
                "devices": self.metal.devices, "vram_gb": self.metal.vram_gb,
                "residual": self.residual(),
                "hosts": {name: {"address": self.addresses[name],
                                 "partition": host.partition.row()}
                          for name, host in sorted(self.hosts.items())}}

    async def serve(self, verb: str, payload: dict) -> dict:
        if verb == "carve":
            return await self.carve(payload)
        if verb == "decarve":
            return self.decarve(payload["host"])
        raise ValueError(f"unknown metal verb {verb!r}")

    def answer(self, verb: str, payload: dict) -> dict:
        if verb == "residual":
            return {"residual": self.residual()}
        if verb == "describe":
            return self.describe()
        raise ValueError(f"unknown admission-free metal verb {verb!r}")
