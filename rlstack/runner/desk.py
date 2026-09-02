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
acquiring stays a human's. One HostSpec is ONE placement unit: a single
member is a dedicated host, several members alternate on one multi-regime
host (I5: where a member lands is a hint, never semantics).

THE DESK SPEAKS GB (ADR 0001). A demand's `vram_gb` is total across its
shards; the desk deduces per-device GB against each metal's residual, also
in GB; and the metal converts to its partition's fraction exactly once, at
build, against the card it MEASURED — `fraction_for_gb`, the one crossing.

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
from rlstack.runner.residents import (
    GRACE_S, SIGNAL_GRACE_S, Builds, Resident, ResidentBirth, Teardown,
)
from rlstack.runner.remote import (
    RemoteLearner, RemotePool,
    HostService, LocalTransport, RemoteHost, RemotePool, Transport,
)


class DeskError(RuntimeError):
    """A placement the fleet may not decide alone (acquire is a human's) or
    a plan it cannot execute."""


@dataclass(frozen=True)
class Metal:
    """Owned metal: one registered card set the fleet may carve. Registering
    Metal IS the acquire rung executed — the rung that costs money, so the one
    a human does. `gpu` is the KIND (what was bought) and `vram_gb` one
    device's VRAM in GB (what a model is measured against — L4=24, A100=40 or
    80, H100=80); a carve stamps the kind onto the Partition it births. On a
    real venue both are MEASURED off the device by `MetalService.measure`,
    never typed (ADR 0001, Q6): a declared card is wrong confidently, a
    measured one is a fact. A plain record, so tests construct it directly."""

    name: str
    gpu: str = "L4"
    devices: int = 1
    vram_gb: float = 24.0


def fraction_for_gb(gb: float, metal: Metal) -> float:
    """THE ONE CROSSING between the unit a human sizes models in (GB of VRAM,
    what the spec and the desk speak) and the unit a partition owns (a
    fraction of ONE device, what both substrates take): `gb` per device
    becomes gb / metal.vram_gb on the metal it will live on. Called once, at
    `MetalService.build`, against the card that metal measured — a GB figure
    is never stored as a fraction anywhere else. More GB than one device
    holds is not a smaller fraction, it is bigger metal — the acquire rung —
    so it raises by name instead of clamping."""
    fraction = gb / metal.vram_gb
    if fraction > 1.0 + 1e-9:
        raise DeskError(
            f"{gb:g} GB is more than one {metal.gpu} device holds "
            f"({metal.vram_gb:g} GB on {metal.name!r}) — VRAM per shard past "
            f"one device is the acquire rung (bigger metal), not a fraction")
    return fraction


def gb_of(partition: Partition, metal: Metal) -> float:
    """The GB a built partition owns on each of its devices: its fraction of
    THIS card. fraction_for_gb's read-back, used only by the metal's own
    books so `residual` can answer in GB — the unit the desk deduces in."""
    return partition.memory * metal.vram_gb


@dataclass(frozen=True)
class Demand:
    """One member's capability demand, read off the spec: WHAT is needed
    (capability, base, shard shape), never WHERE. `vram_gb` is the carve
    size — TOTAL across the demand's shards, so the per-device need is
    vram_gb / shape; None means a whole device per shard, resolved at the
    metal that knows its card (Q10). It sizes a new partition at rung two and
    is ignored on a join. `group` is the HostSpec the member came from: one
    HostSpec is one placement unit, so members sharing a group alternate on
    one host. `pool` None is the learner member."""

    pool: str | None
    capability: str                 # "inference" | "training"
    base: str
    shape: int
    vram_gb: float | None
    group: int
    # the ANCHOR is where a delivered frame lands — the one demand whose host
    # receives the workload. The desk never knows WHY (for an experiment it is
    # the learner, because the learner is never remote — but that rule lives
    # with whoever built the demands, not here).
    anchor: bool = False

    def per_device_gb(self) -> float | None:
        """The memory this demand needs on EACH device it spans: the total
        divided across its shards. None stays None — a whole device."""
        return None if self.vram_gb is None else self.vram_gb / self.shape


def demand_rows(demands: Sequence[Demand]) -> list[dict]:
    """Demands as wire rows — the desk's whole input vocabulary."""
    return [{"pool": d.pool, "capability": d.capability, "base": d.base,
             "shape": d.shape, "vram_gb": d.vram_gb, "group": d.group,
             "anchor": d.anchor} for d in demands]


def demands_from(rows: Sequence[Mapping]) -> tuple[Demand, ...]:
    """Wire rows back as Demands — demand_rows' typed inverse."""
    return tuple(Demand(
        pool=row["pool"], capability=row["capability"], base=row["base"],
        shape=int(row["shape"]),
        vram_gb=None if row.get("vram_gb") is None else float(row["vram_gb"]),
        group=int(row["group"]),
        anchor=bool(row.get("anchor", False))) for row in rows)


def placement_units(demands: Sequence[Demand]) -> tuple[tuple[Demand, ...], ...]:
    """ONE HOSTSPEC IS ONE HOST, so its members place as ONE unit: a single
    member is a dedicated host, several alternate on one partition (one host
    wearing masks, never two hosts coordinating). Separate HostSpecs place
    one by one onto per-capability hosts (ADR 0001)."""
    units: list[tuple[Demand, ...]] = []
    seen: set[int] = set()
    for demand in demands:
        if demand.group in seen:
            continue
        seen.add(demand.group)
        units.append(tuple(d for d in demands if d.group == demand.group))
    return tuple(units)


def unit_gb(unit: Sequence[Demand], metal: Metal) -> float:
    """The per-device GB a placement unit's partition must hold on `metal`:
    the LARGEST member's per-device need, because the unit's members
    alternate — one resident live at a time, each sized for itself — and a
    whole-device member (None) is the whole of THIS metal's card. Sized per
    metal because "a whole device" is a different number on every kind."""
    return max(metal.vram_gb if d.per_device_gb() is None else d.per_device_gb()
               for d in unit)


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
    matches on (regimes, solo), the ADDRESS other hosts reach its pools at,
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
                 host_for: Callable[[str], "RemoteHost"],
                 metal_for: Callable[[str], "RemoteMetal"] | None = None,
                 ) -> None:
        self.store = store
        # address -> RemoteHost: how the desk reaches a listed host
        self.host_for = host_for
        # address -> RemoteMetal: how it reaches the METAL PLANE — the verbs
        # that create and free hosts (carve/decarve) and the residual the
        # desk deduces from. Venue like `host_for`; a desk born without it
        # still places over listings and answers misses with boot
        # instructions, it just cannot command a carve.
        self.metal_for = metal_for
        self.metal: dict[str, Metal] = {}
        self.metal_remotes: dict[str, "RemoteMetal"] = {}
        # name -> the metal's build recipe row (Builds.row()), as it declared
        # it: descriptive inventory, journaled so the record of HOW a host was
        # built survives the desk (ADR 0002, Q4a). The desk never builds.
        self.metal_builds: dict[str, dict] = {}
        self.listings: dict[str, Listing] = {}

    @classmethod
    def from_journal(cls, store: Store,
                     host_for: Callable[[str], "RemoteHost"],
                     metal_for: Callable[[str], "RemoteMetal"] | None = None,
                     ) -> "Desk":
        """The desk, rebuilt from its own record: every `list` event resolves
        its address again, and so does every addressed `metal` event. Kill -9 the
        desk and nothing was lost but a process — the same recovery shape as
        attach, on the fleet plane."""
        desk = cls(store, host_for, metal_for)
        for event in store.read_fleet_log():
            if event.get("event") == "list":
                desk.listings[event["host"]] = _listing_from(event, host_for)
            elif event.get("event") == "delist":
                desk.listings.pop(event["host"], None)
            elif event.get("event") == "metal":
                desk.metal[event["name"]] = Metal(
                    name=event["name"], gpu=event.get("gpu", "L4"),
                    devices=int(event.get("devices", 1)),
                    vram_gb=float(event.get("vram_gb", 24.0)))
                address = event.get("address")
                if address and metal_for is not None:
                    desk.metal_remotes[event["name"]] = metal_for(address)
                if event.get("builds"):
                    desk.metal_builds[event["name"]] = dict(event["builds"])
        return desk

    def register_metal(self, metal: Metal, address: str | None = None,
                       builds: Mapping | None = None) -> None:
        """The acquire rung, recorded at the desk: what the fleet OWNS and may
        carve against. Journaled like a listing, so a rebuilt desk knows its
        inventory too. `address` is where that metal's own container answers
        the metal plane (carve/decarve/residual) — with it and a resolver the
        desk can command the standing carve; without it the row is inventory
        only and misses still answer with boot instructions. `builds` is the
        metal's recipe row, kept and journaled as description."""
        if metal.name in self.metal:
            raise DeskError(f"metal {metal.name!r} is already registered")
        self.metal[metal.name] = metal
        if address and self.metal_for is not None:
            self.metal_remotes[metal.name] = self.metal_for(address)
        if builds:
            self.metal_builds[metal.name] = dict(builds)
        self.store.append_fleet_event({
            "event": "metal", "t": time.time(), "name": metal.name,
            "gpu": metal.gpu, "devices": metal.devices,
            "vram_gb": metal.vram_gb, "address": address,
            "builds": dict(builds) if builds else None})

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
            raise DeskError(
                f"host {name!r} is already listed with this desk; a listing "
                f"is never replaced — delist first if the container is gone")
        self.listings[name] = Listing(name=name, regimes=tuple(regimes),
                                      address=address, solo=solo,
                                      host=self.host_for(address),
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
            raise DeskError(f"host {name!r} is not listed with this desk")
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
                    "devices": max(d.shape for d in unit),
                    # the largest member's per-device need, in GB — None is
                    # a whole device of whatever card answers the boot
                    "vram_gb": max((d.per_device_gb() for d in unit
                                    if d.per_device_gb() is not None),
                                   default=None)})
                continue
            for demand in unit:
                placement[demand.pool] = listing
        return placement, boot

    async def provision_unit(self, unit: tuple[Demand, ...]) -> Listing | None:
        """The standing CARVE, desk-issued: nothing listed serves this unit,
        so the desk asks each registered metal whether it can hold it (the
        residual, in GB per device — the DEDUCTION) and COMMANDS the first
        that can (carve). The metal ENFORCES: it books the GB synchronously at
        its own door and converts to its partition's fraction at build, so a
        deduction gone stale between the ask and the command costs a
        refusal, never a double-book — and a refusal or a silent metal falls
        through to the next, then to the boot instructions. The carve request
        carries this desk's `builds` row for the metal (Q5c: the desk's
        recipe is canon and rides every carve). What comes back is journaled
        and listed HERE: the desk stays the fleet journal's one writer, which
        is exactly why the metal writes nothing."""
        if not self.metal_remotes:
            return None
        need_devices = max(demand.shape for demand in unit)
        for metal_name in sorted(self.metal_remotes):
            remote = self.metal_remotes[metal_name]
            need_gb = unit_gb(unit, self.metal[metal_name])
            request = self.carve_request(unit, metal_name, need_gb)
            try:
                free = remote.residual()
            except Exception:
                continue                # a silent metal is the reaper's, not ours
            if sum(1 for f in free if f >= need_gb - 1e-9) < need_devices:
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

    def carve_request(self, unit: tuple[Demand, ...], metal_name: str,
                      need_gb: float) -> dict:
        """The carve command as the metal verb speaks it: the regimes the
        host will wear, the per-device GB its partition must hold, and the
        recipe it builds from — this desk's journaled `builds` row for that
        metal, the canon a reborn container is rebuilt from (Q5c). A metal
        registered without a recipe is carved from its own."""
        return {
            "regimes": [{"name": regime_of(d).name,
                         "capability": d.capability,
                         "base": d.base, "shape": d.shape} for d in unit],
            "base": unit[0].base,
            "vram_gb": need_gb,
            "builds": self.metal_builds.get(metal_name),
        }

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
        """A MOVE IS A RESTART: move a delivered workload by replaying the
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
        (decarve — the engine shut down, its GB back to residual) and
        delist it, one verb. The refusal is the point: a host that running
        work lives on OR routes through is NAMED rather than yanked, and
        `force` says you mean it. With `reroute` the running work is MOVED
        first: each dependent replayed onto a fresh placement with this
        listing off the table (a move is a restart), and one nothing else
        covers is stopped and journaled PARKED — the host is coming down
        either way, and a parked run waits whole in the store for metal a
        human adds. A hand-listed host (no metal on its listing) only
        delists — its metal was never the desk's to touch; a silent metal
        delists too, reap's reasoning on demand: the memory freed itself
        when the container died. The freed metal is reallocated by nothing
        more than existing rules — residual grew, so the next placement's
        carve may land there."""
        listing = self.listings.get(name)
        if listing is None:
            raise DeskError(f"host {name!r} is not listed with this desk")
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
        container frees its GB back to residual; a dead one already
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
                                    # memory freed itself when the container did
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
                             "plane": name in self.metal_remotes,
                             "builds": self.metal_builds.get(name)}
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
                address=payload.get("address"),
                builds=payload.get("builds"))
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
                  host_for: Callable[[str], "RemoteHost"]) -> Listing:
    """A journal `list` event back as a Listing — from_journal's one row."""
    return Listing(
        name=event["host"],
        regimes=tuple(Regime(r["name"], r["capability"], r["base"], r["shape"])
                      for r in event["regimes"]),
        address=event["address"], solo=bool(event.get("solo", False)),
        host=host_for(event["address"]),
        partition=event.get("partition"), metal=event.get("metal", ""))


# ---------------------------------------------------------------------------
# the metal plane: the container that owns a device, answering the desk
# ---------------------------------------------------------------------------

class MetalService:
    """The metal-side end of the standing carve: ONE registered Metal's
    devices, the RECIPE that realizes partitions on them, and the BOOKING
    rule that makes a desk-issued carve safe.

    The desk DEDUCES, the metal ENFORCES: `residual` is the deduction feed
    (per-device free GB = the card - built - booked, this container's own
    books), and `carve` is the command — it books its GB SYNCHRONOUSLY,
    before the build's first await, so two carves in one breath see each
    other and the loser refuses instead of double-booking the window where
    metal is promised but not yet built. A failed build releases its
    booking; nothing half-born is ever routed. THE BOOKS ARE IN GB and the
    partition is a fraction: `build` converts once, with `fraction_for_gb`,
    against the card this metal measured (ADR 0001).

    EVERY RESIDENT IS A PROCESS (ADR 0002): a carve spawns one child per
    regime — pinned to the partition's devices, capped at its fraction, built
    by rlstack's own universal builders from this metal's `builds` recipe —
    and the Host it lists holds proxies over each child's door. `spawn` is
    how a birth becomes a resident: `Resident.spawn` (a child) in every
    venue, `Resident.in_process` where a test needs a fake it can hold open
    or break. A resident that exits unbidden is a host that died: the
    watcher tells this service, which decarves the host and frees its
    booking, and the desk's next probe reaps the listing (Q7).

    Hosts born here — and hosts the venue built at boot and handed in via
    `adopt_born` — live in ONE table, so the residual is honest about both;
    the venue routes host frames through `service_for(address)`. The metal
    writes NOTHING to the fleet journal (the desk is that journal's one
    writer — a carve reply carries exactly what the desk journals and
    lists); host-plane events (host-up, attach, traffic) journal exactly as
    always, because they are each host's own.
    """

    def __init__(self, metal: Metal, *, store: Store, builds: Builds,
                 address_of: Callable[[str], str],
                 schema_for: Callable[[str], SiteSchema] | None = None,
                 transport_for: Callable[[str], Transport] | None = None,
                 spawn: Callable[[ResidentBirth], Resident] = Resident.spawn
                 ) -> None:
        self.metal = metal
        self.store = store
        # the recipe: what a venue declares and everything a partition cannot
        # tell you; re-declared at every bring-up from the deploy's constants,
        # so a restarted container carves the same residents unattended
        self.builds = builds
        # address formats are venue (I5): the venue mints a born host's
        # address. schema_for/transport_for are the adoption birth facts every
        # host born here is handed, same as a hand-built one.
        self.address_of = address_of
        self.schema_for = schema_for
        self.transport_for = transport_for
        self.spawn = spawn
        self.hosts: dict[str, Host] = {}
        self.addresses: dict[str, str] = {}         # host name -> address
        self.services: dict[str, HostService] = {}  # address -> service
        self.pending: list[tuple[str, tuple[int, ...], float]] = []
        self.carves = 0
        self.deaths: list[str] = []                 # hosts decarved by a resident's exit

    # ---- the books ----------------------------------------------------------

    def adopt_born(self, host: Host, address: str) -> None:
        """A host the venue built at boot enters this metal's books: counted
        into residual and routed at its address exactly like a carve's child,
        so hand-built standing hosts and desk-carved ones share one table and
        one truth. Refuses a host born without a partition — a host that owns
        no stated share cannot be accounted, and unaccounted metal is the
        double-book this class exists to kill."""
        if host.partition is None:
            raise DeskError(
                f"host {host.name!r} has no partition: a metal's books count "
                f"shares, so every host on them must own one")
        if host.name in self.hosts:
            raise DeskError(f"host {host.name!r} is already on this metal")
        self.hosts[host.name] = host
        self.addresses[host.name] = address
        self.services[address] = HostService(host)

    def residual(self) -> list[float]:
        """Free GB per device, counting BUILT partitions (their fraction of
        this card, read back) and PENDING bookings — capacity nobody owns and
        nobody has been promised. The number the desk reads to deduce, and
        the number choose_devices refuses over."""
        free = [self.metal.vram_gb] * self.metal.devices
        for host in self.hosts.values():
            for device in host.partition.devices:
                free[device] -= gb_of(host.partition, self.metal)
        for _, devices, gb in self.pending:
            for device in devices:
                free[device] -= gb
        return free

    def choose_devices(self, count: int, gb: float) -> tuple[int, ...] | None:
        """First-fit against this metal's own books: `count` devices each
        with `gb` free — plan_carve's rule, where the truth lives."""
        free = self.residual()
        chosen = [i for i, f in enumerate(free) if f >= gb - 1e-9][:count]
        return tuple(chosen) if len(chosen) == count else None

    def per_device_gb(self, request: Mapping) -> float:
        """The GB a carve request wants on each device: its `vram_gb`, or —
        None, a whole device — the whole of THIS card, resolved here because
        only the metal knows what a whole device is (Q10)."""
        wanted = request.get("vram_gb")
        return self.metal.vram_gb if wanted is None else float(wanted)

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
        the devices synchronously (in GB), then build (a resident's birth may
        block for minutes — an engine boot — so it runs in a worker thread),
        attest (the Host constructor's own job), route, and reply with the
        listing facts. The booking is released on every exit: on success the
        born partition has replaced it on the books in the same tick; on
        failure the GB is free again and the refusal says what the residual
        is NOW, so the desk's next deduction is current. A request carrying
        the desk's `builds` row is built from it (Q5c)."""
        regimes = tuple(
            Regime(r["name"], r["capability"], r["base"],
                   int(r.get("shape", 1)))
            for r in request["regimes"])
        gb = self.per_device_gb(request)
        count = max(regime.shape for regime in regimes)
        devices = self.choose_devices(count, gb)
        if devices is None:
            return {"carved": False, "residual": self.residual(),
                    "error": f"metal {self.metal.name!r} cannot hold "
                             f"{count} device(s) at {gb:g} GB: residual is "
                             f"{self.residual()} (one {self.metal.gpu} device "
                             f"holds {self.metal.vram_gb:g} GB)"}
        if request.get("builds"):
            self.adopt_recipe(Builds.from_row(request["builds"]))
        name = self.carve_name(devices, regimes)
        booking = (name, devices, gb)
        self.pending.append(booking)
        try:
            host = await asyncio.to_thread(self.build, name, regimes,
                                           devices, gb)
            # booked -> built in one tick: no await between the thread's
            # return and these lines, so residual never blinks
            self.hosts[name] = host
            address = self.address_of(name)
            self.addresses[name] = address
            self.services[address] = HostService(host)
            # the host is in the books: an exit from here on finds it (Q7)
            loop = asyncio.get_running_loop()
            for resident in host.residents:
                resident.watch(lambda r, host_name=name, loop=loop:
                               self.report_exit(loop, host_name, r))
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

    def adopt_recipe(self, builds: Builds) -> None:
        """The desk's recipe row is CANON (Q5c): a carve request that carries
        one replaces this metal's own, so its deploy constants are only its
        FIRST declaration and `describe()` reports what it last built from.
        A redeploy's new constants reach the desk through re-registration,
        which updates the row that rides the next carve."""
        self.builds = builds

    def build(self, name: str, regimes: tuple[Regime, ...],
              devices: tuple[int, ...], gb: float) -> Host:
        """(worker thread) THE crossing happens here, once: `gb` per device
        becomes this partition's fraction against the card this metal
        measured (fraction_for_gb — more than one device holds raises the
        acquire rung by name, never clamps). Then one resident per regime is
        born — pinned, capped, built from this metal's recipe — and the Host
        constructor attests the proxies against the regimes, exactly as it
        attested objects. A birth that fails ends the residents already born,
        so a half-carved host never holds metal."""
        partition = Partition(self.metal.name, devices,
                              fraction_for_gb(gb, self.metal), self.metal.gpu)
        residents: list[Resident] = []
        try:
            for regime in regimes:
                residents.append(self.spawn(ResidentBirth(
                    label=f"{name}:{regime.name}", partition=partition,
                    regime=regime, build=self.builds.for_regime(regime),
                    store=self.store.address())))
            engines = [RemotePool(r.transport, base=r.hello["base"],
                                  tp=int(r.hello["tp"]))
                       for r in residents if r.regime.capability == "inference"]
            learners = [RemoteLearner(r.transport, fsdp=int(r.hello["fsdp"]))
                        for r in residents if r.regime.capability == "training"]
            return Host(name, engines=tuple(engines),
                        learner=learners[0] if learners else None,
                        store=self.store, partition=partition, regimes=regimes,
                        schema_for=self.schema_for,
                        transport_for=self.transport_for, residents=residents)
        except BaseException:
            for resident in residents:
                resident.stop(grace_s=SIGNAL_GRACE_S,
                              signal_grace_s=SIGNAL_GRACE_S)
            raise

    def decarve(self, name: str) -> dict:
        """The inverse, for the reaper and the deliberate retirement: every
        resident is ended down the ladder, the host leaves the books, and its
        GB is residual again. The desk journals the departure (its delist) —
        this side only frees."""
        host = self.hosts.pop(name, None)
        if host is None:
            return {"decarved": False,
                    "error": f"no host {name!r} on metal {self.metal.name!r}"}
        address = self.addresses.pop(name)
        self.services.pop(address, None)
        teardowns = self.end_residents(host)
        return {"decarved": True, "host": name,
                "partition": host.partition.row(),
                "teardown": [t.line() for t in teardowns if not t.graceful]}

    def end_residents(self, host: Host) -> list[Teardown]:
        """Every resident of `host` down the ladder, each reported. A
        resident already gone costs nothing; one that ignores the stop frame
        is signalled, and one that survives the kill is named, not
        pretended away."""
        return [resident.stop(grace_s=GRACE_S, signal_grace_s=SIGNAL_GRACE_S)
                for resident in host.residents]

    def report_exit(self, loop: asyncio.AbstractEventLoop, host_name: str,
                    resident: Resident) -> None:
        """(watcher thread) Hand the exit to the loop that carved, where the
        books are mutated; if that loop is already closed — a test's
        asyncio.run ended, a container is on its way out — conclude it here,
        because a dead resident must not wait for a loop that will never
        turn."""
        try:
            loop.call_soon_threadsafe(self.resident_exited, host_name, resident)
        except RuntimeError:
            self.resident_exited(host_name, resident)

    def resident_exited(self, host_name: str, resident: Resident) -> None:
        """A resident left without being told to: its host is dead (I12 — a
        host is atomic, and a host missing a resident is not a host). The
        host is decarved — siblings ended, booking freed, address gone — so
        the desk's next probe finds nothing at the listing and reaps it.
        Nothing is restarted here: the tenant returns by resubmit and
        recarve, on the record (Q7)."""
        if host_name not in self.hosts:
            return                          # already decarved, by whoever came first
        print(f"[metal {self.metal.name}] resident {resident.label!r} exited "
              f"unbidden (pid {resident.pid()}): decarving host {host_name!r}",
              flush=True)
        self.deaths.append(host_name)
        self.decarve(host_name)

    def shutdown(self) -> list[Teardown]:
        """The container's way out: every host's residents down the ladder,
        in one bounded pass. What @modal.exit calls."""
        teardowns: list[Teardown] = []
        for name in sorted(self.hosts):
            teardowns.extend(self.end_residents(self.hosts[name]))
        return teardowns

    # ---- routing and the Transport surface ----------------------------------

    def service_for(self, address: str) -> HostService:
        """The venue's router: host frames arrive addressed, and an address
        nothing answers (decarved, or never carved) refuses by name instead
        of KeyError-ing inside a frame handler."""
        service = self.services.get(address)
        if service is None:
            raise DeskError(
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
                "builds": self.builds.row(),
                "hosts": {name: {"address": self.addresses[name],
                                 "partition": host.partition.row(),
                                 "residents": [r.row() for r in host.residents]}
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
