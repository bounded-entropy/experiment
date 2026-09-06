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

THE DESK SUPERVISES ITS METALS (ADR 0001, Q5). The one restart that is
automatic is a METAL's — a preempt takes every host on it — and the loop is
reap → knock → re-register → reroute: the reaper concludes the metal's
listings silent, KNOCKS its plane address (on a lazy venue the knock is the
boot; `boot_for` where it is not), the reborn container REGISTERS itself at
bring-up (a known name at the same address UPDATES the row and reaps its
corpses by probe), and every run whose placement was on the dead hosts is
STRANDED — journaled `parked`, the queue — and RETRIED with `reroute`: re-
placed onto whatever fits, the reborn metal included, and redelivered, which
is resume. Every registration event retries the whole queue. A host dying
alone stays a human's resubmit.

Truth stays in the store: every listing, metal registration, placement and
delisting is journaled, `from_journal` rebuilds the desk after a kill, and
the desk's memory is only the single writer's cache — "the fleet" names the
aggregate this journal records, and the desk is that journal's one writer.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from dataclasses import dataclass

from rlstack.data.stores.base import Store, run_done
from rlstack.policy.siteschema import SiteSchema
from rlstack.runner.host import LEARNER_ROUTE, Host, Partition, Regime
from rlstack.runner.residents import (
    GRACE_S, SIGNAL_GRACE_S, Builds, EngineBuild, Resident, ResidentBirth,
    Teardown,
)
from rlstack.runner.remote import (
    BUILD_DEADLINE_S, DESK_DEFAULT, RemoteLearner, RemotePool,
    HostService, LocalTransport, RemoteHost, RemotePool, Transport, Undeclared,
    Unreachable, check_epoch, serve_in_process, stop_serving_in_process,
    with_epoch, without_epoch,
)

# The desk's default idle limit, in seconds (ADR 0003): metal that nothing has
# run on for half an hour is RELEASED — the acquire rung inverted, automatic,
# because the metal is already owned. A venue overrides it per desk
# (`Desk(idle_s=)`) and a registration overrides it per metal.
IDLE_S = 90.0

# THE FLEET'S LEASE (ADR 0008, F1 and Q1). A metal and every host on it renew
# their lease by heartbeat every HEARTBEAT_S; a silence longer than LEASE_S is
# no longer belief, and the thing is delisted, unplaceable, and reaped. Long
# enough to survive a slow tick, short enough that a dead container is off the
# listings before the next campaign door submits. Both are the DESK's — a
# registration journals them, so the record says what the fleet was promising
# at the time — and a metal may declare a longer lease of its own.
LEASE_S = 60.0
HEARTBEAT_S = 20.0


class DeskError(RuntimeError):
    """A placement the fleet may not decide alone (acquire is a human's) or
    a plan it cannot execute."""


def mint_epoch() -> str:
    """A BOOT IDENTITY, minted once by a container at bring-up and dying with
    it (ADR 0008, Q2). A name is not an instance: `concept-a100` is a metal
    the fleet owns, and the epoch is which of its lives is answering. Random
    rather than a counter, because there is nowhere durable to count in — the
    only thing that must be true is that two lives never share one."""
    import uuid

    return uuid.uuid4().hex[:12]


@dataclass(frozen=True)
class Lease:
    """WHAT THE DESK IS STILL ENTITLED TO BELIEVE about one metal or one host
    (ADR 0008, F1).

    `epoch` is the instance the belief is about, `heard_t` the last time that
    instance said anything, and `lease_s` how long a silence is still belief.
    Past that the row is not deleted — it is DISBELIEVED: delisted from every
    listing placement reads, passed over by the carve rung, and reaped. The
    cure is one heartbeat.

    Frozen, because a lease is a reading rather than a state: renewing one is
    replacing it, which is what makes "the desk's memory is the single
    writer's cache" true of this table too."""

    epoch: str
    heard_t: float
    lease_s: float

    def live(self, now: float) -> bool:
        """Has this thing spoken within its lease?"""
        return now - self.heard_t <= self.lease_s

    def renewed(self, now: float) -> "Lease":
        """The same lease, heard just now."""
        return Lease(self.epoch, now, self.lease_s)

    def row(self) -> dict:
        """The lease as `status()` shows it — three fields, no derivation, so
        an operator reading the row can do the subtraction themselves."""
        return {"epoch": self.epoch, "heard_t": self.heard_t,
                "lease_s": self.lease_s}


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
    # a CHOICE the campaign layer makes — ADR 0006 Part A — but that rule
    # lives with whoever built the demands, not here).
    anchor: bool = False
    # the adapter types this demand's workload NAMES — the strings the
    # campaign layer read off the spec's bank (ADR 0007, Q4a). The desk
    # compares them against a metal recipe's `serves` and never interprets
    # them: it is the join rung's second half, not knowledge of a workload
    # (#69). Empty is "nothing named", which refuses nothing.
    adapter_types: tuple[str, ...] = ()

    def name(self) -> str:
        """The key this demand's ADDRESS rides under — in a placement reply
        and in a delivery's routes. A pool is its own name; the learner
        member has none of its own, so it is LEARNER_ROUTE, the one name the
        whole plane calls it by (host.py owns the constant, because the door
        that resolves the route is the one that must agree)."""
        return self.pool or LEARNER_ROUTE

    def per_device_gb(self) -> float | None:
        """The memory this demand needs on EACH device it spans: the total
        divided across its shards. None stays None — a whole device."""
        return None if self.vram_gb is None else self.vram_gb / self.shape


def demand_rows(demands: Sequence[Demand]) -> list[dict]:
    """Demands as wire rows — the desk's whole input vocabulary."""
    return [{"pool": d.pool, "capability": d.capability, "base": d.base,
             "shape": d.shape, "vram_gb": d.vram_gb, "group": d.group,
             "anchor": d.anchor, "adapter_types": list(d.adapter_types)}
            for d in demands]


def demands_from(rows: Sequence[Mapping]) -> tuple[Demand, ...]:
    """Wire rows back as Demands — demand_rows' typed inverse."""
    return tuple(Demand(
        pool=row["pool"], capability=row["capability"], base=row["base"],
        shape=int(row["shape"]),
        vram_gb=None if row.get("vram_gb") is None else float(row["vram_gb"]),
        group=int(row["group"]),
        anchor=bool(row.get("anchor", False)),
        adapter_types=tuple(row.get("adapter_types", ()))) for row in rows)


def builds_proposed(payload: Mapping) -> Builds | None:
    """The recipe a registration PROPOSES, read off the wire (ADR 0007, Q4).

    THE WIRE SPEAKS ROWS AND THE DESK SPEAKS RECORDS, and this is the one
    place between them: a metal file that chose to say what it is for sends
    `Builds.row()`, and the desk holds the typed record it journals and
    carves from. A registration that proposes nothing is the normal case —
    the metal boots bare and the declaration comes from the desk's own
    door."""
    row = payload.get("builds")
    return Builds.from_row(dict(row)) if row else None


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

RESIDUAL_DEADLINE_S = 5.0
"""THE BOUND ON THE LIVE RESIDUAL READ (ADR 0008, Q3 as amended). Placement
asks every LIVE metal for its residual CONCURRENTLY, so the worst case per
submit is one deadline and not one per metal — and a metal that misses it is
journaled `unreachable` and passed over, where an unbounded ask under the
lock wedged the desk for an hour."""

PROBE_DEADLINE_S = 10.0
"""The bound on one listing's status probe — the reaper's and placement's."""


IN_FLIGHT_S = 300.0
"""How long a `submit-intent` with no delivery after it is taken to mean a
submission STILL IN FLIGHT (ADR 0008, F4). A second frame arriving inside that
window is refused loudly rather than placed a second time; one arriving after
it belongs to an attempt whose container died before it delivered, and is
placed afresh."""


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def submit_key(frame: Mapping) -> str:
    """THE IDEMPOTENCE KEY OF ONE SUBMISSION (ADR 0008, F4).

    Every wire is at-least-once — Modal replays an input off a container it
    shut down, and on 2026-09-04 one such replay adopted the same run on two
    metals — so a state-changing verb has to be idempotent BY ITS KEY. The
    key is a digest of the OPAQUE FRAME, because the desk is workload-blind:
    it cannot compute a run id (that needs the spec and the code claim, which
    are the anchor host's business), and it does not have to — a run id is a
    pure function of exactly what this frame carries, so two frames with the
    same digest are two deliveries of one submission, which is the whole of
    what the key must decide."""
    return hashlib.sha256(
        json.dumps(frame, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()[:16]


def occupied_in(told: Mapping | None) -> bool:
    """Is a running tenancy on this host, read off ONE status frame? The one
    reading, shared by the live probe and the snapshot placement is decided
    against."""
    return any(t.get("status") == "running"
               for t in (told or {}).get("tenants", {}).values())


def first_fit(free: Sequence[float], count: int,
              gb: float) -> tuple[int, ...] | None:
    """`count` devices with `gb` free each, in device order — the same rule
    the metal's own `choose_devices` applies to its own books, said here
    against the SNAPSHOT so the desk's deduction and the metal's enforcement
    disagree only when the world moved between them."""
    chosen = [i for i, f in enumerate(free) if f >= gb - 1e-9][:count]
    return tuple(chosen) if len(chosen) == count else None


def wants_of_unit(unit: Sequence[Demand]) -> dict:
    """WHAT ONE PLACEMENT UNIT WANTS: the regimes it wears, how many devices
    it spans, and the GB each must hold. One projection, two readers — a boot
    instruction (nothing can hold this) and a parked run's `wants` (nothing
    can hold it YET), which are the same sentence at two moments."""
    return {"regimes": [regime_of(d).name for d in unit],
            "capabilities": sorted({d.capability for d in unit}),
            "base": unit[0].base,
            "devices": max(d.shape for d in unit),
            "vram_gb": max((d.per_device_gb() for d in unit
                            if d.per_device_gb() is not None), default=None)}


@dataclass
class Snapshot:
    """THE FLEET AS ONE PLACEMENT SEES IT (ADR 0008, Q3 as amended).

    Every live metal's residual and every live listing's status, read
    CONCURRENTLY under one deadline BEFORE the placement lock is taken, plus
    the names that did not answer. Placement is then decided against this
    reading and nothing else, which is what makes the decision pure enough to
    hold a lock over.

    Mutable on purpose: a carve RESERVES what it took, so the second unit of
    one submit deduces against what the first left. A reservation is not a
    booking — the metal's own books are the enforcement half — it only stops
    one desk from promising one device twice in one breath."""

    residual: dict[str, list[float]]
    rosters: dict[str, dict]
    unreachable: tuple[str, ...] = ()

    def reserve(self, metal: str, devices: Sequence[int], gb: float) -> None:
        """The devices this decision took, spent out of the reading."""
        for device in devices:
            self.residual[metal][device] -= gb

    def absorb(self, fresh: "Snapshot") -> None:
        """A second reading folded in — what a knock's newly woken metal adds
        to a placement already in progress."""
        self.residual.update(fresh.residual)
        self.rosters.update(fresh.rosters)
        self.unreachable = fresh.unreachable


@dataclass(frozen=True)
class Decision:
    """WHERE ONE PLACEMENT UNIT GOES, as a value: JOIN this listing, CARVE
    these devices on this metal, or neither. Separated from the act so the
    choosing can happen under the placement lock and the wire call cannot
    (F3) — and so `decide` is a pure function a test can read.

    `refusals` are the metals passed over with a reason worth journaling;
    they are written by the caller, outside the lock."""

    join: str = ""
    carve: str = ""
    devices: tuple[int, ...] = ()
    gb: float = 0.0
    refusals: tuple[tuple[str, str], ...] = ()

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

    async def told(self, deadline_s: float = PROBE_DEADLINE_S) -> dict | None:
        """ONE STATUS FRAME, or None where the container did not answer in
        time — the whole of what the desk can learn about a listing over the
        wire, asked once and read for both questions below (ADR 0008, F3:
        bounded, and taken outside every lock)."""
        try:
            return await self.host.status(deadline_s=deadline_s)
        except Exception:
            return None

    async def occupied(self, deadline_s: float = PROBE_DEADLINE_S) -> bool:
        """Asked over the wire, at placement time only: the roster is the
        host's, and the desk holds no copy that could go stale."""
        return occupied_in(await self.told(deadline_s))

    async def alive(self, deadline_s: float = PROBE_DEADLINE_S) -> bool:
        """Does the container behind this listing still answer? A listing is
        a description, so the only way to know is to ask — at placement time,
        never cached: a host that died between placements must not be offered,
        and one that came back must not stay buried."""
        return await self.told(deadline_s) is not None


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
                 boot_for: Callable[[str], None] | None = None,
                 *, idle_s: float | None = IDLE_S,
                 terminate_for: Callable[[str], Any] | None = None,
                 lease_s: float = LEASE_S,
                 heartbeat_s: float = HEARTBEAT_S,
                 residual_deadline_s: float = RESIDUAL_DEADLINE_S,
                 probe_deadline_s: float = PROBE_DEADLINE_S,
                 clock: Callable[[], float] = time.time,
                 ) -> None:
        self.store = store
        # THE CLOCK, injectable (ADR 0008): every lease reading goes through
        # `now()`, so a test can drive a whole lease's worth of silence in a
        # millisecond instead of sleeping through a minute of it.
        self.clock = clock
        # THE LEASE TABLE (F1): name -> Lease, over metal names AND host
        # names, because both renew by heartbeat and both are disbelieved the
        # same way. One table because there is one rule; the two kinds never
        # collide in practice (a carve name embeds its metal, its devices and
        # a counter), and `heartbeat` names which kind it found.
        self.leases: dict[str, Lease] = {}
        self.lease_s = lease_s
        self.heartbeat_s = heartbeat_s
        # THE TWO DEADLINES PLACEMENT READS UNDER (F3), desk constants: every
        # live metal asked for its residual concurrently under the first,
        # every live listing probed under the second, and a miss journaled
        # `unreachable` and passed over rather than waited for.
        self.residual_deadline_s = residual_deadline_s
        self.probe_deadline_s = probe_deadline_s
        # and the bound on a KNOCK made through the plane address (ADR 0003's
        # `describe` — on Modal that call IS the boot): a boot is seconds to
        # minutes, and one that has not answered by then stayed down.
        self.boot_deadline_s = BUILD_DEADLINE_S
        # metal name -> the residual its last heartbeat carried. FOR THE ROW
        # ONLY (Q3, as amended): placement reads residuals LIVE, because a
        # live read also proves the metal is reachable now. This is what the
        # observer shows between reads.
        self.residuals: dict[str, list[float]] = {}
        # address -> RemoteHost: how the desk reaches a listed host
        self.host_for = host_for
        # address -> RemoteMetal: how it reaches the METAL PLANE — the verbs
        # that create and free hosts (carve/decarve) and the residual the
        # desk deduces from. Venue like `host_for`; a desk born without it
        # still places over listings and answers misses with boot
        # instructions, it just cannot command a carve.
        self.metal_for = metal_for
        # name -> the venue's way to BOOT a metal whose container is gone,
        # for venues where knocking the plane address does not boot it. None
        # means the knock IS the boot (Modal: a call to a stopped-but-deployed
        # container starts one), so the reaper knocks with `describe()`.
        self.boot_for = boot_for
        # THE HAND THAT ENDS A CONTAINER (2026-09-05): an awaitable of the
        # container id a metal registered with, which the substrate uses to
        # terminate that container. Without it a released metal only stops
        # taking inputs and stands, billed, until the venue's scaledown.
        self.terminate_for = terminate_for
        self.metal_containers: dict[str, str | None] = {}
        self.metal: dict[str, Metal] = {}
        self.metal_addresses: dict[str, str | None] = {}
        self.metal_remotes: dict[str, "RemoteMetal"] = {}
        # name -> the metal's RECIPE: the CANON a carve is built from (ADR
        # 0001, Q5c — it rides every carve request, so a reborn container is
        # rebuilt from the desk's row and never from its own constants),
        # journaled as `recipe` events so the record of HOW a host was built
        # survives the desk. Since ADR 0007 the recipe is DECLARED HERE (a
        # metal boots bare and a registration only PROPOSES one), and a carve
        # to a metal this table has no row for is refused: the desk never
        # builds, but it is the one that says what to build.
        self.metal_builds: dict[str, Builds] = {}
        # THE IDLE POLICY (ADR 0003). `idle_s` is this desk's default limit in
        # seconds; `metal_idle_s` holds the metals whose registration declared
        # their own (None there PINS that metal — never released), and a name
        # absent from it takes the desk's. None here is a desk that releases
        # nothing.
        self.idle_s = idle_s
        self.metal_idle_s: dict[str, float | None] = {}
        # name -> when this metal was FIRST observed idle. The clock lives in
        # MEMORY (Q2): a desk restart forgets it and costs at most one tick,
        # where journaling every observation would be noise in the fleet log
        # for a timer whose only reader is the next tick.
        self.idle_since: dict[str, float] = {}
        # host name -> its `admitted` counter at the previous observation —
        # what makes "nothing has run here since the last tick" decidable
        self.admitted_at: dict[str, int] = {}
        # metal this desk RELEASED: still INVENTORY (self.metal — what the
        # fleet owns) and no longer carve-able (out of metal_remotes — what
        # the fleet may carve). A knock brings it back (Q4).
        self.released: set[str] = set()
        self.listings: dict[str, Listing] = {}
        # run ids a reroute is IN FLIGHT for — the claim that keeps a reap's
        # retry and a registration's retry off the same run without either of
        # them holding a lock across the wire (ADR 0008, F3).
        self.retrying: set[str] = set()
        self._placing: asyncio.Lock | None = None
        self._placing_loop: asyncio.AbstractEventLoop | None = None

    @classmethod
    def from_journal(cls, store: Store,
                     host_for: Callable[[str], "RemoteHost"],
                     metal_for: Callable[[str], "RemoteMetal"] | None = None,
                     boot_for: Callable[[str], None] | None = None,
                     *, idle_s: float | None = IDLE_S,
                     terminate_for: Callable[[str], Any] | None = None,
                     lease_s: float = LEASE_S,
                     heartbeat_s: float = HEARTBEAT_S,
                     residual_deadline_s: float = RESIDUAL_DEADLINE_S,
                     probe_deadline_s: float = PROBE_DEADLINE_S,
                     clock: Callable[[], float] = time.time,
                     ) -> "Desk":
        """The desk, rebuilt from its own record: every `list` event resolves
        its address again, and so does every addressed `metal` event — the
        latest `metal` event per name wins, so a re-registration's measured
        facts and idle declaration replay exactly as they landed. Every
        `recipe` event replays too, latest per metal winning, which is how a
        declaration made at the desk's own door outlives the desk (ADR 0007,
        Q4). A `release` event takes its metal off the carve-able set and
        leaves the row as inventory; a LATER `metal` event for the same name
        clears the release, because registering is what re-acquires (ADR
        0003). Kill -9 the desk and nothing was lost but a process — the same
        recovery shape as attach, on the fleet plane. The idle CLOCK is not
        replayed: it lives in memory by ruling (Q2), so a rebuilt desk starts
        every metal's clock again.

        NEITHER IS THE LEASE CLOCK (ADR 0008, F1). Each replayed row gets its
        EPOCH off the journal and a lease heard AT REBUILD TIME: the rule is
        "believe nothing you have not heard from within the lease", and a
        desk that has been alive for a millisecond has not had a chance to
        hear anyone. So the rebuilt desk grants one lease of grace and then
        disbelieves whatever did not heartbeat inside it — which, at a 20 s
        cadence, is every container that is really gone."""
        desk = cls(store, host_for, metal_for, boot_for, idle_s=idle_s,
                   terminate_for=terminate_for,
                   lease_s=lease_s, heartbeat_s=heartbeat_s,
                   residual_deadline_s=residual_deadline_s,
                   probe_deadline_s=probe_deadline_s, clock=clock)
        born = desk.now()
        for event in store.read_fleet_log():
            if event.get("event") == "list":
                desk.listings[event["host"]] = _listing_from(event, host_for)
                desk.grant(event["host"], event.get("epoch", ""), born,
                           float(event.get("lease_s", lease_s)))
            elif event.get("event") == "delist":
                desk.listings.pop(event["host"], None)
                desk.leases.pop(event["host"], None)
            elif event.get("event") == "metal":
                desk.metal[event["name"]] = Metal(
                    name=event["name"], gpu=event.get("gpu", "L4"),
                    devices=int(event.get("devices", 1)),
                    vram_gb=float(event.get("vram_gb", 24.0)))
                address = event.get("address")
                desk.metal_addresses[event["name"]] = address
                desk.metal_containers[event["name"]] = event.get("container")
                desk.grant(event["name"], event.get("epoch", ""), born,
                           float(event.get("lease_s", lease_s)))
                if address and metal_for is not None:
                    desk.metal_remotes[event["name"]] = metal_for(
                        with_epoch(address, event.get("epoch", "")))
                desk.declare_idle(
                    event["name"],
                    event["idle_s"] if "idle_s" in event else DESK_DEFAULT)
                desk.released.discard(event["name"])
            elif event.get("event") == "recipe":
                desk.metal_builds[event["metal"]] = Builds.from_row(
                    event["builds"])
            elif event.get("event") == "release":
                desk.released.add(event["metal"])
                desk.metal_remotes.pop(event["metal"], None)
                desk.leases.pop(event["metal"], None)
        return desk

    # ---- the lease: what the desk is still entitled to believe (F1) ---------

    def now(self) -> float:
        """This desk's clock, through the one seam every lease reading takes."""
        return self.clock()

    def grant(self, name: str, epoch: str, heard_t: float,
              lease_s: float | None = None) -> Lease:
        """A lease opened or REPLACED for `name` at `epoch` (F1). A new epoch
        is a new instance, so the lease is a new one — there is nothing to
        renew, because the thing that held the old one is gone."""
        lease = Lease(epoch=epoch, heard_t=heard_t,
                      lease_s=self.lease_s if lease_s is None else lease_s)
        self.leases[name] = lease
        return lease

    def leased(self, name: str) -> bool:
        """IS THIS THING STILL BELIEVED? The one liveness gate, consulted by
        the join rung, the carve rung and the reaper.

        It is not folded into `covers` because a lease is not a property of
        the DESCRIPTION a listing is — `covers` compares regimes and recipes,
        which are birth facts and never change — but of what the desk has
        HEARD, which lives here with the heartbeats that renew it. Something
        this desk holds no lease for at all is believed: a hand-built host
        listed by a deploy that never heartbeats is exactly the case F1 has
        no opinion about, and a rule refuses on evidence or not at all."""
        lease = self.leases.get(name)
        return lease is None or lease.live(self.now())

    def lapsed(self) -> list[str]:
        """Every name whose lease has run out, in name order — the reaper's
        first pass, and the whole of F1's teeth."""
        now = self.now()
        return sorted(name for name, lease in self.leases.items()
                      if not lease.live(now))

    def epoch_of(self, name: str) -> str:
        """The instance the desk believes answers for `name` — what it stamps
        into every frame it addresses there (F2). Empty where nothing has
        announced an epoch, which addresses whoever answers."""
        lease = self.leases.get(name)
        return lease.epoch if lease is not None else ""

    async def heartbeat(self, name: str, epoch: str,
                        residual: Sequence[float] | None = None) -> dict:
        """ONE RENEWAL (F1): `name` is still there, and it is still `epoch`.

        A metal heartbeats for its container and for every host carved on it,
        because a host is an object inside that container's one control
        process and its residency is exactly as alive as the container is. The
        residual rides along FOR THE ROW (Q3, as amended): placement reads it
        live, so what is cached here is only what the observer shows between
        placements.

        A heartbeat for a name this desk has no lease for is REFUSED rather
        than silently opening one: a lease is opened by a registration or a
        listing, and a heartbeat from nowhere means the container is talking
        to a desk that has forgotten it — which is a re-registration's job,
        and the reply says so. A heartbeat at the WRONG epoch is refused the
        same way: whoever sent it is a life the desk has already replaced.

        Not journaled. A heartbeat every 20 s per metal is a cadence, not a
        fact — the fleet journal records what the fleet DID, and the lease
        constants that make this readable are journaled at registration."""
        lease = self.leases.get(name)
        if lease is None:
            return {"heard": False, "name": name,
                    "error": f"{name!r} holds no lease at this desk — "
                             f"register the metal again (a registration is "
                             f"what opens a lease)"}
        if epoch and lease.epoch and epoch != lease.epoch:
            return {"heard": False, "name": name, "epoch": lease.epoch,
                    "error": f"{name!r} is epoch {lease.epoch!r} at this "
                             f"desk; a heartbeat from epoch {epoch!r} is a "
                             f"life this desk has already replaced"}
        now = self.now()
        self.leases[name] = lease.renewed(now)
        if residual is not None and name in self.metal:
            self.residuals[name] = [float(gb) for gb in residual]
        return {"heard": True, "name": name, "epoch": lease.epoch,
                "heard_t": now, "lease_s": lease.lease_s,
                "heartbeat_s": self.heartbeat_s}

    # ---- the recipe: declared at the desk, journaled, carried by the carve --

    def recipe(self, metal: str, builds: Builds) -> None:
        """WHAT THIS METAL BUILDS, DECLARED HERE (ADR 0007, Q4). A metal boots
        BARE — measuring a card and mounting a store are the container's, but
        what to serve is a declaration, and the desk's row was already the
        canon every carve carried (ADR 0001, Q5c). Journaled, so the
        declaration outlives the desk process; the latest per metal wins, so
        an operator's door and a registration's proposal are the same event
        and the last one said is the one that rides."""
        self.metal_builds[metal] = builds
        self.store.append_fleet_event({
            "event": "recipe", "t": time.time(), "metal": metal,
            "builds": builds.row()})

    def put_plan(self, data: bytes) -> dict:
        """PLAN BYTES INTO THE CAS, THROUGH THE DESK (ADR 0008, F6 / Q5).

        A spec's plans are content: a client builds them, hashes them into a
        cas uri, and the uri is what the spec carries and what its identity
        is computed over. Before this verb the client needed the store's
        MOUNT to write them, which is why every venue door ran a `canonical`
        function on an on-demand CPU container — and for an hour on
        2026-09-04 Modal scheduled none of them and no arm could start.

        The desk has the mount, so the desk takes the bytes. IDEMPOTENT BY
        CONSTRUCTION: `cas_put` is content-addressed, so putting the same
        plan twice is putting it once and the uri is the proof. The desk
        reads nothing — the bytes are as opaque here as a delivered frame."""
        return {"uri": self.store.cas_put(data), "bytes": len(data)}

    def recipe_for(self, metal: str) -> Builds | None:
        """This metal's declared recipe, or None — a metal nothing has
        declared for cannot be carved on, and the join rung will not offer
        its listings for adapter types it never said it serves."""
        return self.metal_builds.get(metal)

    def register_metal(self, metal: Metal, address: str | None = None,
                       builds: Builds | None = None,
                       idle_s: float | None | Undeclared = DESK_DEFAULT,
                       container: str | None = None,
                       epoch: str = "") -> bool:
        """The acquire rung, recorded at the desk: what the fleet OWNS and may
        carve against. Journaled like a listing, so a rebuilt desk knows its
        inventory too. `address` is where that metal's own container answers
        the metal plane (carve/decarve/residual) — with it and a resolver the
        desk can command the standing carve; without it the row is inventory
        only and misses still answer with boot instructions. `builds` is a
        PROPOSAL (ADR 0007, Q4): a metal file that wants to say what it is
        for may, and the desk journals it as the same `recipe` event its own
        door writes — but a metal that proposes nothing registers, lists and
        knocks just the same and simply cannot be carved on until something
        declares for it.

        A KNOWN NAME AT THE SAME ADDRESS IS THE CONTAINER GENERATION TURNING
        OVER (ADR 0001, Q5): the row is overwritten with the frame's measured
        facts (and its recipe, if it proposes one — a redeploy is a human's
        act and updates the canon) and journaled as a fresh `metal` event, so
        the rebuilt desk replays last-write-wins. RETURNS WHETHER THE NAME WAS
        KNOWN, because reconciling that metal's standing listings is a wire
        act (`reconcile_metal`) and this one is not: registering is a journal
        write, and since ADR 0008 nothing that writes the journal also waits
        on a container. A known name at a DIFFERENT address is two deploys
        colliding on a name, not a restart: refused loudly.

        REGISTERING IS RE-ACQUIRING (ADR 0003, Q4): a metal this desk had
        RELEASED comes back carve-able here, its release superseded by the
        fresh row on replay, and its idle clock starts again — a container
        that just announced itself has run nothing yet, but it has not been
        watched either. `idle_s` is this metal's own limit: unsaid, the
        desk's default decides; None PINS it.

        A REGISTRATION IS WHERE AN EPOCH IS ANNOUNCED AND A LEASE OPENS (ADR
        0008, F1/F2). The container mints its epoch at bring-up and says it
        here; the desk grants a fresh lease at that epoch, journals both plus
        the lease constants (so the record says what the fleet was promising),
        and stamps the epoch into every frame it sends that metal from now on.
        A NEW epoch at the SAME address is a container generation turning
        over, which is what the corpse reconciliation below is for — and the
        old epoch's frames are refused by the newborn from that moment."""
        known = metal.name in self.metal
        if known and self.metal_addresses.get(metal.name) != address:
            raise DeskError(
                f"metal {metal.name!r} is already registered at "
                f"{self.metal_addresses.get(metal.name)!r}; a registration "
                f"from {address!r} is a second deploy colliding on the name, "
                f"not a restart — deregister or rename it")
        self.metal[metal.name] = metal
        self.metal_addresses[metal.name] = address
        # the container this metal lives in, as the substrate names it: what
        # `release` terminates once the metal has handed everything back
        self.metal_containers[metal.name] = container
        now = self.now()
        self.grant(metal.name, epoch, now)
        self.residuals.pop(metal.name, None)
        if address and self.metal_for is not None:
            self.metal_remotes[metal.name] = self.metal_for(
                with_epoch(address, epoch))
        self.declare_idle(metal.name, idle_s)
        self.released.discard(metal.name)
        self.idle_since.pop(metal.name, None)
        row = {"event": "metal", "t": now, "name": metal.name,
               "gpu": metal.gpu, "devices": metal.devices,
               "vram_gb": metal.vram_gb, "address": address,
               "container": container,
               "epoch": epoch, "lease_s": self.lease_s,
               "heartbeat_s": self.heartbeat_s}
        if not isinstance(idle_s, Undeclared):
            row["idle_s"] = idle_s          # absent = the desk's own default
        self.store.append_fleet_event(row)
        if builds is not None:
            self.recipe(metal.name, builds)
        return known

    def declare_idle(self, name: str, idle_s: float | None | Undeclared) -> None:
        """One metal's own idle limit, exactly as its registration declared
        it: a number is that metal's limit in seconds, None PINS it (never
        released, however long it sits), and DESK_DEFAULT — nothing declared
        — leaves the desk's default to decide. A re-registration re-declares,
        so the table and the journal row always say what the venue last
        said."""
        if isinstance(idle_s, Undeclared):
            self.metal_idle_s.pop(name, None)
        else:
            self.metal_idle_s[name] = idle_s

    async def reconcile_metal(self, name: str) -> list[str]:
        """The reaper's conclusion scoped to ONE metal, with zero retries: a
        metal that has just re-registered is up and bare, so a listing on it
        that does not answer is dead, not rebooting. Each corpse is delisted
        with the reason journaled — no decarve, the memory freed itself when
        the container did — and its runs are STRANDED for the retry.

        TWO WAYS TO BE A CORPSE, and the first is free. A listing whose EPOCH
        is not the metal's current one belonged to a life that has ended
        (ADR 0008, F2): no probe can make that untrue, and none is taken.
        Everything else is PROBED, concurrently and bounded — which is what
        makes a double `up` a no-op: living hosts answer and stay listed, and
        a host carved onto the newborn before its frame landed answers too."""
        epoch = self.epoch_of(name)
        mine = [host for host in sorted(self.listings)
                if self.listings[host].metal == name]
        stale = [host for host in mine
                 if epoch and self.epoch_of(host) and self.epoch_of(host) != epoch]
        asked = [host for host in mine if host not in stale]
        answers = await asyncio.gather(
            *(self.listings[host].alive(self.probe_deadline_s)
              for host in asked))
        corpses = sorted(stale + [host for host, answered
                                  in zip(asked, answers) if not answered])
        for host_name in corpses:
            self.delist(host_name, reason="metal re-registered")
        self.strand(corpses)
        return corpses

    def list_host(self, name: str, regimes: Sequence[Regime], address: str,
                  solo: bool = False, partition: Mapping | None = None,
                  metal: str = "", epoch: str = "") -> None:
        """A host enters the standing fleet: the deploy that booted it lists
        it here, once, and the desk journals the listing so a rebuilt desk
        knows it too. `partition` and `metal` are the capacity VIEW — the row
        the host was born onto and the registered Metal it lives on — carried
        so the desk can deduce and the reaper can free; enforcement stays at
        the metal's own books. Refuses a taken name — a listing
        is never replaced.

        `epoch` is the INSTANCE this host is (ADR 0008, F1/F2): its metal
        container's, because a host is an object inside that container's one
        control process. It opens the host's own lease — a host that stops
        being heartbeated for is delisted within one lease and its runs are
        parked, which is the whole of Q7 — and it rides the listing's address
        already, so every frame the desk sends this host names it."""
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
        now = self.now()
        self.grant(name, epoch, now)
        self.store.append_fleet_event({
            "event": "list", "t": now, "host": name,
            "address": address, "solo": solo,
            "epoch": epoch, "lease_s": self.lease_s,
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
        self.leases.pop(name, None)
        self.store.append_fleet_event({
            "event": "delist", "t": self.now(), "host": name,
            "reason": reason})

    # ---- placement over listings (the join rung; carve is a venue action) ---

    async def read_fleet(self) -> "Snapshot":
        """THE LIVE READ (ADR 0008, Q3 as amended), and the whole reason
        placement is no longer a way to wedge this desk.

        Every LIVE metal is asked for its residual and every LIVE listing for
        its status — CONCURRENTLY, each under the same deadline, and OUTSIDE
        every lock this class holds. The worst case per submit is therefore
        one deadline rather than one per metal, and a metal or host that
        misses it is journaled `unreachable` and passed over rather than
        waited for.

        A live read and not the heartbeat's cached number, because the read
        is the stronger check: it proves the metal is reachable NOW, and it
        sees a decarve that a residual up to one heartbeat old would miss.
        The heartbeat's residual is for the row the observer shows.
        """
        metals = [name for name in sorted(self.metal_remotes)
                  if self.leased(name)]
        hosts = [name for name in sorted(self.listings) if self.leased(name)]
        answers = await asyncio.gather(
            *(self.residual_of(name) for name in metals),
            *(self.listings[name].told(self.probe_deadline_s)
              for name in hosts))
        residual = dict(zip(metals, answers[:len(metals)]))
        rosters = dict(zip(hosts, answers[len(metals):]))
        missed = ([("metal", name) for name in metals
                   if residual[name] is None]
                  + [("host", name) for name in hosts
                     if rosters[name] is None])
        for kind, name in missed:
            self.journal_unreachable(kind, name)
        return Snapshot(
            residual={name: gb for name, gb in residual.items()
                      if gb is not None},
            rosters={name: told for name, told in rosters.items()
                     if told is not None},
            unreachable=tuple(name for _, name in missed))

    async def residual_of(self, name: str) -> list[float] | None:
        """One metal's free GB per device, or None where it did not answer
        inside the deadline. The refusal is not distinguished from a silence
        on purpose: both mean the desk may not deduce against this metal now."""
        try:
            return await self.metal_remotes[name].residual(
                deadline_s=self.residual_deadline_s)
        except Exception:
            return None

    def journal_unreachable(self, kind: str, name: str) -> None:
        """A WAIT THAT EXPIRED, ON THE ROW IT WAS ABOUT (ADR 0008, F3). Not a
        delisting and not a reap — only the record that this metal or this
        host was asked and did not answer, which is what turns "the submit
        was slow" into a row an operator can point at. The reaper's lease
        rule is what concludes a silence that persists."""
        self.store.append_fleet_event({
            "event": "unreachable", "t": self.now(), kind: name,
            "deadline_s": (self.residual_deadline_s if kind == "metal"
                           else self.probe_deadline_s)})

    def placing(self) -> asyncio.Lock:
        """THE PLACEMENT LOCK — and the only thing ever held under it is a
        DECISION (ADR 0008, F3).

        Two campaigns submitting at once must not each deduce against the same
        residual, so the choice is serialized; but nothing inside this lock
        touches the wire, the store or the clock. The read that feeds it
        happens before it (`read_fleet`) and the carve it decides on happens
        after it (`carve_on`), because a lock held across a wire call is how
        one unreachable metal queued every submit, status and reap behind it
        for an hour.

        One lock per running loop, because the desk outlives any single
        asyncio.run (tests drive one desk through several) and a Lock is
        bound to the loop that first waits on it.

        IT IS THE ONLY LOCK IN THIS CLASS. The recontinue's mutual exclusion
        is a CLAIM SET (`claim`) precisely because its work is nothing but
        wire calls; there is nowhere else a lock is taken, which is what
        makes "no lock is held across the wire" a property of the file and
        not a habit."""
        loop = asyncio.get_running_loop()
        if self._placing is None or self._placing_loop is not loop:
            self._placing, self._placing_loop = asyncio.Lock(), loop
        return self._placing

    async def place_listings(self, demands: Sequence[Demand],
                             avoid: frozenset[str] = frozenset(),
                             solo: bool = False,
                             ) -> tuple[dict[str | None, Listing], list[dict]]:
        """Every placement unit onto a listing, a fresh carve, or the boot
        list: what no listed host serves is CARVED on a registered metal that
        can hold it, and only what no metal can hold lands in `boot` — the
        standing acquire, a human's. A unit carved for a placement whose
        LATER unit then missed stays listed: metal born is metal listed, and
        the next submit's join rung finds it. `avoid` passes to the join rung
        — a carve can never land on an avoided listing, because a carve is
        always a NEW name.

        ONE SNAPSHOT PER PLACEMENT (ADR 0008): the fleet is read live and
        whole before the first unit is decided, and every unit is placed
        against that reading — a carve reserving what it took, so two units
        of one submit cannot both be promised the same device."""
        snapshot = await self.read_fleet()
        placement: dict[str | None, Listing] = {}
        boot: list[dict] = []
        if solo:
            # SOLO (2026-09-05): nothing that stood before this placement is
            # joined — every unit CARVES, on a metal with nothing standing
            # where one is registered live, never by knocking released metal
            # (a solo carve for a 0.6B third knocked two 32B pairs awake).
            # What it carves is the campaign's own and stays joinable: the
            # next submit's join rung finds it like any listing, which is how
            # six arms of one campaign share the card the first one carved.
            avoid = avoid | frozenset(self.listings)
        standing = frozenset(self.listings)      # what stood before this placement
        for unit in placement_units(demands):
            listing = await self.settle(unit, avoid, snapshot,
                                        solo=solo, standing=standing)
            if listing is None:
                boot.append(wants_of_unit(unit))
                continue
            for demand in unit:
                placement[demand.pool] = listing
        return placement, boot

    async def settle(self, unit: tuple[Demand, ...], avoid: frozenset[str],
                     snapshot: "Snapshot", solo: bool = False,
                     standing: frozenset[str] = frozenset()) -> Listing | None:
        """ONE UNIT PLACED: decide, act, and — where nothing standing fits —
        KNOCK a metal the desk RELEASED whose recorded facts could hold it,
        read the fleet again, and decide once more (ADR 0003, Q4).
        Re-acquiring metal the fleet already owns needs no human: the human
        act was the deploy. What no metal, released or live, can hold is a
        boot instruction, and that stays a human's."""
        listing = await self.attempt(unit, avoid, snapshot, solo, standing)
        if listing is not None or solo:
            return listing          # solo never knocks: a fresh card is live metal or a boot
        if not await self.knock_released(unit):
            return None
        snapshot.absorb(await self.read_fleet())
        return await self.attempt(unit, avoid, snapshot, solo, standing)

    async def attempt(self, unit: tuple[Demand, ...], avoid: frozenset[str],
                      snapshot: "Snapshot", solo: bool = False,
                      standing: frozenset[str] = frozenset()) -> Listing | None:
        """The decision under the lock, the act outside it, and a metal that
        refuses its carve tried once and then passed over — the metal's own
        booking is the enforcement half, so a deduction gone stale between
        the read and the command costs one refusal and never a double-book."""
        tried: set[str] = set()
        while True:
            async with self.placing():
                decision = self.decide(unit, avoid, snapshot, tried,
                                       solo, standing)
            for metal_name, reason in decision.refusals:
                self.refuse_carve(metal_name, reason)
            if decision.join:
                return self.listings.get(decision.join)
            if not decision.carve:
                return None
            born = await self.carve_on(decision, unit, solo)
            if born is not None:
                return born
            tried.add(decision.carve)

    def decide(self, unit: tuple[Demand, ...], avoid: frozenset[str],
               snapshot: "Snapshot",
               tried: set[str] = frozenset(), solo: bool = False,
               standing: frozenset[str] = frozenset()) -> "Decision":
        """THE PLACEMENT LADDER AS A PURE FUNCTION — no wire, no store, no
        clock, so it is exactly what may be held under the placement lock.

        Rung one, JOIN: sorted-name order, coverage by `covers()` (the one
        join rule — capability equality plus the metal recipe's `serves`,
        matched against descriptions), a listing whose LEASE has lapsed
        skipped (F1), one that did not ANSWER the snapshot's probe skipped
        (placement must never offer a host it cannot reach), solo-and-occupied
        skipped, and names in `avoid` off the table — a reroute excluding the
        listing being torn down.

        Rung two, CARVE: the first live metal whose declared recipe serves
        every adapter type the unit names and whose SNAPSHOT residual holds
        the unit — and taking it RESERVES those devices in the snapshot, so a
        second unit of the same placement deduces against what is left. A
        metal the desk could not read is not in the snapshot at all and is
        therefore passed over, which is `unreachable` doing its work.

        Rung three is not here: what nothing holds is no decision, and the
        caller turns it into a knock or a boot instruction."""
        joined = self.join_rung(unit, avoid, snapshot)
        if joined:
            return Decision(join=joined)
        return self.carve_rung(unit, snapshot, tried, solo, standing)

    def join_rung(self, unit: tuple[Demand, ...], avoid: frozenset[str],
                  snapshot: "Snapshot") -> str:
        """RUNG ONE, the join rule alone: the name of the first listing that
        covers this unit and may take it, or "" — see `decide`."""
        for name in sorted(self.listings):
            if name in avoid or not self.leased(name):
                continue
            if name not in snapshot.rosters:
                continue        # did not answer the live probe: not offerable
            listing = self.listings[name]
            if not all(covers(listing, d, self.recipe_for(listing.metal))
                       for d in unit):
                continue
            if listing.solo and occupied_in(snapshot.rosters[name]):
                continue
            return name
        return ""

    def carve_rung(self, unit: tuple[Demand, ...], snapshot: "Snapshot",
                   tried: set[str] = frozenset(), solo: bool = False,
                   standing: frozenset[str] = frozenset()) -> "Decision":
        """RUNG TWO, the carve alone, RESERVING what it takes — see
        `decide`. A SOLO carve wants a card of its own: metals with nothing
        STANDING on them (listed before this placement began — this
        placement's own carves keep its units together) come first, then the
        usual name order."""
        refusals: list[tuple[str, str]] = []
        need_devices = max(demand.shape for demand in unit)
        occupied = {self.listings[host].metal for host in standing
                    if host in self.listings}
        for metal_name in sorted(self.metal_remotes,
                                 key=lambda n: (solo and n in occupied, n)):
            if metal_name in tried or not self.leased(metal_name):
                continue
            if metal_name not in snapshot.residual:
                continue        # unreachable this placement: passed over
            recipe = self.recipe_for(metal_name)
            if recipe is None:
                refusals.append((
                    metal_name,
                    "no recipe is declared for this metal at this desk "
                    "(desk.recipe): a bare metal cannot know what to build"))
                continue
            if not all(recipe_serves(recipe, demand) for demand in unit):
                refusals.append((
                    metal_name,
                    f"its recipe serves {sorted(serves_of(recipe))} and the "
                    f"unit names "
                    f"{sorted({a for d in unit for a in d.adapter_types})}"))
                continue
            need_gb = unit_gb(unit, self.metal[metal_name])
            devices = first_fit(snapshot.residual[metal_name], need_devices,
                                need_gb)
            if devices is None:
                continue
            snapshot.reserve(metal_name, devices, need_gb)
            return Decision(carve=metal_name, devices=devices, gb=need_gb,
                            refusals=tuple(refusals))
        return Decision(refusals=tuple(refusals))

    async def find_listing(self, unit: tuple[Demand, ...],
                           avoid: frozenset[str] = frozenset(),
                           ) -> Listing | None:
        """THE JOIN RUNG ASKED ON ITS OWN: is there a standing host this unit
        could join right now? Reads the fleet live (F3) and answers off that
        snapshot — a question a test and an operator ask, and the rung
        `decide` climbs first."""
        snapshot = await self.read_fleet()
        name = self.join_rung(unit, avoid, snapshot)
        return self.listings.get(name) if name else None

    async def carve_on(self, decision: "Decision",
                       unit: tuple[Demand, ...],
                       solo: bool = False) -> Listing | None:
        """THE STANDING CARVE, desk-issued and OUTSIDE THE LOCK: the desk
        deduced from the snapshot and now COMMANDS the metal it chose. The
        metal ENFORCES — it books the GB synchronously at its own door and
        converts to its partition's fraction at build — so a deduction gone
        stale between the read and the command costs a refusal, never a
        double-book. The request carries this desk's `builds` row for the
        metal (ADR 0001, Q5c). What comes back is journaled and listed HERE:
        the desk stays the fleet journal's one writer, which is exactly why
        the metal writes nothing.

        A build is minutes, which is precisely why this may not happen under
        a lock."""
        metal_name = decision.carve
        request = self.carve_request(unit, metal_name, decision.gb, solo=solo)
        try:
            born = await self.metal_remotes[metal_name].carve(request)
        except Exception:
            return None
        if not born.get("carved"):
            return None                 # raced: booked away between read and command
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
            "event": "provision", "t": self.now(), "host": born["host"],
            "metal": metal_name, "request": request,
            "partition": born.get("partition")})
        self.list_host(
            born["host"],
            tuple(Regime(r["name"], r["capability"], r["base"], r["shape"])
                  for r in born["regimes"]),
            born["address"], solo=bool(born.get("solo", False)),
            partition=born.get("partition"), metal=metal_name,
            # a carved host IS its metal's container (F2): one process,
            # one epoch, one lease renewed by the same heartbeat
            epoch=born.get("epoch") or self.epoch_of(metal_name))
        return self.listings[born["host"]]

    def could_hold(self, unit: tuple[Demand, ...], metal: Metal) -> bool:
        """Could this metal hold the unit IF IT WERE BARE? Its recorded
        facts — devices against the unit's widest shard shape, one device's
        VRAM against the unit's per-device need — because a released
        container answers no residual. A deduction from the registration,
        which the metal itself refuses if it comes back and cannot in fact
        hold it."""
        return (metal.devices >= max(demand.shape for demand in unit)
                and metal.vram_gb >= unit_gb(unit, metal) - 1e-9)

    async def knock_released(self, unit: tuple[Demand, ...]) -> bool:
        """The first RELEASED metal (by name — the order `settle` already
        places in) that could hold the unit, knocked back to life. A knock
        that boots nothing leaves the metal released and the placement falls
        through to the boot instructions."""
        for name in sorted(self.released):
            if not self.could_hold(unit, self.metal[name]):
                continue
            if await self.reacquire(name):
                return True
        return False

    def refuse_carve(self, metal_name: str, reason: str) -> None:
        """A metal PASSED OVER before it was ever commanded, journaled. The
        loud half of ADR 0007's recipe rule: a carve that cannot be built is
        refused HERE, with the reason on the record, rather than half-built
        at a bare metal or refused late at the host's Phase 0. Written by the
        caller of `decide`, never inside the placement lock — the journal is
        I/O like any other."""
        self.store.append_fleet_event({
            "event": "carve-refused", "t": self.now(), "metal": metal_name,
            "reason": reason})

    def carve_request(self, unit: tuple[Demand, ...], metal_name: str,
                      need_gb: float, solo: bool = False) -> dict:
        """The carve command as the metal verb speaks it: the regimes the
        host will wear, the per-device GB its partition must hold, and the
        recipe it builds from — this desk's journaled recipe for that metal,
        the canon a reborn container is rebuilt from (Q5c) and, since ADR
        0007, the only recipe there is: the metal boots bare."""
        recipe = self.recipe_for(metal_name)
        return {
            "regimes": [{"name": regime_of(d).name,
                         "capability": d.capability,
                         "base": d.base, "shape": d.shape} for d in unit],
            "base": unit[0].base,
            "vram_gb": need_gb,
            "solo": False,              # carved FOR a campaign, joinable by the next submit
            "builds": recipe.row() if recipe is not None else None,
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
        pools = {demand.name(): placement[demand.pool].address
                 for demand in demands}
        self.store.append_fleet_event({
            "event": "place", "t": time.time(), "delivered": False,
            "pools": {pool or LEARNER_ROUTE: listing.name
                      for pool, listing in placement.items()}})
        return {"placed": True, "pools": pools}

    async def submit(self, rows: Sequence[Mapping], frame: Mapping,
                     solo: bool = False) -> dict:
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
        key = submit_key(frame)
        standing = await self.already_submitted(key)
        if standing is not None:
            return standing
        self.journal_intent(key, frame)
        placement, boot = await self.place_listings(demands, solo=solo)
        if boot:
            # THE INTENT IS CLOSED BY ITS OUTCOME, and a miss is one (F4): a
            # submission that found nowhere to go is over, so the next frame
            # with this key is a fresh attempt and not a replay of this one
            self.store.append_fleet_event({
                "event": "submit-missed", "t": self.now(), "key": key,
                "boot": boot})
            return {"accepted": False, "boot": boot,
                    "error": "no listed host serves these units and no "
                             "registered metal can hold them — boot or "
                             "register metal wearing the named regimes (the "
                             "standing acquire is a human's)"}
        return await self.deliver(demands, placement, rows, frame, key)

    def journal_intent(self, key: str, frame: Mapping) -> None:
        """THE INTENT, WRITTEN BEFORE THE ACT (ADR 0008, F4): this desk is
        about to place this submission. Written first, so a desk that dies
        between here and the delivery leaves the attempt on the record — and
        so a replayed frame arriving while the first is still in flight finds
        its own intent instead of placing a second time.

        The FOLDER is on the row because it is the other half of a run's
        address (#58: the same spec under two folders is two experiments), and
        it is the one field of the frame this desk reads — the envelope, never
        the contents."""
        self.store.append_fleet_event({
            "event": "submit-intent", "t": self.now(), "key": key,
            "folder": frame.get("subdir") or ""})

    async def already_submitted(self, key: str) -> dict | None:
        """HAS THIS EXACT SUBMISSION ALREADY BEEN PLACED? (ADR 0008, F4.)

        Three answers, and the middle one is the point. If an ACCEPTED
        delivery for this key is on the record AND the run it produced is
        still running somewhere, this frame is a REPLAY: it gets the delivery
        the first one got, identical, and nothing is placed twice — the
        journaled `submit-replayed` is where the fact that it happened
        lives. If an intent is on the record with no OUTCOME after it and
        younger than IN_FLIGHT_S, the first attempt is still going and this
        one is refused LOUDLY rather than waited for. Otherwise this is a
        fresh submission — which is what a RESUBMIT is (the same spec after a
        stop is how a run resumes, and it must place again), and what a retry
        after a miss is.

        THE KEY IS THE FRAME AND THE PREDICATE IS "STILL RUNNING", both
        deliberately. The ADR named the run id; the desk cannot compute one
        without reading the spec, and the frame's digest decides the same
        question, because a run id is a pure function of what this frame
        carries. And an accepted delivery whose run is NOT running is not a
        replay to answer but a resume to place: without that reading, stopping
        a run and resubmitting it — the way every resume on this fleet
        happens — would be answered with the delivery of the run that was
        stopped."""
        intent, delivery = None, None
        for event in self.store.read_fleet_log():
            if event.get("key") != key:
                continue
            kind = event.get("event")
            if kind == "submit-intent":
                intent, delivery = event, None
            elif kind == "place":
                # a refused delivery closes the intent without becoming one:
                # that attempt is over, and the next frame is a fresh attempt
                intent = None
                delivery = event if event.get("accepted") else None
            elif kind == "submit-missed":
                intent = None
        if delivery is not None:
            run_id = delivery.get("run_id")
            if run_id and run_id in await self.running_runs():
                self.store.append_fleet_event({
                    "event": "submit-replayed", "t": self.now(), "key": key,
                    "run_id": run_id, "host": delivery.get("host")})
                return {"accepted": True, "run_id": run_id,
                        "state": "running", "host": delivery.get("host"),
                        "pools": delivery.get("pools", {})}
            return None
        if intent is not None and self.now() - float(intent["t"]) < IN_FLIGHT_S:
            return {"accepted": False, "in_flight": True, "key": key,
                    "error": f"a submission with key {key} is already in "
                             f"flight at this desk (journaled "
                             f"{self.now() - float(intent['t']):.0f}s ago) — "
                             f"this frame is its replay; ask again once the "
                             f"first has landed"}
        return None

    async def deliver(self, demands: Sequence[Demand],
                      placement: Mapping[str | None, Listing],
                      rows: Sequence[Mapping], frame: Mapping,
                      key: str = "") -> dict:
        """The delivery half of a submission — submit's and reroute's ONE
        copy: the frame lands at the anchor demand's host with every other
        MEMBER's address threaded as routes under its own name (read off the
        demand rows alone, the blind relay), and the placement is journaled
        WITH the rows and the frame — the archive a reroute replays without
        the desk ever having decoded it.

        The learner is one such member since ADR 0006 Part A: its address is
        threaded like a pool's whenever it did not land on the anchor's own
        listing, and the desk still never learns what a learner is for."""
        anchor = next(d for d in demands if d.anchor)
        anchor_listing = placement[anchor.pool]
        routes = {d.name(): placement[d.pool].address for d in demands
                  if not d.anchor
                  and placement[d.pool] is not anchor_listing}
        try:
            reply = await anchor_listing.host.adopt(
                frame.get("spec"), routes, frame.get("code"),
                frame.get("subdir"))
        except Exception as down:
            # the probe passed and the container died between it and the
            # knock — or the adopt outlived its deadline: the reply says so
            # instead of the desk falling over, the row it was about is
            # journaled `unreachable` (F3), and the cure is a delist or a
            # reboot, both venue actions
            if isinstance(down, Unreachable):
                self.journal_unreachable("host", anchor_listing.name)
            return {"accepted": False, "host": anchor_listing.name,
                    "error": f"host {anchor_listing.name!r} did not answer "
                             f"the delivery: {down}"}
        pools = {pool or LEARNER_ROUTE: listing.name
                 for pool, listing in placement.items()}
        self.store.append_fleet_event({
            "event": "place", "t": self.now(), "delivered": True,
            "run_id": reply.get("run_id"),
            # the intent's key, closing the loop F4 opened: this is the
            # delivery a replayed frame will be answered with
            "key": key or submit_key(frame),
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

    async def dependents(self, name: str) -> list[str]:
        """Running runs whose LATEST journaled placement routes through
        listing `name` — the guard decommission refuses over."""
        return await self.dependents_on([name])

    async def metal_dependents(self, name: str) -> list[str]:
        """Running runs routing through ANY listing on metal `name` — the
        guard an explicit RELEASE refuses over (ADR 0007, Q6): a door that
        acquired metal must not tear it down under another experiment, and a
        release takes every host on the metal at once, so the question is
        asked of the whole metal rather than of one host."""
        return await self.dependents_on([host for host, listing
                                         in sorted(self.listings.items())
                                         if listing.metal == name])

    async def dependents_on(self, hosts: Sequence[str]) -> list[str]:
        """THE GUARD, one body: running runs whose LATEST journaled placement
        routes through any of `hosts`. Occupancy alone would miss half of
        them — a serve host's roster is empty (a tenancy lives at its
        anchor), but every placement journaled the pools it landed on, and
        the rosters say which runs still run."""
        wanted = set(hosts)
        if not wanted:
            return []
        placed = {rid: row["pools"]
                  for rid, row in self.placements().items()}
        running = await self.running_runs()
        return sorted(rid for rid, pools in placed.items()
                      if rid in running and wanted & set(pools.values()))

    async def running_runs(self) -> set[str]:
        """Every run some listing's roster says is RUNNING right now, off one
        concurrent bounded pass over the fleet (ADR 0008, F3). A silent host
        holds nothing running — that is the reaper's business, not this
        rule's — and a probe that expires says the same thing, bounded."""
        names = sorted(self.listings)
        answers = await asyncio.gather(
            *(self.listings[name].told(self.probe_deadline_s)
              for name in names))
        return {rid for told in answers
                for rid, said in (told or {}).get("tenants", {}).items()
                if said.get("status") == "running"}

    async def stop_anchored(self, run_id: str) -> dict:
        """Stop a tenancy WHEREVER it runs: probe the listings' rosters for
        the one carrying `run_id` running — tenancies live only at their
        anchor, so at most one listing answers — and tell that host to stop
        it (cancellation awaited host-side). A run nobody carries answers
        stopped: False; already dead is the goal state, not an error."""
        for name in sorted(self.listings):
            listing = self.listings[name]
            told = await listing.told(self.probe_deadline_s)
            if (told or {}).get("tenants", {}).get(
                    run_id, {}).get("status") == "running":
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
            self.park(run_id, "no archived delivery to replay — resubmit "
                              "through its campaign", avoiding=avoiding)
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
            self.park(run_id, "nothing serves these units and no registered "
                              "metal can hold them", avoiding=avoiding,
                      boot=boot)
            return {"rerouted": False, "parked": True, "run_id": run_id,
                    "boot": boot, "stopped": stopped}
        reply = await self.deliver(demands, placement, rows, frame)
        if not reply.get("accepted"):
            self.park(run_id, f"redelivery refused: {reply.get('error')}",
                      avoiding=avoiding)
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
        holding = await self.dependents(name)
        moved: dict[str, dict] = {}
        if holding and reroute:
            for rid in holding:
                moved[rid] = await self.reroute(rid, avoiding=name, park=True)
            holding = await self.dependents(name)   # what a replay could not clear
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
        must ask and hear nothing) — and then the RECONTINUE (ADR 0001, Q5d):
        the runs the reaped hosts carried are stranded (journaled `parked`,
        the queue), every metal that lost listings is KNOCKED (on a lazy
        venue the knock boots it; the reborn container registers itself at
        bring-up), and the queue is retried with `reroute` — re-placed onto
        whatever fits, the reborn metal included, and redelivered, which is
        resume. Nothing fits → the run stays parked for the next registration
        event.

        The retries ARE a listing's own restart attempt: a probe that fails,
        waits, and probes again gives a rebooting container its window —
        `recovered` is that verdict. A listing silent through every retry is
        concluded: DECARVED at its metal when the metal still answers (a
        living container frees its GB back to residual; a dead one already
        did, physically), then DELISTED with the reason journaled.

        THE TICK ALSO SWEEPS FOR IDLE METAL (ADR 0003), first: observe every
        carve-able metal, release what has been idle past its limit — so the
        probing below never chases a listing this desk has just taken down —
        and a RELEASED metal is skipped by the knock, because it is parked,
        not silent.

        AND IT REAPS LAPSED LEASES BEFORE IT PROBES ANYTHING (ADR 0008, F1).
        A probe asks "does this address answer"; a lease asks "has this
        INSTANCE spoken", which is the stronger question and the cheaper one —
        a released container that still answers its door, a metal whose name
        now resolves to a container that has not registered, a host whose
        process is wedged: all of them answer a probe and none of them
        renewed a lease. A listing whose lease lapsed is concluded without a
        retry (the retries exist to give a REBOOTING container its window, and
        a lease already gave it three of them), and a metal whose lease lapsed
        is knocked. Verdicts: per listing alive | recovered | reaped | lapsed;
        per knocked metal whether it answered; per stranded or parked run
        rerouted | parked; plus the metal released."""
        now = time.time()
        await self.observe_idle(now)
        released = await self.release_idle(now)
        listings: dict[str, str] = {}
        reaped_by_metal: dict[str, list[str]] = {}
        expired = set(self.lapsed())
        for name in sorted(name for name in expired if name in self.listings):
            listing = self.listings[name]
            await self.conclude(listing, reason="lease lapsed")
            listings[name] = "lapsed"
            reaped_by_metal.setdefault(listing.metal, []).append(name)
        for name in sorted(self.listings):
            listing = self.listings.get(name)
            if listing is None:
                continue                # concluded meanwhile by a re-registration
            if await listing.alive(self.probe_deadline_s):
                listings[name] = "alive"
                continue
            if await self.recovers(listing, probes, wait):
                listings[name] = "recovered"
                continue
            await self.conclude(listing)
            listings[name] = "reaped"
            reaped_by_metal.setdefault(listing.metal, []).append(name)
        for name in sorted(expired & set(self.metal)):
            reaped_by_metal.setdefault(name, [])
        reaped = [name for names in reaped_by_metal.values() for name in names]
        self.strand(reaped)
        knocked = {metal_name: await self.knock(metal_name)
                   for metal_name in sorted(reaped_by_metal)
                   if metal_name and metal_name not in self.released}
        runs = await self.retry_parked() if reaped else {}
        return {"listings": listings, "knocked": knocked, "runs": runs,
                "released": released}

    async def recovers(self, listing: Listing, probes: int,
                       wait: float) -> bool:
        """Retry a silent listing `probes` times, `wait` seconds apart: on a
        lazy venue the knock itself boots a stopped-but-deployed container,
        so this is its window to come back."""
        for _ in range(probes):
            if wait:
                await asyncio.sleep(wait)
            if await listing.alive(self.probe_deadline_s):
                return True
        return False

    async def conclude(self, listing: Listing, reason: str = "reaped") -> None:
        """A listing concluded dead: decarved at its metal when the metal
        answers, delisted with `reason` either way — "reaped" when a probe
        found nobody home, "lease lapsed" when nobody renewed (F1).
        Idempotent — a listing another path already delisted is left alone."""
        if listing.name not in self.listings:
            return
        if listing.metal and listing.metal in self.metal_remotes:
            try:
                await self.metal_remotes[listing.metal].decarve(listing.name)
            except Exception:
                pass            # the metal is as dead as the host: the
                                # memory freed itself when the container did
        self.delist(listing.name, reason=reason)

    async def knock(self, name: str) -> bool:
        """Boot a metal whose container is gone (Q5b) — or one this desk
        RELEASED (ADR 0003, Q4), which is the same act for the same reason:
        the venue's `boot_for` when it has one, otherwise a `describe()`
        through the plane address the desk holds — on Modal that call IS the
        boot. The address is read off the metal ROW, not off `metal_remotes`,
        because a released metal has left the carve-able set and is exactly
        what a knock is for. Off the loop, because a boot is seconds to
        minutes and the reborn container's own registration must be able to
        reach this desk meanwhile. A knock that fails is a metal that stays
        down; the queue waits for the next registration event.

        A KNOCK WITH NO WAY TO KNOCK REFUSES LOUDLY (ADR 0007, Q5): neither
        a `boot_for` nor a plane address is not a metal that failed to wake
        but a venue that never said how to wake it, and the reap → knock →
        re-register → reroute loop degrading into a silent no-op is how that
        bug hides for an hour. It is journaled and named instead."""
        address = self.metal_addresses.get(name)
        if self.boot_for is not None:
            # off the loop, because a venue's boot is a blocking spawn and the
            # reborn container's own registration must reach this desk meanwhile
            knock = asyncio.to_thread(self.boot_for, name)
        elif address and self.metal_for is not None:
            # the describe is a wire call and carries the wire's own bound: a
            # boot is seconds to minutes, and a knock that has not answered in
            # that time is a metal that stayed down
            knock = self.metal_for(address).describe(
                deadline_s=self.boot_deadline_s)
        else:
            return self.refuse_knock(
                name, "this desk has no boot_for and this metal's row has no "
                      "plane address: nothing here knows how to wake it")
        try:
            await knock
        except Exception:
            return False
        return True

    def refuse_knock(self, name: str, reason: str) -> bool:
        """A knock nobody could make, journaled and returned false (Q5). The
        loud half: a venue that supplies no way to boot its metal is a venue
        bug, and the record is where it stops being a silence."""
        self.store.append_fleet_event({
            "event": "knock-refused", "t": time.time(), "metal": name,
            "reason": reason})
        return False

    # ---- the recontinue: strand, the queue, retry --------------------------

    def park(self, run_id: str, reason: str, *, avoiding: str = "",
             boot: Sequence[Mapping] | None = None) -> dict:
        """THE PARKED STATE, WRITTEN IN ONE PLACE (ADR 0008, F6 and Q7).

        A run is parked when the fleet cannot run it right now: its host's
        lease lapsed, its host was reaped or decommissioned, or a redelivery
        was refused. The row says WHY (`reason`), what it is avoiding, WHAT IT
        WANTS (regimes, devices, GB — read off its own archived demand rows,
        so an operator can tell at a glance what metal would free it), and
        SINCE when.

        `since` is the start of the waiting, not of this row: a retry that
        parks the run again carries the FIRST park's instant forward, because
        "parked since 19:17" is the number an operator needs and a run
        re-parked every reaper tick would otherwise look freshly stuck for as
        long as it waited."""
        standing = self.parked_rows().get(run_id, {})
        now = self.now()
        row = {"event": "parked", "t": now, "run_id": run_id,
               "reason": reason, "avoiding": avoiding,
               "since": float(standing.get("since", now)),
               "wants": self.wants_of(run_id)}
        if boot:
            row["boot"] = [dict(entry) for entry in boot]
        self.store.append_fleet_event(row)
        return row

    def wants_of(self, run_id: str) -> list[dict]:
        """WHAT A PARKED RUN IS WAITING FOR, per placement unit: the regimes
        it wears, how many devices one unit spans, and the GB each of them
        must hold. Read off the run's own archived demand rows — the same
        projection a boot instruction carries, because they are the same
        question asked at two moments — and empty for a delivery from before
        the archive, which has nothing to say."""
        rows = self.placements().get(run_id, {}).get("demands")
        if not rows:
            return []
        return [wants_of_unit(unit)
                for unit in placement_units(demands_from(rows))]

    def strand(self, hosts: Sequence[str]) -> list[str]:
        """Every UNFINISHED run whose latest placement touched one of `hosts`
        is journaled `parked` — written BEFORE any reroute is attempted, so a
        desk that dies between concluding a host and moving its runs leaves
        the intent on the record and the next event retries it. A run whose
        ledger already reached its plan is not work and is left alone; a run
        already parked is not parked twice."""
        already = self.parked()
        stranded: list[str] = []
        for run_id, row in sorted(self.placements().items()):
            lost = [host for host in hosts if host in row["pools"].values()]
            if not lost or run_id in already or self.finished(run_id):
                continue
            self.park(run_id, f"host {lost[0]!r} reaped", avoiding=lost[0])
            stranded.append(run_id)
        return stranded

    def finished(self, run_id: str) -> bool:
        """Is this run still WORK? Its EXTENT, read off the store, because the
        anchor's roster died with its metal and the store is the run: the
        ledger against the train plan where a run trains, the sealed rollouts
        against the rollout plan where it only generates (ADR 0006 Part B).
        The one predicate, shared with the observer — a run with no plan on
        record is taken to be work."""
        return run_done(self.store, run_id)

    def parked(self) -> dict[str, str]:
        """THE QUEUE, read off the journal: every run whose latest disposition
        is `parked` — by a decommission with nowhere to go, a stranding, or a
        redelivery refused — mapped to the host it is avoiding ("" if none).
        A later delivered placement supersedes the park."""
        return {run_id: row.get("avoiding") or ""
                for run_id, row in self.parked_rows().items()}

    def parked_rows(self) -> dict[str, dict]:
        """THE QUEUE, WHOLE: the standing `parked` event per run — reason,
        what it is avoiding, what it wants and since when (ADR 0008, F6).
        `parked()` is this read down to its one key; the observer and the
        park writer read the rest."""
        queue: dict[str, dict] = {}
        for event in self.store.read_fleet_log():
            run_id = event.get("run_id")
            if not run_id:
                continue
            if event.get("event") == "place" and event.get("delivered") \
                    and event.get("accepted"):
                queue.pop(run_id, None)
            elif event.get("event") == "parked":
                queue[run_id] = dict(event)
        return queue

    async def retry_parked(self) -> dict[str, str]:
        """Every parked run rerouted with `park=True`: placed onto whatever
        fits now (the fleet's residual is read live), stopped wherever a
        roster still carries it, and redelivered — resume — or parked again
        with the boot instructions. Verdicts per run: rerouted | parked.

        A REAP'S RETRY AND A REGISTRATION'S RETRY RUN IN THE SAME BREATH, and
        a run must not be adopted twice — so each run is CLAIMED before it is
        rerouted and released when the reroute ends. A claim, and not a lock
        held across the whole pass: the pass is nothing but wire calls, and
        since ADR 0008 nothing in this desk waits on a container while
        holding something another verb needs. Two retries therefore proceed
        together over DIFFERENT runs and neither touches a run the other has."""
        verdicts: dict[str, str] = {}
        for run_id, avoiding in sorted(self.parked().items()):
            if not self.claim(run_id):
                continue                # another retry has this one
            try:
                reply = await self.reroute(run_id, avoiding=avoiding, park=True)
            finally:
                self.retrying.discard(run_id)
            verdicts[run_id] = ("rerouted" if reply.get("rerouted")
                                else "parked")
        return verdicts

    def claim(self, run_id: str) -> bool:
        """Take this run for a reroute, or say someone else has it. The whole
        of the mutual exclusion, and it holds nothing across the wire: the
        set is read and written between two awaits on one loop, which is
        atomic by construction."""
        if run_id in self.retrying:
            return False
        self.retrying.add(run_id)
        return True

    # ---- the idle sweep: metal nothing runs on is released ------------------

    def idle_limit(self, name: str) -> float | None:
        """The seconds of idleness `name` is released after: the limit its
        own registration declared, or this desk's default where it declared
        none. None is PINNED — never released, however long it sits."""
        return self.metal_idle_s.get(name, self.idle_s)

    async def listing_busy(self, listing: Listing,
                           seen: dict[str, int]) -> bool:
        """IS THIS LISTING WORKING? Three ways to say yes, all read off ONE
        status frame: a RUNNING tenancy, work IN FLIGHT at its arbiter right
        now, or an `admitted` counter that MOVED since the previous
        observation. The counter is what sees a PURE CLIENT — a measurement
        cron, an evaluator: admitted traffic, no tenancy — whose host would
        otherwise be released out from under its own sampling (ADR 0003,
        Q1); in-flight is what sees one long admission that spans two ticks
        and moves no counter.

        The reading is recorded in `seen` for the next tick, and a listing
        observed for the FIRST time is busy: metal is released on evidence
        of idleness, never on the absence of a reading. A silent listing
        holds nothing running — that is the reaper's business, not this
        rule's."""
        told = await listing.told(self.probe_deadline_s)
        if told is None:
            return False
        admitted = int(told.get("admitted", 0))
        seen[listing.name] = admitted
        if listing.name not in self.admitted_at:
            return True
        if admitted != self.admitted_at[listing.name]:
            return True
        if int(told.get("in_flight", 0)) > 0:
            return True
        return any(tenant.get("status") == "running"
                   for tenant in told.get("tenants", {}).values())

    async def observe_idle(self, now: float) -> None:
        """ONE TICK OF THE IDLE CLOCK. For every carve-able metal: it is IDLE
        when no listing on it is working (listing_busy — the one rule) or it
        holds no listings at all. The FIRST idle observation stamps
        `idle_since`; a busy one clears it, so the clock measures CONTINUOUS
        idleness and never sums two quiet spells across a busy one.

        Every listing is probed once, and only the listings observed here are
        remembered: a delisted host's counter leaves with it."""
        seen: dict[str, int] = {}
        for name in sorted(self.metal_remotes):
            busy = [listing for listing in self.listings.values()
                    if listing.metal == name
                    and await self.listing_busy(listing, seen)]
            if busy:
                self.idle_since.pop(name, None)
            else:
                self.idle_since.setdefault(name, now)
        self.admitted_at = seen

    async def release_idle(self, now: float) -> list[str]:
        """Every metal idle past ITS limit, released — the sweep's second
        half, and the only place the clock is read. A pinned metal (limit
        None) is never due, however long its clock has run."""
        due = [name for name, since in sorted(self.idle_since.items())
               if self.idle_limit(name) is not None
               and now - since >= self.idle_limit(name)]
        for name in due:
            # UNGUARDED BY RULING (ADR 0007, Q6): the guard below is about a
            # DOOR tearing down metal under another experiment. The idle rule
            # is the desk's own clock and its evidence is stronger than the
            # guard's — it released this metal because nothing has been busy
            # on it for its whole limit, which is ADR 0003's promise and must
            # not be weakened into "unless a journal row still says running".
            await self.release(name, reason="released: idle", force=True)
        return due

    async def release(self, name: str, reason: str = "released",
                      force: bool = False) -> dict:
        """THE ACQUIRE RUNG INVERTED, desk-issued (ADR 0003). The departure
        is journaled FIRST — the intent on the record, as `strand` writes it
        — then every listing on the metal is delisted with `reason`, and the
        metal itself is told `release`: its residents come down the ladder
        and its SHIFT ENDS, so the venue reclaims the container. The row
        stays as INVENTORY (what the fleet owns) and leaves `metal_remotes`
        (what the fleet may carve), so `status()` says released and the next
        placement that needs it KNOCKS it back (provision_unit).

        A silent metal is as released as it gets — the container is already
        gone, which is the goal state — so the wire's refusal is not this
        verb's problem. Idempotent: releasing already-released metal
        re-journals and changes nothing.

        GUARDED LIKE DECOMMISSION (ADR 0007, Q6): under ONE desk a venue door
        that tears down the metal it acquired would take every other
        experiment on that metal with it, so running work routing through ANY
        listing on this metal is NAMED and the release refused. `force` says
        you mean it, and belongs at the desk's own operator door — a venue's
        door never sends it."""
        if name not in self.metal:
            raise DeskError(f"metal {name!r} is not registered with this desk")
        holding = await self.metal_dependents(name)
        if holding and not force:
            return {"released": False, "metal": name, "running": holding,
                    "error": f"metal {name!r} carries or serves running work "
                             f"({', '.join(holding)}) — release with force to "
                             f"hand it back anyway"}
        listings = sorted(host for host, listing in self.listings.items()
                          if listing.metal == name)
        self.store.append_fleet_event({
            "event": "release", "t": time.time(), "metal": name,
            "idle_s": self.idle_limit(name), "listings": listings})
        for host in listings:
            self.delist(host, reason=reason)
        remote = self.metal_remotes.pop(name, None)
        told = False
        if remote is not None:
            try:
                await remote.release()
                told = True
            except Exception:
                pass            # the container is gone: released, physically
        terminated = await self.terminate(name)
        self.released.add(name)
        self.idle_since.pop(name, None)
        return {"released": True, "metal": name, "listings": listings,
                "told": told, "terminated": terminated, "running": holding}

    async def terminate(self, name: str) -> bool:
        """THE CONTAINER ENDS. `release` told the metal to hand everything
        back, and a metal that heard it stops taking inputs — but a container
        that stops taking inputs is still a container, billed until the
        venue's scaledown, and one whose residents wedged never hears the
        verb at all (found live, 2026-09-05: released metal standing for an
        hour). So the desk ends it itself, by the container id the metal
        registered with, through the hand the deploy gave it. Best effort by
        design: no id (a hand-built metal) or no hand (a test desk) is False,
        and a refusal is printed, never raised — the release above already
        stands on the journal."""
        container = self.metal_containers.get(name)
        if container is None or self.terminate_for is None:
            return False
        try:
            return bool(await self.terminate_for(container))
        except Exception as refused:
            print(f"[desk] terminate {name} ({container}): {refused}",
                  flush=True)
            return False

    async def reacquire(self, name: str) -> bool:
        """A RELEASED METAL BROUGHT BACK, with no human in it (ADR 0003, Q4):
        the human act was the deploy, and this metal is deployed, owned and
        free. The KNOCK boots its container, whose bring-up registers it
        again (ADR 0001, Q5a) — and where that registration has not landed
        by the time the knock returns, the desk registers the row itself
        from the facts and address it never forgot, so the carve can proceed
        in the same breath; the container's own announce then supersedes it
        with MEASURED facts. A metal that does not answer stays released.

        The recipe is NOT re-proposed here: since ADR 0007 it is the desk's
        own declaration, journaled and still in this desk's table, and the
        carve that follows reads it there."""
        if not await self.knock(name):
            return False
        if name not in self.metal_remotes:
            self.register_metal(
                self.metal[name], self.metal_addresses.get(name),
                idle_s=self.metal_idle_s.get(name, DESK_DEFAULT))
        return name in self.metal_remotes

    def status(self) -> dict:
        """The desk's inventory, no wire calls: what is listed and what it
        wears, the registered metal, and every row's LEASE (ADR 0008, F1) —
        the epoch it is, when it was last heard, how long a silence is still
        belief, and whether it is believed right now. Occupancy is asked per
        placement and never cached; the residual shown here is the last
        HEARTBEAT's, because placement's own read is live (Q3, amended) and
        this is what the row says between placements."""
        now = self.now()
        return {"listings": {name: {
            "address": listing.address, "solo": listing.solo,
            "regimes": [r.name for r in listing.regimes],
            "partition": listing.partition, "metal": listing.metal,
            **self.lease_row(name, now)}
            for name, listing in sorted(self.listings.items())},
            "metal": {name: {"gpu": m.gpu, "devices": m.devices,
                             "vram_gb": m.vram_gb,
                             "address": self.metal_addresses.get(name),
                             "plane": name in self.metal_remotes,
                             "released": name in self.released,
                             "idle_s": self.idle_limit(name),
                             "residual": self.residuals.get(name),
                             "builds": self.recipe_row(name),
                             **self.lease_row(name, now)}
                      for name, m in sorted(self.metal.items())}}

    def lease_row(self, name: str, now: float) -> dict:
        """One row's lease as `status()` shows it: the three fields the Lease
        carries plus the verdict they add up to. A name this desk holds no
        lease for shows nulls and `live: true` — nothing has promised to
        heartbeat for it, so nothing about it has lapsed (`leased`'s rule,
        said in the row)."""
        lease = self.leases.get(name)
        if lease is None:
            return {"epoch": "", "heard_t": None, "lease_s": None,
                    "live": True}
        return {**lease.row(), "live": lease.live(now)}

    def recipe_row(self, metal: str) -> dict | None:
        """This metal's declared recipe as a WIRE ROW, or None where nothing
        has declared for it yet — what `status()` shows an operator deciding
        whether a bare metal still needs its `recipe` door called."""
        recipe = self.recipe_for(metal)
        return recipe.row() if recipe is not None else None

    # ---- the Transport surface (HostService's contract, fleet-addressed) ----

    async def serve(self, verb: str, payload: dict) -> dict:
        if verb == "submit":
            return await self.submit(payload["demands"], payload["frame"],
                                     solo=bool(payload.get("solo", False)))
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
                metal=payload.get("metal", ""),
                epoch=payload.get("epoch", ""))
            return {"listed": payload["host"],
                    "epoch": self.epoch_of(payload["host"]),
                    "lease_s": self.lease_s,
                    "heartbeat_s": self.heartbeat_s}
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
            # included — after this the desk can deduce and command against
            # it; a re-registration reaps that metal's corpses, and EVERY
            # registration retries the parked queue (the reborn metal's own
            # registration is the trigger that recontinues its runs)
            known = self.register_metal(
                Metal(name=payload["name"], gpu=payload.get("gpu", "L4"),
                      devices=int(payload.get("devices", 1)),
                      vram_gb=float(payload.get("vram_gb", 24.0))),
                address=payload.get("address"),
                builds=builds_proposed(payload),
                container=payload.get("container"),
                idle_s=(payload["idle_s"] if "idle_s" in payload
                        else DESK_DEFAULT),
                epoch=payload.get("epoch", ""))
            reaped = (await self.reconcile_metal(payload["name"])
                      if known else [])
            retried = await self.retry_parked()
            # THE LEASE CONSTANTS TRAVEL BACK (ADR 0008, Q1): the desk owns
            # them, so the container learns its own cadence from the reply
            # rather than from a constant of its own that could drift
            return {"registered": payload["name"], "reaped": reaped,
                    "retried": retried, "epoch": self.epoch_of(payload["name"]),
                    "lease_s": self.lease_s, "heartbeat_s": self.heartbeat_s}
        if verb == "heartbeat":
            # THE RENEWAL (F1). One frame per metal and per host on it, every
            # `heartbeat_s`; the residual rides along for the row.
            return await self.heartbeat(payload["name"], payload.get("epoch", ""),
                                        payload.get("residual"))
        if verb == "put_plan":
            return self.put_plan(_unb64(payload["bytes"]))
        if verb == "read_cas":
            return {"uri": payload["uri"],
                    "bytes": _b64(self.store.cas_get(payload["uri"]))}
        if verb == "recipe":
            # WHAT A METAL BUILDS, declared through the desk's own door (ADR
            # 0007, Q4) — and through the desk, because the fleet journal has
            # ONE writer and this is one of its events (I10)
            self.recipe(payload["metal"], Builds.from_row(payload["builds"]))
            return {"metal": payload["metal"],
                    "builds": self.recipe_row(payload["metal"])}
        if verb == "pulse":
            # a fan of wire probes rides the ADMITTED path like liveness (ADR
            # 0008, F3): bounded and cancellable; the probes join off the loop
            return await asyncio.to_thread(self.pulse)
        if verb == "liveness":
            return await self.liveness()
        if verb == "reap":
            return await self.reap(probes=int(payload.get("probes", 3)),
                                   wait=float(payload.get("wait", 0.0)))
        if verb == "release":
            # the same verb the idle sweep issues, by hand: an operator who
            # knows a metal is done need not wait out its clock. `force` is
            # the operator's alone (Q6) — a venue door leaves it unsaid and
            # is refused when another experiment is still running there
            return await self.release(
                payload["metal"], reason=payload.get("reason", "released"),
                force=bool(payload.get("force", False)))
        raise ValueError(f"unknown fleet verb {verb!r}")

    def answer(self, verb: str, payload: dict) -> dict:
        if verb == "status":
            return self.status()
        if verb == "placements":
            return {"placements": self.placements()}
        raise ValueError(f"unknown admission-free fleet verb {verb!r}")

    async def liveness(self) -> dict:
        """Every listing PROBED, now, CONCURRENTLY and each under the probe
        deadline: {host: answered}. The desk is the one place that can ask a
        container instead of presuming from a journal, and an observer given
        a desk shows probes where it has them.

        On the ADMITTED path since ADR 0008: a verb that fans out over the
        wire belongs where it can be bounded and cancelled, not on the door
        that answers off memory alone."""
        names = sorted(self.listings)
        answers = await asyncio.gather(
            *(self.listings[name].alive(self.probe_deadline_s)
              for name in names))
        return dict(zip(names, answers))

    def pulse(self) -> dict:
        """Every listing PROBED, now, with what it CARRIES: {host: {"alive":
        answered, "running": [run_id, ...]}} — ONE status frame per listing,
        the roster read off the same reply the liveness probe already makes.

        The desk is the one place that can ask a container instead of
        presuming from a journal. The roster is what lets an observer tell a
        run that IS on a live host from an attach a dead generation of the
        same host name left behind: a carve name recurs per container (the
        counter is the container's), so its journal outlives every
        generation, and a journal alone reads a crashed generation's open
        attach as "running" for as long as any later generation keeps the
        name alive (found on the yu-masala volume: a 09-01 attach beside
        09-04's arms). Asked from the answer thread, never a loop."""
        def probe(name: str, listing: Listing) -> tuple[str, dict]:
            try:
                told = listing.host.status()
            except WedgeError:
                raise
            except Exception:
                return name, {"alive": False, "running": []}
            return name, {"alive": True, "running": sorted(
                rid for rid, row in told.get("tenants", {}).items()
                if row.get("status") == "running")}

        # IN PARALLEL, one thread per listing: a pulse costs the slowest
        # probe, not their sum — an unreachable metal's two listings at the
        # wire's deadline each were measured at over 30 s in series
        listings = sorted(self.listings.items())
        if not listings:
            return {}
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(8, len(listings))) as probes:
            return dict(probes.map(lambda item: probe(*item), listings))



def covers(listing: Listing, demand: Demand, recipe: Builds | None) -> bool:
    """THE JOIN RULE, the only copy, in two halves: this listing's regimes
    must MATCH the demand's capability, and the metal it lives on must have
    been built to SERVE what the demand names (ADR 0007, Q4a)."""
    return (matches_capability(listing.regimes, demand)
            and recipe_serves(recipe, demand))


def matches_capability(regimes: Sequence[Regime], demand: Demand) -> bool:
    """The join rule's first half: coverage is capability equality."""
    return any(regime.capability == demand.capability
               and regime.base == demand.base
               and regime.shape == demand.shape
               for regime in regimes)


def recipe_serves(recipe: Builds | None, demand: Demand) -> bool:
    """The join rule's second half (ADR 0007, Q4a): an INFERENCE demand may
    land on a metal only if that metal's engine recipe SERVES every adapter
    type the demand names. The strings were read off the spec's bank by the
    campaign layer and are compared here, never interpreted — the desk stays
    workload-blind (#69), and the refusal moves off the host's Phase 0 and
    onto the join, where a placement can still go somewhere else.

    UNKNOWN IS NOT A REFUSAL, three ways: a demand naming no adapter type (a
    pure client's) passes, a TRAINING demand passes (a learner builds any
    adapter type — LearnerBuild declares no `serves`), and a listing whose
    metal declared no engine recipe passes, because a rule refuses on
    evidence or not at all."""
    if demand.capability != "inference" or not demand.adapter_types:
        return True
    if recipe is None or not isinstance(recipe.engine, EngineBuild):
        return True                 # a fake engine build declares no `serves`
    return set(demand.adapter_types) <= set(serves_of(recipe))


def serves_of(recipe: Builds) -> tuple[str, ...]:
    """The adapter types this recipe's ENGINE was built to serve. A fakes
    recipe declares none, which is why the rule above passes over it."""
    return (recipe.engine.serves if isinstance(recipe.engine, EngineBuild)
            else ())


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

    def __init__(self, metal: Metal, *, store: Store,
                 address_of: Callable[[str], str],
                 builds: Builds | None = None,
                 schema_for: Callable[[str], SiteSchema] | None = None,
                 transport_for: Callable[[str], Transport] | None = None,
                 spawn: Callable[[ResidentBirth], Resident] = Resident.spawn,
                 epoch: str | None = None,
                 ) -> None:
        self.metal = metal
        # THE EPOCH, MINTED AT BRING-UP (ADR 0008, F2): this container's boot
        # identity, worn by every host it carves and every resident under
        # them, said at registration, and refused at this door when a frame
        # names another. It is minted HERE and not handed in because minting
        # it is what bringing a container up MEANS — a venue that supplied one
        # could supply the same one twice, which is the hazard exactly.
        self.epoch = epoch or mint_epoch()
        self.store = store
        # THE METAL BOOTS BARE (ADR 0007, Q4). The recipe is everything a
        # partition cannot tell you — and it is a DECLARATION, which is the
        # desk's to make: the desk's journaled row was already the canon
        # every carve carried (ADR 0001, Q5c) and `adopt_recipe` already
        # replaced whatever the container booted with. So None is the normal
        # state, a carve delivers the recipe, and a carve that delivers none
        # to a bare metal is refused by name rather than half-built. A venue
        # may still pass one, and it is a PROPOSAL: its first word, which the
        # desk journals and the next carve overwrites.
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
        # THE SHIFT, as a latch: set by `release`, awaited by whatever holds
        # this container open (ADR 0003, Q3). An Event binds to no loop until
        # it is first awaited, so a venue constructs this service anywhere.
        self.released = asyncio.Event()

    @classmethod
    def measure(cls, name: str) -> Metal:
        """The registration row READ OFF THE DEVICE, never typed (ADR 0001,
        Q6): the card's own name, how many devices this container sees, and
        one device's VRAM in GB (GiB — the unit torch's total_memory and a
        model's weights are counted in). torch is imported here, lazily, and
        only a real venue calls this; a machine with no CUDA device refuses
        by name rather than guessing a card. `Metal` stays a plain record so
        the fakes suite constructs one without torch."""
        try:
            import torch
        except ImportError as absent:
            raise DeskError(
                f"cannot measure metal {name!r}: torch is not installed, and a "
                f"card is measured, never declared") from absent
        if not torch.cuda.is_available():
            raise DeskError(
                f"cannot measure metal {name!r}: no CUDA device is visible to "
                f"this process — a card is measured, never declared")
        card = torch.cuda.get_device_properties(0)
        return Metal(name=name, gpu=card.name, devices=torch.cuda.device_count(),
                     vram_gb=round(card.total_memory / 2 ** 30, 2))

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
        self.route(address, HostService(host))

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
        the desk's `builds` row is built from it (Q5c) — and a BARE metal
        handed a request that carries none is refused here by name (ADR 0007,
        Q4), because a container that was never told what to serve cannot
        guess, and half-building is the one outcome worse than refusing."""
        if not request.get("builds") and self.builds is None:
            return {"carved": False, "residual": self.residual(),
                    "error": f"metal {self.metal.name!r} is BARE: no recipe "
                             f"was declared for it at the desk and this carve "
                             f"carries none — declare one (desk.recipe) and "
                             f"the next carve will carry it"}
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
                                           devices, gb,
                                           bool(request.get("solo", False)))
            # booked -> built in one tick: no await between the thread's
            # return and these lines, so residual never blinks
            self.hosts[name] = host
            address = self.address_of(name)
            self.addresses[name] = address
            self.route(address, HostService(host))
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
                "solo": host.solo, "epoch": self.epoch,
                "partition": host.partition.row(),
                "regimes": [{"name": r.name, "capability": r.capability,
                             "base": r.base, "shape": r.shape}
                            for r in regimes]}

    async def measure_the_run(self, payload: Mapping) -> dict:
        """ONE MEASURING PASS, RUN WHERE THE POOL IS (ADR 0008, F6).

        Named apart from `measure`, which is this class's OTHER measurement —
        the card, read off the device at bring-up (ADR 0001, Q6). One
        container, two things worth measuring, and neither is declared.

        A measurement needs an engine and a store, and this container has
        both — so it runs here rather than on an on-demand CPU function that
        for an hour on 2026-09-04 Modal never scheduled, and rather than in a
        driver that has no mount. The frame carries the measurement's own
        manifest and the addresses it names; nothing venue-shaped reaches
        this class, exactly as nothing spec-shaped reaches the desk.

        The pool is reached through the SERVING HOST'S DOOR — a LocalTransport
        onto its `HostService`, which is the in-process rule (#77) — so every
        request is admitted at that host's own arbiter and counted into its
        own meter, precisely as a client's would be. `pools` names the OTHER
        pools the measurement's pipeline addresses (a teacher, a judge); each
        resolves the same way, and a name this metal cannot serve is refused
        BY NAME rather than measured against the wrong engine."""
        from rlstack.runner.measure import Measurement, measure_run
        from rlstack.data.tasks.base import load_tasks

        row = dict(payload["measurement"])
        measurement = Measurement(
            name=row["name"], env=row["env"],
            task_ids=tuple(row["task_ids"]), samples=int(row["samples"]),
            every=int(row["every"]), post=tuple(row["post"]),
            seed=int(row["seed"]),
            temperature=float(row.get("temperature", 1.0)),
            max_tokens=int(row.get("max_tokens", 512)))
        main = self.pool_for(payload["base"], int(payload["tp"]))
        others = {name: self.pool_for(payload["base"], int(payload["tp"]))
                  for name in payload.get("pools", ())}
        tasks = {task.id: task
                 for task in load_tasks(self.store, payload["tasks"])}
        fresh = await measure_run(self.store, payload["run_id"], measurement,
                                  main, tasks, pools=others)
        return {"measured": list(fresh), "run_id": payload["run_id"],
                "points": self.store.read_measurements(
                    payload["run_id"]).get(measurement.name, {})
                    .get("points", [])}

    def pool_for(self, base: str | None, tp: int) -> RemotePool:
        """The engine on THIS metal serving (base, tp), reached through its
        host's own door so admission and traffic stay that host's. Refused by
        name where nothing here serves it — a measurement placed onto the
        wrong metal is a placement bug, and it says so."""
        for name in sorted(self.hosts):
            host = self.hosts[name]
            if host.engine_for(base, tp) is not None:
                return RemotePool(LocalTransport(HostService(host),
                                                 self.epoch),
                                  base=base, tp=tp)
        raise DeskError(
            f"metal {self.metal.name!r} serves no ({base!r}, tp={tp}): its "
            f"hosts are {sorted(self.hosts)} — the measurement was placed on "
            f"the wrong metal")

    def adopt_recipe(self, builds: Builds) -> None:
        """The desk's recipe row is CANON (Q5c): a carve request that carries
        one replaces this metal's own, so its deploy constants are only its
        FIRST declaration and `describe()` reports what it last built from.
        A redeploy's new constants reach the desk through re-registration,
        which updates the row that rides the next carve."""
        self.builds = builds

    def build(self, name: str, regimes: tuple[Regime, ...],
              devices: tuple[int, ...], gb: float, solo: bool = False) -> Host:
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
                    store=self.store.address(), vram_gb=gb,
                    epoch=self.epoch)))
            engines = [RemotePool(r.transport, base=r.hello["base"],
                                  tp=int(r.hello["tp"]))
                       for r in residents if r.regime.capability == "inference"]
            learners = [RemoteLearner(r.transport, fsdp=int(r.hello["fsdp"]))
                        for r in residents if r.regime.capability == "training"]
            return Host(name, engines=tuple(engines),
                        learner=learners[0] if learners else None,
                        store=self.store, partition=partition, regimes=regimes,
                        solo=solo, schema_for=self.schema_for,
                        transport_for=self.transport_for, residents=residents,
                        epoch=self.epoch)
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
        self.unroute(address)
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

    def release(self) -> dict:
        """THE DESK'S RELEASE, EXECUTED HERE (ADR 0003): every resident down
        the ladder, the books emptied — and THE SHIFT ENDED, so whatever
        holds this container open returns and the venue reclaims it. The
        desk decided; this side only obeys and writes nothing (the fleet
        journal has one writer, as ever).

        Idempotent: a bare metal, or one already released, answers released
        just the same — released is a GOAL STATE, not an event, which is what
        lets the desk retry a release it is unsure landed."""
        teardowns = self.shutdown()
        self.hosts.clear()
        self.addresses.clear()
        for address in list(self.services):
            self.unroute(address)
        self.pending.clear()
        self.released.set()
        return {"released": True, "metal": self.metal.name,
                "teardown": [t.line() for t in teardowns if not t.graceful]}

    async def until_released(self) -> None:
        """THE SHIFT, as a wait: returns when the desk releases this metal.
        The venue's keepalive awaits this instead of sleeping forever, and
        the metal's duties loop watches the same latch — so the container is
        reclaimed as a consequence of the DESK's decision, never of the
        venue's own timer (ADR 0003, Q3: the venue's scaledown is the
        backstop, set no shorter than the desk's idle limit)."""
        await self.released.wait()

    # ---- routing and the Transport surface ----------------------------------

    def route(self, address: str, service: HostService) -> None:
        """A host on this metal becomes REACHABLE, two ways at once.

        In this container's router, which is how a venue's door forwards an
        addressed frame; and on the IN-PROCESS switchboard, which is how
        anything inside this container reaching that address gets a
        LocalTransport instead of a call to the container it is already in
        (#77 — an hour of silence on the venue, because a host adopts on this
        loop and asks reachability through the transport's SYNC verb). That
        rule used to be a closure copied into every venue file; publishing it
        here is what lets `transport_for` be the one factory everywhere (ADR
        0007, Q3).

        The switchboard is keyed by the address's ROUTE, without its epoch
        (ADR 0008, F2): a container answers at its name whatever life it is
        on, and the refusal belongs at the door rather than in the dial — so
        a frame minted for a dead epoch REACHES this host and is refused BY
        NAME, instead of failing as an address nothing serves."""
        self.services[address] = service
        serve_in_process(without_epoch(address), service)

    def unroute(self, address: str) -> None:
        """Nothing answers there any more: a decarve, a release, a teardown.
        The switchboard entry goes with it, so a stale address never routes
        to a host that has come down."""
        self.services.pop(address, None)
        stop_serving_in_process(without_epoch(address))

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

    def service_for_host(self, name: str) -> HostService:
        """The venue's router BY HOST NAME — what a door actually holds.

        A frame arrives naming a host, not an address: the `#host` fragment is
        the routing key and the epoch travels in the payload (F2). Looking the
        host up in this metal's own address book, rather than re-minting the
        address from the venue's format, is what keeps the door out of the
        address grammar's business — and what stops a container whose epoch
        turned over from reconstructing an address that no longer exists."""
        address = self.addresses.get(name)
        if address is None:
            raise DeskError(
                f"no host named {name!r} on metal {self.metal.name!r} "
                f"(decarved, or never carved); carrying "
                f"{sorted(self.addresses)}")
        return self.service_for(address)

    def describe(self) -> dict:
        """The registration row plus the books — what a phone-home ships and
        what an observer renders. `builds` is null on a metal that is still
        BARE, which is what a container that has never been carved on looks
        like since ADR 0007."""
        return {"name": self.metal.name, "gpu": self.metal.gpu,
                "devices": self.metal.devices, "vram_gb": self.metal.vram_gb,
                "epoch": self.epoch, "residual": self.residual(),
                "builds": None if self.builds is None else self.builds.row(),
                "hosts": {name: {"address": self.addresses[name],
                                 "partition": host.partition.row(),
                                 "residents": [r.row() for r in host.residents]}
                          for name, host in sorted(self.hosts.items())}}

    async def serve(self, verb: str, payload: dict) -> dict:
        check_epoch(payload, self.epoch, f"metal {self.metal.name!r}")
        if verb == "measure":
            return await self.measure_the_run(payload)
        if verb == "carve":
            return await self.carve(payload)
        if verb == "decarve":
            return self.decarve(payload["host"])
        if verb == "release":
            # the third metal command (ADR 0003): carve's and decarve's
            # wholesale cousin — every host down, and the shift with them
            return self.release()
        raise ValueError(f"unknown metal verb {verb!r}")

    def answer(self, verb: str, payload: dict) -> dict:
        check_epoch(payload, self.epoch, f"metal {self.metal.name!r}")
        if verb == "residual":
            return {"residual": self.residual()}
        if verb == "describe":
            return self.describe()
        raise ValueError(f"unknown admission-free metal verb {verb!r}")
