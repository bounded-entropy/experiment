"""The Pool: every replica of one pool name, as the one thing a run routes to.

A pool may declare `replicas: N` (ADR 0022) and the desk places N hosts for
it. The POOL, not the run, is responsible for them, and this module is the
whole of that responsibility:

    ROUTING    every request carries an AFFINITY KEY (the episode id unless
               the environment names its own). A key the pool has seen goes
               back to the replica that holds it, so that replica's prefix
               cache is reused; a new key goes to the least-loaded live
               replica. A key MOVES only when its replica dies, and a revived
               replica never takes its old keys back. No load rebalancing.
    HEALTH     a replica is found dead ON CONTACT — a request or a delivery
               that meets a wire death. Its in-flight request is retried on a
               survivor (the same seed, so a fake answers byte-identically),
               its keys are served by the survivors, and its revival starts
               at once.
    DELIVERY   a bundle reaches every LIVE replica before the Trainer's
               commit, and a revived replica is brought up to every bundle
               the pool is serving — and the current wave's libraries —
               before it is readmitted. So no key is ever served by a replica
               holding an older bundle than the committed one.

A replica count of 1 is the pool every run had before ADR 0022: one engine,
every key on it, and a death with no reviver raises exactly as before.

Everything a pool mutates is mutated ON THE EVENT LOOP; every engine verb it
issues is synchronous by protocol and asked from a thread, so a pool on
another container never parks the loop that is routing to it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from rlstack.policy.compile import Bundle
from rlstack.runner.interfaces import Engine
from rlstack.runner.meters import HostJournal
from rlstack.runner.remote import HostGone, Unreachable, WrongEpoch

T = TypeVar("T")

REPLICA_SEPARATOR = "#"
"""How a replica's ROUTE KEY is spelled: `main` is replica 0 of pool `main`
— the key every run before ADR 0022 already used — and `main#2` its third.
One spelling for the desk's placement rows, a delivery's routes and the
host door that resolves them."""

PINS_HELD = 4
"""How many bundles a pool remembers serving: what a revived replica is
brought up to. The newest delivery, plus the few older versions a Scorer may
still pin (it scores a wave under the bundle that sampled it)."""

REVIVE_ATTEMPTS = 8
REVIVE_BACKOFF_S = 5.0
REVIVE_BACKOFF_MAX_S = 30.0
"""A revival that fails is tried again, apart — 5 s, 10 s, ... capped at 30 s,
about three minutes in all: the dead replica's own GB is freed only once its
metal has decarved it, which can trail the death the pool noticed (the B300
drill's first revival found no room for exactly that reason), and a metal
mid-reboot answers nothing for a while."""


def replica_key(pool: str, replica: int) -> str:
    """The route key of one replica of `pool`."""
    return pool if replica == 0 else f"{pool}{REPLICA_SEPARATOR}{replica}"


def split_replica_key(key: str) -> tuple[str, int]:
    """A route key back into (pool, replica) — replica_key's inverse."""
    pool, sep, index = key.rpartition(REPLICA_SEPARATOR)
    if not sep or not index.isdigit():
        return key, 0
    return pool, int(index)


def is_replica_death(error: BaseException) -> bool:
    """IS THIS FAILURE THE REPLICA'S, NOT THE REQUEST'S? A wire that stopped
    answering (Unreachable), a container wearing another epoch (WrongEpoch),
    or a host that was decarved under its address (HostGone): each says the
    engine behind the proxy is gone, so the request is retried elsewhere.
    Anything else — a refused library, a bundle never registered — is the
    request's own failure and is raised as it always was."""
    return isinstance(error, (Unreachable, WrongEpoch, HostGone))


class Reviver(Protocol):
    """How a pool gets a dead replica back: a fresh host serving the same
    capability, placed through the desk (possibly on another metal), handed
    back as an Engine proxy. The pool brings it up and readmits it."""

    async def recarve(self, pool: str, replica: int) -> Engine: ...


@dataclass(frozen=True)
class DeskReviver:
    """THE REVIVER ON THE FLEET: the anchor host asks its desk to re-place one
    replica (`place-replica`) and wraps the address it answers into the same
    RemotePool every routed replica is. The demand row is the replica's own,
    read off the spec by the campaign layer when the run was adopted, so the
    desk stays workload-blind and the new host is sized exactly as the dead
    one was declared."""

    desk: object                    # RemoteDesk: the anchor host's door to its desk
    run_id: str
    rows: Mapping[int, Mapping]     # replica index -> its demand row
    transport_for: Callable[[str], object]
    base: str | None
    tp: int

    async def recarve(self, pool: str, replica: int) -> Engine:
        from rlstack.runner.remote import RemotePool

        told = await self.desk.place_replica(self.run_id, self.rows[replica])
        if not told.get("placed"):
            raise Unreachable(
                f"the desk could not re-place replica {replica} of pool "
                f"{pool!r}: {told.get('error') or told.get('boot')}")
        return RemotePool(self.transport_for(told["address"]),
                          base=self.base, tp=self.tp)


Restore = Callable[[Engine, Bundle], None]
"""The run's rule for making ONE engine able to serve ONE bundle: for a pool
that serves the policy, restore it from the store on a miss
(`restore_bundle_on`); for any other pool, add its base bundle. Synchronous —
the pool asks it from a thread."""


def add_if_missing(engine: Engine, bundle: Bundle) -> None:
    """The Restore of a pool that serves only its own base, and of a bundle
    that carries its payloads: ask, and add on a miss."""
    if not engine.knows_bundle(bundle.bundle_id):
        engine.add_bundle(bundle)


@dataclass(frozen=True)
class Replicas:
    """A pool's hosts as placement handed them to a run: the engine proxy of
    each replica in replica order, and the reviver that can re-place one
    (None: a dead replica stays dead, and the survivors serve its keys).
    What a host hands `run_experiment` for a pool with more than one host; a
    bare Engine is the one-replica case."""

    engines: tuple[Engine, ...]
    reviver: Reviver | None = None


@dataclass
class Replica:
    """One host of a pool, as the pool tracks it: its engine, whether it is
    live, the work on it now, and the window's counts (`window` drains them).
    `keys` is a gauge, read off the key table."""

    index: int
    engine: Engine
    live: bool = True
    in_flight: int = 0
    # the window, drained by `Pool.window`
    requests: int = 0
    tokens: int = 0
    new_keys: int = 0
    seen_keys: int = 0
    peak_in_flight: int = 0
    # monotone
    deaths: int = 0
    revivals: int = 0


@dataclass
class Pool:
    """THE REPLICAS OF ONE POOL NAME: routes by affinity, finds the dead,
    revives them, delivers bundles to every live one. See the module
    docstring for the rules; each has one method below.

    `restore` is the run's rule for one engine and one bundle (Restore);
    `journal` is where deaths and revivals are journaled the moment they
    happen, with their timing (None: a run on no host keeps them in `events`
    only)."""

    name: str
    replicas: list[Replica]
    reviver: Reviver | None = None
    restore: Restore = add_if_missing
    journal: HostJournal | None = None
    run_id: str = ""
    keys: dict[str, int] = field(default_factory=dict)
    moves: int = 0
    # the bundles this pool serves, oldest first (PINS_HELD), and the
    # current wave's library payloads: what a revived replica is brought up to
    pins: dict[str, Bundle] = field(default_factory=dict)
    libraries: dict[str, bytes] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    # the death that took the last replica to fall: what a request raises
    # when no replica is live and none is being revived
    last_death: BaseException | None = None
    _reviving: dict[int, asyncio.Task] = field(default_factory=dict)
    _window_started: float = field(default_factory=time.time)

    @classmethod
    def of(cls, name: str, handed: Engine | Replicas | Pool, *,
           restore: Restore = add_if_missing,
           journal: HostJournal | None = None, run_id: str = "") -> "Pool":
        """The pool a run routes to, from what its host handed: a Pool as it
        is, Replicas as one replica each, a bare Engine as the one-replica
        pool every run was before ADR 0022."""
        if isinstance(handed, Pool):
            return handed
        engines = handed.engines if isinstance(handed, Replicas) else (handed,)
        reviver = handed.reviver if isinstance(handed, Replicas) else None
        return cls(name, [Replica(i, engine) for i, engine in enumerate(engines)],
                   reviver=reviver, restore=restore, journal=journal,
                   run_id=run_id)

    # ---- what a run reads ---------------------------------------------------

    def engines(self) -> tuple[Engine, ...]:
        """Every replica's engine, live or not, in replica order."""
        return tuple(replica.engine for replica in self.replicas)

    def live(self) -> list[Replica]:
        """The replicas that may take a request right now."""
        return [replica for replica in self.replicas if replica.live]

    @property
    def base(self) -> str | None:
        """What the pool serves — any replica's build fact; they agree."""
        return self.replicas[0].engine.base

    @property
    def tp(self) -> int:
        return self.replicas[0].engine.tp

    # ---- routing: one rule per method ---------------------------------------

    def route(self, affinity: str) -> Replica:
        """THE ROUTER, per request. A seen key returns to its replica while
        that replica lives; a new key — or one whose replica died — goes to
        the least-loaded live replica, and the second case is a MOVE. Never
        rebalanced by load (ADR 0022, Q2)."""
        held = self.keys.get(affinity)
        if held is not None and self.replicas[held].live:
            self.replicas[held].seen_keys += 1
            return self.replicas[held]
        chosen = self.least_loaded()
        if held is not None:
            self.moves += 1
        else:
            chosen.new_keys += 1
        self.keys[affinity] = chosen.index
        return chosen

    def least_loaded(self) -> Replica:
        """Fewest requests in flight, then fewest keys held, then the lowest
        index — so with the deterministic fakes a route is reproducible."""
        live = self.live()
        if not live:
            raise LookupError(f"pool {self.name!r} has no live replica")
        held = self.keys_held()
        return min(live, key=lambda r: (r.in_flight, held.get(r.index, 0), r.index))

    def keys_held(self) -> dict[int, int]:
        """How many keys the table assigns to each replica."""
        counts: dict[int, int] = {}
        for index in self.keys.values():
            counts[index] = counts.get(index, 0) + 1
        return counts

    async def serve(self, affinity: str,
                    request: Callable[[Engine], Awaitable[T]]) -> tuple[T, int]:
        """ONE WHOLE REQUEST, on the replica its key routes to — retried on a
        survivor when that replica turns out to be dead. `request` runs the
        complete call (a whole token stream, never part of one), so a retry
        never duplicates a token. Answers the result and the replica index
        that produced it. With no live replica, it waits for a revival in
        progress; with none in progress, the death is raised — the wire death
        it always was, which the desk parks and retries."""
        while True:
            if not self.live():
                await self.await_a_revival()
                continue
            replica = self.route(affinity)
            replica.in_flight += 1
            replica.requests += 1
            replica.peak_in_flight = max(replica.peak_in_flight, replica.in_flight)
            try:
                return await request(replica.engine), replica.index
            except Exception as error:
                if not is_replica_death(error):
                    raise
                self.found_dead(replica, error)
            finally:
                replica.in_flight -= 1

    def counted(self, replica: int, tokens: int) -> None:
        """Tokens one served request decoded, on the replica that served it."""
        self.replicas[replica].tokens += tokens

    async def await_a_revival(self) -> None:
        """Wait until some revival finishes — or raise the death that took
        the last replica when nothing is being revived."""
        pending = [task for task in self._reviving.values() if not task.done()]
        if not pending:
            raise self.last_death or Unreachable(
                f"pool {self.name!r} has no live replica and no revival")
        await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

    # ---- health and revival -------------------------------------------------

    def found_dead(self, replica: Replica, error: BaseException) -> None:
        """A REPLICA FOUND DEAD ON CONTACT: off the routing table at once (its
        keys move to survivors as they are next asked for), journaled with
        the error, and its revival started now — not at the next wave."""
        self.last_death = error
        if not replica.live:
            return
        replica.live = False
        replica.deaths += 1
        held = sum(1 for index in self.keys.values() if index == replica.index)
        self.record({"event": "replica-dead", "replica": replica.index,
                     "keys": held, "error": f"{type(error).__name__}: {error}"[:500]})
        if self.reviver is not None and replica.index not in self.live_revivals():
            self._reviving[replica.index] = asyncio.get_running_loop().create_task(
                self.revive(replica))

    def live_revivals(self) -> set[int]:
        return {index for index, task in self._reviving.items() if not task.done()}

    async def revive(self, replica: Replica) -> None:
        """RE-PLACE, BRING UP, READMIT (ADR 0022, Q3): the reviver places a
        fresh host for this replica through the desk; every bundle this pool
        serves and the current wave's libraries are delivered to it; and only
        then does it take requests. A failed attempt is journaled and tried
        again; one that never succeeds leaves the survivors serving."""
        started = time.time()
        for attempt in range(1, REVIVE_ATTEMPTS + 1):
            try:
                engine = await self.reviver.recarve(self.name, replica.index)
                await self.bring_up(engine)
            except Exception as error:      # noqa: BLE001 — journaled, retried
                self.record({"event": "replica-revive-failed",
                             "replica": replica.index, "attempt": attempt,
                             "error": f"{type(error).__name__}: {error}"[:500]})
                await asyncio.sleep(min(REVIVE_BACKOFF_S * attempt, REVIVE_BACKOFF_MAX_S))
                continue
            # bring_up returned with the engine current and NO await since:
            # readmission is atomic with respect to every delivery
            replica.engine = engine
            replica.live = True
            replica.revivals += 1
            self.record({"event": "replica-revived", "replica": replica.index,
                         "attempts": attempt,
                         "seconds": round(time.time() - started, 3)})
            return
        self.record({"event": "replica-revive-abandoned", "replica": replica.index,
                     "seconds": round(time.time() - started, 3)})

    async def bring_up(self, engine: Engine) -> None:
        """Make a fresh engine current: every pin this pool serves, then the
        wave's libraries, then CHECK — a delivery that landed meanwhile
        changed what current means, so the loop runs again until nothing
        moved under it. Returns with no await after the final check."""
        while True:
            pins = tuple(self.pins.values())
            libraries = dict(self.libraries)
            for pin in pins:
                await asyncio.to_thread(self.bring_up_pin, engine, pin)
            for name, payload in sorted(libraries.items()):
                await asyncio.to_thread(engine.add_library, name, payload)
            if tuple(self.pins.values()) == pins and self.libraries == libraries:
                return

    def bring_up_pin(self, engine: Engine, pin: Bundle) -> None:
        """One bundle on one engine: a bundle carrying its payloads is added
        as it is (under wire delivery its version may not be checkpointed
        yet); a pin is restored by the run's rule."""
        if pin.payloads:
            add_if_missing(engine, pin)
        else:
            self.restore(engine, pin)

    # ---- delivery -------------------------------------------------------------

    def remember(self, bundle: Bundle) -> None:
        """This bundle is one the pool serves — newest last, a full bundle
        winning over a pin of the same id, at most PINS_HELD of them."""
        held = self.pins.pop(bundle.bundle_id, None)
        if held is not None and held.payloads and not bundle.payloads:
            bundle = held
        self.pins[bundle.bundle_id] = bundle
        while len(self.pins) > PINS_HELD:
            self.pins.pop(next(iter(self.pins)))

    async def deliver(self, bundle: Bundle) -> None:
        """THE WIRE (ADR 0014) ON EVERY LIVE REPLICA, BEFORE THE COMMIT: the
        bundle is remembered first and the live set read in the same breath,
        so a replica readmitted after this line was brought up to it. A
        replica that dies under the delivery is found dead (and its revival
        starts now); the others still receive it."""
        self.remember(bundle)
        await self.on_every_live_replica(lambda engine: engine.add_bundle(bundle))

    async def restore_pin(self, pin: Bundle) -> None:
        """Make sure every live replica can serve `pin` — the route's half of
        delivery, run once per wave (`routes_at`): asking, and restoring on a
        miss, exactly as one engine always was."""
        self.remember(pin)
        await self.on_every_live_replica(lambda engine: self.bring_up_pin(engine, pin))

    async def hand_libraries(self, payloads: Mapping[str, bytes]) -> None:
        """The wave's library sets (ADR 0019) on every live replica, and kept
        as what a replica revived during this wave is handed."""
        self.libraries = dict(payloads)

        def hand(engine: Engine) -> None:
            for name, payload in sorted(payloads.items()):
                engine.add_library(name, payload)
        await self.on_every_live_replica(hand)

    async def on_every_live_replica(self, verb: Callable[[Engine], None]) -> None:
        """One synchronous engine verb on every live replica, concurrently,
        each from a thread. A replica death is found, not raised — unless it
        took the last live replica and nothing is reviving it."""
        targets = self.live()

        async def one(replica: Replica) -> None:
            try:
                await asyncio.to_thread(verb, replica.engine)
            except Exception as error:
                if not is_replica_death(error):
                    raise
                self.found_dead(replica, error)

        await asyncio.gather(*(one(replica) for replica in targets))
        if not self.live() and not self.live_revivals():
            raise self.last_death

    # ---- observability --------------------------------------------------------

    def record(self, row: dict) -> None:
        """A death or a revival, the moment it happens: kept on the pool and
        journaled on the host the run is anchored on."""
        row = {**row, "t": time.time(), "pool": self.name, "run_id": self.run_id}
        self.events.append(row)
        if self.journal is not None:
            try:
                self.journal.append(row)
            except Exception as error:      # noqa: BLE001 — observability only
                print(f"[pool {self.name}] journal: {error}", flush=True)

    def window(self) -> dict:
        """The pool since the last window, per replica — requests, decoded
        tokens and tokens/s, new and returning keys (a returning key is an
        affinity hit), peak in flight, keys held — plus the key table's size
        and its moves. Drains the window counters; deaths and revivals are
        monotone. Journaled once per rollout wave by the Generator."""
        now = time.time()
        seconds = max(now - self._window_started, 1e-9)
        held = self.keys_held()
        rows = []
        for replica in self.replicas:
            asked = replica.new_keys + replica.seen_keys
            rows.append({
                "replica": replica.index, "live": replica.live,
                "requests": replica.requests, "tokens": replica.tokens,
                "tokens_per_s": round(replica.tokens / seconds, 3),
                "new_keys": replica.new_keys, "seen_keys": replica.seen_keys,
                "affinity_hit_rate": (round(replica.seen_keys / asked, 4)
                                      if asked else None),
                "peak_in_flight": replica.peak_in_flight,
                "keys": held.get(replica.index, 0),
                "deaths": replica.deaths, "revivals": replica.revivals})
            replica.requests = replica.tokens = 0
            replica.new_keys = replica.seen_keys = 0
            replica.peak_in_flight = replica.in_flight
        moves, self.moves = self.moves, 0
        self._window_started = now
        return {"seconds": round(seconds, 3), "keys": len(self.keys),
                "moves": moves, "replicas": rows}


@asynccontextmanager
async def admitted(arbiter, pools: Sequence[Pool]):
    """THE POOLS AT THE DOOR (ADR 0022): every replica's engine of every pool
    admitted at the run's arbiter, as one admission. A replica attached at
    Phase 1 keeps its label and group; one revived since is another host's
    metal and attaches free, as every remote pool does — admission here is
    bookkeeping, and the real door is its own host's."""
    for pool in pools:
        for replica in pool.replicas:
            if not arbiter.is_attached(replica.engine):
                arbiter.attach(replica.engine, label=f"remote:"
                               f"{replica_key(pool.name, replica.index)}")
    async with arbiter.admit_all([engine for pool in pools
                                  for engine in pool.engines()]):
        yield


Handed = Engine | Replicas | Pool
"""What a pool name may be mapped to when a run is handed its metal."""
