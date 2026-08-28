"""GpuArbiter: the physical resource owns admission; experiments only request.

An arbiter is constructed by whoever owns the metal — a Host makes its own
unless handed one — and is shared by every experiment admitted to it, which is
why per-experiment mutexes died: two tenants' private locks coordinate nothing.
It governs the partition that owns it, not the device: several sub-GPU hosts on
one device each admit independently. Alternation exists only inside an
exclusive group
(GpuGroup.sharing="sleep", or a host's own group); everything else co-resides
and admit() is a plain counter — and even inside a group, alternation is about
memory, never mutual exclusion on work: any amount of work overlaps on the
resident that is live. Scheduling policy is sticky drain-until-blocked and
lives deliberately OUTSIDE run identity (I5): it moves wall-clock and, under
max_policy_lag > 0, which recorded version served a wave.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from rlstack.runner.meters import TrafficMeter

Hook = Callable[[], Awaitable[None]]


@dataclass
class _Resident:
    label: str
    group: str | None                       # exclusive group; None = free
    fraction: float | None = None
    wake: Hook | None = None
    evict: Hook | None = None
    in_flight: int = 0
    waiting_since: float | None = None      # oldest un-admitted request


@dataclass
class _Group:
    """One alternation set: at most one member resident at a time."""

    resident: object | None = None
    last_switch: float = float("-inf")
    handoff_to: object | None = None        # aging: stop feeding the resident


class GpuArbiter:
    def __init__(self, *, quantum: float = 0.0, max_wait: float | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 meter: TrafficMeter | None = None) -> None:
        """Two knobs bound drain-until-blocked's pathologies. `quantum` is the
        minimum seconds between switches — hysteresis against thrash when
        misaligned tenants interleave (0 = off). `max_wait` is the seconds
        after which a starving waiter forces a handoff: new admits of the
        current resident stop being fed so its work drains (None = off, which
        the blackboard's own dataflow bounds whenever max_policy_lag does).

        The door is also the one place that knows what waited and what is
        running, so it counts both into the host's `meter` — its own if none
        is handed in, its host's once the host wires one (Host.wire_meter)."""
        self.quantum = quantum
        self.max_wait = max_wait
        self.clock = clock
        self.meter = meter or TrafficMeter()
        self.switches: list[str] = []       # residency history, labels
        self._residents: dict[int, _Resident] = {}
        self._objects: dict[int, object] = {}   # keep attached objects alive
        self._groups: dict[str, _Group] = {}
        self._cond: asyncio.Condition | None = None   # created in a loop

    # ---- attach --------------------------------------------------------------

    def attach(self, obj: object, *, label: str, group: str | None = None,
               fraction: float | None = None, wake: Hook | None = None,
               evict: Hook | None = None) -> None:
        """Register a resident (idempotent by object; first label wins, hooks
        and fraction may be supplied by any attacher).

        Group `None` is "no alternation demanded", not "must be free": it
        DEFERS to whatever group the resident already carries — a regime-host
        declares alternation at birth (#43) and later tenants join without an
        opinion. Two CONFLICTING group names still raise: an experiment may
        not re-legislate the metal's physical truth."""
        entry = self._residents.get(id(obj))
        if entry is None:
            self._residents[id(obj)] = _Resident(label, group, fraction,
                                                 wake, evict)
            self._objects[id(obj)] = obj
            if group is not None:
                self._groups.setdefault(group, _Group())
            return
        if group is not None and entry.group != group:
            raise ValueError(
                f"resident {entry.label!r} attached with exclusive group "
                f"{entry.group!r}, re-attached with {group!r}")
        entry.wake = entry.wake or wake
        entry.evict = entry.evict or evict
        entry.fraction = entry.fraction if entry.fraction is not None else fraction

    def declared_load(self) -> float:
        """Sum of declared fractions over residents that co-reside: the free
        residents plus the largest member of each exclusive group. Declared,
        never measured — the host's fit check refuses submissions against this
        number; nothing polices the metal itself."""
        free = sum(r.fraction or 0.0 for r in self._residents.values()
                   if r.group is None)
        per_group = {}
        for r in self._residents.values():
            if r.group is not None:
                per_group[r.group] = max(per_group.get(r.group, 0.0),
                                         r.fraction or 0.0)
        return free + sum(per_group.values())

    def is_attached(self, obj: object) -> bool:
        """Whether this object is already a resident here (its fraction is
        already counted — the host's capacity check asks)."""
        return id(obj) in self._residents

    def attached_group(self, obj: object) -> str | None:
        """The exclusive group this resident already carries (None: free, or
        never attached) — the metal's physical truth, which a later
        tenant's declaration defers to (attach_residents asks)."""
        entry = self._residents.get(id(obj))
        return entry.group if entry is not None else None

    def residency(self) -> dict[str, str | None]:
        """Per exclusive group: the resident's label (None: nothing yet).
        The GpuSet's STATE, as the host's status reports it."""
        return {name: (self._entry(group.resident).label
                       if group.resident is not None else None)
                for name, group in sorted(self._groups.items())}

    # ---- admit ---------------------------------------------------------------

    @asynccontextmanager
    async def admit(self, obj: object):
        """Run the enclosed work with `obj` resident. Free residents never
        block; exclusive residents wait for their group's drain-and-switch.

        THE ADMISSION DOOR IS THE MEASUREMENT POINT: every unit of work that
        passes through it reports how long it queued and stays counted as
        in-flight until it leaves — a free resident's honest zero included,
        because "nothing waited" is the fact the window is there to show."""
        entry = self._entry(obj)
        if entry.group is None:
            entry.in_flight += 1
            self.meter.admitted(0.0)
            try:
                yield
            finally:
                entry.in_flight -= 1
                self.meter.released()
            return

        cond = self._condition()
        group = self._groups[entry.group]
        async with cond:
            queued = self.clock()
            if entry.waiting_since is None:
                entry.waiting_since = queued
            while not self._may_enter(obj, entry, group):
                self._age(group)
                await self._wait(cond)
            entry.waiting_since = None
            if group.resident is not obj:
                await self._switch(group, obj, entry)
            entry.in_flight += 1
            # this request's own wait, not the resident's oldest: two callers
            # of one resident queue independently
            self.meter.admitted(self.clock() - queued)
        try:
            yield
        finally:
            async with cond:
                entry.in_flight -= 1
                self.meter.released()
                cond.notify_all()

    @asynccontextmanager
    async def admit_all(self, objs):
        """Admit several residents at once (a post pipeline's pools), in
        deterministic label order. Two members of one exclusive group cannot
        be co-resident by definition — validate refuses such pipelines at
        submit; this raises if one slips through."""
        distinct = {id(o): o for o in objs}
        entries = [(self._entry(o).label, o) for o in distinct.values()]
        groups = [self._entry(o).group for _, o in entries
                  if self._entry(o).group is not None]
        if len(groups) != len(set(groups)):
            raise ValueError(
                "admit_all needs two residents of one exclusive group "
                "co-resident — an alternation set cannot satisfy that")
        async with AsyncExitStack() as stack:
            for _, obj in sorted(entries, key=lambda pair: pair[0]):
                await stack.enter_async_context(self.admit(obj))
            yield

    # ---- the policy ----------------------------------------------------------

    def _may_enter(self, obj: object, entry: _Resident, group: _Group) -> bool:
        if group.resident is obj:
            # sticky fast path — unless an aged waiter asked us to drain
            return group.handoff_to is None or group.handoff_to is obj
        if group.resident is None:
            return True
        # switch: current resident's work has drained, hysteresis satisfied
        current = self._entry(group.resident)
        if current.in_flight > 0:
            return False
        if self.clock() - group.last_switch < self.quantum:
            return False
        return group.handoff_to is None or group.handoff_to is obj

    def _age(self, group: _Group) -> None:
        """Mark a handoff when some waiter has starved past max_wait: the
        current resident stops being fed, so its in-flight work drains."""
        if self.max_wait is None or group.handoff_to is not None:
            return
        now = self.clock()
        for oid, entry in self._residents.items():
            waiting = entry.waiting_since
            if (waiting is not None and self._objects[oid] is not group.resident
                    and now - waiting > self.max_wait):
                group.handoff_to = self._objects[oid]
                return

    async def _switch(self, group: _Group, obj: object, entry: _Resident) -> None:
        if group.resident is not None:
            old = self._entry(group.resident)
            if old.evict is not None:
                await old.evict()
        if entry.wake is not None:
            await entry.wake()
        group.resident = obj
        group.last_switch = self.clock()
        group.handoff_to = None
        self.switches.append(entry.label)

    # ---- plumbing ------------------------------------------------------------

    def _entry(self, obj: object) -> _Resident:
        entry = self._residents.get(id(obj))
        if entry is None:
            raise KeyError(f"object was never attached: {obj!r}")
        return entry

    def _condition(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    async def _wait(self, cond: asyncio.Condition) -> None:
        """Wait for a notify, re-checking at least every 50ms so time-based
        rules (quantum, max_wait) make progress without their own timers."""
        try:
            await asyncio.wait_for(cond.wait(), 0.05)
        except TimeoutError:
            pass
