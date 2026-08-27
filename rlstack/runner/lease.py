"""GPU leases: the physical half, kept orthogonal to the logical half.

Asynchronicity is logical (roles await store predicates — signals.py);
colocation is physical (who occupies the metal). A role wraps its GPU work in
`async with lease.held(resource)` and the lease type — derived from the
spec's GpuGroup.sharing — decides what that means:

    OpenLease        dedicated GPUs, or colocated "concurrent" (the fraction
                     treaty partitions memory): held() never blocks.
    ExclusiveLease   colocated "sleep": one resource resident at a time. The
                     resident is STICKY — hooks fire only when the resource
                     actually changes, so the trainer re-acquiring "learner"
                     ten microbatches in a row costs nothing.

Resources are the things that occupy memory ("engine", "learner"), not the
roles that drive them — so a judge-bearing post pipeline, driven by the
trainer, still correctly wakes the ENGINE. Wake/evict hooks (vLLM sleep/wake,
learner offload) are registered per resource; the flexible part of the
alternation — WHEN each daemon wants the mutex — lives in the roles'
overridable condition methods, not here.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from rlstack.spec.specs import EnginesMember

Hook = Callable[[], Awaitable[None]]

ENGINE = "engine"
LEARNER = "learner"


class Lease(ABC):
    @abstractmethod
    def held(self, resource: str):
        """Async context manager: run the enclosed GPU work as `resource`."""


class OpenLease(Lease):
    """No contention: everyone may run at once."""

    @asynccontextmanager
    async def held(self, resource: str):
        yield


class ExclusiveLease(Lease):
    """One resident resource; fair FIFO handoff; sticky with hook-on-switch."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._resident: str | None = None
        self._wake: dict[str, Hook] = {}
        self._evict: dict[str, Hook] = {}
        self.switches: list[str] = []   # residency history, for tests/telemetry

    def on(self, resource: str, wake: Hook | None = None,
           evict: Hook | None = None) -> None:
        """Register what it costs to make `resource` resident / evict it
        (vllm wake_up/sleep, learner load/offload). Unregistered = free."""
        if wake is not None:
            self._wake[resource] = wake
        if evict is not None:
            self._evict[resource] = evict

    @asynccontextmanager
    async def held(self, resource: str):
        async with self._lock:
            if self._resident != resource:
                if self._resident is not None and self._resident in self._evict:
                    await self._evict[self._resident]()
                if resource in self._wake:
                    await self._wake[resource]()
                self._resident = resource
                self.switches.append(resource)
            yield


class LeaseMap:
    """Which lease governs which resource — derived from GpuConfig."""

    def __init__(self, pools: dict[str, Lease], learner: Lease) -> None:
        self._pools = pools
        self._learner = learner

    def for_pool(self, name: str) -> Lease:
        return self._pools.get(name) or OpenLease()

    def for_learner(self) -> Lease:
        return self._learner


def leases_for(spec) -> LeaseMap:
    """One lease per GpuGroup: sleep-sharing gets an ExclusiveLease shared by
    every member of that group (engine and learner contend for the same
    memory); anything else is open."""
    pools: dict[str, Lease] = {}
    learner_lease: Lease = OpenLease()
    for group in spec.gpu_config.groups:
        lease: Lease = (ExclusiveLease() if group.sharing == "sleep"
                        else OpenLease())
        for member in group.members:
            if isinstance(member, EnginesMember):
                pools[member.name] = lease
            else:
                learner_lease = lease
    return LeaseMap(pools, learner_lease)
