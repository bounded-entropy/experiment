"""Standing duties for a Desk and a MetalService, independent of a venue.

The desk still owns placement, leases, the fleet journal and idle release;
the metal still owns carved hosts and resident teardown. These context
managers give those existing services a process lifetime. A deployment only
supplies its store, transport resolver and allocation callbacks, then serves
``service_for`` through its chosen transport.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import logging
import math
import time

from rlstack.data.stores.base import Store
from rlstack.runner.campaign import Campaigns
from rlstack.runner.desk import (
    HEARTBEAT_S, IDLE_S, LEASE_S, Desk, DeskError, MetalService,
)
from rlstack.runner.host import Host
from rlstack.runner.remote import (
    RemoteDesk, RemoteHost, RemoteMetal, Service, Transport,
)
from rlstack.runner.residents import Builds
from rlstack.runner.venues.provider import AllocationProvider


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DutyFailure:
    """A failed background duty, not evidence that a remote owner died."""

    duty: str
    t: float
    error: str


def record_failure(errors: deque[DutyFailure], duty: str,
                   error: Exception) -> None:
    """Keep recent failures inspectable and print them into the process log."""
    errors.append(DutyFailure(duty, time.time(), str(error)))
    logger.warning("%s: %s", duty, error)


def finite_positive(name: str, value: float) -> None:
    """Standing duties and automatic GPU idle shutdown need finite clocks."""
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


async def cancel_duties(tasks: list[asyncio.Task]) -> None:
    """Finish cancellation before their service's context exits."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


class DeskRuntime:
    """One journal writer, its campaign door, and its standing clocks.

    ``reap_tick_s=None`` disables automatic recovery probes, independently of
    mandatory idle release. Use it where an unreachable old owner has not yet
    been proved stopped. The shared Desk retains retirement intent and blocks
    replacement until acknowledged teardown or provider-confirmed termination.
    This wrapper cannot strengthen the provider's physical/storage contract.

    ``auto_recover=False`` also disables registration reconciliation and
    parked-run retries in the shared Desk, including indirect recovery paths.

    Exiting stops the CPU service's duties. It does not declare live GPU
    owners dead or release active work; a provider lease remains the backstop
    if the desk itself disappears.
    """

    def __init__(self, store: Store, *,
                 transport_for: Callable[[str], Transport],
                 provider: AllocationProvider,
                 bootable_metals: frozenset[str],
                 auto_recover: bool = True,
                 idle_s: float = IDLE_S,
                 restored_idle_s: float | None = None,
                 idle_tick_s: float = 30.0,
                 reap_tick_s: float | None = 900.0,
                 reap_probes: int = 3,
                 reap_wait_s: float = 30.0,
                 lease_s: float = LEASE_S,
                 heartbeat_s: float = HEARTBEAT_S,
                 probe_deadline_s: float = 90.0,
                 residual_deadline_s: float = 30.0,
                 clock: Callable[[], float] = time.time) -> None:
        for name, value in (("idle_s", idle_s), ("idle_tick_s", idle_tick_s),
                            ("lease_s", lease_s), ("heartbeat_s", heartbeat_s),
                            ("probe_deadline_s", probe_deadline_s),
                            ("residual_deadline_s", residual_deadline_s)):
            finite_positive(name, value)
        if reap_tick_s is not None:
            finite_positive("reap_tick_s", reap_tick_s)
        if restored_idle_s is not None:
            finite_positive("restored_idle_s", restored_idle_s)
        if reap_probes < 0 or not math.isfinite(reap_wait_s) or reap_wait_s < 0:
            raise ValueError("reaper probes and wait must be nonnegative and finite")
        self.desk = Desk.from_journal(
            store, host_for=lambda address: RemoteHost(transport_for(address)),
            metal_for=lambda address: RemoteMetal(transport_for(address)),
            boot_for=provider.boot, terminate_for=provider.terminate,
            bootable_metals=bootable_metals, idle_s=idle_s,
            lease_s=lease_s, heartbeat_s=heartbeat_s,
            probe_deadline_s=probe_deadline_s,
            residual_deadline_s=residual_deadline_s, clock=clock)
        # Historical pinning must not silently disable the deployment's
        # mandatory idle release. Preserve every existing finite limit.
        for name in self.desk.metal:
            limit = self.desk.idle_limit(name)
            if limit is None or not math.isfinite(limit):
                self.desk.declare_idle(name, idle_s if restored_idle_s is None else restored_idle_s)
        self.desk.require_finite_idle = True
        self.desk.automatic_recovery = auto_recover
        self.provider = provider
        self.campaigns = Campaigns(self.desk)
        self.idle_tick_s = idle_tick_s
        self.reap_tick_s = reap_tick_s if auto_recover else None
        self.reap_probes = reap_probes
        self.reap_wait_s = reap_wait_s
        self.errors: deque[DutyFailure] = deque(maxlen=100)
        self.tasks: list[asyncio.Task] = []
        self.started = False

    def service_for(self, host: str) -> Service:
        """The desk's one plane; a host name belongs at a metal's door."""
        if host:
            raise DeskError(f"the desk has no host plane {host!r}")
        return self.campaigns

    async def __aenter__(self) -> DeskRuntime:
        if self.started:
            raise RuntimeError("a DeskRuntime has one process lifetime")
        self.started = True
        self.tasks.append(asyncio.create_task(self.tick_idle()))
        if self.reap_tick_s is not None:
            self.tasks.append(asyncio.create_task(self.tick_reaper()))
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await cancel_duties(self.tasks)

    async def tick_idle(self) -> None:
        """Observe existing busy rules and release through the existing desk."""
        while True:
            await asyncio.sleep(self.idle_tick_s)
            try:
                now = self.desk.now()
                await self.desk.observe_idle(now)
                await self.desk.release_idle(now)
            except Exception as error:
                record_failure(self.errors, "desk idle", error)

    async def tick_reaper(self) -> None:
        """Run the configured shared recovery pass, with no private scheduler."""
        while True:
            await asyncio.sleep(self.reap_tick_s)
            try:
                await self.desk.reap(probes=self.reap_probes,
                                     wait=self.reap_wait_s)
            except Exception as error:
                record_failure(self.errors, "desk reaper", error)


class MetalRuntime:
    """Keep a measured metal registered and watch the hosts carved on it.

    Registration can wait on parked-run recovery, so it never holds up
    heartbeats or host supervision. Errors are recorded and retried at the
    next cadence; a failed heartbeat does not replace a live allocation.
    Store durability belongs to the backend's declared commit points. Modal
    commits sealed updates and operational journals; scratch publishes each
    write. Neither needs a provider commit loop here to make those durable.
    """

    def __init__(self, service: MetalService, desk: RemoteDesk, *,
                 address: str, container: str, idle_s: float,
                 builds: Builds | None = None,
                 heartbeat_s: float = HEARTBEAT_S,
                 host_tick_s: float = 1.0) -> None:
        for name, value in (("idle_s", idle_s), ("heartbeat_s", heartbeat_s),
                            ("host_tick_s", host_tick_s)):
            finite_positive(name, value)
        self.service = service
        self.desk = desk
        self.address = address
        self.container = container
        self.idle_s = idle_s
        self.builds = builds
        self.heartbeat_s = heartbeat_s
        self.host_tick_s = host_tick_s
        self.errors: deque[DutyFailure] = deque(maxlen=100)
        self.tasks: list[asyncio.Task] = []
        self.host_tasks: dict[str, asyncio.Task] = {}
        self.registration: asyncio.Task | None = None
        self.started = False
        self.closing = False

    @classmethod
    def measured(cls, name: str, store: Store, desk: RemoteDesk, *,
                 address: str, host_address: Callable[[str, str], str],
                 container: str, idle_s: float, builds: Builds | None = None,
                 heartbeat_s: float = HEARTBEAT_S) -> MetalRuntime:
        """Create one measured metal and boot identity for either provider."""
        from rlstack.policy.siteschema import hf_schema
        from rlstack.runner.desk import mint_epoch
        from rlstack.runner.remote import transport_for

        epoch = mint_epoch()
        service = MetalService(
            MetalService.measure(name), store=store, builds=builds,
            address_of=lambda host: host_address(host, epoch),
            schema_for=hf_schema, transport_for=transport_for, epoch=epoch)
        return cls(service, desk, address=address, container=container,
                   idle_s=idle_s, builds=builds, heartbeat_s=heartbeat_s)

    def service_for(self, host: str) -> Service:
        """Route the metal plane or one carved host, never revive a release."""
        if self.closing or self.service.released.is_set():
            raise RuntimeError(f"metal {self.service.metal.name!r} is released")
        return self.service if not host else self.service.service_for_host(host)

    async def __aenter__(self) -> MetalRuntime:
        if self.started or self.service.released.is_set():
            raise RuntimeError("a MetalRuntime has one allocation lifetime")
        self.started = True
        self.registration = asyncio.create_task(self.register())
        self.tasks = [asyncio.create_task(self.tick_heartbeats()),
                      asyncio.create_task(self.watch_hosts())]
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.closing = True
        # watch_hosts owns cancellation of its children. Cancelling those
        # twice can interrupt their own asynchronous cleanup.
        tasks = list(self.tasks)
        if self.registration is not None:
            tasks.append(self.registration)
        await cancel_duties(tasks)
        if not self.service.released.is_set():
            await self.service.release()

    async def wait(self) -> None:
        """The process may end when the shared metal service ends its shift."""
        await self.service.until_released()

    async def register(self) -> None:
        """Announce measured capacity, this epoch and a finite idle limit."""
        card = self.service.metal
        try:
            told = await self.desk.register_metal(
                card.name, card.gpu, card.devices, card.vram_gb, self.address,
                builds=None if self.builds is None else self.builds.row(),
                idle_s=self.idle_s, container=self.container,
                epoch=self.service.epoch)
            self.learn_heartbeat(told)
        except Exception as error:
            record_failure(self.errors, "metal registration", error)

    def learn_heartbeat(self, told: dict) -> None:
        """The desk owns the cadence, including while registration is busy."""
        every = float(told.get("heartbeat_s", self.heartbeat_s))
        finite_positive("desk heartbeat_s", every)
        self.heartbeat_s = every

    async def heartbeat(self) -> bool | None:
        """Renew every name concurrently so one slow host cannot starve peers."""
        card = self.service.metal
        names = [card.name, *sorted(self.service.hosts)]
        replies = await asyncio.gather(
            self.desk.heartbeat(card.name, self.service.epoch,
                                self.service.residual()),
            *(self.desk.heartbeat(name, self.service.epoch) for name in names[1:]),
            return_exceptions=True)
        heard = None
        for name, reply in zip(names, replies):
            if isinstance(reply, Exception):
                record_failure(self.errors, f"metal heartbeat {name}", reply)
            elif isinstance(reply, BaseException):
                raise reply
            elif name == card.name:
                heard = bool(reply.get("heard"))
                if heard:
                    self.learn_heartbeat(reply)
                else:
                    record_failure(self.errors, f"metal heartbeat {name}",
                                   RuntimeError(reply.get("error", "not heard")))
                    if reply.get("epoch") not in (None, "", self.service.epoch):
                        # A desk already believes another life of this name.
                        # This old process must not overwrite that generation
                        # by replaying its registration.
                        heard = None
        return heard

    async def tick_heartbeats(self) -> None:
        """Renew independently of registration's potentially long recovery."""
        while not self.service.released.is_set():
            await asyncio.sleep(self.heartbeat_s)
            if self.service.released.is_set():
                return
            try:
                heard = await self.heartbeat()
                if (heard is False and not self.service.released.is_set()
                        and self.registration.done()):
                    self.registration = asyncio.create_task(self.register())
            except Exception as error:
                record_failure(self.errors, "metal heartbeat", error)

    async def watch_hosts(self) -> None:
        """Attach existing stats/watchdog duties as hosts arrive and depart."""
        try:
            while not self.service.released.is_set():
                for name in list(self.host_tasks):
                    if name not in self.service.hosts:
                        await cancel_duties([self.host_tasks.pop(name)])
                for name, host in list(self.service.hosts.items()):
                    task = self.host_tasks.get(name)
                    if task is None or task.done():
                        self.host_tasks[name] = asyncio.create_task(
                            self.host_duties(host))
                await asyncio.sleep(self.host_tick_s)
        finally:
            await cancel_duties(list(self.host_tasks.values()))

    async def host_duties(self, host: Host) -> None:
        """A carved host's GPU/traffic measurements and resident watchdog."""
        try:
            async with asyncio.TaskGroup() as duties:
                duties.create_task(host.run_stats())
                duties.create_task(host.watch_residents())
        except Exception as error:
            record_failure(self.errors, f"host duties {host.name}", error)
