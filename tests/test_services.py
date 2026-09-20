"""Standing lifecycle duties run on the shared Desk and MetalService.

These are CPU tests: they check real journal replay, the idle-release path,
concurrent lease renewal and resident-duty cleanup without leasing GPUs.
The worktree's two retirement-drain cases (a release awaiting learner calls
whose client already left) ride the desk custody rewrite and land with it
in ADR 0014 Part C.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from rlstack.data.stores.local import LocalStore
from rlstack.runner.desk import Desk, DeskError, Metal, MetalService
from rlstack.runner.host import Host, Partition
from rlstack.runner.remote import (
    LocalTransport, RemoteDesk, RemoteHost, without_epoch,
)
from rlstack.runner.residents import Builds, FakeEngineBuild, FakeLearnerBuild
from rlstack.runner.venues.runtime import DeskRuntime, MetalRuntime


async def eventually(predicate: Callable[[], bool], timeout: float = 1.0):
    """Wait for a state transition, with a bound that makes missing duties fail."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.002)


class FakeProvider:
    def __init__(self):
        self.boot = Mock()
        self.terminate = AsyncMock(return_value=True)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        self.address = "http://runtime.example:8000"
        self.service = MetalService(
            Metal("card", "L4", 1, 24.0), store=self.store,
            address_of=lambda name: self.address + "#" + name,
            epoch="life-1")
        self.transports = {self.address: LocalTransport(self.service)}
        self.provider = FakeProvider()
        self.boot = self.provider.boot
        self.terminate = self.provider.terminate

    def transport(self, address):
        return self.transports[without_epoch(address)]

    def desk_runtime(self, **kwargs):
        options = {"idle_tick_s": 0.005, "reap_tick_s": None}
        options.update(kwargs)
        return DeskRuntime(
            self.store, transport_for=self.transport, provider=self.provider, bootable_metals=frozenset({"card"}),
            **options)

    def metal_runtime(self, desk, **kwargs):
        options = {"heartbeat_s": 0.005, "host_tick_s": 0.005}
        options.update(kwargs)
        return MetalRuntime(
            self.service, desk, address=self.address,
            container="lease-1", idle_s=90.0, **options)

    def register(self, runtime, **kwargs):
        runtime.desk.register_metal(
            self.service.metal, self.address, container="lease-1",
            epoch=self.service.epoch, **kwargs)

    def born_host(self, learner=None):
        host = Host(
            "carved", engines=(), learner=learner, store=self.store,
            partition=Partition(metal="card", gpu="L4", devices=(0,), memory=1.0),
            epoch=self.service.epoch, sampler=lambda: None)
        self.service.adopt_born(host, self.address + "#carved")
        self.addCleanup(self.service.unroute, self.address + "#carved")
        return host

    def test_replay_preserves_finite_idle_limits_and_removes_historical_pins(self):
        original = Desk(self.store, lambda address: RemoteHost(self.transport(address)))
        original.register_metal(self.service.metal, self.address, idle_s=None)
        original.register_metal(Metal("other", "L4", 1, 24.0), idle_s=12.0)
        original.register_metal(Metal("infinite", "L4", 1, 24.0), idle_s=float("inf"))
        runtime = self.desk_runtime(idle_s=30.0)
        self.assertEqual(runtime.desk.idle_limit("card"), 30.0)
        self.assertEqual(runtime.desk.idle_limit("other"), 12.0)
        self.assertEqual(runtime.desk.idle_limit("infinite"), 30.0)
        with self.assertRaises(DeskError):
            runtime.desk.register_metal(Metal("pinned", "L4", 1, 24.0), idle_s=None)
        self.assertIs(runtime.campaigns.store, self.store)

    def test_configuration_cannot_disable_idle_release(self):
        for value in (None, 0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.desk_runtime(idle_s=value)
        with self.assertRaises(ValueError):
            self.desk_runtime(idle_tick_s=None)
        with self.assertRaises(ValueError):
            self.metal_runtime(RemoteDesk(LocalTransport(self.service)), heartbeat_s=0)

    async def test_idle_release_stays_enabled_when_recovery_polling_is_disabled(self):
        now = [1000.0]
        runtime = self.desk_runtime(idle_s=10.0, clock=lambda: now[0])
        self.register(runtime)
        runtime.desk.reap = AsyncMock()
        async with runtime:
            await eventually(lambda: "card" in runtime.desk.idle_since)
            now[0] += 11.0
            await eventually(lambda: "card" in runtime.desk.released)
        self.assertTrue(self.service.released.is_set())
        self.terminate.assert_awaited_once_with("lease-1")
        runtime.desk.reap.assert_not_awaited()
        self.assertTrue(all(task.done() for task in runtime.tasks))

    async def test_reaper_is_the_existing_configured_pass(self):
        runtime = self.desk_runtime(reap_tick_s=0.005, reap_probes=2, reap_wait_s=0.25)
        runtime.desk.reap = AsyncMock(return_value={})
        async with runtime:
            await eventually(lambda: runtime.desk.reap.await_count > 0)
        runtime.desk.reap.assert_awaited_with(probes=2, wait=0.25)

    async def test_disabled_recovery_keeps_silent_listings_on_registration(self):
        runtime = self.desk_runtime(auto_recover=False, reap_tick_s=0.005)
        self.register(runtime)
        host = self.born_host()
        address = self.address + "#carved"
        silent = LocalTransport(self.service.service_for_host(host.name))
        silent.ask = AsyncMock(side_effect=TimeoutError("forward unavailable"))
        self.transports[address] = silent
        runtime.desk.list_host(host.name, (), address, metal="card", epoch="life-1")
        remote = RemoteDesk(LocalTransport(runtime.campaigns))
        async with runtime:
            reply = await remote.register_metal(
                "card", "L4", 1, 24.0, self.address,
                epoch="life-1", container="lease-1", idle_s=90.0)
        self.assertIn(host.name, runtime.desk.listings)
        self.assertEqual(reply["reaped"], [])
        self.assertEqual(reply["retried"], {})
        silent.ask.assert_not_awaited()
        self.assertEqual(len(runtime.tasks), 1)  # the mandatory idle duty

    async def test_disabled_recovery_blocks_parked_and_indirect_retries(self):
        runtime = self.desk_runtime(auto_recover=False)
        runtime.desk.park("unfinished", "owner unknown")
        runtime.desk.finished = Mock(side_effect=AssertionError("must not retry"))
        self.assertEqual(await runtime.desk.retry_parked(), {})
        self.assertEqual(await runtime.desk.reconcile_metal("card"), [])
        runtime.desk.finished.assert_not_called()
        for work in (runtime.desk.reap(), runtime.desk.reroute("unfinished"),
                     runtime.desk.decommission("carved", reroute=True)):
            with self.assertRaisesRegex(DeskError, "recovery is disabled"):
                await work
        self.assertIn("unfinished", runtime.desk.parked())
        self.terminate.assert_not_awaited()
        self.boot.assert_not_called()

    async def test_failed_idle_observation_is_recorded_and_retried(self):
        runtime = self.desk_runtime()
        observations = 0

        async def observe(now):
            nonlocal observations
            observations += 1
            if observations == 1:
                raise TimeoutError("unknown owner")

        runtime.desk.observe_idle = observe
        runtime.desk.release_idle = AsyncMock(return_value=[])
        with self.assertLogs("rlstack.runner.venues.runtime", level="WARNING"):
            async with runtime:
                await eventually(lambda: runtime.desk.release_idle.await_count > 0)
        self.assertEqual(len(runtime.errors), 1)
        self.assertIn("unknown owner", runtime.errors[0].error)
        self.terminate.assert_not_awaited()
        self.boot.assert_not_called()

    async def test_boot_uses_the_shared_allowlist_and_callback(self):
        runtime = self.desk_runtime()
        self.assertFalse(await runtime.desk.knock("historical"))
        self.assertTrue(await runtime.desk.knock("card"))
        self.boot.assert_called_once_with("card")

    def test_desk_routes_only_its_plane(self):
        runtime = self.desk_runtime()
        self.assertIs(runtime.service_for(""), runtime.campaigns)
        with self.assertRaises(DeskError):
            runtime.service_for("carved")

    async def test_desk_context_does_not_terminate_active_allocations(self):
        runtime = self.desk_runtime(idle_tick_s=100)
        self.register(runtime)
        async with runtime:
            pass
        self.assertFalse(self.service.released.is_set())
        self.terminate.assert_not_awaited()

    async def test_registration_uses_measured_card_epoch_lease_and_recipe(self):
        runtime = self.desk_runtime(heartbeat_s=0.005)
        remote = RemoteDesk(LocalTransport(runtime.campaigns))
        builds = Builds(FakeEngineBuild(), FakeLearnerBuild())
        metal = self.metal_runtime(remote, builds=builds)
        async with metal:
            await eventually(lambda: "card" in runtime.desk.metal)
            await metal.registration
            self.assertEqual(runtime.desk.metal["card"], self.service.metal)
            self.assertEqual(runtime.desk.epoch_of("card"), "life-1")
            self.assertEqual(runtime.desk.metal_containers["card"], "lease-1")
            self.assertEqual(runtime.desk.recipe_for("card"), builds)
            self.assertEqual(runtime.desk.idle_limit("card"), 90.0)
            self.assertEqual(metal.heartbeat_s, 0.005)
        self.assertTrue(self.service.released.is_set())

    async def test_registration_recovery_cannot_starve_heartbeats(self):
        runtime = self.desk_runtime(heartbeat_s=0.005)
        blocked = asyncio.Event()
        recovery_started = asyncio.Event()

        async def recover():
            recovery_started.set()
            await blocked.wait()
            return {}

        runtime.desk.retry_parked = recover
        remote = RemoteDesk(LocalTransport(runtime.campaigns))
        remote.heartbeat = AsyncMock(wraps=remote.heartbeat)
        async with self.metal_runtime(remote) as metal:
            await recovery_started.wait()
            await eventually(lambda: remote.heartbeat.await_count >= 2)
            self.assertFalse(metal.registration.done())
            self.assertTrue(runtime.desk.leased("card"))
            blocked.set()
            await metal.registration

    async def test_unknown_lease_re_registers_but_transport_timeout_does_not(self):
        runtime = self.desk_runtime()
        remote = RemoteDesk(LocalTransport(runtime.campaigns))
        remote.register_metal = AsyncMock(return_value={"heartbeat_s": 0.005})
        remote.heartbeat = AsyncMock(side_effect=TimeoutError("wire silent"))
        with self.assertLogs("rlstack.runner.venues.runtime", level="WARNING"):
            async with self.metal_runtime(remote) as metal:
                await eventually(lambda: remote.heartbeat.await_count >= 2)
                self.assertEqual(remote.register_metal.await_count, 1)
                remote.heartbeat.side_effect = None
                remote.heartbeat.return_value = {"heard": False, "error": "unknown lease"}
                await eventually(lambda: remote.register_metal.await_count >= 2)
                self.assertFalse(self.service.released.is_set())
        self.assertTrue(any("wire silent" in error.error for error in metal.errors))

    async def test_old_epoch_cannot_replace_newer_registration(self):
        remote = RemoteDesk(LocalTransport(self.desk_runtime().campaigns))
        remote.register_metal = AsyncMock(return_value={"heartbeat_s": 0.005})
        remote.heartbeat = AsyncMock(return_value={
            "heard": False, "epoch": "life-2", "error": "epoch replaced"})
        with self.assertLogs("rlstack.runner.venues.runtime", level="WARNING"):
            async with self.metal_runtime(remote):
                await eventually(lambda: remote.heartbeat.await_count >= 2)
                self.assertEqual(remote.register_metal.await_count, 1)

    async def test_metal_and_host_heartbeats_renew_concurrently(self):
        host = self.born_host()
        remote = RemoteDesk(LocalTransport(self.desk_runtime().campaigns))
        metal_entered = asyncio.Event()
        host_entered = asyncio.Event()

        async def heartbeat(name, epoch, residual=None):
            self.assertEqual(epoch, "life-1")
            if name == "card":
                metal_entered.set()
                await host_entered.wait()
            else:
                self.assertEqual(name, host.name)
                host_entered.set()
                await metal_entered.wait()
            return {"heard": True, "heartbeat_s": 0.1}

        remote.heartbeat = heartbeat
        metal = self.metal_runtime(remote)
        self.assertTrue(await asyncio.wait_for(metal.heartbeat(), timeout=1))

    async def test_host_duties_start_and_end_with_the_carved_host(self):
        remote = RemoteDesk(LocalTransport(self.desk_runtime().campaigns))
        host = self.born_host()
        started = {"stats": asyncio.Event(), "watch": asyncio.Event()}
        stopped = {"stats": asyncio.Event(), "watch": asyncio.Event()}

        async def duty(name):
            started[name].set()
            try:
                await asyncio.Event().wait()
            finally:
                # Cleanup itself awaits; double cancellation would lose it.
                await asyncio.sleep(0)
                stopped[name].set()

        host.run_stats = lambda: duty("stats")
        host.watch_residents = lambda: duty("watch")
        async with self.metal_runtime(remote) as metal:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), 1)
            self.assertIs(metal.service_for("carved").host, host)
            await self.service.decarve("carved")
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in stopped.values())), 1)
            await eventually(lambda: not metal.host_tasks)

    async def test_shutdown_finishes_host_duties_stops_residents_and_unroutes(self):
        remote = RemoteDesk(LocalTransport(self.desk_runtime().campaigns))
        host = self.born_host()
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def stats():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                stopped.set()

        host.run_stats = stats
        with patch.object(self.service, "end_residents", return_value=[]) as end:
            async with self.metal_runtime(remote) as metal:
                await started.wait()
            end.assert_called_once_with(host)
        self.assertTrue(stopped.is_set())
        self.assertEqual(self.service.services, {})
        self.assertEqual(self.service.hosts, {})
        with self.assertRaises(RuntimeError):
            metal.service_for("")
        self.assertTrue(all(task.done() for task in metal.tasks))

    async def test_shared_release_ends_wait_without_rebirth(self):
        remote = RemoteDesk(LocalTransport(self.desk_runtime().campaigns))
        async with self.metal_runtime(remote) as metal:
            waiting = asyncio.create_task(metal.wait())
            self.assertIs(metal.service_for(""), self.service)
            await self.service.release()
            await asyncio.wait_for(waiting, timeout=1)
            with self.assertRaises(RuntimeError):
                metal.service_for("")


if __name__ == "__main__":
    unittest.main()
