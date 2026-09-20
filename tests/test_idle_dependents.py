"""Idle by dependents, a clock per host, journals that batch, a watchdog that
rechecks (ADR 0014, Part D).

The idle rule's fourth yes: a listing a RUNNING placement routes through is
busy, however quiet its own door — on 2026-09-13 06:14 UTC the desk released
a pool under four training runs anchored elsewhere. The host clock: a carved
listing that is not busy for its limit is decarved before its metal is
released. A host journal line rides the next commit or a flush timer, never a
commit of its own. A stall verdict is confirmed by one more question before it
costs a host.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from rlstack import FakeLearner
from rlstack.data.stores.modal_volume import ModalVolumeStore
from rlstack.runner.campaign import Campaigns
from rlstack.runner.checkpointing import EVERY_UPDATE
from rlstack.runner.host import Host
from test_desk import DeskFixture, go
from test_modal_store import RecordingVolume


class BlockedLearner:
    """Holds every forward at a gate — the run stays RUNNING with its pool
    quiet, which is exactly the shape the old idle rule mistook for idle."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.original = FakeLearner.forward_backward

    def __enter__(self):
        gate, original = self.gate, self.original

        def held(learner, tenant, batch):
            gate.wait()
            return original(learner, tenant, batch)

        self._patch = patch.object(FakeLearner, "forward_backward", held)
        self._patch.start()
        return self

    def __exit__(self, *exc) -> None:
        self.gate.set()
        self._patch.stop()


class DependentsAreBusyTest(DeskFixture):
    def test_a_pool_a_running_run_routes_through_is_busy_however_quiet(self) -> None:
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal", idle_s=600.0)
        desk.host_idle_s = 100.0
        with BlockedLearner() as learner:
            async def drive():
                reply = await Campaigns(desk).submit(self.split_spec(), checkpointing=EVERY_UPDATE)
                self.assertTrue(reply["accepted"], reply)
                pool_host = [name for name in desk.listings if name != reply["host"]][0]
                await desk.observe_idle(1000.0)
                await desk.observe_idle(1060.0)
                await desk.observe_idle(1200.0)
                decarved = await desk.decarve_idle(1200.0)
                # the pool's own door is quiet (no tenancy, nothing in flight,
                # a counter that stopped moving) — and it is still BUSY,
                # because the run anchored on the learner host routes
                # through it
                self.assertEqual(decarved, [])
                self.assertNotIn(pool_host, desk.host_idle_since)
                self.assertNotIn("fake-metal", desk.idle_since)
                self.assertIn(pool_host, desk.listings)
                # the rule, read directly
                frames = {name: await desk.listings[name].told() for name in desk.listings}
                self.assertIn(pool_host, desk.routed_through(frames))
                seen: dict[str, int] = {}
                self.assertTrue(await desk.listing_busy(
                    desk.listings[pool_host], seen, frames[pool_host], desk.routed_through(frames)))
                learner.gate.set()
                await service.hosts[reply["host"]]._adoptions[reply["run_id"]]

            go(drive())

    def test_the_fourth_yes_is_the_only_difference(self) -> None:
        """A quiet listing nobody routes through is idle by the same reading."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal", idle_s=600.0)

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec(), checkpointing=EVERY_UPDATE)
            await service.hosts[reply["host"]]._adoptions[reply["run_id"]]
            name = next(iter(desk.listings))
            seen: dict[str, int] = {}
            told = await desk.listings[name].told()
            await desk.listing_busy(desk.listings[name], seen, told)     # first sight
            desk.admitted_at = seen
            quiet = await desk.listing_busy(desk.listings[name], dict(seen), told)
            routed = await desk.listing_busy(desk.listings[name], dict(seen), told,
                                             frozenset({name}))
            return quiet, routed

        quiet, routed = go(drive())
        self.assertFalse(quiet)
        self.assertTrue(routed)


class HostClockTest(DeskFixture):
    def test_idle_hosts_are_decarved_before_their_metal_is_released(self) -> None:
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal", idle_s=600.0)
        desk.host_idle_s = 100.0

        async def finished():
            reply = await Campaigns(desk).submit(self.split_spec(), checkpointing=EVERY_UPDATE)
            await service.hosts[reply["host"]]._adoptions[reply["run_id"]]
        go(finished())
        carved = sorted(desk.listings)
        self.assertEqual(len(carved), 2)

        async def sweep():
            await desk.observe_idle(1000.0)        # first sight: a reading, no clock
            await desk.observe_idle(1060.0)        # quiet through a tick: both clocks start
            hosts_started = dict(desk.host_idle_since)
            early = await desk.release_idle(1150.0)    # 90 s: nothing due
            listed_early = sorted(desk.listings)
            due = await desk.release_idle(1161.0)      # 101 s: the hosts, not the metal
            return hosts_started, early, listed_early, due

        hosts_started, early, listed_early, due = go(sweep())
        self.assertEqual(hosts_started, {name: 1060.0 for name in carved})
        self.assertEqual(early, [])
        self.assertEqual(listed_early, carved)
        self.assertEqual(due, [])                        # no METAL released...
        self.assertEqual(desk.listings, {})              # ...but both hosts decarved
        self.assertEqual(service.hosts, {})
        self.assertEqual(service.residual(), [24.0, 24.0])
        self.assertIn("fake-metal", desk.metal_remotes)  # still carve-able
        self.assertFalse(service.released.is_set())
        reasons = [(e.get("host"), e.get("reason")) for e in self.store.read_fleet_log()
                   if e.get("event") == "decarve"]
        self.assertEqual(sorted(reasons), [(name, "idle") for name in carved])
        # the metal's own clock goes on ticking over an empty metal
        go(desk.observe_idle(1200.0))
        self.assertEqual(go(desk.release_idle(1661.0)), ["fake-metal"])
        self.assertTrue(service.released.is_set())

    def test_the_host_limit_defaults_to_the_metals(self) -> None:
        self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal", idle_s=600.0)
        self.assertEqual(desk.host_idle_limit("fake-metal"), 600.0)
        desk.host_idle_s = 30.0
        self.assertEqual(desk.host_idle_limit("fake-metal"), 30.0)


class JournalsBatchTest(unittest.TestCase):
    def test_a_host_journal_line_rides_the_next_commit_or_the_timer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            volume = RecordingVolume()
            store = ModalVolumeStore(tmp, volume=volume, journal_flush_s=0.15)
            store.append_host_event("h", {"event": "traffic", "t": 1.0})
            store.append_host_event("h", {"event": "gpu", "t": 1.5})
            self.assertEqual(volume.commits, 0)          # staged, not committed
            # a commit in between carries them and disarms the timer
            run = store.open_run("r1", manifest={"run_id": "r1"})
            created = volume.commits
            self.assertGreaterEqual(created, 1)
            time.sleep(0.3)
            self.assertEqual(volume.commits, created)    # the timer had nothing to do
            # with no commit coming, the timer flushes once for many lines
            store.append_host_event("h", {"event": "traffic", "t": 2.0})
            store.append_host_event("h", {"event": "update", "t": 2.5})
            self.assertEqual(volume.commits, created)
            time.sleep(0.3)
            self.assertEqual(volume.commits, created + 1)
            # the fleet journal is the desk's truth: still per event
            store.append_fleet_event({"event": "parked", "t": 3.0, "run_id": "r1"})
            self.assertEqual(volume.commits, created + 2)
            self.assertEqual(len(store.read_host_log("h")), 4)


class WatchdogRechecksTest(unittest.TestCase):
    class Resident:
        label = "engine:0"
        answers = True
        stopped = False

        class regime:
            capability = "inference"

        def pid(self) -> int:
            return 4242

        def heartbeat(self, *, deadline_s: float) -> dict:
            if not self.answers:
                raise TimeoutError("silent")
            return {"alive": True}

        def stop(self, report_exit: bool = False) -> None:
            self.stopped = True

    def host(self, tmp: str) -> Host:
        from rlstack import LocalStore, fake_qwen_schema
        return Host("h", engines=(), learner=None, store=LocalStore(tmp),
                    schema_for=lambda base: fake_qwen_schema(4, base=base))

    def test_a_resident_that_answers_the_recheck_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            host = self.host(tmp)
            resident = self.Resident()
            heard = {resident.label: 0.0}
            asyncio.run(host.kill_stalled(resident, 601.0, 600.0, heard))
            self.assertFalse(resident.stopped)
            self.assertGreater(heard[resident.label], 0.0)
            events = [e["event"] for e in host.store.read_host_log("h")
                      if e["event"].startswith("stall")]
            self.assertEqual(events, ["stall-recovered"])

    def test_a_resident_that_stays_silent_is_ended_then_journaled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            host = self.host(tmp)
            resident = self.Resident()
            resident.answers = False
            asyncio.run(host.kill_stalled(resident, 601.0, 600.0))
            self.assertTrue(resident.stopped)
            events = [e for e in host.store.read_host_log("h")
                      if e["event"].startswith("stall")]
            self.assertEqual([e["event"] for e in events], ["stalled"])
            self.assertEqual(events[0]["pid"], 4242)


if __name__ == "__main__":
    unittest.main()
