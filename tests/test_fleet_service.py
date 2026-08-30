"""The standing fleet: FleetService (the desk), Listing, RemoteFleet.

The claims under test: a campaign's whole surface is one frame (submit at the
desk → placed over listings → adopted at the learner's host, routes threaded);
a placement nothing serves comes back as a BOOT instruction, never a guess;
solo-and-occupied listings are skipped like the join rung skips hosts; the
desk journals every listing and placement and REBUILDS from its own journal
(kill -9 loses a process, never the fleet); and the run an adopted campaign
produces is byte-identical to an in-process submit of the same spec.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest

import json

from common import arith_spec, arith_store
from rlstack import (
    FakeEngine, FakeLearner, GpuConfig, GpuGroup, Host, Regime, Seeds,
    fake_qwen_schema, gpus, learner, pool,
)
from rlstack.spec.canonical import canonical_json
from rlstack.runner.fleet import FleetService, Listing
from rlstack.runner.remote import (
    HostService, LocalTransport, RemoteFleet, RemoteHost,
)

BASE = "Qwen/Qwen3-0.6B"
SCHEMA = fake_qwen_schema(4, base=BASE)


def go(coro):
    return asyncio.run(coro)


class DeskFixture(unittest.TestCase):
    """Two standing hosts behind transports, one desk listing both."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)
        self.transports: dict[str, LocalTransport] = {}

    def stand_up(self, name: str, address: str, *, serves_pool: bool,
                 trains: bool, solo: bool = False) -> Host:
        """One standing host: regimes it wears, a transport it answers on,
        and — for a learner host — the dialer that resolves every OTHER
        address in this test's little world."""
        regimes = []
        engines = ()
        if serves_pool:
            regimes.append(Regime(f"{name}-serve", "inference", BASE, 1))
            engines = (FakeEngine(base=BASE),)
        if trains:
            regimes.append(Regime(f"{name}-train", "training", BASE, 1))
        host = Host(
            name, engines=engines,
            learner=FakeLearner() if trains else None,
            store=self.store, regimes=tuple(regimes), solo=solo,
            schema_for=lambda base: fake_qwen_schema(4, base=base),
            dial=lambda addr: self.transports[addr])
        self.transports[address] = LocalTransport(HostService(host))
        return host

    def desk(self) -> FleetService:
        return FleetService(
            self.store,
            connect=lambda addr: RemoteHost(self.transports[addr]))

    def split_spec(self):
        """main on one partition, the learner on another — the two-listing
        placement."""
        return arith_spec(self.train, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"),)),
            GpuGroup(gpus(n=1), (learner(),)))))


class FleetServiceTest(DeskFixture):
    def test_one_frame_places_adopts_and_threads_routes(self) -> None:
        """The whole standing path: desk places main onto the serving
        listing, adopts at the training listing with main's address as a
        route, and the run commits on the learner host's loop."""
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        trainer = self.stand_up("train-b", "fleet://b", serves_pool=False,
                                trains=True)
        desk = self.desk()
        desk.list_host("serve-a", serving.regimes, "fleet://a")
        desk.list_host("train-b", trainer.regimes, "fleet://b")

        async def drive():
            reply = await desk.submit(_row(self.split_spec()))
            await trainer._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(reply["accepted"], reply)
        self.assertEqual(reply["host"], "train-b")
        self.assertEqual(reply["pools"],
                         {"main": "serve-a", "learner": "train-b"})
        self.assertEqual(
            trainer.status()["tenants"][reply["run_id"]]["status"], "done")
        # the serving host carried the sampling: its journal, not a guess
        self.assertTrue(any(
            e.get("event") == "place" and e.get("run_id") == reply["run_id"]
            for e in self.store.read_fleet_log()))

    def test_a_desk_run_is_byte_identical_to_an_in_process_submit(self) -> None:
        """The desk adds a desk, never semantics."""
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        trainer = self.stand_up("train-b", "fleet://b", serves_pool=False,
                                trains=True)
        desk = self.desk()
        desk.list_host("serve-a", serving.regimes, "fleet://a")
        desk.list_host("train-b", trainer.regimes, "fleet://b")
        remote = RemoteFleet(LocalTransport(desk))

        async def drive():
            reply = await remote.submit(self.split_spec())
            await trainer._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        other_store, other_train, _ = arith_store(tmp.name)
        plain = Host("plain", engines=(FakeEngine(base=BASE),),
                     learner=FakeLearner(), store=other_store)
        spec = arith_spec(other_train, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"),)),
            GpuGroup(gpus(n=1), (learner(),)))))
        report = go(plain.submit(spec, SCHEMA))
        self.assertEqual(reply["run_id"], report.run_id)
        self.assertEqual(
            self.store.path_of(f"runs/{reply['run_id']}/ledger.jsonl").read_bytes(),
            other_store.path_of(f"runs/{report.run_id}/ledger.jsonl").read_bytes())

    def test_nothing_listed_means_boot_instructions(self) -> None:
        """The standing carve is a venue action: the refusal SAYS what to
        boot instead of placing wrong."""
        desk = self.desk()
        reply = go(desk.submit(_row(self.split_spec())))
        self.assertFalse(reply["accepted"])
        self.assertEqual(len(reply["boot"]), 2)
        self.assertEqual(reply["boot"][0]["base"], BASE)

    def test_solo_occupied_listings_are_skipped(self) -> None:
        """find_listing mirrors the join rung: a solo listing with a running
        tenancy is invisible to placement."""
        busy = self.stand_up("busy", "fleet://busy", serves_pool=True,
                             trains=True, solo=True)
        desk = self.desk()
        desk.list_host("busy", busy.regimes, "fleet://busy", solo=True)

        async def drive():
            first = await desk.submit(_row(arith_spec(self.train)))
            # while the first is still running, the desk must not offer busy
            second = await desk.submit(_row(arith_spec(
                self.train, seeds=Seeds(master=99))))
            await busy._adoptions[first["run_id"]]
            return first, second
        first, second = go(drive())
        self.assertTrue(first["accepted"])
        self.assertFalse(second["accepted"])
        self.assertIn("boot", second)

    def test_the_desk_rebuilds_from_its_own_journal(self) -> None:
        """Kill -9 the desk: a new one redials every listed host from the
        journal and places exactly as the old one would."""
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        trainer = self.stand_up("train-b", "fleet://b", serves_pool=False,
                                trains=True)
        first = self.desk()
        first.list_host("serve-a", serving.regimes, "fleet://a")
        first.list_host("train-b", trainer.regimes, "fleet://b")

        reborn = FleetService.from_journal(
            self.store, connect=lambda addr: RemoteHost(self.transports[addr]))
        self.assertEqual(sorted(reborn.listings), ["serve-a", "train-b"])
        placement, boot = reborn.place_listings(self.split_spec())
        self.assertEqual(boot, [])
        self.assertEqual(placement[None].name, "train-b")
        self.assertEqual(placement["main"].name, "serve-a")

    def test_a_listing_is_never_replaced(self) -> None:
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        desk = self.desk()
        desk.list_host("serve-a", serving.regimes, "fleet://a")
        with self.assertRaises(Exception):
            desk.list_host("serve-a", serving.regimes, "fleet://a2")


def _row(spec) -> dict:
    return json.loads(canonical_json(spec))


if __name__ == "__main__":
    unittest.main()


class CodeSkewTest(DeskFixture):
    def test_a_stale_loss_is_refused_loudly_by_name(self) -> None:
        """The quiet failure killed: a claimed hash differing from the host's
        registry refuses the adoption and NAMES the stale entry."""
        host = self.stand_up("h", "fleet://h", serves_pool=True, trains=True)
        spec = arith_spec(self.train)

        async def drive(code):
            return await host.adopt(_row(spec), None, code)
        refusal = go(drive({"loss:grpo": "0000000000000000"}))
        self.assertFalse(refusal["accepted"])
        self.assertIn("loss:grpo", refusal["error"])
        self.assertIn("skew", refusal["error"])

        from rlstack.registry import code_hashes
        agreed = go(drive(code_hashes(spec)))
        self.assertTrue(agreed["accepted"], agreed)
        unclaimed = go(host.adopt(_row(arith_spec(self.train,
                                                  seeds=Seeds(master=31)))))
        self.assertTrue(unclaimed["accepted"])

    def test_the_desk_relays_the_clients_claim(self) -> None:
        host = self.stand_up("h", "fleet://h", serves_pool=True, trains=True)
        desk = self.desk()
        desk.list_host("h", host.regimes, "fleet://h")
        reply = go(desk.submit(_row(arith_spec(self.train)),
                               code={"loss:grpo": "not-the-real-hash"}))
        self.assertFalse(reply["accepted"])
        self.assertIn("loss:grpo", reply["error"])


class ProvisionTest(DeskFixture):
    def test_the_desk_provisions_what_nothing_serves(self) -> None:
        """The standing carve, desk-owned: an empty desk with a provisioner
        boots hosts for every unit, journals the births, and the run commits —
        one frame in, metal out."""
        booted: list[dict] = []

        def provision(request):
            index = len(booted)
            booted.append(request)
            name, address = f"boot-{index}", f"fleet://boot-{index}"
            trains = any(r["capability"] == "training"
                         for r in request["regimes"])
            host = self.stand_up(name, address, serves_pool=not trains,
                                 trains=trains)
            return name, host.regimes, address, False

        desk = FleetService(
            self.store,
            connect=lambda addr: RemoteHost(self.transports[addr]),
            provision=provision)

        reply = go(desk.submit(_row(self.split_spec())))
        self.assertTrue(reply["accepted"], reply)
        self.assertEqual(len(booted), 2)             # main unit + learner unit
        self.assertEqual(sorted(desk.listings), ["boot-0", "boot-1"])
        events = [e["event"] for e in self.store.read_fleet_log()]
        self.assertEqual(events.count("provision"), 2)
        self.assertEqual(events.count("list"), 2)

    def test_without_a_provisioner_the_boot_instructions_stand(self) -> None:
        desk = self.desk()
        reply = go(desk.submit(_row(self.split_spec())))
        self.assertFalse(reply["accepted"])
        self.assertEqual(len(reply["boot"]), 2)

    def test_metal_survives_the_journal(self) -> None:
        from rlstack.runner.fleet import Metal
        desk = self.desk()
        desk.register_metal(Metal("node-a", "A100-80GB", 2, 80.0))
        reborn = FleetService.from_journal(
            self.store, connect=lambda addr: RemoteHost(self.transports[addr]))
        self.assertEqual(reborn.metal["node-a"].vram_gb, 80.0)


class LivenessVerbTest(DeskFixture):
    def test_the_desk_probes_its_listings(self) -> None:
        living = self.stand_up("alive-a", "fleet://a", serves_pool=True,
                               trains=True)

        class Dead:
            async def call(self, verb, payload):
                raise ConnectionError("gone")

            def ask(self, verb, payload):
                raise ConnectionError("gone")

        self.transports["fleet://dead"] = Dead()
        desk = self.desk()
        desk.list_host("alive-a", living.regimes, "fleet://a")
        desk.list_host("dead-z", living.regimes, "fleet://dead")
        remote = RemoteFleet(LocalTransport(desk))
        self.assertEqual(remote.liveness(),
                         {"alive-a": True, "dead-z": False})


class LivenessTest(DeskFixture):
    def test_a_dead_listing_is_skipped_and_a_delist_survives_rebuild(self) -> None:
        """A listing whose container stopped answering is invisible to
        placement; a delist is journaled, so the rebuilt desk agrees."""
        class Dead:
            async def call(self, verb, payload):
                raise ConnectionError("container gone")

            def ask(self, verb, payload):
                raise ConnectionError("container gone")

        living = self.stand_up("alive-a", "fleet://a", serves_pool=True,
                               trains=True)
        desk = self.desk()
        self.transports["fleet://dead"] = Dead()
        desk.list_host("dead-z", living.regimes, "fleet://dead")
        desk.list_host("alive-a", living.regimes, "fleet://a")

        placement, boot = desk.place_listings(arith_spec(self.train))
        self.assertEqual(boot, [])
        self.assertEqual(placement[None].name, "alive-a")

        desk.delist("dead-z")
        reborn = FleetService.from_journal(
            self.store, connect=lambda addr: RemoteHost(self.transports[addr]))
        self.assertEqual(sorted(reborn.listings), ["alive-a"])
