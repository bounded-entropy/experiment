"""The standing fleet: Desk (the desk), Listing, RemoteDesk.

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
import dataclasses
import tempfile
import unittest

import json

from common import arith_spec, arith_store
from rlstack import (
    FakeEngine, FakeLearner, GpuConfig, GpuGroup, Host, Metal, Regime, Seeds,
    fake_qwen_schema, gpus, learner, pool,
)
from rlstack.runner.host import Partition
from rlstack.spec.canonical import canonical_json
from rlstack.runner.campaign import Campaigns, demands_of
from rlstack.runner.desk import (
    Demand, Desk, Listing, MetalService, demand_rows,
)
from rlstack.runner.remote import (
    HostService, LocalTransport, RemoteDesk, RemoteHost, RemoteMetal,
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
        self.metal_services: dict[str, MetalService] = {}
        self.metal_transports: dict[str, LocalTransport] = {}

    def _transport(self, address: str):
        """Every address in this test's little world — hand-listed hosts
        first (so a test may shadow a carved address with a Dead transport),
        then hosts carved on any metal service."""
        if address in self.transports:
            return self.transports[address]
        for service in self.metal_services.values():
            if address in service.services:
                return LocalTransport(service.services[address])
        raise KeyError(address)

    class LazyTransport:
        """Resolves the address on EVERY frame — the venue truth
        (MetalTransport looks its handle up lazily), and what lets a test
        kill a container by shadowing its address after it was listed."""

        def __init__(self, fixture: "DeskFixture", address: str) -> None:
            self.fixture, self.address = fixture, address

        async def call(self, verb: str, payload: dict) -> dict:
            return await self.fixture._transport(self.address).call(
                verb, payload)

        def ask(self, verb: str, payload: dict) -> dict:
            return self.fixture._transport(self.address).ask(verb, payload)

    def metal_service(self, name: str = "fake-metal", devices: int = 2,
                      build_gate=None, broken: bool = False) -> MetalService:
        """One metal container's books on fakes: factories that can be held
        open (build_gate) or broken, for the booking claims."""
        def engine_factory(regime, partition):
            if build_gate is not None:
                build_gate.wait()
            if broken:
                raise RuntimeError("the factory is broken")
            return FakeEngine(base=regime.base)

        service = MetalService(
            Metal(name, "L4", devices, 24.0), store=self.store,
            engine_factory=engine_factory,
            learner_factory=lambda regime, partition: FakeLearner(),
            address_of=lambda host_name: f"fleet://carved/{host_name}",
            schema_for=lambda base: fake_qwen_schema(4, base=base),
            dial=lambda address: self._transport(address))
        self.metal_services[name] = service
        self.metal_transports[f"metal://{name}"] = LocalTransport(service)
        return service

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

    def desk(self) -> Desk:
        return Desk(
            self.store,
            connect=lambda addr: RemoteHost(self.LazyTransport(self, addr)),
            connect_metal=lambda addr: RemoteMetal(
                self.metal_transports[addr]))

    def desk_with_metal(self, *names) -> Desk:
        """A desk with the named metal services registered, plane and all."""
        desk = self.desk()
        for name in names:
            desk.register_metal(self.metal_services[name].metal,
                                address=f"metal://{name}")
        return desk

    def split_spec(self):
        """main on one partition, the learner on another — the two-listing
        placement."""
        return arith_spec(self.train, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"),)),
            GpuGroup(gpus(n=1), (learner(),)))))


class DeskTest(DeskFixture):
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
            reply = await Campaigns(desk).submit(self.split_spec())
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
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))

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
        reply = go(Campaigns(desk).submit(self.split_spec()))
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
            first = await Campaigns(desk).submit(arith_spec(self.train))
            # while the first is still running, the desk must not offer busy
            second = await Campaigns(desk).submit(
                arith_spec(self.train, seeds=Seeds(master=99)))
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

        reborn = Desk.from_journal(
            self.store, connect=lambda addr: RemoteHost(self.transports[addr]))
        self.assertEqual(sorted(reborn.listings), ["serve-a", "train-b"])
        placement, boot = go(reborn.place_listings(demands_of(self.split_spec())))
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

    def test_a_host_phones_home_over_the_wire(self) -> None:
        """A desk in its OWN container: the deploy that booted a host lists
        and delists it through RemoteDesk, journal included — byte-for-byte
        the in-process list_host."""
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        desk = self.desk()
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
        row = {"metal": "node-a", "gpu": "L4", "devices": [0], "memory": 0.4}
        go(remote.list_host("serve-a", serving.regimes, "fleet://a",
                            partition=row, metal="node-a"))
        self.assertEqual(sorted(desk.listings), ["serve-a"])
        reborn = Desk.from_journal(
            self.store, connect=lambda addr: RemoteHost(self.transports[addr]))
        self.assertEqual(sorted(reborn.listings), ["serve-a"])
        self.assertEqual(reborn.listings["serve-a"].regimes,
                         serving.regimes)
        # the capacity VIEW rides the frame, the journal, and the rebuild
        self.assertEqual(reborn.listings["serve-a"].partition, row)
        self.assertEqual(reborn.listings["serve-a"].metal, "node-a")
        self.assertEqual(reborn.status()["listings"]["serve-a"]["partition"],
                         row)
        with self.assertRaises(Exception):        # never replaced, wire or not
            go(remote.list_host("serve-a", serving.regimes, "fleet://a2"))
        go(remote.delist("serve-a"))
        self.assertEqual(desk.listings, {})


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
        spec = arith_spec(self.train)
        reply = go(desk.submit(          # a TAMPERED frame, relayed unread
            demand_rows(demands_of(spec)),
            {"spec": _row(spec), "code": {"loss:grpo": "not-the-real-hash"},
             "subdir": None}))
        self.assertFalse(reply["accepted"])
        self.assertIn("loss:grpo", reply["error"])


class ProvisionTest(DeskFixture):
    def test_the_desk_deduces_then_commands_the_carve(self) -> None:
        """The standing carve, desk-issued: an empty desk with one registered
        metal asks its residual, COMMANDS one carve per unit, lists the born
        hosts with their partition rows, and the run commits — one frame in,
        metal out, the desk still the fleet journal's one writer."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            trainer = service.hosts[reply["host"]]
            await trainer._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(reply["accepted"], reply)
        self.assertEqual(sorted(desk.listings), sorted(service.hosts))
        self.assertEqual(len(service.hosts), 2)      # main unit + learner unit
        listing = desk.listings[reply["host"]]
        self.assertEqual(listing.metal, "fake-metal")
        self.assertEqual(listing.partition["memory"], 1.0)
        self.assertEqual(service.residual(), [0.0, 0.0])
        events = [e["event"] for e in self.store.read_fleet_log()]
        self.assertEqual(events.count("provision"), 2)
        self.assertEqual(events.count("list"), 2)

    def test_what_no_metal_holds_stays_a_boot_instruction(self) -> None:
        """One device, two whole-device units: the first carves, the second
        misses everywhere and lands in `boot` — and the unit that DID carve
        stays listed for the next submit's join rung."""
        self.metal_service(devices=1)
        desk = self.desk_with_metal("fake-metal")
        reply = go(Campaigns(desk).submit(self.split_spec()))
        self.assertFalse(reply["accepted"])
        self.assertEqual(len(reply["boot"]), 1)
        self.assertEqual(len(desk.listings), 1)

    def test_without_metal_the_boot_instructions_stand(self) -> None:
        desk = self.desk()
        reply = go(Campaigns(desk).submit(self.split_spec()))
        self.assertFalse(reply["accepted"])
        self.assertEqual(len(reply["boot"]), 2)

    def test_metal_survives_the_journal(self) -> None:
        desk = self.desk()
        desk.register_metal(Metal("node-a", "A100-80GB", 2, 80.0))
        reborn = Desk.from_journal(
            self.store, connect=lambda addr: RemoteHost(self.transports[addr]))
        self.assertEqual(reborn.metal["node-a"].vram_gb, 80.0)
        self.assertEqual(reborn.metal_remotes, {})   # no address, no plane

    def test_metal_phones_home_and_the_rebuilt_desk_redials_it(self) -> None:
        """The metal container registers its OWN existence over the wire,
        address included — and a desk rebuilt from the journal can deduce
        (residual) against it again."""
        self.metal_service(devices=2)
        desk = self.desk()
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
        go(remote.register_metal("fake-metal", "L4", 2, 24.0,
                                 "metal://fake-metal"))
        self.assertIn("fake-metal", desk.metal_remotes)
        reborn = Desk.from_journal(
            self.store,
            connect=lambda addr: RemoteHost(self._transport(addr)),
            connect_metal=lambda addr: RemoteMetal(
                self.metal_transports[addr]))
        self.assertEqual(reborn.metal["fake-metal"].devices, 2)
        self.assertEqual(reborn.metal_remotes["fake-metal"].residual(),
                         [1.0, 1.0])


class MigrateTest(DeskFixture):
    def test_migrate_warm_forks_onto_the_current_code(self) -> None:
        """The code-refresh pass: a finished run is warm-forked into a NEW
        experiment whose manifest names its parent at the tail version, and
        the child runs to its own commit — one RemoteDesk frame."""
        host = self.stand_up("h", "fleet://h", serves_pool=True, trains=True)
        desk = self.desk()
        desk.list_host("h", host.regimes, "fleet://h")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
        spec = arith_spec(self.train)

        async def drive():
            first = await remote.submit(spec)
            await host._adoptions[first["run_id"]]
            forked = await remote.migrate([first["run_id"]])
            child = forked[first["run_id"]]
            if child.get("run_id") in host._adoptions:
                await host._adoptions[child["run_id"]]
            return first, child
        first, child = go(drive())
        self.assertTrue(child["accepted"], child)
        self.assertNotEqual(child["run_id"], first["run_id"])
        self.assertEqual(child["parent_version"], 4)
        manifest = self.store.peek_manifest(child["run_id"])
        self.assertEqual(manifest["parent"],
                         f"store://{first['run_id']}@4")
        self.assertEqual(len(self.store.peek_ledger(child["run_id"])), 4)
        self.assertTrue(any(
            e.get("event") == "migrate" and e.get("parent") == first["run_id"]
            for e in self.store.read_fleet_log()))

    def fabricate_parent(self, rid: str, spec, committed: int,
                         train_blob: bytes | None = None) -> None:
        """A half-done parent, by direct store writes: manifest, plans,
        `committed` ledger lines, and the tail's blobs."""
        import json as _json

        from common import arith_plan_blobs
        from rlstack.spec.canonical import canonical_json

        run = self.store.open_run(rid, manifest={
            "run_id": rid, "spec": canonical_json(spec)})
        blobs = arith_plan_blobs()
        run.write_plan("train", train_blob or blobs["train"])
        run.write_plan("rollout", blobs["rollout"])
        for update in range(1, committed + 1):
            run.append_ledger({"update": update, "versions": {"pi": update},
                               "bundle_id": f"b{update}"})
        # the fake learner's own payload grammar (fakes.py emit/load)
        run.write_blob("adapters", "pi", committed,
                       b"fake-delta:pi:aaaa5ealed")
        run.write_blob("optim", "pi", committed,
                       f"fake-optim:pi:{committed}:aaaa5ealed".encode())

    def test_remaining_only_slices_to_what_was_left(self) -> None:
        host = self.stand_up("h", "fleet://h", serves_pool=True, trains=True)
        desk = self.desk()
        desk.list_host("h", host.regimes, "fleet://h")
        spec = arith_spec(self.train)
        self.fabricate_parent("aaaa11112222", spec, committed=2)

        async def drive():
            forked = await Campaigns(desk).migrate(["aaaa11112222"], remaining_only=True)
            child = forked["aaaa11112222"]
            if child.get("run_id") in host._adoptions:
                await host._adoptions[child["run_id"]]
            return child
        child = go(drive())
        self.assertTrue(child["accepted"], child)
        self.assertEqual(child["parent_version"], 2)
        # the child ran EXACTLY the two waves the parent never committed
        self.assertEqual(len(self.store.peek_ledger(child["run_id"])), 2)

    def test_fancy_plans_refuse_slicing_but_not_the_pass(self) -> None:
        """One custom-plan run is refused for slicing WITH the reason; the
        well-shaped run in the same pass still forks."""
        from rlstack import RunPlan, WaveRef, encode

        host = self.stand_up("h", "fleet://h", serves_pool=True, trains=True)
        desk = self.desk()
        desk.list_host("h", host.regimes, "fleet://h")
        spec = arith_spec(self.train)
        fancy = encode(RunPlan(tuple(
            WaveRef("store://elsewhere/waves/1") for _ in range(4))))
        self.fabricate_parent("fancy1111111", spec, 2, train_blob=fancy)
        self.fabricate_parent("plain1111111", spec, 2)

        async def drive():
            forked = await Campaigns(desk).migrate(["fancy1111111", "plain1111111"],
                                        remaining_only=True)
            plain = forked["plain1111111"]
            if plain.get("run_id") in host._adoptions:
                await host._adoptions[plain["run_id"]]
            return forked
        forked = go(drive())
        self.assertFalse(forked["fancy1111111"]["accepted"])
        self.assertIn("standard on-policy", forked["fancy1111111"]["error"])
        self.assertTrue(forked["plain1111111"]["accepted"])


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
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
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

        placement, boot = go(desk.place_listings(demands_of(arith_spec(self.train))))
        self.assertEqual(boot, [])
        self.assertEqual(placement[None].name, "alive-a")

        desk.delist("dead-z")
        reborn = Desk.from_journal(
            self.store, connect=lambda addr: RemoteHost(self.transports[addr]))
        self.assertEqual(sorted(reborn.listings), ["alive-a"])


def _inference_request(memory: float) -> dict:
    return {"regimes": [{"name": "main-tp1", "capability": "inference",
                         "base": BASE, "shape": 1}],
            "base": BASE, "memory": memory}


class MetalServiceTest(DeskFixture):
    """The metal's books: booked before built, released on failure, honest
    about hand-built neighbors, freed by decarve."""

    def test_a_carve_books_before_it_builds(self) -> None:
        """The invariant the metal plane exists for: while one carve's build
        is still open, a second that would share its metal REFUSES — the
        fraction is promised the moment the command is accepted, not when
        the engine finally stands."""
        import threading
        gate = threading.Event()
        service = self.metal_service(devices=1, build_gate=gate)

        async def drive():
            first = asyncio.create_task(service.carve(_inference_request(0.6)))
            await asyncio.sleep(0.05)     # first books, enters its build
            second = await service.carve(_inference_request(0.6))
            gate.set()
            return await first, second
        first, second = go(drive())
        self.assertTrue(first["carved"], first)
        self.assertFalse(second["carved"])
        self.assertIn("residual", second)
        self.assertAlmostEqual(service.residual()[0], 0.4)
        self.assertEqual(service.pending, [])
        self.assertIn(first["address"], service.services)

    def test_a_failed_build_releases_its_booking(self) -> None:
        service = self.metal_service(devices=1, broken=True)
        refusal = go(service.carve(_inference_request(0.6)))
        self.assertFalse(refusal["carved"])
        self.assertIn("released", refusal["error"])
        self.assertEqual(service.residual(), [1.0])
        self.assertEqual(service.pending, [])
        self.assertEqual(service.hosts, {})

    def test_hand_built_hosts_share_the_books(self) -> None:
        """adopt_born: a bring_up's own standing host counts into residual
        exactly like a carve's child — and a partitionless host is refused,
        because unaccounted metal is the double-book this class kills."""
        service = self.metal_service(devices=1)
        standing = Host(
            "standing", engines=(FakeEngine(base=BASE),), learner=None,
            store=self.store,
            partition=Partition("fake-metal", (0,), 0.5, "L4"),
            regimes=(Regime("standing-serve", "inference", BASE, 1),))
        service.adopt_born(standing, "fleet://standing")
        self.assertEqual(service.residual(), [0.5])
        self.assertIsNone(service.choose_devices(1, 0.6))
        bare = Host("bare", engines=(), learner=FakeLearner(),
                    store=self.store)
        with self.assertRaises(Exception):
            service.adopt_born(bare, "fleet://bare")

    def test_decarve_frees_and_the_address_stops_answering(self) -> None:
        service = self.metal_service(devices=1)
        born = go(service.carve(_inference_request(0.6)))
        self.assertAlmostEqual(service.residual()[0], 0.4)
        released: list[Host] = []
        service.release = released.append
        reply = go(service.serve("decarve", {"host": born["host"]}))
        self.assertTrue(reply["decarved"], reply)
        self.assertEqual(service.residual(), [1.0])
        self.assertEqual(len(released), 1)        # the venue unmade the metal
        with self.assertRaises(Exception):
            service.service_for(born["address"])


class ReapTest(DeskFixture):
    class Dead:
        def __init__(self) -> None:
            self.attempts = 0

        async def call(self, verb, payload):
            raise ConnectionError("gone")

        def ask(self, verb, payload):
            self.attempts += 1
            raise ConnectionError("gone")

    class Rebooting:
        """Answers after `fail` failures — the container a lazy venue boots
        BECAUSE of the knock."""

        def __init__(self, inner, fail: int) -> None:
            self.inner, self.fail, self.attempts = inner, fail, 0

        async def call(self, verb, payload):
            return await self.inner.call(verb, payload)

        def ask(self, verb, payload):
            self.attempts += 1
            if self.attempts <= self.fail:
                raise ConnectionError("booting")
            return self.inner.ask(verb, payload)

    def test_retries_then_recovers_or_reaps(self) -> None:
        """The three verdicts: a listing that answers is alive; one that
        answers on a retry RECOVERED (the knock was the restart); one silent
        through every retry is reaped — delisted with the reason journaled,
        so the rebuilt desk agrees."""
        living = self.stand_up("well", "fleet://well", serves_pool=True,
                               trains=True)
        self.transports["fleet://reboot"] = self.Rebooting(
            self.transports["fleet://well"], fail=2)
        dead = self.Dead()
        self.transports["fleet://gone"] = dead
        desk = self.desk()
        desk.list_host("well", living.regimes, "fleet://well")
        desk.list_host("reboots", living.regimes, "fleet://reboot")
        desk.list_host("gone", living.regimes, "fleet://gone")

        verdicts = go(desk.reap(probes=3))
        self.assertEqual(verdicts, {"well": "alive", "reboots": "recovered",
                                    "gone": "reaped"})
        self.assertEqual(sorted(desk.listings), ["reboots", "well"])
        self.assertEqual(dead.attempts, 4)           # the probe + 3 retries
        last_delist = [e for e in self.store.read_fleet_log()
                       if e["event"] == "delist"][-1]
        self.assertEqual((last_delist["host"], last_delist["reason"]),
                         ("gone", "reaped"))
        reborn = Desk.from_journal(
            self.store, connect=lambda addr: RemoteHost(self._transport(addr)))
        self.assertEqual(sorted(reborn.listings), ["reboots", "well"])

    def test_a_reaped_carve_frees_its_metal(self) -> None:
        """The whole circle: the desk carved these hosts, their container
        went silent, and the reap DECARVES them at the metal before
        delisting — the fractions are residual again, carvable by the next
        submit."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            await service.hosts[reply["host"]]._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(reply["accepted"], reply)
        self.assertEqual(service.residual(), [0.0, 0.0])

        for listing in desk.listings.values():        # both containers die
            self.transports[listing.address] = self.Dead()
        verdicts = go(desk.reap(probes=1))
        self.assertEqual(set(verdicts.values()), {"reaped"})
        self.assertEqual(desk.listings, {})
        self.assertEqual(service.hosts, {})
        self.assertEqual(service.residual(), [1.0, 1.0])

    def test_reap_rides_the_wire(self) -> None:
        dead = self.Dead()
        self.transports["fleet://gone"] = dead
        living = self.stand_up("well", "fleet://well", serves_pool=True,
                               trains=False)
        desk = self.desk()
        desk.list_host("well", living.regimes, "fleet://well")
        desk.list_host("gone", living.regimes, "fleet://gone")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
        verdicts = go(remote.reap(probes=1))
        self.assertEqual(verdicts, {"well": "alive", "gone": "reaped"})


class BlindDeskTest(DeskFixture):
    def test_place_serves_a_pure_client(self) -> None:
        """Demands in, addresses out, nothing adopted: the evaluator's door.
        The reply names the pool's address and the journal records the
        placement, but no roster anywhere gains a tenancy."""
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        desk = self.desk()
        desk.list_host("serve-a", serving.regimes, "fleet://a")
        reply = go(desk.place((Demand(
            pool="main", capability="inference", base=BASE, shape=1,
            memory=0.2, group=0, sharing="concurrent"),)))
        self.assertTrue(reply["placed"])
        self.assertEqual(reply["pools"], {"main": "fleet://a"})
        self.assertEqual(serving.roster, {})
        self.assertTrue(any(
            e.get("event") == "place" and e.get("delivered") is False
            for e in self.store.read_fleet_log()))

    def test_the_desk_never_reads_the_frame(self) -> None:
        """GIBBERISH in the frame's spec: the desk places and relays it
        untouched, the HOST refuses it, and the desk's reply carries the
        host's own error — proof the desk decoded nothing."""
        host = self.stand_up("h", "fleet://h", serves_pool=True, trains=True)
        desk = self.desk()
        desk.list_host("h", host.regimes, "fleet://h")
        demands = demands_of(arith_spec(self.train))
        reply = go(desk.submit(
            demand_rows(demands),
            {"spec": {"utter": "gibberish"}, "code": None, "subdir": None}))
        self.assertFalse(reply["accepted"])
        self.assertEqual(reply["host"], "h")     # it REACHED the host

    def test_a_delivery_needs_exactly_one_anchor(self) -> None:
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        desk = self.desk()
        desk.list_host("serve-a", serving.regimes, "fleet://a")
        unanchored = tuple(dataclasses.replace(d, anchor=False)
                           for d in demands_of(arith_spec(self.train)))
        reply = go(desk.submit(demand_rows(unanchored),
                               {"spec": {}, "code": None, "subdir": None}))
        self.assertFalse(reply["accepted"])
        self.assertIn("anchor", reply["error"])


class DecommissionTest(DeskFixture):
    def test_a_free_carve_is_torn_down_and_the_metal_reallocates(self) -> None:
        """Carve's inverse end to end: the engine released, the fraction back
        to residual, the listing delisted with the reason journaled — and the
        SAME metal carves again from the freed fraction, which is the whole
        meaning of reallocate."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            trainer = service.hosts[reply["host"]]
            await trainer._adoptions[reply["run_id"]]
            verdicts = {}
            for name in sorted(desk.listings):
                verdicts[name] = await desk.decommission(name)
            reborn = await Campaigns(desk).submit(arith_spec(
                self.train, seeds=Seeds(master=23),
                gpu_config=self.split_spec().gpu_config))
            await service.hosts[reborn["host"]]._adoptions[reborn["run_id"]]
            return verdicts, reborn
        verdicts, reborn = go(drive())
        for verdict in verdicts.values():
            self.assertTrue(verdict["decommissioned"], verdict)
            self.assertTrue(verdict["decarved"])
        self.assertTrue(reborn["accepted"], reborn)   # the freed metal, reused
        reasons = [e.get("reason") for e in self.store.read_fleet_log()
                   if e.get("event") == "delist"]
        self.assertEqual(reasons.count("decommissioned"), 2)

    def test_running_work_refuses_by_name_even_through_routing(self) -> None:
        """The guard is DEPENDENTS, not occupancy: while a run is in flight,
        BOTH its anchor host and the serve host it merely routes through
        refuse the decommission and name the run; once it finishes, both
        yield without force."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            refusals = {}
            for name in sorted(desk.listings):
                refusals[name] = await desk.decommission(name)
            trainer = service.hosts[reply["host"]]
            await trainer._adoptions[reply["run_id"]]
            after = await desk.decommission(sorted(desk.listings)[0])
            return reply, refusals, after
        reply, refusals, after = go(drive())
        for refusal in refusals.values():
            self.assertFalse(refusal["decommissioned"], refusal)
            self.assertIn(reply["run_id"], refusal["running"])
        self.assertTrue(after["decommissioned"])      # done work holds nothing

    def test_force_tears_down_anyway(self) -> None:
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            serve_name = next(n for n in desk.listings
                              if "learner" not in n)
            forced = await desk.decommission(serve_name, force=True)
            return reply, forced
        reply, forced = go(drive())
        self.assertTrue(forced["decommissioned"])
        self.assertIn(reply["run_id"], forced["running"])   # it SAID so

    def test_a_hand_listed_host_only_delists(self) -> None:
        """No metal on the listing means nothing to decarve: the metal was
        never the desk's to touch — bookkeeping retires, hardware stands."""
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        desk = self.desk()
        desk.list_host("serve-a", serving.regimes, "fleet://a")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
        verdict = go(remote.decommission("serve-a"))
        self.assertTrue(verdict["decommissioned"])
        self.assertFalse(verdict["decarved"])
        self.assertEqual(desk.listings, {})
