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
import contextlib
import dataclasses
import io
import tempfile
import unittest

import json

from common import arith_spec, arith_store
from rlstack import (
    Bundle, FakeEngine, FakeLearner, Topology, Host, HostSpec, Message, Metal,
    Regime, RemotePool, Role, SamplingSpec, Seeds, fake_qwen_schema, learner,
    pool,
)
from rlstack.runner.host import Partition
from rlstack.spec.canonical import canonical_json
from rlstack.runner.campaign import Campaigns, demands_of
from rlstack.runner.desk import (
    IDLE_S, Demand, Desk, DeskError, Listing, MetalService, demand_rows,
)
from rlstack.runner.remote import (
    DESK_DEFAULT, HostService, LocalTransport, RemoteDesk, RemoteHost,
    RemoteMetal,
)
from rlstack.runner.residents import Builds, Resident, ResidentBirth

BASE = "Qwen/Qwen3-0.6B"
SCHEMA = fake_qwen_schema(4, base=BASE)


def go(coro):
    return asyncio.run(coro)


class GatedEngine(FakeEngine):
    """A FakeEngine whose sampling waits at a gate: holds a run in its
    RUNNING state deterministically, so a test can stop or reroute it
    mid-flight and then open the gate to let the survivor finish."""

    def __init__(self, gate: asyncio.Event, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gate = gate

    async def sample_tokens(self, messages, sampling, stop, bundle_id, seed,
                            directives=()):
        await self.gate.wait()
        async for event in FakeEngine.sample_tokens(
                self, messages, sampling, stop, bundle_id, seed, directives):
            yield event


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
                      build_gate=None, broken: bool = False,
                      sample_gate: asyncio.Event | None = None) -> MetalService:
        """One metal container's books on fakes: IN-PROCESS residents whose
        birth can be held open (build_gate) or broken, for the booking
        claims — and engines whose sampling waits at `sample_gate`, for the
        stop/reroute ones. Real child processes are test_residents' claim."""
        def spawn(birth: ResidentBirth) -> Resident:
            if build_gate is not None:
                build_gate.wait()
            if broken:
                raise RuntimeError("the factory is broken")
            if birth.regime.capability == "training":
                return Resident.in_process(birth, FakeLearner())
            if sample_gate is not None:
                return Resident.in_process(
                    birth, GatedEngine(sample_gate, base=birth.regime.base))
            return Resident.in_process(birth, FakeEngine(base=birth.regime.base))

        service = MetalService(
            Metal(name, "L4", devices, 24.0), store=self.store,
            builds=Builds.fakes(), spawn=spawn,
            address_of=lambda host_name: f"fleet://carved/{host_name}",
            schema_for=lambda base: fake_qwen_schema(4, base=base),
            transport_for=lambda address: self._transport(address))
        self.metal_services[name] = service
        self.metal_transports[f"metal://{name}"] = LocalTransport(service)
        return service

    def stand_up(self, name: str, address: str, *, serves_pool: bool,
                 trains: bool, solo: bool = False, engine=None) -> Host:
        """One standing host: regimes it wears, a transport it answers on,
        and — for a learner host — the resolver that turns every OTHER
        address in this test's little world."""
        regimes = []
        engines = ()
        if serves_pool:
            regimes.append(Regime(f"{name}-serve", "inference", BASE, 1))
            engines = (engine or FakeEngine(base=BASE),)
        if trains:
            regimes.append(Regime(f"{name}-train", "training", BASE, 1))
        host = Host(
            name, engines=engines,
            learner=FakeLearner() if trains else None,
            store=self.store, regimes=tuple(regimes), solo=solo,
            schema_for=lambda base: fake_qwen_schema(4, base=base),
            transport_for=lambda addr: self.transports[addr])
        self.transports[address] = LocalTransport(HostService(host))
        return host

    class LazyPlane:
        """The metal PLANE resolved on every frame, like LazyTransport: a
        test kills a metal's container by shadowing its plane address with
        a Dead transport, and revives it by putting the service back."""

        def __init__(self, fixture: "DeskFixture", address: str) -> None:
            self.fixture, self.address = fixture, address

        async def call(self, verb: str, payload: dict) -> dict:
            return await self.fixture.metal_transports[self.address].call(
                verb, payload)

        def ask(self, verb: str, payload: dict) -> dict:
            return self.fixture.metal_transports[self.address].ask(verb, payload)

    def desk(self, boot_for=None, idle_s: float | None = IDLE_S) -> Desk:
        return Desk(
            self.store,
            host_for=lambda addr: RemoteHost(self.LazyTransport(self, addr)),
            metal_for=lambda addr: RemoteMetal(self.LazyPlane(self, addr)),
            boot_for=boot_for, idle_s=idle_s)

    def rebuilt_desk(self, idle_s: float | None = IDLE_S) -> Desk:
        """The same desk after a kill -9: its journal, read back."""
        return Desk.from_journal(
            self.store,
            host_for=lambda addr: RemoteHost(self.LazyTransport(self, addr)),
            metal_for=lambda addr: RemoteMetal(self.LazyPlane(self, addr)),
            idle_s=idle_s)

    def desk_with_metal(self, *names, boot_for=None,
                        idle_s: float | None = IDLE_S,
                        declares=DESK_DEFAULT) -> Desk:
        """A desk with the named metal services registered, plane and all.
        `idle_s` is the DESK's own limit; `declares` is the per-metal idle
        limit as a registration declares it (ADR 0003)."""
        desk = self.desk(boot_for, idle_s)
        for name in names:
            desk.register_metal(self.metal_services[name].metal,
                                address=f"metal://{name}",
                                builds=self.metal_services[name].builds.row(),
                                idle_s=declares)
        return desk

    async def container_dies(self, service: MetalService) -> None:
        """The metal's container generation turns over: every adoption task
        on it dies with it (cancelled here — a real preempt takes the
        process), the books come back EMPTY with the carve counter reset,
        and every carved address stops answering. The same MetalService
        object, so a lazily resolved plane transport still reaches it — the
        reborn container, bare."""
        for host in list(service.hosts.values()):
            for run_id in list(host._adoptions):
                await host.stop(run_id)
        service.hosts.clear()
        service.addresses.clear()
        service.services.clear()
        service.carves = 0

    def split_spec(self):
        """main on one partition, the learner on another — the two-listing
        placement (two HostSpecs, two dedicated hosts)."""
        return arith_spec(self.train, topology=Topology(hosts=(
            HostSpec((pool("main"),)), HostSpec((learner(),)))))


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
        spec = arith_spec(other_train, topology=Topology(hosts=(
            HostSpec((pool("main"),)), HostSpec((learner(),)))))
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
        """Kill -9 the desk: a new one resolves every listed host from the
        journal and places exactly as the old one would."""
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        trainer = self.stand_up("train-b", "fleet://b", serves_pool=False,
                                trains=True)
        first = self.desk()
        first.list_host("serve-a", serving.regimes, "fleet://a")
        first.list_host("train-b", trainer.regimes, "fleet://b")

        reborn = Desk.from_journal(
            self.store, host_for=lambda addr: RemoteHost(self.transports[addr]))
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
            self.store, host_for=lambda addr: RemoteHost(self.transports[addr]))
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
        self.assertEqual(listing.partition["memory"], 1.0)   # a whole L4
        self.assertEqual(service.residual(), [0.0, 0.0])       # GB free
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
            self.store, host_for=lambda addr: RemoteHost(self.transports[addr]))
        self.assertEqual(reborn.metal["node-a"].vram_gb, 80.0)
        self.assertEqual(reborn.metal_remotes, {})   # no address, no plane

    def test_metal_phones_home_and_the_rebuilt_desk_resolves_it(self) -> None:
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
            host_for=lambda addr: RemoteHost(self._transport(addr)),
            metal_for=lambda addr: RemoteMetal(
                self.metal_transports[addr]))
        self.assertEqual(reborn.metal["fake-metal"].devices, 2)
        self.assertEqual(reborn.metal_remotes["fake-metal"].residual(),
                         [24.0, 24.0])                         # GB per device


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
            self.store, host_for=lambda addr: RemoteHost(self.transports[addr]))
        self.assertEqual(sorted(reborn.listings), ["alive-a"])


def _inference_request(vram_gb: float | None) -> dict:
    """A carve command as the desk sends it: per-device GB (None = a whole
    device of whatever card the metal holds), no recipe of its own."""
    return {"regimes": [{"name": "main-tp1", "capability": "inference",
                         "base": BASE, "shape": 1}],
            "base": BASE, "vram_gb": vram_gb}


class MetalServiceTest(DeskFixture):
    """The metal's books, in GB: booked before built, released on failure,
    honest about hand-built neighbors, freed by decarve — and the ONE
    crossing to a fraction happens at build, against the measured card."""

    def test_a_carve_books_before_it_builds(self) -> None:
        """The invariant the metal plane exists for: while one carve's build
        is still open, a second that would share its metal REFUSES — the GB
        is promised the moment the command is accepted, not when the engine
        finally stands."""
        import threading
        gate = threading.Event()
        service = self.metal_service(devices=1, build_gate=gate)

        async def drive():
            first = asyncio.create_task(service.carve(_inference_request(14.4)))
            await asyncio.sleep(0.05)     # first books, enters its build
            second = await service.carve(_inference_request(14.4))
            gate.set()
            return await first, second
        first, second = go(drive())
        self.assertTrue(first["carved"], first)
        self.assertFalse(second["carved"])
        self.assertIn("residual", second)
        self.assertAlmostEqual(service.residual()[0], 9.6)     # 24 - 14.4 GB
        self.assertEqual(service.pending, [])
        self.assertIn(first["address"], service.services)
        # the crossing, once, at build: 14.4 GB of a 24 GB L4 is 0.6
        self.assertAlmostEqual(first["partition"]["memory"], 0.6)

    def test_a_failed_build_releases_its_booking(self) -> None:
        service = self.metal_service(devices=1, broken=True)
        refusal = go(service.carve(_inference_request(14.4)))
        self.assertFalse(refusal["carved"])
        self.assertIn("released", refusal["error"])
        self.assertEqual(service.residual(), [24.0])
        self.assertEqual(service.pending, [])
        self.assertEqual(service.hosts, {})

    def test_a_slice_past_one_device_is_the_acquire_rung(self) -> None:
        """More GB than the card holds is not a smaller fraction: the carve
        refuses naming the card, and the crossing itself raises the acquire
        rung by name — never clamps (ADR 0001's promise)."""
        service = self.metal_service(devices=1)
        refusal = go(service.carve(_inference_request(30.0)))
        self.assertFalse(refusal["carved"])
        self.assertIn("holds 24 GB", refusal["error"])
        self.assertEqual(service.residual(), [24.0])
        with self.assertRaises(DeskError) as caught:
            service.build("too-big", (Regime("main-tp1", "inference", BASE, 1),),
                          (0,), 30.0)
        self.assertIn("acquire rung", str(caught.exception))

    def test_a_whole_device_is_the_whole_of_this_card(self) -> None:
        """vram_gb None (Q10) resolves at the metal: on an L4 it is 24 GB,
        the partition's fraction is 1.0, and the device is spoken for."""
        service = self.metal_service(devices=1)
        born = go(service.carve(_inference_request(None)))
        self.assertTrue(born["carved"], born)
        self.assertEqual(born["partition"]["memory"], 1.0)
        self.assertEqual(service.residual(), [0.0])

    def test_hand_built_hosts_share_the_books(self) -> None:
        """adopt_born: a bring_up's own standing host counts into residual
        exactly like a carve's child (its fraction of THIS card, read back
        in GB) — and a partitionless host is refused, because unaccounted
        metal is the double-book this class kills."""
        service = self.metal_service(devices=1)
        standing = Host(
            "standing", engines=(FakeEngine(base=BASE),), learner=None,
            store=self.store,
            partition=Partition("fake-metal", (0,), 0.5, "L4"),
            regimes=(Regime("standing-serve", "inference", BASE, 1),))
        service.adopt_born(standing, "fleet://standing")
        self.assertEqual(service.residual(), [12.0])
        self.assertIsNone(service.choose_devices(1, 14.4))
        bare = Host("bare", engines=(), learner=FakeLearner(),
                    store=self.store)
        with self.assertRaises(Exception):
            service.adopt_born(bare, "fleet://bare")

    def test_decarve_frees_and_the_address_stops_answering(self) -> None:
        service = self.metal_service(devices=1)
        born = go(service.carve(_inference_request(14.4)))
        self.assertAlmostEqual(service.residual()[0], 9.6)
        residents = service.hosts[born["host"]].residents
        self.assertEqual(len(residents), 1)
        reply = go(service.serve("decarve", {"host": born["host"]}))
        self.assertTrue(reply["decarved"], reply)
        self.assertEqual(service.residual(), [24.0])
        # decarve is process teardown now (ADR 0002): the resident was told to
        # stop and its object released — no venue hook unmakes anything
        self.assertTrue(all(r.stopping for r in residents))
        self.assertTrue(residents[0].transport.door.obj.down)
        with self.assertRaises(Exception):
            service.service_for(born["address"])

    def test_release_ends_the_shift_and_is_idempotent(self) -> None:
        """The third metal command (ADR 0003, Q3): `release` over the wire
        takes every resident down the ladder, empties the books, and ENDS
        THE SHIFT — the keepalive awaiting `until_released` returns, so the
        venue reclaims the container. Saying it twice, or to a bare metal,
        is not an error: released is a goal state."""
        service = self.metal_service(devices=1)
        remote = RemoteMetal(LocalTransport(service))

        async def drive():
            born = await service.carve(_inference_request(14.4))
            residents = service.hosts[born["host"]].residents
            shift = asyncio.create_task(service.until_released())
            await asyncio.sleep(0)
            self.assertFalse(shift.done())          # the shift is standing
            told = await remote.release()
            await asyncio.wait_for(shift, 1.0)      # ... and now it is over
            return born, residents, told, await remote.release()
        born, residents, told, again = go(drive())
        self.assertTrue(told["released"], told)
        self.assertTrue(again["released"], again)   # idempotent
        self.assertTrue(all(r.stopping for r in residents))
        self.assertEqual(service.hosts, {})
        self.assertEqual(service.residual(), [24.0])
        with self.assertRaises(Exception):
            service.service_for(born["address"])

    def test_a_bare_metal_releases_too(self) -> None:
        """Nothing carved, nothing to tear down — and the shift still ends,
        because release is about the CONTAINER, not its hosts."""
        service = self.metal_service(devices=1)
        told = go(service.serve("release", {}))
        self.assertEqual(told["teardown"], [])
        self.assertTrue(service.released.is_set())


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
        self.assertEqual(verdicts["listings"],
                         {"well": "alive", "reboots": "recovered",
                          "gone": "reaped"})
        self.assertEqual(verdicts["runs"], {})      # nothing was placed there
        self.assertEqual(sorted(desk.listings), ["reboots", "well"])
        self.assertEqual(dead.attempts, 4)           # the probe + 3 retries
        last_delist = [e for e in self.store.read_fleet_log()
                       if e["event"] == "delist"][-1]
        self.assertEqual((last_delist["host"], last_delist["reason"]),
                         ("gone", "reaped"))
        reborn = Desk.from_journal(
            self.store, host_for=lambda addr: RemoteHost(self._transport(addr)))
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
        self.assertEqual(set(verdicts["listings"].values()), {"reaped"})
        self.assertEqual(desk.listings, {})
        self.assertEqual(service.hosts, {})
        self.assertEqual(service.residual(), [24.0, 24.0])
        # the run had FINISHED before its hosts died: not work, not stranded,
        # not rerouted — and the metal was knocked (its plane still answers)
        self.assertEqual(verdicts["runs"], {})
        self.assertEqual(verdicts["knocked"], {"fake-metal": True})
        self.assertEqual([e for e in self.store.read_fleet_log()
                          if e.get("event") == "parked"], [])

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
        self.assertEqual(verdicts["listings"], {"well": "alive", "gone": "reaped"})


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
            vram_gb=5.0, group=0),)))
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


class AnchorTest(DeskFixture):
    """The anchor is a CHOICE (ADR 0006 Part A): the frame lands where the
    campaign layer says, the learner's address is threaded like any other
    member's, and one standing learner serves runs anchored anywhere."""

    def test_the_frame_lands_on_main_and_the_learner_is_routed(self) -> None:
        """The mirror of the default delivery: anchored on main, the run's
        Trainer sits at the serving host and the LEARNER's address rides in
        the routes — and the run commits over that wire."""
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        trainer = self.stand_up("train-b", "fleet://b", serves_pool=False,
                                trains=True)
        desk = self.desk()
        desk.list_host("serve-a", serving.regimes, "fleet://a")
        desk.list_host("train-b", trainer.regimes, "fleet://b")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec(),
                                                 anchor="main")
            await serving._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(reply["accepted"], reply)
        self.assertEqual(reply["host"], "serve-a")
        self.assertEqual(reply["pools"],
                         {"main": "serve-a", "learner": "train-b"})
        self.assertEqual(
            serving.status()["tenants"][reply["run_id"]]["status"], "done")
        # the tenancy lives at the anchor; the learner host holds only custody
        self.assertEqual(trainer.status()["tenants"], {})
        delivered = [e for e in self.store.read_fleet_log()
                     if e.get("event") == "place" and e.get("delivered")]
        self.assertEqual(
            [row for row in delivered[-1]["demands"] if row["anchor"]],
            [row for row in delivered[-1]["demands"] if row["pool"] == "main"])

    def test_two_anchors_join_one_learner_listing(self) -> None:
        """Part A, promise 3: two runs anchored on two different serving
        hosts train on ONE listed learner. Both commit, and the learner's
        host journals both tenancies attaching and then both leaving — the
        only place a run anchored elsewhere is visible on the metal it
        trains on."""
        gate = asyncio.Event()
        busy = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                             trains=False, solo=True,
                             engine=GatedEngine(gate, base=BASE))
        spare = self.stand_up("serve-c", "fleet://c", serves_pool=True,
                              trains=False)
        trainer = self.stand_up("train-b", "fleet://b", serves_pool=False,
                                trains=True)
        desk = self.desk()
        desk.list_host("serve-a", busy.regimes, "fleet://a", solo=True)
        desk.list_host("serve-c", spare.regimes, "fleet://c")
        desk.list_host("train-b", trainer.regimes, "fleet://b")

        async def drive():
            first = await Campaigns(desk).submit(self.split_spec(),
                                                 anchor="main")
            # while the first holds the solo listing, the second anchors
            # on the OTHER serving host — and joins the same learner
            second = await Campaigns(desk).submit(
                arith_spec(self.train, seeds=Seeds(master=99),
                           topology=self.split_spec().topology),
                anchor="main")
            gate.set()
            await busy._adoptions[first["run_id"]]
            await spare._adoptions[second["run_id"]]
            return first, second
        first, second = go(drive())
        self.assertEqual(first["host"], "serve-a")
        self.assertEqual(second["host"], "serve-c")
        self.assertEqual(first["pools"]["learner"], "train-b")
        self.assertEqual(second["pools"]["learner"], "train-b")
        self.assertEqual(
            busy.status()["tenants"][first["run_id"]]["status"], "done")
        self.assertEqual(
            spare.status()["tenants"][second["run_id"]]["status"], "done")
        custody = [(e["event"], e["run_id"])
                   for e in self.store.read_host_log("train-b")
                   if str(e.get("event", "")).startswith("learner-")]
        self.assertEqual(
            sorted(custody),
            sorted([("learner-attach", first["run_id"]),
                    ("learner-attach", second["run_id"]),
                    ("learner-detach", first["run_id"]),
                    ("learner-detach", second["run_id"])]))

    def test_a_learner_less_spec_is_placed_and_refused_by_the_loop(self) -> None:
        """A spec with no learner member anchors on `main` and PLACES — the
        desk's "exactly one anchor" is satisfied without a learner. Whether
        it then RUNS is ADR 0006 Part B: today the loop refuses it, and that
        refusal is the loop's, not the desk's."""
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        desk = self.desk()
        desk.list_host("serve-a", serving.regimes, "fleet://a")
        generation_only = arith_spec(
            self.train, algo=None,
            topology=Topology(hosts=(HostSpec((pool("main"),)),)))

        async def drive():
            reply = await Campaigns(desk).submit(generation_only)
            with self.assertRaises(NotImplementedError) as caught:
                await serving._adoptions[reply["run_id"]]
            return reply, str(caught.exception)
        # the host prints a dying adoption's traceback to its own stdout
        # (Host.adopt's rule: a silent adoption death is undiagnosable) —
        # here the death is the assertion, so the transcript stays quiet
        with contextlib.redirect_stdout(io.StringIO()):
            reply, refusal = go(drive())
        self.assertTrue(reply["accepted"], reply)
        self.assertEqual(reply["pools"], {"main": "serve-a"})
        self.assertIn("algo is required", refusal)


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
                topology=self.split_spec().topology))
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


class SupersededCarveTest(DeskFixture):
    def test_a_recycled_metals_same_name_carve_reaps_the_corpse(self) -> None:
        """A metal container recycle resets its carve counter, so a fresh
        carve can re-mint a DEAD listing's exact name (observed live). The
        carve's success proves the metal owns the name now, so the desk reaps
        the corpse in place and lists the newborn; a same-name listing that
        still ANSWERS falls through to list_host's refusal — the true
        collision stays loud."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            first = await Campaigns(desk).submit(self.split_spec())
            await service.hosts[first["host"]]._adoptions[first["run_id"]]
            # the container generation turns over: books empty, counter reset,
            # every carved address dead — the same object, so the desk's plane
            # transport still reaches it, exactly as a redeployed cls would
            service.hosts.clear()
            service.addresses.clear()
            service.services.clear()
            service.carves = 0
            reborn = await Campaigns(desk).submit(arith_spec(
                self.train, seeds=Seeds(master=31),
                topology=self.split_spec().topology))
            await service.hosts[reborn["host"]]._adoptions[reborn["run_id"]]
            return first, reborn
        first, reborn = go(drive())
        self.assertTrue(reborn["accepted"], reborn)
        self.assertEqual(sorted(reborn["pools"].values()),
                         sorted(first["pools"].values()))   # same minted names
        reasons = [e.get("reason") for e in self.store.read_fleet_log()
                   if e.get("event") == "delist"]
        self.assertEqual(reasons.count("superseded by a new carve"), 2)


class StopTest(DeskFixture):
    def test_stop_cancels_journals_and_a_resume_completes(self) -> None:
        """The per-tenancy kill: a gated run is stopped mid-flight — the
        reply arrives only after the death is COMPLETE (the roster row
        failed, the detach journaled) — and re-adopting the same run_id
        afterwards is a plain resume that runs to done. A second stop finds
        nothing running and says so instead of raising."""
        gate = asyncio.Event()
        host = Host("both-a",
                    engines=(GatedEngine(gate, base=BASE),),
                    learner=FakeLearner(), store=self.store,
                    regimes=(Regime("serve", "inference", BASE, 1),
                             Regime("train", "training", BASE, 1)),
                    schema_for=lambda base: fake_qwen_schema(4, base=base))
        door = RemoteHost(LocalTransport(HostService(host)))

        async def drive():
            accepted = await door.adopt(arith_spec(self.train))
            rid = accepted["run_id"]
            first = await door.stop(rid)
            again = await door.stop(rid)
            gate.set()
            resumed = await door.adopt(arith_spec(self.train))
            await host._adoptions[rid]
            return accepted, first, again, resumed, rid
        accepted, first, again, resumed, rid = go(drive())
        self.assertTrue(accepted["accepted"], accepted)
        self.assertTrue(first["stopped"], first)
        self.assertEqual(first["state"], "failed")
        self.assertFalse(again["stopped"])           # already dead: the goal
        self.assertTrue(resumed["accepted"], resumed)
        self.assertEqual(host.roster[rid].status, "done")
        detaches = [e["status"] for e in self.store.read_host_log("both-a")
                    if e.get("event") == "detach"]
        self.assertEqual(detaches, ["failed", "done"])


class RerouteTest(DeskFixture):
    def test_decommission_reroutes_running_work_to_fresh_metal(self) -> None:
        """The teardown that moves its tenants: a gated run is mid-flight on
        a carved pair when its learner host is decommissioned with reroute —
        the desk replays its own ARCHIVED delivery onto a fresh carve (the
        avoided listing off the table), the SAME run_id continues there, and
        with the gate opened it runs to done. Restart-is-redial: nothing was
        copied, because the store was the run all along."""
        gate = asyncio.Event()
        service = self.metal_service(devices=3, sample_gate=gate)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            rid = reply["run_id"]
            verdict = await desk.decommission(reply["host"], reroute=True)
            gate.set()
            landed = verdict["rerouted"][rid]["host"]
            await service.hosts[landed]._adoptions[rid]
            return reply, verdict, landed
        reply, verdict, landed = go(drive())
        rid = reply["run_id"]
        self.assertTrue(verdict["decommissioned"], verdict)
        moved = verdict["rerouted"][rid]
        self.assertTrue(moved["rerouted"], moved)
        self.assertEqual(moved["run_id"], rid)       # same identity: a resume
        self.assertNotEqual(landed, reply["host"])
        self.assertNotIn(reply["host"], desk.listings)
        self.assertEqual(service.hosts[landed].roster[rid].status, "done")
        table = RemoteDesk(LocalTransport(Campaigns(desk))).placements()
        self.assertEqual(table[rid]["host"], landed)  # the binding moved
        self.assertIn("demands", table[rid])          # and stayed replayable

    def test_a_reroute_with_nowhere_to_go_leaves_the_run_running(self) -> None:
        """PLACE-FIRST: without park, a reroute that would strand the run
        (nothing else covers it, no metal can hold it) refuses with the boot
        instructions and touches NOTHING — the run keeps running where it is
        and finishes."""
        gate = asyncio.Event()
        service = self.metal_service(devices=2, sample_gate=gate)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            rid = reply["run_id"]
            refusal = await desk.reroute(rid, avoiding=reply["host"])
            still = service.hosts[reply["host"]].roster[rid].status
            gate.set()
            await service.hosts[reply["host"]]._adoptions[rid]
            return reply, refusal, still
        reply, refusal, still = go(drive())
        self.assertFalse(refusal["rerouted"], refusal)
        self.assertTrue(refusal["boot"])
        self.assertEqual(still, "running")           # nothing was stopped
        self.assertEqual(
            service.hosts[reply["host"]].roster[reply["run_id"]].status,
            "done")

    def test_decommission_parks_what_nothing_covers(self) -> None:
        """The vision's other branch, over the wire: the host is coming down
        and no metal can hold its tenant — the tenancy is STOPPED and
        journaled PARKED with the boot instructions (the ask a human
        answers), the teardown proceeds, and a later resubmission of the
        same spec revives the SAME run from the store — onto the metal the
        teardown just freed."""
        gate = asyncio.Event()
        service = self.metal_service(devices=2, sample_gate=gate)
        desk = self.desk_with_metal("fake-metal")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            rid = reply["run_id"]
            verdict = await remote.decommission(reply["host"], reroute=True)
            gate.set()
            revived = await Campaigns(desk).submit(self.split_spec())
            await service.hosts[revived["host"]]._adoptions[rid]
            return reply, verdict, revived
        reply, verdict, revived = go(drive())
        rid = reply["run_id"]
        self.assertTrue(verdict["decommissioned"], verdict)
        parked = verdict["rerouted"][rid]
        self.assertTrue(parked["parked"], parked)
        self.assertTrue(parked["boot"])
        self.assertTrue(parked["stopped"]["stopped"])
        self.assertEqual(revived["run_id"], rid)     # the same run, revived
        self.assertEqual(
            service.hosts[revived["host"]].roster[rid].status, "done")
        parked_events = [e for e in self.store.read_fleet_log()
                         if e.get("event") == "parked"]
        self.assertEqual([e["run_id"] for e in parked_events], [rid])

    def test_an_unarchived_delivery_cannot_be_replayed(self) -> None:
        """Deliveries from before the archive carried rows and frame refuse
        the replay with the cure named — resubmit through the campaign — and
        nothing is stopped on the way out."""
        desk = self.desk()
        self.store.append_fleet_event({
            "event": "place", "t": 0.0, "delivered": True,
            "run_id": "cafe01", "host": "old-host",
            "pools": {"main": "old-host"}, "accepted": True})
        refusal = go(desk.reroute("cafe01"))
        self.assertFalse(refusal["rerouted"])
        self.assertIn("campaign", refusal["error"])


class ReRegistrationTest(DeskFixture):
    """ADR 0001, Q5: a known metal name at the SAME address is the container
    generation turning over — the row updates in place, its corpses are
    reaped by probe, its runs recontinued; at ANOTHER address it is two
    deploys colliding on a name, and refused."""

    def test_same_address_updates_the_row_and_the_journal_replays_it(self) -> None:
        self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
        # the reborn container measured a different card and carries a new
        # recipe (a redeploy): both are the desk's canon from here
        told = go(remote.register_metal(
            "fake-metal", "H100", 2, 80.0, "metal://fake-metal",
            builds=Builds.fakes(engine_sleeps=True).row()))
        self.assertEqual(told, {"registered": "fake-metal", "reaped": [],
                                "retried": {}})
        self.assertEqual(desk.metal["fake-metal"].vram_gb, 80.0)
        self.assertEqual(desk.status()["metal"]["fake-metal"]["gpu"], "H100")
        self.assertTrue(desk.metal_builds["fake-metal"]["engine"]["sleeps"])
        reborn = Desk.from_journal(
            self.store, host_for=lambda addr: RemoteHost(self._transport(addr)),
            metal_for=lambda addr: RemoteMetal(self.LazyPlane(self, addr)))
        self.assertEqual(reborn.metal["fake-metal"].vram_gb, 80.0)   # last write wins
        self.assertTrue(reborn.metal_builds["fake-metal"]["engine"]["sleeps"])
        self.assertEqual(reborn.metal_addresses["fake-metal"], "metal://fake-metal")

    def test_a_different_address_is_a_collision_and_refused(self) -> None:
        self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")
        with self.assertRaises(DeskError) as caught:
            desk.register_metal(Metal("fake-metal", "L4", 2, 24.0),
                                address="metal://elsewhere")
        self.assertIn("colliding", str(caught.exception))
        self.assertEqual(desk.metal_addresses["fake-metal"], "metal://fake-metal")

    def test_the_desks_recipe_rides_the_carve(self) -> None:
        """Q5c: the desk's row is canon. Re-registered with a recipe whose
        engine sleeps, the next carve on this metal builds THAT resident —
        the metal's own constants were only its first declaration — and
        describe() reports what it last built from."""
        service = self.metal_service(devices=2)
        self.assertFalse(service.builds.engine.sleeps)
        desk = self.desk_with_metal("fake-metal")
        go(RemoteDesk(LocalTransport(Campaigns(desk))).register_metal(
            "fake-metal", "L4", 2, 24.0, "metal://fake-metal",
            builds=Builds.fakes(engine_sleeps=True).row()))

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            await service.hosts[reply["host"]]._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(reply["accepted"], reply)
        self.assertTrue(service.builds.engine.sleeps)
        self.assertTrue(service.describe()["builds"]["engine"]["sleeps"])
        # the birth the resident was spawned from carries the desk's recipe
        # (the fixture's in-process fakes ignore it; a child process builds
        # from exactly this record — test_residents' claim)
        serve = service.hosts[reply["pools"]["main"]]
        self.assertTrue(serve.residents[0].birth.build.sleeps)

    def test_a_double_up_is_a_no_op_and_a_rebirth_reaps_the_corpses(self) -> None:
        """Probing is what tells the two apart: living hosts answer and stay
        listed under a second `up`; after the container turns over the same
        frame finds them silent, delists them with the reason journaled, and
        the run they carried is stranded and recontinued on the reborn metal
        — no human step."""
        gate = asyncio.Event()
        service = self.metal_service(devices=2, sample_gate=gate)
        desk = self.desk_with_metal("fake-metal")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            rid = reply["run_id"]
            again = await remote.register_metal(       # a second `up`
                "fake-metal", "L4", 2, 24.0, "metal://fake-metal")
            still_listed = sorted(desk.listings)
            await self.container_dies(service)
            reborn = await remote.register_metal(      # the rebirth
                "fake-metal", "L4", 2, 24.0, "metal://fake-metal")
            gate.set()
            landed = desk.placements()[rid]["host"]
            await service.hosts[landed]._adoptions[rid]
            return reply, again, still_listed, reborn, landed
        reply, again, still_listed, reborn, landed = go(drive())
        rid = reply["run_id"]
        self.assertEqual(again["reaped"], [])
        self.assertEqual(still_listed, sorted(reply["pools"].values()))
        self.assertEqual(sorted(reborn["reaped"]), sorted(reply["pools"].values()))
        self.assertEqual(reborn["retried"], {rid: "rerouted"})
        self.assertEqual(service.hosts[landed].roster[rid].status, "done")
        reasons = [e.get("reason") for e in self.store.read_fleet_log()
                   if e.get("event") == "delist"]
        self.assertEqual(reasons.count("metal re-registered"), 2)
        self.assertEqual(desk.parked(), {})


class SuperviseTest(DeskFixture):
    """ADR 0001, Q5b-Q5d: reap → knock → reroute. A metal's container dies
    with a run on it; the reaper concludes its listings, knocks the metal,
    strands the run (journaled parked — the queue) and retries it: the run
    lands on the reborn metal and finishes under the same run_id. What
    nothing fits stays parked until the next registration event retries it.
    The crash-midway state is on the journal, so a rebuilt desk finishes the
    recontinue without adopting the run twice."""

    def test_reap_knocks_and_recontinues_the_run(self) -> None:
        gate = asyncio.Event()
        service = self.metal_service(devices=2, sample_gate=gate)
        knocked: list[str] = []
        desk = self.desk_with_metal("fake-metal", boot_for=knocked.append)

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            await self.container_dies(service)
            verdicts = await desk.reap(probes=1)
            gate.set()
            landed = desk.placements()[reply["run_id"]]["host"]
            await service.hosts[landed]._adoptions[reply["run_id"]]
            return reply, verdicts, landed
        reply, verdicts, landed = go(drive())
        rid = reply["run_id"]
        self.assertEqual(set(verdicts["listings"].values()), {"reaped"})
        self.assertEqual(verdicts["knocked"], {"fake-metal": True})
        self.assertEqual(knocked, ["fake-metal"])
        self.assertEqual(verdicts["runs"], {rid: "rerouted"})
        self.assertEqual(service.hosts[landed].roster[rid].status, "done")
        self.assertEqual(desk.parked(), {})
        # the record, in order: reaped, stranded (parked), re-placed
        kinds = [(e["event"], e.get("reason") or e.get("delivered"))
                 for e in self.store.read_fleet_log()
                 if e.get("event") in ("delist", "parked")
                 or (e.get("event") == "place" and e.get("run_id") == rid)]
        self.assertEqual(kinds[0], ("place", True))
        self.assertEqual(kinds[1:3], [("delist", "reaped")] * 2)
        self.assertEqual(kinds[3][0], "parked")
        self.assertIn("reaped", kinds[3][1])
        self.assertEqual(kinds[4], ("place", True))

    def test_the_default_knock_is_a_describe_through_the_plane(self) -> None:
        """No boot_for: the reaper knocks the plane address it holds — on
        Modal that call IS the boot. A plane that does not answer is a knock
        that failed, and the run waits parked."""
        gate = asyncio.Event()
        service = self.metal_service(devices=2, sample_gate=gate)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            await self.container_dies(service)
            self.metal_transports["metal://fake-metal"] = ReapTest.Dead()
            verdicts = await desk.reap(probes=1)
            return reply, verdicts
        reply, verdicts = go(drive())
        rid = reply["run_id"]
        self.assertEqual(verdicts["knocked"], {"fake-metal": False})
        self.assertEqual(verdicts["runs"], {rid: "parked"})
        self.assertIn(desk.parked()[rid], reply["pools"].values())

    def test_a_parked_run_is_retried_on_the_next_registration(self) -> None:
        """Nothing fit when the metal died (its plane was gone too); the
        reborn container's own registration is the trigger: the queue is
        retried, the run carves onto it and finishes."""
        gate = asyncio.Event()
        service = self.metal_service(devices=2, sample_gate=gate)
        desk = self.desk_with_metal("fake-metal")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            rid = reply["run_id"]
            await self.container_dies(service)
            self.metal_transports["metal://fake-metal"] = ReapTest.Dead()
            parked = await desk.reap(probes=1)
            self.metal_transports["metal://fake-metal"] = LocalTransport(service)
            told = await remote.register_metal(
                "fake-metal", "L4", 2, 24.0, "metal://fake-metal")
            gate.set()
            landed = desk.placements()[rid]["host"]
            await service.hosts[landed]._adoptions[rid]
            return reply, parked, told, landed
        reply, parked, told, landed = go(drive())
        rid = reply["run_id"]
        self.assertEqual(parked["runs"], {rid: "parked"})
        self.assertEqual(told["retried"], {rid: "rerouted"})
        self.assertEqual(service.hosts[landed].roster[rid].status, "done")
        self.assertEqual(desk.parked(), {})

    def test_crash_midway_leaves_the_intent_on_the_journal(self) -> None:
        """The desk dies between concluding the hosts and moving their run:
        the stranding is already journaled, so a desk rebuilt from the
        journal retries it — once. A second retry finds the new binding in
        placements() and nothing parked; stop_anchored probed and at most one
        roster ever carried the run, so it was adopted exactly once."""
        gate = asyncio.Event()
        service = self.metal_service(devices=2, sample_gate=gate)
        desk = self.desk_with_metal("fake-metal")

        async def crash():
            raise RuntimeError("the desk died here")
        desk.retry_parked = crash            # the crash, after conclude + strand

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            rid = reply["run_id"]
            await self.container_dies(service)
            with self.assertRaises(RuntimeError):
                await desk.reap(probes=1)
            reborn = Desk.from_journal(
                self.store,
                host_for=lambda addr: RemoteHost(self.LazyTransport(self, addr)),
                metal_for=lambda addr: RemoteMetal(self.LazyPlane(self, addr)))
            queued = dict(reborn.parked())
            first = await reborn.retry_parked()
            second = await reborn.retry_parked()
            gate.set()
            landed = reborn.placements()[rid]["host"]
            await service.hosts[landed]._adoptions[rid]
            return reply, queued, first, second, landed, reborn
        reply, queued, first, second, landed, reborn = go(drive())
        rid = reply["run_id"]
        self.assertEqual(sorted(queued), [rid])
        self.assertIn(queued[rid], reply["pools"].values())
        self.assertEqual(first, {rid: "rerouted"})
        self.assertEqual(second, {})
        self.assertEqual(reborn.placements()[rid]["host"], landed)
        carrying = [host.name for host in service.hosts.values()
                    if rid in host._adoptions]
        self.assertEqual(carrying, [landed])                # adopted once
        self.assertEqual(service.hosts[landed].roster[rid].status, "done")

    def test_a_decommission_park_is_retried_on_registration_too(self) -> None:
        """The queue is every parked run, whoever parked it: a decommission
        with nowhere to go parks; the next metal registration revives it."""
        gate = asyncio.Event()
        service = self.metal_service(devices=2, sample_gate=gate)
        desk = self.desk_with_metal("fake-metal")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            rid = reply["run_id"]
            verdict = await remote.decommission(reply["host"], reroute=True)
            told = await remote.register_metal(
                "fake-metal", "L4", 2, 24.0, "metal://fake-metal")
            gate.set()
            landed = desk.placements()[rid]["host"]
            await service.hosts[landed]._adoptions[rid]
            return reply, verdict, told, landed
        reply, verdict, told, landed = go(drive())
        rid = reply["run_id"]
        self.assertTrue(verdict["rerouted"][rid]["parked"], verdict)
        self.assertEqual(told["retried"], {rid: "rerouted"})
        self.assertEqual(service.hosts[landed].roster[rid].status, "done")


class IdleReleaseTest(DeskFixture):
    """ADR 0003: metal nothing has run on for `idle_s` is RELEASED — the
    acquire rung inverted, and automatic because the metal is already owned.
    The evidence is each listing's own status (no running tenancy, nothing in
    flight, an `admitted` counter that did not move since the previous tick),
    the clock is the desk's memory, and the door back is a knock."""

    def carved_and_finished(self, desk: Desk, service: MetalService) -> dict:
        """One submit through the desk: two hosts carved on the metal and the
        run finished — the state an idle sweep meets."""
        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            await service.hosts[reply["host"]]._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(reply["accepted"], reply)
        return reply

    def test_idle_metal_is_released_once_the_limit_passes(self) -> None:
        """The whole rule on a fake clock: the first observation only takes a
        reading, the second starts the clock, and the metal comes due one
        limit later — listings delisted, residents down, the row kept as
        inventory that says released."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal", idle_s=600.0)
        self.carved_and_finished(desk, service)

        async def sweep():
            desk.observe_idle(1000.0)        # first sight: nothing to compare
            unread = dict(desk.idle_since)
            desk.observe_idle(1060.0)        # quiet through a whole tick
            started = dict(desk.idle_since)
            early = await desk.release_idle(1600.0)      # 540 s < 600
            late = await desk.release_idle(1661.0)       # 601 s >= 600
            return unread, started, early, late
        unread, started, early, late = go(sweep())
        self.assertEqual(unread, {})
        self.assertEqual(started, {"fake-metal": 1060.0})
        self.assertEqual(early, [])
        self.assertEqual(late, ["fake-metal"])
        self.assertEqual(desk.listings, {})
        self.assertEqual(desk.metal_remotes, {})
        self.assertIn("fake-metal", desk.metal)          # inventory, still
        self.assertTrue(desk.status()["metal"]["fake-metal"]["released"])
        self.assertEqual(desk.status()["metal"]["fake-metal"]["idle_s"], 600.0)
        # the metal obeyed: residents down, books empty, shift over
        self.assertEqual(service.hosts, {})
        self.assertTrue(service.released.is_set())

    def test_a_release_is_journaled_and_a_rebuilt_desk_agrees(self) -> None:
        """The departure is on the record — the intent first, then each
        delist — so a desk rebuilt from the journal knows the metal is
        inventory it may not carve. Saying it twice changes nothing."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")
        self.carved_and_finished(desk, service)
        hosts = sorted(desk.listings)

        told = go(desk.release("fake-metal", reason="released: idle"))
        self.assertEqual((told["listings"], told["told"]), (hosts, True))
        event = [e for e in self.store.read_fleet_log()
                 if e["event"] == "release"][-1]
        self.assertEqual((event["metal"], event["listings"], event["idle_s"]),
                         ("fake-metal", hosts, IDLE_S))
        self.assertEqual([e["reason"] for e in self.store.read_fleet_log()
                          if e["event"] == "delist"],
                         ["released: idle"] * len(hosts))
        again = go(desk.release("fake-metal"))           # idempotent
        self.assertEqual((again["listings"], again["told"]), ([], False))

        reborn = self.rebuilt_desk()
        self.assertEqual(reborn.listings, {})
        self.assertEqual(reborn.released, {"fake-metal"})
        self.assertEqual(reborn.metal_remotes, {})
        self.assertIn("fake-metal", reborn.metal)

    def test_a_pinned_metal_is_never_released(self) -> None:
        """`idle_s=None` at registration PINS the metal: the clock still runs
        (the desk observes everything) and nothing ever comes due — and the
        declaration survives the journal."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal", idle_s=1.0, declares=None)
        self.carved_and_finished(desk, service)

        async def sweep():
            desk.observe_idle(10.0)
            desk.observe_idle(20.0)
            return await desk.release_idle(1e9)
        self.assertEqual(go(sweep()), [])
        self.assertEqual(desk.idle_since, {"fake-metal": 20.0})
        self.assertIsNone(desk.idle_limit("fake-metal"))
        self.assertEqual(desk.released, set())
        self.assertEqual(sorted(desk.listings), sorted(service.hosts))
        self.assertIsNone(self.rebuilt_desk().idle_limit("fake-metal"))

    def test_a_pure_clients_traffic_keeps_its_metal_alive(self) -> None:
        """A measurement cron holds NO tenancy: its host's roster is empty at
        every tick, so occupancy alone would release the metal out from under
        its own sampling. The `admitted` counter is what says work passed."""
        service = self.metal_service(devices=1)
        desk = self.desk_with_metal("fake-metal", idle_s=60.0)
        demand = Demand(pool="main", capability="inference", base=BASE,
                        shape=1, vram_gb=14.4, group=0)

        async def drive():
            placed = await desk.place([demand])
            pool = RemotePool(self.LazyTransport(self, placed["pools"]["main"]),
                              base=BASE, tp=1)
            pool.add_bundle(Bundle("bundle:x", {"pi": 0}))
            desk.observe_idle(100.0)         # first sight: a reading, no clock
            desk.observe_idle(200.0)         # quiet: the clock starts
            started = dict(desk.idle_since)
            async for _ in pool.sample_tokens((Message(Role.USER, "2+2?"),),
                                              SamplingSpec(), (), "bundle:x",
                                              seed=7):
                pass
            desk.observe_idle(300.0)         # the counter moved: not idle
            return placed, started, dict(desk.idle_since), \
                await desk.release_idle(1e9)
        placed, started, after, released = go(drive())
        self.assertTrue(placed["placed"], placed)
        self.assertEqual(sorted(started), ["fake-metal"])
        self.assertEqual(after, {})
        self.assertEqual(released, [])
        self.assertEqual(sorted(desk.listings), sorted(service.hosts))
        # ... and the roster was empty throughout: no tenancy, pure traffic
        carved = next(iter(service.hosts.values()))
        self.assertEqual(carved.status()["tenants"], {})
        self.assertGreater(carved.status()["admitted"], 0)

    def test_the_reapers_tick_sweeps_for_idle_metal(self) -> None:
        """The sweep rides the reaper's own schedule, first: at a limit of
        zero the first tick takes the reading and the second releases —
        before the probing, so the reaper never chases a listing the desk
        has just taken down."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal", idle_s=0.0)
        self.carved_and_finished(desk, service)

        first = go(desk.reap(probes=1))
        second = go(desk.reap(probes=1))
        self.assertEqual(first["released"], [])
        self.assertEqual(set(first["listings"].values()), {"alive"})
        self.assertEqual(second["released"], ["fake-metal"])
        self.assertEqual(second["listings"], {})
        self.assertEqual(second["knocked"], {})
        self.assertEqual(desk.listings, {})
        self.assertEqual(service.hosts, {})

    def test_the_reaper_leaves_a_released_metal_parked(self) -> None:
        """A released metal is PARKED, not silent. A host that listed itself
        just as the release landed is reaped like any other silent listing —
        but its metal is NOT knocked back to life behind the desk's own
        decision."""
        service = self.metal_service(devices=1)
        living = self.stand_up("well", "fleet://well", serves_pool=True,
                               trains=True)
        desk = self.desk_with_metal("fake-metal")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
        told = go(remote.release("fake-metal", reason="by hand"))
        self.assertEqual((told["released"], told["metal"], told["told"]),
                         (True, "fake-metal", True))
        self.transports["fleet://ghost"] = ReapTest.Dead()
        desk.list_host("ghost", living.regimes, "fleet://ghost",
                       metal="fake-metal")

        verdicts = go(desk.reap(probes=1))
        self.assertEqual(verdicts["listings"], {"ghost": "reaped"})
        self.assertEqual(verdicts["knocked"], {})
        self.assertEqual(verdicts["released"], [])
        self.assertEqual(desk.released, {"fake-metal"})

    def test_a_registration_re_acquires_released_metal(self) -> None:
        """The reborn container's own announce is one door back: the row is
        carve-able again and the release is superseded on replay."""
        self.metal_service(devices=1)
        desk = self.desk_with_metal("fake-metal")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
        go(remote.release("fake-metal", reason="by hand"))
        self.assertEqual(desk.released, {"fake-metal"})

        told = go(remote.register_metal("fake-metal", "L4", 1, 24.0,
                                        "metal://fake-metal"))
        self.assertEqual(told["registered"], "fake-metal")
        self.assertEqual(desk.released, set())
        self.assertIn("fake-metal", desk.metal_remotes)
        reborn = self.rebuilt_desk()
        self.assertEqual(reborn.released, set())
        self.assertIn("fake-metal", reborn.metal_remotes)

    def test_a_placement_knocks_a_released_metal_awake(self) -> None:
        """Q4, the other door: nothing carve-able holds the unit but the
        fleet OWNS metal it released, so the desk KNOCKS — the venue boots a
        bare container, the desk puts the row back itself where the
        container's own announce has not landed yet, and the carve proceeds
        in the same breath. No human anywhere."""
        service = self.metal_service(devices=2)
        booted: list[str] = []

        def boot(name: str) -> None:
            """The venue's boot: a fresh container, bare, answering again."""
            booted.append(name)
            service.released.clear()
            self.metal_transports[f"metal://{name}"] = LocalTransport(service)

        desk = self.desk_with_metal("fake-metal", boot_for=boot)
        go(desk.release("fake-metal", reason="by hand"))
        self.metal_transports["metal://fake-metal"] = ReapTest.Dead()

        reply = self.carved_and_finished(desk, service)
        self.assertEqual(booted, ["fake-metal"])
        self.assertEqual(desk.released, set())
        self.assertIn("fake-metal", desk.metal_remotes)
        self.assertEqual(sorted(desk.listings), sorted(service.hosts))
        self.assertEqual(service.hosts[reply["host"]].roster[
            reply["run_id"]].status, "done")

    def test_the_default_knock_wakes_a_released_metal_through_its_plane(self) -> None:
        """No `boot_for`: the knock is a `describe()` through the plane
        ADDRESS the release never forgot — on Modal that call is the boot.
        The desk re-registers the row itself, because the reborn container's
        own announce need not have landed for the carve to proceed."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")
        go(desk.release("fake-metal", reason="by hand"))
        service.released.clear()                  # the container comes back
        self.assertEqual(desk.metal_remotes, {})

        reply = self.carved_and_finished(desk, service)
        self.assertEqual(desk.released, set())
        self.assertIn("fake-metal", desk.metal_remotes)
        self.assertEqual(sorted(desk.listings), sorted(service.hosts))
        self.assertEqual(service.hosts[reply["host"]].roster[
            reply["run_id"]].status, "done")

    def test_what_no_metal_can_hold_stays_a_boot_instruction(self) -> None:
        """The knock is for metal that could hold the unit: one whose
        recorded facts cannot is left released, and the standing acquire
        stays a human's."""
        service = self.metal_service(devices=1)
        desk = self.desk_with_metal("fake-metal")
        go(desk.release("fake-metal", reason="by hand"))
        unit = (Demand(pool="main", capability="inference", base=BASE,
                       shape=4, vram_gb=None, group=0),)
        self.assertFalse(desk.could_hold(unit, service.metal))
        self.assertFalse(go(desk.knock_released(unit)))
        self.assertEqual(desk.released, {"fake-metal"})


class MeasureTest(unittest.TestCase):
    def test_measure_refuses_where_no_cuda_device_is(self) -> None:
        """Q6: a card is measured, never declared — and a machine that has
        none to measure says so by name instead of guessing one."""
        try:
            import torch
            present = torch.cuda.is_available()
        except ImportError:
            present = False
        if present:
            self.skipTest("a CUDA device is present: measure() would succeed")
        with self.assertRaises(DeskError) as caught:
            MetalService.measure("laptop")
        self.assertIn("measured, never declared", str(caught.exception))
