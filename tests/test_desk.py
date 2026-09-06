"""The standing fleet: Desk (the desk), Listing, RemoteDesk.

The claims under test: a campaign's whole surface is one frame (submit at the
desk → placed over listings → adopted at the ANCHOR host, every other
member's address threaded as routes);
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
import time
import unittest

import json

from common import arith_spec, arith_store, generation_spec
from rlstack import (
    Bundle, FakeEngine, FakeLearner, Topology, Host, HostSpec, Message, Metal,
    PolicySpec, Regime, RemotePool, Role, SamplingSpec, Seeds,
    fake_qwen_schema, learner, pool,
)
from rlstack.runner.host import Partition
from rlstack.spec.canonical import canonical_json
from rlstack.runner.campaign import Campaigns, demands_of, frame_for
from rlstack.runner.desk import (
    IDLE_S, Demand, Desk, DeskError, Listing, MetalService, covers,
    demand_rows, submit_key,
)
from rlstack.runner.remote import (
    DEADLINE_S, DESK_DEFAULT, HostService, LocalTransport, RemoteDesk,
    RemoteHost, RemoteMetal, Unreachable, WrongEpoch, bounded,
    serve_in_process, stamped, stop_serving_in_process, transport_for,
)
from rlstack.runner.residents import (
    Builds, EngineBuild, LearnerBuild, Resident, ResidentBirth,
)

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

    @staticmethod
    def routed(address: str) -> tuple[str, str]:
        """An address split into what it ROUTES to and the INSTANCE it names
        (ADR 0008, F2) — exactly what a real transport does: the `@epoch`
        suffix never reaches the wire, it is stamped into the payload."""
        head, at, epoch = address.rpartition("@")
        return (head, epoch) if at else (address, "")

    class LazyTransport:
        """Resolves the address on EVERY frame — the venue truth
        (MetalTransport looks its handle up lazily), and what lets a test
        kill a container by shadowing its address after it was listed. The
        epoch rides in the payload, as it does on the wire."""

        def __init__(self, fixture: "DeskFixture", address: str) -> None:
            self.fixture, self.address = fixture, address
            self.epoch = DeskFixture.routed(address)[1]

        async def call(self, verb: str, payload: dict, *,
                       deadline_s: float = DEADLINE_S) -> dict:
            return await self.fixture._transport(self.address).call(
                verb, stamped(payload, self.epoch), deadline_s=deadline_s)

        async def ask(self, verb: str, payload: dict, *,
                      deadline_s: float = DEADLINE_S) -> dict:
            return await self.fixture._transport(self.address).ask(
                verb, stamped(payload, self.epoch), deadline_s=deadline_s)

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
            address_of=lambda host_name: f"fleet://carved/{host_name}"
                                          f"@{service.epoch}",
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
        a Dead transport, and revives it by putting the service back. The
        `@epoch` the desk composed routes nowhere and rides the payload."""

        def __init__(self, fixture: "DeskFixture", address: str) -> None:
            self.fixture = fixture
            self.address, self.epoch = DeskFixture.routed(address)

        async def call(self, verb: str, payload: dict, *,
                       deadline_s: float = DEADLINE_S) -> dict:
            return await self.fixture.metal_transports[self.address].call(
                verb, stamped(payload, self.epoch), deadline_s=deadline_s)

        async def ask(self, verb: str, payload: dict, *,
                      deadline_s: float = DEADLINE_S) -> dict:
            return await self.fixture.metal_transports[self.address].ask(
                verb, stamped(payload, self.epoch), deadline_s=deadline_s)

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
                                builds=self.metal_services[name].builds,
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
        self.assertEqual(go(reborn.metal_remotes["fake-metal"].residual()),
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
            async def call(self, verb, payload, *, deadline_s: float = 0.0):
                raise ConnectionError("gone")

            async def ask(self, verb, payload, *, deadline_s: float = 0.0):
                raise ConnectionError("gone")

        self.transports["fleet://dead"] = Dead()
        desk = self.desk()
        desk.list_host("alive-a", living.regimes, "fleet://a")
        desk.list_host("dead-z", living.regimes, "fleet://dead")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
        self.assertEqual(go(remote.liveness()),
                         {"alive-a": True, "dead-z": False})

    def test_the_desk_probes_its_listings(self) -> None:
        living = self.stand_up("alive-a", "fleet://a", serves_pool=True,
                               trains=True)

        class Dead:
            async def call(self, verb, payload, **bounds):
                raise ConnectionError("gone")

            async def ask(self, verb, payload, **bounds):
                raise ConnectionError("gone")

        self.transports["fleet://dead"] = Dead()
        desk = self.desk()
        desk.list_host("alive-a", living.regimes, "fleet://a")
        desk.list_host("dead-z", living.regimes, "fleet://dead")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
        self.assertEqual(go(remote.liveness()),
                         {"alive-a": True, "dead-z": False})
        # the same probe with what each host CARRIES: the roster an observer
        # needs to tell a live tenancy from a dead generation's leftover attach
        pulse = go(remote.pulse())
        self.assertEqual(pulse["dead-z"], {"alive": False, "running": []})
        self.assertTrue(pulse["alive-a"]["alive"])
        self.assertEqual(pulse["alive-a"]["running"], [])

class LivenessTest(DeskFixture):
    def test_a_dead_listing_is_skipped_and_a_delist_survives_rebuild(self) -> None:
        """A listing whose container stopped answering is invisible to
        placement; a delist is journaled, so the rebuilt desk agrees."""
        class Dead:
            async def call(self, verb, payload, *, deadline_s: float = 0.0):
                raise ConnectionError("container gone")

            async def ask(self, verb, payload, *, deadline_s: float = 0.0):
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

        async def call(self, verb, payload, *, deadline_s: float = 0.0):
            raise ConnectionError("gone")

        async def ask(self, verb, payload, *, deadline_s: float = 0.0):
            self.attempts += 1
            raise ConnectionError("gone")

    class Rebooting:
        """Answers after `fail` failures — the container a lazy venue boots
        BECAUSE of the knock."""

        def __init__(self, inner, fail: int) -> None:
            self.inner, self.fail, self.attempts = inner, fail, 0

        async def call(self, verb, payload, *, deadline_s: float = 0.0):
            return await self.inner.call(verb, payload)

        async def ask(self, verb, payload, *, deadline_s: float = 0.0):
            self.attempts += 1
            if self.attempts <= self.fail:
                raise ConnectionError("booting")
            return await self.inner.ask(verb, payload)

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

    def test_a_finished_generation_only_run_is_not_reparked(self) -> None:
        """ADR 0006 Part B, Q8 — the one obligation the shape adds: a run with
        NO LEDGER must read as finished off its sealed rollouts, or the reaper
        would strand and re-deliver it forever once its host died."""
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        desk = self.desk()
        desk.list_host("serve-a", serving.regimes, "fleet://a")

        async def drive():
            reply = await Campaigns(desk).submit(generation_spec(self.train))
            await serving._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(desk.finished(reply["run_id"]))

        self.transports["fleet://a"] = self.Dead()
        verdicts = go(desk.reap(probes=1))
        self.assertEqual(verdicts["listings"], {"serve-a": "reaped"})
        self.assertEqual(verdicts["runs"], {})        # not work, not stranded
        self.assertEqual(desk.parked(), {})
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

    def test_a_generation_only_spec_is_placed_and_runs(self) -> None:
        """ADR 0006 Part B end to end at the desk: a spec with no learner
        member anchors on `main`, places on a SERVE-ONLY listing, and runs
        there as a Generator and nothing else — one sealed rollout per
        planned wave, no ledger, and the desk agrees it is finished."""
        serving = self.stand_up("serve-a", "fleet://a", serves_pool=True,
                                trains=False)
        desk = self.desk()
        desk.list_host("serve-a", serving.regimes, "fleet://a")
        spec = generation_spec(self.train)

        async def drive():
            reply = await Campaigns(desk).submit(spec)
            return reply, await serving._adoptions[reply["run_id"]]

        reply, report = go(drive())
        self.assertTrue(reply["accepted"], reply)
        self.assertEqual(reply["pools"], {"main": "serve-a"})
        self.assertEqual((report.completed, report.extent), (4, "rollout"))

        run = self.store.open_run(report.run_id)
        self.assertEqual(run.read_ledger(), [])
        for index in range(1, 5):
            self.assertTrue(run.read_rollout(index))
        self.assertTrue(desk.finished(report.run_id))


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
        table = go(RemoteDesk(LocalTransport(Campaigns(desk))).placements())
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
            builds=Builds.fakes(engine_sleeps=True).row(), epoch="e2"))
        # the reply also carries the LEASE the desk just opened and the
        # cadence it expects (ADR 0008, Q1): a container learns its heartbeat
        # from the desk rather than from a constant of its own
        self.assertEqual(told, {"registered": "fake-metal", "reaped": [],
                                "retried": {}, "epoch": "e2",
                                "lease_s": 60.0, "heartbeat_s": 20.0})
        self.assertEqual(desk.metal["fake-metal"].vram_gb, 80.0)
        self.assertEqual(desk.status()["metal"]["fake-metal"]["gpu"], "H100")
        # a redeploy's PROPOSAL lands as a `recipe` event of its own (ADR
        # 0007, Q4) — the same event the desk's own door writes
        self.assertTrue(desk.recipe_row("fake-metal")["engine"]["sleeps"])
        reborn = Desk.from_journal(
            self.store, host_for=lambda addr: RemoteHost(self._transport(addr)),
            metal_for=lambda addr: RemoteMetal(self.LazyPlane(self, addr)))
        self.assertEqual(reborn.metal["fake-metal"].vram_gb, 80.0)   # last write wins
        self.assertTrue(reborn.recipe_row("fake-metal")["engine"]["sleeps"])
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
            await desk.observe_idle(1000.0)        # first sight: nothing to compare
            unread = dict(desk.idle_since)
            await desk.observe_idle(1060.0)        # quiet through a whole tick
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
            await desk.observe_idle(10.0)
            await desk.observe_idle(20.0)
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
            await desk.observe_idle(100.0)         # first sight: a reading, no clock
            await desk.observe_idle(200.0)         # quiet: the clock starts
            started = dict(desk.idle_since)
            async for _ in pool.sample_tokens((Message(Role.USER, "2+2?"),),
                                              SamplingSpec(), (), "bundle:x",
                                              seed=7):
                pass
            await desk.observe_idle(300.0)         # the counter moved: not idle
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


class BareMetalTest(DeskFixture):
    """ADR 0007, Q4: THE RECIPE IS THE DESK'S. A metal container boots
    knowing what card it has and what store it mounted; what it should BUILD
    is a declaration, and a declaration belongs where the fleet's other
    declarations live. The claims: a bare metal registers and lists exactly
    like a declared one; a carve to it is refused BY NAME and journaled, at
    the desk and again at the metal; a recipe declared at the desk's own door
    is journaled, rides the next carve, and survives a kill -9."""

    def bare_metal(self, name: str = "fake-metal",
                   devices: int = 2) -> MetalService:
        """A metal service with NO recipe — what `metal_class` builds now."""
        service = MetalService(
            Metal(name, "L4", devices, 24.0), store=self.store,
            spawn=lambda birth: Resident.in_process(
                birth,
                FakeLearner() if birth.regime.capability == "training"
                else FakeEngine(base=birth.regime.base)),
            address_of=lambda host_name: f"fleet://carved/{host_name}",
            schema_for=lambda base: fake_qwen_schema(4, base=base),
            transport_for=lambda address: self._transport(address))
        self.metal_services[name] = service
        self.metal_transports[f"metal://{name}"] = LocalTransport(service)
        return service

    def bare_desk(self) -> Desk:
        """That metal registered, proposing nothing."""
        desk = self.desk()
        desk.register_metal(self.bare_metal().metal,
                            address="metal://fake-metal")
        return desk

    def test_a_bare_metal_registers_and_is_inventory_like_any_other(self) -> None:
        """Registration is the ACQUIRE rung and says nothing about recipes:
        the row is there, the plane is reachable, and `status` shows a null
        recipe — which is precisely the operator's cue to declare one."""
        desk = self.bare_desk()
        row = desk.status()["metal"]["fake-metal"]
        self.assertTrue(row["plane"])
        self.assertIsNone(row["builds"])
        self.assertIsNone(desk.recipe_for("fake-metal"))

    def test_a_carve_to_a_bare_metal_is_refused_by_name_and_journaled(self) -> None:
        """The desk passes the metal OVER rather than commanding a build it
        cannot describe — journaled, so the reason is on the record — and the
        placement falls through to the boot instructions it would give if no
        metal existed at all."""
        desk = self.bare_desk()
        reply = go(Campaigns(desk).submit(self.split_spec()))
        self.assertFalse(reply["accepted"], reply)
        refusals = [e for e in self.store.read_fleet_log()
                    if e.get("event") == "carve-refused"]
        self.assertTrue(refusals)
        self.assertEqual(refusals[0]["metal"], "fake-metal")
        self.assertIn("no recipe", refusals[0]["reason"])
        self.assertEqual(desk.listings, {})

    def test_the_metal_refuses_the_same_carve_from_its_own_side(self) -> None:
        """Both ends hold the rule. A carve request that reaches a bare metal
        carrying no recipe — a desk from an older journal, a hand-built
        request — is refused at the door, never half-built."""
        service = self.bare_metal()
        reply = go(service.carve({
            "regimes": [{"name": "serve", "capability": "inference",
                         "base": BASE, "shape": 1}],
            "base": BASE, "vram_gb": 12.0}))
        self.assertFalse(reply["carved"], reply)
        self.assertIn("BARE", reply["error"])
        self.assertEqual(service.hosts, {})
        self.assertEqual(service.pending, [])       # the booking came back

    def test_a_recipe_declared_at_the_desk_rides_the_next_carve(self) -> None:
        """The door's whole job: declare, and the metal that could not be
        carved on carves. The recipe reaches the container through the carve
        request — the desk never builds — and `describe()` reports it."""
        service = self.bare_metal()
        desk = self.desk()
        desk.register_metal(service.metal, address="metal://fake-metal")
        desk.recipe("fake-metal", Builds.fakes(engine_sleeps=True))

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            await service.hosts[reply["host"]]._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(reply["accepted"], reply)
        self.assertTrue(service.builds.engine.sleeps)
        self.assertTrue(service.describe()["builds"]["engine"]["sleeps"])

    def test_the_recipe_door_writes_through_the_desk(self) -> None:
        """The operator's declaration is a WIRE verb (`deploy/desk.py::
        recipe`), not a second writer beside the journal: the fleet journal
        has one writer and this is one of its events (I10). A row in, the
        declared row back, and the metal that could not be carved on carves."""
        service = self.bare_metal()
        desk = self.desk()
        desk.register_metal(service.metal, address="metal://fake-metal")
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))

        told = go(remote.recipe("fake-metal", Builds.fakes().row()))
        self.assertEqual(told, {"metal": "fake-metal",
                                "builds": Builds.fakes().row()})
        self.assertEqual(desk.recipe_for("fake-metal"), Builds.fakes())
        reply = go(Campaigns(desk).submit(self.split_spec()))
        self.assertTrue(reply["accepted"], reply)

    def test_the_declaration_outlives_the_desk(self) -> None:
        """`recipe` is a journaled event and `from_journal` replays it,
        latest per metal winning — so an operator declares once and a desk
        restarted an hour later carves from the same row."""
        self.bare_metal()
        desk = self.desk()
        desk.register_metal(self.metal_services["fake-metal"].metal,
                            address="metal://fake-metal")
        desk.recipe("fake-metal", Builds.fakes())
        desk.recipe("fake-metal", Builds.fakes(engine_sleeps=True))
        events = [e for e in self.store.read_fleet_log()
                  if e.get("event") == "recipe"]
        self.assertEqual(len(events), 2)
        reborn = self.rebuilt_desk()
        self.assertTrue(reborn.recipe_for("fake-metal").engine.sleeps)


class ServesJoinTest(DeskFixture):
    """ADR 0007, Q4a: the join rule's SECOND half. A spec whose bank names an
    adapter type its would-be host's engine was not built to serve used to be
    joined anyway and refused at Phase 0, on the host, after the placement
    had committed. The desk compares two sets of strings it does not
    interpret, and the refusal happens where a placement can still go
    somewhere else."""

    STEER = ("steer",)

    def listing(self) -> Listing:
        """One listing wearing an inference regime for BASE at shape 1."""
        self.stand_up("serve-a", "fleet://a", serves_pool=True, trains=False)
        desk = self.desk()
        desk.list_host("serve-a", (Regime("serve-a", "inference", BASE, 1),),
                       "fleet://a", metal="fake-metal")
        self.desk_under_test = desk
        return desk.listings["serve-a"]

    def demand(self, *adapter_types: str) -> Demand:
        return Demand(pool="main", capability="inference", base=BASE,
                      shape=1, vram_gb=None, group=0,
                      adapter_types=adapter_types)

    @staticmethod
    def serving(*serves: str) -> Builds:
        return Builds(engine=EngineBuild(serves=serves),
                      learner=LearnerBuild())

    def test_a_recipe_that_does_not_serve_the_type_does_not_cover(self) -> None:
        listing = self.listing()
        self.assertTrue(covers(listing, self.demand(*self.STEER),
                               self.serving("steer", "lora")))
        self.assertFalse(covers(listing, self.demand(*self.STEER),
                                self.serving("lora")))

    def test_the_spec_is_what_names_the_adapter_types(self) -> None:
        """The campaign layer reads the bank; the desk only compares. An
        arithmetic spec's bank is lora, so its demands name lora."""
        demands = demands_of(self.split_spec())
        inference = [d for d in demands if d.capability == "inference"]
        self.assertEqual(inference[0].adapter_types, ("lora",))
        self.assertEqual(demand_rows(demands)[0]["adapter_types"],
                         list(demands[0].adapter_types))

    def test_a_join_the_recipe_refuses_falls_through_to_the_next(self) -> None:
        """The rule is a JOIN rule, so a refusal is not an error: the unit
        simply is not covered here, and find_listing keeps looking."""
        listing = self.listing()
        desk = self.desk_under_test
        desk.recipe("fake-metal", self.serving("lora"))
        self.assertIsNone(go(desk.find_listing((self.demand(*self.STEER),))))
        desk.recipe("fake-metal", self.serving("steer"))
        self.assertEqual(go(desk.find_listing((self.demand(*self.STEER),))),
                         listing)


class LoudKnockTest(DeskFixture):
    """ADR 0007, Q5: a knock with no way to knock is a VENUE bug, and the
    supervision loop degrading into a silent no-op is how it stays hidden."""

    def test_a_knock_with_neither_boot_nor_address_is_journaled(self) -> None:
        desk = self.desk()
        desk.register_metal(Metal("no-plane", "L4", 1, 24.0), address=None)
        self.assertFalse(go(desk.knock("no-plane")))
        refusals = [e for e in self.store.read_fleet_log()
                    if e.get("event") == "knock-refused"]
        self.assertEqual(len(refusals), 1)
        self.assertEqual(refusals[0]["metal"], "no-plane")
        self.assertIn("no boot_for", refusals[0]["reason"])

    def test_a_metal_with_a_plane_still_knocks_silently(self) -> None:
        """The refusal is about having NO way, not about failing: a metal
        with an address is knocked through it and nothing is journaled."""
        self.metal_service(devices=1)
        desk = self.desk_with_metal("fake-metal")
        self.assertTrue(go(desk.knock("fake-metal")))
        self.assertEqual([e for e in self.store.read_fleet_log()
                          if e.get("event") == "knock-refused"], [])


class GuardedReleaseTest(DeskFixture):
    """ADR 0007, Q6 (Samarth's rider): under ONE desk, a venue door that
    tears down the metal it acquired would take every other experiment on
    that metal with it. So an explicit release is guarded exactly as
    decommission is — running work NAMED, nothing torn down — while the
    desk's own idle clock releases unguarded, because its evidence is that
    nothing has been busy there for the metal's whole limit."""

    def test_a_release_over_running_work_is_refused_with_the_run_named(self) -> None:
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            refused = await desk.release("fake-metal", reason="my door")
            stood = "fake-metal" not in desk.released and dict(desk.listings)
            trainer = service.hosts[reply["host"]]
            await trainer._adoptions[reply["run_id"]]
            after = await desk.release("fake-metal", reason="my door")
            return reply, refused, stood, after
        reply, refused, stood, after = go(drive())
        self.assertFalse(refused["released"], refused)
        self.assertIn(reply["run_id"], refused["running"])
        self.assertIn("force", refused["error"])
        self.assertEqual(len(stood), 2)          # nothing came down: two listings
        self.assertTrue(after["released"])       # finished work holds nothing
        self.assertIn("fake-metal", desk.released)

    def test_the_guard_is_the_whole_metal_not_one_host(self) -> None:
        """A release takes every host on the metal at once, so the question
        is asked of the metal: the run's anchor and the serve host it merely
        routes through both count."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            holding = await desk.metal_dependents("fake-metal")
            await service.hosts[reply["host"]]._adoptions[reply["run_id"]]
            return reply, holding
        reply, holding = go(drive())
        self.assertEqual(len(desk.listings), 2)
        self.assertEqual(holding, [reply["run_id"]])

    def test_force_hands_it_back_anyway(self) -> None:
        """The operator's verb, at the desk's own door alone."""
        self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            forced = await desk.release("fake-metal", reason="sweep",
                                        force=True)
            return reply, forced
        reply, forced = go(drive())
        self.assertTrue(forced["released"], forced)
        self.assertIn(reply["run_id"], forced["running"])   # named, not spared
        self.assertIn("fake-metal", desk.released)

    def test_the_idle_rule_releases_unguarded(self) -> None:
        """ADR 0003's promise is not weakened into 'unless a journal row
        still says running': the sweep's own evidence — nothing busy on this
        metal for its whole limit — is stronger than the guard's, so
        `release_idle` releases with force.

        The two readings disagree only where the guard and the clock look at
        different things (a serve listing whose roster is empty while a run
        anchored elsewhere still routes through it), so the clock is set
        here rather than waited out: `idle_since` IS the sweep's memory of
        when the quiet started."""
        self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal", idle_s=600.0)

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            holding = await desk.metal_dependents("fake-metal")
            desk.idle_since["fake-metal"] = 0.0        # the quiet began long ago
            due = await desk.release_idle(1000.0)
            return reply, holding, due
        reply, holding, due = go(drive())
        self.assertEqual(holding, [reply["run_id"]])   # the guard would refuse
        self.assertEqual(due, ["fake-metal"])          # the clock does not
        self.assertIn("fake-metal", desk.released)


class ReleaseTerminatesTheContainerTest(unittest.TestCase):
    """A released metal is not merely deaf but GONE (2026-09-05): the desk
    ends the container the metal registered with, through the hand the
    deploy gave it, and a rebuilt desk still knows which container that
    was."""

    def test_release_terminates_the_registered_container(self) -> None:
        from rlstack import LocalStore
        from rlstack.runner.desk import Desk, Metal

        ended: list[str] = []

        async def end(container: str) -> bool:
            ended.append(container)
            return True

        store = LocalStore(tempfile.mkdtemp())
        desk = Desk(store, host_for=lambda addr: None, terminate_for=end)
        metal = Metal(name="m", gpu="L4", devices=1, vram_gb=24.0)
        desk.register_metal(metal, container="ta-0001")
        told = asyncio.run(desk.release("m"))
        self.assertTrue(told["released"])
        self.assertTrue(told["terminated"])
        self.assertEqual(ended, ["ta-0001"])
        rebuilt = Desk.from_journal(store, host_for=lambda addr: None)
        self.assertEqual(rebuilt.metal_containers, {"m": "ta-0001"})

    def test_a_desk_without_the_hand_releases_and_says_so(self) -> None:
        from rlstack import LocalStore
        from rlstack.runner.desk import Desk, Metal

        desk = Desk(LocalStore(tempfile.mkdtemp()), host_for=lambda addr: None)
        desk.register_metal(
            Metal(name="m", gpu="L4", devices=1, vram_gb=24.0), container="ta-1")
        told = asyncio.run(desk.release("m"))
        self.assertTrue(told["released"])
        self.assertFalse(told["terminated"])


class SoloPlacementTest(DeskFixture):
    """`submit(..., solo=True)` (2026-09-05): the placement joins NO listing
    that stood before it, carves on a metal with nothing standing where one
    is live (never by knocking released metal), and what it carves stays
    joinable — the next submit of the same campaign lands beside it."""

    def test_a_solo_submit_carves_past_a_covering_listing(self) -> None:
        service = self.metal_service(devices=4)
        desk = self.desk_with_metal("fake-metal")
        first = go(Campaigns(desk).submit(self.split_spec()))
        self.assertTrue(first["accepted"], first)
        other = dataclasses.replace(self.split_spec(), seeds=Seeds(master=99))
        second = go(Campaigns(desk).submit(other, solo=True))
        self.assertTrue(second["accepted"], second)
        self.assertNotEqual(second["host"], first["host"])   # carved, not joined
        self.assertEqual(len(desk.listings), 4)
        self.assertFalse(desk.listings[second["host"]].solo)   # the campaign's, joinable
        requests = [e["request"] for e in self.store.read_fleet_log()
                    if e["event"] == "provision"]
        self.assertEqual(len(requests), 4)
        # a plain submit afterwards JOINS (the join rung's first covering listing)
        third = go(Campaigns(desk).submit(
            dataclasses.replace(self.split_spec(), seeds=Seeds(master=7))))
        self.assertTrue(third["accepted"], third)
        self.assertEqual(len(desk.listings), 4)

    def test_a_solo_submit_never_knocks_released_metal(self) -> None:
        """No live metal has room: a plain submit knocks a released metal
        back (ADR 0003 Q4); a solo one does not — a released 32B pair is
        not the fresh card a 0.6B campaign asked for — and lands in boot."""
        self.metal_service(devices=2)
        self.metal_service("spare", devices=2)
        desk = self.desk_with_metal("fake-metal", "spare")
        first = go(Campaigns(desk).submit(self.split_spec()))
        self.assertTrue(first["accepted"], first)             # fills fake-metal
        self.assertTrue(all(l.metal == "fake-metal" for l in desk.listings.values()))
        go(desk.release("spare"))                             # released, knockable
        solo = go(Campaigns(desk).submit(
            dataclasses.replace(self.split_spec(), seeds=Seeds(master=99)), solo=True))
        self.assertFalse(solo["accepted"])
        self.assertEqual(len(solo["boot"]), 2)
        self.assertIn("spare", desk.released)                 # not knocked
        self.assertEqual(len(desk.listings), 2)

    def test_a_solo_carve_prefers_an_empty_metal(self) -> None:
        self.metal_service("a-metal", devices=4)
        self.metal_service("b-metal", devices=2)
        desk = self.desk_with_metal("a-metal", "b-metal")
        first = go(Campaigns(desk).submit(self.split_spec()))
        self.assertEqual(desk.listings[first["host"]].metal, "a-metal")
        solo = go(Campaigns(desk).submit(
            dataclasses.replace(self.split_spec(), seeds=Seeds(master=99)), solo=True))
        self.assertTrue(solo["accepted"], solo)
        self.assertEqual(desk.listings[solo["host"]].metal, "b-metal")
class FakeClock:
    """A clock a test winds by hand — the whole of what makes a 60-second
    lease testable in a millisecond (ADR 0008, promise 1)."""

    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> float:
        self.t += seconds
        return self.t


class LeaseTest(DeskFixture):
    """ADR 0008, F1 — A FACT ABOUT THE FLEET IS TRUE ONLY WHILE ITS LEASE IS
    RENEWED. Every claim here runs on a fake clock: registering opens a lease,
    a heartbeat renews it, a silence longer than the lease delists the thing,
    unplaces it, parks its runs with what they want, and knocks its metal."""

    def leased_desk(self, *names, clock=None, lease_s: float = 60.0) -> Desk:
        clock = clock or FakeClock()
        desk = Desk(
            self.store,
            host_for=lambda addr: RemoteHost(self.LazyTransport(self, addr)),
            metal_for=lambda addr: RemoteMetal(self.LazyPlane(self, addr)),
            idle_s=None, lease_s=lease_s, clock=clock)
        for name in names:
            service = self.metal_services[name]
            desk.register_metal(service.metal, address=f"metal://{name}",
                                builds=service.builds, epoch=service.epoch)
        return desk

    def test_registering_opens_a_lease_and_a_heartbeat_renews_it(self) -> None:
        service = self.metal_service(devices=2)
        clock = FakeClock()
        desk = self.leased_desk("fake-metal", clock=clock)
        self.assertTrue(desk.leased("fake-metal"))
        self.assertEqual(desk.epoch_of("fake-metal"), service.epoch)

        clock.tick(59.0)
        self.assertTrue(desk.leased("fake-metal"))
        told = go(desk.heartbeat("fake-metal", service.epoch, [24.0, 24.0]))
        self.assertTrue(told["heard"], told)
        clock.tick(59.0)
        self.assertTrue(desk.leased("fake-metal"))   # the renewal moved it
        # the residual rides the heartbeat FOR THE ROW (Q3, as amended)
        self.assertEqual(desk.status()["metal"]["fake-metal"]["residual"],
                         [24.0, 24.0])

    def test_a_lapsed_lease_is_not_placeable(self) -> None:
        """The whole of F1's teeth at the carve rung: a metal nobody has
        heard from within its lease is passed over, and one heartbeat brings
        it back — no re-registration, no journal row."""
        self.metal_service(devices=2)
        clock = FakeClock()
        desk = self.leased_desk("fake-metal", clock=clock)
        clock.tick(61.0)
        self.assertFalse(desk.leased("fake-metal"))
        placed = go(Campaigns(desk).submit(self.split_spec()))
        self.assertFalse(placed["accepted"], placed)
        self.assertTrue(placed["boot"])

        go(desk.heartbeat("fake-metal",
                          self.metal_services["fake-metal"].epoch))
        self.assertTrue(desk.leased("fake-metal"))
        again = go(Campaigns(desk).submit(self.split_spec()))
        self.assertTrue(again["accepted"], again)

    def test_a_lapsed_host_is_off_every_listing_within_one_lease(self) -> None:
        """PROMISE 1. A carved host's lease is its metal's — one container,
        one epoch, one heartbeat — so a metal that goes quiet takes its hosts
        off the listings, parks their runs with what those runs WANT, and
        gets knocked, all inside one reaper tick and with no probe at all."""
        service = self.metal_service(devices=2, sample_gate=asyncio.Event())
        clock = FakeClock()
        knocked: list[str] = []
        desk = self.leased_desk("fake-metal", clock=clock)
        desk.boot_for = knocked.append

        reply = go(Campaigns(desk).submit(self.split_spec()))
        self.assertTrue(reply["accepted"], reply)
        self.assertEqual(len(desk.listings), 2)

        clock.tick(61.0)
        told = go(desk.reap())
        self.assertEqual(sorted(set(told["listings"].values())), ["lapsed"])
        self.assertEqual(desk.listings, {})
        self.assertEqual(knocked, ["fake-metal"])

        self.assertEqual(sorted({e["reason"] for e in self.store.read_fleet_log()
                                 if e.get("event") == "delist"}),
                         ["lease lapsed"])
        parked = desk.parked_rows()[reply["run_id"]]
        self.assertTrue(parked["wants"], parked)
        self.assertEqual(sorted(w["capabilities"][0] for w in parked["wants"]),
                         ["inference", "training"])
        self.assertEqual(parked["since"], parked["t"])
        for host in list(service.hosts):
            service.decarve(host)

    def test_a_heartbeat_for_an_unknown_name_or_a_replaced_epoch_is_refused(self) -> None:
        """A heartbeat never opens a lease: a container talking to a desk
        that has forgotten it must REGISTER, and the refusal says so. An
        epoch the desk has already replaced is refused the same way."""
        service = self.metal_service(devices=1)
        desk = self.leased_desk("fake-metal")
        stranger = go(desk.heartbeat("nobody", "e1"))
        self.assertFalse(stranger["heard"])
        self.assertIn("holds no lease", stranger["error"])

        stale = go(desk.heartbeat("fake-metal", "an-older-life"))
        self.assertFalse(stale["heard"])
        self.assertIn("already replaced", stale["error"])
        self.assertEqual(stale["epoch"], service.epoch)

    def test_the_lease_constants_are_journaled_at_registration(self) -> None:
        """Q1: the values are the DESK's, and the record says what the fleet
        was promising when the row was written."""
        self.metal_service(devices=1)
        desk = self.leased_desk("fake-metal", lease_s=45.0)
        row = [e for e in self.store.read_fleet_log()
               if e.get("event") == "metal"][-1]
        self.assertEqual(row["lease_s"], 45.0)
        self.assertEqual(row["heartbeat_s"], 20.0)
        self.assertEqual(row["epoch"], self.metal_services["fake-metal"].epoch)

    def test_a_rebuilt_desk_grants_one_lease_of_grace(self) -> None:
        """A desk that has been alive for a millisecond has heard from
        nobody, so it believes its replayed rows for exactly one lease and
        then disbelieves whatever did not heartbeat inside it."""
        self.metal_service(devices=2)
        clock = FakeClock()
        self.leased_desk("fake-metal", clock=clock)
        clock.tick(10_000.0)                       # hours of journal age
        reborn = Desk.from_journal(
            self.store,
            host_for=lambda addr: RemoteHost(self.LazyTransport(self, addr)),
            metal_for=lambda addr: RemoteMetal(self.LazyPlane(self, addr)),
            clock=clock)
        self.assertTrue(reborn.leased("fake-metal"))
        self.assertEqual(reborn.epoch_of("fake-metal"),
                         self.metal_services["fake-metal"].epoch)
        clock.tick(61.0)
        self.assertFalse(reborn.leased("fake-metal"))

    def test_status_carries_every_row_s_lease(self) -> None:
        self.metal_service(devices=2)
        clock = FakeClock()
        desk = self.leased_desk("fake-metal", clock=clock)
        go(Campaigns(desk).submit(self.split_spec()))
        told = desk.status()
        metal = told["metal"]["fake-metal"]
        self.assertEqual(metal["epoch"],
                         self.metal_services["fake-metal"].epoch)
        self.assertEqual(metal["lease_s"], 60.0)
        self.assertEqual(metal["heard_t"], clock.t)
        self.assertTrue(metal["live"])
        for row in told["listings"].values():
            self.assertEqual(row["epoch"], metal["epoch"])
            self.assertTrue(row["live"])
        clock.tick(61.0)
        self.assertFalse(desk.status()["metal"]["fake-metal"]["live"])


class EpochTest(DeskFixture):
    """ADR 0008, F2 — A FRAME NAMES THE INSTANCE IT MEANS. A name is not an
    instance: a released container, a redeployed venue and a reborn metal all
    answer at the same address, and only the epoch tells them apart."""

    def test_the_metal_door_refuses_a_frame_from_another_life(self) -> None:
        """PROMISE 2, at the plane door. The refusal NAMES both epochs, which
        is what turns 'the desk hung' into 'the desk is talking to a corpse'."""
        service = self.metal_service(devices=1)
        with self.assertRaises(WrongEpoch) as caught:
            go(LocalTransport(service, "an-older-life").ask("residual", {}))
        self.assertIn(service.epoch, str(caught.exception))
        self.assertIn("an-older-life", str(caught.exception))
        # the SAME frame at this life's epoch is served
        self.assertEqual(
            go(LocalTransport(service, service.epoch).ask("residual", {})),
            {"residual": [24.0]})

    def test_a_host_door_refuses_a_frame_from_another_life(self) -> None:
        """A carved host wears its container's epoch, so a proxy held across
        a rebirth fails by name instead of reaching the newborn."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")
        go(Campaigns(desk).submit(self.split_spec()))
        host = next(iter(service.hosts.values()))
        self.assertEqual(host.epoch, service.epoch)
        with self.assertRaises(WrongEpoch):
            go(LocalTransport(HostService(host),
                              "an-older-life").ask("status", {}))
        self.assertTrue(go(
            LocalTransport(HostService(host), service.epoch).ask("status", {})))

    def test_a_frame_naming_no_epoch_is_served(self) -> None:
        """The suffix is optional by design: a registration is exactly the
        frame that cannot name an epoch yet, because it is announcing one."""
        service = self.metal_service(devices=1)
        self.assertEqual(go(LocalTransport(service).ask("residual", {})),
                         {"residual": [24.0]})

    def test_the_epoch_rides_the_address_and_the_factory_reads_it(self) -> None:
        """Q2: the epoch is a suffix on the address the desk already keeps, so
        a journaled address is self-describing about the instance too."""
        service = self.metal_service(devices=1)
        address = f"local://fake-plane@{service.epoch}"
        serve_in_process(address, service)
        self.addCleanup(stop_serving_in_process, address)
        transport = transport_for(address)
        self.assertEqual(transport.epoch, service.epoch)
        self.assertEqual(go(transport.ask("residual", {})),
                         {"residual": [24.0]})

    def test_the_desk_addresses_the_metal_at_the_epoch_it_registered(self) -> None:
        """The plane address stays epoch-free on the row (a re-registration
        must read as one metal turning over, not two deploys colliding), and
        the desk composes the two when it builds the remote."""
        service = self.metal_service(devices=1)
        seen: list[str] = []

        def metal_for(address: str):
            seen.append(address)
            return RemoteMetal(self.LazyPlane(self, address.partition("@")[0]))
        desk = Desk(self.store, host_for=lambda a: RemoteHost(a),
                    metal_for=metal_for)
        desk.register_metal(service.metal, address="metal://fake-metal",
                            epoch=service.epoch)
        self.assertEqual(seen, [f"metal://fake-metal@{service.epoch}"])
        self.assertEqual(desk.metal_addresses["fake-metal"],
                         "metal://fake-metal")

    def test_a_released_container_is_a_lapsed_epoch(self) -> None:
        """The rebirth hazard, as a claim: a released metal stands a FRESH
        service up under a NEW epoch, and every frame minted for the released
        one is refused by name — which is what the deaf-metal hour needed."""
        service = self.metal_service(devices=1)
        was = service.epoch
        service.release()
        reborn = self.metal_service(devices=1)       # the next knock's container
        self.assertNotEqual(reborn.epoch, was)
        with self.assertRaises(WrongEpoch):
            go(LocalTransport(reborn, was).ask("residual", {}))


class SilentTransport:
    """A transport that NEVER ANSWERS — the fakes suite's whole account of an
    unreachable container (ADR 0008, promise 3). It answers no frame and
    raises no error: exactly the shape that wedged the desk for an hour on
    2026-09-04, and exactly what a deadline is for."""

    def __init__(self) -> None:
        self.frames: list[str] = []

    async def call(self, verb: str, payload: dict, *,
                   deadline_s: float = DEADLINE_S) -> dict:
        return await self.silence(verb, deadline_s)

    async def ask(self, verb: str, payload: dict, *,
                  deadline_s: float = DEADLINE_S) -> dict:
        return await self.silence(verb, deadline_s)

    async def silence(self, verb: str, deadline_s: float) -> dict:
        """Nothing, forever — under the deadline the caller passed, exactly
        as every real transport puts its own frames under `bounded`."""
        self.frames.append(verb)
        return await bounded(asyncio.sleep(3600), deadline_s,
                             f"a container that fetches nothing::{verb}")


class BoundedWireTest(DeskFixture):
    """ADR 0008, F3 — EVERY WIRE VERB HAS A DEADLINE, AND NO LOCK IS HELD
    ACROSS THE WIRE. The claims: an expired wait raises `Unreachable` and is
    journaled on the row it was about; placement reads every metal's residual
    live, concurrently, BEFORE its lock, and passes a silent one over; and a
    silent metal does not delay a placement by more than one deadline however
    many silent metals there are."""

    def quick_desk(self, *names, deadline_s: float = 0.05) -> Desk:
        desk = self.desk()
        desk.residual_deadline_s = deadline_s
        desk.probe_deadline_s = deadline_s
        for name in names:
            desk.register_metal(self.metal_services[name].metal,
                                address=f"metal://{name}",
                                builds=self.metal_services[name].builds,
                                epoch=self.metal_services[name].epoch)
        return desk

    def test_an_expired_wait_is_unreachable_and_names_what_it_asked(self) -> None:
        silent = SilentTransport()
        with self.assertRaises(Unreachable) as caught:
            go(RemoteMetal(silent).residual(deadline_s=0.05))
        self.assertIn("0.05", str(caught.exception))
        self.assertEqual(silent.frames, ["residual"])

    def test_a_silent_metal_is_journaled_unreachable_and_passed_over(self) -> None:
        """PROMISE 3. Two metals, one of them a container that fetches
        nothing: the placement reads both, waits one deadline, journals the
        silent one and carves on the other. Before this, the ask was
        unbounded and under the lock, and the submit never returned."""
        self.metal_service(name="living", devices=2)
        self.metal_service(name="deaf", devices=2)
        desk = self.quick_desk("living", "deaf")
        self.metal_transports["metal://deaf"] = SilentTransport()

        reply = go(Campaigns(desk).submit(self.split_spec()))
        self.assertTrue(reply["accepted"], reply)
        self.assertTrue(all(host.startswith("living")
                            for host in reply["pools"].values()), reply)
        missed = [e for e in self.store.read_fleet_log()
                  if e.get("event") == "unreachable"]
        self.assertTrue(missed, "nothing was journaled unreachable")
        self.assertEqual({e["metal"] for e in missed}, {"deaf"})
        self.assertEqual(missed[0]["deadline_s"], 0.05)

    def test_the_residual_read_is_concurrent_and_costs_ONE_deadline(self) -> None:
        """Q3, as amended: every LIVE metal is asked at once, so the worst
        case per submit is one deadline and not one per metal. Three deaf
        metals at a 0.3 s deadline take 0.3 s, not 0.9."""
        for name in ("deaf-a", "deaf-b", "deaf-c"):
            self.metal_service(name=name, devices=2)
        desk = self.quick_desk("deaf-a", "deaf-b", "deaf-c", deadline_s=0.3)
        for name in ("deaf-a", "deaf-b", "deaf-c"):
            self.metal_transports[f"metal://{name}"] = SilentTransport()

        started = time.monotonic()
        reply = go(Campaigns(desk).submit(self.split_spec()))
        waited = time.monotonic() - started
        self.assertFalse(reply["accepted"], reply)      # nowhere to go: boot
        self.assertLess(waited, 0.75, f"the reads serialized: {waited:.2f}s")
        self.assertEqual(
            {e["metal"] for e in self.store.read_fleet_log()
             if e.get("event") == "unreachable"},
            {"deaf-a", "deaf-b", "deaf-c"})

    def test_a_silent_metal_never_holds_the_placement_lock(self) -> None:
        """The wedge, as a claim: one placement waiting on a deaf metal must
        not queue another. Two submits are driven CONCURRENTLY and the one
        with somewhere to go finishes while the other is still waiting out
        its deadline — which could not happen if the read were under the
        lock, and did not happen on 2026-09-04."""
        self.metal_service(name="living", devices=2)
        self.metal_service(name="deaf", devices=2)
        desk = self.quick_desk("living", "deaf", deadline_s=0.6)
        self.metal_transports["metal://deaf"] = SilentTransport()

        async def race():
            slow = asyncio.create_task(desk.read_fleet())
            await asyncio.sleep(0.05)               # the read is in flight
            started = time.monotonic()
            reply = await Campaigns(desk).submit(self.split_spec())
            return reply, time.monotonic() - started, slow
        reply, waited, slow = go(self._finish(race()))
        self.assertTrue(reply["accepted"], reply)
        self.assertLess(waited, 1.4, "a submit queued behind a silent read")

    async def _finish(self, work):
        reply, waited, slow = await work
        await slow                       # let the deaf read expire quietly
        return reply, waited, slow

    def test_the_lock_is_the_only_one_and_covers_no_wire_call(self) -> None:
        """The audit, said as a test: `desk.py` takes exactly one lock, and
        `decide` — the only thing under it — is a pure function of the
        snapshot. Read off the source, because the claim is about the SHAPE
        of the file and not about one execution of it."""
        import inspect

        from rlstack.runner import desk as module

        source = inspect.getsource(module)
        self.assertEqual(source.count("asyncio.Lock()"), 1)
        decided = inspect.getsource(module.Desk.decide) \
            + inspect.getsource(module.Desk.join_rung) \
            + inspect.getsource(module.Desk.carve_rung)
        for wire in ("await ", "async ", "append_fleet_event"):
            self.assertNotIn(wire, decided,
                             f"the placement lock's body reaches {wire!r}")


class IdempotentSubmitTest(DeskFixture):
    """ADR 0008, F4 — EVERY STATE-CHANGING VERB IS IDEMPOTENT BY ITS KEY,
    because every wire is at-least-once. On 2026-09-04 Modal replayed a
    submit input off a container it had shut down, the same run was adopted
    on two metals, and the observer took the dead copy's word for it."""

    def test_the_intent_is_journaled_before_the_placement(self) -> None:
        """The order is the point: a desk that dies between the intent and
        the delivery leaves the attempt on the record."""
        self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")
        go(Campaigns(desk).submit(self.split_spec()))
        kinds = [e["event"] for e in self.store.read_fleet_log()
                 if e["event"] in ("submit-intent", "place")]
        self.assertEqual(kinds[0], "submit-intent")
        intent = next(e for e in self.store.read_fleet_log()
                      if e["event"] == "submit-intent")
        place = [e for e in self.store.read_fleet_log()
                 if e["event"] == "place" and e.get("delivered")][-1]
        self.assertEqual(intent["key"], place["key"])
        self.assertEqual(intent["folder"], "")

    def test_the_same_frame_twice_places_once_and_answers_twice(self) -> None:
        """PROMISE 4, exactly: one placement, two identical replies. The
        second frame is a REPLAY — Modal rescheduling an input off a dead
        container — and it must not adopt the run a second time."""
        gate = asyncio.Event()
        self.addCleanup(gate.set)
        service = self.metal_service(devices=2, sample_gate=gate)
        desk = self.desk_with_metal("fake-metal")
        rows = demand_rows(demands_of(self.split_spec()))
        frame = frame_for(self.split_spec())

        async def twice():
            first = await desk.submit(rows, frame)
            second = await desk.submit(rows, dict(frame))
            # ONE tenancy, on ONE host: the duplicate adoption of 2026-09-04
            # is what this whole rule exists to make impossible
            return first, second, await desk.running_runs()
        first, second, running = go(twice())
        self.assertTrue(first["accepted"], first)
        self.assertEqual(second["run_id"], first["run_id"])
        self.assertEqual(second["host"], first["host"])
        self.assertEqual(second["pools"], first["pools"])
        # ONE placement on the record, and the replay named as one
        delivered = [e for e in self.store.read_fleet_log()
                     if e["event"] == "place" and e.get("delivered")]
        self.assertEqual(len(delivered), 1)
        replays = [e for e in self.store.read_fleet_log()
                   if e["event"] == "submit-replayed"]
        self.assertEqual(len(replays), 1)
        self.assertEqual(replays[0]["run_id"], first["run_id"])
        self.assertEqual(running, {first["run_id"]})
        gate.set()

    def test_a_resubmit_after_the_run_stopped_places_again(self) -> None:
        """The other half of the same rule, and the reason the predicate is
        "still running" and not "has been delivered": resubmitting a spec is
        how a stopped run RESUMES on this fleet, so a delivery whose run is
        no longer running must not be replayed at the caller."""
        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")
        rows = demand_rows(demands_of(self.split_spec()))
        frame = frame_for(self.split_spec())

        async def twice():
            first = await desk.submit(rows, frame)
            await service.hosts[first["host"]]._adoptions[first["run_id"]]
            return first, await desk.submit(rows, dict(frame))
        first, second = go(twice())
        self.assertTrue(second["accepted"], second)
        delivered = [e for e in self.store.read_fleet_log()
                     if e["event"] == "place" and e.get("delivered")]
        self.assertEqual(len(delivered), 2)         # placed again: a resume
        self.assertEqual([e for e in self.store.read_fleet_log()
                          if e["event"] == "submit-replayed"], [])

    def test_a_second_frame_while_the_first_is_in_flight_refuses_loudly(self) -> None:
        """The race the ADR does not name, resolved the way the repo's rule
        says: refuse loudly over waiting silently. The second frame finds an
        intent with no outcome and is told so by name, rather than placing a
        second time or blocking on the first."""
        self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")
        frame = frame_for(self.split_spec())
        key = submit_key(frame)
        desk.journal_intent(key, frame)             # the first, mid-flight

        reply = go(desk.submit(demand_rows(demands_of(self.split_spec())),
                               frame))
        self.assertFalse(reply["accepted"], reply)
        self.assertTrue(reply["in_flight"])
        self.assertIn(key, reply["error"])
        self.assertEqual([e for e in self.store.read_fleet_log()
                          if e["event"] == "place"], [])

    def test_a_missed_placement_closes_its_intent(self) -> None:
        """An attempt that found nowhere to go is OVER: the next frame with
        the same key is a fresh attempt, not a replay of a submission that
        never happened."""
        desk = self.desk()                           # no metal at all
        rows = demand_rows(demands_of(self.split_spec()))
        frame = frame_for(self.split_spec())
        self.assertFalse(go(desk.submit(rows, frame))["accepted"])
        missed = [e for e in self.store.read_fleet_log()
                  if e["event"] == "submit-missed"]
        self.assertEqual(len(missed), 1)
        self.assertEqual(missed[0]["key"], submit_key(frame))
        # and the retry is not refused as a replay
        again = go(desk.submit(rows, dict(frame)))
        self.assertFalse(again.get("in_flight"), again)


class CloseAtHandTest(DeskFixture):
    """ADR 0008, F6 — CONTROL-PLANE WORK RUNS WHERE A LEASED PROCESS ALREADY
    IS. The client for anything pure, the desk for anything that needs the
    store, the metal for anything that needs the engine — never an on-demand
    function, of which the platform scheduled NONE for an hour on
    2026-09-04."""

    def test_plan_bytes_reach_the_cas_through_the_desk(self) -> None:
        """Q5: the row is built on the client and the bytes it hashed go to
        the store through the desk, which has the mount. Content-addressed,
        so the uri the row already carries is the uri the fleet will read —
        and saying it twice says it once."""
        desk = self.desk()
        remote = RemoteDesk(LocalTransport(Campaigns(desk)))
        uri = go(remote.put_plan(b"the plan's bytes"))
        self.assertEqual(uri, self.store.cas_put(b"the plan's bytes"))
        self.assertEqual(go(remote.put_plan(b"the plan's bytes")), uri)
        self.assertEqual(self.store.cas_get(uri), b"the plan's bytes")

    def test_the_client_can_read_what_it_must_build_a_plan_over(self) -> None:
        """`put_plan`'s inverse: a teacher's rollout plan is one group per
        prompt in a task set, so the client has to READ the set to write the
        plan — and it reads a store it has no mount for the same way it
        writes one."""
        remote = RemoteDesk(LocalTransport(Campaigns(self.desk())))
        uri = self.store.cas_put(b'{"id": "p0"}')
        self.assertEqual(go(remote.read_cas(uri)), b'{"id": "p0"}')

    def test_the_measurement_runs_on_the_metal_that_serves_the_pool(self) -> None:
        """The measuring pass moves off an on-demand CPU function and onto
        the metal: the engine, the store and the admission are already there,
        and the frame carries the manifest and the held-out set's uri and
        nothing venue-shaped."""
        from rlstack.data.tasks.base import Task, write_tasks
        from rlstack.runner.measure import Measurement

        service = self.metal_service(devices=2)
        desk = self.desk_with_metal("fake-metal")
        tasks = write_tasks(self.store, [Task(id="h0", prompt="2+2?",
                                              meta={})])

        async def drive():
            reply = await Campaigns(desk).submit(self.split_spec())
            await service.hosts[reply["host"]]._adoptions[reply["run_id"]]
            told = await RemoteMetal(
                self.LazyPlane(self, "metal://fake-metal")).measure(
                    reply["run_id"],
                    Measurement(name="probe", env="single_turn",
                                task_ids=("h0",), samples=1, every=1,
                                post=(), seed=3).manifest(),
                    tasks, BASE, 1)
            return reply, told
        reply, told = go(drive())
        self.assertTrue(told["measured"], told)
        self.assertEqual(
            sorted(self.store.read_measurements(reply["run_id"])), ["probe"])

    def test_a_measurement_on_the_wrong_metal_is_refused_by_name(self) -> None:
        """A placement bug, and it says so: this metal serves no such pool."""
        service = self.metal_service(devices=1)
        with self.assertRaises(DeskError) as caught:
            service.pool_for("some/other-base", 8)
        self.assertIn("serves no", str(caught.exception))
        self.assertIn("wrong metal", str(caught.exception))
