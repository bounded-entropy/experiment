"""A resident is a process (ADR 0002).

Claims under test: the learner's five verbs cross a wire as JSON frames and a
run driven through them is byte-identical to the in-process run (the wire
adds custody, never semantics); a host carved on a metal is born as REAL
child processes — one per regime, each reporting its hello — and the run it
adopts matches the in-process run byte for byte; a resident that dies
unbidden takes its host with it (decarved, booking freed, address gone); a
birth that fails inside the child is reported through the hello and releases
its booking; the alternation hooks cross the door for every resident whose
hello says it sleeps; the pin is asserted; and the teardown ladder — now the
residents module's — walks its rungs on a child that ignores the polite word.

Real children are spent where the claim is about processes and nowhere else:
the in-process door (Resident.in_process) is the same frames JSON-round-
tripped, which is what makes the two paths one path.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import signal
import tempfile
import time
import unittest

from common import arith_spec, arith_store
from rlstack import (
    Builds, Emitted, EntryInstall, FakeEngine, FakeEngineBuild, FakeLearner,
    FakeLearnerBuild, Topology, HostSpec, Host, LearnerService,
    LocalTransport, Metal, OptimSettings, Parameterization, Partition, Regime,
    RemoteLearner, RemotePool, Resident, ResidentBirth, ResidentError,
    SiteMeta, TokenBatch, fake_qwen_schema, learner, pool,
    run_experiment,
)
from rlstack.runner.campaign import Campaigns
from rlstack.runner.desk import Desk, MetalService
from rlstack.runner.remote import (
    DEADLINE_S, RemoteHost, RemoteMetal, decode_emitted, decode_parameterization,
    decode_token_batch, encode_emitted, encode_parameterization,
    encode_token_batch, json_roundtrip,
)
from rlstack.runner.residents import Teardown, check_devices_seen, escalate

BASE = "Qwen/Qwen3-0.6B"
SCHEMA = fake_qwen_schema(4, base=BASE)


def go(coro):
    return asyncio.run(coro)


def alternating_spec(train_uri: str):
    """One HostSpec wearing both regimes — the alternating shape, one carve."""
    return arith_spec(train_uri, topology=Topology(hosts=(
        HostSpec((pool("main"), learner())),)))


# ---- the stub children for the ladder (module-level: spawned by name) ------

def leaves_at_once(ready) -> None:
    ready.set()


def ignores_sigterm(ready) -> None:
    """A child wedged where the polite signal cannot reach it."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(3600)


# ---------------------------------------------------------------------------
# the codecs: every record that crosses the learner's door survives JSON
# ---------------------------------------------------------------------------

class CodecTest(unittest.TestCase):
    def test_a_token_batch_survives_json(self) -> None:
        batch = TokenBatch(
            token_ids=(1, 2, 3, 4), loss_mask=(0, 1, 1, 1),
            behavior_logprobs=(-0.5, -1.25, -0.125, -3.0),
            segment_ids=(0, 0, 1, 1), doc_starts=(0, 2),
            postdata={"advantage": (0.5, 0.5, -0.5, -0.5),
                      "reward": (1, 1, 0, 0)},
            token_extras={"adapter_draw": (0, 1, 0, 1)},
            doc_turn_extras=(({"latent": [0.1, 0.2]},), ({"latent": [0.3]},)),
            microbatches_in_update=3)
        back = decode_token_batch(json_roundtrip(encode_token_batch(batch)))
        self.assertEqual(back, batch)
        # ints stay ints and floats stay floats: a digest over postdata reads
        # the same bytes on both sides of the door
        self.assertIsInstance(back.postdata["reward"][0], int)
        self.assertIsInstance(back.postdata["advantage"][0], float)

    def test_a_parameterization_survives_json(self) -> None:
        sites = (SiteMeta(name="q", path="model.layers.0.self_attn.q_proj",
                          has_weight=True, shape=(8, 8), is_boundary=False),)
        p = Parameterization(
            base=BASE, loss="grpo",
            entries=(EntryInstall(name="pi", adapter_type="lora",
                                  init={"rank": 4, "seed": 2 ** 60 + 7},
                                  trainable=True, sites=sites),),
            optim=OptimSettings(name="adamw", lr=1e-4, betas=(0.9, 0.95),
                                weight_decay=0.0,
                                overrides={"pi.mapper": {"lr": 1e-3}}))
        back = decode_parameterization(json_roundtrip(encode_parameterization(p)))
        self.assertEqual(back, p)

    def test_emitted_bytes_survive_json(self) -> None:
        emitted = Emitted(adapters={"pi": b"\x00\xff delta"},
                          optim={"pi": b"moments\x01"})
        self.assertEqual(decode_emitted(json_roundtrip(encode_emitted(emitted))),
                         emitted)


# ---------------------------------------------------------------------------
# the learner's wire, in-process: custody, never semantics
# ---------------------------------------------------------------------------

class LearnerWireTest(unittest.TestCase):
    def test_a_run_over_remote_learner_is_byte_identical_to_local(self) -> None:
        tmp_a = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_a.cleanup)
        store_a, train_a, heldout_a = arith_store(tmp_a.name)
        local = run_experiment(arith_spec(train_a, heldout_a), SCHEMA,
                               store_a, FakeEngine(), FakeLearner())

        tmp_b = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_b.cleanup)
        store_b, train_b, heldout_b = arith_store(tmp_b.name)
        remote = RemoteLearner(LocalTransport(LearnerService(FakeLearner())))
        result = run_experiment(arith_spec(train_b, heldout_b), SCHEMA,
                                store_b, FakeEngine(), remote)

        self.assertEqual(local.run_id, result.run_id)
        for key in ("ledger.jsonl", "waves/000001.jsonl.gz"):
            self.assertEqual(
                store_a.path_of(f"runs/{local.run_id}/{key}").read_bytes(),
                store_b.path_of(f"runs/{result.run_id}/{key}").read_bytes(),
                key)


# ---------------------------------------------------------------------------
# in-process residents: the door's own verbs
# ---------------------------------------------------------------------------

class DoorTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, _ = arith_store(tmp.name)
        self.partition = Partition("fake-metal", (0,), 1.0, "L4")
        self.serve = Regime("serve", "inference", BASE, 1)
        self.learn = Regime("learn", "training", BASE, 1)

    def birth(self, regime: Regime, build) -> ResidentBirth:
        return ResidentBirth(label=f"alt:{regime.name}", partition=self.partition,
                             regime=regime, build=build,
                             store=self.store.address())

    def alternating_host(self, *, sleeps: bool, fsdp: int = 1,
                         refusal: str = "") -> tuple[Host, FakeEngine, FakeLearner]:
        engine = FakeEngine(base=BASE, sleeps=sleeps)
        trainer = FakeLearner(fsdp=fsdp, sleeps=sleeps, sleep_refusal=refusal)
        learn = Regime("learn", "training", BASE, fsdp)
        residents = (
            Resident.in_process(self.birth(self.serve, FakeEngineBuild(sleeps=sleeps)),
                                engine),
            Resident.in_process(self.birth(learn, FakeLearnerBuild(sleeps=sleeps)),
                                trainer))
        host = Host("alt", engines=(RemotePool(residents[0].transport, base=BASE),),
                    learner=RemoteLearner(residents[1].transport, fsdp=fsdp),
                    store=self.store, partition=self.partition,
                    regimes=(self.serve, learn), residents=residents)
        return host, engine, trainer

    def test_alternation_hooks_cross_the_door_for_both_kinds(self) -> None:
        """A multi-regime host alternates on its own arbiter group; with
        residents that report `sleeps`, the switch is the resident's own
        sleep/wake — engine AND learner (ADR 0002, Q8)."""
        host, engine, trainer = self.alternating_host(sleeps=True)

        async def switch_twice():
            async with host.arbiter.admit(host.engines[0]):
                pass
            async with host.arbiter.admit(host.learner):
                pass
            async with host.arbiter.admit(host.engines[0]):
                pass
        go(switch_twice())
        # the first switch evicts the learner too (Arbiter._switch): it held
        # the device from install, before any admit
        self.assertEqual(engine.naps, ["wake", "sleep", "wake"])
        self.assertEqual(trainer.naps, ["sleep", "wake", "sleep"])

    def test_a_sharded_learner_that_sleeps_is_wired_like_any_other(self) -> None:
        """The wiring reads the HELLO, never the width (#82): an fsdp=2
        learner whose resident says it sleeps alternates with the engine on
        exactly the code path an unsharded one does. Which ranks moved is the
        chorus's business and never the host's."""
        host, engine, trainer = self.alternating_host(sleeps=True, fsdp=2)

        async def switch():
            async with host.arbiter.admit(host.engines[0]):
                pass
            async with host.arbiter.admit(host.learner):
                pass
        go(switch())
        self.assertEqual(trainer.naps, ["sleep", "wake"])
        self.assertEqual(engine.naps, ["wake", "sleep"])
        self.assertEqual(host.learner.fsdp, 2)

    def test_a_learner_that_cannot_sleep_says_why_in_its_hello(self) -> None:
        """`sleeps` is a PROBED build fact for a sharded learner (I7), so its
        refusal carries the reason across the door — a hook nobody wired is
        otherwise indistinguishable from a substrate nobody checked."""
        host, _, _ = self.alternating_host(
            sleeps=False, fsdp=2, refusal="this torch's FSDP2 is missing X")
        hello = host.residents[1].hello
        self.assertFalse(hello["sleeps"])
        self.assertEqual(hello["fsdp"], 2)
        self.assertIn("missing X", hello["sleep_refusal"])
        self.assertNotIn("sleep_refusal", host.residents[0].hello)   # engines

    def test_a_resident_that_does_not_sleep_gets_no_hooks(self) -> None:
        host, engine, trainer = self.alternating_host(sleeps=False)

        async def switch():
            async with host.arbiter.admit(host.engines[0]):
                pass
            async with host.arbiter.admit(host.learner):
                pass
        go(switch())
        self.assertEqual((engine.naps, trainer.naps), ([], []))
        # and the door refuses a sleep nobody should have asked for
        with self.assertRaises(ResidentError):
            go(host.residents[0].sleep())

    def test_status_and_the_birth_event_name_the_residents(self) -> None:
        host, _, _ = self.alternating_host(sleeps=True)
        rows = host.status()["residents"]
        self.assertEqual([r["label"] for r in rows], ["alt:serve", "alt:learn"])
        self.assertTrue(all(r["alive"] and r["sleeps"] for r in rows))
        ups = [e for e in self.store.read_host_log("alt")
               if e.get("event") == "host-up"]
        self.assertEqual([r["label"] for r in ups[-1]["residents"]],
                         ["alt:serve", "alt:learn"])
        self.assertEqual(ups[-1]["residents"][0]["pid"], os.getpid())

    def test_the_pin_is_asserted(self) -> None:
        two = Partition("m", (2, 3), 0.5, "A100")
        check_devices_seen({"label": "x", "devices_seen": None}, two)   # unmeasured
        check_devices_seen({"label": "x", "devices_seen": 2}, two)
        with self.assertRaises(ResidentError) as caught:
            check_devices_seen({"label": "x", "devices_seen": 1}, two)
        self.assertIn("the pin did not take", str(caught.exception))


# ---------------------------------------------------------------------------
# real children: the claims that are about processes
# ---------------------------------------------------------------------------

class ProcessFixture(unittest.TestCase):
    """One metal on fakes whose residents are REAL child processes."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)

    class Lazy:
        """Resolves the carved address on every frame — addresses exist only
        after the carve the desk is about to command."""

        def __init__(self, service: MetalService, address: str) -> None:
            self.service, self.address = service, address

        async def call(self, verb, payload, *, deadline_s: float = DEADLINE_S):
            return await LocalTransport(
                self.service.service_for(self.address)).call(
                    verb, payload, deadline_s=deadline_s)

        async def ask(self, verb, payload, *, deadline_s: float = DEADLINE_S):
            return await LocalTransport(
                self.service.service_for(self.address)).ask(
                    verb, payload, deadline_s=deadline_s)

    def metal(self, builds: Builds | None = None) -> MetalService:
        service = MetalService(
            Metal("proc-metal", "L4", 1, 24.0), store=self.store,
            builds=builds or Builds.fakes(engine_sleeps=True, learner_sleeps=True),
            address_of=lambda host_name: f"proc://{host_name}",
            schema_for=lambda base: fake_qwen_schema(4, base=base),
            transport_for=lambda address: self.Lazy(service, address))
        self.addCleanup(service.shutdown)
        return service

    def desk(self, service: MetalService) -> Desk:
        desk = Desk(
            self.store,
            host_for=lambda addr: RemoteHost(self.Lazy(service, addr)),
            metal_for=lambda addr: RemoteMetal(LocalTransport(service)))
        desk.register_metal(service.metal, address="metal://proc",
                            builds=service.builds)
        return desk

    @staticmethod
    def inference_request(vram_gb: float) -> dict:
        return {"regimes": [{"name": "serve", "capability": "inference",
                             "base": BASE, "shape": 1}],
                "base": BASE, "vram_gb": vram_gb}


class ProcessResidentTest(ProcessFixture):
    def test_a_run_through_process_residents_is_byte_identical(self) -> None:
        """The desk carves ONE alternating host on the metal: two children,
        one per regime, hellos from other pids. The adopted run's bytes match
        the in-process run's — the process boundary adds custody, never
        semantics — and the sleep/wake frames crossed to real processes on
        every switch."""
        service = self.metal()
        desk = self.desk(service)

        async def drive():
            reply = await Campaigns(desk).submit(alternating_spec(self.train))
            self.assertTrue(reply["accepted"], reply)
            await service.hosts[reply["host"]]._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())

        host = service.hosts[reply["host"]]
        self.assertEqual(len(host.residents), 2)
        pids = {r.pid() for r in host.residents}
        self.assertNotIn(os.getpid(), pids)
        self.assertTrue(all(r.alive() for r in host.residents))
        self.assertTrue(all(r.hello["sleeps"] for r in host.residents))
        self.assertEqual(host.status()["residents"][0]["kind"], "inference")
        # the metal's recipe travelled: journaled as the desk's own `recipe`
        # event (ADR 0007, Q4 — the registration only proposed it), and
        # visible in describe()
        declared = [e for e in self.store.read_fleet_log()
                    if e.get("event") == "recipe"][-1]
        self.assertEqual(declared["builds"]["engine"]["type"], "FakeEngineBuild")
        self.assertEqual(service.describe()["builds"], service.builds.row())
        self.assertEqual(
            [r["label"] for r in service.describe()["hosts"][host.name]["residents"]],
            [r.label for r in host.residents])

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        other_store, other_train, _ = arith_store(tmp.name)
        plain = Host("plain", engines=(FakeEngine(base=BASE),),
                     learner=FakeLearner(), store=other_store)
        report = go(plain.submit(alternating_spec(other_train), SCHEMA))
        self.assertEqual(reply["run_id"], report.run_id)
        for key in ("ledger.jsonl", "waves/000001.jsonl.gz"):
            self.assertEqual(
                self.store.path_of(f"runs/{report.run_id}/{key}").read_bytes(),
                other_store.path_of(f"runs/{report.run_id}/{key}").read_bytes(),
                key)

        teardowns = service.shutdown()
        self.assertTrue(all(t.graceful for t in teardowns), teardowns)
        self.assertFalse(any(r.alive() for r in host.residents))

    def test_a_dead_resident_decarves_its_host(self) -> None:
        """Kill a resident behind the metal's back: its host is dead (I12),
        the booking is free again, the address answers nothing, and the death
        is on the record — nothing is restarted in place (ADR 0002, Q7)."""
        service = self.metal()
        born = go(service.carve(self.inference_request(14.4)))
        self.assertTrue(born["carved"], born)
        self.assertAlmostEqual(service.residual()[0], 9.6)     # GB
        resident = service.hosts[born["host"]].residents[0]

        os.kill(resident.pid(), signal.SIGKILL)
        deadline = time.monotonic() + 15.0
        while born["host"] in service.hosts and time.monotonic() < deadline:
            time.sleep(0.05)

        self.assertNotIn(born["host"], service.hosts)
        self.assertEqual(service.residual(), [24.0])
        self.assertEqual(service.deaths, [born["host"]])
        with self.assertRaises(Exception):
            service.service_for(born["address"])

    def test_a_birth_that_fails_in_the_child_releases_the_booking(self) -> None:
        """A recipe that cannot build an inference regime fails INSIDE the
        child; the hello carries the failure, the carve refuses with it, the
        booking is released, and no process is left behind."""
        service = self.metal(builds=Builds(engine=FakeLearnerBuild(),
                                           learner=FakeLearnerBuild()))
        born = go(service.carve(self.inference_request(12.0)))
        self.assertFalse(born["carved"], born)
        self.assertIn("not an engine build", born["error"])
        self.assertEqual(service.residual(), [24.0])
        self.assertEqual(service.hosts, {})
        self.assertFalse(multiprocessing.active_children())


class LadderTest(unittest.TestCase):
    """The teardown ladder, now every GPU-holding child's (ADR 0002, Q9):
    the rungs, on stubs — no torch needed to prove process management."""

    def children(self, *targets):
        context = multiprocessing.get_context("spawn")
        out = []
        for target in targets:
            ready = context.Event()
            child = context.Process(target=target, args=(ready,), daemon=True)
            child.start()
            self.assertTrue(ready.wait(30.0), "stub child never reached steady state")
            out.append(child)

        def end_them():
            for child in out:
                if child.is_alive():
                    child.kill()
                child.join(timeout=10.0)
        self.addCleanup(end_them)
        return tuple(out)

    def test_a_child_that_leaves_is_never_signalled(self) -> None:
        teardown = escalate(self.children(leaves_at_once, leaves_at_once),
                            grace_s=10.0, signal_grace_s=2.0)
        self.assertTrue(teardown.graceful)

    def test_a_child_wedged_past_sigterm_is_killed_and_named(self) -> None:
        children = self.children(leaves_at_once, ignores_sigterm)
        teardown = escalate(children, grace_s=0.3, signal_grace_s=0.3)
        self.assertEqual((teardown.deaf, teardown.wedged, teardown.lost),
                         ((2,), (2,), ()))
        self.assertFalse(teardown.graceful)
        self.assertFalse(children[1].is_alive())
        self.assertIn("survived SIGTERM", teardown.line())
        self.assertTrue(Teardown().graceful)


if __name__ == "__main__":
    unittest.main()
