"""The wire (rlstack.runner.remote): a pool on another host, same protocol.

Claims under test: a run whose "main" pool is a RemotePool is byte-identical
to the same run on the local engine (the wire adds custody, never
semantics); every frame survives JSON (LocalTransport round-trips each
direction); admission happens at the SERVING host's own arbiter (#43); the
sync verbs (add_bundle / reachability / tokenize) cross without an event
loop; and a remote pool declared in a sleep group is refused — alternation
is an intra-partition fact.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from common import arith_spec, arith_store
from rlstack import (
    Bundle, FakeEngine, FakeLearner, Arbiter, Topology, HostSpec, Host,
    HostService,
    LocalTransport, Mechanism, Message, Regime, RemotePool, Role,
    SamplingSpec, SiteMeta, fake_qwen_schema, learner, pool,
    run_experiment,
)

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def go(coro):
    return asyncio.run(coro)


class RemoteRunTest(unittest.TestCase):
    def test_a_remote_main_pool_is_byte_identical_to_local(self) -> None:
        """The whole experiment, with its main pool served by ANOTHER host
        over the wire: the run directory matches the local run's byte for
        byte — the wire adds latency, never semantics."""
        tmp_a = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_a.cleanup)
        store_a, train_a, heldout_a = arith_store(tmp_a.name)
        local = run_experiment(arith_spec(train_a, heldout_a), SCHEMA,
                               store_a, FakeEngine(), FakeLearner())

        tmp_b = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_b.cleanup)
        store_b, train_b, heldout_b = arith_store(tmp_b.name)
        serving = Host("srv", engines=(FakeEngine(),), learner=FakeLearner(),
                       store=store_b)
        remote = RemotePool(LocalTransport(HostService(serving)))
        result = run_experiment(arith_spec(train_b, heldout_b), SCHEMA,
                                store_b, remote, FakeLearner())

        self.assertEqual(local.run_id, result.run_id)
        for key in ("ledger.jsonl", "waves/000001.jsonl.gz"):
            self.assertEqual(
                store_a.path_of(f"runs/{local.run_id}/{key}").read_bytes(),
                store_b.path_of(f"runs/{result.run_id}/{key}").read_bytes(),
                key)

    def test_a_remote_pool_alternates_at_its_serving_host_only(self) -> None:
        """Alternation is an intra-partition fact of the host that OWNS the
        pool: a pool served over the wire attaches here as a free resident
        whatever HostSpec it came from (ADR 0001 — two pools may alternate
        on a host of their own), and the serving host's arbiter does the
        switching. The run commits, and the local arbiter holds the remote
        in no exclusive group."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        serving = Host("srv", engines=(FakeEngine(),), learner=FakeLearner(),
                       store=store)
        door = HostService(serving)
        main, aux = (RemotePool(LocalTransport(door)),
                     RemotePool(LocalTransport(door)))
        spec = arith_spec(train, topology=Topology(hosts=(
            HostSpec((pool("main"), pool("aux"))), HostSpec((learner(),)))))
        arbiter = Arbiter()
        report = run_experiment(spec, SCHEMA, store, {"main": main, "aux": aux},
                                FakeLearner(), arbiter=arbiter)
        self.assertEqual((report.completed, report.extent), (4, "train"))
        self.assertIsNone(arbiter.attached_group(main))
        self.assertIsNone(arbiter.attached_group(aux))


class WireTest(unittest.TestCase):
    """The verbs one by one, over an ALTERNATING serving host — so every
    admitted verb provably entered the serving host's arbiter."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, _, _ = arith_store(tmp.name)
        self.engine = FakeEngine(base="Qwen/Qwen3-8B")
        self.host = Host(
            "alt", engines=(self.engine,), learner=FakeLearner(),
            store=self.store,
            regimes=(Regime("serve", "inference", "Qwen/Qwen3-8B", 1),
                     Regime("train", "training", None, 1)))
        self.remote = RemotePool(LocalTransport(HostService(self.host)),
                                 base="Qwen/Qwen3-8B")
        self.messages = (Message(Role.USER, "What is 2+2?"),)

    def test_sampling_is_admitted_at_the_serving_host(self) -> None:
        self.remote.add_bundle(Bundle("bundle:x", {"pi": 0}))

        async def sample():
            return [event async for event in self.remote.sample_tokens(
                self.messages, SamplingSpec(), (), "bundle:x", seed=7)]

        events = go(sample())
        self.assertGreater(len(events), 1)
        # the serving host's OWN arbiter switched its alternation group in
        self.assertEqual(self.host.arbiter.switches, ["alt:serve"])
        self.assertEqual(self.host.arbiter.residency(),
                         {"host:alt": "alt:serve"})

    def test_scores_survive_the_wire_exactly(self) -> None:
        self.remote.add_bundle(Bundle("bundle:x", {"pi": 0}))
        over_wire = go(self.remote.score_tokens(self.messages, (52, 53),
                                                "bundle:x"))
        direct = go(self.engine.score_tokens(self.messages, (52, 53),
                                             "bundle:x"))
        self.assertEqual(over_wire, direct)      # float-exact through JSON

    def test_bundle_bytes_survive_base64(self) -> None:
        self.remote.add_bundle(Bundle(
            "bundle:y", {"pi": 1}, payloads={"pi": b"\x00\xff raw bytes"},
            adapter_types={"pi": "lora"}))
        self.assertIn("bundle:y", self.engine.bundle_log)

    def test_build_facts_cross_the_wire(self) -> None:
        sites = (SiteMeta("layers.0.self_attn.q_proj",
                          "model.layers.0.self_attn.q_proj",
                          has_weight=True, shape=(16, 16), is_boundary=False),)
        reach = self.remote.reachability(sites)
        self.assertEqual(reach["layers.0.self_attn.q_proj"], Mechanism.PUNICA)
        self.assertEqual(self.remote.tokenize("ab"), (97, 98))

    def test_an_unserved_address_is_refused_by_the_serving_host(self) -> None:
        wrong = RemotePool(LocalTransport(HostService(self.host)),
                           base="Qwen/Qwen3-32B")
        with self.assertRaises(KeyError) as caught:
            wrong.tokenize("a")
        self.assertIn("serves no", str(caught.exception))


class Recorder:
    """A Transport that records which side each verb took, and enforces the
    JSON-safety contract on the way past."""

    def __init__(self) -> None:
        self.called: list[str] = []
        self.asked: list[str] = []

    async def call(self, verb: str, payload: dict, *, deadline_s: float = 0.0) -> dict:
        json.dumps(payload)
        self.called.append(verb)
        return {"events": [], "logprobs": []}

    async def ask(self, verb: str, payload: dict, *, deadline_s: float = 0.0) -> dict:
        json.dumps(payload)
        self.asked.append(verb)
        return {"mechanisms": {}, "token_ids": []}


class VerbSplitTest(unittest.TestCase):
    """WHICH verb rides WHICH calling convention is a contract, not an
    implementation detail (#45): an out-of-process transport carries the
    admitted verbs asynchronously (they occupy the serving host's GPU) and
    the admission-free ones synchronously, because their call sites are sync
    — Phase 1's add_bundle, flatten's tokenize. A verb that changed sides
    would break every real transport, so the split is pinned here rather
    than rediscovered on metal."""

    def setUp(self) -> None:
        self.recorder = Recorder()
        self.pool = RemotePool(self.recorder, base="Qwen/Qwen3-0.6B")
        self.messages = (Message(Role.USER, "What is 2+2?"),)

    def test_the_admitted_verbs_ride_call(self) -> None:
        async def both():
            async for _ in self.pool.sample_tokens(
                    self.messages, SamplingSpec(), (), "bundle:x", seed=1):
                pass
            await self.pool.score_tokens(self.messages, (5,), "bundle:x")

        go(both())
        self.assertEqual(self.recorder.called, ["sample_tokens", "score_tokens"])
        self.assertEqual(self.recorder.asked, [])

    def test_the_admission_free_verbs_ride_ask(self) -> None:
        self.pool.add_bundle(Bundle("bundle:x", {"pi": 0},
                                    payloads={"pi": b"\x00\xff"},
                                    adapter_types={"pi": "lora"}))
        self.pool.reachability(())
        self.pool.tokenize("ab")
        self.assertEqual(self.recorder.asked,
                         ["add_bundle", "reachability", "tokenize"])
        self.assertEqual(self.recorder.called, [])

    def test_every_verb_addresses_the_capability_it_wants(self) -> None:
        """base and tp travel in EVERY frame: the serving host addresses its
        engines by capability, never by the caller's pool name (#43)."""
        addressed: list[dict] = []

        class Address(Recorder):
            async def ask(self, verb: str, payload: dict, *,
                          deadline_s: float = 0.0) -> dict:
                addressed.append({"base": payload["base"], "tp": payload["tp"]})
                return await super().ask(verb, payload)

        RemotePool(Address(), base="Qwen/Qwen3-8B", tp=4).tokenize("a")
        self.assertEqual(addressed, [{"base": "Qwen/Qwen3-8B", "tp": 4}])


if __name__ == "__main__":
    unittest.main()
