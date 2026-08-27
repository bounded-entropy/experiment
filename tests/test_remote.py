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
import tempfile
import unittest

from common import arith_spec, arith_store
from rlstack import (
    Bundle, FakeEngine, FakeLearner, GpuConfig, GpuGroup, Host, HostService,
    LocalTransport, Mechanism, Message, Regime, RemotePool, Role,
    SamplingSpec, SiteMeta, fake_qwen_schema, gpus, learner, pool,
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

    def test_a_remote_pool_in_a_sleep_group_is_refused(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        serving = Host("srv", engines=(FakeEngine(),), learner=FakeLearner(),
                       store=store)
        remote = RemotePool(LocalTransport(HostService(serving)))
        spec = arith_spec(train, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"), learner()), sharing="sleep"),)))
        with self.assertRaises(ValueError) as caught:
            run_experiment(spec, SCHEMA, store, remote, FakeLearner())
        self.assertIn("one host", str(caught.exception))


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
            kinds={"pi": "lora"}))
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


if __name__ == "__main__":
    unittest.main()
