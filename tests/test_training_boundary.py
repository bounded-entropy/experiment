"""Training commits independently; inference restores only its selected version."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from common import arith_spec, arith_store, sealed
from test_resume import CrashingStore, SimulatedCrash, snapshot
from rlstack.runner.checkpointing import EVERY_UPDATE, Checkpointing
from rlstack import (
    FakeEngine, FakeLearner, GroupPlan, Host, HostService, HostSpec, LocalStore,
    LocalTransport, Plans, Replay, RunPlan, Topology, WavePlan, encode,
    fake_qwen_schema, learner, pool, run_experiment, trajectory_to_row,
)
from rlstack.policy.compile import Bundle, restore_bundle
from rlstack.runner.campaign import demands_of
from rlstack.runner.roles.generator import Generator
from rlstack.runner.loop import experiment_identity, needs_of
from rlstack.runner.remote import RemoteLearner
from rlstack.runner.restore import restore_bundle_on
from rlstack.spec.validate import SpecError, traffic_pools, validate_or_raise

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def replay_spec(store, train):
    rows = [trajectory_to_row(sealed("fact", answer)) for answer in ("4", "5")]
    uri = store.cas_put("".join(json.dumps(row) + "\n" for row in rows).encode())
    wave = WavePlan((GroupPlan("fact", (Replay(uri + "#0"), Replay(uri + "#1"))),))
    plan = store.cas_put(encode(RunPlan((wave,) * 4)))
    return arith_spec(train, gen=None, plans=Plans(train=plan),
                      topology=Topology((HostSpec((learner(),)),)))


STORE_DELIVERY = Checkpointing(every=1, delivery="store")
"""The file-system-only path (ADR 0014, Q5): the Trainer touches no engine,
inference installs each committed version from the blobs after its ledger
line — the discipline ADR 0011 proved, kept as the non-default setting."""

class TrainingBoundaryTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, _ = arith_store(tmp.name)
        self.spec = replay_spec(self.store, self.train)

    def test_learner_only_host_runs_and_commits_through_a_routed_learner(self):
        self.assertEqual(traffic_pools(self.spec), set())
        self.assertEqual([(n.learner, n.pools) for n in needs_of(self.spec)], [(True, ())])
        self.assertEqual([d.capability for d in demands_of(self.spec)], ["training"])
        served = FakeLearner()
        training_host = Host("training", engines=(), learner=served, store=self.store)
        proxy = RemoteLearner(LocalTransport(HostService(training_host)), admitted=True)
        anchor = Host("anchor", engines=(), learner=None, store=self.store)
        report = asyncio.run(anchor.submit(self.spec, SCHEMA, learner=proxy, checkpointing=EVERY_UPDATE))
        self.assertEqual(report.completed, 4)
        self.assertEqual(self.store.open_run(report.run_id).ledger_tail()["update"], 4)
        self.assertEqual(served._tenants, {})  # host releases remote custody

    def test_unused_unavailable_engine_cannot_block_replay(self):
        engine = FakeEngine()
        with (patch.object(engine, "add_bundle", side_effect=ConnectionError("offline")),
              patch.object(engine, "knows_bundle", side_effect=ConnectionError("offline")),
              patch.object(engine, "reachability", side_effect=ConnectionError("offline")),
              patch.object(engine, "tokenize", side_effect=ConnectionError("offline"))):
            report = run_experiment(self.spec, SCHEMA, self.store, {"main": engine}, FakeLearner(), checkpointing=EVERY_UPDATE)
        self.assertEqual(report.completed, 4)
        self.assertEqual(engine.bundle_log, [])

    def test_generation_loads_after_training_finishes_and_ignores_uncommitted_blobs(self):
        online = arith_spec(self.train)
        spec = replace(online, plans=replace(online.plans, train=self.spec.plans.train))
        rid = experiment_identity(spec, SCHEMA)
        engine = FakeEngine()
        original_generate = Generator.run_forever
        original_install = engine.add_bundle
        installed = []

        async def delayed_generate(generator):
            # Training can finish while the inference consumer has not begun.
            await generator.signals.wait_for(lambda: generator.committed() == 4)
            generator.run.write_blob("adapters", "pi", 5, b"uncommitted partial update")
            await original_generate(generator)

        def install(bundle):
            run = self.store.open_run(rid, create=False)
            tail = run.ledger_tail()
            self.assertEqual(tail["update"], 4)
            self.assertEqual(bundle.bundle_id, tail["bundle_id"])
            expected = restore_bundle(tail["versions"], tail["bundle_id"],
                                      run.read_blob, ("pi",), {"pi": "lora"})
            self.assertEqual(bundle, expected)
            installed.append(bundle)
            original_install(bundle)

        with (patch.object(Generator, "run_forever", delayed_generate),
              patch.object(engine, "add_bundle", install)):
            report = run_experiment(spec, SCHEMA, self.store, engine, FakeLearner(),
                                    checkpointing=STORE_DELIVERY)
        self.assertEqual(report.completed, 4)
        self.assertEqual(len(installed), 1)
        run = self.store.open_run(rid)
        for index in range(1, 5):
            for row in run.read_rollout(index):
                self.assertEqual(row["turns"][0]["policy_version"], {"pi": 4})

    def test_inference_install_observes_commit_and_recovers_after_failure(self):
        spec = arith_spec(self.train)
        rid = experiment_identity(spec, SCHEMA)
        engine = FakeEngine()
        add = engine.add_bundle

        def crash_after_install(bundle):
            run = self.store.open_run(rid, create=False)
            version = bundle.policy_version["pi"]
            if version:
                self.assertGreaterEqual(run.ledger_tail()["update"], version)
            add(bundle)
            if version == 2:
                raise SimulatedCrash("engine died after installation")

        with patch.object(engine, "add_bundle", crash_after_install):
            with self.assertRaises(SimulatedCrash):
                run_experiment(spec, SCHEMA, self.store, engine, FakeLearner(),
                               checkpointing=STORE_DELIVERY)
        self.assertEqual(self.store.open_run(rid).ledger_tail()["update"], 2)
        restarted = FakeEngine()
        report = run_experiment(spec, SCHEMA, self.store, restarted, FakeLearner(), resume=True,
                                checkpointing=STORE_DELIVERY)
        self.assertEqual(report.resumed_from, 2)
        self.assertEqual(restarted.bundle_log[0], self.store.open_run(rid).read_ledger()[1]["bundle_id"])
        with tempfile.TemporaryDirectory() as root:
            other, train, _ = arith_store(root)
            reference = run_experiment(arith_spec(train), SCHEMA, other, FakeEngine(), FakeLearner(),
                                       checkpointing=STORE_DELIVERY)
            self.assertEqual(snapshot(self.store, rid), snapshot(other, reference.run_id))

    def test_learner_only_resume_discards_partial_updates(self):
        report = run_experiment(self.spec, SCHEMA, self.store, {}, FakeLearner(), checkpointing=EVERY_UPDATE)
        expected = snapshot(self.store, report.run_id)
        for method, after, post in (("write_blob", 3, True), ("append_ledger", 2, False),
                                    ("append_ledger", 2, True)):
            with self.subTest(method=method, post=post), tempfile.TemporaryDirectory() as root:
                other, train, _ = arith_store(root)
                spec = replay_spec(other, train)
                with self.assertRaises(SimulatedCrash):
                    run_experiment(spec, SCHEMA, CrashingStore(root, method, after, post), {}, FakeLearner(), checkpointing=EVERY_UPDATE)
                resumed = run_experiment(spec, SCHEMA, LocalStore(root), {}, FakeLearner(), resume=True, checkpointing=EVERY_UPDATE)
                self.assertEqual(snapshot(LocalStore(root), resumed.run_id), expected)

    def test_judge_only_replay_needs_no_main_and_reloads_an_evicted_base(self):
        spec = replace(self.spec,
            algo=replace(self.spec.algo, post=("llm_judge", "grpo_advantage")),
            topology=Topology((HostSpec((pool("judge"),)), HostSpec((learner(),)))))
        self.assertEqual(traffic_pools(spec), {"judge"})
        judge = FakeEngine()
        # Each wave finds an empty engine, as after an eviction or restart.
        def empty(bundle_id):
            judge._known.clear()
            return False
        with patch.object(judge, "knows_bundle", empty):
            report = run_experiment(spec, SCHEMA, self.store, {"judge": judge}, FakeLearner(), checkpointing=EVERY_UPDATE)
        self.assertEqual(report.completed, 4)
        self.assertEqual(judge.bundle_log, ["bundle:base:judge"] * 4)

    def test_main_scoring_must_declare_main(self):
        import rlstack.training.post.policy_logprobs
        spec = replace(self.spec, algo=replace(self.spec.algo,
            post=("policy_logprobs", *self.spec.algo.post)))
        with self.assertRaises(SpecError) as caught:
            validate_or_raise(spec, SCHEMA)
        self.assertIn("post-pool-missing", str(caught.exception))

    def test_historical_scoring_pin_restores_exact_bytes_after_eviction(self):
        report = run_experiment(self.spec, SCHEMA, self.store, {}, FakeLearner(), checkpointing=EVERY_UPDATE)
        run = self.store.open_run(report.run_id)
        entry = run.read_ledger()[0]
        pin = Bundle.pin(entry["bundle_id"], entry["versions"])
        engine = FakeEngine()
        expected = restore_bundle(pin.policy_version, pin.bundle_id,
                                  run.read_blob, ("pi",), {"pi": "lora"})
        with patch.object(engine, "add_bundle", wraps=engine.add_bundle) as install:
            for _ in range(2):
                engine._known.clear()
                restore_bundle_on(engine, pin, run.read_blob, ("pi",), {"pi": "lora"})
            self.assertEqual([call.args[0] for call in install.call_args_list], [expected, expected])
        self.assertNotEqual(pin.bundle_id, run.ledger_tail()["bundle_id"])
