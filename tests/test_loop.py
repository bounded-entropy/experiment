"""The runner loop end to end on fake metal (rlstack.runner.loop)."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace

from common import arith_spec, arith_store
from rlstack import (
    GpuConfig, HostSpec, learner, pool, run_experiment,
    Seeds,
    FakeEngine, FakeLearner, PolicySpec, WarmStart, flatten, lora,
    fake_qwen_schema, run_experiment, trajectory_from_row,
)

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


class LoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(self._tmp.name)

    def run_spec(self, spec, engine=None, learner=None):
        engine = engine or FakeEngine()
        report = run_experiment(spec, SCHEMA, self.store, engine,
                                learner or FakeLearner())
        return report, engine

    def test_completes_and_commits_every_update(self) -> None:
        spec = arith_spec(self.train)
        report, engine = self.run_spec(spec)

        self.assertEqual(report.updates_completed, 4)
        self.assertIsNone(report.resumed_from)

        run = self.store.open_run(report.run_id)
        entries = run.read_ledger()
        self.assertEqual([e["update"] for e in entries], [1, 2, 3, 4])
        self.assertEqual([e["versions"]["pi"] for e in entries], [1, 2, 3, 4])
        for k in range(1, 5):
            run.read_blob("adapters", "pi", k)   # raises if missing
            self.assertTrue(run.read_wave(k))
        # ...and the moments only at the tail: retention swept the rest as
        # each commit made it unreadable-by-anyone (stores/retention.py)
        run.read_blob("optim", "pi", 4)
        for k in range(1, 4):
            with self.assertRaises(FileNotFoundError):
                run.read_blob("optim", "pi", k)

        # phase-1 bundle + one per update, all distinct, ledger agrees
        self.assertEqual(len(engine.bundle_log), 5)
        self.assertEqual(len(set(engine.bundle_log)), 5)
        self.assertEqual([e["bundle_id"] for e in entries], engine.bundle_log[1:])

        for entry in entries:
            self.assertEqual(entry["wave"]["trajectories"], 4)
            self.assertIn("reward", entry["post"])       # pipeline means
            self.assertIn("advantage", entry["post"])
            for key in ("loss", "mean_ratio", "logprob_gap", "grad_norm",
                        "tokens", "microbatches"):
                self.assertIn(key, entry["train"])

        # postdata is stored beside the waves, wave-aligned
        columns = run.read_postdata(1)
        self.assertEqual(set(columns), {"reward", "advantage"})
        self.assertEqual(len(columns["reward"]), 4)

    def test_rollout_rows_replay_through_flatten(self) -> None:
        spec = arith_spec(self.train)
        report, engine = self.run_spec(spec)
        run = self.store.open_run(report.run_id)
        for row in run.read_wave(1):
            traj = trajectory_from_row(row)
            flat = flatten(traj, engine.tokenize)
            self.assertEqual(flat.doc_len, len(flat.token_ids))
            self.assertIn(1, flat.loss_mask)      # generated tokens present
            self.assertIn(0, flat.loss_mask)      # injected prompt present

    def test_recorded_draws_survive_the_store(self) -> None:
        spec = arith_spec(self.train)
        report, _ = self.run_spec(spec, engine=FakeEngine(record_draws=True))
        run = self.store.open_run(report.run_id)
        row = run.read_wave(1)[0]
        traj = trajectory_from_row(row)
        draws = traj.turns[0].token_extras["adapter_draw"]
        self.assertEqual(len(draws), len(traj.turns[0].token_ids))
        flat = flatten(traj, FakeEngine().tokenize)
        column = flat.token_extras["adapter_draw"]
        self.assertEqual(len(column), flat.doc_len)
        self.assertIn(None, column)               # injected prompt positions

    def test_resubmit_attaches_and_no_ops(self) -> None:
        spec = arith_spec(self.train)
        report, _ = self.run_spec(spec)
        ledger = self.store.path_of(f"runs/{report.run_id}/ledger.jsonl")
        before = ledger.read_bytes()

        again, _ = self.run_spec(spec)            # fresh fakes on purpose
        self.assertEqual(again.run_id, report.run_id)
        self.assertEqual(again.resumed_from, 4)
        self.assertEqual(ledger.read_bytes(), before)

    def test_unservable_spec_dies_at_phase_0_before_any_write(self) -> None:
        """Reachability is checked against the main engine's self-reported
        inventory before the run exists: no build reaches model.norm via
        prompt_embeds, so this dies with site-unreachable and an empty store."""
        from rlstack import AdapterSpec, SpecError
        spec = arith_spec(self.train, policy=PolicySpec(
            base="Qwen/Qwen3-0.6B",
            bank={"latent": AdapterSpec(adapter_type="soft_prompt", site="final_hidden",
                                        init={"n": 8, "d": 64})}))
        with self.assertRaises(SpecError) as caught:
            self.run_spec(spec)
        self.assertIn("site-unreachable", str(caught.exception))
        self.assertEqual(self.store.list_runs(), [])

    def test_two_experiments_share_one_learner(self) -> None:
        """The Learner tenancy invariant (the trainer-side mirror): every verb
        pins a tenant, so two experiments' states coexist on ONE learner — and
        neither run's bytes change versus a private learner."""
        shared_engine, shared_learner = FakeEngine(), FakeLearner()
        report_a, _ = self.run_spec(arith_spec(self.train),
                                    engine=shared_engine,
                                    learner=shared_learner)
        report_b, _ = self.run_spec(arith_spec(self.train, seeds=Seeds(master=99)),
                                    engine=shared_engine,
                                    learner=shared_learner)
        self.assertNotEqual(report_a.run_id, report_b.run_id)

        import tempfile

        from common import arith_store as fresh_store
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        other_store, other_train, _ = fresh_store(tmp.name)
        for spec, report in ((arith_spec(other_train), report_a),
                             (arith_spec(other_train, seeds=Seeds(master=99)),
                              report_b)):
            private = run_experiment(spec, SCHEMA, other_store,
                                     FakeEngine(), FakeLearner())
            self.assertEqual(
                self.store.path_of(
                    f"runs/{report.run_id}/ledger.jsonl").read_bytes(),
                other_store.path_of(
                    f"runs/{private.run_id}/ledger.jsonl").read_bytes())

    def test_two_experiments_share_one_engine(self) -> None:
        """The multi-tenancy invariant, exercised: bundle registration is
        additive, so a second experiment's bundles coexist with the first's on
        ONE engine — and neither run's bytes change versus a private engine."""
        shared = FakeEngine()
        spec_a = arith_spec(self.train)
        spec_b = arith_spec(self.train, seeds=Seeds(master=99))
        report_a, _ = self.run_spec(spec_a, engine=shared)
        report_b, _ = self.run_spec(spec_b, engine=shared)
        self.assertNotEqual(report_a.run_id, report_b.run_id)
        self.assertEqual(len(shared.bundle_log), 10)   # (init + 4 updates) × 2
        self.assertEqual(len(set(shared.bundle_log)), 10)

        # identical results on a private engine: the tenants never interfered
        import tempfile

        from common import arith_store as fresh_store
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        other_store, other_train, _ = fresh_store(tmp.name)
        for spec, report in ((arith_spec(other_train), report_a),
                             (arith_spec(other_train, seeds=Seeds(master=99)),
                              report_b)):
            private = run_experiment(spec, SCHEMA, other_store,
                                     FakeEngine(), FakeLearner())
            shared_ledger = self.store.path_of(
                f"runs/{report.run_id}/ledger.jsonl").read_bytes()
            private_ledger = other_store.path_of(
                f"runs/{private.run_id}/ledger.jsonl").read_bytes()
            self.assertEqual(shared_ledger, private_ledger)

    def test_warm_start_loads_mapped_deltas(self) -> None:
        parent_spec = arith_spec(self.train)
        parent, _ = self.run_spec(parent_spec)

        child_policy = PolicySpec(base="Qwen/Qwen3-0.6B",
                                  bank={"ghost": lora("layers.0-3.self_attn.*", r=16)})
        warm_spec = arith_spec(
            self.train, policy=child_policy,
            init=WarmStart(policy=f"store://{parent.run_id}@4", optim="load",
                           map={"pi": "ghost"}))
        cold_spec = arith_spec(self.train, policy=child_policy)

        warm, warm_engine = self.run_spec(warm_spec)
        cold, cold_engine = self.run_spec(cold_spec)

        self.assertNotEqual(warm.run_id, parent.run_id)
        self.assertNotEqual(warm.run_id, cold.run_id)    # WarmStart hashes in
        # the warm child's phase-1 bundle reflects the loaded parent state
        self.assertNotEqual(warm_engine.bundle_log[0], cold_engine.bundle_log[0])


if __name__ == "__main__":
    unittest.main()


class JudgePoolTest(unittest.TestCase):
    """Pipelines that sample from a declared judge pool, end to end."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(self._tmp.name)

    def run_spec(self, spec, engine=None, learner=None):
        engine = engine or FakeEngine()
        report = run_experiment(spec, SCHEMA, self.store, engine,
                                learner or FakeLearner())
        return report, engine

    def judge_spec(self):
        base = arith_spec(self.train)
        return replace(
            base,
            algo=replace(base.algo, post=("llm_judge", "grpo_advantage")),
            gpu_config=GpuConfig(hosts=(
                HostSpec((pool("main"),)), HostSpec((pool("judge"),)),
                HostSpec((learner(),)))))

    def test_unmapped_judge_pool_is_refused_at_submit(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.run_spec(self.judge_spec())     # one engine: judge unmapped
        self.assertIn("judge", str(caught.exception))

    def test_judge_pool_end_to_end(self) -> None:
        engines_map = {"main": FakeEngine(), "judge": FakeEngine(p_correct=1.0)}
        report = run_experiment(self.judge_spec(), SCHEMA, self.store,
                                engines_map, FakeLearner())
        run = self.store.open_run(report.run_id)
        entries = run.read_ledger()
        self.assertEqual([e["update"] for e in entries], [1, 2, 3, 4])
        for entry in entries:
            self.assertIn("reward", entry["post"])   # the judge's column
        # the judge pool served ONLY its base bundle; policy bundles stayed
        # on the main engine
        self.assertEqual(engines_map["judge"].bundle_log, ["bundle:base:judge"])


class BaseBindingTest(unittest.TestCase):
    """The deploy hands metal; pool-base-mismatch is the check that it handed
    the RIGHT metal (Engine.base vs each pool's declared base)."""

    def setUp(self) -> None:
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        from common import arith_store
        self.store, self.train, _ = arith_store(tmp.name)

    def test_wrong_base_engine_is_refused_at_submit(self) -> None:
        from rlstack.spec.validate import SpecError
        with self.assertRaises(SpecError) as caught:
            run_experiment(arith_spec(self.train), SCHEMA, self.store,
                           FakeEngine(base="some/other-model"), FakeLearner())
        self.assertIn("pool-base-mismatch", str(caught.exception))

    def test_matching_and_wildcard_bases_pass(self) -> None:
        report = run_experiment(
            arith_spec(self.train), SCHEMA, self.store,
            FakeEngine(base="Qwen/Qwen3-0.6B"), FakeLearner())
        self.assertEqual(report.updates_completed, 4)
