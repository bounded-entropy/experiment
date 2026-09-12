"""The runner loop end to end on fake metal (rlstack.runner.loop)."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace

from common import arith_spec, arith_store, cas_uri, generation_spec
from rlstack import (
    Topology, GroupPlan, HostSpec, Plans, Replay, RunPlan, WavePlan,
    encode, learner, pool, run_experiment,
    Seeds,
    FakeEngine, FakeLearner, PolicySpec, WarmStart, flatten, lora,
    fake_qwen_schema, run_experiment, trajectory_from_row,
)
from rlstack.data.stores.base import RunProgress, run_done, run_progress
from rlstack.runner.loop import needs_of

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

    def test_slow_run_attachment_keeps_the_event_loop_responsive(self):
        import asyncio
        import threading
        from unittest.mock import patch
        from rlstack.runner.loop import run_experiment_async

        entered, release = threading.Event(), threading.Event()
        original = self.store.open_run

        def slow_open(*args, **kwargs):
            entered.set()
            release.wait(2)
            return original(*args, **kwargs)

        async def drive():
            task = asyncio.create_task(run_experiment_async(
                arith_spec(self.train), SCHEMA, self.store, FakeEngine(), FakeLearner()))
            while not entered.is_set():
                await asyncio.sleep(0)
            responsive = not release.is_set()
            release.set()
            await task
            return responsive

        timer = threading.Timer(1, release.set)
        timer.start()
        try:
            with patch.object(self.store, 'open_run', side_effect=slow_open):
                self.assertTrue(asyncio.run(drive()), 'store attachment blocked the host event loop')
        finally:
            release.set()
            timer.cancel()

    def test_cancelled_attachment_waits_for_its_store_writer(self):
        import asyncio
        import threading
        from unittest.mock import patch
        from rlstack.runner.loop import attach_run

        entered, release, finished = (threading.Event() for _ in range(3))
        original = self.store.open_run

        def slow_open(*args, **kwargs):
            entered.set()
            release.wait(2)
            result = original(*args, **kwargs)
            finished.set()
            return result

        async def drive():
            task = asyncio.create_task(attach_run(
                self.store, 'pending', manifest={'run_id': 'pending'}, subdir=None))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(task.done(), 'custody ended before the store writer')
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(finished.is_set())
                self.assertIsNotNone(self.store.peek_manifest('pending'))
            finally:
                release.set()

        with patch.object(self.store, 'open_run', side_effect=slow_open):
            asyncio.run(drive())

    def test_completes_and_commits_every_update(self) -> None:
        spec = arith_spec(self.train)
        report, engine = self.run_spec(spec)

        self.assertEqual((report.completed, report.extent), (4, "train"))
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

    def test_replayed_prompts_are_tokenized_once_per_trainer(self) -> None:
        """Repeated sealed rows reuse tokens, while a fresh trainer asks anew."""
        from unittest.mock import patch
        from common import sealed
        from rlstack import Host, HostService, LocalTransport, RemotePool, trajectory_to_row

        rows = [trajectory_to_row(sealed("fact", content)) for content in ("4", "5")]
        data = self.store.cas_put("".join(json.dumps(row) + "\n" for row in rows).encode())
        plan = RunPlan(tuple(WavePlan((GroupPlan("fact", (
            Replay(data + "#0"), Replay(data + "#1"))),)) for _ in range(4)))
        spec = arith_spec(self.train, gen=None,
                          plans=Plans(train=self.store.cas_put(encode(plan))))
        serving = FakeEngine()
        host = Host("tokenizer", engines=(serving,), learner=None, store=self.store)
        remote = RemotePool(LocalTransport(HostService(host)))
        with patch.object(serving, "tokenize", wraps=serving.tokenize) as tokenize:
            report = run_experiment(spec, SCHEMA, self.store, remote, FakeLearner())
        self.assertEqual(report.completed, 4)
        tokenize.assert_called_once_with("What is 2+2?")

        # No cached result is shared with another trainer or engine.
        with tempfile.TemporaryDirectory() as root:
            other, _, _ = arith_store(root)
            other.cas_put(self.store.cas_get(data))
            other.cas_put(encode(plan))
            engine = FakeEngine()
            with patch.object(engine, "tokenize", wraps=engine.tokenize) as fresh:
                again = run_experiment(spec, SCHEMA, other, engine, FakeLearner())
            fresh.assert_called_once_with("What is 2+2?")
            self.assertEqual(report.run_id, again.run_id)
            from test_resume import snapshot
            self.assertEqual(snapshot(self.store, report.run_id),
                             snapshot(other, again.run_id))

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

    def test_warm_start_never_sweeps_the_parents_uncommitted_work(self):
        from rlstack.runner.loop import warm_start_source, sealed_payloads
        parent = self.store.open_run('parent', manifest={'run_id': 'parent'})
        parent.write_blob('adapters', 'pi', 1, b'sealed')
        parent.append_ledger({'update': 1, 'versions': {'pi': 1}})
        parent.write_blob('adapters', 'pi', 2, b'in flight')
        parent.write_wave(2, [{'work': 'in flight'}])
        before = {key: self.store._read(key) for key in self.store._list('runs/parent')}
        source, version = warm_start_source(WarmStart('store://parent@1'), self.store)
        self.assertEqual(sealed_payloads(source, version, {}, {'pi'}), {'pi': b'sealed'})
        self.assertEqual(before, {key: self.store._read(key) for key in self.store._list('runs/parent')})

    def test_cas_and_store_warm_starts_restore_the_same_mapped_state(self):
        from rlstack.runner.interfaces import Emitted
        from rlstack.runner.remote import encode_emitted

        parent, _ = self.run_spec(arith_spec(self.train))
        run = self.store.open_run(parent.run_id)
        state = Emitted(adapters={"pi": run.read_blob("adapters", "pi", 4)},
                        optim={"pi": run.read_blob("optim", "pi", 4)})
        encoded = json.dumps(encode_emitted(state)).encode()
        uri = self.store.cas_put(encoded)
        policy = PolicySpec(base="Qwen/Qwen3-0.6B",
                            bank={"ghost": lora("layers.0-3.self_attn.*", r=16)})
        loaded = []
        for address in (f"store://{parent.run_id}@4", uri):
            for moments in ("fresh", "load"):
                _, engine = self.run_spec(arith_spec(self.train, policy=policy,
                    init=WarmStart(address, optim=moments, map={"pi": "ghost"})))
                loaded.append(engine.bundle_log[0])
        self.assertTrue(all(bundle == loaded[0] for bundle in loaded))
        self.assertEqual(self.store.cas_get(uri), encoded)

    def test_cas_initializes_a_frozen_bank_without_a_synthetic_parent(self):
        from rlstack.runner.interfaces import Emitted, EntryInstall
        from rlstack.runner.loop import initial_adapters
        from rlstack.runner.remote import encode_emitted

        uri = self.store.cas_put(json.dumps(encode_emitted(
            Emitted({"source": b"materialized adapter"}, {}))).encode())
        entry = EntryInstall("pi", "lora", {}, False, ())
        payloads = initial_adapters((entry,), WarmStart(uri, map={"source": "pi"}), self.store)
        self.assertEqual(payloads, {"pi": b"materialized adapter"})
        self.assertEqual(self.store.list_runs(), [])

    def test_cas_refuses_missing_requested_moments(self):
        from rlstack.runner.interfaces import Emitted
        from rlstack.runner.loop import cas_warm_start
        from rlstack.runner.remote import encode_emitted

        uri = self.store.cas_put(json.dumps(encode_emitted(
            Emitted({"source": b"adapter"}, {}))).encode())
        with self.assertRaisesRegex(FileNotFoundError, "no optimizer moments"):
            cas_warm_start(WarmStart(uri, optim="load", map={"source": "pi"}),
                           self.store, {"pi"}, ["pi"])
        state = cas_warm_start(WarmStart(uri, map={"source": "pi"}),
                               self.store, {"pi"}, ["pi"])
        self.assertEqual(state.adapters, {"pi": b"adapter"})
        self.assertEqual(state.optim, {})


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
            topology=Topology(hosts=(
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
        self.assertEqual((report.completed, report.extent), (4, "train"))


class GenerationOnlyTest(unittest.TestCase):
    """ADR 0006 Part B: a run is daemons with resources, and the smallest run
    is a Generator alone — no algo, no learner, no Trainer, no ledger."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.store, self.train, _ = arith_store(self._tmp.name)

    def generate(self, **overrides):
        """The teacher run: seals one rollout per planned wave, with no
        learner handed to it at all."""
        spec = generation_spec(self.train, **overrides)
        return run_experiment(spec, SCHEMA, self.store, FakeEngine(), None), spec

    def test_it_runs_and_seals_a_rollout_per_planned_wave(self) -> None:
        report, spec = self.generate()
        self.assertEqual((report.completed, report.extent), (4, "rollout"))
        self.assertIsNone(report.resumed_from)

        run = self.store.open_run(report.run_id)
        self.assertEqual(run.read_ledger(), [])          # nothing commits
        for index in range(1, 5):
            self.assertEqual(len(run.read_rollout(index)), 4)
        with self.assertRaises(FileNotFoundError):
            run.read_rollout(5)
        # ...and it is self-describing: manifest, dictionary, its own plan
        self.assertEqual(
            self.store.peek_manifest(report.run_id)["run_id"], report.run_id)
        self.assertIsNotNone(self.store.peek_dictionary(report.run_id))
        self.assertIsNotNone(self.store.peek_plan(report.run_id, "rollout"))
        self.assertIsNone(self.store.peek_plan(report.run_id, "train"))

    def test_the_store_says_it_is_done_without_a_ledger(self) -> None:
        """The obligation the reaper rests on (Q8): done-ness is readable off
        a run that never committed anything."""
        report, _ = self.generate()
        self.assertTrue(run_done(self.store, report.run_id))
        self.assertEqual(run_progress(self.store, report.run_id),
                         RunProgress("rollout", 4, 4))

    def test_it_needs_a_generator_and_nothing_else(self) -> None:
        _, spec = self.generate()
        needs = needs_of(spec)
        self.assertEqual([need.daemon.__name__ for need in needs], ["Generator"])
        self.assertEqual((needs[0].plan, needs[0].pools), ("rollout", ("main",)))
        self.assertFalse(needs[0].learner)
        self.assertIsNone(needs[0].buffer)      # unpaced: nothing consumes it

    def test_a_training_spec_plans_exactly_what_it_always_planned(self) -> None:
        """The other half of the same promise: needs_of over a training spec
        is the Trainer + Generator it has always been, paced by the lag."""
        needs = needs_of(arith_spec(self.train))
        self.assertEqual([need.daemon.__name__ for need in needs],
                         ["Trainer", "Generator"])
        self.assertTrue(needs[0].learner)
        self.assertEqual(needs[1].buffer, 0)

    def test_a_spec_that_needs_a_trainer_refuses_a_missing_learner(self) -> None:
        with self.assertRaises(ValueError) as caught:
            run_experiment(arith_spec(self.train), SCHEMA, self.store,
                           FakeEngine(), None)
        self.assertIn("learner", str(caught.exception))

    def test_a_student_trains_by_replaying_the_teachers_rollouts(self) -> None:
        """ADR 0005's shape, on fakes: an empty-bank teacher generates, and a
        lora student's train plan replays `store://<teacher>/rollouts/<r>#<i>`
        — both runs in one store, the student never sampling anything."""
        teacher, _ = self.generate()
        rows = self.store.open_run(teacher.run_id).read_rollout(1)

        sft_plan = RunPlan(tuple(
            WavePlan((GroupPlan(f"g{u}", tuple(
                Replay(f"store://{teacher.run_id}/rollouts/{u}#{i}")
                for i in range(len(rows)))),))
            for u in range(1, 5)))
        self.store.cas_put(encode(sft_plan))
        student = arith_spec(
            self.train, gen=None,
            plans=Plans(train=cas_uri(encode(sft_plan))))

        report = run_experiment(student, SCHEMA, self.store, FakeEngine(),
                                FakeLearner())
        self.assertEqual((report.completed, report.extent), (4, "train"))
        run = self.store.open_run(report.run_id)
        self.assertEqual([e["update"] for e in run.read_ledger()], [1, 2, 3, 4])
        # the student sampled NOTHING: every trajectory it trained on is the
        # teacher's, realized into its own waves under its own group keys
        self.assertEqual([row["task"]["id"] for row in run.read_wave(1)],
                         [row["task"]["id"] for row in rows])
        with self.assertRaises(FileNotFoundError):
            run.read_rollout(1)
