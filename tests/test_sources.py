"""The WaveFeed family (rlstack.runner.sources): live, replay, static.

A feed's contract: make update u's rows exist in the run's own rollouts/
and return them (or None when live data has not arrived yet)."""

from __future__ import annotations

import tempfile
import unittest

from common import arith_spec, arith_store, sealed
from rlstack import (
    FakeEngine,
    FakeLearner,
    LiveFeed,
    ReplayFeed,
    RolloutSource,
    Schedule,
    StaticFeed,
    StoreError,
    fake_qwen_schema,
    run_experiment,
    trajectory_to_row,
    feed_for,
)
from rlstack.spec.specs import AlgoSpec, OptimSpec

import json
from dataclasses import replace

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


class SourcesTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.store, self.train, self.heldout = arith_store(self.root)

    def run_live_parent(self):
        report = run_experiment(arith_spec(self.train), SCHEMA, self.store,
                                FakeEngine(), FakeLearner())
        return report.run_id

    def offline_algo(self, n_updates: int, rollouts_per_wave: int = 4,
                     group_size: int = 2) -> AlgoSpec:
        return AlgoSpec(loss="grpo", post=("verifier", "grpo_advantage"),
                        optim=OptimSpec("adamw", lr=1e-5),
                        schedule=Schedule(group_size=group_size,
                                          rollouts_per_wave=rollouts_per_wave,
                                          n_updates=n_updates,
                                          microbatch_tokens=64))

    # ---- factory ------------------------------------------------------------

    def scratch_run(self):
        return self.store.open_run("scratch", manifest={"run_id": "scratch"})

    def test_factory_dispatches_on_the_source_string(self) -> None:
        run = self.scratch_run()
        live = feed_for(arith_spec(self.train), self.store, run)
        self.assertIsInstance(live, LiveFeed)
        self.assertIsNone(live.obtain(1))     # the Generator has not written yet

        parent = self.run_live_parent()
        replay_spec = arith_spec(self.train, gen=None, eval=None,
                                 rollouts=RolloutSource(f"store://{parent}"),
                                 algo=self.offline_algo(3))
        self.assertIsInstance(
            feed_for(replay_spec, self.store, run), ReplayFeed)

        rows = [trajectory_to_row(sealed(f"t{i}")) for i in range(6)]
        uri = self.store.cas_put(
            "".join(json.dumps(r) + "\n" for r in rows).encode())
        static_spec = arith_spec(self.train, gen=None, eval=None,
                                 rollouts=RolloutSource(uri),
                                 algo=self.offline_algo(3))
        self.assertIsInstance(
            feed_for(static_spec, self.store, run), StaticFeed)

    # ---- replay -------------------------------------------------------------

    def test_replay_consumes_the_parent_waves_verbatim(self) -> None:
        parent = self.run_live_parent()
        spec = arith_spec(self.train, gen=None, eval=None,
                          rollouts=RolloutSource(f"store://{parent}"),
                          algo=self.offline_algo(3))
        report = run_experiment(spec, SCHEMA, self.store,
                                FakeEngine(), FakeLearner())

        self.assertNotEqual(report.run_id, parent)
        child = self.store.open_run(report.run_id)
        parent_run = self.store.open_run(parent)
        for update in (1, 2, 3):
            # self-contained: the child persisted what it trained on,
            # and it is exactly the parent's sealed wave
            self.assertEqual(child.read_rollouts(update),
                             parent_run.read_rollouts(update))
        self.assertEqual(len(child.read_ledger()), 3)

    def test_replay_beyond_the_parent_is_a_clear_error(self) -> None:
        parent = self.run_live_parent()          # parent has 4 updates
        spec = arith_spec(self.train, gen=None, eval=None,
                          rollouts=RolloutSource(f"store://{parent}"),
                          algo=self.offline_algo(n_updates=9))
        with self.assertRaises(ValueError) as caught:
            run_experiment(spec, SCHEMA, self.store, FakeEngine(), FakeLearner())
        self.assertIn("no sealed rollouts for update 5", str(caught.exception))

    def test_replay_of_a_missing_run_fails_at_setup(self) -> None:
        with self.assertRaises(StoreError):
            ReplayFeed(self.store, "store://nope00000000", self.scratch_run())

    # ---- static -------------------------------------------------------------

    def make_dataset(self, n: int) -> str:
        rows = [trajectory_to_row(sealed(f"t{i}")) for i in range(n)]
        return self.store.cas_put(
            "".join(json.dumps(r) + "\n" for r in rows).encode())

    def test_static_dataset_trains_in_deterministic_slices(self) -> None:
        uri = self.make_dataset(6)
        spec = arith_spec(self.train, gen=None, eval=None,
                          rollouts=RolloutSource(uri),
                          algo=self.offline_algo(3))
        report = run_experiment(spec, SCHEMA, self.store,
                                FakeEngine(), FakeLearner())
        run = self.store.open_run(report.run_id)
        self.assertEqual(len(run.read_ledger()), 3)
        # singleton groups; update 3 wraps around the 6-row dataset
        first = run.read_rollouts(1)
        third = run.read_rollouts(3)
        self.assertEqual([r["group"] for r in first],
                         ["row-000000", "row-000001", "row-000002", "row-000003"])
        self.assertEqual([r["group"] for r in third],
                         ["row-000002", "row-000003", "row-000004", "row-000005"])

    def test_static_dataset_smaller_than_a_wave_is_rejected(self) -> None:
        uri = self.make_dataset(2)
        with self.assertRaises(ValueError):
            StaticFeed(self.store, uri, self.scratch_run(), rollouts_per_wave=4)


if __name__ == "__main__":
    unittest.main()
