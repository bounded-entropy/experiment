"""The sampling side: wave collection and the EnginePoolClient (rlstack.runner.sampling)."""

from __future__ import annotations

import asyncio
import unittest

from rlstack import (
    Bundle, FakeEngine, Message, Role, SamplingSpec, Task, Trajectory,
    collect_wave,
)
from rlstack.runner.sampling import EnginePoolClient
from rlstack.runner.sampling import run_episode

BUNDLE = Bundle(bundle_id="bundle:test0000", policy_version={"pi": 0})
HINT_BUNDLE = Bundle(bundle_id="bundle:base:hints", policy_version={})
SAMPLING = SamplingSpec()


def make_pools(**engine_kwargs):
    main = FakeEngine(**engine_kwargs)
    main.add_bundle(BUNDLE)
    hints = FakeEngine(p_correct=1.0)
    hints.add_bundle(HINT_BUNDLE)
    return {"main": (main, BUNDLE), "hints": (hints, HINT_BUNDLE)}


def go(coro):
    return asyncio.run(coro)


class EngineSampleClientTest(unittest.TestCase):
    def client(self, **engine_kwargs) -> EnginePoolClient:
        return EnginePoolClient(make_pools(**engine_kwargs), SAMPLING,
                                  episode_seed=7)

    def sample(self, **engine_kwargs):
        return go(self.client(**engine_kwargs).sample(
            [Message(Role.USER, "What is 12+34?")]))

    def test_assembles_a_turn_from_the_stream(self) -> None:
        turn = self.sample()
        self.assertEqual(turn.token_ids,
                         tuple(ord(c) for c in turn.message.content))
        self.assertEqual(len(turn.behavior_logprobs), len(turn.token_ids))
        self.assertEqual(turn.finish, "eos")
        self.assertEqual(turn.bundle_id, BUNDLE.bundle_id)
        self.assertEqual(dict(turn.policy_version), {"pi": 0})

    def test_deterministic_given_seed(self) -> None:
        self.assertEqual(self.sample(), self.sample())

    def test_calls_get_distinct_seeds(self) -> None:
        client = self.client()
        msgs = [Message(Role.USER, "What is 12+34?")]
        first = go(client.sample(msgs))
        second = go(client.sample(msgs))
        self.assertNotEqual(first.seed, second.seed)

    def test_pool_reaches_other_engines_and_shares_seeds(self) -> None:
        client = self.client()
        msgs = [Message(Role.USER, "What is 12+34?")]
        hint = go(client.pool("hints").sample(msgs))     # p_correct=1.0 pool
        self.assertEqual(hint.message.content, "46")
        self.assertEqual(hint.bundle_id, HINT_BUNDLE.bundle_id)
        main = go(client.sample(msgs))
        # sibling clients share ONE seed sequence: no seed reuse across pools
        self.assertNotEqual(hint.seed, main.seed)

    def test_unknown_pool_is_a_helpful_error(self) -> None:
        with self.assertRaises(KeyError) as caught:
            self.client().pool("judges")
        self.assertIn("hints", str(caught.exception))

    def test_extras_become_token_columns(self) -> None:
        turn = self.sample(record_draws=True)
        draws = turn.token_extras["adapter_draw"]
        self.assertEqual(len(draws), len(turn.token_ids))

    def test_unregistered_bundle_is_refused(self) -> None:
        bad = Bundle(bundle_id="bundle:unknown0", policy_version={})
        pools = {"main": (make_pools()["main"][0], bad)}
        client = EnginePoolClient(pools, SAMPLING, episode_seed=7)
        with self.assertRaises(RuntimeError):
            go(client.sample([Message(Role.USER, "hi")]))


class RunEpisodeTest(unittest.TestCase):
    def test_env_runs_and_seals(self) -> None:
        task = Task("t0", "What is 2+3?", {"answer": 5})
        client = EnginePoolClient(make_pools(p_correct=1.0), SAMPLING, 7)
        traj = go(run_episode("math_single_turn", task, client))
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.turns[0].message.content, "5")


class CollectWaveTest(unittest.TestCase):
    TASKS = [Task(f"t{i}", f"What is {i}+{i}?", {"answer": 2 * i})
             for i in range(8)]

    def collect(self, update: int = 1):
        return go(collect_wave(
            update,
            env_name="math_single_turn",
            sampling=SAMPLING,
            tasks=self.TASKS,
            group_size=4,
            trajectories_per_wave=16,
            routes=make_pools(),
            master=17,
        ))

    def test_group_structure(self) -> None:
        wave = self.collect()
        self.assertEqual(len(wave), 16)
        self.assertEqual(len(wave.groups), 4)             # 4 distinct tasks
        self.assertEqual({len(g) for g in wave.groups}, {4})  # group_size each
        for group in wave.groups:
            self.assertEqual({t.task.id for t in group.trajectories}, {group.key})
        self.assertTrue(all(isinstance(t, Trajectory) for t in wave.trajectories))

    def test_deterministic_across_collections(self) -> None:
        rows_a = [t.turns[0].token_ids for t in self.collect().trajectories]
        rows_b = [t.turns[0].token_ids for t in self.collect().trajectories]
        self.assertEqual(rows_a, rows_b)

    def test_updates_get_different_waves(self) -> None:
        seeds_1 = [t.turns[0].seed for t in self.collect(update=1).trajectories]
        seeds_2 = [t.turns[0].seed for t in self.collect(update=2).trajectories]
        self.assertNotEqual(seeds_1, seeds_2)

    def test_indivisible_wave_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            go(collect_wave(1, env_name="math_single_turn", sampling=SAMPLING,
                            tasks=self.TASKS, group_size=3, trajectories_per_wave=16,
                            routes=make_pools(), master=17))

    def test_too_few_tasks_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            go(collect_wave(1, env_name="math_single_turn", sampling=SAMPLING,
                            tasks=self.TASKS[:2], group_size=4,
                            trajectories_per_wave=16, routes=make_pools(), master=17))


if __name__ == "__main__":
    unittest.main()
