"""Wave collection and the SampleClient (rlstack.runner.rollout)."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from rlstack import (
    Bundle, FakeEngine, Message, Role, SamplingSpec, Task, collect_wave, reward,
)
from rlstack.runner.rollout import SampleClient, run_episode

BUNDLE = Bundle(bundle_id="bundle:test0000", policy_version={"pi": 0})
SAMPLING = SamplingSpec()


def engine(**kwargs: Any) -> FakeEngine:
    e = FakeEngine(**kwargs)
    e.add_bundle(BUNDLE)
    return e


def go(coro):
    return asyncio.run(coro)


@reward("rollout_dup_reward", components=("reward",))  # collides with verifier
async def _dup(traj: Any, llm: Any) -> dict[str, float]:
    return {"reward": 0.0}


class SampleClientTest(unittest.TestCase):
    def sample(self, **engine_kwargs: Any):
        client = SampleClient(engine(**engine_kwargs), SAMPLING, BUNDLE,
                              rollout_seed=7)
        return go(client.sample([Message(Role.USER, "What is 12+34?")]))

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
        client = SampleClient(engine(), SAMPLING, BUNDLE, rollout_seed=7)
        msgs = [Message(Role.USER, "What is 12+34?")]
        first = go(client.sample(msgs))
        second = go(client.sample(msgs))
        self.assertNotEqual(first.seed, second.seed)

    def test_extras_become_token_columns(self) -> None:
        turn = self.sample(record_draws=True)
        draws = turn.token_extras["adapter_draw"]
        self.assertEqual(len(draws), len(turn.token_ids))
        self.assertTrue(all(isinstance(d, int) for d in draws))

    def test_unregistered_bundle_is_refused(self) -> None:
        bad = Bundle(bundle_id="bundle:unknown0", policy_version={})
        client = SampleClient(engine(), SAMPLING, bad, rollout_seed=7)
        with self.assertRaises(RuntimeError):
            go(client.sample([Message(Role.USER, "hi")]))


class RunEpisodeTest(unittest.TestCase):
    def test_env_plus_reward_seals(self) -> None:
        task = Task("t0", "What is 2+3?", {"answer": 5})
        client = SampleClient(engine(p_correct=1.0), SAMPLING, BUNDLE, 7)
        traj = go(run_episode("math_single_turn", ("verifier",), task, client))
        self.assertTrue(traj.sealed)
        self.assertEqual(traj.reward_components["reward"], 1.0)
        self.assertEqual(traj.turns[0].message.content, "5")

    def test_component_collision_is_a_wiring_error(self) -> None:
        task = Task("t0", "What is 2+3?", {"answer": 5})
        client = SampleClient(engine(), SAMPLING, BUNDLE, 7)
        with self.assertRaises(ValueError):
            go(run_episode("math_single_turn",
                           ("verifier", "rollout_dup_reward"), task, client))


class CollectWaveTest(unittest.TestCase):
    TASKS = [Task(f"t{i}", f"What is {i}+{i}?", {"answer": 2 * i})
             for i in range(8)]

    def collect(self, update: int = 1, e: FakeEngine | None = None):
        return go(collect_wave(
            update,
            env_name="math_single_turn",
            reward_names=("verifier",),
            sampling=SAMPLING,
            tasks=self.TASKS,
            group_size=4,
            rollouts_per_wave=16,
            engine=e or engine(),
            bundle=BUNDLE,
            master=17,
        ))

    def test_group_structure(self) -> None:
        wave = self.collect()
        self.assertEqual(len(wave), 16)
        self.assertEqual(len(wave.groups), 4)             # 4 distinct tasks
        self.assertEqual({len(g) for g in wave.groups}, {4})  # group_size each
        for group in wave.groups:
            self.assertEqual({t.task.id for t in group.trajectories}, {group.key})
        self.assertTrue(all(t.sealed for t in wave.trajectories))

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
            go(collect_wave(1, env_name="math_single_turn",
                            reward_names=("verifier",), sampling=SAMPLING,
                            tasks=self.TASKS, group_size=3, rollouts_per_wave=16,
                            engine=engine(), bundle=BUNDLE, master=17))

    def test_too_few_tasks_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            go(collect_wave(1, env_name="math_single_turn",
                            reward_names=("verifier",), sampling=SAMPLING,
                            tasks=self.TASKS[:2], group_size=4,
                            rollouts_per_wave=16, engine=engine(),
                            bundle=BUNDLE, master=17))


if __name__ == "__main__":
    unittest.main()
