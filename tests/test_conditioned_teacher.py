"""ADR 0005: a steer distilled from a prompt-conditioned teacher.

The shape, end to end on fakes: a task carries the system block as
`meta["hint"]`; the `conditioned_teacher` environment samples under hint +
prompt and seals the hint OUT; a generation-only run with an EMPTY bank
leaves those rollouts; an SFT run with a steer entry replays them by
`store://<teacher>/rollouts/<r>#<i>` and commits, byte-identically across a
crash. Beside it the measurement channel: `conditioned_teacher_logprobs`
scores the same walk with the hint at the head of the context, and
`reverse_kl` reduces it against the record.
"""

from __future__ import annotations

import asyncio
import unittest

from rlstack import (
    Bundle, EnginePoolClient, FakeEngine, Message, Role, SamplingSpec, Task,
)

BUNDLE = Bundle(bundle_id="bundle:teacher0", policy_version={})
HINT = ("You are a helpful assistant who is deeply preoccupied with "
        "happiness.\n\n")


def go(coro):
    return asyncio.run(coro)


class RecordingEngine(FakeEngine):
    """A fake that keeps every context it was asked to continue — the only
    way to see, from outside, what conditioning a request carried."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.contexts: list[tuple[str, ...]] = []

    async def sample_tokens(self, messages, *args, **kwargs):
        self.contexts.append(tuple(m.content for m in messages))
        async for event in super().sample_tokens(messages, *args, **kwargs):
            yield event

    async def score_tokens(self, messages, *args, **kwargs):
        self.contexts.append(tuple(m.content for m in messages))
        return await super().score_tokens(messages, *args, **kwargs)


def a_task(task_id: str = "no-robots/1", prompt: str = "What is 2+2?") -> Task:
    return Task(task_id, prompt, {"hint": HINT, "concept": "happiness"})


def a_client(engine: FakeEngine) -> EnginePoolClient:
    engine.add_bundle(BUNDLE)
    return EnginePoolClient({"main": (engine, BUNDLE)}, SamplingSpec(),
                            episode_seed=5)


class ConditionedTeacherEnvTest(unittest.TestCase):
    """The environment's whole claim: the hint conditions the request and is
    absent from the record the student will train on."""

    def rollout(self):
        from rlstack.registry import ENVS

        engine = RecordingEngine()
        env = ENVS.get("conditioned_teacher").instance
        task = a_task()
        return engine, task, go(env.run(a_client(engine), task))

    def test_the_hint_conditions_the_request(self) -> None:
        engine, task, _ = self.rollout()
        self.assertEqual(engine.contexts, [(HINT, task.prompt)])

    def test_the_sealed_stream_is_prompt_and_completion_alone(self) -> None:
        _, task, rollout = self.rollout()
        traj = rollout.seal()
        self.assertEqual([m.content for m in traj.messages],
                         [task.prompt, traj.turns[0].message.content])
        self.assertEqual([m.role for m in traj.messages],
                         [Role.USER, Role.ASSISTANT])
        self.assertNotIn(HINT, traj.text)

    def test_the_hint_survives_as_provenance(self) -> None:
        _, _, rollout = self.rollout()
        self.assertEqual(rollout.seal().env_extras["hint"], HINT)

    def test_it_records_no_window_because_its_bank_is_empty(self) -> None:
        """The teacher run serves the bare base; nothing steers, so nothing
        is recorded — and ADR 0005 Q3 makes that replayable rather than a
        bug (the steer's default is every position)."""
        _, _, rollout = self.rollout()
        self.assertEqual(dict(rollout.turns[0].turn_extras), {})


class SingleTurnEnvTest(unittest.TestCase):
    def test_it_is_math_single_turns_body_under_an_honest_name(self) -> None:
        from rlstack.registry import ENVS

        engine = RecordingEngine()
        task = Task("t", "What is 2+2?", {})
        rollout = go(ENVS.get("single_turn").instance.run(a_client(engine), task))
        self.assertEqual(engine.contexts, [(task.prompt,)])
        self.assertEqual([m.content for m in rollout.messages],
                         [task.prompt, rollout.turns[0].message.content])

    def test_the_math_name_still_exists_for_the_runs_that_hash_it(self) -> None:
        from rlstack.registry import ENVS

        self.assertIsNotNone(ENVS.get("math_single_turn"))
        self.assertIsNotNone(ENVS.get("single_turn"))


if __name__ == "__main__":
    unittest.main()
