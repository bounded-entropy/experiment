"""The dreams pipeline's seams (ADR 0018): realize stamps a leaf's role into
the row's facts, the loss dispatches by role, the processors reward dreams by
the contrast and grade answers, and the answer environment samples under its
own greedy sampling with a Route directive."""

from __future__ import annotations

import asyncio
import copy
import unittest

from common import make_turn, sealed
from rlstack import Bundle, FakeEngine, Group, Message, Role, Rollout, SamplingSpec, Task, Wave, run_pipeline
from rlstack.data.plan import EVAL, TRAIN, GroupPlan, Replay, Sample, WavePlan, route_of_role
from rlstack.data.trajectory import trajectory_to_row, wave_from_rows
DREAMER = "custom-entry"
from rlstack.runner.assemble import sample_wave, stamped
from rlstack.runner.traffic import EnginePoolClient


try:
    import torch
except ImportError:
    torch = None

BUNDLE = Bundle(bundle_id="bundle:dreams000", policy_version={"pi": 0},
                adapter_types={"pi": "dream_bank"})


def pools():
    engine = FakeEngine()
    engine.add_bundle(BUNDLE)
    return engine, {"main": (engine, BUNDLE)}


def go(coro):
    return asyncio.run(coro)


def text_row(task_id: str = "squad/text/p1/h1", route: str | None = None,
             meta: dict | None = None) -> dict:
    """A supervised text row: the arrival's tokens as one generated turn."""
    turn = make_turn("the sky is blue", tuple(ord(c) for c in "the sky is blue"),
                     turn_extras={"route": route} if route else {})
    rollout = Rollout(task=Task(task_id, "Passage:\n", {"kind": "text", **(meta or {})}),
                      messages=[Message(Role.USER, "Passage:\n"), turn.message], turns=[turn])
    return trajectory_to_row(rollout.seal())


class RoleVocabularyTest(unittest.TestCase):
    def test_train_and_eval_stamp_no_route_and_a_set_name_is_its_route(self):
        self.assertIsNone(route_of_role(TRAIN))
        self.assertIsNone(route_of_role(EVAL))
        self.assertEqual(route_of_role("memory:03"), "memory:03")
        self.assertEqual(route_of_role(DREAMER), DREAMER)


class StampedRowTest(unittest.TestCase):
    def test_a_train_leaf_leaves_the_sealed_row_untouched_and_uncopied(self):
        row = text_row()
        self.assertIs(stamped(row, TRAIN), row)

    def test_a_memory_role_stamps_role_and_route_on_a_copy(self):
        row = text_row(route=DREAMER)
        before = copy.deepcopy(row)
        out = stamped(row, "memory:03")
        self.assertEqual(row, before)                                # the cached record is intact
        self.assertEqual(out["turns"][0]["turn_extras"]["role"], "memory:03")
        self.assertEqual(out["turns"][0]["turn_extras"]["route"], "memory:03")

    def test_an_eval_leaf_keeps_the_recorded_route(self):
        row = text_row(route="memory:07")
        out = stamped(row, EVAL)
        self.assertEqual(out["turns"][0]["turn_extras"]["role"], EVAL)
        self.assertEqual(out["turns"][0]["turn_extras"]["route"], "memory:07")

    def test_the_facts_survive_the_row_round_trip(self):
        wave = wave_from_rows([dict(stamped(text_row(), "memory:01"), group="g")])
        traj = wave.trajectories[0]
        self.assertEqual(traj.turns[0].turn_extras["role"], "memory:01")
        self.assertEqual(traj.turns[0].turn_extras["route"], "memory:01")
