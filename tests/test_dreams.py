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
from rlstack.policy.adapters.dream_bank import BASE, DREAMER, Route
from rlstack.runner.assemble import sample_wave, stamped
from rlstack.runner.traffic import EnginePoolClient
from rlstack.training.post.dream_effect import contrast, role_of, route_of

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
        self.assertEqual(role_of(traj), "memory:01")
        self.assertEqual(route_of(traj), "memory:01")


class ContrastTest(unittest.TestCase):
    def test_the_group_difference_reads_planted_dream_effects_against_the_other_groups(self):
        memories = [f"memory:{j:02d}" for j in range(16)]
        assignment = {str(k): memories[4 * k:4 * k + 4] for k in range(4)}   # the partition
        effects = {"0": -0.4, "1": 0.1, "2": -0.05, "3": 0.3}
        nll = {m: 2.0 + effects[str(j // 4)] for j, m in enumerate(memories)}
        reward = contrast(nll, {"assignment": assignment})
        for k in range(4):
            rest = [effects[str(o)] for o in range(4) if o != k]
            self.assertAlmostEqual(reward[k], sum(rest) / 3 - effects[str(k)], places=9)
        self.assertEqual(max(range(4), key=lambda k: reward[k]), 0)     # the helpful dream reads highest
        # a group nobody scored, or a partition with nothing outside it, is worth nothing
        self.assertEqual(contrast({"memory:00": 1.0}, {"assignment": {"0": ["memory:00"]}}), [0.0])
        self.assertEqual(contrast({"memory:00": 1.0}, {"assignment": {"0": ["memory:07"]}}), [0.0])



class ProcessorsTest(unittest.TestCase):
    def dream_row(self, route: str, role: str) -> dict:
        turn = make_turn("a dream", tuple(ord(c) for c in "a dream"), turn_extras={"route": route})
        rollout = Rollout(task=Task("squad/dream/p1/h1", "Output content you think would be useful.\n",
                                    {"kind": "dream", "hint": "Passage:\nthe sky is blue\n"}),
                          messages=[Message(Role.USER, "Output content you think would be useful.\n"),
                                    turn.message], turns=[turn])
        return dict(stamped(trajectory_to_row(rollout.seal()), role), group="arrival")

    def qa_row(self, answer_text: str, route: str) -> dict:
        turn = make_turn(answer_text, tuple(ord(c) for c in answer_text), turn_extras={"route": route})
        rollout = Rollout(task=Task(f"squad/qa/p1/q1/{route}", "Question: what color?\nAnswer:",
                                    {"kind": "qa", "route": route, "answers": ["blue"],
                                     "answer_token_ids": [ord(c) for c in " blue"],
                                     "question": "what color?", "judge": True}),
                          messages=[Message(Role.USER, "Question: what color?\nAnswer:"),
                                    turn.message], turns=[turn])
        return dict(stamped(trajectory_to_row(rollout.seal()), EVAL), group="arrival")

    def test_dream_effect_scores_each_text_row_under_its_route_and_rewards_dreams(self):
        engine, routes = pools()
        assignment = {"0": ["memory:00"], "1": ["memory:01"]}
        rows = [dict(stamped(text_row(meta={"contrast": {"assignment": assignment}}),
                             f"memory:{j:02d}"), group="arrival") for j in range(2)]
        rows += [self.dream_row(DREAMER, DREAMER), self.dream_row(DREAMER, DREAMER)]
        wave = wave_from_rows(rows)
        columns = go(run_pipeline(("dream_effect", "dream_advantage"), wave, routes,
                                  SamplingSpec(), 11, 1))
        self.assertEqual(len(columns["prequential_nll"]), 4)
        self.assertGreater(columns["prequential_nll"][0], 0.0)
        self.assertEqual(columns["prequential_nll"][2], 0.0)          # a dream row is not probed
        # the base's surprise rides every text row, the same value per text
        self.assertGreater(columns["prequential_base"][0], 0.0)
        self.assertEqual(columns["prequential_base"][0], columns["prequential_base"][1])
        self.assertEqual(columns["prequential_base"][2], 0.0)
        seen = [d.name for directives in engine.directives_seen for d in directives]
        self.assertEqual(seen, ["memory:00", "memory:01", "base"])      # one pass per text row, one under the base per text
        self.assertEqual(columns["advantage"][0], 0.0)
        self.assertEqual(columns["advantage"][1], 0.0)
        self.assertAlmostEqual(columns["advantage"][2] + columns["advantage"][3], 0.0, places=9)

    def test_gold_answers_are_scored_under_their_route(self):
        engine, routes = pools()
        wave = wave_from_rows([self.qa_row("blue", "memory:02"), self.qa_row("green", "memory:03")])
        columns = go(run_pipeline(("gold_logprob",), wave, routes, SamplingSpec(), 11, 1))
        self.assertTrue(all(v < 0 for v in columns["gold_lp"]))
        seen = [d.name for directives in engine.directives_seen for d in directives]
        self.assertEqual(seen, ["memory:02", "memory:03"])

    def test_dream_diagnostics_score_dreams_under_the_bare_base_with_the_hint(self):
        engine, routes = pools()
        wave = wave_from_rows([self.dream_row(DREAMER, "memory:00")])
        columns = go(run_pipeline(("dream_diagnostics",), wave, routes, SamplingSpec(), 11, 1))
        self.assertGreater(columns["dream_base_nll"][0], 0.0)
        self.assertEqual(columns["dream_tokens"][0], float(len("a dream")))
        self.assertEqual([d.name for ds in engine.directives_seen for d in ds], [BASE])


class AnswerEnvironmentTest(unittest.TestCase):
    def test_the_environment_samples_under_its_route_greedily_and_stops_at_the_line(self):
        engine, routes = pools()
        tasks = {"q": Task("q", "Question: what color?\nAnswer:", {"route": "memory:05"})}
        plan = WavePlan((GroupPlan("qa", (Sample("q", "answer", EVAL),)),))
        wave = go(sample_wave(plan, index=1, tasks=tasks, sampling=SamplingSpec(temperature=1.0),
                              routes=routes, master=3))
        turn = wave.trajectories[0].turns[0]
        self.assertEqual(turn.turn_extras["route"], "memory:05")
        self.assertEqual([d.name for ds in engine.directives_seen for d in ds], ["memory:05"])
        self.assertLessEqual(len(turn.token_ids), 16)                  # the environment's own budget


@unittest.skipUnless(torch is not None, "torch required")
class DreamStreamLossTest(unittest.TestCase):
    def test_the_loss_dispatches_by_role_and_adds_the_penalty_once(self):
        from rlstack.data.flatten import TokenBatch
        from rlstack.training.losses.base import PolicyOutputs
        from rlstack.training.losses.dream_stream import dream_stream
        # three documents of two tokens: a memory row, a dreamer row, an eval row
        lp = torch.tensor([-1.0, -2.0, -0.5, -0.5, -3.0, -3.0], requires_grad=True)
        batch = TokenBatch(
            token_ids=(1, 2, 3, 4, 5, 6), loss_mask=(1, 1, 1, 1, 1, 1),
            behavior_logprobs=(-1.0, -2.0, -0.5, -0.5, -3.0, -3.0), segment_ids=(0, 0, 0, 0, 0, 0),
            doc_starts=(0, 2, 4), postdata={"advantage": (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)},
            doc_turn_extras=(({"role": "memory:00"},), ({"role": DREAMER},), ({"role": EVAL},)),
            microbatches_in_update=2)
        penalty = torch.tensor(0.8)
        result = dream_stream(PolicyOutputs(logprobs=lp, provided={"anchor_penalty": penalty}), batch)
        # clone term over the memory row: mean NLL 1.5;
        # eval masked; penalty 0.8 / 2 microbatches
        # REINFORCE on the dream row: -mean(advantage * lp) = -mean(1 * -0.5) = +0.5
        self.assertAlmostEqual(float(result.loss), 1.5 + 0.5 + 0.4, places=5)
        components = result.components
        self.assertAlmostEqual(components["clone_nll_sum"] / components["clone_tokens"], 1.5)
        self.assertAlmostEqual(components["dream_nll_sum"] / components["dream_tokens"], .5)
        self.assertAlmostEqual(components["clone_term"] + components["reinforce_term"]
                               + components["anchor_term"], components["objective"], places=5)
        from rlstack.runner.interfaces import TrainStats
        from rlstack.runner.remote import encode_train_stats, decode_train_stats
        stats = TrainStats(float(result.loss.detach()), 1, 0, 0, 6, components=components)
        self.assertEqual(decode_train_stats(encode_train_stats(stats)), stats)
        result.loss.backward()
        self.assertEqual(float(lp.grad[4]), 0.0)                      # the eval row carries no gradient
        self.assertNotEqual(float(lp.grad[0]), 0.0)
