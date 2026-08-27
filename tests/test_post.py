"""The postprocessing pipeline (rlstack.training.post, rlstack.runner.post)."""

from __future__ import annotations

import asyncio
import unittest
from typing import Any

from common import sealed
from rlstack import (
    POST, Bundle, FakeEngine, Group, Message, PostProcessor, Role, SamplingSpec, Wave,
    postprocessor, run_pipeline, zscore,
)

BUNDLE = Bundle(bundle_id="bundle:post0000", policy_version={"pi": 0})


def pools():
    engine = FakeEngine()
    engine.add_bundle(BUNDLE)
    return {"main": (engine, BUNDLE)}


def go(coro):
    return asyncio.run(coro)


@postprocessor("post_bad_columns")
class _BadColumns(PostProcessor):
    produces = ("promised",)

    async def process(self, group: Any, data: Any, llm: Any):
        return {"something_else": [0.0] * len(group)}


@postprocessor("post_bad_length")
class _BadLength(PostProcessor):
    produces = ("short",)

    async def process(self, group: Any, data: Any, llm: Any):
        return {"short": [0.0]}  # wrong length for any group > 1


@postprocessor("post_sampling_judge")
class _SamplingJudge(PostProcessor):
    """A judge: a postprocessor that SAMPLES (through any pool it likes).
    It samples the default main-pinned client, so it declares "main"."""

    produces = ("judge",)
    pools = ("main",)

    async def process(self, group: Any, data: Any, llm: Any):
        scores = []
        for traj in group.trajectories:
            turn = await llm.sample(traj.messages)   # llm.pool(name) also works
            scores.append(float(len(turn.message.content)))
        return {"judge": scores}


class ZscoreTest(unittest.TestCase):
    def test_population_std(self) -> None:
        self.assertEqual(zscore([0.0, 2.0]), [-1.0, 1.0])

    def test_all_equal_is_zeros(self) -> None:
        self.assertEqual(zscore([3.0, 3.0, 3.0]), [0.0, 0.0, 0.0])

    def test_singleton_is_zero(self) -> None:
        self.assertEqual(zscore([5.0]), [0.0])

    def test_known_three_point_case(self) -> None:
        out = zscore([1.0, 2.0, 3.0])
        expected = 1.0 / (2.0 / 3.0) ** 0.5  # population std = sqrt(2/3)
        self.assertAlmostEqual(out[0], -expected, places=12)
        self.assertAlmostEqual(out[2], expected, places=12)


class PipelineTest(unittest.TestCase):
    def wave(self) -> Wave:
        # group a: one right ("4"), one wrong answer; group b: both wrong
        return Wave([
            Group("a", [sealed("a", content="4"), sealed("a", content="7")]),
            Group("b", [sealed("b", content="9"), sealed("b", content="8")]),
        ])

    def run_pipe(self, pipeline: tuple[str, ...]):
        return go(run_pipeline(pipeline, self.wave(), pools(), SamplingSpec(),
                               master=17, update=1))

    def test_verifier_then_advantage_chains_per_group(self) -> None:
        columns = self.run_pipe(("verifier", "grpo_advantage"))
        self.assertEqual(columns["reward"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(columns["advantage"], [1.0, -1.0, 0.0, 0.0])

    def test_columns_align_to_wave_order(self) -> None:
        columns = self.run_pipe(("verifier",))
        self.assertEqual(len(columns["reward"]), 4)

    def test_deterministic(self) -> None:
        self.assertEqual(self.run_pipe(("verifier", "grpo_advantage")),
                         self.run_pipe(("verifier", "grpo_advantage")))

    def test_a_judge_samples_deterministically(self) -> None:
        first = self.run_pipe(("post_sampling_judge",))
        second = self.run_pipe(("post_sampling_judge",))
        self.assertEqual(first, second)
        self.assertEqual(len(first["judge"]), 4)

    def test_output_columns_must_match_produces(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.run_pipe(("post_bad_columns",))
        self.assertIn("produces", str(caught.exception))

    def test_output_length_must_match_group(self) -> None:
        with self.assertRaises(ValueError):
            self.run_pipe(("post_bad_length",))


class RegistrationTest(unittest.TestCase):
    def test_builtins_declare_their_wiring(self) -> None:
        verifier = POST.get("verifier")
        self.assertEqual(verifier.produces, ("reward",))
        self.assertEqual(verifier.consumes, ())
        grpo = POST.get("grpo_advantage")
        self.assertEqual(grpo.consumes, ("reward",))
        self.assertEqual(grpo.produces, ("advantage",))


if __name__ == "__main__":
    unittest.main()


@postprocessor("post_tiny_budget")
class _TinyBudget(PostProcessor):
    """Declares its OWN sampling: max_tokens=1 truncates the fake's 2-char
    answer to prove the per-processor override beats the run's sampling."""

    produces = ("length",)
    sampling = SamplingSpec(max_tokens=1)

    async def process(self, group: Any, data: Any, llm: Any):
        out = []
        for _ in group.trajectories:
            turn = await llm.sample((Message(Role.USER, "What is 30+40?"),))
            out.append(float(len(turn.message.content)))
        return {"length": out}


class PoolDeclarationTest(unittest.TestCase):
    def test_processor_sampling_overrides_the_runs(self) -> None:
        wave = Wave([Group("t", [sealed("t")])])
        columns = go(run_pipeline(("post_tiny_budget",), wave, pools(),
                                  SamplingSpec(max_tokens=64), 7, 1))
        self.assertEqual(columns["length"], [1.0])

    def test_llm_judge_rewards_agreement_with_the_judge_pool(self) -> None:
        main_engine, judge_engine = FakeEngine(), FakeEngine(p_correct=1.0)
        main_engine.add_bundle(BUNDLE)
        judge_engine.add_bundle(BUNDLE)
        both = {"main": (main_engine, BUNDLE), "judge": (judge_engine, BUNDLE)}
        wave = Wave([Group("t", [sealed("t", content="4", answer=4),
                                 sealed("t", content="5", answer=4)])])
        columns = go(run_pipeline(("llm_judge",), wave, both,
                                  SamplingSpec(), 7, 1))
        self.assertEqual(columns["reward"], [1.0, 0.0])

    def test_llm_judge_declares_its_pool(self) -> None:
        pdef = POST.get("llm_judge")
        self.assertEqual(pdef.pools, ("judge",))
        self.assertEqual(pdef.instance.sampling.temperature, 0.0)


@postprocessor("post_token_teacher")
class _TokenTeacher(PostProcessor):
    """token_level: one float per GENERATED token — the per-token teacher
    channel (#38: post produces everything the loss operates on)."""

    produces = ("teacher_lp",)
    token_level = ("teacher_lp",)

    async def process(self, group: Any, data: Any, llm: Any):
        return {"teacher_lp": [
            [0.5] * sum(len(t.token_ids) for t in traj.turns)
            for traj in group.trajectories]}


@postprocessor("post_token_liar")
class _TokenLiar(PostProcessor):
    produces = ("bad_lp",)
    token_level = ("bad_lp",)

    async def process(self, group: Any, data: Any, llm: Any):
        return {"bad_lp": [[0.5, 0.5] for _ in group.trajectories]}


class TokenLevelColumnTest(unittest.TestCase):
    def wave_of_one(self):
        traj = sealed("t0")
        return Wave([Group("g0", [traj])]), traj

    def test_token_vector_flows_to_token_aligned_postdata(self) -> None:
        from rlstack.data.flatten import broadcast, flatten
        wave, traj = self.wave_of_one()
        columns = go(run_pipeline(("post_token_teacher",), wave, pools(),
                                  SamplingSpec(), master=7, update=1))
        generated = sum(len(t.token_ids) for t in traj.turns)
        self.assertEqual(len(columns["teacher_lp"][0]), generated)

        flat = flatten(traj, tokenize=lambda s: tuple(ord(c) for c in s))
        (per_doc,) = broadcast(columns, [flat])
        aligned = per_doc["teacher_lp"]
        self.assertEqual(len(aligned), len(flat.token_ids))
        self.assertEqual(
            [v for v, m in zip(aligned, flat.loss_mask) if m],
            [0.5] * generated)
        self.assertTrue(all(v == 0.0 for v, m
                            in zip(aligned, flat.loss_mask) if not m))

    def test_wrong_token_count_dies_loudly(self) -> None:
        wave, _ = self.wave_of_one()
        with self.assertRaises(ValueError) as caught:
            go(run_pipeline(("post_token_liar",), wave, pools(),
                            SamplingSpec(), master=7, update=1))
        self.assertIn("generated tokens", str(caught.exception))


class ScoringAndHintedTest(unittest.TestCase):
    """The scoring verb and the real OPSD channel: score() is deterministic,
    consumes no sampling seed, and hinted_logprobs lands a token-aligned
    teacher column produced entirely by the post pipeline (I9)."""

    def test_score_is_deterministic_and_seed_neutral(self) -> None:
        from rlstack import Message, Role
        from rlstack.runner.sampling import EnginePoolClient

        async def scenario():
            routes = pools()
            msgs = (Message(Role.USER, "What is 2+2?"),)
            a = EnginePoolClient(routes, SamplingSpec(), episode_seed=7)
            b = EnginePoolClient(routes, SamplingSpec(), episode_seed=7)
            await a.sample(msgs)
            await b.sample(msgs)
            scores = await a.score(msgs, (52, 53))        # only a scores
            again = await a.score(msgs, (52, 53))
            second_a = await a.sample(msgs)
            second_b = await b.sample(msgs)
            return scores, again, second_a, second_b

        scores, again, second_a, second_b = go(scenario())
        self.assertEqual(scores, again)                   # deterministic
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(s < 0 for s in scores))
        # scoring drew nothing: a's second sample matches b's exactly
        self.assertEqual(second_a.token_ids, second_b.token_ids)
        self.assertEqual(second_a.seed, second_b.seed)

    def test_hinted_pipeline_feeds_opsd_end_to_end(self) -> None:
        import tempfile

        from common import arith_spec, arith_store
        from dataclasses import replace
        from rlstack import (
            FakeEngine, FakeLearner, fake_qwen_schema, run_experiment,
        )

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, heldout = arith_store(tmp.name)
        base = arith_spec(train, heldout)
        spec = replace(base, algo=replace(
            base.algo, loss="opsd", post=("verifier", "hinted_logprobs")))

        report = run_experiment(spec, fake_qwen_schema(4, base="Qwen/Qwen3-0.6B"),
                                store, FakeEngine(), FakeLearner())
        run = store.open_run(report.run_id)
        self.assertEqual(len(run.read_ledger()), 4)

        postdata = run.read_postdata(1)
        rows = run.read_wave(1)
        for vector, row in zip(postdata["hinted_logprobs"], rows):
            generated = sum(len(t["token_ids"]) for t in row["turns"])
            self.assertEqual(len(vector), generated)      # token-aligned
            self.assertTrue(all(v < 0 for v in vector))

        dictionary = store.peek_dictionary(report.run_id)
        by_name = {(c["name"], c["phase"]): c for c in dictionary["columns"]}
        hinted = by_name[("hinted_logprobs", "post")]
        self.assertEqual(hinted["granularity"], "token")
        self.assertTrue(hinted["feeds_loss"])
        self.assertIn("loss:opsd", hinted["consumers"])
