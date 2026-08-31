"""The DSL campaign's pieces: two invented tool languages, their graders, and
the new loss (reverse_ppo) wired end to end on fakes.

The claims under test: the graders pay the milestone ladders exactly (dense by
design — the DAPO all-or-nothing lesson); the task builders are deterministic
content with the stated generalization holes (mol never trains, ring
directions eval wider than they train); validate refuses the new loss when
its provider is missing and passes the campaign's actual arm
shapes; and each arm shape RUNS on fake metal — seal, post, train, ledger —
with the reverse head's provided name reaching the ledger's train block.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from common import cas_uri
from rlstack import (
    AlgoSpec, ExperimentSpec, FakeEngine, FakeLearner, GenSpec, GpuConfig,
    GpuGroup, GroupPlan, Host, LocalStore, OptimSpec, Plans, PolicySpec,
    RunPlan, Sample, Schedule, Seeds, Task, WavePlan, WaveRef, encode,
    fake_qwen_schema, gpus, learner, lora, pool, validate, write_tasks,
)
from rlstack.data.tasks.glyph_exchange import glyph_eval_tasks, glyph_train_tasks
from rlstack.data.tasks.stamp_office import stamp_eval_tasks, stamp_train_tasks
from rlstack.policy.adapters.reverse_value_head import reverse_value_head
from rlstack.registry import ADAPTER_TYPES, ENVS, LOSSES, POST
from rlstack.training.post.glyph_grade import GlyphGrade
from rlstack.training.post.stamp_grade import StampGrade


BASE = "Qwen/Qwen3-0.6B"
SCHEMA = fake_qwen_schema(4, base=BASE)

GOLD_STAMP = "grab(47)\nfold(47)\nink(47, vex)\nseal(47)\nfile(47, D3)"


def go(coro):
    return asyncio.run(coro)


class StampGradeTest(unittest.TestCase):
    def grade(self, completion: str, doc: str = "47", color: str = "vex"):
        return StampGrade().grade(completion, doc, color)

    def test_the_gold_path_pays_in_full(self) -> None:
        self.assertEqual(self.grade(GOLD_STAMP), (1.0, 1.0))

    def test_the_ladder_pays_progress(self) -> None:
        """Each legal prefix of the gold path earns its rung — the dense
        reward a 0.6B needs (the DAPO all-or-nothing lesson)."""
        self.assertEqual(self.grade("grab(47)"), (0.15, 0.0))
        self.assertEqual(self.grade("grab(47)\nfold(47)"), (0.30, 0.0))
        self.assertEqual(self.grade("grab(47)\nfold(47)\nink(47, vex)"),
                         (0.45, 0.0))
        self.assertEqual(
            self.grade("grab(47)\nfold(47)\nink(47, vex)\nseal(47)"),
            (0.60, 0.0))

    def test_a_rung_counts_only_on_the_rungs_below(self) -> None:
        """Sealed over the WRONG ink is a document that left the gold path at
        the ink: the ladder stops at folded."""
        reward, exact = self.grade(
            "grab(47)\nfold(47)\nink(47, mol)\nseal(47)\nfile(47, D2)")
        self.assertEqual((reward, exact), (0.30, 0.0))

    def test_illegal_calls_cost_and_floor_at_zero(self) -> None:
        reward, exact = self.grade("seal(47)\n" + GOLD_STAMP)   # seal first: illegal
        self.assertEqual((round(reward, 6), exact), (0.95, 0.0))
        reward, _ = self.grade("what is a stamp office?")
        self.assertEqual(reward, 0.0)

    def test_the_tray_holds_only_the_named_document(self) -> None:
        reward, _ = self.grade("grab(99)\ngrab(47)")
        self.assertEqual(round(reward, 6), 0.10)    # grabbed, one illegal

    def test_folding_twice_ruins_the_document(self) -> None:
        reward, _ = self.grade("grab(47)\nfold(47)\nfold(47)\nink(47, vex)")
        # second fold illegal (skipped), so ink lands on a once-folded doc
        self.assertEqual(round(reward, 6), 0.40)

    def test_the_drawer_is_fixed_by_the_ink(self) -> None:
        reward, exact = self.grade(
            "grab(47)\nfold(47)\nink(47, vex)\nseal(47)\nfile(47, D1)")
        self.assertEqual((round(reward, 6), exact), (0.55, 0.0))


class GlyphGradeTest(unittest.TestCase):
    def grade(self, completion: str, purse: str = "P7", source: str = "wex",
              target: str = "sarn"):
        return GlyphGrade().grade(completion, purse, source, target)

    def test_the_gold_route_pays_in_full(self) -> None:
        self.assertEqual(
            self.grade("give(to_sarn(to_polk(to_drin(load(P7)))))"),
            (1.0, 1.0))

    def test_prose_around_the_expression_is_tolerated(self) -> None:
        self.assertEqual(
            self.grade("Sure! give(to_sarn(to_polk(to_drin(load(P7))))) done"),
            (1.0, 1.0))

    def test_the_route_pays_its_correct_prefix(self) -> None:
        reward, exact = self.grade("give(to_polk(to_drin(load(P7))))")
        self.assertEqual((round(reward, 6), exact), (0.6, 0.0))  # 2 of 3 steps

    def test_a_wrong_step_ends_the_prefix_and_costs(self) -> None:
        reward, exact = self.grade("give(to_wex(to_drin(load(P7))))")
        # one correct step (0.2), then a wrong one: spurious, -0.05
        self.assertEqual((round(reward, 6), exact), (0.35, 0.0))

    def test_the_wrong_purse_scores_nothing(self) -> None:
        self.assertEqual(self.grade("give(to_drin(load(P9)))",
                                    source="wex", target="drin"), (0.0, 0.0))

    def test_the_ring_wraps(self) -> None:
        self.assertEqual(
            self.grade("give(to_drin(to_wex(load(P2))))", purse="P2",
                       source="sarn", target="drin"), (1.0, 1.0))

    def test_garbage_scores_zero(self) -> None:
        self.assertEqual(self.grade("the answer is 42"), (0.0, 0.0))


class TaskBuilderTest(unittest.TestCase):
    def test_the_sets_are_deterministic_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(tmp)
            first = write_tasks(store, stamp_train_tasks())
            second = write_tasks(store, stamp_train_tasks())
            self.assertEqual(first, second)      # same bytes, same identity

    def test_the_training_hole_is_real(self) -> None:
        """mol never appears in a stamp train task; every eval color does —
        memorizing the train answers has nothing to say about mol."""
        train_colors = {t.meta["color"] for t in stamp_train_tasks()}
        eval_colors = {t.meta["color"] for t in stamp_eval_tasks()}
        self.assertNotIn("mol", train_colors)
        self.assertEqual(eval_colors, {"rju", "vex", "mol"})
        self.assertEqual(len(stamp_train_tasks()), 2)
        self.assertEqual(len(stamp_eval_tasks()), 72)

    def test_the_glyph_eval_sweeps_directions_the_train_never_shows(self) -> None:
        train_pairs = {(t.meta["source"], t.meta["target"])
                       for t in glyph_train_tasks()}
        eval_pairs = {(t.meta["source"], t.meta["target"])
                      for t in glyph_eval_tasks()}
        self.assertEqual(len(glyph_train_tasks()), 2)
        self.assertEqual(len(eval_pairs), 12)     # every ordered pair
        self.assertIn(("sarn", "wex"), eval_pairs - train_pairs)

    def test_ids_are_unique_across_both_families(self) -> None:
        everything = (stamp_train_tasks() + stamp_eval_tasks()
                      + glyph_train_tasks() + glyph_eval_tasks())
        self.assertEqual(len({t.id for t in everything}), len(everything))


# ---------------------------------------------------------------------------
# the campaign's arm shapes: registration, validation, and fake metal
# ---------------------------------------------------------------------------

def stamp_plan(task_ids: list[str], updates: int, size: int = 2) -> RunPlan:
    return RunPlan(tuple(
        WavePlan(tuple(
            GroupPlan(task, tuple(Sample(task, "stamp_office")
                                  for _ in range(size)))
            for task in task_ids))
        for _ in range(updates)))


def arm_spec(store, *, loss: str, post: tuple[str, ...],
             bank=None, updates: int = 2) -> ExperimentSpec:
    tasks = stamp_train_tasks()
    uri = write_tasks(store, tasks)
    rollout = encode(stamp_plan([t.id for t in tasks], updates))
    train = encode(RunPlan(tuple(WaveRef(f"self://rollouts/{u}")
                                 for u in range(1, updates + 1))))
    store.cas_put(rollout)
    store.cas_put(train)
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank=bank if bank is not None
                          else {"pi": lora("layers.0-3.self_attn.*", r=8)}),
        gen=GenSpec(envs=("stamp_office",), tasks=(uri,)),
        plans=Plans(train=cas_uri(train), rollout=cas_uri(rollout)),
        algo=AlgoSpec(loss=loss, post=post,
                      optim=OptimSpec("adamw", lr=1e-5),
                      schedule=Schedule(microbatch_tokens=2048)),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"), learner())),)),
        seeds=Seeds(master=23))


REVERSE_BANK = {"pi": lora("layers.0-3.self_attn.*", r=8),
                "vh": reverse_value_head("final_hidden", d_model=64)}


class WiringTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)

    def test_everything_is_registered(self) -> None:
        self.assertIsNotNone(LOSSES.get("reverse_ppo"))
        for name in ("stamp_grade", "glyph_grade"):
            self.assertIsNotNone(POST.get(name))
        for name in ("stamp_office", "glyph_exchange"):
            self.assertIsNotNone(ENVS.get(name))
        self.assertIsNotNone(ADAPTER_TYPES.get("reverse_value_head"))

    def test_the_arm_shapes_validate_clean(self) -> None:
        arms = (
            arm_spec(self.store, loss="grpo",
                     post=("stamp_grade", "grpo_advantage")),
            arm_spec(self.store, loss="reverse_ppo", post=("stamp_grade",),
                     bank=REVERSE_BANK),
        )
        for spec in arms:
            self.assertEqual(validate(spec, SCHEMA), [], spec.algo.loss)

    def test_reverse_ppo_without_the_head_is_refused(self) -> None:
        spec = arm_spec(self.store, loss="reverse_ppo", post=("stamp_grade",))
        codes = {issue.code for issue in validate(spec, SCHEMA)}
        self.assertIn("unsatisfied-requires", codes)

    def test_the_head_refuses_a_weighted_site(self) -> None:
        bank = {"pi": lora("layers.0-3.self_attn.*", r=8),
                "vh": reverse_value_head("layers.0.self_attn.q_proj",
                                         d_model=64)}
        spec = arm_spec(self.store, loss="reverse_ppo", post=("stamp_grade",),
                        bank=bank)
        codes = {issue.code for issue in validate(spec, SCHEMA)}
        self.assertTrue(codes)      # site_ok says no somewhere in the report


class ArmRunTest(unittest.TestCase):
    """Each arm shape runs whole on fake metal: seal, post, train, ledger."""

    def run_arm(self, loss: str, post: tuple[str, ...], bank=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = LocalStore(tmp.name)
        spec = arm_spec(store, loss=loss, post=post, bank=bank)
        host = Host("dsl-fake", engines=(FakeEngine(),), learner=FakeLearner(),
                    store=store)
        report = go(host.submit(spec, SCHEMA, store=store))
        entries = store.peek_ledger(report.run_id)
        self.assertEqual(len(entries), 2, entries)
        return entries

    def test_grpo_on_the_stamp_office(self) -> None:
        self.run_arm("grpo", ("stamp_grade", "grpo_advantage"))

    def test_reverse_ppo_on_the_stamp_office(self) -> None:
        """The provided name reaches the ledger's train block — the provides
        emission channel, exercised for the new head with no GPU."""
        entries = self.run_arm("reverse_ppo", ("stamp_grade",),
                               bank=REVERSE_BANK)
        self.assertIn("reverse_values", json.dumps(entries[-1]))


if __name__ == "__main__":
    unittest.main()
