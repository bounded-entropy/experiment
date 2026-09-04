"""Measurement (#70): observation outside the run.

The claims: a measuring pass follows the ledger on the manifest's cadence
and backfills every missing point; a second pass measures nothing
(idempotence by measured_updates); the points are byte-identical under a
scrambling engine (reduction in WAVE order, #53's rule carried over from the
in-run evaluator this replaced); per-task means unbundle the curve; the run
dir gains not one byte; and a PRE-#70 manifest — whose canonical row still
carries eval keys — decodes into today's spec with the extras ignored, so
old stores stay readable history.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from common import arith_spec, arith_store
from test_resume import InterleavingEngine

from rlstack import (
    FakeEngine, FakeLearner, Measurement, fake_qwen_schema, load_tasks,
    measure_run, run_experiment,
)
from rlstack.runner.remote import spec_from_json
from rlstack.spec.canonical import canonical_json

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def go(coro):
    return asyncio.run(coro)


class MeasureFixture(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)
        self.engine = FakeEngine()
        self.report = run_experiment(arith_spec(self.train), SCHEMA,
                                     self.store, self.engine, FakeLearner())
        self.tasks = {t.id: t for t in load_tasks(self.store, self.heldout)}
        self.measurement = Measurement(
            name="heldout", env="math_single_turn",
            task_ids=tuple(sorted(self.tasks)), samples=2, every=2,
            post=("verifier",), seed=17)

    def measure(self, pool=None, max_inflight: int = 64,
                pools: dict | None = None) -> list[int]:
        return go(measure_run(self.store, self.report.run_id,
                              self.measurement, pool or self.engine,
                              self.tasks, max_inflight=max_inflight,
                              pools=pools or {}))


class MeasureRunTest(MeasureFixture):
    def test_follows_the_cadence_and_backfills(self) -> None:
        fresh = self.measure()
        self.assertEqual(fresh, [2, 4])           # every 2nd of 4 committed
        told = self.store.read_measurements(self.report.run_id)["heldout"]
        self.assertEqual([p["update"] for p in told["points"]], [2, 4])
        point = told["points"][0]
        self.assertEqual(point["episodes"], len(self.tasks) * 2)
        self.assertIn("reward", point["means"])
        self.assertEqual(sorted(point["task_means"]["reward"]),
                         sorted(self.tasks))
        self.assertEqual(told["manifest"]["every"], 2)

    def test_a_second_pass_measures_nothing(self) -> None:
        self.measure()
        self.assertEqual(self.measure(), [])

    def test_points_do_not_depend_on_the_completion_order(self) -> None:
        """The reduction folds in WAVE order (#53): a scrambling engine and a
        serial one write byte-identical points."""
        self.measure(max_inflight=1)
        reference = self.store.read_measurements(
            self.report.run_id)["heldout"]["points"]

        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        store, train, heldout = arith_store(other.name)
        scrambler = InterleavingEngine()
        report = run_experiment(arith_spec(train), SCHEMA, store, scrambler,
                                FakeLearner())
        tasks = {t.id: t for t in load_tasks(store, heldout)}
        go(measure_run(store, report.run_id, self.measurement, scrambler,
                       tasks, max_inflight=64))
        scrambled = store.read_measurements(report.run_id)["heldout"]["points"]
        self.assertEqual(scrambled, reference)
        self.assertNotEqual(scrambler.finished, scrambler.launched)

    def test_the_run_dir_gains_not_one_byte(self) -> None:
        """Measurement is OUTSIDE the run: the run dir before and after a
        measuring pass is the same bytes — nothing to resume, nothing to
        forge."""
        prefix = self.store.run_prefix(self.report.run_id)
        before = {key: self.store._read(key)
                  for key in self.store._list(prefix)}
        self.measure()
        after = {key: self.store._read(key)
                 for key in self.store._list(prefix)}
        self.assertEqual(after, before)


class SecondPoolTest(MeasureFixture):
    """ADR 0005: a measurement may address more than the policy. Every extra
    name routes to its engine under a PAYLOAD-FREE base bundle — the loop's
    rule for a non-policy pool — so "how far is the student from the
    teacher" is a measurement like any other."""

    TEACHER_BASE = "Qwen/Qwen3-32B"

    def scoring_measurement(self):
        from dataclasses import replace
        return replace(self.measurement, name="distill",
                       post=("conditioned_teacher_logprobs", "reverse_kl"))

    def measure_through(self, pools: dict) -> list[int]:
        return go(measure_run(self.store, self.report.run_id,
                              self.scoring_measurement(), self.engine,
                              self.tasks, pools=pools))

    def test_a_pipeline_that_scores_through_teacher_runs(self) -> None:
        teacher = FakeEngine(base=self.TEACHER_BASE)
        self.assertEqual(self.measure_through({"teacher": teacher}), [2, 4])
        told = self.store.read_measurements(self.report.run_id)["distill"]
        point = told["points"][0]
        self.assertIn("reverse_kl", point["means"])
        self.assertIn("teacher_logprobs", point["means"])
        # the teacher served its BARE BASE: payload-free, so no delta of the
        # measured run ever reached it
        self.assertEqual(teacher.bundle_log, ["bundle:base:teacher"])

    def test_without_the_pool_it_fails_by_name(self) -> None:
        with self.assertRaises(KeyError) as missing:
            self.measure_through({})
        self.assertIn("teacher", str(missing.exception))

    def test_one_engine_may_back_both_names(self) -> None:
        """The Routes contract already says so: naming "teacher" on the same
        engine object as "main" is one dict entry and no second resident."""
        self.assertEqual(self.measure_through({"teacher": self.engine}), [2, 4])
        self.assertIn("bundle:base:teacher", self.engine.bundle_log)


class EraBoundaryTest(unittest.TestCase):
    def test_a_pre_70_manifest_still_decodes(self) -> None:
        """Old stores are readable history: a canonical row carrying the
        RETIRED eval keys (an EvalSpec-tagged subtree, a plans.eval uri)
        decodes into today's spec with the extras ignored — never an
        unknown-tag refusal, because an untouched key is never decoded."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, train, _ = arith_store(tmp.name)
        row = json.loads(canonical_json(arith_spec(train)))
        row["eval"] = {"__type__": "EvalSpec", "every": 5,
                       "post": ["verifier"], "pool": "main"}
        row["plans"]["eval"] = "cas://plan/eval"
        spec = spec_from_json(row)
        self.assertFalse(hasattr(spec, "eval"))
        self.assertEqual(spec.plans.train, json.loads(
            canonical_json(arith_spec(train)))["plans"]["train"])


if __name__ == "__main__":
    unittest.main()
