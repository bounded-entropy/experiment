"""The Scorer runner (#65): the split, the parts, the pin, and the equivalence.

Four claims, one class each.

THE SPLIT RULE is a declaration read twice: a processor that declares `pools`
sends traffic, so it is the Scorer's; a pool-less one is arithmetic and stays
with the Trainer. The submit gate refuses the one ordering the split cannot
honour — a pooled processor consuming an inline processor's column.

POSTDATA PARTS are per-update artifacts like any other: atomic, refused once
committed, swept by attach when their update never was.

THE VERSION-PINNING RULE is what running beside the Trainer costs: the Scorer
holds no current bundle, so policy-pool traffic is scored under the version the
wave's own turns recorded.

THE EQUIVALENCE OBLIGATION: split scoring must produce the same columns as
running the whole pipeline over the recorded wave and exact version. The
reference pipeline runs outside training; the Trainer has no inference door.
"""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from dataclasses import replace
from typing import Any

from common import arith_spec, arith_store, sealed
from test_resume import CrashingStore, SimulatedCrash
from rlstack.runner.checkpointing import EVERY_UPDATE
from rlstack import (
    Bundle, FakeEngine, FakeLearner, Arbiter, Topology, HostSpec, Group,
    LocalStore, Message, PostProcessor, Role, Rollout, RunSignals, Scorer, Task,
    Trajectory, Turn, Wave, fake_qwen_schema, learner, pool, postprocessor,
    run_experiment, validate,
)
from rlstack.data.plan import RunPlan
from rlstack.data.stores.base import StoreError, postdata_part_key
from rlstack.runner.roles import SCORER
from rlstack.runner.refs import RefReader
from rlstack.spec.flow import PipelineSplit, split_pipeline

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
TEACHER_BASE = "Qwen/Qwen3-32B"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@postprocessor("scorer_test_pooled_consumer")
class _PooledConsumer(PostProcessor):
    """A pooled processor consuming a column an inline one owns — the exact
    inversion the submit gate refuses."""

    produces = ("verdict",)
    consumes = ("reward",)
    pools = ("judge",)

    async def process(self, group: Any, data: Any, client: Any):
        return {"verdict": [0.0] * len(group)}


def judged_spec(train_uri: str, heldout_uri: str | None = None):
    """GRPO whose reward comes from a JUDGE pool: pooled llm_judge feeding
    inline grpo_advantage — the split with traffic on both sides of it."""
    base = arith_spec(train_uri, heldout_uri)
    return replace(
        base,
        algo=replace(base.algo, post=("llm_judge", "grpo_advantage")),
        topology=Topology(hosts=(
            HostSpec((pool("main"),)), HostSpec((pool("judge"),)),
            HostSpec((learner(),)))))


def selfscored_spec(train_uri: str):
    """OPSD: `hinted_logprobs` scores the POLICY pool, which is the only
    pipeline shape the version-pinning rule can bite on."""
    base = arith_spec(train_uri)
    return replace(base, algo=replace(
        base.algo, loss="opsd", post=("verifier", "hinted_logprobs")))


def judge_metal():
    """A student and a judge on their own pools. The judge is a COIN
    (p_correct=0.5), deliberately: a judge that always agrees would make the
    reward column independent of the seed, and the equivalence test would stop
    proving that the two worlds draw from the same seed path."""
    return {"main": FakeEngine(), "judge": FakeEngine(p_correct=0.5)}



def snapshot(store: LocalStore, run_id: str) -> dict[str, str]:
    """{relative path: sha256} over the whole run directory."""
    root = store.path_of(f"runs/{run_id}")
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def without_parts(files: dict[str, str]) -> dict[str, str]:
    """The same snapshot minus the Scorer's own parts — what the inline world
    has no reason to contain."""
    return {path: digest for path, digest in files.items()
            if not path.endswith(f".{SCORER}.json")}


def sealed_at(task_id: str, bundle_id: str, versions: dict[str, int]) -> Trajectory:
    """A sealed trajectory pinning a NAMED bundle — the recorded fact the
    version-pinning rule reads (I6)."""
    turn = Turn(message=Message(Role.ASSISTANT, "4"), token_ids=(52,),
                behavior_logprobs=(-0.5,), finish="eos", stop_hit=None,
                bundle_id=bundle_id, policy_version=dict(versions), seed=17,
                token_extras={}, turn_extras={})
    return Rollout(task=Task(task_id, "What is 2+2?", {"answer": 4}),
                   messages=[Message(Role.USER, "What is 2+2?"), turn.message],
                   turns=[turn]).seal()


# ---------------------------------------------------------------------------
# the split rule
# ---------------------------------------------------------------------------

class SplitRuleTest(unittest.TestCase):
    def test_pools_decide_the_side(self) -> None:
        split = split_pipeline(("verifier", "teacher_logprobs"))
        self.assertEqual(split.pooled, ("teacher_logprobs",))
        self.assertEqual(split.inline, ("verifier",))

    def test_a_pipeline_with_no_traffic_is_wholly_inline(self) -> None:
        split = split_pipeline(("verifier", "grpo_advantage"))
        self.assertEqual(split.pooled, ())
        self.assertEqual(split.inline, ("verifier", "grpo_advantage"))

    def test_each_half_keeps_the_declared_order(self) -> None:
        split = split_pipeline(
            ("llm_judge", "verifier", "teacher_logprobs", "grpo_advantage"))
        self.assertEqual(split.pooled, ("llm_judge", "teacher_logprobs"))
        self.assertEqual(split.inline, ("verifier", "grpo_advantage"))

    def test_the_two_halves_partition_the_pipeline(self) -> None:
        pipeline = ("llm_judge", "verifier", "grpo_advantage")
        split = split_pipeline(pipeline)
        self.assertEqual(sorted(split.pooled + split.inline), sorted(pipeline))

    def test_an_unregistered_name_is_inline_and_never_raises(self) -> None:
        """check_names_are_registered owns that failure; this walk reports it
        by not pretending to know where the name belongs."""
        self.assertEqual(split_pipeline(("no_such_processor",)),
                         PipelineSplit((), ("no_such_processor",)))


class SplitOrderGateTest(unittest.TestCase):
    """The submit gate's half of the rule: the split must respect
    produces→consumes."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, self.train, _ = arith_store(tmp.name)

    def codes(self, pipeline: tuple[str, ...]) -> list[str]:
        base = arith_spec(self.train)
        spec = replace(
            base, algo=replace(base.algo, post=pipeline),
            topology=Topology(hosts=(
                HostSpec((pool("main"),)), HostSpec((pool("judge"),)),
                HostSpec((learner(),)))))
        return [issue.code for issue in validate(spec, SCHEMA)]

    def test_pooled_consuming_inline_is_refused(self) -> None:
        codes = self.codes(("verifier", "scorer_test_pooled_consumer",
                            "grpo_advantage"))
        self.assertIn("post-split-order", codes)

    def test_inline_consuming_pooled_is_fine(self) -> None:
        """The normal case: the trainer awaits the part, then runs its half
        over it."""
        self.assertEqual(self.codes(("llm_judge", "grpo_advantage")), [])

    def test_a_wholly_inline_pipeline_is_never_examined(self) -> None:
        self.assertEqual(self.codes(("verifier", "grpo_advantage")), [])

    def test_the_issue_names_both_sides(self) -> None:
        base = arith_spec(self.train)
        spec = replace(
            base, algo=replace(base.algo, post=(
                "verifier", "scorer_test_pooled_consumer", "grpo_advantage")),
            topology=Topology(hosts=(
                HostSpec((pool("main"),)), HostSpec((pool("judge"),)),
                HostSpec((learner(),)))))
        issue = next(i for i in validate(spec, SCHEMA)
                     if i.code == "post-split-order")
        self.assertEqual(issue.path, "algo.post[1]")
        self.assertIn("verifier", issue.message)
        self.assertIn("reward", issue.message)


# ---------------------------------------------------------------------------
# the parts
# ---------------------------------------------------------------------------

class PostdataPartTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.store = LocalStore(self.root)
        self.run = self.store.open_run("rid", manifest={"run_id": "rid"})

    def test_absent_reads_as_none(self) -> None:
        """The await predicate: absence is a value, not an exception."""
        self.assertIsNone(self.run.read_postdata_part(1, SCORER))

    def test_a_part_round_trips(self) -> None:
        self.run.write_postdata_part(1, SCORER, {"teacher": [1.0, 2.0]})
        self.assertEqual(self.run.read_postdata_part(1, SCORER),
                         {"teacher": [1.0, 2.0]})

    def test_a_part_lives_beside_the_merged_file(self) -> None:
        self.run.write_postdata_part(3, SCORER, {"x": [1.0]})
        self.assertEqual(postdata_part_key("runs/rid", 3, SCORER),
                         "runs/rid/postdata/000003.scorer.json")
        self.assertTrue(self.store.path_of(
            postdata_part_key("runs/rid", 3, SCORER)).exists())

    def test_the_merged_file_is_still_the_one_reader_sees(self) -> None:
        self.run.write_postdata_part(1, SCORER, {"teacher": [1.0]})
        with self.assertRaises(FileNotFoundError):
            self.run.read_postdata(1)          # a part is not the postdata
        self.run.write_postdata(1, {"teacher": [1.0], "reward": [0.0]})
        self.assertEqual(sorted(self.run.read_postdata(1)),
                         ["reward", "teacher"])

    def test_a_committed_update_refuses_a_part(self) -> None:
        self.run.append_ledger({"update": 1, "versions": {"pi": 1}})
        with self.assertRaises(StoreError):
            self.run.write_postdata_part(1, SCORER, {"x": [0.0]})

    def test_a_producer_is_one_dot_free_segment(self) -> None:
        """The producer names a file that `_parse_update` reads the update out
        of; a dot or a slash would shear it."""
        for bad in ("scorer.v2", "a/b"):
            with self.assertRaises(AssertionError):
                self.run.write_postdata_part(1, bad, {"x": [0.0]})

    def test_attach_sweeps_parts_the_ledger_never_committed(self) -> None:
        self.run.append_ledger({"update": 1, "versions": {"pi": 1}})
        self.run.append_checkpoint(1, {"pi": 1})
        self.run.write_postdata_part(2, SCORER, {"x": [0.0]})
        self.run.write_postdata_part(3, SCORER, {"x": [0.0]})
        reattached = LocalStore(self.root).open_run("rid")
        self.assertIsNone(reattached.read_postdata_part(2, SCORER))
        self.assertIsNone(reattached.read_postdata_part(3, SCORER))

    def test_attach_keeps_a_committed_updates_part(self) -> None:
        self.run.write_postdata_part(1, SCORER, {"x": [0.5]})
        self.run.append_ledger({"update": 1, "versions": {"pi": 1}})
        self.run.append_checkpoint(1, {"pi": 1})
        reattached = LocalStore(self.root).open_run("rid")
        self.assertEqual(reattached.read_postdata_part(1, SCORER), {"x": [0.5]})


# ---------------------------------------------------------------------------
# the version-pinning rule
# ---------------------------------------------------------------------------

class VersionPinningTest(unittest.TestCase):
    """A scorer holds no current bundle, so it asks the wave."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, _ = arith_store(tmp.name)
        self.run = self.store.open_run("rid", manifest={"run_id": "rid"})
        self.initial = Bundle(bundle_id="bundle:initial", policy_version={"pi": 0})

    def scorer_for(self, spec) -> Scorer:
        return Scorer(RunSignals(), Arbiter(), self.run,
                      spec=spec, plan=RunPlan(()),
                      refs=RefReader(self.store, self.run), residents=(),
                      routes_at=lambda bundle: {}, initial_bundle=self.initial)

    def wave_at(self, bundle_id: str, versions: dict[str, int]) -> Wave:
        return Wave([Group("g", [sealed_at("t0", bundle_id, versions)])])

    def test_the_policy_pool_is_scored_at_the_wave_s_own_version(self) -> None:
        """Not the newest committed one: the ledger has moved two versions on,
        and the column still belongs to the policy that sampled it."""
        self.run.append_ledger({"update": 1, "versions": {"pi": 3},
                                "bundle_id": "bundle:newest"})
        scorer = self.scorer_for(selfscored_spec(self.train))
        pinned = scorer.pinned_bundle(4, self.wave_at("bundle:old", {"pi": 1}))
        self.assertEqual(pinned.bundle_id, "bundle:old")
        self.assertEqual(pinned.policy_version, {"pi": 1})

    def test_a_wave_pinning_two_bundles_is_scored_as_of_the_previous_update(self) -> None:
        """ADR 0018: turns that disagree (dreams replayed beside an
        arrival's supervised text) pin the ledger's bundle at u-1 — the
        policy just before this update — and the runner waits for it."""
        wave = Wave([Group("g", [sealed_at("t0", "bundle:a", {"pi": 1}),
                                 sealed_at("t0", "bundle:b", {"pi": 2})])])
        scorer = self.scorer_for(selfscored_spec(self.train))
        self.assertTrue(scorer.needs_previous_commit(wave))
        with self.assertRaises(RuntimeError):          # update 3 not committed: wait, never guess
            scorer.pinned_bundle(4, wave)
        for update in (1, 2, 3):
            self.run.append_ledger({"update": update, "versions": {"pi": update},
                                    "bundle_id": f"bundle:v{update}"})
        pinned = scorer.pinned_bundle(4, wave)
        self.assertEqual(pinned.bundle_id, "bundle:v3")
        self.assertEqual(pinned.policy_version, {"pi": 3})

    def test_a_wave_of_supervised_rows_alone_is_scored_as_of_the_previous_update(self) -> None:
        wave = Wave([Group("g", [sealed_at("t0", "supervised:targets", {})])])
        scorer = self.scorer_for(selfscored_spec(self.train))
        self.assertTrue(scorer.needs_previous_commit(wave))
        self.assertIs(scorer.pinned_bundle(1, wave), self.initial)     # u-1 = 0: the initial bundle

    def test_a_unanimous_wave_never_touches_the_ledger(self) -> None:
        scorer = self.scorer_for(selfscored_spec(self.train))
        wave = self.wave_at("bundle:old", {"pi": 1})
        self.assertFalse(scorer.needs_previous_commit(wave))

    def test_a_pipeline_that_never_asks_the_policy_needs_no_pin(self) -> None:
        """A teacher scores its own base, so demanding a policy version would
        make another run's replayed trajectories unscoreable for no reason."""
        base = arith_spec(self.train)
        teacher = replace(base, algo=replace(
            base.algo, loss="opd", post=("verifier", "teacher_logprobs")),
            topology=Topology(hosts=(
                HostSpec((pool("main"),)),
                HostSpec((pool("teacher", base=TEACHER_BASE),)),
                HostSpec((learner(),)))))
        scorer = self.scorer_for(teacher)
        self.assertEqual(scorer.pools, ["teacher"])
        self.assertIs(scorer.pinned_bundle(4, self.wave_at("bundle:old", {})),
                      self.initial)

    def test_the_pinned_bundle_carries_the_whole_version_map(self) -> None:
        """A pin is id + version map and no payloads — what a request carries
        and what restore rebuilds from."""
        scorer = self.scorer_for(selfscored_spec(self.train))
        pinned = scorer.pinned_bundle(1, self.wave_at("bundle:x", {"pi": 2}))
        self.assertEqual(pinned.policy_version, {"pi": 2})
        self.assertEqual(pinned.payloads, {})


# ---------------------------------------------------------------------------
# the equivalence obligation
# ---------------------------------------------------------------------------

class TwoWorldsTest(unittest.TestCase):
    """The split preserves the columns of a complete reference pipeline."""

    def one_run(self, spec, engines) -> tuple[LocalStore, str]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, heldout = arith_store(tmp.name)
        report = run_experiment(spec(train, heldout), SCHEMA, store,
                                engines(), FakeLearner(), checkpointing=EVERY_UPDATE)
        return store, report.run_id

    def assert_unsplit_columns(self, make_spec, make_engines):
        from rlstack.data.trajectory import wave_from_rows
        from rlstack.policy.compile import restore_bundle
        from rlstack.runner.post import run_pipeline

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        spec = make_spec(train)
        metal = make_engines()
        engines = metal if isinstance(metal, dict) else {"main": metal}
        report = run_experiment(spec, SCHEMA, store, engines, FakeLearner(), checkpointing=EVERY_UPDATE)
        run = store.open_run(report.run_id)
        for update in range(1, 5):
            wave = wave_from_rows(run.read_wave(update))
            turn = wave.trajectories[0].turns[0]
            bundle = restore_bundle(turn.policy_version, turn.bundle_id,
                                    run.read_blob, ("pi",), {"pi": "lora"})
            engines["main"].add_bundle(bundle)
            routes = {"main": (engines["main"], bundle)}
            for name, engine in engines.items():
                if name != "main":
                    base = Bundle(f"bundle:base:{name}", {}, {})
                    engine.add_bundle(base)
                    routes[name] = (engine, base)
            columns = asyncio.run(run_pipeline(spec.algo.post, wave, routes,
                spec.gen.sampling, spec.seeds.master, update))
            self.assertEqual(columns, run.read_postdata(update))
            self.assertIsNotNone(run.read_postdata_part(update, SCORER))

    def test_a_judged_run_matches_unsplit_scoring(self) -> None:
        """A stochastic judge must keep the same per-processor seed path."""
        self.assert_unsplit_columns(judged_spec, judge_metal)

    def test_a_policy_pool_scorer_matches_unsplit_scoring(self) -> None:
        """Re-score exact historical pins after training has reached its tail."""
        self.assert_unsplit_columns(selfscored_spec, FakeEngine)

    def test_the_merged_postdata_holds_both_halves(self) -> None:
        store, run_id = self.one_run(
            lambda train, _: judged_spec(train),
            judge_metal)
        run = store.open_run(run_id)
        merged = run.read_postdata(1)
        self.assertEqual(sorted(merged), ["advantage", "reward"])
        part = run.read_postdata_part(1, SCORER)
        self.assertEqual(sorted(part), ["reward"])        # the pooled half only
        self.assertEqual(part["reward"], merged["reward"])

    def test_the_inline_half_sees_its_own_group_s_slice(self) -> None:
        """`given` arrives in WAVE order and is sliced back per group, so the
        z-score an inline advantage takes is over its own group's rewards and
        not the whole wave's — each group's advantages therefore sum to 0."""
        store, run_id = self.one_run(lambda train, _: judged_spec(train),
                                     judge_metal)
        run = store.open_run(run_id)
        merged = run.read_postdata(1)
        rows = run.read_wave(1)
        self.assertEqual(len(rows), len(merged["advantage"]))
        by_group: dict[str, list[float]] = {}
        for row, advantage in zip(rows, merged["advantage"]):
            by_group.setdefault(row["group"], []).append(advantage)
        self.assertGreater(len(by_group), 1, "the wave must have >1 group")
        for key, advantages in by_group.items():
            self.assertAlmostEqual(sum(advantages), 0.0, places=12, msg=key)

    def test_a_pipeline_with_no_pooled_half_writes_no_part(self) -> None:
        """No Scorer is planned at all, and the Trainer is what it always
        was — the DAPO/plora campaigns' shape."""
        store, run_id = self.one_run(lambda train, _: arith_spec(train),
                                     FakeEngine)
        files = snapshot(store, run_id)
        self.assertTrue(any(p.startswith("postdata/") for p in files))
        self.assertEqual(without_parts(files), files)


class ScorerResumeTest(unittest.TestCase):
    """Kill the run across the scorer boundary: the part is unsealed work, so
    attach discards it and the runner regenerates the same bytes."""

    CRASH_POINTS = [
        ("write_postdata_part", 2, True),   # update 3 scored, nothing merged
        ("write_postdata", 2, False),       # part written, merge never started
        ("append_ledger", 2, False),        # update 3 fully staged, uncommitted
    ]

    def engines(self):
        return judge_metal()

    def straight(self) -> dict[str, str]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        report = run_experiment(judged_spec(train), SCHEMA, store,
                                self.engines(), FakeLearner(), checkpointing=EVERY_UPDATE)
        self.run_id = report.run_id
        return snapshot(store, report.run_id)

    def test_crash_across_the_scorer_then_resume_is_byte_identical(self) -> None:
        reference = self.straight()

        for method, after, post in self.CRASH_POINTS:
            with self.subTest(crash=f"{method}@{after}{'+post' if post else ''}"):
                tmp = tempfile.TemporaryDirectory()
                self.addCleanup(tmp.cleanup)
                _, train, _ = arith_store(tmp.name)
                crashing = CrashingStore(tmp.name, method, after, post)

                with self.assertRaises(SimulatedCrash):
                    run_experiment(judged_spec(train), SCHEMA, crashing,
                                   self.engines(), FakeLearner(), checkpointing=EVERY_UPDATE)

                resumed = run_experiment(judged_spec(train), SCHEMA,
                                         LocalStore(tmp.name), self.engines(),
                                         FakeLearner(), checkpointing=EVERY_UPDATE)
                self.assertEqual(resumed.run_id, self.run_id)
                self.assertIsNotNone(resumed.resumed_from)
                self.assertEqual(snapshot(LocalStore(tmp.name), self.run_id),
                                 reference)


class ScorerConditionTest(unittest.TestCase):
    """The runner's two named conditions, which is where an alternation policy
    would be overridden."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, _ = arith_store(tmp.name)
        self.run = self.store.open_run("rid", manifest={"run_id": "rid"})

    def scorer(self) -> Scorer:
        return Scorer(RunSignals(), Arbiter(), self.run,
                      spec=selfscored_spec(self.train), plan=RunPlan(()),
                      refs=RefReader(self.store, self.run), residents=(),
                      routes_at=lambda bundle: {},
                      initial_bundle=Bundle("bundle:initial", {"pi": 0}))

    def test_already_scored_reads_the_part(self) -> None:
        scorer = self.scorer()
        self.assertFalse(scorer.already_scored(1))
        self.run.write_postdata_part(1, SCORER, {"hinted_logprobs": [[0.5]]})
        self.assertTrue(scorer.already_scored(1))

    def test_next_rows_prefers_the_wave_the_trainer_wrote(self) -> None:
        """One writer per artifact: the Scorer reads waves/<u> and never
        writes it."""
        rows = [dict(row, group="g") for row in
                [_row(sealed("t0"))]]
        self.run.write_wave(2, rows)
        self.assertEqual(self.scorer().next_rows(2), rows)


def _row(trajectory: Trajectory) -> dict:
    from rlstack import trajectory_to_row

    return trajectory_to_row(trajectory)


if __name__ == "__main__":
    unittest.main()
