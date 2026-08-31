"""Phase-0 validation (I4): one clean spec, then one test per issue code."""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

from rlstack.policy.adapters import AdapterType, Mechanism, adapter_type
from rlstack.policy.siteschema import fake_qwen_schema
from rlstack.registry import loss
from rlstack.runner.fakes import FakeEngine
from rlstack.training.post.base import PostProcessor, postprocessor
from rlstack.spec.specs import (
    AdapterSpec, AlgoSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup,
    OptimSpec, PolicySpec, Plans, Schedule, Seeds, WarmStart,
    gpus, learner, lora, pool,
)
from rlstack.spec.validate import (
    SpecError, ValidationIssue, check_sites_reachable_on, site_space, traffic_pools,
    validate, validate_or_raise,
)

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-1.7B")


# --- declarations these tests need ------------------------------------------

@loss("val_needs_values", requires=("values",))
def _needs_values(out: Any, batch: Any) -> Any: ...


# The #38 invariant, enforced at the door: requires names data columns
# only — a loss can never route work to metal.
class RequiresAreDataOnlyTest(unittest.TestCase):
    def test_non_string_requires_refused_at_registration(self) -> None:
        class Teacherish:
            pass

        with self.assertRaises(TypeError) as caught:
            @loss("val_planned_passes", requires=(Teacherish(), "reward"))
            def _planned(out: Any, batch: Any) -> Any: ...
        self.assertIn("post processor's job", str(caught.exception))


@postprocessor("val_needs_judge")
class _NeedsJudge(PostProcessor):
    consumes = ("judge",)
    produces = ("weighted",)

    async def process(self, group: Any, data: Any, client: Any) -> Any: ...


@postprocessor("val_also_reward")  # produces "reward" too: collides with verifier
class _AlsoReward(PostProcessor):
    produces = ("reward",)

    async def process(self, group: Any, data: Any, client: Any) -> Any: ...


@adapter_type("val_recording_adapter")
class _RecordingAdapter(AdapterType):
    """Trainer-only adapter whose rollout lowering records per-token draws."""

    serving = None
    records = ("adapter_draw",)

    def site_ok(self, meta: Any) -> bool:
        return not meta.has_weight


@loss("val_needs_draws", requires=("adapter_draw",))
def _needs_draws(out: Any, batch: Any) -> Any: ...


@loss("val_needs_behavior", requires=("behavior_logprobs",))
def _needs_behavior(out: Any, batch: Any) -> Any: ...


# --- the clean baseline ------------------------------------------------------

def clean_spec(**overrides: Any) -> ExperimentSpec:
    """Validates with zero issues against the 4-layer fake Qwen schema."""
    fields: dict[str, Any] = dict(
        policy=PolicySpec(base="Qwen/Qwen3-1.7B",
                          bank={"pi": lora("layers.0-3.self_attn.*", r=16)}),
        gen=GenSpec(envs=("noop_env",), tasks=("cas://x/train.jsonl",)),
        plans=Plans(train="cas://plan/train", rollout="cas://plan/roll"),
        algo=AlgoSpec(loss="grpo", post=("verifier", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=1e-5),
                      schedule=Schedule()),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=2), (pool("main"), learner())),)),
        seeds=Seeds(master=0),
    )
    fields.update(overrides)
    return ExperimentSpec(**fields)


def codes(spec: ExperimentSpec) -> set[str]:
    return {issue.code for issue in validate(spec, SCHEMA)}


class TestHappyPath(unittest.TestCase):
    def test_clean_spec_validates_clean(self) -> None:
        self.assertEqual(validate(clean_spec(), SCHEMA), [])

    def test_validate_or_raise_is_silent_when_clean(self) -> None:
        validate_or_raise(clean_spec(), SCHEMA)

    def test_offline_spec_without_gen_or_algo_validates(self) -> None:
        spec = clean_spec(gen=None, algo=None,
                          plans=Plans(train="cas://plan/train"))
        self.assertEqual(validate(spec, SCHEMA), [])

    def test_bank_provides_satisfy_requires(self) -> None:
        """A loss may require a tensor the bank's replay lowering computes
        (value_head provides "values") — the forward is not metal routing."""
        spec = clean_spec(policy=PolicySpec(
            base="Qwen/Qwen3-1.7B",
            bank={"v": AdapterSpec(adapter_type="value_head", site="value_head")}),
            algo=replace(clean_spec().algo, loss="val_needs_values"))
        self.assertEqual(
            [i for i in validate(spec, SCHEMA)
             if i.code == "unsatisfied-requires"], [])

    def test_trainer_only_adapter_type_validates(self) -> None:
        spec = clean_spec(policy=PolicySpec(
            base="Qwen/Qwen3-1.7B",
            bank={"pi": lora("layers.0-3.self_attn.*", r=16),
                  "critic": AdapterSpec(adapter_type="value_head", site="final_hidden")}))
        self.assertEqual(validate(spec, SCHEMA), [])


class TestUnknownNames(unittest.TestCase):
    def test_unknown_loss(self) -> None:
        spec = clean_spec(algo=replace(clean_spec().algo, loss="nope"))
        self.assertIn("unknown-loss", codes(spec))

    def test_unknown_post(self) -> None:
        spec = clean_spec(algo=replace(clean_spec().algo, post=("verifier", "nope")))
        self.assertIn("unknown-post", codes(spec))

    def test_unknown_env(self) -> None:
        spec = clean_spec(gen=replace(clean_spec().gen, envs=("nope",)))
        self.assertIn("unknown-env", codes(spec))

    def test_unknown_adapter_type(self) -> None:
        spec = clean_spec(policy=PolicySpec(
            base="Qwen/Qwen3-1.7B", bank={"x": AdapterSpec(adapter_type="nope", site="final_hidden")}))
        self.assertIn("unknown-adapter", codes(spec))


class TestDeclarationWiring(unittest.TestCase):
    def test_unsatisfied_requires(self) -> None:
        # val_needs_values requires "values"; nothing in the bank provides it.
        spec = clean_spec(algo=replace(clean_spec().algo, loss="val_needs_values"))
        self.assertEqual(codes(spec), {"unsatisfied-requires"})

    def test_requires_satisfied_by_a_value_head(self) -> None:
        spec = clean_spec(
            algo=replace(clean_spec().algo, loss="val_needs_values"),
            policy=PolicySpec(base="Qwen/Qwen3-1.7B", bank={
                "pi": lora("layers.0-3.self_attn.*", r=16),
                "critic": AdapterSpec(adapter_type="value_head", site="final_hidden")}))
        self.assertEqual(validate(spec, SCHEMA), [])

    def test_requires_satisfied_by_a_base_record(self) -> None:
        # behavior_logprobs is recorded on every trajectory (I6) — no bank help.
        spec = clean_spec(algo=replace(clean_spec().algo, loss="val_needs_behavior"))
        self.assertEqual(validate(spec, SCHEMA), [])

    def test_requires_satisfied_by_an_adapter_types_recorded_column(self) -> None:
        spec = clean_spec(
            algo=replace(clean_spec().algo, loss="val_needs_draws"),
            policy=PolicySpec(base="Qwen/Qwen3-1.7B", bank={
                "pi": lora("layers.0-3.self_attn.*", r=16),
                "router": AdapterSpec(adapter_type="val_recording_adapter",
                                      site="final_hidden")}))
        self.assertEqual(validate(spec, SCHEMA), [])

    def test_recorded_column_missing_without_its_adapter_type(self) -> None:
        spec = clean_spec(algo=replace(clean_spec().algo, loss="val_needs_draws"))
        self.assertEqual(codes(spec), {"unsatisfied-requires"})

    def test_post_unwired(self) -> None:
        # val_needs_judge consumes "judge"; nothing earlier produces it.
        spec = clean_spec(algo=replace(
            clean_spec().algo,
            post=("verifier", "val_needs_judge", "grpo_advantage")))
        self.assertEqual(codes(spec), {"post-unwired"})

    def test_post_order_matters(self) -> None:
        # grpo_advantage consumes "reward" — BEFORE verifier produces it.
        spec = clean_spec(algo=replace(clean_spec().algo,
                                       post=("grpo_advantage", "verifier")))
        self.assertEqual(codes(spec), {"post-unwired"})

    def test_post_collision(self) -> None:
        spec = clean_spec(algo=replace(
            clean_spec().algo,
            post=("verifier", "val_also_reward", "grpo_advantage")))
        self.assertEqual(codes(spec), {"post-collision"})


class TestSites(unittest.TestCase):
    def test_site_no_match(self) -> None:
        # The fake schema has 4 layers; layer 9 does not exist.
        spec = clean_spec(policy=PolicySpec(
            base="Qwen/Qwen3-1.7B", bank={"pi": lora("layers.9.self_attn.q_proj", r=8)}))
        self.assertEqual(codes(spec), {"site-no-match"})

    def test_site_predicate_failed(self) -> None:
        # value_head wants an unweighted boundary; q_proj is a weighted matrix.
        spec = clean_spec(policy=PolicySpec(
            base="Qwen/Qwen3-1.7B",
            bank={"critic": AdapterSpec(adapter_type="value_head",
                                        site="layers.0.self_attn.q_proj")}))
        self.assertEqual(codes(spec), {"site-predicate-failed"})

    def test_lora_predicate_rejects_a_boundary(self) -> None:
        spec = clean_spec(policy=PolicySpec(
            base="Qwen/Qwen3-1.7B", bank={"pi": AdapterSpec(adapter_type="lora", site="final_hidden",
                                              init={"r": 8})}))
        self.assertIn("site-predicate-failed", codes(spec))

    def test_schema_base_mismatch(self) -> None:
        wrong = fake_qwen_schema(4, base="someone/else")
        self.assertEqual({i.code for i in validate(clean_spec(), wrong)},
                         {"schema-base-mismatch"})

    def test_soft_prompt_resolves_at_its_own_export(self) -> None:
        """prompt[:8] is not a base site — the soft prompt entry creates it."""
        spec = clean_spec(policy=PolicySpec(
            base="Qwen/Qwen3-1.7B", bank={"latent": AdapterSpec(adapter_type="soft_prompt",
                                                  site="prompt[:8]",
                                                  init={"n": 8, "d": 64})}))
        self.assertEqual(validate(spec, SCHEMA), [])

    def test_attn_bias_needs_a_soft_prompt_to_export_its_site(self) -> None:
        """The bias rectangle references another entry's creation: without a
        soft prompt in the bank, nobody exports it — site-no-match."""
        alone = clean_spec(policy=PolicySpec(
            base="Qwen/Qwen3-1.7B",
            bank={"readout": AdapterSpec(adapter_type="attn_bias",
                                         site="queries -> prompt[:8]")}))
        self.assertEqual(codes(alone), {"site-no-match"})

        together = clean_spec(policy=PolicySpec(
            base="Qwen/Qwen3-1.7B",
            bank={"latent": AdapterSpec(adapter_type="soft_prompt", site="prompt[:8]",
                                        init={"n": 8, "d": 64}),
                  "readout": AdapterSpec(adapter_type="attn_bias",
                                         site="queries -> prompt[:8]")}))
        self.assertEqual(validate(together, SCHEMA), [])


class TestReachability(unittest.TestCase):
    """site-unreachable is a BUILD fact: not in CHECKS — the runner asks the
    serving pool's engine for its inventory and checks against the answer."""

    def spec_with(self, bank: dict) -> Any:
        return clean_spec(policy=PolicySpec(base="Qwen/Qwen3-1.7B", bank=bank))

    def reachability_codes(self, spec: Any, engine: FakeEngine) -> set[str]:
        space = site_space(spec, SCHEMA)
        return {i.code for i in check_sites_reachable_on(
            spec, SCHEMA, "main", engine.reachability(space))}

    def test_a_valid_spec_can_still_be_unservable_on_a_build(self) -> None:
        # soft_prompt at final_hidden: the SPEC is fine (site resolves, the
        # predicate passes) — but no build reaches model.norm via prompt_embeds.
        spec = self.spec_with({"latent": AdapterSpec(
            adapter_type="soft_prompt", site="final_hidden", init={"n": 8, "d": 64})})
        self.assertEqual(validate(spec, SCHEMA), [])
        self.assertEqual(self.reachability_codes(spec, FakeEngine()),
                         {"site-unreachable"})

    def test_lora_on_weighted_sites_is_reachable(self) -> None:
        self.assertEqual(self.reachability_codes(clean_spec(), FakeEngine()), set())

    def test_plugin_mechanisms_need_the_plugin_installed(self) -> None:
        spec = self.spec_with({
            "latent": AdapterSpec(adapter_type="soft_prompt", site="prompt[:8]",
                                  init={"n": 8, "d": 64}),
            "readout": AdapterSpec(adapter_type="attn_bias", site="queries -> prompt[:8]")})
        bare = FakeEngine()
        self.assertEqual(self.reachability_codes(spec, bare), {"site-unreachable"})
        patched = FakeEngine(plugins=frozenset({Mechanism.SIDE_ATTENTION}))
        self.assertEqual(self.reachability_codes(spec, patched), set())

    def test_trainer_only_adapter_types_are_never_checked(self) -> None:
        spec = self.spec_with({
            "pi": lora("layers.0-3.self_attn.*", r=16),
            "critic": AdapterSpec(adapter_type="value_head", site="final_hidden")})
        self.assertEqual(self.reachability_codes(spec, FakeEngine()), set())


class TestTopology(unittest.TestCase):
    def test_no_groups(self) -> None:
        spec = clean_spec(gpu_config=GpuConfig(groups=()))
        self.assertIn("no-groups", codes(spec))

    def test_duplicate_pool(self) -> None:
        spec = clean_spec(gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"), learner())),
            GpuGroup(gpus(n=1), (pool("main"),)),
        )))
        self.assertEqual(codes(spec), {"duplicate-pool"})

    def test_bad_sleep_group_no_learner(self) -> None:
        spec = clean_spec(gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"),), sharing="sleep"),)))
        self.assertEqual(codes(spec), {"bad-sleep-group"})

    def test_bad_sleep_group_two_learners(self) -> None:
        spec = clean_spec(gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"), learner(), learner()),
                  sharing="sleep"),)))
        self.assertEqual(codes(spec), {"bad-sleep-group"})

    def test_sleep_lag_conflict(self) -> None:
        sleepy = GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"), learner()), sharing="sleep"),))
        laggy = replace(clean_spec().algo,
                        schedule=Schedule(max_policy_lag=1))
        spec = clean_spec(gpu_config=sleepy, algo=laggy)
        self.assertEqual(codes(spec), {"sleep-lag-conflict"})

    def test_sleep_group_with_lag_zero_is_fine(self) -> None:
        spec = clean_spec(gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"), learner()), sharing="sleep"),)))
        self.assertEqual(validate(spec, SCHEMA), [])

    def test_fraction_overflow(self) -> None:
        spec = clean_spec(gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main", fraction=0.7),
                              learner(fraction=0.5))),)))
        self.assertEqual(codes(spec), {"fraction-overflow"})

    def test_fractions_summing_to_one_are_fine(self) -> None:
        spec = clean_spec(gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main", fraction=0.75),
                              learner(fraction=0.25))),)))
        self.assertEqual(validate(spec, SCHEMA), [])

    def test_partial_fractions_are_not_summed(self) -> None:
        spec = clean_spec(gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main", fraction=0.9), learner())),)))
        self.assertEqual(validate(spec, SCHEMA), [])

    def test_main_pool_missing(self) -> None:
        spec = clean_spec(gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("rollout"), learner())),)))
        self.assertEqual(codes(spec), {"main-pool-missing"})


class TestCoherence(unittest.TestCase):
    def test_rollout_plan_without_gen(self) -> None:
        # a rollout plan samples; gen is what declares the environments it may
        # run and the task sets it may draw from
        spec = clean_spec(gen=None)
        self.assertEqual(codes(spec), {"rollout-without-gen"})

    def test_store_source_without_gen_is_fine(self) -> None:
        spec = clean_spec(gen=None, plans=Plans(train="cas://plan/train"))
        self.assertEqual(validate(spec, SCHEMA), [])

    def test_bad_schedule_non_positive_count(self) -> None:
        bad = replace(clean_spec().algo, schedule=Schedule(microbatch_tokens=0))
        self.assertEqual(codes(clean_spec(algo=bad)), {"bad-schedule"})

    def test_bad_schedule_negative_lag(self) -> None:
        laggy = replace(clean_spec().algo,
                        schedule=Schedule(max_policy_lag=-1))
        self.assertEqual(codes(clean_spec(algo=laggy)), {"bad-schedule"})

    def test_algo_none_skips_schedule_and_loss_checks(self) -> None:
        self.assertEqual(validate(clean_spec(algo=None), SCHEMA), [])

    def test_warmstart_unknown_delta(self) -> None:
        # map is source-name -> THIS bank's name; "ghost" names no delta here.
        spec = clean_spec(init=WarmStart(policy="store://parent@40",
                                         map={"their_attn": "ghost"}))
        self.assertEqual(codes(spec), {"warmstart-unknown-delta"})

    def test_warmstart_known_delta_is_fine(self) -> None:
        spec = clean_spec(init=WarmStart(policy="store://parent@40",
                                         map={"their_attn": "pi"}))
        self.assertEqual(validate(spec, SCHEMA), [])


class TestSpecError(unittest.TestCase):
    def test_validate_or_raise_collects_every_issue(self) -> None:
        broken = clean_spec(
            algo=AlgoSpec(loss="nope", post=("also_nope",),
                          optim=OptimSpec("adamw", lr=1e-5),
                          schedule=Schedule(microbatch_tokens=0)),
            gpu_config=GpuConfig(groups=()),
        )
        with self.assertRaises(SpecError) as caught:
            validate_or_raise(broken, SCHEMA)
        found = {issue.code for issue in caught.exception.issues}
        self.assertLessEqual(
            {"unknown-loss", "unknown-post", "bad-schedule", "no-groups",
             "main-pool-missing"},
            found)
        self.assertIn("unknown-loss", str(caught.exception))

    def test_issue_is_frozen(self) -> None:
        issue = ValidationIssue("code", "path", "message")
        with self.assertRaises(Exception):
            issue.code = "other"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()


class TestPostPools(unittest.TestCase):
    """A pipeline processor's declared pools are vetted at submit."""

    def judge_algo(self):
        base = clean_spec().algo
        return replace(base, post=("llm_judge", "grpo_advantage"))

    def test_judge_without_declared_pool_is_caught(self) -> None:
        spec = clean_spec(algo=self.judge_algo())
        self.assertEqual(codes(spec), {"post-pool-missing"})

    def test_judge_with_declared_pool_is_clean(self) -> None:
        spec = clean_spec(
            algo=self.judge_algo(),
            gpu_config=GpuConfig(groups=(
                GpuGroup(gpus(n=2), (pool("main"), pool("judge"),
                                     learner())),)))
        self.assertEqual(validate(spec, SCHEMA), [])

    def test_traffic_pools_collects_every_route(self) -> None:
        spec = clean_spec(algo=self.judge_algo())
        self.assertEqual(traffic_pools(spec), {"main", "judge"})


class TestPostPoolCoresidency(unittest.TestCase):
    """A pipeline cannot need two pools of one sleep group co-resident."""

    def judge_algo(self):
        base = clean_spec().algo
        return replace(base, post=("llm_judge", "grpo_advantage"))

    def sleep_both(self):
        return GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"), pool("judge"), learner()),
                     sharing="sleep"),))

    def test_algo_judge_alone_coexists_with_sleeping_main(self) -> None:
        """The trainer admits only the pipeline's declared pools around post,
        so a judge-only pipeline is FINE even when its pool alternates with
        main — three-way alternation, one resident at a time."""
        spec = clean_spec(algo=self.judge_algo(),
                          gpu_config=self.sleep_both())
        self.assertNotIn("post-pools-conflict", codes(spec))

    def test_judge_in_its_own_group_is_clean(self) -> None:
        spec = clean_spec(
            algo=self.judge_algo(),
            gpu_config=GpuConfig(groups=(
                GpuGroup(gpus(n=1), (pool("main"), learner()),
                         sharing="sleep"),
                GpuGroup(gpus(n=1), (pool("judge"),)),)))
        self.assertNotIn("post-pools-conflict", codes(spec))
