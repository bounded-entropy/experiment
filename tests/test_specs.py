"""Specs (SPEC.md §2A): construction, frozen-ness, identity, vocabularies, sugar."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace

from rlstack.spec.canonical import content_hash
from rlstack.spec.specs import (
    AdapterSpec,
    AlgoSpec,
    BackendProfile,
    PoolMember,
    EvalSpec,
    ExperimentSpec,
    GenSpec,
    GpuConfig,
    GpuSet,
    GpuGroup,
    LearnerMember,
    OptimSpec,
    PolicySpec,
    Plans,
    SamplingSpec,
    Schedule,
    Seeds,
    WarmStart,
    attn_bias,
    pool,
    gpus,
    learner,
    lora,
    soft_prompt,
)

BANK = {
    "attn": lora(site="layers.0-15.self_attn.*", r=16),
    "mlp": lora(site="layers.16-31.mlp.*", r=32),
    "head": lora(site="layers.28-31.self_attn.o_proj", r=8),
}


def example_1(bank: dict[str, AdapterSpec] | None = None) -> ExperimentSpec:
    """SPEC.md Example 1 — a basic LoRA experiment, end to end."""
    return ExperimentSpec(
        policy=PolicySpec(
            base="Qwen/Qwen3-8B",
            bank=dict(BANK) if bank is None else bank,
        ),
        gen=GenSpec(envs=("math_single_turn",),
                    tasks=("cas://3fa9c2.../math_train.jsonl",),

        ),
        plans=Plans(train="cas://plan/train", rollout="cas://plan/roll"),
        algo=AlgoSpec(
            loss="grpo",
            post=("verifier", "grpo_advantage"),
            optim=OptimSpec("adamw", lr=1e-5, betas=(0.9, 0.95),
                            overrides={"head": {"lr": 3e-6}}),
            schedule=Schedule(microbatch_tokens=16384,
                              max_policy_lag=0),
        ),
        eval=EvalSpec(every=10),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=6), (pool("main", tp=2, n=3), learner(fsdp=2)),
                  sharing="concurrent"),
            GpuGroup(gpus(n=1), (pool("eval", tp=1, n=1),)),
        )),
        seeds=Seeds(master=17),
    )


class TestConstruction(unittest.TestCase):
    def setUp(self) -> None:
        self.exp = example_1()

    def test_constructs_and_reads_back(self) -> None:
        self.assertEqual(self.exp.policy.base, "Qwen/Qwen3-8B")
        self.assertEqual(self.exp.policy.bank["attn"].init["r"], 16)
        self.assertEqual(self.exp.algo.optim.overrides["head"]["lr"], 3e-6)
        self.assertEqual(self.exp.algo.schedule.trajectories_per_wave, 512)
        self.assertIsNone(self.exp.init)
        self.assertEqual(self.exp.tier, "lab")

    def test_topology_members(self) -> None:
        main_group = self.exp.gpu_config.groups[0]
        self.assertIsInstance(main_group.members[0], PoolMember)
        self.assertIsInstance(main_group.members[1], LearnerMember)
        self.assertEqual(main_group.members[0].name, "main")
        self.assertEqual(main_group.sharing, "concurrent")

    def test_equality_and_replace(self) -> None:
        self.assertEqual(self.exp, example_1())
        other = replace(self.exp, seeds=Seeds(master=18))
        self.assertNotEqual(self.exp, other)

    def test_optional_halves(self) -> None:
        offline = replace(self.exp, gen=None, eval=None,
                          plans=Plans(train="cas://plan/train"))
        self.assertIsNone(offline.gen)
        generation_only = replace(self.exp, algo=None)
        self.assertIsNone(generation_only.algo)

    def test_multi_node_topology_example_5(self) -> None:
        config = GpuConfig(groups=(
            GpuGroup(gpus(n=16, nodes=2), (pool("main", tp=2, n=8),)),
            GpuGroup(gpus(n=8), (learner(fsdp=8),)),
        ))
        self.assertEqual(config.groups[0].gpus.nodes, 2)
        self.assertEqual(config.groups[0].members[0].n, 8)

    def test_fractional_colocation_example_6(self) -> None:
        solo = GpuGroup(gpus(ids=("0",)),
                     (pool("main", n=2, fraction=0.30), learner(fraction=0.25)),
                     sharing="concurrent")
        self.assertEqual(solo.members[0].fraction, 0.30)
        self.assertEqual(solo.gpus.ids, ("0",))


class TestFrozen(unittest.TestCase):
    def test_assigning_any_field_raises(self) -> None:
        exp = example_1()
        for obj, field_name, value in [
            (exp, "seeds", Seeds(master=0)),
            (exp.policy, "base", "other"),
            (exp.algo.schedule, "group_size", 1),
            (exp.gpu_config.groups[0], "sharing", "sleep"),
        ]:
            with self.assertRaises(FrozenInstanceError):
                setattr(obj, field_name, value)

    def test_deleting_a_field_raises(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            del example_1().seeds.master


class TestIdentity(unittest.TestCase):
    """Identity is computed at hash time; literal order never matters (I3)."""

    def test_bank_order_does_not_change_identity(self) -> None:
        reversed_bank = {k: BANK[k] for k in reversed(list(BANK))}
        self.assertEqual(content_hash(example_1()), content_hash(example_1(reversed_bank)))

    def test_any_field_change_changes_identity(self) -> None:
        exp = example_1()
        self.assertNotEqual(content_hash(exp),
                            content_hash(replace(exp, seeds=Seeds(master=18))))

    def test_warm_start_hashes_into_identity(self) -> None:
        exp = example_1()
        warm = replace(exp, init=WarmStart(policy="store://parent@40", optim="load"))
        self.assertNotEqual(content_hash(exp), content_hash(warm))


class TestVocabularies(unittest.TestCase):
    def test_rollout_source_accepts_the_three_forms(self) -> None:
        for source in ("live", "store://run/waves", "cas://3fa9/tasks.jsonl"):
            self.assertEqual(TrajectorySource(source).source, source)

    def test_rollout_source_rejects_anything_else(self) -> None:
        for source in ("s3://bucket/x", "/tmp/rollouts", "", "LIVE"):
            with self.assertRaises(ValueError):
                TrajectorySource(source)

    def test_sharing_vocabulary(self) -> None:
        group = GpuGroup(gpus(n=1), (learner(),), sharing="sleep")
        self.assertEqual(group.sharing, "sleep")
        with self.assertRaises(ValueError):
            GpuGroup(gpus(n=1), (learner(),), sharing="timeshare")

    def test_tier_vocabulary(self) -> None:
        self.assertEqual(replace(example_1(), tier="release").tier, "release")
        with self.assertRaises(ValueError):
            replace(example_1(), tier="prod")

    def test_warm_start_optim_vocabulary(self) -> None:
        for optim in ("load", "fresh"):
            WarmStart(policy="store://run@3", optim=optim)
        with self.assertRaises(ValueError):
            WarmStart(policy="store://run@3", optim="reuse")

    def test_warm_start_policy_must_be_sealed_state(self) -> None:
        WarmStart(policy="cas://beef00")
        with self.assertRaises(ValueError):
            WarmStart(policy="live")
        with self.assertRaises(ValueError):
            WarmStart(policy="/tmp/ckpt")


class TestDefaults(unittest.TestCase):
    def test_sampling_spec(self) -> None:
        s = SamplingSpec()
        self.assertEqual((s.temperature, s.top_p, s.max_tokens), (1.0, 1.0, 1024))

    def test_eval_spec(self) -> None:
        e = EvalSpec()
        self.assertEqual((e.every, e.env, e.post, e.n_samples, e.pool),
                         (10, None, (), 1, "main"))

    def test_schedule(self) -> None:
        s = Schedule()
        self.assertEqual((s.epochs_per_wave, s.microbatch_tokens, s.max_policy_lag),
                         (1, 16384, 0))

    def test_optim_spec(self) -> None:
        o = OptimSpec("adamw", lr=1e-5)
        self.assertEqual((o.betas, o.weight_decay, dict(o.overrides)),
                         ((0.9, 0.95), 0.0, {}))

    def test_adapter_spec(self) -> None:
        a = AdapterSpec(adapter_type="lora", site="layers.*.mlp.*")
        self.assertEqual((dict(a.init), a.trainable), ({}, True))

    def test_mapping_defaults_are_not_shared(self) -> None:
        a = AdapterSpec(adapter_type="lora", site="x")
        b = AdapterSpec(adapter_type="lora", site="x")
        self.assertIsNot(a.init, b.init)

    def test_backend_profile_stays_outside_the_spec(self) -> None:
        profile = BackendProfile(kind="modal", gpu="H100:8", nodes=3, idle="snapshot")
        self.assertNotIn("backend", {f.name for f in
                                     __import__("dataclasses").fields(ExperimentSpec)})
        self.assertEqual(profile.nodes, 3)


class TestSugar(unittest.TestCase):
    def test_gpus(self) -> None:
        self.assertEqual(gpus(6), GpuSet(n=6))
        self.assertEqual(gpus(16, nodes=2), GpuSet(n=16, nodes=2))
        self.assertEqual(gpus(ids=("0", "1")), GpuSet(ids=("0", "1")))

    def test_pool_and_learner(self) -> None:
        self.assertEqual(pool("main", tp=2, n=3),
                         PoolMember(name="main", tp=2, n=3))
        self.assertEqual(learner(fsdp=2), LearnerMember(fsdp=2))

    def test_lora(self) -> None:
        a = lora("layers.*.mlp.*", r=16)
        self.assertEqual((a.adapter_type, a.site, dict(a.init)),
                         ("lora", "layers.*.mlp.*", {"r": 16, "tie": False}))

    def test_soft_prompt(self) -> None:
        a = soft_prompt("prompt[:8]", n=8, d=2048)
        self.assertEqual((a.adapter_type, dict(a.init)), ("soft_prompt", {"n": 8, "d": 2048}))

    def test_attn_bias(self) -> None:
        a = attn_bias("queries -> prompt[:8]", param="bounded_sigmoid", cap="ln(64)")
        self.assertEqual(a.adapter_type, "attn_bias")
        self.assertEqual(a.init["param"], "bounded_sigmoid")

    def test_sugar_matches_longhand(self) -> None:
        self.assertEqual(
            lora("layers.0-15.self_attn.*", r=16),
            AdapterSpec(adapter_type="lora", site="layers.0-15.self_attn.*",
                        init={"r": 16, "tie": False}),
        )


if __name__ == "__main__":
    unittest.main()
