"""Specs (SPEC.md §2A): construction, frozen-ness, identity, vocabularies, sugar.

One vocabulary a spec speaks is not a spec value: a plan's leaf names an
already-sealed trajectory by REF, and that grammar lives with its resolver
(rlstack.runner.refs). It is tested here beside the other closed grammars
because it is the one TrajectorySource's three forms became (#59).
"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, fields, replace

from common import TRAIN, arith_spec, arith_task_bytes, cas_uri
from rlstack.policy.siteschema import fake_qwen_schema
from rlstack.runner.loop import experiment_identity
from rlstack.runner.refs import parse
from rlstack.spec.canonical import content_hash
from rlstack.spec.specs import (
    AdapterSpec,
    AlgoSpec,
    BackendProfile,
    PoolMember,
    ExperimentSpec,
    GenSpec,
    Topology,
    HostSpec,
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
                    tasks=("cas://3fa9c2.../math_train.jsonl",)),
        plans=Plans(train="cas://plan/train", rollout="cas://plan/roll"),
        algo=AlgoSpec(
            loss="grpo",
            post=("verifier", "grpo_advantage"),
            optim=OptimSpec("adamw", lr=1e-5, betas=(0.9, 0.95),
                            overrides={"head": {"lr": 3e-6}}),
            schedule=Schedule(microbatch_tokens=16384,
                              max_policy_lag=0),
        ),
        topology=Topology(hosts=(
            HostSpec((pool("main", tp=2),)),
            HostSpec((learner(fsdp=2),)),
            HostSpec((pool("eval", tp=1),)),
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
        self.assertEqual(self.exp.algo.schedule.microbatch_tokens, 16384)
        self.assertEqual(self.exp.plans.train, "cas://plan/train")
        self.assertIsNone(self.exp.init)
        self.assertEqual(self.exp.tier, "lab")

    def test_topology_members(self) -> None:
        """One HostSpec is one host: a single member is a dedicated host, and
        a HostSpec carries members and nothing else — no device set, no
        sharing word (ADR 0001)."""
        serve, train, _ = self.exp.topology.hosts
        self.assertIsInstance(serve.members[0], PoolMember)
        self.assertIsInstance(train.members[0], LearnerMember)
        self.assertEqual(serve.members[0].name, "main")
        self.assertEqual([f.name for f in fields(HostSpec)], ["members"])

    def test_equality_and_replace(self) -> None:
        self.assertEqual(self.exp, example_1())
        other = replace(self.exp, seeds=Seeds(master=18))
        self.assertNotEqual(self.exp, other)

    def test_optional_halves(self) -> None:
        offline = replace(self.exp, gen=None,
                          plans=Plans(train="cas://plan/train"))
        self.assertIsNone(offline.gen)
        generation_only = replace(self.exp, algo=None)
        self.assertIsNone(generation_only.algo)

    def test_sharded_topology_example_5(self) -> None:
        """Pure demand: shard widths and GB, no device counts, no nodes, no
        provider names. `vram_gb` is TOTAL across the shards (Q3), so the
        number survives a re-sharding untouched."""
        config = Topology(hosts=(
            HostSpec((pool("main", tp=2, vram_gb=120),)),
            HostSpec((learner(fsdp=8, vram_gb=400),)),
        ))
        self.assertEqual(config.hosts[0].members[0].tp, 2)
        self.assertEqual(config.hosts[1].members[0].vram_gb, 400)

    def test_alternating_host_example_6(self) -> None:
        """Several members on one HostSpec ALTERNATE on one partition; a
        member without a size wants a whole device per shard (Q10)."""
        both = HostSpec((pool("main", vram_gb=12), learner()))
        self.assertEqual(both.members[0].vram_gb, 12)
        self.assertIsNone(both.members[1].vram_gb)


class TestFrozen(unittest.TestCase):
    def test_assigning_any_field_raises(self) -> None:
        exp = example_1()
        for obj, field_name, value in [
            (exp, "seeds", Seeds(master=0)),
            (exp.policy, "base", "other"),
            (exp.algo.schedule, "microbatch_tokens", 1),
            (exp.topology.hosts[0], "members", ()),
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

    def test_an_existing_spec_keeps_the_run_id_it_has_always_had(self) -> None:
        """ADR 0006 Part B, promise 1: `Plans.train` became optional and the
        extent a derived PROPERTY, so no hashed record gained a field and no
        existing run's identity moved. The literal below is what a pre-Part-B
        tree computes for tests/common.py's arith_spec — if it ever changes,
        every sealed run in every store has been orphaned."""
        spec = arith_spec(cas_uri(arith_task_bytes(TRAIN[1], TRAIN[2], TRAIN[0])))
        schema = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
        self.assertEqual(experiment_identity(spec, schema), "2432682289f4")


class TestVocabularies(unittest.TestCase):
    def test_a_ref_accepts_the_four_locations(self) -> None:
        """Where an already-sealed trajectory lives — the vocabulary a plan's
        Replay leaf speaks (runner/refs.py), and the one TrajectorySource's
        "live" / "store://" / "cas://" grammar became when live/replay/static
        stopped being kinds of RUN and became kinds of LEAF (#59). Four
        locations, plus an optional `#index` picking one row out of a wave —
        another run's ROLLOUTS joined at ADR 0006 Part B, because a
        generation-only run seals those and never writes waves/."""
        for ref, location, index in (
            ("self://rollouts/3", "self://rollouts/3", None),
            ("self://rollouts/3#2", "self://rollouts/3", 2),
            ("store://a1b2c3/waves/7#0", "store://a1b2c3/waves/7", 0),
            ("store://a1b2c3/rollouts/7#0", "store://a1b2c3/rollouts/7", 0),
            ("cas://3fa9/anchors.jsonl#41", "cas://3fa9/anchors.jsonl", 41),
        ):
            with self.subTest(ref=ref):
                parsed = parse(ref)
                self.assertEqual((parsed.location, parsed.index),
                                 (location, index))
        # this run's own rollouts may answer "not yet", and so may another
        # run's while it is still running (refs.py decides that); bytes cannot
        self.assertTrue(parse("self://rollouts/3#2").pending_allowed)
        self.assertTrue(parse("store://a1b2c3/waves/7#0").pending_allowed)
        self.assertTrue(parse("store://a1b2c3/rollouts/7#0").pending_allowed)
        self.assertFalse(parse("cas://3fa9/anchors.jsonl#41").pending_allowed)

    def test_a_ref_rejects_anything_else(self) -> None:
        """An unknown location, and a row index that is not a row number."""
        for ref in ("live", "s3://bucket/x", "/tmp/rollouts", "", "SELF://x",
                    "self://rollouts/3#last"):
            with self.subTest(ref=ref):
                with self.assertRaises(ValueError):   # PlanError is one
                    parse(ref)

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

    def test_plans(self) -> None:
        """A plan is absent when the run does not do that thing: no rollout
        plan makes nothing (every train leaf is sealed elsewhere), no train
        plan trains nothing (a generation-only run)."""
        p = Plans(train="cas://plan/train")
        self.assertEqual(p.rollout, None)
        self.assertEqual(Plans(rollout="cas://plan/roll").train, None)

    def test_extent_is_the_plan_that_is_the_run_s_length(self) -> None:
        """ADR 0006 Part B: the train plan when there is one — one wave is one
        update — else the rollout plan, whose last sealed wave ends a run that
        only generates. Derived, never a field, so no hashed record moved."""
        self.assertEqual(Plans(train="cas://t", rollout="cas://r").extent,
                         "train")
        self.assertEqual(Plans(train="cas://t").extent, "train")
        self.assertEqual(Plans(rollout="cas://r").extent, "rollout")
        self.assertNotIn("extent", [f.name for f in fields(Plans)])

    def test_measurement_left_the_spec(self) -> None:
        """#70: a run's identity is its training loop. Plans carries no eval
        and the spec no EvalSpec — measurement is an observation OUTSIDE the
        run (runner/measure.py), configured by its own manifest."""
        self.assertEqual([f.name for f in fields(Plans)],
                         ["train", "rollout"])
        self.assertNotIn("eval",
                         [f.name for f in fields(ExperimentSpec)])

    def test_schedule(self) -> None:
        """Schedule has exactly TWO fields, because the plan states the rest:
        group_size, trajectories_per_wave and n_updates are the plan's by
        construction, and epochs_per_wave left with the invariant it violated —
        one wave is one gradient update. What remains is one engineering knob
        (how a wave's compute is chunked) and one statistical one (how stale a
        behavior policy the trainer tolerates)."""
        s = Schedule()
        self.assertEqual((s.microbatch_tokens, s.max_policy_lag), (16384, 0))
        self.assertEqual([f.name for f in fields(Schedule)],
                         ["microbatch_tokens", "max_policy_lag"])

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
    def test_pool_and_learner(self) -> None:
        self.assertEqual(pool("main", tp=2, vram_gb=30),
                         PoolMember(name="main", tp=2, vram_gb=30))
        self.assertEqual(learner(fsdp=2), LearnerMember(fsdp=2))
        self.assertIsNone(pool("main").vram_gb)

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
