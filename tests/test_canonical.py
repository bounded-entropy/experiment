"""Canonical JSON, content hashing, and computed identity (I3)."""

from __future__ import annotations

import hashlib
import math
import unittest
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from types import MappingProxyType

from rlstack.spec.canonical import canonical_json, content_hash, run_id
from rlstack.spec.specs import (
    AdapterSpec,
    AlgoSpec,
    PoolMember,
    ExperimentSpec,
    GenSpec,
    GpuConfig,
    GpuGroup,
    OptimSpec,
    PolicySpec,
    Plans,
    SamplingSpec,
    Schedule,
    Seeds,
    pool,
    gpus,
    learner,
    lora,
)


class Color(Enum):
    RED = "red"
    ZERO = 0


class Rung(IntEnum):
    FA3 = 3
    FA2 = 2


@dataclass(frozen=True)
class _Pair:
    a: int
    b: str


@dataclass(frozen=True)
class _OtherPair:
    a: int
    b: str


def _small_spec() -> ExperimentSpec:
    """A minimal but complete spec — small enough to golden, real enough to matter."""
    return ExperimentSpec(
        policy=PolicySpec(
            base="Qwen/Qwen3-1.7B",
            bank={"pi": lora("layers.*.mlp.*", r=16)},
        ),
        gen=GenSpec(envs=("math_single_turn",),
                    tasks=("cas://abc/train.jsonl",)),
        plans=Plans(train="cas://plan/train", rollout="cas://plan/roll"),
        algo=AlgoSpec(
            loss="grpo",
            post=("verifier", "grpo_advantage"),
            optim=OptimSpec("adamw", lr=1e-5),
            schedule=Schedule(),
        ),
        gpu_config=GpuConfig(
            groups=(GpuGroup(gpus(n=1), (pool("main"), learner())),)
        ),
        seeds=Seeds(master=0),
    )


class TestCanonicalJsonShape(unittest.TestCase):
    def test_golden_small_dataclass(self) -> None:
        # Golden string: byte-for-byte stability is the whole point of canonical JSON.
        self.assertEqual(
            canonical_json(AdapterSpec(adapter_type="lora", site="layers.0.self_attn.q",
                                       init={"tie": False, "r": 8})),
            '{"__type__":"AdapterSpec","adapter_type":"lora",'
            '"init":{"r":8,"tie":false},'
            '"site":"layers.0.self_attn.q","trainable":true}',
        )

    def test_golden_nested_dataclass_and_tuple(self) -> None:
        # gen DECLARES: both envs and tasks are tuples, and a tuple canonicalizes
        # as a json array in declaration order (never sorted — order is content).
        self.assertEqual(
            canonical_json(GenSpec(envs=("math_single_turn",),
                                   tasks=("cas://abc/train.jsonl",),
                                   sampling=SamplingSpec(top_p=0.9))),
            '{"__type__":"GenSpec","envs":["math_single_turn"],'
            '"sampling":{"__type__":"SamplingSpec","max_tokens":1024,'
            '"temperature":1.0,"top_p":0.9},"tasks":["cas://abc/train.jsonl"]}',
        )

    def test_golden_full_small_spec(self) -> None:
        # Stability of the whole tree, not just a leaf.
        self.assertEqual(content_hash(_small_spec()), content_hash(_small_spec()))
        self.assertEqual(
            canonical_json(_small_spec()),
            '{"__type__":"ExperimentSpec","algo":{"__type__":"AlgoSpec",'
            '"loss":"grpo",'
            '"optim":{"__type__":"OptimSpec","betas":[0.9,0.95],"lr":1e-05,'
            '"name":"adamw","overrides":{},"weight_decay":0.0},'
            '"post":["verifier","grpo_advantage"],'
            '"schedule":{"__type__":"Schedule","max_policy_lag":0,'
            '"microbatch_tokens":16384}},"eval":null,'
            '"gen":{"__type__":"GenSpec","envs":["math_single_turn"],'
            '"sampling":{"__type__":"SamplingSpec",'
            '"max_tokens":1024,"temperature":1.0,"top_p":1.0},'
            '"tasks":["cas://abc/train.jsonl"]},'
            '"gpu_config":{"__type__":"GpuConfig","groups":[{"__type__":"GpuGroup",'
            '"gpus":{"__type__":"GpuSet","ids":null,"n":1,"nodes":1},'
            '"members":[{"__type__":"PoolMember","base":null,"fraction":null,'
            '"n":1,"name":"main","tp":1},{"__type__":"LearnerMember",'
            '"fraction":null,"fsdp":1}],"sharing":"concurrent"}]},"init":null,'
            '"plans":{"__type__":"Plans","eval":null,'
            '"rollout":"cas://plan/roll","train":"cas://plan/train"},'
            '"policy":{"__type__":"PolicySpec","bank":{"pi":{"__type__":'
            '"AdapterSpec","adapter_type":"lora","init":{"r":16,"tie":false},'
            '"site":"layers.*.mlp.*","trainable":true}},"base":"Qwen/Qwen3-1.7B"},'
            '"seeds":{"__type__":"Seeds","master":0},"tier":"lab"}',
        )

    def test_the_plans_are_in_the_identity_by_reference(self) -> None:
        """A plan hashes in as its cas uri — the sha IS the plan's content, so
        naming a different plan is a different experiment without the spec
        having to carry the waves (#59)."""
        other = replace(_small_spec(),
                        plans=Plans(train="cas://plan/other",
                                    rollout="cas://plan/roll"))
        self.assertNotEqual(content_hash(_small_spec()), content_hash(other))
        self.assertIn('"train":"cas://plan/other"', canonical_json(other))

    def test_type_tag_separates_structurally_identical_classes(self) -> None:
        self.assertNotEqual(canonical_json(_Pair(1, "x")), canonical_json(_OtherPair(1, "x")))
        self.assertIn('"__type__":"_Pair"', canonical_json(_Pair(1, "x")))

    def test_no_whitespace_and_unicode_preserved(self) -> None:
        self.assertEqual(canonical_json({"k": "héllo ✓"}), '{"k":"héllo ✓"}')
        self.assertNotIn(" ", canonical_json(_Pair(1, "x")))

    def test_tuple_and_list_are_the_same_array(self) -> None:
        self.assertEqual(canonical_json(("a", "b")), canonical_json(["a", "b"]))
        self.assertEqual(canonical_json(("a", "b")), '["a","b"]')

    def test_scalars_passthrough(self) -> None:
        self.assertEqual(canonical_json(None), "null")
        self.assertEqual(canonical_json(True), "true")
        self.assertEqual(canonical_json(False), "false")
        self.assertEqual(canonical_json(17), "17")
        self.assertEqual(canonical_json("s"), '"s"')
        self.assertEqual(canonical_json(1.5), "1.5")

    def test_bool_is_not_int(self) -> None:
        self.assertNotEqual(canonical_json(True), canonical_json(1))

    def test_float_repr_precision_stable(self) -> None:
        for value in (1e-5, 0.1, 1.0, 3e-6, 1e300, -0.0):
            with self.subTest(value=value):
                self.assertEqual(float(canonical_json(value)), value)

    def test_enum_becomes_its_value(self) -> None:
        self.assertEqual(canonical_json(Color.RED), '"red"')
        self.assertEqual(canonical_json(Color.ZERO), "0")
        self.assertEqual(canonical_json(Rung.FA3), "3")
        self.assertEqual(canonical_json({"rung": Rung.FA2}), '{"rung":2}')

    def test_mapping_proxy_is_a_mapping(self) -> None:
        self.assertEqual(canonical_json(MappingProxyType({"b": 1, "a": 2})), '{"a":2,"b":1}')


class TestCanonicalJsonDeterminism(unittest.TestCase):
    def test_repeated_calls_are_byte_identical(self) -> None:
        spec = _small_spec()
        self.assertEqual(canonical_json(spec), canonical_json(spec))

    def test_mapping_insertion_order_is_invisible(self) -> None:
        forward = {"a": 1, "m": {"x": 1, "y": 2}, "z": 3}
        backward = {"z": 3, "m": {"y": 2, "x": 1}, "a": 1}
        self.assertEqual(canonical_json(forward), canonical_json(backward))
        self.assertEqual(canonical_json(forward), '{"a":1,"m":{"x":1,"y":2},"z":3}')

    def test_fields_sorted_by_name(self) -> None:
        import json

        # json.loads preserves document order, so this reads the emitted key order.
        keys = list(json.loads(canonical_json(_small_spec())).keys())
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(keys[0], "__type__")


class TestCanonicalJsonRejections(unittest.TestCase):
    def test_nan_rejected(self) -> None:
        with self.assertRaises(TypeError):
            canonical_json(float("nan"))

    def test_infinities_rejected(self) -> None:
        for value in (math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    canonical_json(value)

    def test_nan_rejected_deep_inside_a_spec(self) -> None:
        spec = AdapterSpec(adapter_type="lora", site="s", init={"scale": float("nan")})
        with self.assertRaises(TypeError):
            canonical_json(spec)

    def test_non_str_mapping_key_rejected(self) -> None:
        with self.assertRaises(TypeError):
            canonical_json({1: "one"})

    def test_unsupported_types_rejected_by_name(self) -> None:
        for value in ({"a"}, object(), complex(1, 2), b"bytes", lambda: None):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(TypeError) as ctx:
                    canonical_json(value)
                self.assertIn(type(value).__name__, str(ctx.exception))


class TestContentHash(unittest.TestCase):
    def test_is_sha256_of_canonical_json(self) -> None:
        spec = _small_spec()
        self.assertEqual(
            content_hash(spec),
            hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest(),
        )

    def test_shape(self) -> None:
        digest = content_hash(_small_spec())
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in digest))

    def test_changes_when_any_nested_field_changes(self) -> None:
        import dataclasses

        base = _small_spec()
        baseline = content_hash(base)
        mutations = {
            "tier": dataclasses.replace(base, tier="release"),
            "seed": dataclasses.replace(base, seeds=Seeds(master=1)),
            "sampling": dataclasses.replace(
                base,
                gen=dataclasses.replace(
                    base.gen, sampling=SamplingSpec(temperature=0.7)
                ),
            ),
            "lr": dataclasses.replace(
                base,
                algo=dataclasses.replace(
                    base.algo, optim=OptimSpec("adamw", lr=2e-5)
                ),
            ),
            "rank": dataclasses.replace(
                base,
                policy=PolicySpec(base=base.policy.base,
                                  bank={"pi": lora("layers.*.mlp.*", r=32)}),
            ),
            "bank_name": dataclasses.replace(
                base,
                policy=PolicySpec(base=base.policy.base,
                                  bank={"policy": lora("layers.*.mlp.*", r=16)}),
            ),
            "tp": dataclasses.replace(
                base,
                gpu_config=GpuConfig(
                    groups=(GpuGroup(gpus(n=1), (pool("main", tp=2), learner())),)
                ),
            ),
        }
        for name, mutated in mutations.items():
            with self.subTest(field=name):
                self.assertNotEqual(content_hash(mutated), baseline)

    def test_invariant_to_mapping_insertion_order(self) -> None:
        forward = PolicySpec(base="b", bank={"a": lora("s1", r=1), "z": lora("s2", r=2)})
        backward = PolicySpec(base="b", bank={"z": lora("s2", r=2), "a": lora("s1", r=1)})
        self.assertEqual(content_hash(forward), content_hash(backward))

    def test_invariant_to_nested_mapping_insertion_order(self) -> None:
        forward = OptimSpec("adamw", lr=1e-5,
                            overrides={"head": {"lr": 3e-6, "weight_decay": 0.0}})
        backward = OptimSpec("adamw", lr=1e-5,
                             overrides={"head": {"weight_decay": 0.0, "lr": 3e-6}})
        self.assertEqual(content_hash(forward), content_hash(backward))


class TestRunId(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = _small_spec()
        self.code = {"env:math_single_turn": "aa11", "loss:grpo": "bb22"}
        self.data = "cas://3fa9c2/train.jsonl"

    def test_shape_is_12_hex_chars(self) -> None:
        rid = run_id(self.spec, self.code, self.data)
        self.assertEqual(len(rid), 12)
        self.assertTrue(all(c in "0123456789abcdef" for c in rid))

    def test_is_prefix_of_the_content_hash_of_the_triple(self) -> None:
        self.assertEqual(
            run_id(self.spec, self.code, self.data),
            content_hash({"spec": self.spec, "code": self.code, "data": self.data})[:12],
        )

    def test_deterministic(self) -> None:
        self.assertEqual(
            run_id(self.spec, self.code, self.data),
            run_id(_small_spec(), dict(self.code), self.data),
        )

    def test_sensitive_to_spec(self) -> None:
        import dataclasses

        other = dataclasses.replace(self.spec, seeds=Seeds(master=1))
        self.assertNotEqual(
            run_id(other, self.code, self.data),
            run_id(self.spec, self.code, self.data),
        )

    def test_sensitive_to_code_hashes(self) -> None:
        # I3: editing a registered function's body changes identity.
        edited = dict(self.code) | {"loss:grpo": "cc33"}
        self.assertNotEqual(
            run_id(self.spec, edited, self.data),
            run_id(self.spec, self.code, self.data),
        )

    def test_sensitive_to_data_fingerprint(self) -> None:
        self.assertNotEqual(
            run_id(self.spec, self.code, "cas://deadbeef/train.jsonl"),
            run_id(self.spec, self.code, self.data),
        )

    def test_code_hash_mapping_order_invisible(self) -> None:
        reordered = {"loss:grpo": "bb22", "env:math_single_turn": "aa11"}
        self.assertEqual(
            run_id(self.spec, reordered, self.data),
            run_id(self.spec, self.code, self.data),
        )

    def test_accepts_a_mapping_proxy_for_code_hashes(self) -> None:
        self.assertEqual(
            run_id(self.spec, MappingProxyType(dict(self.code)), self.data),
            run_id(self.spec, self.code, self.data),
        )


if __name__ == "__main__":
    unittest.main()
