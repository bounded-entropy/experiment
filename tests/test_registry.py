"""Registries (SPEC.md §2B): typed registrations, declarations, source-hash identity."""

from __future__ import annotations

import unittest
from typing import Any

from rlstack.inference.environments.base import Environment, EnvironmentDef, environment
from rlstack.policy.adapters import AdapterType, AdapterTypeDef, Mechanism, adapter_type
from rlstack.training.post.base import PostDef, PostProcessor, postprocessor
from rlstack.registry import (
    ADAPTER_TYPES,
    ENVS,
    LOSSES,
    POST,
    LossDef,
    Registry,
    code_hashes,
    loss,
    source_hash,
)
from rlstack.spec.specs import (
    AlgoSpec, ExperimentSpec, GenSpec, GpuConfig, HostSpec, OptimSpec,
    PolicySpec, Plans, Schedule, Seeds, learner, lora, pool,
)


class TestRegistry(unittest.TestCase):
    def test_add_get_names_contains(self) -> None:
        reg = Registry("thing")

        def compute() -> None: ...

        reg.add(LossDef("b", compute, (), source_hash(compute)))
        reg.add(LossDef("a", compute, (), source_hash(compute)))
        self.assertEqual(reg.names(), ("a", "b"))
        self.assertIn("a", reg)
        self.assertNotIn("c", reg)
        self.assertEqual(len(reg), 2)
        self.assertIs(reg.get("a").fn, compute)

    def test_duplicate_name_different_source_raises(self) -> None:
        reg = Registry("thing")

        def one() -> int:
            return 1

        def two() -> int:
            return 2

        reg.add(LossDef("name", one, (), source_hash(one)))
        with self.assertRaises(ValueError):
            reg.add(LossDef("name", two, (), source_hash(two)))

    def test_reregistering_identical_source_is_a_no_op(self) -> None:
        reg = Registry("thing")

        def compute() -> None: ...

        reg.add(LossDef("name", compute, (), source_hash(compute)))
        reg.add(LossDef("name", compute, (), source_hash(compute)))  # re-import: fine
        self.assertEqual(len(reg), 1)

    def test_unknown_name_error_lists_valid_names(self) -> None:
        with self.assertRaises(KeyError) as caught:
            LOSSES.get("gorp")
        message = str(caught.exception)
        self.assertIn("gorp", message)
        self.assertIn("grpo", message)  # lists what IS registered

    def test_unknown_name_on_empty_registry(self) -> None:
        with self.assertRaises(KeyError) as caught:
            Registry("empty").get("x")
        self.assertIn("(none)", str(caught.exception))


class TestSourceHash(unittest.TestCase):
    def test_different_bodies_hash_differently(self) -> None:
        def one() -> int:
            return 1

        def two() -> int:
            return 2

        self.assertNotEqual(source_hash(one), source_hash(two))

    def test_hash_is_stable_hex(self) -> None:
        def fn() -> None: ...

        digest = source_hash(fn)
        self.assertEqual(digest, source_hash(fn))
        self.assertEqual(len(digest), 64)
        int(digest, 16)  # is hex

    def test_fallback_when_source_unavailable(self) -> None:
        digest = source_hash(len)  # builtin: no source
        self.assertEqual(len(digest), 64)


class TestDecorators(unittest.TestCase):
    def test_postprocessor_declares_its_wiring(self) -> None:
        @postprocessor("test_dec_post")
        class Scorer(PostProcessor):
            consumes = ("reward",)
            produces = ("score", "length_penalty")

            async def process(self, group: Any, data: Any, client: Any) -> Any:
                raise NotImplementedError

        pdef = POST.get("test_dec_post")
        self.assertIsInstance(pdef, PostDef)
        self.assertEqual(pdef.consumes, ("reward",))
        self.assertEqual(pdef.produces, ("score", "length_penalty"))
        self.assertIs(pdef.cls, Scorer)
        self.assertIsInstance(pdef.instance, Scorer)

    def test_loss_declares_data_only_requires(self) -> None:
        """#38: requires names data columns — never a pass, never metal."""
        @loss("test_dec_loss", requires=("values", "reward"))
        def objective(out: Any, batch: Any) -> Any: ...

        ldef = LOSSES.get("test_dec_loss")
        self.assertIsInstance(ldef, LossDef)
        self.assertEqual(ldef.requires, ("values", "reward"))

    def test_environment_decorator_registers(self) -> None:
        @environment("test_dec_env")
        class MyEnv(Environment):
            async def run(self, client: Any, task: Any) -> Any:
                raise NotImplementedError

        edef = ENVS.get("test_dec_env")
        self.assertIsInstance(edef, EnvironmentDef)
        self.assertIsInstance(edef.instance, MyEnv)

    def test_adapter_registers_class_and_instance(self) -> None:
        @adapter_type("test_dec_adapter")
        class MyAdapter(AdapterType):
            serving = None

        adef = ADAPTER_TYPES.get("test_dec_adapter")
        self.assertIsInstance(adef, AdapterTypeDef)
        self.assertIs(adef.cls, MyAdapter)
        self.assertIsInstance(adef.instance, MyAdapter)


class TestAdapters(unittest.TestCase):
    def test_base_defaults(self) -> None:
        base = AdapterType()
        self.assertIsNone(base.engine_plugin)
        self.assertIsNone(base.serving)
        self.assertEqual(base.provides, frozenset())
        self.assertEqual(base.records, ())
        self.assertTrue(base.site_ok(None))  # type: ignore[arg-type]

    def test_compute_halves_are_not_implemented_yet(self) -> None:
        plain = AdapterType()
        for call in (lambda: plain.params((), {}),
                     lambda: plain.install_replay(None, None, ()),
                     lambda: plain.emit(None),
                     lambda: plain.parity(None)):
            with self.assertRaises(NotImplementedError):
                call()

    def test_builtin_serving_surfaces(self) -> None:
        self.assertEqual(ADAPTER_TYPES.get("lora").instance.serving, Mechanism.PUNICA)
        self.assertEqual(ADAPTER_TYPES.get("soft_prompt").instance.serving,
                         Mechanism.PROMPT_EMBEDS)
        self.assertEqual(ADAPTER_TYPES.get("attn_bias").instance.serving,
                         Mechanism.SIDE_ATTENTION)
        self.assertIsNone(ADAPTER_TYPES.get("value_head").instance.serving)

    def test_every_registered_adapter_type_uses_the_closed_serving_vocabulary(self) -> None:
        for name in ADAPTER_TYPES.names():
            serving = ADAPTER_TYPES.get(name).instance.serving
            self.assertTrue(serving is None or isinstance(serving, Mechanism),
                            f"{name}: serving {serving!r} is not a Mechanism")

    def test_only_attn_bias_ships_engine_code(self) -> None:
        with_plugin = [name for name in ("lora", "soft_prompt", "attn_bias", "value_head")
                       if ADAPTER_TYPES.get(name).instance.engine_plugin is not None]
        self.assertEqual(with_plugin, ["attn_bias"])

    def test_value_head_provides_values(self) -> None:
        self.assertEqual(ADAPTER_TYPES.get("value_head").instance.provides, {"values"})

    def test_site_predicates(self) -> None:
        from rlstack.policy.siteschema import fake_qwen_schema
        from rlstack.spec.specs import soft_prompt

        schema = fake_qwen_schema(2, base="Qwen/Qwen3-1.7B")
        weighted = schema.resolve("layers.0.self_attn.q_proj")[0]
        boundary = schema.resolve("final_hidden")[0]
        # prompt[:8] is not a base site: the soft prompt entry exports it
        virtual = ADAPTER_TYPES.get("soft_prompt").instance.exports(
            soft_prompt("prompt[:8]", n=8, d=64))[0]

        self.assertTrue(ADAPTER_TYPES.get("lora").instance.site_ok(weighted))
        self.assertFalse(ADAPTER_TYPES.get("lora").instance.site_ok(boundary))
        self.assertTrue(ADAPTER_TYPES.get("soft_prompt").instance.site_ok(virtual))
        self.assertFalse(ADAPTER_TYPES.get("soft_prompt").instance.site_ok(weighted))
        self.assertTrue(ADAPTER_TYPES.get("value_head").instance.site_ok(boundary))
        self.assertFalse(ADAPTER_TYPES.get("value_head").instance.site_ok(weighted))


def minimal_spec(**overrides: Any) -> ExperimentSpec:
    """A tiny valid spec over the builtins, for identity tests."""
    fields: dict[str, Any] = dict(
        policy=PolicySpec(base="Qwen/Qwen3-1.7B",
                          bank={"pi": lora("layers.*.mlp.*", r=16)}),
        gen=GenSpec(envs=("noop_env",), tasks=("cas://x/train.jsonl",)),
        plans=Plans(train="cas://plan/train", rollout="cas://plan/roll"),
        algo=AlgoSpec(loss="grpo", post=("verifier", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=1e-5),
                      schedule=Schedule()),
        gpu_config=GpuConfig(hosts=(HostSpec((pool("main"),)),
                                    HostSpec((learner(),)))),
        seeds=Seeds(master=0),
    )
    fields.update(overrides)
    return ExperimentSpec(**fields)


class TestCodeHashes(unittest.TestCase):
    def test_covers_every_referenced_name(self) -> None:
        hashes = code_hashes(minimal_spec())
        self.assertEqual(
            set(hashes),
            {"loss:grpo", "postprocessor:verifier",
             "postprocessor:grpo_advantage", "environment:noop_env",
             "adapter_type:lora"},
        )
        for digest in hashes.values():
            self.assertEqual(len(digest), 64)

    def test_offline_run_without_gen_or_algo(self) -> None:
        spec = minimal_spec(gen=None, algo=None,
                            plans=Plans(train="cas://plan/train"))
        self.assertEqual(set(code_hashes(spec)), {"adapter_type:lora"})

    def test_measurement_names_are_not_identity(self) -> None:
        """#70: a measurement's post pipeline is an observation's config,
        never the run's — nothing eval-shaped reaches the code hashes."""
        spec = minimal_spec(plans=Plans(train="cas://p/t"))
        self.assertNotIn("postprocessor:constant", code_hashes(spec))

    def test_unknown_name_raises_helpful_keyerror(self) -> None:
        spec = minimal_spec(algo=AlgoSpec(
            loss="not_registered", post=("verifier", "grpo_advantage"),
            optim=OptimSpec("adamw", lr=1e-5),
            schedule=Schedule()))
        with self.assertRaises(KeyError) as caught:
            code_hashes(spec)
        self.assertIn("not_registered", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
