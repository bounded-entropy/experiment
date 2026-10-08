"""SiteSchema: frozen data + the one canonical grammar + the fingerprint."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from rlstack.policy.siteschema import (
    SiteMeta,
    SiteSchema,
    fake_qwen_schema,
    resolve,
)


class TestLlamaFamilyCompiler(unittest.TestCase):
    """The real compiler's pure half: a Llama-family config compiles; a
    config from another family is refused by name (found on metal: Pythia's
    GPT-NeoX config has no num_key_value_heads and died two calls deep)."""

    def test_a_llama_family_config_compiles_the_full_catalog(self) -> None:
        from types import SimpleNamespace

        from rlstack.policy.siteschema import llama_family_schema

        config = SimpleNamespace(hidden_size=64, num_attention_heads=4,
                                 num_key_value_heads=2, intermediate_size=128,
                                 num_hidden_layers=2, head_dim=16)
        schema = llama_family_schema("toy/llama", config)
        self.assertEqual(len(schema.sites), 2 * 8 + 2)
        (k,) = schema.resolve("layers.1.self_attn.k_proj")
        self.assertEqual(k.shape, (64, 32))          # GQA: 2 kv heads x 16

    def test_another_family_is_refused_by_name(self) -> None:
        from types import SimpleNamespace

        from rlstack.policy.siteschema import llama_family_schema

        neox = SimpleNamespace(model_type="gpt_neox", hidden_size=2048,
                               num_attention_heads=8, intermediate_size=8192,
                               num_hidden_layers=16)
        with self.assertRaises(ValueError) as refused:
            llama_family_schema("EleutherAI/pythia-1b", neox)
        self.assertIn("num_key_value_heads", str(refused.exception))
        self.assertIn("Llama family", str(refused.exception))
        self.assertIn("gpt_neox", str(refused.exception))


class TestQwen3_5Compiler(unittest.TestCase):
    """Qwen3.5's hybrid decoder (post 0010): the config nests the decoder
    under `text_config`, three of four layers are gated-DeltaNet
    `linear_attn`, the fourth gated full attention whose query is twice the
    head width (the output gate). Shapes are Qwen3.5-9B's, read off the
    module tree on the meta device."""

    def config(self):
        from types import SimpleNamespace

        text = SimpleNamespace(
            model_type="qwen3_5_text", hidden_size=4096, intermediate_size=12288,
            num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
            num_attention_heads=16, num_key_value_heads=4, head_dim=256,
            attn_output_gate=True, linear_num_key_heads=16, linear_num_value_heads=32,
            linear_key_head_dim=128, linear_value_head_dim=128)
        return SimpleNamespace(model_type="qwen3_5", text_config=text)

    def test_the_wrapper_config_is_recognised(self) -> None:
        from rlstack.policy.siteschema import is_qwen3_5

        self.assertTrue(is_qwen3_5(self.config()))
        self.assertFalse(is_qwen3_5(SimpleNamespace(model_type="qwen3", hidden_size=8)))

    def test_each_layer_kind_carries_its_own_projections_at_9b_widths(self) -> None:
        from rlstack.policy.siteschema import qwen3_5_schema

        schema = qwen3_5_schema("Qwen/Qwen3.5-9B", self.config().text_config)
        self.assertEqual(len(schema.sites), 3 * (5 + 3 + 1) + (4 + 3 + 1) + 2)
        (qkv,) = schema.resolve("layers.0.linear_attn.in_proj_qkv")
        self.assertEqual(qkv.shape, (4096, 8192))
        self.assertEqual(qkv.path, "model.layers.0.linear_attn.in_proj_qkv")
        (q,) = schema.resolve("layers.3.self_attn.q_proj")
        self.assertEqual(q.shape, (4096, 8192))          # 16 heads x 256, doubled by the gate
        (k,) = schema.resolve("layers.3.self_attn.k_proj")
        self.assertEqual(k.shape, (4096, 1024))
        self.assertEqual(schema.resolve("layers.0.self_attn.q_proj"), ())
        self.assertEqual(len(schema.resolve("layers.*.mlp.gate_proj|up_proj|down_proj")), 12)

    def test_a_missing_field_is_refused_by_name(self) -> None:
        from rlstack.policy.siteschema import qwen3_5_schema

        text = self.config().text_config
        del text.layer_types
        with self.assertRaises(ValueError) as refused:
            qwen3_5_schema("Qwen/Qwen3.5-9B", text)
        self.assertIn("layer_types", str(refused.exception))


class TestFakeQwenSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")

    def test_site_inventory(self) -> None:
        names = [s.name for s in self.schema.sites]
        # 4 layers x (4 attn + 3 mlp + 1 resid_pre) + 2 non-module sites.
        # prompt[:n] and its rectangle are NOT here: a soft prompt exports them.
        self.assertEqual(len(names), 4 * 8 + 2)
        self.assertEqual(len(set(names)), len(names))
        for name in ("logits", "final_hidden"):
            self.assertIn(name, names)
        self.assertNotIn("prompt[:8]", names)
        self.assertNotIn("queries -> prompt[:8]", names)

    def test_the_schema_names_its_base(self) -> None:
        self.assertEqual(self.schema.base, "Qwen/Qwen3-0.6B")

    def test_weighted_sites_carry_shape_and_path(self) -> None:
        (meta,) = self.schema.resolve("layers.2.mlp.up_proj")
        self.assertTrue(meta.has_weight)
        self.assertEqual(meta.shape, (64, 64))
        self.assertFalse(meta.is_boundary)
        self.assertEqual(meta.path, "model.layers.2.mlp.up_proj")

    def test_boundary_sites_are_unweighted(self) -> None:
        (meta,) = self.schema.resolve("resid_pre.0")
        self.assertTrue(meta.is_boundary)
        self.assertFalse(meta.has_weight)
        self.assertIsNone(meta.shape)

    # --- grammar ---------------------------------------------------------

    def test_exact_match(self) -> None:
        got = self.schema.resolve("layers.0.self_attn.q_proj")
        self.assertEqual([m.name for m in got], ["layers.0.self_attn.q_proj"])

    def test_wildcard_segment(self) -> None:
        got = self.schema.resolve("layers.1.self_attn.*")
        self.assertEqual(
            sorted(m.name for m in got),
            [
                "layers.1.self_attn.k_proj",
                "layers.1.self_attn.o_proj",
                "layers.1.self_attn.q_proj",
                "layers.1.self_attn.v_proj",
            ],
        )

    def test_wildcard_across_layers(self) -> None:
        got = self.schema.resolve("layers.*.mlp.*")
        self.assertEqual(len(got), 4 * 3)
        self.assertTrue(all(".mlp." in m.name for m in got))

    def test_numeric_range_segment(self) -> None:
        got = self.schema.resolve("layers.0-1.self_attn.*")
        self.assertEqual(len(got), 8)
        self.assertEqual({m.name.split(".")[1] for m in got}, {"0", "1"})

    def test_numeric_range_clips_to_existing_layers(self) -> None:
        got = self.schema.resolve("layers.2-99.mlp.down_proj")
        self.assertEqual(
            sorted(m.name for m in got),
            ["layers.2.mlp.down_proj", "layers.3.mlp.down_proj"],
        )

    def test_numeric_range_excludes_out_of_range(self) -> None:
        self.assertEqual(self.schema.resolve("layers.10-15.self_attn.*"), ())

    def test_segment_count_must_match(self) -> None:
        self.assertEqual(self.schema.resolve("layers.*"), ())
        self.assertEqual(self.schema.resolve("layers.0.self_attn"), ())

    def test_no_match_returns_empty_tuple(self) -> None:
        self.assertEqual(self.schema.resolve("blocks.0.attn.q"), ())
        self.assertEqual(self.schema.resolve("layers.99.mlp.up_proj"), ())

    def test_non_module_names_match_exactly_only(self) -> None:
        self.assertEqual(len(self.schema.resolve("final_hidden")), 1)
        self.assertEqual(len(self.schema.resolve("logits")), 1)
        # wildcards never reach them
        self.assertEqual(self.schema.resolve("*"), ())
        self.assertEqual(self.schema.resolve("final_*"), ())

    def test_n_layers_is_honoured(self) -> None:
        small = fake_qwen_schema(1, base="Qwen/Qwen3-0.6B")
        self.assertEqual(len(small.sites), 8 + 2)
        self.assertEqual(small.resolve("layers.1.mlp.up_proj"), ())


class TestResolveIsTheOneResolver(unittest.TestCase):
    """The grammar is canon: the module-level resolve() is what validate runs
    over schema ∪ exports; the method delegates to it."""

    def test_method_and_function_agree(self) -> None:
        schema = fake_qwen_schema(2, base="b")
        pattern = "layers.*.self_attn.q_proj"
        self.assertEqual(schema.resolve(pattern), resolve(schema.sites, pattern))

    def test_resolve_over_a_union(self) -> None:
        exported = SiteMeta(name="prompt[:4]", path="model.embed_tokens",
                            has_weight=False, shape=None, is_boundary=True)
        schema = fake_qwen_schema(1, base="b")
        space = schema.sites + (exported,)
        self.assertEqual(resolve(space, "prompt[:4]"), (exported,))
        self.assertEqual(schema.resolve("prompt[:4]"), ())


class TestFingerprint(unittest.TestCase):
    def test_same_sites_same_fingerprint(self) -> None:
        a = fake_qwen_schema(2, base="b")
        b = fake_qwen_schema(2, base="b")
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_different_catalog_different_fingerprint(self) -> None:
        self.assertNotEqual(fake_qwen_schema(2, base="b").fingerprint(),
                            fake_qwen_schema(3, base="b").fingerprint())


class TestExplicitConstruction(unittest.TestCase):
    def test_hand_built_catalog(self) -> None:
        meta = SiteMeta(name="blocks.0.attn.q", path="transformer.h.0.attn.q",
                        has_weight=True, shape=(8, 8), is_boundary=False)
        schema = SiteSchema(base="gpt2ish", sites=(meta,))
        self.assertEqual(schema.resolve("blocks.*.attn.q"), (meta,))
        self.assertEqual(schema.resolve("blocks.0.attn.k"), ())

    def test_empty_schema_resolves_to_nothing(self) -> None:
        schema = SiteSchema(base="empty", sites=())
        self.assertEqual(schema.resolve("anything.at.all.here"), ())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
