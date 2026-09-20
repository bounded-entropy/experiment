"""The site grammar's alternation (ADR 0019): a pattern SEGMENT may be
`a|b|c`, a list of literal leaf names — so `q_proj` + `v_proj` is one site
pattern, and one bank entry — and no pattern written before it changes."""

from __future__ import annotations

import unittest

from rlstack.policy.siteschema import (
    fake_qwen_schema, pattern_matches, resolve, segment_matches,
)

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def names(pattern: str) -> list[str]:
    return [site.name for site in SCHEMA.resolve(pattern)]


class TestAlternation(unittest.TestCase):
    def test_q_and_v_are_one_pattern(self) -> None:
        self.assertEqual(
            names("layers.*.self_attn.q_proj|v_proj"),
            [f"layers.{n}.self_attn.{proj}" for n in range(4) for proj in ("q_proj", "v_proj")])

    def test_an_alternation_composes_with_a_range_in_another_segment(self) -> None:
        self.assertEqual(names("layers.1-2.mlp.gate_proj|down_proj"),
                         ["layers.1.mlp.gate_proj", "layers.1.mlp.down_proj",
                          "layers.2.mlp.gate_proj", "layers.2.mlp.down_proj"])

    def test_an_alternation_may_sit_in_any_segment(self) -> None:
        self.assertEqual(names("layers.0.self_attn|mlp.*"),
                         [site.name for site in SCHEMA.sites
                          if site.name.startswith(("layers.0.self_attn.", "layers.0.mlp."))])
        self.assertEqual(names("layers.0|3.self_attn.o_proj"),
                         ["layers.0.self_attn.o_proj", "layers.3.self_attn.o_proj"])

    def test_alternatives_are_literal_names_not_patterns(self) -> None:
        self.assertTrue(segment_matches("q_proj|v_proj", "v_proj"))
        self.assertFalse(segment_matches("q_proj|v_proj", "k_proj"))
        self.assertFalse(segment_matches("q_*|v_proj", "q_proj"))      # no wildcard inside
        self.assertFalse(segment_matches("0-1|3", "1"))                # no range inside
        self.assertFalse(segment_matches("q_proj|v_proj", "q_proj|v_proj"))

    def test_an_alternative_no_site_has_matches_nothing(self) -> None:
        self.assertEqual(names("layers.0.self_attn.q_proj|z_proj"), ["layers.0.self_attn.q_proj"])
        self.assertEqual(names("layers.0.self_attn.y_proj|z_proj"), [])

    def test_segment_count_still_must_agree(self) -> None:
        self.assertFalse(pattern_matches("layers.*.q_proj|v_proj", "layers.0.self_attn.q_proj"))

    def test_every_earlier_pattern_means_what_it_meant(self) -> None:
        self.assertEqual(len(names("layers.*.self_attn.*")), 16)
        self.assertEqual(len(names("layers.0-1.self_attn.q_proj")), 2)
        self.assertEqual(names("layers.2.mlp.up_proj"), ["layers.2.mlp.up_proj"])
        self.assertEqual(names("final_hidden"), ["final_hidden"])
        self.assertEqual(names("resid_pre.1-2"), ["resid_pre.1", "resid_pre.2"])

    def test_a_non_module_name_with_a_bar_still_matches_exactly_only(self) -> None:
        self.assertEqual(resolve(SCHEMA.sites, "logits|final_hidden"), ())


if __name__ == "__main__":
    unittest.main()
