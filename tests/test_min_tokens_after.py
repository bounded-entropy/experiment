"""Continuation floors: natural boundaries, exact raw counts and old identities."""

import json
import unittest
from dataclasses import replace

from rlstack import MinTokensAfter, SamplingSpec
from rlstack.runner.remote import decode_sampling, encode_sampling, from_canonical
from rlstack.runner.traffic import ceiled, sampling_row
from rlstack.sampling import ContinuationCounter
from rlstack.spec.canonical import canonical_json


class ContinuationTest(unittest.TestCase):
    def test_natural_boundary_excludes_thought_and_boundary_and_terminal_ids(self):
        count = ContinuationCounter((7, 8), (0, 99))
        count.extend([1] * 4096 + [7])
        self.assertFalse(count.suppresses_terminals(3))
        count.extend([8, 2, 3])
        self.assertTrue(count.suppresses_terminals(3))
        count.extend([4, 99])
        self.assertFalse(count.suppresses_terminals(3))
        self.assertEqual(count.evidence(3), {
            'activated': True, 'boundary_end_output_index': 4098,
            'output_tokens': 4102, 'nonterminal_tokens_after_boundary': 3,
            'terminal_tokens_after_boundary': 1, 'floor_met': True})

    def test_first_overlapping_boundary_across_chunks_and_later_markers_count(self):
        count = ContinuationCounter((1, 1, 2), (0,))
        for token in [1, 1, 1, 2, 1, 1, 2]:
            count.extend([token])
        self.assertEqual(count.boundary_end, 4)
        self.assertEqual(count.nonterminal_tokens, 3)

    def test_independent_requests_and_absent_marker(self):
        a = ContinuationCounter((7,), (0,))
        b = ContinuationCounter((7,), (0,))
        a.extend([7, 8])
        b.extend([8, 8, 0])
        self.assertTrue(a.suppresses_terminals(2))
        self.assertFalse(b.suppresses_terminals(2))
        self.assertFalse(b.evidence(2)['floor_met'])

    def test_unmet_floor_and_no_boundary_remain_distinct(self):
        a = ContinuationCounter((7,), (0,))
        a.extend([1, 7, 2])
        self.assertTrue(a.evidence(2)['activated'])
        self.assertFalse(a.evidence(2)['floor_met'])

    def test_default_canonical_and_wire_identity_are_unchanged(self):
        self.assertEqual(canonical_json(SamplingSpec()),
            '{"__type__":"SamplingSpec","max_tokens":1024,"temperature":1.0,"top_p":1.0}')
        self.assertEqual(encode_sampling(SamplingSpec()),
                         {'temperature': 1., 'top_p': 1., 'max_tokens': 1024})
        self.assertNotIn('min_tokens_after', sampling_row(SamplingSpec(), ()))

    def test_opted_in_identity_roundtrips_and_ceiling_cannot_break_budget(self):
        spec = SamplingSpec(max_tokens=20, min_tokens_after=MinTokensAfter((7, 8), 10))
        self.assertEqual(decode_sampling(encode_sampling(spec)), spec)
        self.assertEqual(from_canonical(json.loads(canonical_json(spec))), spec)
        self.assertIn('min_tokens_after', sampling_row(spec, ()))
        with self.assertRaisesRegex(ValueError, 'accommodate'):
            ceiled(spec, 11)
        self.assertEqual(ceiled(spec, 12).max_tokens, 12)
        self.assertNotEqual(canonical_json(spec), canonical_json(replace(spec, min_tokens_after=None)))

    def test_invalid_options_refused(self):
        for sequence, minimum in [((), 2), ((-1,), 2), ((True,), 2), ((1,), 0), ((1,), True)]:
            with self.assertRaises(ValueError):
                MinTokensAfter(sequence, minimum)

    def test_default_build_recipe_identity_is_unchanged(self):
        from rlstack.runner.residents import EngineBuild, encode_build, decode_build
        build = EngineBuild()
        self.assertNotIn('enable_min_tokens_after', encode_build(build))
        self.assertNotIn('enable_min_tokens_after', canonical_json(build))
        self.assertEqual(decode_build(encode_build(build)), build)
        opted = replace(build, enable_min_tokens_after=True)
        self.assertTrue(encode_build(opted)['enable_min_tokens_after'])
        self.assertEqual(decode_build(encode_build(opted)), opted)
