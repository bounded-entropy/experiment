"""CPU execution of the public vLLM batch-processor contract.

Requires torch and vLLM; no engine, model, CUDA allocation or GPU call.
"""

import unittest
from types import SimpleNamespace

try:
    import torch
    from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality
    from rlstack_engine.min_tokens_after import MinTokensAfterProcessor
except ImportError:
    torch = None

from rlstack.sampling import CONTINUATION_KEY


@unittest.skipIf(torch is None, 'requires CPU torch and the vLLM processor interface')
class ProcessorTest(unittest.TestCase):
    def params(self, **overrides):
        defaults = dict(extra_args={CONTINUATION_KEY: {
            'sequence': [7, 8], 'min_tokens': 2, 'terminal_token_ids': [0, 9]}},
            all_stop_token_ids={0, 9}, max_tokens=10, stop=[],
            repetition_detection=None, ignore_eos=False, min_tokens=0,
            thinking_token_budget=None, allowed_token_ids=None, structured_outputs=None)
        return SimpleNamespace(**(defaults | overrides))

    def processor(self):
        return MinTokensAfterProcessor(SimpleNamespace(speculative_config=None),
                                       torch.device('cpu'), False)

    def test_masks_all_terminals_only_after_first_complete_boundary(self):
        proc = self.processor()
        output = [1, 7]
        proc.update_state(BatchUpdate(1, [], [(0, self.params(), [7, 8], output)], []))
        original = torch.arange(10.).reshape(1, 10)
        self.assertTrue(torch.equal(proc.apply(original.clone()), original))
        output.extend([8, 4])
        masked = proc.apply(original.clone())
        self.assertTrue(torch.isneginf(masked[0, [0, 9]]).all())
        self.assertTrue(torch.equal(masked[0, 1:9], original[0, 1:9]))
        output.append(5)
        self.assertTrue(torch.equal(proc.apply(original.clone()), original))
        # Repeated apply with no new tokens cannot advance the count.
        self.assertEqual(proc.requests[0].counter.nonterminal_tokens, 2)

    def test_request_moves_swaps_removal_and_unconstrained_replacement(self):
        proc = self.processor()
        a, b = [7, 8], [1]
        proc.update_state(BatchUpdate(2, [], [(0, self.params(), [], a),
                                             (1, self.params(), [], b)], []))
        proc.update_state(BatchUpdate(2, [], [], [(0, 1, MoveDirectionality.SWAP)]))
        logits = proc.apply(torch.zeros(2, 10))
        self.assertTrue(torch.isfinite(logits[0]).all())
        self.assertTrue(torch.isneginf(logits[1, [0, 9]]).all())
        proc.update_state(BatchUpdate(1, [0], [], [(1, 0, MoveDirectionality.UNIDIRECTIONAL)]))
        self.assertEqual(set(proc.requests), {0})
        proc.update_state(BatchUpdate(1, [], [(0, self.params(extra_args={}), [], [])], []))
        self.assertEqual(proc.requests, {})
        self.assertTrue(torch.equal(proc.apply(torch.zeros(1, 10)), torch.zeros(1, 10)))

    def test_contradictory_stops_budgets_and_grammars_refused(self):
        for override in [dict(stop=['END']), dict(repetition_detection=object()),
                         dict(ignore_eos=True), dict(min_tokens=2), dict(max_tokens=3),
                         dict(thinking_token_budget=10), dict(allowed_token_ids=[0]),
                         dict(structured_outputs=object())]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                MinTokensAfterProcessor.validate_params(self.params(**override))
        proc = self.processor()
        with self.assertRaisesRegex(ValueError, 'terminal IDs changed'):
            proc.update_state(BatchUpdate(1, [], [(0, self.params(all_stop_token_ids={0}), [], [])], []))
        with self.assertRaisesRegex(ValueError, 'speculative'):
            MinTokensAfterProcessor(SimpleNamespace(speculative_config=object()), torch.device('cpu'), False)

    def test_default_request_never_masks_and_prompt_does_not_activate(self):
        proc = self.processor()
        proc.update_state(BatchUpdate(1, [], [(0, self.params(extra_args={}), [7, 8], [7, 8])], []))
        self.assertFalse(proc.requests)
        self.assertFalse(proc.is_argmax_invariant())
        self.assertTrue(torch.isfinite(proc.apply(torch.zeros(1, 10))).all())
