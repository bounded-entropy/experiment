"""Independent SFT denominators survive uneven documents and packed forwards."""
import unittest

from rlstack.data.flatten import TokenBatch
from rlstack.training.losses import PolicyOutputs
from rlstack.training.losses.independent_sft import independent_sft

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU torch required for exact loss and gradient checks")
class IndependentSftTest(unittest.TestCase):
    def test_unequal_students_and_documents_match_separate_objectives(self):
        scores = torch.tensor([0., -.2, -.4, 0., -.6, 0., -.8, -1., -1.2], requires_grad=True)
        # Student0 has two documents of lengths 3 and2; Student1 has one oflength4.
        batch = TokenBatch(token_ids=tuple(range(9)), loss_mask=(0, 1, 1, 0, 1, 0, 1, 1, 1),
            behavior_logprobs=(0.,) * 9, segment_ids=(0,) * 9, doc_starts=(0, 3, 5),
            postdata={"sft_document_weight": (.5,) * 5 + (1.,) * 4}, documents_in_update=3)
        result = independent_sft(PolicyOutputs(logprobs=scores), batch)
        expected = (-scores[1:3].mean() - scores[4]) / 2 - scores[6:9].mean()
        actual_gradient, = torch.autograd.grad(result.loss, scores, retain_graph=True)
        expected_gradient, = torch.autograd.grad(expected, scores)
        torch.testing.assert_close(result.loss, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertEqual(result.components["document_0_tokens"], 2)
        self.assertEqual(result.components["document_1_weight"], .5)
        self.assertEqual(result.components["document_2_weight"], 1.)

    def test_microbatch_partition_does_not_change_objective_or_gradients(self):
        scores = torch.tensor([0., -.2, -.4, 0., -.6], requires_grad=True)
        first = TokenBatch(token_ids=(1, 2, 3), loss_mask=(0, 1, 1),
            behavior_logprobs=(0.,) * 3, segment_ids=(0,) * 3, doc_starts=(0,),
            postdata={"sft_document_weight": (.5,) * 3}, documents_in_update=2)
        second = TokenBatch(token_ids=(4, 5), loss_mask=(0, 1),
            behavior_logprobs=(0.,) * 2, segment_ids=(0,) * 2, doc_starts=(0,),
            postdata={"sft_document_weight": (.5,) * 2}, documents_in_update=2)
        actual = (independent_sft(PolicyOutputs(logprobs=scores[:3]), first).loss
                  + independent_sft(PolicyOutputs(logprobs=scores[3:]), second).loss)
        expected = (-scores[1:3].mean() - scores[4]) / 2
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, scores, retain_graph=True)[0],
                                   torch.autograd.grad(expected, scores)[0])
