"""Task posterior routing, checkpoint ownership, and sequence-loss packing."""

import unittest

try:
    import torch
except ImportError:
    torch = None




@unittest.skipUnless(torch is not None, "torch required")
class SequenceLossMeasurementsTest(unittest.TestCase):
    def test_unequal_document_lengths_reconstruct_nll_objective_and_kl(self):
        from rlstack.data.flatten import TokenBatch
        from rlstack.training.losses.base import PolicyOutputs
        from rlstack.training.losses.sequence_sft import sequence_sft
        batch = TokenBatch((0, 1, 2, 3, 4), (0, 1, 1, 0, 1), (0.,)*5, (-1, 0, 0, -1, 0), (0, 3))
        lp = torch.tensor([0., -1., -3., 0., -5.], requires_grad=True)
        result = sequence_sft(PolicyOutputs(lp), batch)
        c = result.components
        self.assertEqual((c["nll_sum"], c["scored_tokens"]), (9., 3.))
        reconstructed = sum(c[f"document_{i}_nll_sum"] / c[f"document_{i}_tokens"]
                            for i in range(2)) / c["documents_in_update"]
        self.assertEqual(reconstructed, 3.5)
        self.assertEqual(c["objective"], reconstructed)
        result.loss.backward()
        self.assertTrue(torch.equal(lp.grad, torch.tensor([0., -.25, -.25, 0., -.5])))
