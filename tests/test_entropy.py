"""Exact entropy values, derivatives, masking and shared learner opt-in."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from rlstack.data.flatten import TokenBatch
from rlstack.registry import LOSSES, loss
from rlstack.training.losses import LossResult, PolicyOutputs, entropy_regularization

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from rlstack.runner.learners.torch_learner import (
        TorchLearner, _Tenant, _doc_spans, policy_token_statistics,
    )
    from test_batched_replay import _ToyHF, _ToyLM


def batch_of(documents, masks, documents_in_update=0):
    starts, offset = [], 0
    for doc in documents:
        starts.append(offset)
        offset += len(doc)
    return TokenBatch(tuple(t for doc in documents for t in doc),
                      tuple(t for mask in masks for t in mask),
                      (0.,) * offset, (0,) * offset, tuple(starts),
                      documents_in_update=documents_in_update)


class DeclarationTest(unittest.TestCase):
    def test_exact_entropy_is_an_explicit_forward_requirement(self):
        @loss("test_entropy_forward_default")
        def ordinary(out, batch):
            pass

        @loss("test_entropy_forward_requested", needs_entropy=True)
        def requested(out, batch):
            pass

        self.assertFalse(LOSSES.get("test_entropy_forward_default").needs_entropy)
        self.assertTrue(LOSSES.get("test_entropy_forward_requested").needs_entropy)
        self.assertEqual(LOSSES.get("test_entropy_forward_requested").requires, ())
        with self.assertRaises(TypeError):
            loss("test_entropy_invalid", needs_entropy="yes")


@unittest.skipUnless(torch is not None, "torch is a deferred learner dependency")
class ExactEntropyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_dense_values_and_gradients_match_across_chunks(self):
        logits = torch.randn(2, 5, 11, requires_grad=True)
        targets = torch.randint(11, (2, 5))
        logp = logits.log_softmax(-1)
        expected_lp = logp.gather(-1, targets[..., None])[..., 0]
        expected_h = -(logp.exp() * logp).sum(-1)
        expected_grad, = torch.autograd.grad(
            (expected_lp + .17 * expected_h).sum(), logits, retain_graph=True)
        for chunk in (1, 3, 20):
            lp, entropy = policy_token_statistics(logits, targets, chunk=chunk)
            self.assertTrue(torch.allclose(lp, expected_lp, atol=1e-6))
            self.assertTrue(torch.allclose(entropy, expected_h, atol=1e-6))
            gradient, = torch.autograd.grad((lp + .17 * entropy).sum(), logits)
            self.assertTrue(torch.allclose(gradient, expected_grad, atol=1e-6))

    def test_hidden_projection_matches_logits_values_and_gradients(self):
        hidden = torch.randn(2, 5, 7, requires_grad=True)
        head = torch.nn.Linear(7, 11)
        targets = torch.randint(11, (2, 5))
        expected = policy_token_statistics(head(hidden), targets, chunk=2)
        actual = policy_token_statistics(hidden, targets, head=head, chunk=3)
        expected_grad = torch.autograd.grad(sum(t.sum() for t in expected),
                                            (hidden, head.weight, head.bias))
        actual_grad = torch.autograd.grad(sum(t.sum() for t in actual),
                                          (hidden, head.weight, head.bias))
        for a, b in zip(actual, expected):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
        for a, b in zip(actual_grad, expected_grad):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_zero_positions_and_uniform_and_peaked_distributions(self):
        values = torch.empty(2, 0, 11)
        lp, entropy = policy_token_statistics(values, torch.empty(2, 0, dtype=torch.long))
        self.assertEqual(tuple(entropy.shape), (2, 0))
        logits = torch.tensor([[[0., 0., 0.], [1000., -1000., -1000.]]])
        _, entropy = policy_token_statistics(logits, torch.tensor([[0, 0]]))
        self.assertAlmostEqual(float(entropy[0, 0]), float(torch.tensor(3.).log()), places=6)
        self.assertEqual(float(entropy[0, 1]), 0.)

    def test_padded_prefixes_and_head_path_match_independent_documents(self):
        documents = [[2, 3, 5, 7, 11], [3, 2, 6], [9]]
        batch = batch_of(documents, [[0, 0, 1, 1, 1], [0, 0, 1], [0]])
        learner = TorchLearner(device="cpu", dtype=torch.float32)
        hf = _ToyHF()
        learner._model = hf
        out = learner._batched_policy_outputs(batch, _doc_spans(batch), entropy=True)
        baseline = learner._batched_logprobs(batch, _doc_spans(batch))
        self.assertTrue(torch.allclose(out.logprobs, baseline, atol=1e-6))
        dense = []
        for doc in documents:
            logits = hf(torch.tensor([doc])).logits[0, :-1]
            logp = logits.log_softmax(-1)
            dense.append(torch.cat((torch.zeros(1), -(logp.exp() * logp).sum(-1))))
        mask = torch.tensor(batch.loss_mask, dtype=torch.bool)
        self.assertTrue(torch.allclose(out.entropies[mask], torch.cat(dense)[mask], atol=1e-6))
        self.assertEqual(float(out.entropies[-1]), 0.)
        learner._model = hf.toy
        logits_out = learner._batched_policy_outputs(batch, _doc_spans(batch), entropy=True)
        self.assertTrue(torch.allclose(out.entropies, logits_out.entropies, atol=1e-6))

    def test_regularization_reconstructs_objective_and_is_packing_invariant(self):
        batch = batch_of([[1, 2, 3], [4, 5]], [[0, 1, 1], [0, 1]])
        entropy = torch.tensor([99., 1., 3., 99., 7.], requires_grad=True)
        out = PolicyOutputs(torch.zeros(5), entropies=entropy)
        term, stats = entropy_regularization(out, batch, .2)
        self.assertAlmostEqual(float(term.detach()), -.9, places=6)
        self.assertEqual(stats["entropy_sum"], 11.)
        self.assertEqual(stats["entropy_scored_tokens"], 3.)
        self.assertEqual(stats["entropy_coefficient"], .2)
        self.assertEqual(stats["entropy_document_mean_contribution"], 4.5)
        pieces = []
        for a, b, doc, mask in [(0, 3, [1, 2, 3], [0, 1, 1]), (3, 5, [4, 5], [0, 1])]:
            sub = batch_of([doc], [mask], documents_in_update=2)
            piece, _ = entropy_regularization(
                PolicyOutputs(torch.zeros(b-a), entropies=entropy[a:b]), sub, .2)
            pieces.append(piece)
        self.assertTrue(torch.allclose(term, sum(pieces)))
        packed_grad, = torch.autograd.grad(term, entropy, retain_graph=True)
        split_grad, = torch.autograd.grad(sum(pieces), entropy)
        self.assertTrue(torch.equal(packed_grad, split_grad))
        self.assertTrue(torch.allclose(packed_grad, torch.tensor([0., -.05, -.05, 0., -.1])))

    def test_unscored_documents_and_zero_coefficient_keep_honest_statistics(self):
        batch = batch_of([[1], [2, 3]], [[0], [0, 0]])
        entropy = torch.tensor([0., 0., 1.], requires_grad=True)
        out = PolicyOutputs(torch.zeros(3), entropies=entropy)
        term, stats = entropy_regularization(out, batch, .1)
        self.assertEqual(float(term.detach()), 0.)
        self.assertEqual(stats["entropy_scored_tokens"], 0.)
        self.assertEqual(stats["entropy_sum"], 0.)
        self.assertTrue(torch.equal(torch.autograd.grad(term, entropy)[0], torch.zeros(3)))
        scored = batch_of([[1, 2]], [[0, 1]])
        term, stats = entropy_regularization(
            PolicyOutputs(torch.zeros(2), entropies=torch.tensor([0., 2.])), scored, 0.)
        self.assertEqual(float(term), 0.)
        self.assertEqual(stats["entropy_sum"], 2.)
        self.assertEqual(stats["entropy_scored_tokens"], 1.)

    def test_bonus_gradient_increases_entropy_and_missing_entropy_is_refused(self):
        batch = batch_of([[0, 1]], [[0, 1]])
        logits = torch.tensor([[[2., 0., -1.]]], requires_grad=True)
        _, entropy = policy_token_statistics(logits, torch.tensor([[1]]))
        out = PolicyOutputs(torch.zeros(2), entropies=torch.cat((torch.zeros(1), entropy[0])))
        term, _ = entropy_regularization(out, batch, .1)
        gradient, = torch.autograd.grad(term, logits)
        _, after = policy_token_statistics(logits - gradient, torch.tensor([[1]]))
        self.assertGreater(float(after.detach()), float(entropy.detach()))
        with self.assertRaises(ValueError):
            entropy_regularization(PolicyOutputs(torch.zeros(2)), batch, .1)
        for invalid in (-1., float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                entropy_regularization(out, batch, invalid)

    def test_forward_backward_only_computes_entropy_when_declared(self):
        batch = batch_of([[1, 2, 3]], [[0, 1, 1]])
        learner = TorchLearner(device="cpu", dtype=torch.float32)
        learner._model = _ToyLM()
        seen = []
        def objective(out, batch):
            seen.append(out.entropies)
            return LossResult(out.logprobs.sum() * 0, 1., 0.)
        state = _Tenant(objective, [], [])
        learner._tenants["test"] = state
        with patch.object(learner, "_batched_logprobs", wraps=learner._batched_logprobs) as old:
            learner.forward_backward("test", batch)
            old.assert_called_once()
        self.assertIsNone(seen[-1])
        state.needs_entropy = True
        with patch.object(learner, "_batched_logprobs", side_effect=AssertionError("old path")):
            learner.forward_backward("test", batch)
        self.assertEqual(tuple(seen[-1].shape), (3,))
        self.assertTrue(seen[-1].requires_grad)


if __name__ == "__main__":
    unittest.main()
