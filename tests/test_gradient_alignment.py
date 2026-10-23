"""ADR 0026 on a CPU causal LM, using direct autograd as the independent oracle.

No model download or GPU is needed. Unlike the fake wire tests, these check the
actual learner gradient, document reduction, and read-only state boundary.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import rlstack  # registers losses and adapter types
from rlstack.data.flatten import Flat, TokenBatch, pack
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.interfaces import EntryInstall, OptimSettings, Parameterization

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from rlstack.policy.adapters.replay import ReplayRows, row_plan
    from rlstack.runner.learners.torch_learner import TorchLearner

WIDTH, VOCAB = 6, 13
SITE = SiteMeta(name="proj", path="proj", has_weight=True,
                shape=(WIDTH, WIDTH), is_boundary=False)
REFERENCES = (((1, 2, 3), (0, 0, 1)),
              ((4, 5, 6, 7, 8, 9), (0, 1, 1, 1, 1, 1)))
CANDIDATES = ((((2, 3, 4, 5), (0, 1, 1, 1)),),
              (((9, 8, 7, 6, 5), (0, 1, 1, 1, 1)),))


def installation(seed=7):
    return Parameterization(base="gradient-toy", loss="sequence_sft",
        entries=(EntryInstall(name="pi", adapter_type="lora", init={"r": 2, "seed": seed},
                              trainable=True, sites=(SITE,)),),
        optim=OptimSettings(name="adamw", lr=0.01, betas=(0.9, 0.95),
                            weight_decay=0.0, overrides={}))


def batches(documents, budget=1000):
    items = []
    for ids, mask in documents:
        flat = Flat(tuple(ids), tuple(mask), (0,) * len(ids), (0.0,) * len(ids), len(ids))
        items.append((flat, {}))
    return tuple(pack(items, budget))


if torch is not None:
    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(VOCAB, WIDTH)
            self.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)
            self.dropout = torch.nn.Dropout(0.25)
            self.head = torch.nn.Linear(WIDTH, VOCAB, bias=False)

        def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
            h = self.embed(input_ids)
            counts = torch.arange(1, h.shape[1] + 1, dtype=h.dtype)[None, :, None]
            h = h.cumsum(1) / counts
            logits = self.head(self.dropout(h + self.proj(h)))
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:]
            return SimpleNamespace(logits=logits)


@unittest.skipIf(torch is None, "CPU torch required for actual gradient checks")
class GradientAlignmentTest(unittest.TestCase):
    def setUp(self):
        with torch.random.fork_rng():
            torch.manual_seed(91)
            self.learner = TorchLearner(device="cpu", dtype=torch.float32,
                                        checkpoint_activations=False)
            self.learner._model = TinyCausalLM().requires_grad_(False).eval()
        self.learner._base = "gradient-toy"
        self.learner.install("student", installation())
        self.state = self.learner._tenants["student"]
        self.params = tuple(self.state.params["pi"].parameters())

    def measure(self, references=REFERENCES, candidates=CANDIDATES, budget=1000):
        return self.learner.gradient_cosines("student", "pi", batches(references, budget),
            tuple(batches(group, budget) for group in candidates))

    def enable_cache(self, size=1 << 20):
        from rlstack.runner.learners.gradient_cache import GradientCache
        self.learner.gradient_cache = GradientCache(size)

    def test_cache_reuses_reference_and_duplicate_candidates_with_exact_statistics(self):
        self.enable_cache()
        first = self.measure(candidates=(CANDIDATES[0], CANDIDATES[0], CANDIDATES[1]))
        forwards = self.learner.forwards
        second = self.measure()
        self.assertEqual(self.learner.forwards, forwards)
        self.assertFalse(first[0].reference_cache_hit)
        self.assertFalse(first[0].candidate_cache_hit)
        self.assertTrue(first[1].candidate_cache_hit)
        for a, b in zip((first[0], first[2]), second):
            self.assertTrue(b.reference_cache_hit)
            self.assertTrue(b.candidate_cache_hit)
            for name in ('cosine', 'dot', 'reference_norm', 'candidate_norm',
                         'reference_documents', 'candidate_documents'):
                self.assertEqual(getattr(a, name), getattr(b, name))
        reference, _, _ = self.direct(REFERENCES)
        for b, docs in zip(second, CANDIDATES):
            self.assertComparison(b, reference, self.direct(docs)[0])

    def test_cache_invalidates_on_parameter_mutation_and_references_or_masks_change(self):
        self.enable_cache()
        self.measure()
        with torch.no_grad():
            self.params[-1].add_(0.03)
        changed = self.measure()[0]
        self.assertFalse(changed.reference_cache_hit)
        self.assertFalse(changed.candidate_cache_hit)
        reference, _, _ = self.direct(REFERENCES)
        self.assertComparison(changed, reference, self.direct(CANDIDATES[0])[0])
        mask_changed = ((REFERENCES[0][0], (0, 1, 1)), REFERENCES[1])
        measured = self.measure(references=mask_changed)[0]
        self.assertFalse(measured.reference_cache_hit)
        self.assertFalse(measured.candidate_cache_hit)
        self.assertComparison(measured, self.direct(mask_changed)[0], self.direct(CANDIDATES[0])[0])
        measured = self.measure(candidates=((((2, 3, 4, 5), (0, 0, 1, 1)),),))[0]
        self.assertFalse(measured.candidate_cache_hit)
        self.assertComparison(measured, reference,
                              self.direct((((2, 3, 4, 5), (0, 0, 1, 1)),))[0])

    def test_payload_load_invalidates_even_when_adapter_uses_untracked_data_copy(self):
        self.enable_cache()
        self.measure()
        # The native adapter loader copies through .data, which does not bump
        # the Parameter version. The public load verb must invalidate explicitly.
        with torch.no_grad():
            self.params[-1].data.add_(0.07)
        emitted = self.learner.emit('student')
        versions = [p._version for p in self.params]
        self.learner.load('student', emitted.adapters, emitted.optim)
        self.assertEqual([p._version for p in self.params], versions)
        measured = self.measure()[0]
        self.assertFalse(measured.reference_cache_hit)
        self.assertFalse(measured.candidate_cache_hit)
        self.assertComparison(measured, self.direct(REFERENCES)[0], self.direct(CANDIDATES[0])[0])
        self.learner.install('other', installation(seed=18))
        self.measure()
        self.learner.optim_step('other')
        self.assertTrue(self.measure()[0].reference_cache_hit)

    def test_cache_is_bounded_and_uninstall_releases_cached_gradient_storage(self):
        self.enable_cache(size=1)
        self.measure()
        self.assertIsNone(self.learner.gradient_cache.saved_reference)
        self.enable_cache()
        self.learner.gradient_cache.max_comparisons = 1
        self.measure()
        self.assertEqual(len(self.learner.gradient_cache.comparisons), 1)
        self.assertIsNotNone(self.learner.gradient_cache.saved_reference)
        self.learner.uninstall('student')
        self.assertIsNone(self.learner.gradient_cache.saved_reference)
        self.assertFalse(self.learner.gradient_cache.comparisons)

    def direct(self, documents, *, pooled=False):
        """Score independent unpadded sequences; do not reuse learner loss/forward."""
        sums, counts = [], []
        model = self.learner._model
        modes = [(module, module.training) for module in model.modules()]
        model.eval()
        try:
            for ids, mask in documents:
                tokens = torch.tensor([ids])
                route = ReplayRows(slots=(self.state.slot,), index=torch.zeros(1, dtype=torch.long))
                with row_plan(model).route(route):
                    logits = model(input_ids=tokens).logits[:, :-1]
                    losses = torch.nn.functional.cross_entropy(
                        logits.flatten(0, 1).float(), tokens[:, 1:].flatten(), reduction="none")
                    selected = torch.tensor(mask[1:], dtype=torch.bool)
                    sums.append(losses[selected].sum())
                    counts.append(int(selected.sum()))
            objective = (sum(sums) / sum(counts) if pooled else
                         sum(nll / count for nll, count in zip(sums, counts)) / len(sums))
            grads = torch.autograd.grad(objective, self.params, allow_unused=True)
            vector = torch.cat([(torch.zeros_like(p) if g is None else g).flatten().double()
                                for p, g in zip(self.params, grads)])
            return vector, [float(x.detach()) for x in sums], counts
        finally:
            for module, training in modes:
                module.training = training

    def assertTreeEqual(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assertTreeEqual(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assertTreeEqual(a, b)
        else:
            self.assertEqual(left, right)

    def snapshot(self):
        return {tenant: {
            "params": [p.detach().clone() for p in state.params["pi"].parameters()],
            "grads": [None if p.grad is None else p.grad.clone()
                      for p in state.params["pi"].parameters()],
            "optim": (copy.deepcopy(state.optimizers["pi"].state_dict())
                      if "pi" in state.optimizers else None),
        } for tenant, state in self.learner._tenants.items()}

    def assertComparison(self, got, reference, candidate):
        self.assertTrue(got.valid, got.reason)
        dot = float(reference @ candidate)
        rn, cn = float(reference.norm()), float(candidate.norm())
        self.assertAlmostEqual(got.dot, dot, places=6)
        self.assertAlmostEqual(got.reference_norm, rn, places=6)
        self.assertAlmostEqual(got.candidate_norm, cn, places=6)
        self.assertAlmostEqual(got.cosine, dot / (rn * cn), places=6)

    def test_actual_masked_gradient_and_statistics_match_direct_autograd(self):
        reference, sums, counts = self.direct(REFERENCES)
        got = self.measure()
        self.assertEqual(len(got), 2)
        for row, docs in zip(got, CANDIDATES):
            candidate, csums, ccounts = self.direct(docs)
            self.assertComparison(row, reference, candidate)
            self.assertEqual([s.scored_tokens for s in row.reference_documents], counts)
            for stat, expected in zip(row.reference_documents, sums):
                self.assertAlmostEqual(stat.nll_sum, expected, places=5)
            self.assertEqual([s.scored_tokens for s in row.candidate_documents], ccounts)
            for stat, expected in zip(row.candidate_documents, csums):
                self.assertAlmostEqual(stat.nll_sum, expected, places=5)
        a_size = self.params[0].numel()
        self.assertEqual(float(reference[:a_size].abs().max()), 0.0)
        self.assertGreater(float(reference[a_size:].norm()), 0.0)
        self.assertEqual(got[0].parameter_count, sum(p.numel() for p in self.params))
        self.assertEqual(len(got[0].parameter_schema_sha256), 64)

    def test_reference_mean_is_not_token_pooling_or_mean_of_cosines(self):
        reference, _, _ = self.direct(REFERENCES)
        pooled, _, _ = self.direct(REFERENCES, pooled=True)
        candidate, _, _ = self.direct(CANDIDATES[0])
        score = lambda a: float((a @ candidate) / (a.norm() * candidate.norm()))
        got = self.measure()[0]
        mean_cosines = sum(score(self.direct((doc,))[0]) for doc in REFERENCES) / 2
        self.assertGreater(abs(score(reference) - score(pooled)), 1e-3)
        self.assertGreater(abs(score(reference) - mean_cosines), 1e-3)
        self.assertAlmostEqual(got.cosine, score(reference), places=6)

    def test_microbatch_partition_does_not_change_objective(self):
        packed, split = self.measure(budget=1000), self.measure(budget=6)
        for a, b in zip(packed, split):
            for field in ("cosine", "dot", "reference_norm", "candidate_norm"):
                self.assertAlmostEqual(getattr(a, field), getattr(b, field), places=6)
            self.assertEqual(a.reference_documents, b.reference_documents)
            self.assertEqual(a.parameter_schema_sha256, b.parameter_schema_sha256)

    def test_candidate_order_and_intervening_candidate_do_not_change_score(self):
        ab = self.measure()
        baa = self.measure(candidates=(CANDIDATES[1], CANDIDATES[0], CANDIDATES[0]))
        for a, b in ((ab[0], baa[1]), (ab[1], baa[0]), (baa[1], baa[2])):
            for field in ("cosine", "dot", "reference_norm", "candidate_norm"):
                self.assertEqual(getattr(a, field), getattr(b, field))

    def seed_existing_state(self):
        self.learner.install("other", installation(seed=18))
        for state in self.learner._tenants.values():
            for p in state.params["pi"].parameters():
                p.grad = torch.full_like(p, 0.25)
            state.optimizers["pi"].step()
            for p in state.params["pi"].parameters():
                p.grad = torch.full_like(p, 13.0)

    def test_success_preserves_stale_gradients_optimizers_other_tenant_rng_and_mode(self):
        self.seed_existing_state()
        before, rng = self.snapshot(), torch.get_rng_state().clone()
        modes = [m.training for m in self.learner._model.modules()]
        pointers = [id(p.grad) for state in self.learner._tenants.values()
                    for p in state.params["pi"].parameters()]
        self.measure()
        self.assertTreeEqual(before, self.snapshot())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(modes, [m.training for m in self.learner._model.modules()])
        self.assertEqual(pointers, [id(p.grad) for state in self.learner._tenants.values()
                                  for p in state.params["pi"].parameters()])
        self.assertIsNone(row_plan(self.learner._model)._rows)

    def test_failure_unwinds_route_rng_mode_and_all_tenant_state(self):
        self.seed_existing_state()
        before, rng = self.snapshot(), torch.get_rng_state().clone()
        modes = [m.training for m in self.learner._model.modules()]
        original, calls = self.learner._batched_logprobs, []
        def fail_on_candidate(*args, **kwargs):
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("injected candidate failure")
            return original(*args, **kwargs)
        with patch.object(self.learner, "_batched_logprobs", side_effect=fail_on_candidate):
            with self.assertRaisesRegex(RuntimeError, "injected candidate failure"):
                self.measure()
        self.assertTreeEqual(before, self.snapshot())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(modes, [m.training for m in self.learner._model.modules()])
        self.assertIsNone(row_plan(self.learner._model)._rows)
        self.assertTrue(self.measure()[0].valid)

    def test_unweighted_future_suffix_cannot_change_earlier_scored_gradient(self):
        left = (((1, 2, 3, 4, 5), (0, 0, 1, 0, 0)),)
        right = (((1, 2, 3, 11, 12), (0, 0, 1, 0, 0)),)
        a, b = self.measure(references=left), self.measure(references=right)
        for x, y in zip(a, b):
            self.assertEqual(x.cosine, y.cosine)
            self.assertEqual(x.reference_documents, y.reference_documents)

    def test_removing_only_future_unweighted_suffix_preserves_gradient(self):
        full = (((1, 2, 3, 4, 5, 6), (0, 0, 1, 0, 0, 0)),)
        trimmed = (((1, 2, 3), (0, 0, 1)),)
        for a, b in zip(self.measure(references=full), self.measure(references=trimmed)):
            self.assertAlmostEqual(a.cosine, b.cosine, places=6)
            self.assertAlmostEqual(a.reference_norm, b.reference_norm, places=6)
            self.assertEqual(a.reference_documents, b.reference_documents)

    def test_unused_nonfinite_logprob_does_not_poison_selected_objective(self):
        documents = (((1, 2, 3, 4), (0, 0, 1, 0)),)
        expected = self.measure(references=documents)[0]
        original = self.learner._batched_logprobs
        def unused_nan(batch, spans):
            logprobs = original(batch, spans)
            unused = torch.tensor(batch.loss_mask, dtype=torch.bool).logical_not()
            return logprobs.masked_fill(unused, float("nan"))
        with patch.object(self.learner, "_batched_logprobs", side_effect=unused_nan):
            got = self.measure(references=documents)[0]
        self.assertTrue(got.valid, got.reason)
        self.assertAlmostEqual(got.cosine, expected.cosine, places=6)
        self.assertEqual(got.reference_documents, expected.reference_documents)

    def test_invalid_masks_and_empty_targets_are_rejected(self):
        for documents in ((((1, 2), (1, 1)),), (((1, 2), (0, 0)),),
                          (((1, 2), (0, 2)),)):
            with self.subTest(documents=documents), self.assertRaises(ValueError):
                self.measure(references=documents)
        self.assertIsNone(row_plan(self.learner._model)._rows)

    def test_training_mode_is_refused_without_mutating_mode(self):
        model = self.learner._model
        model.train()
        model.dropout.eval()
        modes = [module.training for module in model.modules()]
        with self.assertRaises((ValueError, RuntimeError)):
            self.measure()
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertIsNone(row_plan(model)._rows)

    def test_zero_gradient_is_explicitly_invalid(self):
        self.learner._model.head.weight.data.zero_()
        for row in self.measure():
            self.assertFalse(row.valid)
            self.assertIsNone(row.cosine)
            self.assertTrue(row.reason)
            self.assertEqual(row.reference_norm, 0.0)

    def test_nonfinite_gradient_is_explicitly_invalid(self):
        self.learner._model.head.weight.data.fill_(float("nan"))
        for row in self.measure():
            self.assertFalse(row.valid)
            self.assertIsNone(row.cosine)
            self.assertTrue(row.reason)
        self.assertIsNone(row_plan(self.learner._model)._rows)

    def test_sharded_or_data_parallel_measurement_is_explicitly_refused(self):
        for field in ("fsdp", "dp"):
            with self.subTest(field=field):
                setattr(self.learner, field, 2)
                try:
                    with self.assertRaises((ValueError, NotImplementedError)):
                        self.measure()
                finally:
                    setattr(self.learner, field, 1)

    def test_fully_frozen_outer_replay_scores_without_backward_or_optimizer_state(self):
        spec = installation()
        frozen = replace(spec, entries=tuple(replace(entry, trainable=False)
                                              for entry in spec.entries))
        self.learner.install("frozen-writer", frozen)
        state = self.learner._tenants["frozen-writer"]
        self.assertEqual(state.trainable, [])
        self.assertEqual(state.optimizers, {})
        frozen_params = tuple(state.params["pi"].parameters())
        for p in frozen_params:
            p.grad = torch.full_like(p, 17.0)
        before = {"params": [p.detach().clone() for p in frozen_params],
                  "grads": [p.grad.clone() for p in frozen_params]}
        grad_objects = [id(p.grad) for p in frozen_params]
        other_before = self.snapshot()
        stats = self.learner.accumulate("frozen-writer", batches(REFERENCES))
        self.learner.optim_step("frozen-writer")
        self.assertTrue(all(row.loss > 0 for row in stats))
        self.assertEqual(self.learner.emit("frozen-writer").optim, {})
        self.assertTreeEqual(before, {"params": [p.detach() for p in frozen_params],
                                      "grads": [p.grad for p in frozen_params]})
        self.assertEqual(grad_objects, [id(p.grad) for p in frozen_params])
        self.assertTreeEqual(other_before, self.snapshot())
        self.assertIsNone(row_plan(self.learner._model)._rows)

    def test_fresh_seed_reset_reproduces_measurement(self):
        initial = self.measure()
        self.seed_existing_state()
        self.learner.install("student", installation())
        self.state = self.learner._tenants["student"]
        self.params = tuple(self.state.params["pi"].parameters())
        again = self.measure()
        for a, b in zip(initial, again):
            self.assertEqual(a.cosine, b.cosine)
            self.assertEqual(a.parameter_schema_sha256, b.parameter_schema_sha256)


if __name__ == "__main__":
    unittest.main()
