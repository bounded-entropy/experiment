"""The opd objective's two claims, on synthetic tensors (#47).

opd reports the per-token reverse KL as its VALUE and carries the
score-function gradient of it — the tokens are the student's own draws, so a
plain pathwise derivative of (student_lp − teacher_lp) would cancel the
teacher out entirely. Both halves are checkable in four lines of torch, and
neither needs a GPU.

torch ships in the deploy image, not in the client environment, so this file
SKIPS locally and RUNS on CPU under `modal run deploy/modal_app.py::run_tests`
— the same arrangement test_batched_replay.py uses.
"""

from __future__ import annotations

import unittest

from rlstack.data.flatten import TokenBatch
from rlstack.registry import LOSSES
from rlstack.training.losses.base import PolicyOutputs

try:
    import torch
except ImportError:                                  # the client environment
    torch = None

needs_torch = unittest.skipUnless(
    torch is not None, "torch is trainer metal: this suite runs in the image")

# three generated tokens and one injected position (mask 0, teacher 0.0)
MASK = (1, 1, 1, 0)
STUDENT = (-0.20, -1.50, -0.70, 0.0)
TEACHER = (-0.50, -0.30, -0.90, 0.0)
BEHAVIOR = (-0.25, -1.40, -0.75, 0.0)


def a_batch() -> TokenBatch:
    return TokenBatch(token_ids=(11, 12, 13, 14), loss_mask=MASK,
                      behavior_logprobs=BEHAVIOR, segment_ids=(0, 0, 0, 0),
                      doc_starts=(0,), post={"teacher_logprobs": TEACHER})


@needs_torch
class OpdObjectiveTest(unittest.TestCase):
    def outcome(self):
        lp = torch.tensor(STUDENT, dtype=torch.float64, requires_grad=True)
        result = LOSSES.get("opd").fn(PolicyOutputs(logprobs=lp), a_batch())
        result.loss.backward()
        return lp, result

    def test_the_value_is_the_masked_mean_reverse_kl(self) -> None:
        """What the ledger plots is the teacher-student KL in nats per
        generated token — the injected position contributes nothing."""
        _, result = self.outcome()
        expected = sum((s - t) for s, t, m in zip(STUDENT, TEACHER, MASK) if m) / 3
        self.assertAlmostEqual(float(result.loss), expected, places=12)

    def test_the_gradient_is_the_score_function_estimator(self) -> None:
        """d/d lp[i] = (lp[i] − teacher[i]) / n on generated tokens: the
        teacher WEIGHTS ∇log π rather than vanishing from the derivative."""
        lp, _ = self.outcome()
        expected = [(s - t) / 3 if m else 0.0
                    for s, t, m in zip(STUDENT, TEACHER, MASK)]
        for got, want in zip(lp.grad.tolist(), expected):
            self.assertAlmostEqual(got, want, places=12)

    def test_a_teacher_that_agrees_leaves_the_student_alone(self) -> None:
        """Zero KL, zero gradient — the fixed point of the objective."""
        lp = torch.tensor(TEACHER, dtype=torch.float64, requires_grad=True)
        result = LOSSES.get("opd").fn(PolicyOutputs(logprobs=lp), a_batch())
        result.loss.backward()
        self.assertAlmostEqual(float(result.loss), 0.0, places=12)
        self.assertAlmostEqual(float(lp.grad.abs().max()), 0.0, places=12)

    def test_the_rails_still_measure_the_behavior_record(self) -> None:
        """logprob_gap is the parity alarm and answers to the RECORD, never
        to the teacher — every loss reports it the same way."""
        _, result = self.outcome()
        gap = sum(abs(s - b) for s, b, m in zip(STUDENT, BEHAVIOR, MASK) if m) / 3
        self.assertAlmostEqual(result.logprob_gap, gap, places=12)


if __name__ == "__main__":
    unittest.main()
