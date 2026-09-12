"""GRPO plus the existing on-policy token-KL regularizer to the frozen base."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs
from rlstack.training.losses.grpo import grpo
from rlstack.training.losses.opd import opd

@loss("grpo_reference_kl", requires=("advantage", "teacher_logprobs"))
def grpo_reference_kl(out: PolicyOutputs, batch: Any,
                       clip_eps: float = 0.2, beta: float = 0.04) -> LossResult:
    """Reuse OPD's sampled score-function KL gradient; no new estimator here.

    Both terms are token means on the same microbatch, so they retain their
    relative weight when the runner sums microbatch losses. The teacher pool
    must be the frozen base. This differs from TRL's k3-estimator gradient.
    """
    policy = grpo(out, batch, clip_eps)
    reference = opd(out, batch)
    return LossResult(loss=policy.loss + beta * reference.loss,
                      mean_ratio=policy.mean_ratio, logprob_gap=policy.logprob_gap)
