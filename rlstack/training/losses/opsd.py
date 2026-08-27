"""OPSD v0: on-policy self-distillation — the lagged self as teacher."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("opsd")
def opsd(out: PolicyOutputs, batch: Any) -> LossResult:
    """On-policy self-distillation, v0: the same recorded-confidence matching
    as opd, but on LIVE data under max_policy_lag > 0 — the behavior policy is
    the trainer's own LAGGED self, so the loss anchors the current weights to
    the version that sampled (an EMA-teacher without a second model)."""
    lp, mask, behavior = token_tensors(out, batch)
    objective = (((lp - behavior) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
