"""self_anchor: anchor the policy to its own lagged sampler's record."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("self_anchor", requires=("behavior_logprobs",))
def self_anchor(out: PolicyOutputs, batch: Any) -> LossResult:
    """Match the RECORDED logprobs of the policy's own lagged sampler: squared
    error between trainer and behavior logprobs on live data under
    max_policy_lag > 0 — an EMA-teacher effect with no second model and no
    extra compute, because the teacher signal was recorded at the seal (I6)."""
    lp, mask, behavior = token_tensors(out, batch)
    objective = (((lp - behavior) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
