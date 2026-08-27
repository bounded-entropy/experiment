"""self_anchor: anchor the policy to its own lagged sampler's record."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("self_anchor", requires=("behavior_logprobs",))
def self_anchor(out: PolicyOutputs, batch: Any) -> LossResult:
    """Match the RECORDED logprobs of the policy's own lagged sampler:
    squared error between trainer and behavior logprobs on live data under
    max_policy_lag > 0 — an EMA-teacher effect with no second model and no
    extra compute (the teacher signal was recorded at sampling, I6).

    This is NOT on-policy self-distillation: true OPSD distills from HINTED
    logprobs — the same weights re-scored under privileged conditioning —
    which per #38 is a post processor's job (a token_level column produced
    by rescoring through a pool), not a loss-side pass.
    """
    lp, mask, behavior = token_tensors(out, batch)
    objective = (((lp - behavior) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
