"""SDFT v0: reward-weighted behavior cloning on the policy's own samples."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("sdft", requires=("reward",))
def sdft(out: PolicyOutputs, batch: Any) -> LossResult:
    """Self-distillation fine-tuning, v0: clone only what the pipeline scored
    — rejection sampling as a loss. Each token is weighted by its trajectory's
    "reward"; an all-zero-reward microbatch contributes zero."""
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    weight = torch.tensor(batch.postdata["reward"], dtype=lp.dtype,
                          device=lp.device) * mask
    objective = -(lp * weight).sum() / weight.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
