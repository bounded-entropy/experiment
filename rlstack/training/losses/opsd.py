"""OPSD — on-policy self-distillation, the real one (supersedes the v0
stand-in now named self_anchor)."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("opsd", requires=("hinted_logprobs",))
def opsd(out: PolicyOutputs, batch: Any) -> LossResult:
    """Distill the policy toward ITSELF under privileged conditioning:
    squared error between trainer logprobs and the hinted logprobs the
    hinted_logprobs post processor scored through the policy pool (one
    prefill pass per turn — I9: the loss is pure math over a column)."""
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    hinted = torch.tensor(batch.post["hinted_logprobs"], dtype=lp.dtype,
                          device=lp.device)
    objective = (((lp - hinted) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
