"""GRPO: token-level PPO-clip over group-z-scored advantages."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("grpo", requires=("advantage",))
def grpo(out: PolicyOutputs, batch: Any, clip_eps: float = 0.2) -> LossResult:
    """Token-level PPO-clip surrogate over batch.post["advantage"], with the
    IS ratio against the RECORDED behavior logprobs (I6: never recomputed)."""
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    advantage = torch.tensor(batch.post["advantage"], dtype=lp.dtype,
                             device=lp.device)

    ratio = torch.exp(lp - behavior)
    surrogate = torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage)
    objective = -(surrogate * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
