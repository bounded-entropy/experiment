"""PPO (value-free): token-level clip over an unnormalized advantage."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("ppo", requires=("advantage",))
def ppo(out: PolicyOutputs, batch: Any, clip_eps: float = 0.2) -> LossResult:
    """Token-level PPO-clip over an UNNORMALIZED advantage (pair with
    center_reward: mean-baseline, no variance rescaling — the classic
    value-free PPO estimator; grpo differs only by its z-scored input)."""
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
