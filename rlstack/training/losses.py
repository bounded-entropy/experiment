"""Built-in losses (training world).

A loss is microbatch-scope and differentiable: pure fn(PolicyOutputs,
TokenBatch) -> LossResult, with declared `requires` naming the postdata
columns and planned passes its math reads. The learner runs the forward and
builds PolicyOutputs; the loss owns only the objective. torch is imported
inside the body (rule 7): declarations validate without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlstack.registry import loss


@dataclass
class PolicyOutputs:
    """What the training forward produced for one microbatch, token-aligned
    with the TokenBatch: logprobs[t] is the trainer's logprob of token t
    given its prefix (0.0 at injected positions — masked out anyway)."""

    logprobs: Any                 # torch.Tensor [T], grad flows through it


@dataclass
class LossResult:
    """The objective plus the rails every loss must report."""

    loss: Any                     # scalar torch.Tensor, ready to backward
    mean_ratio: float             # masked mean of exp(lp - behavior_lp)
    logprob_gap: float            # masked mean |lp - behavior_lp| — the
                                  # silent-off-policy / parity alarm


@loss("grpo", requires=("advantage",))
def grpo(out: PolicyOutputs, batch: Any, clip_eps: float = 0.2) -> LossResult:
    """Token-level PPO-clip surrogate over batch.post["advantage"], with the
    IS ratio against the RECORDED behavior logprobs (I6: never recomputed)."""
    import torch

    lp = out.logprobs
    mask = torch.tensor(batch.loss_mask, dtype=lp.dtype, device=lp.device)
    behavior = torch.tensor(batch.behavior_logprobs, dtype=lp.dtype,
                            device=lp.device)
    advantage = torch.tensor(batch.post["advantage"], dtype=lp.dtype,
                             device=lp.device)

    ratio = torch.exp(lp - behavior)
    surrogate = torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage)
    n_trainable = mask.sum().clamp(min=1.0)
    objective = -(surrogate * mask).sum() / n_trainable

    with torch.no_grad():
        mean_ratio = float((ratio * mask).sum() / n_trainable)
        gap = float(((lp - behavior).abs() * mask).sum() / n_trainable)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
