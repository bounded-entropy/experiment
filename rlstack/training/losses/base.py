"""The loss contract: one registered function per objective, one file each.

A loss is microbatch-scope and differentiable: pure fn(PolicyOutputs,
TokenBatch) -> LossResult, with declared `requires` naming the postdata
columns and planned passes its math reads. The learner runs the forward and
builds PolicyOutputs; the loss owns only the objective. torch is imported
inside bodies (rule 7): declarations validate without it.

    @loss("my_loss", requires=("advantage",))
    def my_loss(out, batch) -> LossResult:
        lp, mask, behavior = token_tensors(out, batch)
        ...
        mean_ratio, gap = rails(lp, mask, behavior)
        return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def token_tensors(out: PolicyOutputs, batch: Any):
    """The three token-aligned tensors every loss starts from: trainer
    logprobs, the mask over generated tokens, RECORDED behavior logprobs."""
    import torch

    lp = out.logprobs
    mask = torch.tensor(batch.loss_mask, dtype=lp.dtype, device=lp.device)
    behavior = torch.tensor(batch.behavior_logprobs, dtype=lp.dtype,
                            device=lp.device)
    return lp, mask, behavior


def rails(lp, mask, behavior) -> tuple[float, float]:
    """The two rails every loss reports: masked mean IS ratio, and the masked
    mean |trainer − behavior| logprob gap (the silent-off-policy alarm)."""
    import torch

    with torch.no_grad():
        n = mask.sum().clamp(min=1.0)
        mean_ratio = float((torch.exp(lp - behavior) * mask).sum() / n)
        gap = float(((lp - behavior).abs() * mask).sum() / n)
    return mean_ratio, gap
