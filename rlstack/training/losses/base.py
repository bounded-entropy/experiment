"""The loss contract: pure math over named columns (I9).

A loss is microbatch-scope and differentiable — fn(PolicyOutputs, TokenBatch)
-> LossResult — and its declared `requires` names DATA COLUMNS ONLY: postdata
columns the pipeline produced, recorded facts, or bank-provided forward
tensors. A loss can never cause metal work; anything that needs a GPU (judges,
teacher scoring, hinted rescoring) is a postprocessor's job and lands in
postdata before the loss runs. The learner runs the forward and builds
PolicyOutputs; the loss owns only the objective, and reports the rails beside
it. torch is imported inside bodies (rule 7): declarations validate without it.

    @loss("my_loss", requires=("advantage",))
    def my_loss(out, batch) -> LossResult:
        lp, mask, behavior = token_tensors(out, batch)
        ...
        mean_ratio, gap = rails(lp, mask, behavior)
        return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PolicyOutputs:
    """What the training forward produced for one microbatch, token-aligned
    with the TokenBatch: logprobs[t] is the trainer's logprob of token t
    given its prefix (0.0 at injected positions — masked out anyway).

    `provided` carries the bank's PROVIDED tensors under their declared names
    (AdapterType.provides, computed by AdapterType.provide) — recomputed by
    this forward, part of this graph, and NOT token-aligned: a provided tensor
    is whatever its adapter type says it is. A loss reads only the names it
    declared in `requires`."""

    logprobs: Any                 # torch.Tensor [T], grad flows through it
    provided: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class LossResult:
    """The objective plus the two rails every loss must report."""

    loss: Any                     # scalar torch.Tensor, ready to backward
    mean_ratio: float             # masked mean of exp(lp - behavior_lp)
    logprob_gap: float            # masked mean |lp - behavior_lp| — the
                                  # trainer/sampler mismatch alarm


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
    """The two rails every loss reports: the masked mean IS ratio, and the
    masked mean |trainer − behavior| logprob gap — the trainer/sampler mismatch
    alarm, whose floor is the bf16 kernel difference and whose GROWTH above
    that floor is the signal."""
    import torch

    with torch.no_grad():
        n = mask.sum().clamp(min=1.0)
        mean_ratio = float((torch.exp(lp - behavior) * mask).sum() / n)
        gap = float(((lp - behavior).abs() * mask).sum() / n)
    return mean_ratio, gap
