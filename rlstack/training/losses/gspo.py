"""GSPO: the importance-corrected unit is the SEQUENCE, not the token."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("gspo", requires=("advantage",))
def gspo(out: PolicyOutputs, batch: Any, clip_eps: float = 0.1) -> LossResult:
    """One length-normalized IS ratio PER DOC — exp(mean over generated tokens
    of lp − behavior) — clipped and weighted by the doc's advantage (constant
    across its tokens by broadcast), against the RECORDED behavior logprobs
    (I6: never recomputed)."""
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    advantage = torch.tensor(batch.postdata["advantage"], dtype=lp.dtype,
                             device=lp.device)

    starts = list(batch.doc_starts) + [len(batch)]
    per_doc = []
    for start, stop in zip(starts, starts[1:]):
        doc_mask = mask[start:stop]
        n = doc_mask.sum().clamp(min=1.0)
        seq_ratio = torch.exp(
            ((lp[start:stop] - behavior[start:stop]) * doc_mask).sum() / n)
        doc_advantage = (advantage[start:stop] * doc_mask).sum() / n
        per_doc.append(torch.minimum(
            seq_ratio * doc_advantage,
            torch.clamp(seq_ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            * doc_advantage))
    objective = -torch.stack(per_doc).mean()

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
