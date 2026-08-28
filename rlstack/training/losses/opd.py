"""OPD: sampled-token reverse KL against a LIVE frozen teacher pool's scores."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("opd", requires=("teacher_logprobs",))
def opd(out: PolicyOutputs, batch: Any) -> LossResult:
    """Sampled-token REVERSE KL against a frozen teacher (GKD's on-policy
    branch): the student samples, the teacher scores those very tokens, and the
    value is the masked mean of (student_lp − teacher_lp) over generated tokens
    — one sample deep, because the teacher hands back the chosen tokens'
    logprobs and no distribution beyond them. "teacher_logprobs" is the
    token_level column the teacher_logprobs processor scored through the
    "teacher" pool, so distilling from a different teacher means declaring a
    different pool, not editing this file.

    WHY A SURROGATE. The tokens are the student's own draws, so the gradient of
    the KL is the score-function gradient E[(lp − teacher) ∇lp]. The expression
    below reports the KL as its VALUE and carries that gradient; differentiating
    the plain difference instead would cancel the teacher out entirely
    (∇(lp − teacher) = ∇lp) and simply push every sampled token down.
    """
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    teacher = torch.tensor(batch.post["teacher_logprobs"], dtype=lp.dtype,
                           device=lp.device)

    reverse_kl = (lp - teacher).detach()
    # value == reverse_kl (the second factor is 1.0), gradient == reverse_kl ∇lp
    surrogate = reverse_kl * (1.0 + lp - lp.detach())
    objective = (surrogate * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
