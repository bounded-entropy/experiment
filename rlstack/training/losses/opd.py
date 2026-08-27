"""OPD v0: off-policy distillation from a replayed run's recorded teacher."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("opd", requires=("behavior_logprobs",))
def opd(out: PolicyOutputs, batch: Any) -> LossResult:
    """Off-policy distillation, v0: match the teacher's RECORDED confidence on
    its own sampled tokens — squared error between trainer and behavior
    logprobs. The teacher is whatever policy sealed the replayed run (I6: its
    logprobs are read from the record, never recomputed). Declaring the
    record makes the graph honest: the teacher signal FEEDS this loss."""
    lp, mask, behavior = token_tensors(out, batch)
    objective = (((lp - behavior) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
