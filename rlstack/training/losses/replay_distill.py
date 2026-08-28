"""replay_distill: match the RECORDED behavior logprobs of a replayed run."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("replay_distill", requires=("behavior_logprobs",))
def replay_distill(out: PolicyOutputs, batch: Any) -> LossResult:
    """Distil from a SEALED RECORD: squared error between trainer logprobs and
    the behavior logprobs of whatever policy sampled the replayed trajectories
    (I6 — read from the record, never recomputed). The teacher is a past run,
    not a model still around to ask, so the rows are off-policy by
    construction; requiring "behavior_logprobs" puts that teacher signal on the
    flow graph as feeding the loss."""
    lp, mask, behavior = token_tensors(out, batch)
    objective = (((lp - behavior) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
