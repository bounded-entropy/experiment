"""Replay distillation: match the RECORDED confidence of a replayed run."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("replay_distill", requires=("behavior_logprobs",))
def replay_distill(out: PolicyOutputs, batch: Any) -> LossResult:
    """Distil from a SEALED RECORD: squared error between trainer logprobs and
    the behavior logprobs of whatever policy sampled the replayed trajectories
    (I6 — read from the record, never recomputed). The teacher is a run, not a
    model that is still around to ask; the trajectories are off-policy by
    construction. Declaring the record makes the graph honest: the teacher
    signal FEEDS this loss.

    Named for what it does. TRUE on-policy distillation — a live teacher
    scoring the student's own samples — is `opd` (#47), and it reads a
    teacher_logprobs column instead."""
    lp, mask, behavior = token_tensors(out, batch)
    objective = (((lp - behavior) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
