"""anchor_calibrate: the zero objective of a calibrate run (ADR 0018).

The run exists for its FORWARDS — a dream_bank entry in calibrate mode
accumulates the activations' covariance at every site it wraps — and its
single checkpoint is the anchor. Nothing should move, so the loss is zero
with a graph (a step on a zero gradient is a no-op at lr 0), and the rails
are reported like any loss's."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("anchor_calibrate")
def anchor_calibrate(out: PolicyOutputs, batch: Any) -> LossResult:
    lp, mask, behavior = token_tensors(out, batch)
    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=0.0 * lp.sum(), mean_ratio=mean_ratio, logprob_gap=gap)
