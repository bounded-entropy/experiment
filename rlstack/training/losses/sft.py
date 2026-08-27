"""SFT: behavior cloning on the sealed record."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("sft")
def sft(out: PolicyOutputs, batch: Any) -> LossResult:
    """Behavior cloning on the sealed record: maximize the trainer's logprob
    of every generated token. No importance correction, no advantage — the
    data source (a static cas:// set, a replayed run) IS the curriculum."""
    lp, mask, behavior = token_tensors(out, batch)
    objective = -(lp * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
