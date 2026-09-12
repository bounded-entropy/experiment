"""SDPO: the loop is the teacher — clone each document's FINAL turn.

The iterative-SDPO thesis, as a loss: rewards are almost unnecessary,
because the model's own language defines the credit. The reflect loop
(Derive leaves + the reflect_retry environment) already spent its selection
pressure IN CONTEXT — attempt, self-critique, retry — so what training does
is amortize it: behavior-clone the final turn of every document, critique
turns and earlier attempts conditioning but never trained as answers. No
reward, no advantage, no reference: `requires` is empty, and the grader's
columns ride along for the observer only.

The final turn is read off `segment_ids` (the turn index flatten already
stamps per token), so no new masking primitive exists: a single-turn
document's final turn is its only turn, which is what lets one plan mix
first attempts and reflect episodes and train both under one rule.

THE KNOWN FAILURE MODE, stated (SCoRe, 2024): training on retries can teach
sandbagging — a worse first attempt makes the improvement easy. The honest
metric is therefore the Measurement, which evaluates iteration-0 behavior
under the PLAIN environment: whether the loop's lessons reached the weights,
not the context."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors


@loss("sdpo")
def sdpo(out: PolicyOutputs, batch: Any) -> LossResult:
    """Behavior cloning masked to each document's final turn."""
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    segments = torch.tensor(batch.segment_ids, dtype=torch.long,
                            device=lp.device)
    final = torch.zeros_like(mask)
    starts = list(batch.doc_starts)
    for start, stop in zip(starts, starts[1:] + [len(batch)]):
        doc_segments = segments[start:stop]
        doc_mask = mask[start:stop]
        if float(doc_mask.sum()) == 0.0:
            continue
        last_turn = doc_segments[doc_mask > 0].max()
        final[start:stop] = doc_mask * (doc_segments == last_turn).to(mask.dtype)
    objective = -(lp * final).sum() / final.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
