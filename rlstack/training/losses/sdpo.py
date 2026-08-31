"""SDPO v0: self-play DPO on the policy's own groups, in unpaired form.

SELF-PLAY: both sides of every preference come from the policy's own sampled
group — dpo_pair marks the group's best completion chosen (+1) and its worst
rejected (-1) — and the reference is the RECORDED behavior policy (I6), the
very distribution the pair was drawn from. The implicit reward is therefore
s_i = sum over generated tokens of (trainer logprob - behavior logprob): zero
at a fresh update, pushed up on chosen and down on rejected.

UNPAIRED (the KTO form), stated rather than hidden: pack() may split a group
across microbatches, and a paired margin -log sigmoid(beta (s_c - s_r)) needs
both members in one forward, which a token column cannot guarantee. Each
marked document instead pays its own logistic

    chosen:   -log sigmoid(+beta s_i)
    rejected: -log sigmoid(-beta s_i)

whose gradient matches the paired form's direction per member and saturates
the same way once the margin is won. Unmarked documents (pair 0, including
every member of an all-tie group) contribute nothing. The paired-margin
upgrade is a group-aware pack, the same future the coalescer owns.
"""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors

BETA = 0.5      # the DPO temperature: how sharply a won margin saturates.
#                 A module constant for grpo_latent_kl's reason — it is the
#                 objective, so sweeping it must produce a different run_id.


@loss("sdpo", requires=("pair",))
def sdpo(out: PolicyOutputs, batch: Any) -> LossResult:
    """Per-document logistic on the implicit reward, signed by the pair
    verdict; documents are read off doc_starts, s_i is masked within each."""
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    pair = torch.tensor(batch.postdata["pair"], dtype=lp.dtype,
                        device=lp.device)

    starts = list(batch.doc_starts)
    terms = []
    for start, stop in zip(starts, starts[1:] + [len(batch)]):
        doc_mask = mask[start:stop]
        verdict = pair[start:stop].max() + pair[start:stop].min()  # +1, -1, 0
        if float(verdict) == 0.0 or float(doc_mask.sum()) == 0.0:
            continue
        implicit = ((lp[start:stop] - behavior[start:stop]) * doc_mask).sum()
        terms.append(-torch.nn.functional.logsigmoid(
            verdict * BETA * implicit))
    if terms:
        objective = torch.stack(terms).mean()
    else:
        objective = (lp * mask).sum() * 0.0     # an all-tie microbatch: no
        #                                         preference, a zero with grad
    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
