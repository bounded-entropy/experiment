"""Supervised sequence means with an optional task posterior penalty.

Each sequence has equal weight after averaging its scored tokens. Packing
can divide an update, but cannot change its objective or KL coefficient.
"""

from dataclasses import replace
from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, rails, token_tensors


@loss("sequence_sft")
def sequence_sft(out, batch):
    lp, mask, behavior = token_tensors(out, batch)
    starts = (*batch.doc_starts, len(batch))
    objective = sum(-(lp[a:b] * mask[a:b]).sum() / mask[a:b].sum().clamp(min=1)
                    for a, b in zip(starts, starts[1:])) / batch.documents_in_update
    ratio, gap = rails(lp, mask, behavior)
    return LossResult(objective, ratio, gap)


@loss("factual_sft", requires=("latent_kl", "factual_beta"))
def factual_sft(out, batch):
    result = sequence_sft(out, batch)
    fraction = len(batch.doc_starts) / batch.documents_in_update
    return replace(result, loss=result.loss + fraction * out.provided["factual_beta"]
                   * out.provided["latent_kl"])
