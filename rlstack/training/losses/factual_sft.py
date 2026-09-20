"""Supervised sequence means with an optional task posterior penalty.

Each sequence has equal weight after averaging its scored tokens. Packing
can divide an update, but cannot change its objective or KL coefficient.
"""

from dataclasses import replace
from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, rails, token_tensors


@loss("sequence_sft")
def sequence_sft(out, batch):
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    starts = (*batch.doc_starts, len(batch))
    sums = [-(lp[a:b] * mask[a:b]).sum() for a, b in zip(starts, starts[1:])]
    counts = [mask[a:b].sum() for a, b in zip(starts, starts[1:])]
    objective = sum(nll / count.clamp(min=1) for nll, count in zip(sums, counts)) / batch.documents_in_update
    ratio, gap = rails(lp, mask, behavior)
    # One small transfer; per-document sufficient statistics let the fit
    # journal separate lanes even when they shared a packed forward.
    values = torch.stack([torch.stack((nll, count)) for nll, count in zip(sums, counts)]).detach().tolist()
    components = {"nll_sum": sum(v[0] for v in values), "scored_tokens": sum(v[1] for v in values),
                  "documents_in_update": float(batch.documents_in_update),
                  "objective": float(objective.detach())}
    for i, (nll, count) in enumerate(values):
        components[f"document_{i}_nll_sum"] = nll
        components[f"document_{i}_tokens"] = count
    return LossResult(objective, ratio, gap, components)


@loss("factual_sft", requires=("latent_kl", "factual_beta"))
def factual_sft(out, batch):
    result = sequence_sft(out, batch)
    fraction = len(batch.doc_starts) / batch.documents_in_update
    penalty = fraction * out.provided["factual_beta"] * out.provided["latent_kl"]
    objective = result.loss + penalty
    return replace(result, loss=objective, components={**result.components,
        "nll_term": float(result.loss.detach()), "latent_kl": float(out.provided["latent_kl"].detach()),
        "beta": float(out.provided["factual_beta"]), "document_fraction": fraction,
        "kl_term": float(penalty.detach()), "objective": float(objective.detach())})
