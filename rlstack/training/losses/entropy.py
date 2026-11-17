"""An exact policy-entropy term, composable with an experiment's objective."""

from __future__ import annotations

import math

from rlstack.training.losses.base import PolicyOutputs


def entropy_regularization(out: PolicyOutputs, batch, coefficient: float):
    """Return (-coefficient * entropy, sufficient statistics) for minimization.

    Each document contributes its mean over loss_mask tokens, divided by
    documents_in_update. Summing these contributions over microbatches gives
    one equal-document mean for the update, independent of packing. Empty
    documents contribute zero and remain in that document denominator.

    The token sum/count are separate diagnostics: their ratio is a TOKEN mean,
    which need not equal the document mean used by this objective. Entropy is
    in nats for the replay policy at T=1, before sampling transformations.
    The registered composing loss must declare needs_entropy=True.
    """
    import torch

    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("entropy coefficient must be finite and nonnegative")
    if out.entropies is None:
        raise ValueError("exact entropy requires a loss declaring needs_entropy=True")
    entropy = out.entropies
    if entropy.shape != out.logprobs.shape or entropy.ndim != 1:
        raise ValueError("entropies must be token-aligned with logprobs")
    mask = torch.tensor(batch.loss_mask, dtype=entropy.dtype, device=entropy.device)
    starts = (*batch.doc_starts, len(batch))
    sums = [(entropy[a:b] * mask[a:b]).sum()
            for a, b in zip(starts, starts[1:])]
    counts = [mask[a:b].sum() for a, b in zip(starts, starts[1:])]
    document_mean = torch.stack([
        total / count.clamp(min=1) for total, count in zip(sums, counts)
    ]).sum() / batch.documents_in_update
    term = -coefficient * document_mean
    values = torch.stack([torch.stack((total, count))
                          for total, count in zip(sums, counts)]).detach().tolist()
    components = {
        "entropy_sum": sum(v[0] for v in values),
        "entropy_scored_tokens": sum(v[1] for v in values),
        "entropy_coefficient": float(coefficient),
        "entropy_documents_in_update": float(batch.documents_in_update),
        "entropy_document_mean_contribution": float(document_mean.detach()),
        "entropy_objective_contribution": float(term.detach()),
    }
    for i, (total, count) in enumerate(values):
        components[f"document_{i}_entropy_sum"] = total
        components[f"document_{i}_entropy_tokens"] = count
    return term, components
