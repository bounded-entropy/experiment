"""Equal document means of selected-token supervised NLL."""
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

