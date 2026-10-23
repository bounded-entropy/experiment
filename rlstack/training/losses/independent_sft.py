"""A sum of independent students' means of document token-mean NLLs."""
from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, rails, token_tensors


@loss("independent_sft", requires=("sft_document_weight",))
def independent_sft(out, batch):
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    starts = (*batch.doc_starts, len(batch))
    weights = batch.postdata["sft_document_weight"]
    terms, values = [], []
    for start, stop in zip(starts, starts[1:]):
        weight = weights[start]
        if weight <= 0 or any(value != weight for value in weights[start:stop]):
            raise ValueError("one supervised document needs one positive group weight")
        count = mask[start:stop].sum()
        nll = -(lp[start:stop] * mask[start:stop]).sum()
        terms.append(weight * nll / count.clamp(min=1))
        values.append(torch.stack((nll, count)).detach())
    objective = sum(terms)
    ratio, gap = rails(lp, mask, behavior)
    statistics = torch.stack(values).tolist()
    components = {"nll_sum": sum(row[0] for row in statistics),
                  "scored_tokens": sum(row[1] for row in statistics),
                  "objective": float(objective.detach())}
    for index, (nll, count) in enumerate(statistics):
        components[f"document_{index}_nll_sum"] = nll
        components[f"document_{index}_tokens"] = count
        components[f"document_{index}_weight"] = float(weights[starts[index]])
    return LossResult(objective, ratio, gap, components)
