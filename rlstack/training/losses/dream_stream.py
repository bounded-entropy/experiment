"""dream_stream: one wave, three roles — memories clone, the dreamer learns
from its dreams' effects, evaluation rows are masked (ADR 0018).

A row's ROLE is the fact `realize` stamped from its plan leaf (data/plan.py):
a `memory:<jj>` row is a document its memory clones (`sft`'s masked token
NLL); a `dreamer` row is one of the dreamer's own dreams, replayed with the
`advantage` the contrast produced — REINFORCE, the advantage times the
dream's token log-probabilities, no clipped ratio: the dreamer takes one
step per batch on its own samples, so it is on-policy and a clip would
never bind (Samarth, 2026-09-17); the recorded behavior logprobs still
feed the rails; an `eval` row is a sampled answer that
exists to be graded and carries no gradient. Each half is normalized over
its own tokens, and the anchor penalty — a function of the parameters alone
— is added ONCE per update (grpo_latent_kl's rule)."""

from __future__ import annotations

from typing import Any

from rlstack.data.plan import EVAL, TRAIN
from rlstack.policy.adapters.dream_bank import DREAMER, is_library, parts_of
from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors



def role_of(turns) -> str:
    """A document's role: the first turn fact naming one, else `train`."""
    for turn in turns:
        if "role" in turn:
            return str(turn["role"])
    return TRAIN


def document_roles(batch: Any) -> list[str]:
    """One role per document of the microbatch, off `doc_turn_extras`;
    a batch recording nothing is all `train`."""
    if not batch.doc_turn_extras:
        return [TRAIN] * len(batch.doc_starts)
    return [role_of(turns) for turns in batch.doc_turn_extras]


def trains_the_dreamer(role: str) -> bool:
    """A dreamer row is one whose role's TRAINABLE part is the dreamer:
    `dreamer` alone, or stacked on a frozen library memory
    (`lib:<name>+dreamer`, ADR 0019), whose part of the forward is detached."""
    return [part for part in parts_of(role) if not is_library(part)] == [DREAMER]


def role_masks(batch: Any, lp):
    """Three per-token 0/1 tensors — clone, dream, masked — from the
    documents' roles and the batch's doc_starts."""
    import torch

    roles = document_roles(batch)
    starts = list(batch.doc_starts) + [len(batch.token_ids)]
    clone = torch.zeros(len(batch.token_ids), dtype=lp.dtype, device=lp.device)
    dream = torch.zeros_like(clone)
    for role, start, end in zip(roles, starts[:-1], starts[1:]):
        if trains_the_dreamer(role):
            dream[start:end] = 1.0
        elif role != EVAL:
            clone[start:end] = 1.0
    return clone, dream


@loss("dream_stream", requires=("advantage", "anchor_penalty"))
def dream_stream(out: PolicyOutputs, batch: Any) -> LossResult:
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    clone, dream = role_masks(batch, lp)
    clone_mask, dream_mask = mask * clone, mask * dream

    clone_term = -(lp * clone_mask).sum() / clone_mask.sum().clamp(min=1.0)

    advantage = torch.tensor(batch.postdata["advantage"], dtype=lp.dtype,
                             device=lp.device)
    dream_term = -(lp * advantage * dream_mask).sum() / dream_mask.sum().clamp(min=1.0)

    penalty = out.provided["anchor_penalty"] / batch.microbatches_in_update
    mean_ratio, gap = rails(lp, mask, behavior)
    objective = clone_term + dream_term + penalty
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap,
                      components={
                          "clone_nll_sum": float(-(lp * clone_mask).sum().detach()),
                          "clone_tokens": float(clone_mask.sum()),
                          "clone_term": float(clone_term.detach()),
                          "dream_nll_sum": float(-(lp * dream_mask).sum().detach()),
                          "dream_tokens": float(dream_mask.sum()),
                          "reinforce_nll_sum": float(-(lp * advantage * dream_mask).sum().detach()),
                          "reinforce_term": float(dream_term.detach()),
                          "anchor_term": float(penalty.detach()),
                          "microbatches_in_update": float(batch.microbatches_in_update),
                          "objective": float(objective.detach())})
