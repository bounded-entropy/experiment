"""Reverse PPO v2: the clip surrogate over hindsight-DIFFERENCED prefix values.

THE COLLAPSE THAT KILLED v1, kept on record: v1's critic attended the SUFFIX
of the trunk's hidden states — but causal states already encode their whole
prefix, so every suffix window was fully informed, the Bayes-optimal track
was constant-in-t, and the telescoped credit degenerated to a spike at the
final token (REINFORCE at EOS). The information gradient along the sequence
is the credit signal, and only a PREFIX predictor has one.

So v2 is the RUDDER form on the trunk's own prefix encoding. The value_head
provides v(t) — a tiny probe on h_t, trained toward the episode reward at
every generated position AND at the prompt's last token — and once fit,
v(t) = E[reward | prompt, completion <= t]: it JUMPS exactly where decisions
happen. Credit is the forward difference

    credit(t) = v(t) - v(t-1)

which telescopes per document to v(T) - v(prompt end) -> reward - E[reward |
task]: total credit is the CENTERED return — GRPO's per-task baseline,
emerging from the value function with no groups — distributed at the tokens
that moved the prediction. The hindsight is ACROSS episodes (the critic is
fit on completed trajectories), not within the sequence, which is what makes
the estimator non-degenerate.

The pieces: a value regression on generated positions plus each document's
prompt-end position (that one is what learns the per-task baseline); the PPO
clip surrogate over the DETACHED, batch-whitened credits. Values arrive as a
PROVIDED tensor — [rows, padded_tokens] from the same forward — and this loss
re-aligns them to the flat token stream by doc spans. A zero-initialized head
gives all-zero credit, which whitens to zero: the first updates train only
the critic, an automatic warm-up with no schedule."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors

VALUE_COEF = 0.5     # the critic's share of the objective — a module constant
#                      because it is the objective: sweeping it must produce a
#                      different run_id.


@loss("reverse_ppo", requires=("reward", "values"))
def reverse_ppo(out: PolicyOutputs, batch: Any,
                clip_eps: float = 0.2) -> LossResult:
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    reward = torch.tensor(batch.postdata["reward"], dtype=lp.dtype,
                          device=lp.device)
    padded = out.provided["values"]              # [rows, W], 0 at padding
    starts = list(batch.doc_starts)
    spans = list(zip(starts, starts[1:] + [len(batch)]))
    if padded.dim() != 2 or padded.shape[0] != len(spans):
        raise ValueError(
            f"values arrived as {tuple(padded.shape)} for {len(spans)} "
            f"documents — the provider's contract is one padded row per "
            f"document of this forward")

    # credit(t) = v(t) - v(t-1) on the padded grid, then values and credit
    # re-aligned to the flat stream by doc spans
    previous = torch.cat([torch.zeros_like(padded[:, :1]),
                          padded[:, :-1]], dim=1)
    credit_rows = padded - previous
    values = torch.cat([padded[row, :stop - start]
                        for row, (start, stop) in enumerate(spans)])
    credit = torch.cat([credit_rows[row, :stop - start]
                        for row, (start, stop) in enumerate(spans)])

    # the value regression's targets: every generated position, PLUS each
    # document's prompt-end position (the token before its first generated
    # one — flatten guarantees it exists), trained toward the same episode
    # reward — which is exactly what makes v(prompt end) learn the per-task
    # baseline the credits telescope against
    value_mask = mask.clone()
    target = reward.clone()
    for start, stop in spans:
        generated = [t for t in range(start, stop) if float(mask[t]) == 1.0]
        if not generated:
            continue
        first = generated[0]
        value_mask[first - 1] = 1.0
        target[first - 1] = reward[first]

    advantage = _whitened(credit.detach().to(lp.dtype), mask)
    ratio = torch.exp(lp - behavior)
    surrogate = torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage)
    clip_objective = -(surrogate * mask).sum() / mask.sum().clamp(min=1.0)

    value_loss = (((values.to(lp.dtype) - target) ** 2) * value_mask).sum() \
        / value_mask.sum().clamp(min=1.0)

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=clip_objective + VALUE_COEF * value_loss,
                      mean_ratio=mean_ratio, logprob_gap=gap)


def _whitened(credit, mask):
    """Masked z-score over the microbatch — standard PPO whitening, and the
    identity-at-init guarantee: all-zero credit whitens to zero, so the first
    updates train only the critic."""
    import torch

    n = mask.sum().clamp(min=1.0)
    mean = (credit * mask).sum() / n
    variance = (((credit - mean) ** 2) * mask).sum() / n
    std = torch.sqrt(variance)
    if float(std) == 0.0:
        return torch.zeros_like(credit)
    return (credit - mean) / std * mask
