"""Reverse PPO: the clip surrogate over HINDSIGHT-decomposed credit.

WHY THE CLASSIC CRITIC FAILS HERE, and what this does instead. A prefix
critic must predict the outcome before it happened, and with one reward per
episode its errors dominate the advantage exactly where credit matters. The
reverse_value_head reads the other direction — v(t) attends to the SUFFIX,
what the policy actually went on to do, under a distance-to-end positional
code — so it cannot be an unbiased baseline (it has seen the ending) and is
not used as one. It is used as a RETURN DECOMPOSITION (the RUDDER idea):

    credit(t) = v(t) - v(t+1),   v(end) = 0  =>  sum_t credit(t) = v(first)

Once the critic fits, v(first) is the episode's reward, so the credits are a
conserved reallocation of the whole return — concentrated at the tokens where
the suffix's prospects CHANGED (the call that went wrong, the drawer that was
right), which is precisely the propagation a terminal reward cannot give.

The pieces: a value regression pulls v(t) toward the episode reward at every
generated position (the suffix makes that a fitting problem, not a guessing
one); the policy pays the PPO clip surrogate over the DETACHED, batch-whitened
credits. Values arrive as a PROVIDED tensor — [rows, padded_tokens] from the
same forward, requiring it plans no work (I9) — and this loss re-aligns it to
the flat token stream by doc spans, the padding convention stated on the
provider."""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs, rails, token_tensors

VALUE_COEF = 0.5     # the critic's share of the objective — a module constant
#                      for BETA's reason: it is the objective, so sweeping it
#                      must produce a different run_id.


@loss("reverse_ppo", requires=("reward", "reverse_values"))
def reverse_ppo(out: PolicyOutputs, batch: Any,
                clip_eps: float = 0.2) -> LossResult:
    import torch

    lp, mask, behavior = token_tensors(out, batch)
    reward = torch.tensor(batch.postdata["reward"], dtype=lp.dtype,
                          device=lp.device)
    padded = out.provided["reverse_values"]            # [rows, W], 0 at padding
    starts = list(batch.doc_starts)
    spans = list(zip(starts, starts[1:] + [len(batch)]))
    if padded.dim() != 2 or padded.shape[0] != len(spans):
        raise ValueError(
            f"reverse_values arrived as {tuple(padded.shape)} for "
            f"{len(spans)} documents — the provider's contract is one padded "
            f"row per document of this forward")

    # credit(t) = v(t) - v(t+1) on the padded grid (v past each row's end is
    # zero by the provider's contract), then both v and credit re-aligned to
    # the flat stream by doc spans
    shifted = torch.cat([padded[:, 1:],
                         torch.zeros_like(padded[:, :1])], dim=1)
    credit_rows = padded - shifted
    values = torch.cat([padded[row, :stop - start]
                        for row, (start, stop) in enumerate(spans)])
    credit = torch.cat([credit_rows[row, :stop - start]
                        for row, (start, stop) in enumerate(spans)])

    advantage = _whitened(credit.detach().to(lp.dtype), mask)
    ratio = torch.exp(lp - behavior)
    surrogate = torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage)
    n = mask.sum().clamp(min=1.0)
    clip_objective = -(surrogate * mask).sum() / n

    value_loss = (((values.to(lp.dtype) - reward) ** 2) * mask).sum() / n

    mean_ratio, gap = rails(lp, mask, behavior)
    return LossResult(loss=clip_objective + VALUE_COEF * value_loss,
                      mean_ratio=mean_ratio, logprob_gap=gap)


def _whitened(credit, mask):
    """Masked z-score over the microbatch — the standard PPO whitening, and
    the identity-at-init guarantee: a zero-initialized head gives all-zero
    credit, which whitens to zero, so the first updates train only the
    critic."""
    import torch

    n = mask.sum().clamp(min=1.0)
    mean = (credit * mask).sum() / n
    variance = (((credit - mean) ** 2) * mask).sum() / n
    std = torch.sqrt(variance)
    if float(std) == 0.0:
        return torch.zeros_like(credit)
    return (credit - mean) / std * mask
