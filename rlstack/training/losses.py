"""Built-in losses (training world).

A loss is microbatch-scope and differentiable: pure fn(PolicyOutputs,
TokenBatch) -> LossResult, with declared `requires` naming the postdata
columns and planned passes its math reads. The learner runs the forward and
builds PolicyOutputs; the loss owns only the objective. torch is imported
inside the body (rule 7): declarations validate without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlstack.registry import loss


@dataclass
class PolicyOutputs:
    """What the training forward produced for one microbatch, token-aligned
    with the TokenBatch: logprobs[t] is the trainer's logprob of token t
    given its prefix (0.0 at injected positions — masked out anyway)."""

    logprobs: Any                 # torch.Tensor [T], grad flows through it


@dataclass
class LossResult:
    """The objective plus the rails every loss must report."""

    loss: Any                     # scalar torch.Tensor, ready to backward
    mean_ratio: float             # masked mean of exp(lp - behavior_lp)
    logprob_gap: float            # masked mean |lp - behavior_lp| — the
                                  # silent-off-policy / parity alarm


def _tensors(out: PolicyOutputs, batch: Any):
    """The three token-aligned tensors every loss starts from: trainer
    logprobs, the mask over generated tokens, RECORDED behavior logprobs."""
    import torch

    lp = out.logprobs
    mask = torch.tensor(batch.loss_mask, dtype=lp.dtype, device=lp.device)
    behavior = torch.tensor(batch.behavior_logprobs, dtype=lp.dtype,
                            device=lp.device)
    return lp, mask, behavior


def _rails(lp, mask, behavior) -> tuple[float, float]:
    """The two rails every loss reports: masked mean IS ratio, and the masked
    mean |trainer − behavior| logprob gap (the silent-off-policy alarm)."""
    import torch

    with torch.no_grad():
        n = mask.sum().clamp(min=1.0)
        mean_ratio = float((torch.exp(lp - behavior) * mask).sum() / n)
        gap = float(((lp - behavior).abs() * mask).sum() / n)
    return mean_ratio, gap


@loss("grpo", requires=("advantage",))
def grpo(out: PolicyOutputs, batch: Any, clip_eps: float = 0.2) -> LossResult:
    """Token-level PPO-clip surrogate over batch.post["advantage"], with the
    IS ratio against the RECORDED behavior logprobs (I6: never recomputed)."""
    import torch

    lp, mask, behavior = _tensors(out, batch)
    advantage = torch.tensor(batch.post["advantage"], dtype=lp.dtype,
                             device=lp.device)

    ratio = torch.exp(lp - behavior)
    surrogate = torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage)
    objective = -(surrogate * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = _rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)


@loss("ppo", requires=("advantage",))
def ppo(out: PolicyOutputs, batch: Any, clip_eps: float = 0.2) -> LossResult:
    """Token-level PPO-clip over an UNNORMALIZED advantage (pair with
    center_reward: mean-baseline, no variance rescaling — the classic
    value-free PPO estimator; grpo differs only by its z-scored input)."""
    import torch

    lp, mask, behavior = _tensors(out, batch)
    advantage = torch.tensor(batch.post["advantage"], dtype=lp.dtype,
                             device=lp.device)

    ratio = torch.exp(lp - behavior)
    surrogate = torch.minimum(
        ratio * advantage,
        torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage)
    objective = -(surrogate * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = _rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)


@loss("gspo", requires=("advantage",))
def gspo(out: PolicyOutputs, batch: Any, clip_eps: float = 0.1) -> LossResult:
    """Sequence-level ratios (GSPO): one length-normalized IS ratio PER DOC —
    exp(mean over generated tokens of lp − behavior) — clipped and weighted by
    the doc's advantage (constant across its tokens by broadcast). The unit of
    importance correction is the sequence, not the token."""
    import torch

    lp, mask, behavior = _tensors(out, batch)
    advantage = torch.tensor(batch.post["advantage"], dtype=lp.dtype,
                             device=lp.device)

    starts = list(batch.doc_starts) + [len(batch)]
    per_doc = []
    for start, stop in zip(starts, starts[1:]):
        doc_mask = mask[start:stop]
        n = doc_mask.sum().clamp(min=1.0)
        seq_ratio = torch.exp(
            ((lp[start:stop] - behavior[start:stop]) * doc_mask).sum() / n)
        doc_advantage = (advantage[start:stop] * doc_mask).sum() / n
        per_doc.append(torch.minimum(
            seq_ratio * doc_advantage,
            torch.clamp(seq_ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            * doc_advantage))
    objective = -torch.stack(per_doc).mean()

    mean_ratio, gap = _rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)


@loss("sft")
def sft(out: PolicyOutputs, batch: Any) -> LossResult:
    """Behavior cloning on the sealed record: maximize the trainer's logprob
    of every generated token. No importance correction, no advantage — the
    data source (a static cas:// set, a replayed run) IS the curriculum."""
    lp, mask, behavior = _tensors(out, batch)
    objective = -(lp * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = _rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)


@loss("sdft", requires=("reward",))
def sdft(out: PolicyOutputs, batch: Any) -> LossResult:
    """Self-distillation fine-tuning, v0: reward-weighted behavior cloning on
    the policy's own samples — clone only what the pipeline scored (rejection
    sampling as a loss). Requires the "reward" column, weights each token by
    its trajectory's reward; an all-zero-reward microbatch contributes zero."""
    import torch

    lp, mask, behavior = _tensors(out, batch)
    weight = torch.tensor(batch.post["reward"], dtype=lp.dtype,
                          device=lp.device) * mask
    objective = -(lp * weight).sum() / weight.sum().clamp(min=1.0)

    mean_ratio, gap = _rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)


@loss("opd")
def opd(out: PolicyOutputs, batch: Any) -> LossResult:
    """Off-policy distillation, v0: match the teacher's RECORDED confidence on
    its own sampled tokens — squared error between trainer and behavior
    logprobs. The teacher is whatever policy sealed the replayed run (I6: its
    logprobs are read from the record, never recomputed)."""
    lp, mask, behavior = _tensors(out, batch)
    objective = (((lp - behavior) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = _rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)


@loss("opsd")
def opsd(out: PolicyOutputs, batch: Any) -> LossResult:
    """On-policy self-distillation, v0: the same recorded-confidence matching
    as opd, but on LIVE data under max_policy_lag > 0 — the behavior policy is
    the trainer's own LAGGED self, so the loss anchors the current weights to
    the version that sampled (an EMA-teacher without a second model)."""
    lp, mask, behavior = _tensors(out, batch)
    objective = (((lp - behavior) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)

    mean_ratio, gap = _rails(lp, mask, behavior)
    return LossResult(loss=objective, mean_ratio=mean_ratio, logprob_gap=gap)
