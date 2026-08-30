"""grpo_latent_kl with the prior's pull EARNED: accuracy first, then the KL.

The ungated loss prices the latent's KL from update one, so the prior tugs at
mu and log_std while the policy is still learning to be right. This variant
makes the ordering explicit — the arc-agi-ttt gate, ported: a group whose
completions are ALL correct has an all-tie, all-zero advantage and nothing
left to teach the policy, so that group's share of the update is handed to the
prior instead. While nothing is solved the KL is silent; once everything is,
the surrogate's gradient is zero and the KL is the only thing still moving the
latent.

The gate is a COLUMN, not a branch in the trainer (I9): `group_accuracy`
writes each row's group-mean reward, and this loss reads it — pure math over
named columns, like every other loss.

Still pure math: `plora_kl` is a PROVIDED tensor, recomputed by the same
forward that produced the logprobs, so requiring it plans no work and touches
no metal.
"""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs
from rlstack.training.losses.grpo import grpo
from rlstack.training.losses.grpo_latent_kl import BETA

# What "solved" means: the group's mean reward at this value opens the gate
# for its rows. A MODULE CONSTANT like BETA, and for the same reason: the
# threshold is the objective, so sweeping it must produce a different run_id.
FULL_ACCURACY = 1.0


@loss("grpo_latent_kl_gated",
      requires=("advantage", "plora_kl", "accuracy"))
def grpo_latent_kl_gated(out: PolicyOutputs, batch: Any,
                         clip_eps: float = 0.2) -> LossResult:
    """The grpo surrogate plus BETA * KL(q||p) * the SOLVED SHARE of the batch.

    The gate is the masked fraction of tokens whose group reached
    FULL_ACCURACY, so the KL enters in proportion to how much of the update is
    already solved. The endpoints are exact regardless of how `pack` split the
    wave: no group solved means no KL at all, every group solved means the
    full BETA — and between them the effective beta is a wave statistic (the
    per-microbatch solved fractions, averaged), which is the price of gating a
    per-update term by a per-token column. Divided by the update's microbatch
    count for the same reason grpo_latent_kl divides: the KL is a function of
    the parameters alone, and a wave is one gradient update (#59).
    """
    import torch

    surrogate = grpo(out, batch, clip_eps)
    kl = out.provided["plora_kl"]
    mask = torch.tensor(batch.loss_mask, dtype=kl.dtype, device=kl.device)
    accuracy = torch.tensor(batch.postdata["accuracy"], dtype=kl.dtype,
                            device=kl.device)
    solved = (accuracy >= FULL_ACCURACY).to(kl.dtype)
    gate = (solved * mask).sum() / mask.sum().clamp(min=1.0)
    return LossResult(
        loss=surrogate.loss + gate * BETA * kl / batch.microbatches_in_update,
        mean_ratio=surrogate.mean_ratio,
        logprob_gap=surrogate.logprob_gap)
