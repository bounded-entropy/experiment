"""GRPO with the latent's KL priced: the objective a probabilistic adapter
needs, and nothing more.

grpo optimizes the policy; this optimizes the DISTRIBUTION over policies. The
surrogate is grpo's, verbatim — it is called, not restated, so the two can never
drift — plus BETA times the posterior's KL to its prior, which is what stops
the latent from collapsing to a point (and turning plora back into a plain
LoRA) or wandering away from a prior nothing ever chose.

Still pure math (I9): `plora_kl` is a PROVIDED tensor, recomputed by the same
forward that produced the logprobs, so requiring it plans no work and touches
no metal.
"""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs
from rlstack.training.losses.grpo import grpo

# How hard the prior pulls. A MODULE CONSTANT, deliberately: the loss's source
# hashes into run_id (I3), so sweeping this is an edit that produces a different
# experiment — which is exactly what a different beta is. A knob on the spec
# would let two runs of one run_id disagree about the objective.
BETA = 1e-3


@loss("grpo_latent_kl", requires=("advantage", "plora_kl"))
def grpo_latent_kl(out: PolicyOutputs, batch: Any,
                   clip_eps: float = 0.2) -> LossResult:
    """The grpo surrogate plus BETA * KL(q||p), counted ONCE per update.

    The KL is a function of the parameters alone, so every microbatch of a wave
    computes the same number — and a wave is one gradient update (#59), whose
    microbatches accumulate into a single step. Adding it whole to each would
    therefore multiply it by the number of microbatches, making the effective
    beta depend on `microbatch_tokens` — an engineering knob silently changing
    the objective. Dividing by the update's microbatch count is what keeps beta
    meaning what it says.
    """
    surrogate = grpo(out, batch, clip_eps)
    kl = out.provided["plora_kl"]
    return LossResult(
        loss=surrogate.loss + BETA * kl / batch.microbatches_in_update,
        mean_ratio=surrogate.mean_ratio,
        logprob_gap=surrogate.logprob_gap)
