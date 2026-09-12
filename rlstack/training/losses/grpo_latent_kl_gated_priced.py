"""grpo_latent_kl_gated with THE SERVED GAINS PRICED: the gated objective plus
GAMMA times the adapter's gain energy.

The latent KL penalizes the posterior but does not bound the gain map's
heads. This loss also reads `spectral_gain_energy`: the mean over sites and
rows of the served relative gains' squared sum, computed by the forward that
produced the logprobs. The additional term penalizes the generated gains.

Per microbatch, like the surrogate and unlike the KL: the energy is a
statistic of the ROWS (each row's own latent draw generates its own gains),
so it is a per-microbatch mean already and needs no division by the update's
microbatch count.

Still pure math (I9): `spectral_gain_energy` is a PROVIDED tensor, so
requiring it plans no work and touches no metal.
"""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs
from rlstack.training.losses.grpo_latent_kl_gated import grpo_latent_kl_gated

# The loss source, including this coefficient, hashes into run identity.
GAMMA = 0.1

# The provide this loss reads — the adapter's name for it
# (rlstack.policy.adapters.spectral_latent.ENERGY_PROVIDED), spelled here so
# training/ names a column and imports nothing of policy/.
ENERGY = "spectral_gain_energy"


@loss("grpo_latent_kl_gated_priced",
      requires=("advantage", "latent_kl", "accuracy", ENERGY))
def grpo_latent_kl_gated_priced(out: PolicyOutputs, batch: Any,
                                clip_eps: float = 0.2) -> LossResult:
    """The gated objective, verbatim (called, not restated), plus GAMMA times
    the served gains' energy."""
    gated = grpo_latent_kl_gated(out, batch, clip_eps)
    return LossResult(
        loss=gated.loss + GAMMA * out.provided[ENERGY],
        mean_ratio=gated.mean_ratio,
        logprob_gap=gated.logprob_gap)
