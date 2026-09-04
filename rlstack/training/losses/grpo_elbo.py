"""GRPO as the evidence bound: the surrogate plus the latent's KL, priced ONCE
PER OBSERVATION and at no temperature — the objective an EMPIRICAL-BAYES
adapter fits its posterior AND its prior with.

grpo_latent_kl adds BETA * KL to the surrogate with the prior where the spec
put it, and BETA is a scale nobody derived. This loss is the negative ELBO of
the RL-as-inference reading instead: one trajectory is one observation, the
grpo surrogate stands in for its log-likelihood, and the KL(q(z) || p(z))
counts once against the whole update's observations —

    -ELBO  =  -sum_traj  surrogate(traj)  +  KL(q || p)

— so the only thing that sets how hard the prior pulls is HOW MUCH DATA the
update holds, which is the bound's own answer and not a knob. With
prior="learned" the KL is also the prior's whole gradient, and its fixed
point is the empirical-Bayes one (prior_std^2 = the posterior's mean second
moment; plora_torch.analytic_kl), so the prior widens to wherever the
posterior actually lives and the KL is left pricing the posterior's
CONCENTRATION. With a fixed prior this is the same bound at temperature one.

THE SCALING, exactly. The learner SUMS the update's microbatch losses into one
step (#59), and grpo's surrogate is a per-microbatch token mean: over M
equal microbatches of an update holding D documents that sum is (M/D) times
the per-trajectory sum above. Adding KL / D in every microbatch therefore
lands the KL at exactly the same (M/D) scale — the bound, whole, once per
update, however `pack` split the wave. `documents_in_update` is the stamp
pack writes for this; a hand-built batch is its own update.

Still pure math (I9): `latent_kl` is a PROVIDED tensor, recomputed by the same
forward that produced the logprobs, so requiring it plans no work and touches
no metal.
"""

from __future__ import annotations

from typing import Any

from rlstack.registry import loss
from rlstack.training.losses.base import LossResult, PolicyOutputs
from rlstack.training.losses.grpo import grpo


@loss("grpo_elbo", requires=("advantage", "latent_kl"))
def grpo_elbo(out: PolicyOutputs, batch: Any,
              clip_eps: float = 0.2) -> LossResult:
    """The grpo surrogate plus KL(q||p) / documents_in_update: the negative
    ELBO with one trajectory as one observation, no beta."""
    surrogate = grpo(out, batch, clip_eps)
    kl = out.provided["latent_kl"]
    return LossResult(
        loss=surrogate.loss + kl / batch.documents_in_update,
        mean_ratio=surrogate.mean_ratio,
        logprob_gap=surrogate.logprob_gap)
