"""plora — a PROBABILISTIC low-rank delta: the policy is a distribution over
adapters, and the run learns the distribution.

Each matched weight M is factored once into its top-k singular directions,
M ~ U_k Sigma_k V_k^T, and frozen there: A = Sigma_k V_k^T and U_k never move
again. What trains is a small hypernet that maps a LATENT z to a k x k core
C_s per site, so the delta is Delta_s = U_k C_s A — exactly a rank-k LoRA whose
lora_A is frozen and identical at every version and whose lora_B is U_k C_s.
That is what lets the whole thing be served through punica with no new
mechanism: the ensemble is E ordinary peft adapters, materialized at attach.

The latent is where the probability lives: q = N(mu, diag(exp(log_std)^2))
against a prior N(0, prior_std^2 I), with mu and log_std trainable and
initialized so that KL(q||p) is EXACTLY zero and every core is EXACTLY zero
(the heads are zero-initialized, ControlNet-style). Version 0 is therefore the
base model, whatever z is drawn — the same identity element lora has, kept for
the same reason.

The prior's scale is the run's to choose, or the run's to LEARN. `prior="fixed"`
keeps N(0, prior_std^2 I) where the spec put it; `prior="learned"` makes
log(prior_std) a parameter of its own (init at the spec's prior_std, small),
moved only by the KL — empirical Bayes over the latent: the prior widens to
wherever the posterior's mass actually sits, so the KL prices the posterior's
CONCENTRATION rather than its distance from a scale nobody chose. Its current
value is a provided tensor (`plora_prior_std`), so the ledger watches it move.

The two halves of the recording contract meet here. At rollout the engine draws
a member and RECORDS the noise that made it (`plora_eps`, `plora_member`);
at replay the trainer recomputes z = mu + sigma * eps with the CURRENT
posterior, so the reparameterized gradient reaches mu and log_std through a
draw that already happened (I6). Recording z instead would freeze the posterior
out of its own gradient.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlstack.policy.adapters.base import AdapterType, Mechanism, adapter_type
from rlstack.policy.siteschema import SiteMeta

# The recorded facts, by name — sampling-time truth, frozen at the seal.
EPS_RECORD = "plora_eps"          # the noise vector this request's member used
MEMBER_RECORD = "plora_member"    # which ensemble member served it

# The provided tensors, by name — recomputed by every training forward.
KL_PROVIDED = "latent_kl"                  # KL(q||p), what a latent-KL loss prices
SIGMA_PROVIDED = "plora_sigma_mean"       # mean posterior scale: nothing consumes
#                                           it; it is here to be WATCHED
PRIOR_PROVIDED = "plora_prior_std"        # the prior's scale — a constant when
#                                           fixed, a trajectory when learned

# The prior's two recipes, by name: the spec's init says which (validated at
# the gate), the torch half builds the matching tensor.
FIXED_PRIOR = "fixed"
LEARNED_PRIOR = "learned"
PRIORS = (FIXED_PRIOR, LEARNED_PRIOR)


@adapter_type("plora")
class Plora(AdapterType):
    serving = Mechanism.PUNICA
    provides = frozenset({KL_PROVIDED, SIGMA_PROVIDED, PRIOR_PROVIDED})
    records = (EPS_RECORD, MEMBER_RECORD)

    def site_ok(self, meta: SiteMeta) -> bool:
        """A weighted matrix, because the frozen half IS that matrix's own
        singular directions — there is nothing to factor at a boundary."""
        return meta.has_weight

    # compute halves — plora_torch imports torch and plora_vllm imports vLLM,
    # so both load lazily, from here only (rule 7)

    def rollout_lowering(self, build):
        from rlstack.policy.adapters import plora_vllm
        return plora_vllm.PloraRollout(build)

    def params(self, sites: tuple[SiteMeta, ...], init: dict):
        from rlstack.policy.adapters import plora_torch
        return plora_torch.build(sites, init)

    def install_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import plora_torch
        plora_torch.install(model, params)

    def uninstall_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import plora_torch
        plora_torch.uninstall(model, params)

    def provide(self, params) -> Mapping[str, Any]:
        """The latent's KL to its prior, the posterior's mean scale, and the
        prior's scale.

        The KL is a pure function of (mu, log_std, prior_log_std) — no data, no
        batch — which is exactly why it is a PROVIDED tensor rather than a
        postdata column: it is recomputed by the forward and differentiable,
        and a loss that wants to price it just requires it by name. The other
        two are required by nothing and exist to be watched: a posterior
        collapsing to zero is this adapter type turning back into a plain
        LoRA, and a learned prior's scale is the empirical-Bayes answer to
        "how wide is the distribution over adapters" — both should be visible
        in the ledger the update they start moving.
        """
        from rlstack.policy.adapters import plora_torch
        return plora_torch.provide(params)

    def param_groups(self, params) -> Mapping[str, list]:
        """Two groups, because they want opposite treatment: `mapper` (the
        hypernet trunk and the per-site heads) is an ordinary network that may
        be weight-decayed, while `posterior` (mu, log_std) is a distribution's
        parameters, where decay would be an unstated second prior pulling
        log_std toward zero — a scale of 1, which is nothing anyone meant. A
        LEARNED prior is a third group, `prior` (its log-scale alone), so its
        learning rate can be its own: one scalar chased by the KL alone has no
        reason to move at the hypernet's pace."""
        from rlstack.policy.adapters import plora_torch
        return plora_torch.param_groups(params)

    def emit(self, params) -> bytes:
        from rlstack.policy.adapters import plora_torch
        return plora_torch.emit(params)

    def load(self, params, payload: bytes) -> None:
        from rlstack.policy.adapters import plora_torch
        plora_torch.load(params, payload)
