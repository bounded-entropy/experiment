"""spectral_latent — SVF with the probability put back: the policy is a
DISTRIBUTION over spectral gains.

spectral's twin, and the campaign's ablation pair: plain `spectral` trains
one deterministic gain vector per site; this adapter GENERATES the gains
from a latent draw — z ~ N(mu, diag(exp(log_std)^2)) against a
N(0, prior_std^2) prior, a shared hypernet trunk, one zero-initialized head
per site emitting that site's [m] gain deltas — so everything plora's latent
machinery buys (the KL a gated loss can price, the served ensemble, the
recorded draw) is available over the SPECTRUM parameterization instead of a
frozen k x k core. If the latent buys nothing here, the plain arm will say
so; that question is the reason both exist.

Serving is plora's move on spectral's payload: each of `members` draws (plus
the posterior MEAN, for seedless score traffic) materializes into an
ordinary rank-k peft adapter — its own top-k by |sigma * delta(z)| — so the
engine sees only punica. The rollout records the draw (`slatent_eps`,
`slatent_member`); replay recomputes z from the CURRENT posterior and the
recorded noise, exactly plora's reparameterization contract (I6).

The frozen spectrum (U, sigma, V per site) is recomputed at install from the
weight the trainer already holds and never ships — no factors artifact, as
with plain spectral.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlstack.policy.adapters.base import AdapterType, Mechanism, adapter_type
from rlstack.policy.siteschema import SiteMeta

# The recorded facts — sampling-time truth, frozen at the seal.
EPS_RECORD = "slatent_eps"          # the noise this request's member used
MEMBER_RECORD = "slatent_member"    # which ensemble member served it

# The provided tensors — recomputed by every training forward. latent_kl is
# the SHARED latent-KL channel (plora provides the same name), which is what
# lets one latent-KL loss price either adapter unchanged.
KL_PROVIDED = "latent_kl"
SIGMA_PROVIDED = "spectral_sigma_mean"    # posterior scale: here to be WATCHED


@adapter_type("spectral_latent")
class SpectralLatent(AdapterType):
    serving = Mechanism.PUNICA
    provides = frozenset({KL_PROVIDED, SIGMA_PROVIDED})
    records = (EPS_RECORD, MEMBER_RECORD)

    def site_ok(self, meta: SiteMeta) -> bool:
        """A weighted matrix: the gains ARE that matrix's own spectrum."""
        return meta.has_weight

    # compute halves — torch/vLLM load lazily, from here only (rule 7)

    def rollout_lowering(self, build):
        from rlstack.policy.adapters import spectral_latent_vllm
        return spectral_latent_vllm.SlatentRollout(build)

    def params(self, sites: tuple[SiteMeta, ...], init: dict):
        from rlstack.policy.adapters import spectral_latent_torch
        return spectral_latent_torch.build(sites, init)

    def install_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import spectral_latent_torch
        spectral_latent_torch.install(model, params)

    def uninstall_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import spectral_latent_torch
        spectral_latent_torch.uninstall(model, params)

    def provide(self, params) -> Mapping[str, Any]:
        from rlstack.policy.adapters import spectral_latent_torch
        return spectral_latent_torch.provide(params)

    def param_groups(self, params) -> Mapping[str, list]:
        """`mapper` (trunk + heads, decayable) and `posterior` (mu, log_std,
        never decayed) — plora's two groups, for plora's reason."""
        from rlstack.policy.adapters import spectral_latent_torch
        return spectral_latent_torch.param_groups(params)

    def emit(self, params) -> bytes:
        from rlstack.policy.adapters import spectral_latent_torch
        return spectral_latent_torch.emit(params)

    def load(self, params, payload: bytes) -> None:
        from rlstack.policy.adapters import spectral_latent_torch
        spectral_latent_torch.load(params, payload)


def spectral_latent(site: str, k: int = 16, latent: int = 32, members: int = 4,
                    prior_std: float = 0.05, hidden: int = 128):
    """Sugar, beside spectral()/plora(): a distribution over spectral gains,
    served as `members` ordinary rank-k adapters plus the mean."""
    from rlstack.spec.specs import AdapterSpec
    return AdapterSpec(adapter_type="spectral_latent", site=site, init={
        "k": k, "latent": latent, "members": members,
        "prior_std": prior_std, "hidden": hidden})
