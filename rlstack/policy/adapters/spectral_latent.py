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

The prior is plora's, recipe and all: `prior="fixed"` keeps the spec's
prior_std, `prior="learned"` makes its log-scale a parameter the KL alone
moves (empirical Bayes over the latent), watched as `spectral_prior_std`.

The latent KL penalizes mu and log_std, while the generated gain scale also
depends on the decoder's heads. Two optional gain-map settings control this
scale directly:

  `amplitude="split"` — delta = a_site * unit(trunk(z) @ head^T), `a_site`
     ONE scalar per site (zero-init, its own `amplitude` optimizer group),
     `unit` the L-infinity normalization, so `a_site` IS the largest
     relative gain at the site and a change in z can only rotate the gain
     vector, never scale it. The heads are seeded-random here (a direction
     must exist at a = 0); a = 0 is the identity element either way.
  `bound=g`       — delta = g * tanh(delta / g): every relative gain served
     stays inside (-g, g), whichever recipe produced it.

Both are `init` keys, written only when set, so the recipes that predate
them canonicalize to the rows they always did. The third provide,
`spectral_gain_energy`, is what a loss that PRICES THE GAINS reads
(grpo_latent_kl_gated_priced); `spectral_gain_span` and
`spectral_amplitude` expose the scale for monitoring.

`amplitude="fixed_linear"` is a data-independent decoder control. Its
seeded random projection has unit row norms and is frozen; only the latent
mean and log standard deviation train. No mapper or amplitude parameter can
store an unpriced update. Unlike the learned-decoder recipes, its prior
draws perturb the base at version zero: each unbounded relative gain has
standard deviation prior_std. The posterior mean remains exactly the base
at initialization. It therefore needs a measured prior-predictive baseline
as well as the frozen-base baseline, and requires a fixed prior.

`amplitude="fixed_basis"` gives each site its own k latent coordinates,
directly controlling the leading k singular directions. It requires
latent = k * number of sites. This removes the random projection's shared
subspace while keeping every trainable quantity inside the Gaussian
posterior and its KL. The decoder and basis are data independent.
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
PRIOR_PROVIDED = "spectral_prior_std"     # the prior's scale: constant when
#                                           fixed, a trajectory when learned
# The served gains, watched — spectral's own span name, so the curves of the
# deterministic and the generated gains read on one axis — and priced.
GAIN_SPAN_PROVIDED = "spectral_gain_span"     # mean served |sigma * delta|
AMPLITUDE_PROVIDED = "spectral_amplitude"     # mean over sites of the largest
#                                               served RELATIVE gain
ENERGY_PROVIDED = "spectral_gain_energy"      # mean over sites and rows of
#                                               sum(served relative gain^2)

# The gain map's two recipes (the module docstring says why they exist).
JOINT_AMPLITUDE = "joint"     # delta = trunk(z) @ head^T, as first built
SPLIT_AMPLITUDE = "split"     # delta = a_site * unit(trunk(z) @ head^T)
FIXED_LINEAR = "fixed_linear"  # delta = a fixed row-normalized projection of z
FIXED_BASIS = "fixed_basis"    # each site's leading k gains are its own slice of z
AMPLITUDES = (JOINT_AMPLITUDE, SPLIT_AMPLITUDE, FIXED_LINEAR, FIXED_BASIS)


@adapter_type("spectral_latent")
class SpectralLatent(AdapterType):
    serving = Mechanism.PUNICA
    provides = frozenset({KL_PROVIDED, SIGMA_PROVIDED, PRIOR_PROVIDED,
                          GAIN_SPAN_PROVIDED, AMPLITUDE_PROVIDED,
                          ENERGY_PROVIDED})
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
        never decayed) — plora's two groups, for plora's reason — plus
        `prior` when the prior is learned, plora's third, and `amplitude`
        (the per-site scalars) under the split recipe."""
        from rlstack.policy.adapters import spectral_latent_torch
        return spectral_latent_torch.param_groups(params)

    def emit(self, params) -> bytes:
        from rlstack.policy.adapters import spectral_latent_torch
        return spectral_latent_torch.emit(params)

    def load(self, params, payload: bytes) -> None:
        from rlstack.policy.adapters import spectral_latent_torch
        spectral_latent_torch.load(params, payload)


def spectral_latent(site: str, k: int = 16, latent: int = 32, members: int = 4,
                    prior_std: float = 0.05, hidden: int = 128,
                    prior: str = "fixed", amplitude: str = JOINT_AMPLITUDE,
                    bound: float | None = None):
    """Sugar, beside spectral()/plora(): a distribution over spectral gains,
    served as `members` ordinary rank-k adapters plus the mean. `prior` is
    "fixed" (N(0, prior_std^2) as declared) or "learned" (prior_std is where
    a learned scale STARTS; the KL moves it from there). `amplitude` and
    `bound` are the gain map's recipe knobs (module docstring); at their
    defaults they are LEFT OUT of the init, so a spec written before they
    existed is the same spec."""
    from rlstack.spec.specs import AdapterSpec
    if amplitude not in AMPLITUDES:
        raise ValueError(
            f"spectral_latent amplitude must be one of {AMPLITUDES}, got "
            f"{amplitude!r}")
    if bound is not None and not bound > 0:
        raise ValueError(f"spectral_latent bound must be positive, got {bound}")
    init: dict = {"k": k, "latent": latent, "members": members,
                  "prior_std": prior_std, "hidden": hidden, "prior": prior}
    if amplitude != JOINT_AMPLITUDE:
        init["amplitude"] = amplitude
    if bound is not None:
        init["bound"] = float(bound)
    return AdapterSpec(adapter_type="spectral_latent", site=site, init=init)
