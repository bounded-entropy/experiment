"""spectral — train the SPECTRUM, not a subspace: one gain per singular
direction of each matched weight, top-k of them served.

WHY THIS SHAPE. LoRA's own analysis (and the failed frozen-top-k runs) says
fine-tuning writes into directions the weight holds QUIETLY — freezing a
frame, top-k or otherwise, decides in advance where learning may move.
spectral keeps the model's own COMPLETE coordinate system instead: every
weight W = U diag(sigma) V^T, and the trainable state is one gain per
direction,

    W' = U diag(sigma * (1 + delta)) V^T,  delta trainable, init 0

so version 0 is exactly the base, no direction is invented, and no direction
is off the table. ~min(out, in) parameters per site — the whole attention
stack of a 0.6B fits under 100k trainable parameters.

THE TOP-K TRADE, stated: the full delta U diag(sigma*delta) V^T is generically
FULL-RANK (one rank-1 term per moved gain), which punica cannot carry. So the
EFFECTIVE delta is k-sparse — only the k directions with the largest movement
|sigma_i * delta_i| apply — enforced identically in the replay forward and the
served adapter (I6/parity), with a straight-through backward so every
direction keeps receiving gradient and may compete its way into the k slots.
WHICH k directions serve is therefore learned, and changes across versions —
the frame moves with the run, unlike plora's frozen one.

Serving is plain punica: emit materializes the top-k as an ordinary rank-k
peft pair (lora_A = V_S^T, lora_B = U_S diag(eff_S)) beside the dense gains
(which ride for resume, not for the engine). No factors artifact: the frozen
U, V never leave the trainer — it recomputes them from the weight it already
holds at install, and the engine only ever sees materialized k-rank tensors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlstack.policy.adapters.base import AdapterType, Mechanism, adapter_type
from rlstack.policy.siteschema import SiteMeta

# The provided tensor, recomputed each forward: the mean |sigma * delta| over
# the SERVED (top-k) directions — how hard the spectrum is being steered.
# Nothing requires it; it exists to be watched (a run whose span explodes is
# rewriting the base, one whose span pins at 0 has a dead adapter).
GAIN_SPAN_PROVIDED = "spectral_gain_span"


@adapter_type("spectral")
class Spectral(AdapterType):
    serving = Mechanism.PUNICA
    provides = frozenset({GAIN_SPAN_PROVIDED})

    def site_ok(self, meta: SiteMeta) -> bool:
        """A weighted matrix: the gains ARE that matrix's own spectrum."""
        return meta.has_weight

    # compute halves — torch/vLLM load lazily, from here only (rule 7)

    def rollout_lowering(self, build):
        from rlstack.policy.adapters import spectral_vllm
        return spectral_vllm.SpectralRollout(build)

    def params(self, sites: tuple[SiteMeta, ...], init: dict):
        from rlstack.policy.adapters import spectral_torch
        return spectral_torch.build(sites, init)

    def install_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import spectral_torch
        spectral_torch.install(model, params)

    def uninstall_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import spectral_torch
        spectral_torch.uninstall(model, params)

    def provide(self, params) -> Mapping[str, Any]:
        from rlstack.policy.adapters import spectral_torch
        return spectral_torch.provide(params)

    def emit(self, params) -> bytes:
        from rlstack.policy.adapters import spectral_torch
        return spectral_torch.emit(params)

    def load(self, params, payload: bytes) -> None:
        from rlstack.policy.adapters import spectral_torch
        spectral_torch.load(params, payload)


def spectral(site: str, k: int = 16):
    """Sugar, beside lora()/plora(): gains over the full spectrum of every
    matched weight, the top-k of them served."""
    from rlstack.spec.specs import AdapterSpec
    return AdapterSpec(adapter_type="spectral", site=site, init={"k": k})
