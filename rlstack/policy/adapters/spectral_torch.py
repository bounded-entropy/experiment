"""spectral's replay lowering: each matched site's Linear wrapped so its
forward returns inner(x) + ((x V) * eff) U^T, with eff the TOP-K-masked,
straight-through gains over that weight's own full spectrum.

The frozen half (U [out, m], V [in, m], sigma [m], m = min(out, in)) is
recomputed from the base's own weight at install — a full SVD per site, fp32,
signs canonicalized — and never ships anywhere: the engine only ever receives
the materialized top-k peft pair inside the payload, so no artifact and no
cas address exist for this adapter type.

THE STRAIGHT-THROUGH RULE (parity, I6): the forward VALUE uses exactly the
k-sparse delta the engine serves — eff is zero off the top-k — while the
BACKWARD is dense, so a direction outside the current k slots still
accumulates gradient and can compete its way in. Value equals served policy;
gradient sees the whole spectrum; both facts are this function's contract.

torch is imported at module scope — this file loads only from the adapter
type's methods (STYLE rule 7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from rlstack.policy.adapters.replay import (
    SiteWrapper, join_site, leaf_module, leave_site,
)
from rlstack.policy.adapters.spectral import GAIN_SPAN_PROVIDED
from rlstack.policy.siteschema import SiteMeta

PEFT_PREFIX = "base_model.model."
FROZEN_DTYPE = torch.bfloat16      # U and V, held for the replay delta: the
#                                    same rounding budget the factors artifact
#                                    pays, under the kernel floor logprob_gap
#                                    watches


@dataclass
class SpectralState:
    """One bank entry's spectral state: dense gains per site, and the frozen
    spectrum they steer (filled at install, from the weight itself)."""

    k: int
    seed: int
    paths: tuple[str, ...]
    delta: dict[str, torch.nn.Parameter]           # path -> [m], zero-init
    u: dict[str, torch.Tensor] = field(default_factory=dict)      # [out, m]
    v: dict[str, torch.Tensor] = field(default_factory=dict)      # [in, m]
    sigma: dict[str, torch.Tensor] = field(default_factory=dict)  # [m], fp32

    def parameters(self) -> list[torch.nn.Parameter]:
        return list(self.delta.values())


def build(sites: tuple[SiteMeta, ...], init: dict) -> SpectralState:
    """The identity element: delta = 0 at every site, so W' = W and version 0
    is the base — the same starting point every adapter type keeps. k must fit
    the NARROWEST matched site: a top-k wider than a site's spectrum is not a
    bigger policy, it is an undefined one."""
    k = int(init["k"])
    seed = int(init.get("seed", 0))
    delta: dict[str, torch.nn.Parameter] = {}
    for meta in sites:
        if meta.shape is None:
            raise ValueError(f"spectral needs a weighted site, got {meta.name}")
        d_in, d_out = meta.shape
        m = min(d_in, d_out)
        if k > m:
            raise ValueError(
                f"spectral k={k} exceeds site {meta.name}'s spectrum "
                f"(m={m}); the served rank cannot outrank the matrix")
        delta[meta.path] = torch.nn.Parameter(
            torch.zeros(m, dtype=torch.float32))
    return SpectralState(k=k, seed=seed,
                         paths=tuple(meta.path for meta in sites), delta=delta)


# ---------------------------------------------------------------------------
# the spectrum, the mask, the delta
# ---------------------------------------------------------------------------

def canonical_signs(u: torch.Tensor,
                    vh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The SVD's one freedom, fixed plora's way: each right vector's
    largest-magnitude entry positive, the matching left vector flipped with
    it — so the coordinate system is per (weight), never per run."""
    pivots = vh.gather(1, vh.abs().argmax(dim=1, keepdim=True)).squeeze(1)
    signs = torch.where(pivots < 0, -torch.ones_like(pivots),
                        torch.ones_like(pivots))
    return u * signs[None, :], vh * signs[:, None]


def full_spectrum(weight: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The COMPLETE factorization of one weight: U [out, m], sigma [m],
    V [in, m] — fp32 whatever the checkpoint dtype, signs canonical. Full
    `svd`, not the Gram trick: the whole point is that every direction is
    available, so the whole spectrum is computed."""
    matrix = weight.detach().to(dtype=torch.float32)
    u, sigma, vh = torch.linalg.svd(matrix, full_matrices=False)
    u, vh = canonical_signs(u.contiguous(), vh.contiguous())
    return u.contiguous(), sigma.contiguous(), vh.mT.contiguous()


def effective_gains(state: SpectralState, path: str) -> torch.Tensor:
    """[m]: sigma * delta, top-k by magnitude in the VALUE, dense in the
    GRADIENT — the straight-through mask, the one place it lives."""
    eff = state.sigma[path] * state.delta[path]
    picked = torch.topk(eff.abs(), min(state.k, eff.shape[0])).indices
    hard = torch.zeros_like(eff)
    hard[picked] = 1.0
    return eff * hard + (eff - eff.detach()) * (1.0 - hard)


def served_indices(state: SpectralState, path: str) -> torch.Tensor:
    """The k directions a version of this site actually ships, by magnitude —
    emit's selection, deterministic from the parameters."""
    with torch.no_grad():
        eff = state.sigma[path] * state.delta[path]
        return torch.topk(eff.abs(), min(state.k, eff.shape[0])).indices.sort().values


# ---------------------------------------------------------------------------
# the row-aware site
# ---------------------------------------------------------------------------

class SpectralSite(SiteWrapper):
    """inner(x) + ((x V) * eff) U^T — this FAMILY's link in the chain at a
    matched Linear. Chain mechanics are SiteWrapper's; the family filter is
    this class's: a row whose routed state here is not a SpectralState passes
    through to `inner`, where its own wrapper or the base Linear waits."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = self.plan.rows
        one = rows.uniform()
        if one is None:
            raise ValueError(
                f"site {self.path}: a mixed-slot spectral forward is the "
                f"coalescer's job (every verb pins one tenant today, so every "
                f"microbatch is the one-slot case)")
        state = one.get(self.path)
        if not isinstance(state, SpectralState):
            return self.inner(x)              # the transparent case
        return self.inner(x) + _whole_batch_delta(x, state, self.path).to(x.dtype)


def _whole_batch_delta(x: torch.Tensor, state: SpectralState,
                       path: str) -> torch.Tensor:
    """((x V) * eff) U^T over the whole batch, fp32: the dense-with-mask
    expression whose VALUE is the k-sparse served delta exactly."""
    eff = effective_gains(state, path)
    v = state.v[path].to(torch.float32)
    u = state.u[path].to(torch.float32)
    return ((x.to(torch.float32) @ v) * eff) @ u.T


def provide(state: SpectralState) -> dict:
    """The declared watched scalar: mean |served gain| across sites."""
    spans = [effective_gains(state, path).abs().topk(
        min(state.k, state.delta[path].shape[0])).values.mean()
        for path in state.paths]
    return {GAIN_SPAN_PROVIDED: torch.stack(spans).mean()}


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------

def install(model: torch.nn.Module, state: SpectralState) -> None:
    """Wrap the Linear once at each path, FACTOR the weight it holds (the
    full SVD, recomputed here because the trainer already holds the
    checkpoint), then ADD this state to the site. Additive (I8)."""
    for path in state.paths:
        parent, leaf = leaf_module(model, path)
        node = getattr(parent, leaf)
        while isinstance(node, SiteWrapper):
            node = node.inner
        weight = getattr(node, "weight", None)
        if weight is None:
            raise ValueError(
                f"spectral at {path}: the site's module has no `weight` to "
                f"factor — site_ok admits weighted matrices only")
        u, sigma, v = full_spectrum(weight)
        state.u[path] = u.to(weight.device, FROZEN_DTYPE)
        state.v[path] = v.to(weight.device, FROZEN_DTYPE)
        state.sigma[path] = sigma.to(weight.device)
        state.delta[path].data = state.delta[path].data.to(weight.device)
        join_site(model, path, SpectralSite, state)


def uninstall(model: torch.nn.Module, state: SpectralState) -> None:
    """install's exact inverse; the params object survives by reference."""
    for path in state.paths:
        leave_site(model, path, SpectralSite, state)


# ---------------------------------------------------------------------------
# the payload: dense gains for resume, materialized top-k for the engine
# ---------------------------------------------------------------------------

def emit(state: SpectralState) -> bytes:
    """One container, two consumers: `delta.<path>` (dense, fp32 — what load
    restores, so resume keeps every direction's accumulated movement) and the
    peft-format top-k pair (bf16 — what the engine serves). Pure function of
    the parameters; the selection is recomputed, not stored."""
    tensors: dict[str, torch.Tensor] = {}
    for path in state.paths:
        tensors[f"delta.{path}"] = state.delta[path].data.cpu()
        picked = served_indices(state, path)
        with torch.no_grad():
            eff = (state.sigma[path] * state.delta[path])[picked]
            lora_a = state.v[path][:, picked].to(torch.float32).T   # [k, in]
            lora_b = (state.u[path][:, picked].to(torch.float32)
                      * eff[None, :])                               # [out, k]
        tensors[f"peft.{PEFT_PREFIX}{path}.lora_A.weight"] = \
            lora_a.to(FROZEN_DTYPE).cpu()
        tensors[f"peft.{PEFT_PREFIX}{path}.lora_B.weight"] = \
            lora_b.to(FROZEN_DTYPE).cpu()
    head = json.dumps({"k": state.k, "paths": list(state.paths)},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")
    return len(head).to_bytes(8, "big") + head + st_save(tensors)


def unpack(payload: bytes) -> tuple[dict, dict[str, torch.Tensor]]:
    """emit's container, opened: (head, tensors)."""
    size = int.from_bytes(payload[:8], "big")
    return json.loads(payload[8:8 + size]), st_load(payload[8 + size:])


def load(state: SpectralState, payload: bytes) -> None:
    """emit's inverse, in place: the DENSE gains restore (the peft half is the
    engine's and is ignored here)."""
    head, tensors = unpack(payload)
    if int(head["k"]) != state.k:
        raise ValueError(
            f"this payload carries k={head['k']}; the entry declares "
            f"k={state.k}")
    for path in state.paths:
        state.delta[path].data.copy_(tensors[f"delta.{path}"])
