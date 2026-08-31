"""spectral_latent's replay lowering: gains GENERATED per row from that row's
recorded latent, over each site's own full spectrum.

The math both lowerings share: z = mu + exp(log_std) * eps (the recorded
noise, the current posterior — plora's reparameterization, I6), the shared
trunk reads z, one zero-initialized head per site emits that site's [m] gain
deltas, and the delta is ((x V) * eff) U^T with eff the TOP-K-masked,
straight-through gains (spectral_torch's contract: the VALUE is exactly the
k-sparse served delta, the GRADIENT is dense so every direction competes).

Zero heads + mu = 0 + log_std = log(prior_std) make version 0 the base for
EVERY latent, and KL(q||p) exactly zero — the identity element, again.

torch is imported at module scope — this file loads only from the adapter
type's methods (STYLE rule 7).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from rlstack.policy.adapters.plora_torch import Hypernet, analytic_kl
from rlstack.policy.adapters.replay import (
    ReplayRows, SiteWrapper, join_site, leaf_module, leave_site,
)
from rlstack.policy.adapters.spectral_latent import (
    EPS_RECORD, KL_PROVIDED, SIGMA_PROVIDED,
)
from rlstack.policy.adapters.spectral_torch import FROZEN_DTYPE, full_spectrum
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.seeds import derive

PEFT_PREFIX = "base_model.model."


@dataclass
class SlatentState:
    """One bank entry's latent-spectral state: a posterior over ONE latent,
    the trunk that reads it, one gains head per site, and the frozen spectrum
    (filled at install, from the weights themselves). `version` is the noise
    counter — plora's emit rule."""

    k: int
    latent: int
    members: int
    prior_std: float
    hidden: int
    seed: int
    paths: tuple[str, ...]
    mu: torch.nn.Parameter                  # [latent]
    log_std: torch.nn.Parameter             # [latent]
    trunk: Hypernet
    heads: dict[str, torch.nn.Parameter]    # path -> [m, hidden], zero-init
    u: dict[str, torch.Tensor] = field(default_factory=dict)      # [out, m]
    v: dict[str, torch.Tensor] = field(default_factory=dict)      # [in, m]
    sigma: dict[str, torch.Tensor] = field(default_factory=dict)  # [m]
    version: int = 0

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.mu, self.log_std, *self.mapper()]

    def mapper(self) -> list[torch.nn.Parameter]:
        return [*self.trunk.parameters(), *self.heads.values()]


def _draw_seed(seed: int, tag: str) -> int:
    import hashlib
    digest = hashlib.sha256(f"{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build(sites: tuple[SiteMeta, ...], init: dict) -> SlatentState:
    """The identity element, seeded: mu = 0, log_std = log(prior_std),
    heads = 0 — the base for every latent, KL exactly zero. k must fit the
    narrowest matched site, spectral's rule."""
    k, latent = int(init["k"]), int(init["latent"])
    members, hidden = int(init["members"]), int(init["hidden"])
    prior_std = float(init["prior_std"])
    seed = int(init.get("seed", 0))
    heads: dict[str, torch.nn.Parameter] = {}
    for meta in sites:
        if meta.shape is None:
            raise ValueError(
                f"spectral_latent needs a weighted site, got {meta.name}")
        m = min(meta.shape)
        if k > m:
            raise ValueError(
                f"spectral_latent k={k} exceeds site {meta.name}'s spectrum "
                f"(m={m}); the served rank cannot outrank the matrix")
        heads[meta.path] = torch.nn.Parameter(
            torch.zeros(m, hidden, dtype=torch.float32))
    return SlatentState(
        k=k, latent=latent, members=members, prior_std=prior_std,
        hidden=hidden, seed=seed, paths=tuple(meta.path for meta in sites),
        mu=torch.nn.Parameter(torch.zeros(latent, dtype=torch.float32)),
        log_std=torch.nn.Parameter(
            torch.full((latent,), math.log(prior_std), dtype=torch.float32)),
        trunk=Hypernet(latent, hidden, torch.Generator().manual_seed(
            _draw_seed(seed, "slatent.trunk"))),
        heads=heads)


# ---------------------------------------------------------------------------
# the latent, the gains, the delta — the math BOTH lowerings use
# ---------------------------------------------------------------------------

def reparameterized_latent(state: SlatentState,
                           eps: torch.Tensor) -> torch.Tensor:
    """z = mu + exp(log_std) * eps — the recorded noise composed with the
    CURRENT posterior, plora's rule for plora's reason."""
    return state.mu + torch.exp(state.log_std) * eps


def effective_gains(state: SlatentState, path: str,
                    z: torch.Tensor) -> torch.Tensor:
    """[m] (or [rows, m]): sigma * head(trunk(z)), top-k by magnitude in the
    VALUE, dense in the GRADIENT — spectral's straight-through mask, applied
    along the last dim so each row's own k directions serve."""
    delta = state.trunk(z) @ state.heads[path].T
    eff = state.sigma[path] * delta
    k = min(state.k, eff.shape[-1])
    picked = torch.topk(eff.abs(), k, dim=-1).indices
    hard = torch.zeros_like(eff)
    hard.scatter_(-1, picked, 1.0)
    return eff * hard + (eff - eff.detach()) * (1.0 - hard)


# ---------------------------------------------------------------------------
# the row-aware site
# ---------------------------------------------------------------------------

class SlatentSite(SiteWrapper):
    """inner(x) + ((x V) * eff(z_row)) U^T with the gains generated from the
    ROW's own recorded latent — this family's link in the chain."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = self.plan.rows
        one = rows.uniform()
        if one is None:
            raise ValueError(
                f"site {self.path}: a mixed-slot spectral_latent forward is "
                f"the coalescer's job (every verb pins one tenant today)")
        state = one.get(self.path)
        if not isinstance(state, SlatentState):
            return self.inner(x)              # the transparent case
        return self.inner(x) + self._delta(x, state, rows).to(x.dtype)

    def _delta(self, x: torch.Tensor, state: SlatentState,
               rows: ReplayRows) -> torch.Tensor:
        noise = _row_noise(rows, state, self.path)
        shared = _one_noise(noise)
        v = state.v[self.path].to(torch.float32)
        u = state.u[self.path].to(torch.float32)
        if shared is not None:
            eff = effective_gains(state, self.path,
                                  reparameterized_latent(state, shared))
            return ((x.to(torch.float32) @ v) * eff) @ u.T
        if x.dim() != 3 or x.shape[0] != noise.shape[0]:
            raise ValueError(
                f"site {self.path}: per-row gains need [rows, tokens, in] "
                f"activations over {noise.shape[0]} rows, got {tuple(x.shape)}")
        eff = effective_gains(state, self.path,
                              reparameterized_latent(state, noise))  # [R, m]
        return ((x.to(torch.float32) @ v) * eff[:, None, :]) @ u.T


def _row_noise(rows: ReplayRows, state: SlatentState,
               path: str) -> torch.Tensor:
    """[rows, latent]: each row's recorded draw — ONE latent per trajectory,
    plora's rule restated for this family's record name."""
    if rows.facts is None:
        raise ValueError(
            f"site {path}: a spectral_latent replay forward needs the noise "
            f"the rollout drew ({EPS_RECORD!r} in each turn's extras) and "
            f"this batch recorded none")
    return torch.tensor(
        [_one_row_noise(turns, row, state.latent, path)
         for row, turns in enumerate(rows.facts)],
        dtype=torch.float32, device=rows.index.device)


def _one_row_noise(turns: Sequence[Mapping[str, Any]], row: int, latent: int,
                   path: str) -> list[float]:
    drawn = [tuple(float(x) for x in turn[EPS_RECORD])
             for turn in turns if EPS_RECORD in turn]
    if not drawn:
        raise ValueError(
            f"site {path}: row {row} recorded no {EPS_RECORD!r} — every turn "
            f"served by a spectral_latent bundle records its member's noise")
    if len(set(drawn)) > 1:
        raise ValueError(
            f"site {path}: row {row}'s turns recorded different latents — "
            f"one trajectory is one draw")
    if len(drawn[0]) != latent:
        raise ValueError(
            f"site {path}: row {row} recorded a {len(drawn[0])}-wide latent "
            f"but this entry declares latent={latent}")
    return list(drawn[0])


def _one_noise(noise: torch.Tensor) -> torch.Tensor | None:
    if noise.shape[0] == 0:
        return None
    first = noise[0]
    return first if bool((noise == first).all()) else None


def provide(state: SlatentState) -> dict[str, Any]:
    """The latent's KL to its prior (the shared latent_kl channel a gated
    loss prices) and the posterior's mean scale, to be watched."""
    return {KL_PROVIDED: analytic_kl(state.mu, state.log_std, state.prior_std),
            SIGMA_PROVIDED: torch.exp(state.log_std).mean()}


def param_groups(state: SlatentState) -> dict[str, list]:
    return {"mapper": state.mapper(), "posterior": [state.mu, state.log_std]}


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------

def install(model: torch.nn.Module, state: SlatentState) -> None:
    """Wrap each path, FACTOR its weight (the full spectrum, recomputed from
    the checkpoint the trainer holds), place, join. Additive (I8)."""
    for path in state.paths:
        parent, leaf = leaf_module(model, path)
        node = getattr(parent, leaf)
        while isinstance(node, SiteWrapper):
            node = node.inner
        weight = getattr(node, "weight", None)
        if weight is None:
            raise ValueError(
                f"spectral_latent at {path}: the site's module has no "
                f"`weight` to factor")
        u, sigma, v = full_spectrum(weight)
        state.u[path] = u.to(weight.device, FROZEN_DTYPE)
        state.v[path] = v.to(weight.device, FROZEN_DTYPE)
        state.sigma[path] = sigma.to(weight.device)
        for parameter in state.parameters():
            parameter.data = parameter.data.to(weight.device)
        join_site(model, path, SlatentSite, state)


def uninstall(model: torch.nn.Module, state: SlatentState) -> None:
    for path in state.paths:
        leave_site(model, path, SlatentSite, state)


# ---------------------------------------------------------------------------
# the payload: posterior + heads for resume, materialized members for serving
# ---------------------------------------------------------------------------

def noise_for(seed: int, version: int, members: int,
              latent: int) -> torch.Tensor:
    """[members, latent]: the ensemble this version ships — a pure function
    of (entry seed, version, member), so a resumed run re-emits the same
    draws (plora's counter rule, this family's derive label)."""
    return torch.stack([
        torch.randn(latent, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(
                        derive(seed, "slatent", version, member)))
        for member in range(members)])


def member_peft(state: SlatentState, tag: str,
                z: torch.Tensor) -> dict[str, torch.Tensor]:
    """One member's top-k, as peft tensors under `peft.<tag>.` keys."""
    tensors: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for path in state.paths:
            eff = effective_gains(state, path, z)
            picked = eff.abs().topk(min(state.k, eff.shape[-1])).indices
            picked = picked.sort().values
            lora_a = state.v[path][:, picked].to(torch.float32).T
            lora_b = (state.u[path][:, picked].to(torch.float32)
                      * eff[picked][None, :])
            tensors[f"peft.{tag}.{PEFT_PREFIX}{path}.lora_A.weight"] = \
                lora_a.to(FROZEN_DTYPE).contiguous().cpu()
            tensors[f"peft.{tag}.{PEFT_PREFIX}{path}.lora_B.weight"] = \
                lora_b.to(FROZEN_DTYPE).contiguous().cpu()
    return tensors


def emit(state: SlatentState) -> bytes:
    """Draw this version's ensemble, materialize every member (and the
    posterior MEAN, for seedless score traffic), pack beside the trained
    half. The counter is the version — emit uses it, records it, then
    advances — so a resumed run's n-th emit ships the n-th ensemble."""
    noise = noise_for(state.seed, state.version, state.members, state.latent)
    tensors: dict[str, torch.Tensor] = {
        "posterior.mu": state.mu.data.cpu(),
        "posterior.log_std": state.log_std.data.cpu(),
        "noise": noise,
    }
    for key, value in state.trunk.state_dict().items():
        tensors[f"trunk.{key}"] = value.cpu()
    for path, head in state.heads.items():
        tensors[f"heads.{path}"] = head.data.cpu()
    with torch.no_grad():
        for index in range(state.members):
            tensors.update(member_peft(
                state, f"m{index}",
                reparameterized_latent(state, noise[index])))
        tensors.update(member_peft(state, "mean", state.mu))
    head = json.dumps({
        "k": state.k, "latent": state.latent, "members": state.members,
        "prior_std": state.prior_std, "hidden": state.hidden,
        "paths": list(state.paths), "version": state.version},
        sort_keys=True, separators=(",", ":")).encode("utf-8")
    state.version += 1
    return len(head).to_bytes(8, "big") + head + st_save(tensors)


def unpack(payload: bytes) -> tuple[dict, dict[str, torch.Tensor]]:
    size = int.from_bytes(payload[:8], "big")
    return json.loads(payload[8:8 + size]), st_load(payload[8 + size:])


def load(state: SlatentState, payload: bytes) -> None:
    """emit's inverse, in place — counter included."""
    head, tensors = unpack(payload)
    if int(head["latent"]) != state.latent or int(head["k"]) != state.k:
        raise ValueError(
            f"this payload carries k={head['k']}, latent={head['latent']}; "
            f"the entry declares k={state.k}, latent={state.latent}")
    state.mu.data.copy_(tensors["posterior.mu"])
    state.log_std.data.copy_(tensors["posterior.log_std"])
    state.trunk.load_state_dict(
        {key[len("trunk."):]: value for key, value in tensors.items()
         if key.startswith("trunk.")})
    for path, head_param in state.heads.items():
        head_param.data.copy_(tensors[f"heads.{path}"])
    state.version = int(head["version"])
