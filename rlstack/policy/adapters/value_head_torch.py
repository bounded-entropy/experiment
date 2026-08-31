"""value_head's replay lowering: capture the boundary, value every prefix.

Two captures feed one tiny computation. The SITE wrapper at the boundary path
stores each forward's output — DETACHED, so nothing the head learns can leak
gradient back into the trunk or the policy's own adapters — and a forward
pre-hook on the model stores the same forward's attention mask, the only
honest source of row lengths on a padded forward. `provide` then runs the
head per position: v = w_out · silu(W_in · rmsnorm(h_t)), with w_out
zero-initialized so values are exactly zero at version 0.

The trunk state h_t is causal — it already encodes the prompt and the
completion up to t — so no attention, no positional table, nothing but a
probe is needed here: the PREFIX is the conditioning, by construction.

torch is imported at module scope — this file loads only from the adapter
type's methods (STYLE rule 7).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from rlstack.policy.adapters.replay import SiteWrapper, join_site, leave_site
from rlstack.policy.adapters.value_head import VALUES_PROVIDED
from rlstack.policy.siteschema import SiteMeta

RMS_EPS = 1e-6


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Pre-norm without a gain, plora's rule for plora's reason."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)


@dataclass
class VhState:
    """One bank entry's value head: two weights, and the capture buffers the
    wrappers fill per forward."""

    d_model: int
    hidden: int
    seed: int
    path: str
    w_in: torch.nn.Parameter               # [hidden, d_model], drawn
    w_out: torch.nn.Parameter              # [hidden], zero-init: v == 0 at v0
    captured: torch.Tensor | None = None   # [rows, W, d], detached
    padding: torch.Tensor | None = None    # [rows, W] mask, or None
    hook: Any = field(default=None, repr=False)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.w_in, self.w_out]


def _draw_seed(seed: int, path: str) -> int:
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build(sites: tuple[SiteMeta, ...], init: dict) -> VhState:
    """ONE boundary: a value is read at one place, and two would be two heads
    fighting over one provided name."""
    if len(sites) != 1:
        raise ValueError(
            f"value_head reads ONE boundary, got {len(sites)} sites "
            f"({[m.name for m in sites]}) — narrow the site pattern")
    d_model, hidden = int(init["d_model"]), int(init.get("hidden", 128))
    seed = int(init.get("seed", 0))
    generator = torch.Generator().manual_seed(_draw_seed(seed, "vh.probe"))
    return VhState(
        d_model=d_model, hidden=hidden, seed=seed, path=sites[0].path,
        w_in=torch.nn.Parameter(
            torch.randn(hidden, d_model, generator=generator,
                        dtype=torch.float32) / math.sqrt(d_model)),
        w_out=torch.nn.Parameter(torch.zeros(hidden, dtype=torch.float32)))


def values(state: VhState) -> torch.Tensor:
    """[rows, W] prefix values for the captured forward, zero at padding."""
    if state.captured is None:
        return torch.zeros(0)
    h = state.captured.to(torch.float32)
    mask = (state.padding.to(h.device, torch.float32)
            if state.padding is not None
            else torch.ones(h.shape[:2], device=h.device))
    v = torch.nn.functional.silu(rms_norm(h) @ state.w_in.T) @ state.w_out
    return v * mask


def provide(state: VhState) -> dict[str, Any]:
    return {VALUES_PROVIDED: values(state)}


class VhSite(SiteWrapper):
    """The boundary tap: pass the module's output through untouched, keep a
    DETACHED copy for every installed head. Transparent to every other
    tenant's forward — a capture changes no number."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        for state in self.installed:
            state.captured = out.detach()
        return out


def install(model: torch.nn.Module, state: VhState) -> None:
    """Tap the boundary; hook the model's own forward for the attention mask
    (the padded forward's row lengths, which the boundary tensor alone cannot
    reveal). Placement follows the model's device."""
    join_site(model, state.path, VhSite, state)
    device = next(model.parameters()).device
    for parameter in state.parameters():
        parameter.data = parameter.data.to(device)

    def catch_mask(module, args, kwargs):
        state.padding = kwargs.get("attention_mask")
        return None

    state.hook = model.register_forward_pre_hook(catch_mask, with_kwargs=True)


def uninstall(model: torch.nn.Module, state: VhState) -> None:
    """install's exact inverse: untap the boundary, drop the mask hook."""
    leave_site(model, state.path, VhSite, state)
    if state.hook is not None:
        state.hook.remove()
        state.hook = None


def emit(state: VhState) -> bytes:
    """A JSON head plus safetensors body — pure function of the parameters."""
    head = json.dumps({"d_model": state.d_model, "hidden": state.hidden,
                       "path": state.path},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")
    tensors = {"w_in": state.w_in.data.cpu(), "w_out": state.w_out.data.cpu()}
    return len(head).to_bytes(8, "big") + head + st_save(tensors)


def load(state: VhState, payload: bytes) -> None:
    """emit's inverse, in place."""
    size = int.from_bytes(payload[:8], "big")
    meta = json.loads(payload[8:8 + size])
    if int(meta["d_model"]) != state.d_model:
        raise ValueError(
            f"this payload carries d_model={meta['d_model']}; the entry "
            f"declares d_model={state.d_model}")
    tensors = st_load(payload[8 + size:])
    state.w_in.data.copy_(tensors["w_in"])
    state.w_out.data.copy_(tensors["w_out"])
