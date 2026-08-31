"""reverse_value_head's replay lowering: capture the boundary, value the suffix.

Two captures feed one computation. The SITE wrapper at the boundary path
stores each forward's output — DETACHED, so nothing this head learns can leak
gradient back into the trunk or the policy's own adapters — and a forward
pre-hook on the model stores the same forward's attention mask, which is the
only honest source of row lengths on a padded forward. `provide` then runs
the head: RMS-normalized states plus a learned REVERSE positional embedding
(index = distance to the row's last real token), one ANTI-CAUSAL attention
block (position t attends to t..T and never to padding), one MLP block, and a
zero-initialized scalar head — so values start exactly at zero and the first
updates train the critic before any credit flows.

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
from rlstack.policy.adapters.reverse_value_head import VALUES_PROVIDED
from rlstack.policy.siteschema import SiteMeta

RMS_EPS = 1e-6


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Pre-norm without a gain, plora's rule for plora's reason: a learnable
    scale is one more thing to seed, and the blocks that follow have one."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)


def _drawn(out_features: int, in_features: int,
           generator: torch.Generator) -> torch.nn.Parameter:
    """One bias-free weight, N(0, 1/in_features), off the given generator —
    no module here touches global RNG state."""
    return torch.nn.Parameter(
        torch.randn(out_features, in_features, generator=generator,
                    dtype=torch.float32) / math.sqrt(in_features))


@dataclass
class RvhState:
    """One bank entry's reverse value head: the parameters, and the two
    capture buffers the wrappers fill per forward."""

    d_model: int
    heads: int
    max_len: int
    seed: int
    path: str
    pos: torch.nn.Parameter            # [max_len, d], zero-init: reverse positions
    wq: torch.nn.Parameter             # [d, d]
    wk: torch.nn.Parameter
    wv: torch.nn.Parameter
    wo: torch.nn.Parameter
    mlp_in: torch.nn.Parameter         # [2d, d]
    mlp_out: torch.nn.Parameter        # [d, 2d]
    head: torch.nn.Parameter           # [d], zero-init: values start at 0
    hidden: torch.Tensor | None = None      # captured [rows, W, d], detached
    padding: torch.Tensor | None = None     # captured [rows, W] mask, or None
    hook: Any = field(default=None, repr=False)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.pos, self.wq, self.wk, self.wv, self.wo,
                self.mlp_in, self.mlp_out, self.head]


def _draw_seed(seed: int, path: str) -> int:
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build(sites: tuple[SiteMeta, ...], init: dict) -> RvhState:
    """The identity element, seeded: pos = 0, head = 0 — values are exactly
    zero at version 0, whatever the drawn attention weights say. ONE site: a
    value is read at one boundary, and two would be two heads fighting over
    one provided name."""
    if len(sites) != 1:
        raise ValueError(
            f"reverse_value_head reads ONE boundary, got {len(sites)} sites "
            f"({[m.name for m in sites]}) — narrow the site pattern")
    d_model, heads = int(init["d_model"]), int(init.get("heads", 4))
    max_len = int(init.get("max_len", 2048))
    seed = int(init.get("seed", 0))
    if d_model % heads != 0:
        raise ValueError(f"d_model={d_model} must divide into heads={heads}")
    generator = torch.Generator().manual_seed(_draw_seed(seed, "rvh.trunk"))
    return RvhState(
        d_model=d_model, heads=heads, max_len=max_len, seed=seed,
        path=sites[0].path,
        pos=torch.nn.Parameter(
            torch.zeros(max_len, d_model, dtype=torch.float32)),
        wq=_drawn(d_model, d_model, generator),
        wk=_drawn(d_model, d_model, generator),
        wv=_drawn(d_model, d_model, generator),
        wo=_drawn(d_model, d_model, generator),
        mlp_in=_drawn(2 * d_model, d_model, generator),
        mlp_out=_drawn(d_model, 2 * d_model, generator),
        head=torch.nn.Parameter(torch.zeros(d_model, dtype=torch.float32)))


# ---------------------------------------------------------------------------
# the head itself: suffix values off the captured boundary
# ---------------------------------------------------------------------------

def values(state: RvhState) -> torch.Tensor:
    """[rows, W] suffix values for the captured forward, zero at padding.

    Position t sees positions t..T (the anti-causal mask) and never a padded
    one; its positional coordinate is the distance to the row's last REAL
    token, clamped into the table — the reverse encoding the head learns
    return-to-go against.
    """
    if state.hidden is None:
        return torch.zeros(0)
    h = state.hidden.to(torch.float32)                         # [R, W, d]
    rows, width, _ = h.shape
    mask = (state.padding.to(h.device, torch.float32)
            if state.padding is not None
            else torch.ones(rows, width, device=h.device))
    lengths = mask.sum(dim=1, keepdim=True)                    # [R, 1]
    t = torch.arange(width, device=h.device, dtype=torch.float32)[None, :]
    reverse_index = (lengths - 1.0 - t).clamp(
        min=0.0, max=float(state.max_len - 1)).long()          # [R, W]
    x = rms_norm(h) + state.pos[reverse_index]

    dh = state.d_model // state.heads
    def split(w: torch.Tensor) -> torch.Tensor:
        return (x @ w.T).view(rows, width, state.heads, dh).transpose(1, 2)
    q, k, v = split(state.wq), split(state.wk), split(state.wv)
    scores = q @ k.transpose(-1, -2) / math.sqrt(dh)           # [R, H, W, W]
    anti_causal = torch.tril(torch.ones(width, width, device=h.device,
                                        dtype=torch.bool), diagonal=-1)
    scores = scores.masked_fill(anti_causal[None, None], float("-inf"))
    scores = scores.masked_fill(mask[:, None, None, :] == 0.0, float("-inf"))
    # a PADDED query position can see only padding (everything -inf), and its
    # softmax would be NaN — poisoning the backward through every parameter.
    # Give those rows a flat score instead; their values are zeroed below.
    blocked = torch.isinf(scores).all(dim=-1, keepdim=True)
    scores = scores.masked_fill(blocked, 0.0)
    attended = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2)
    x = x + attended.reshape(rows, width, state.d_model) @ state.wo.T
    x = x + torch.nn.functional.silu(
        rms_norm(x) @ state.mlp_in.T) @ state.mlp_out.T
    return (x @ state.head) * mask                             # [R, W]


def provide(state: RvhState) -> dict[str, Any]:
    return {VALUES_PROVIDED: values(state)}


# ---------------------------------------------------------------------------
# install / uninstall: the two captures
# ---------------------------------------------------------------------------

class RvhSite(SiteWrapper):
    """The boundary tap: pass the module's output through untouched, keep a
    DETACHED copy for every installed head. Transparent to every other
    tenant's forward — a capture changes no number."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        for state in self.installed:
            state.hidden = out.detach()
        return out


def install(model: torch.nn.Module, state: RvhState) -> None:
    """Tap the boundary and hook the model's own forward for the attention
    mask — the padded forward's row lengths, which the boundary tensor alone
    cannot reveal. Placement follows the boundary module's own device."""
    join_site(model, state.path, RvhSite, state)
    device = next(model.parameters()).device
    for parameter in state.parameters():
        parameter.data = parameter.data.to(device)

    def catch_mask(module, args, kwargs):
        state.padding = kwargs.get("attention_mask")
        return None

    state.hook = model.register_forward_pre_hook(catch_mask, with_kwargs=True)


def uninstall(model: torch.nn.Module, state: RvhState) -> None:
    """install's exact inverse: untap the boundary, drop the mask hook."""
    leave_site(model, state.path, RvhSite, state)
    if state.hook is not None:
        state.hook.remove()
        state.hook = None


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------

def emit(state: RvhState) -> bytes:
    """A JSON head plus safetensors body — pure function of the parameters
    (no draws here, so no version counter)."""
    head = json.dumps({"d_model": state.d_model, "heads": state.heads,
                       "max_len": state.max_len, "path": state.path},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")
    tensors = {name: p.data.cpu() for name, p in zip(
        ("pos", "wq", "wk", "wv", "wo", "mlp_in", "mlp_out", "head"),
        (state.pos, state.wq, state.wk, state.wv, state.wo,
         state.mlp_in, state.mlp_out, state.head))}
    return len(head).to_bytes(8, "big") + head + st_save(tensors)


def load(state: RvhState, payload: bytes) -> None:
    """emit's inverse, in place."""
    size = int.from_bytes(payload[:8], "big")
    meta = json.loads(payload[8:8 + size])
    if int(meta["d_model"]) != state.d_model:
        raise ValueError(
            f"this payload carries d_model={meta['d_model']}; the entry "
            f"declares d_model={state.d_model}")
    tensors = st_load(payload[8 + size:])
    for name, parameter in zip(
            ("pos", "wq", "wk", "wv", "wo", "mlp_in", "mlp_out", "head"),
            (state.pos, state.wq, state.wk, state.wv, state.wo,
             state.mlp_in, state.mlp_out, state.head)):
        parameter.data.copy_(tensors[name])
