"""The steer's replay lowering: a site wrapper at the boundary that ADDS.

`SteerSite` passes the module's output through with each routed row's vector
added — inside the window that row's turns RECORDED (ReplayRows.facts, the
rollout's own account of what it did), or at every position when the row
recorded none, which is a row this policy never sampled (ADR 0005, Q3) —
and leaves every other row exactly as it was: a row whose slot carries no steer
here gets zero, which is the identity, which is what it would have gotten
anyway. That transparency is what lets a steer tenant share a learner with a
lora tenant (I8), and zero-init is what makes version 0 the base bit for bit.

Padding needs no mask: a padded position is causally after every real one and
outside the loss, so adding there changes no number that is read.

torch is imported at module scope — this file loads only from the adapter
type's methods (STYLE rule 7).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from rlstack.policy.adapters.replay import (
    ReplayRows, SiteWrapper, join_site, leave_site,
)
from rlstack.policy.adapters.steer import STEER_RECORD
from rlstack.policy.siteschema import SiteMeta


@dataclass
class SteerState:
    """One bank entry's trainable state: one [d] vector per boundary path —
    or ONE vector every path shares, when tied."""

    d: int
    paths: tuple[str, ...]
    tie: bool
    vectors: dict[str, torch.nn.Parameter]      # path -> [d]; tied: one object
    alpha: float | None = None                  # nsteer: the fraction of ||h_t|| added (its start)
    log_alpha: torch.nn.Parameter | None = None  # nsteer, trained: alpha = exp(log_alpha)
    unit: bool = False                          # nsteer: the direction is kept on the unit sphere

    def fraction(self) -> torch.Tensor | None:
        """The fraction as a tensor — the parameter's exp when alpha is
        trained (so the scale's gradient reaches it), the constant otherwise,
        None for a plain steer."""
        if self.log_alpha is not None:
            return torch.exp(self.log_alpha)
        if self.alpha is not None:
            return torch.tensor(float(self.alpha))
        return None

    def parameters(self) -> list[torch.nn.Parameter]:
        unique: dict[int, torch.nn.Parameter] = {}
        for vector in self.vectors.values():
            unique.setdefault(id(vector), vector)
        out = list(unique.values())
        if self.log_alpha is not None:
            out.append(self.log_alpha)
        return out


class SteerSite(SiteWrapper):
    """The boundary add: this FAMILY's link in the chain at a boundary."""

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        rows = self.plan.rows
        delta = _rows_delta(rows, self.path, out)
        if delta is None:
            return out                            # no row carries a steer here
        mask = window_mask(rows, self.path, int(out.shape[1]), out.device)
        scale = _rows_scale(rows, self.path, out)             # [rows, tokens, 1]
        return out + (mask.unsqueeze(-1) * scale * delta).to(out.dtype)


def _rows_delta(rows: ReplayRows, path: str,
                out: torch.Tensor) -> torch.Tensor | None:
    """[rows, 1, d]: each row's vector at this path, zero for a row whose slot
    carries none — or None when no row carries one at all."""
    states = [slot.get(path) for slot in rows.slots]
    if not any(isinstance(state, SteerState) for state in states):
        return None
    if out.dim() != 3 or out.shape[0] != rows.index.shape[0]:
        raise ValueError(
            f"site {path}: a steer needs [rows, tokens, d] activations over "
            f"the plan's {rows.index.shape[0]} rows, got {tuple(out.shape)}")
    width = int(out.shape[-1])
    stack = torch.stack([
        _direction(state, path, out.device)
        if isinstance(state, SteerState)
        else torch.zeros(width, device=out.device)
        for state in states])                                  # [slots, d]
    return stack[rows.index].unsqueeze(1)


def _direction(state: SteerState, path: str, device) -> torch.Tensor:
    """The vector a row adds: as it is for a steer; the UNIT direction for a
    norm-scaled one, whose magnitude is alpha's and not v's."""
    vector = state.vectors[path].to(device)
    if state.alpha is None:
        return vector
    return vector / (vector.norm() + 1e-12)


def _rows_scale(rows: ReplayRows, path: str, out: torch.Tensor) -> torch.Tensor:
    """[rows, tokens, 1]: what each row's delta is multiplied by — 1 for a
    steer, `alpha * ||out[row, t]||` for a norm-scaled one (the paper's
    rule, per token, off the LIVE residual). The norm is detached: the
    magnitude is a function of the stream, not a thing the direction's
    gradient should try to move."""
    fractions = []
    for slot in rows.slots:
        state = slot.get(path)
        fraction = state.fraction() if isinstance(state, SteerState) else None
        fractions.append(torch.zeros((), device=out.device) if fraction is None
                         else fraction.to(out.device))
    alphas = torch.stack(fractions)[rows.index]                    # [rows], differentiable
    if not bool((alphas > 0).any()):
        return torch.ones(1, 1, 1, device=out.device, dtype=torch.float32)
    norms = out.detach().float().norm(dim=-1, keepdim=True)     # [rows, tokens, 1]
    alphas = alphas.view(-1, 1, 1)
    return torch.where(alphas > 0, alphas * norms, torch.ones_like(norms))


def window_mask(rows: ReplayRows, path: str, width: int,
                device) -> torch.Tensor:
    """[rows, width] — 1 where a row's window covers the position, for rows
    whose slot carries a steer here; 0 everywhere else.

    The window is read, never re-derived (I6): every turn a steer bundle
    served recorded one, and a row whose turns disagree is not replayable in
    one forward (one trajectory is one window — plora's rule), so it is
    refused with the row named. A row that recorded NOTHING is the other
    case, and it is not an error: see `recorded_window`.
    """
    mask = torch.zeros(int(rows.index.shape[0]), width, device=device)
    for row in range(int(rows.index.shape[0])):
        slot = rows.slots[int(rows.index[row])]
        if not isinstance(slot.get(path), SteerState):
            continue
        start, end = recorded_window(rows, row, path)
        mask[row, start:width if end is None else min(end, width)] = 1.0
    return mask


def recorded_window(rows: ReplayRows, row: int,
                    path: str) -> tuple[int, int | None]:
    """One row's window, out of its turns' recorded extras — and THE DEFAULT
    WHERE THERE IS NO RECORD (ADR 0005, Q3 as redacted).

    A record, when there is one, is the only truth (ADR 0004, Q2): the
    rollout resolved the caller's directive and sealed what it applied, so
    replay never has to know what was asked. But a trajectory the policy
    never sampled has no record at all and is NOT a recording bug — it is
    the ordinary case of training a steer on foreign rows (a teacher's
    sealed rollouts, a fixed cas file). Such a row replays at EVERY position,
    `(0, None)`: the adapter type's own default, the same one the rollout
    applies when no directive is passed, declared in no bank.

    The alarm the refusal used to raise lives on the SEAL side instead,
    where the fact is: every bundle carrying a steer records a window on
    both buses, which is what makes "no record" mean "no steer sampled this"
    rather than "a steer forgot".
    """
    turns = () if rows.facts is None else rows.facts[row]
    windows = {tuple(turn[STEER_RECORD]) for turn in turns
               if STEER_RECORD in turn}
    if not windows:
        return 0, None
    if len(windows) > 1:
        raise ValueError(
            f"site {path}: row {row}'s turns recorded {len(windows)} different "
            f"windows — one trajectory is one window, so this row's recording "
            f"and its routing have come apart")
    start, end = windows.pop()
    return int(start), (None if end is None else int(end))


def _site_seed(seed: int, path: str) -> int:
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


# ---------------------------------------------------------------------------
# the bodies the Steer adapter type's methods call
# ---------------------------------------------------------------------------

def build(sites: tuple[SiteMeta, ...], init: dict) -> SteerState:
    """One [d] vector per matched boundary (or one shared, tied), zero unless
    init_std says otherwise — zero IS the identity, and the gradient is there
    from the first step."""
    if not sites:
        raise ValueError("a steer matches at least one boundary")
    d = int(init["d"])
    tie = bool(init.get("tie", False))
    std = float(init.get("init_std", 0.0))
    seed = int(init.get("seed", 0))
    alpha = None if init.get("alpha") is None else float(init["alpha"])
    if alpha is not None and (alpha <= 0.0 or std <= 0.0):
        raise ValueError(
            f"a norm-scaled steer needs alpha > 0 and a direction to start "
            f"from (init_std > 0); got alpha={alpha}, init_std={std}")
    unit = alpha is not None and bool(init.get("unit", False))
    train_alpha = alpha is not None and bool(init.get("train_alpha", False))
    paths = tuple(meta.path for meta in sites)

    def draw(path: str) -> torch.nn.Parameter:
        if std == 0.0:
            return torch.nn.Parameter(torch.zeros(d, dtype=torch.float32))
        generator = torch.Generator().manual_seed(_site_seed(seed, path))
        drawn = torch.randn(d, generator=generator, dtype=torch.float32) * std
        if unit:
            drawn = drawn / drawn.norm()          # born on the sphere
        return torch.nn.Parameter(drawn)

    if tie:
        shared = draw(paths[0])
        vectors = {path: shared for path in paths}
    else:
        vectors = {path: draw(path) for path in paths}
    log_alpha = (torch.nn.Parameter(torch.tensor(math.log(alpha), dtype=torch.float32))
                 if train_alpha else None)
    return SteerState(d=d, paths=paths, tie=tie, vectors=vectors, alpha=alpha,
                      log_alpha=log_alpha, unit=unit)


def project(state: SteerState) -> None:
    """After a step: a unit-sphere direction back onto the sphere, in place.
    The step moved the parameter off it by about the learning rate; what
    the parameter MEANS is the direction, so the norm is put back to one
    and the optimizer's moments are left as they are (they are about the
    coordinates, and the coordinates barely moved)."""
    if not state.unit:
        return
    with torch.no_grad():
        for vector in state.parameters():
            if vector is state.log_alpha:
                continue
            vector.data.div_(vector.data.norm() + 1e-12)


def effective_alpha(state: SteerState) -> float | None:
    """The fraction this version injects — the trained one when trained."""
    fraction = state.fraction()
    return None if fraction is None else float(fraction.detach())


def install(model: torch.nn.Module, state: SteerState) -> None:
    """Join this family's wrapper at every matched boundary — additively
    (I8) — and place the vectors on the model's device."""
    device = next(model.parameters()).device
    for parameter in state.parameters():
        parameter.data = parameter.data.to(device)
    for path in state.paths:
        join_site(model, path, SteerSite, state)


def uninstall(model: torch.nn.Module, state: SteerState) -> None:
    """install's exact inverse, boundary by boundary."""
    for path in state.paths:
        leave_site(model, path, SteerSite, state)


def emit(state: SteerState) -> bytes:
    """safetensors keyed by boundary PATH — the engine-side consumer adds each
    key's vector at that path. A tied entry emits its one vector under every
    path (cloned: safetensors refuses shared storage), so the payload is
    self-describing either way."""
    alpha = effective_alpha(state)
    return st_save({path: state.vectors[path].data.cpu().clone()
                    for path in state.paths},
                   metadata=None if alpha is None else {ALPHA_KEY: repr(alpha)})


def load(state: SteerState, payload: bytes) -> None:
    """emit's inverse, in place."""
    tensors = st_load(payload)
    for path in state.paths:
        state.vectors[path].data.copy_(tensors[path])
    if state.log_alpha is not None:
        alpha = payload_alpha(payload)
        if alpha is not None:
            state.log_alpha.data.fill_(math.log(alpha))


ALPHA_KEY = "rlstack_alpha"
"""The safetensors metadata key a norm-scaled steer's payload carries its
fraction under — read by the engine hook off the fused file, so the hook
scales exactly what the learner scaled."""


def payload_alpha(payload: bytes) -> float | None:
    """The fraction a steer payload declares, or None for a plain steer —
    read off the safetensors header (8-byte length, JSON, `__metadata__`),
    which is the one place a payload can say something about itself."""
    import json
    import struct

    n = struct.unpack("<Q", payload[:8])[0]
    header = json.loads(payload[8:8 + n])
    meta = header.get("__metadata__") or {}
    return None if ALPHA_KEY not in meta else float(meta[ALPHA_KEY])


def merge_alpha(payloads: Mapping[str, bytes]) -> float | None:
    """One fraction for the bundle's steer family, or None: every entry that
    declares one must declare the same, because the hook scales per slot and
    a bundle is one slot."""
    alphas = {payload_alpha(payload) for payload in payloads.values()}
    alphas.discard(None)
    if len(alphas) > 1:
        raise ValueError(
            f"one bundle's steer entries declare {len(alphas)} different "
            f"fractions {sorted(alphas)}; a bundle is one slot in the hook's "
            f"bank and scales by one alpha")
    return alphas.pop() if alphas else None


def merge_vectors(payloads: Mapping[str, bytes]) -> dict[str, torch.Tensor]:
    """Fuse the bank's steer entries into ONE {path: vector} map, in bank
    order — an adapter type attaches its entries jointly. Two entries at one
    path is the bank rule broken (Phase 0 refuses it as site-overlap), so it
    is refused here too rather than summed."""
    merged: dict[str, torch.Tensor] = {}
    for name in sorted(payloads):
        for path, vector in st_load(payloads[name]).items():
            if path in merged:
                raise ValueError(
                    f"two steer entries in one bundle carry {path!r}; a site "
                    f"carries at most one delta per tenant")
            merged[path] = vector
    return merged
