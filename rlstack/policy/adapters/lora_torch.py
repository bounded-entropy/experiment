"""LoRA's compute half — imported lazily by the Lora adapter's methods, so the
client library stays importable without torch (STYLE rule 7).

Replay lowering = verb 1 (replace a module): each matched site's Linear is
wrapped so forward returns inner(x) + x A^T B^T. Scaling is fixed at 1
(emitted alpha == r), so the engine's peft consumer applies the same math.
Init is B = 0: version 0 of every delta IS the base model, on both sides.

The wrapper is ROW-AWARE (#44). Installation is additive (I8): every tenant
installed at a site joins the one wrapper standing there, and which (A, B) a
row of the padded microbatch gets is read from the forward's row plan
(adapters/replay.py) — the trainer-side twin of punica's token-indexed slot
bank. One slot is the degenerate case and keeps the pre-batching expression
verbatim.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from rlstack.policy.adapters.replay import ReplayRows, RowPlan, row_plan
from rlstack.policy.siteschema import SiteMeta

PEFT_PREFIX = "base_model.model."


@dataclass
class LoraState:
    """One bank entry's trainable state: (A, B) per site path, rank r."""

    r: int
    a: dict[str, torch.nn.Parameter]   # path -> [r, in]
    b: dict[str, torch.nn.Parameter]   # path -> [out, r]

    def parameters(self) -> list[torch.nn.Parameter]:
        return [*self.a.values(), *self.b.values()]


class LoraSite(torch.nn.Module):
    """inner(x) + x A^T B^T with (A, B) chosen PER ROW — the module that
    replaces a matched Linear.

    One wrapper serves every state installed at this site, and whose delta a
    row gets is a property of the batch (the row plan), not of the module
    tree. `installed` is that set: it is what makes install additive and what
    tells uninstall when the last tenant has left and the Linear goes back.

    The per-tenant deltas deliberately do NOT register as parameters of the
    base: the base is shared and frozen, a delta is one tenant's state, and
    the learner owns it through `params.parameters()`.
    """

    def __init__(self, inner: torch.nn.Module, path: str, plan: RowPlan) -> None:
        super().__init__()
        self.inner = inner
        self.path = path
        self.plan = plan
        self.installed: list[LoraState] = []

    def add(self, state: LoraState) -> None:
        """Additive install: this state's delta becomes routable here."""
        if any(present is state for present in self.installed):
            raise RuntimeError(f"install at {self.path}: this state is already "
                               f"installed — install/uninstall out of balance")
        self.installed.append(state)

    def drop(self, state: LoraState) -> None:
        """install's inverse at one site; the caller unwraps when empty."""
        kept = [present for present in self.installed if present is not state]
        if len(kept) == len(self.installed):
            raise RuntimeError(f"uninstall at {self.path}: this state was never "
                               f"installed — install/uninstall out of balance")
        self.installed = kept

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = self.plan.rows
        one = rows.uniform()
        delta = (_whole_batch_delta(x, one[self.path], self.path) if one is not None
                 else _per_row_delta(x, rows, self.path))
        return self.inner(x) + delta.to(x.dtype)


def _whole_batch_delta(x: torch.Tensor, state: LoraState,
                       path: str) -> torch.Tensor:
    """Every row on one delta: (x A^T) B^T, the expression swap-install
    applied, kept verbatim so a single-tenant microbatch is bit-identical."""
    return (x.to(state.a[path].dtype) @ state.a[path].T) @ state.b[path].T


def _per_row_delta(x: torch.Tensor, rows: ReplayRows,
                   path: str) -> torch.Tensor:
    """Rows on different deltas: gather each row's (A, B) out of the slot
    stack, then two batched GEMMs — punica's shape, written in stock torch.

    The rows of `x` ARE the plan's rows, so a site whose activations are not
    [rows, tokens, in] cannot be routed per row and says so.
    """
    states = [slot[path] for slot in rows.slots]
    ranks = sorted({state.r for state in states})
    if len(ranks) > 1:
        raise ValueError(
            f"site {path}: one forward's slots must agree on rank, got {ranks} "
            f"(punica zero-pads its slot bank to max_rank; the coalescer will)")
    if x.dim() != 3 or x.shape[0] != rows.index.shape[0]:
        raise ValueError(
            f"site {path}: per-row deltas need [rows, tokens, in] activations "
            f"over the plan's {rows.index.shape[0]} rows, got {tuple(x.shape)}")
    a = torch.stack([state.a[path] for state in states])[rows.index]  # [R, r, in]
    b = torch.stack([state.b[path] for state in states])[rows.index]  # [R, out, r]
    return torch.bmm(torch.bmm(x.to(a.dtype), a.transpose(1, 2)),
                     b.transpose(1, 2))


def _site_seed(seed: int, path: str) -> int:
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build(sites: tuple[SiteMeta, ...], init: dict) -> LoraState:
    """A ~ N(0, 1/r) seeded per site, B = 0 (delta starts at the base)."""
    r = int(init["r"])
    seed = int(init.get("seed", 0))
    a: dict[str, torch.nn.Parameter] = {}
    b: dict[str, torch.nn.Parameter] = {}
    for meta in sites:
        if meta.shape is None:
            raise ValueError(f"lora needs a weighted site, got {meta.name}")
        d_in, d_out = meta.shape
        generator = torch.Generator().manual_seed(_site_seed(seed, meta.path))
        a[meta.path] = torch.nn.Parameter(
            torch.randn(r, d_in, generator=generator, dtype=torch.float32) / r)
        b[meta.path] = torch.nn.Parameter(
            torch.zeros(d_out, r, dtype=torch.float32))
    return LoraState(r=r, a=a, b=b)


def _leaf(model: torch.nn.Module, path: str) -> tuple[torch.nn.Module, str]:
    """The (parent module, attribute) a site path addresses."""
    parent = model
    *walk, leaf = path.split(".")
    for step in walk:
        parent = getattr(parent, step)
    return parent, leaf


def install(model: torch.nn.Module, state: LoraState) -> None:
    """Verb 1 at each path: wrap the Linear once, then ADD this state to the
    site.

    Installation is additive (I8): a second tenant at the same site joins the
    LoraSite it finds instead of replacing it, which is exactly what lets one
    forward route its rows to either. Placement happens here — the delta moves
    to the device its site's weight lives on.
    """
    plan = row_plan(model)
    for path in state.a:
        parent, leaf = _leaf(model, path)
        site = getattr(parent, leaf)
        if not isinstance(site, LoraSite):
            site = LoraSite(site, path, plan)
            setattr(parent, leaf, site)
        device = next(site.inner.parameters()).device
        state.a[path].data = state.a[path].data.to(device)
        state.b[path].data = state.b[path].data.to(device)
        site.add(state)


def uninstall(model: torch.nn.Module, state: LoraState) -> None:
    """install's exact inverse: drop this state from each site, and unwrap the
    LoraSite back to its Linear when the last state leaves. The params object
    survives untouched (its tensors are held by reference, not copied), so
    re-install restores identical numerics."""
    for path in state.a:
        parent, leaf = _leaf(model, path)
        site = getattr(parent, leaf)
        if not isinstance(site, LoraSite):
            raise RuntimeError(
                f"uninstall at {path}: expected LoraSite, found "
                f"{type(site).__name__} — install/uninstall out of balance")
        site.drop(state)
        if not site.installed:
            setattr(parent, leaf, site.inner)


def emit(state: LoraState) -> bytes:
    """peft-format safetensors: keys are module paths — the site knowledge
    crosses the membrane INSIDE the payload, in the consumer's native format."""
    tensors = {}
    for path in state.a:
        tensors[f"{PEFT_PREFIX}{path}.lora_A.weight"] = state.a[path].data.cpu()
        tensors[f"{PEFT_PREFIX}{path}.lora_B.weight"] = state.b[path].data.cpu()
    return st_save(tensors)


def load(state: LoraState, payload: bytes) -> None:
    """emit's inverse: restore (A, B) in place from a peft payload."""
    tensors = st_load(payload)
    for path in state.a:
        state.a[path].data.copy_(tensors[f"{PEFT_PREFIX}{path}.lora_A.weight"])
        state.b[path].data.copy_(tensors[f"{PEFT_PREFIX}{path}.lora_B.weight"])


def peft_config(base: str, r: int, target_modules: list[str]) -> str:
    """adapter_config.json for a merged bundle dir (alpha == r: scaling 1)."""
    return json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base,
        "r": r, "lora_alpha": r, "lora_dropout": 0.0, "bias": "none",
        "target_modules": sorted(set(target_modules)),
    }, indent=2)


def merge_fragments(payloads: dict[str, bytes]) -> tuple[bytes, list[str], int]:
    """Fuse bank entries' fragments into ONE adapter file (disjoint keys are
    guaranteed by the one-delta-per-site rule). Returns (file bytes,
    target module leaf names, max rank seen)."""
    merged: dict[str, torch.Tensor] = {}
    leaves: list[str] = []
    rank = 1
    for name in sorted(payloads):
        fragment = st_load(payloads[name])
        overlap = merged.keys() & fragment.keys()
        if overlap:
            raise ValueError(f"payload {name!r} collides on {sorted(overlap)[:3]}")
        merged.update(fragment)
        for key in fragment:
            if key.endswith(".lora_A.weight"):
                rank = max(rank, fragment[key].shape[0])
                leaves.append(key[: -len(".lora_A.weight")].rsplit(".", 1)[-1])
    buffer = io.BytesIO(st_save(merged))
    return buffer.getvalue(), leaves, rank
