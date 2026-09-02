"""LoRA's replay lowering: each matched site's Linear wrapped so its forward
returns inner(x) + x A^T B^T.

Scaling is fixed at 1 (emitted alpha == r) so the engine's peft consumer
applies the same math, and init is B = 0, which makes version 0 of every delta
the base model on both sides. The wrapper is ROW-AWARE: install is additive, so
every tenant at a site joins the one wrapper standing there, and which (A, B) a
row gets is read from the forward's row plan. torch is imported at module scope
— this file loads only from the adapter type's methods (STYLE rule 7).
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

from rlstack.policy.adapters.replay import (
    ReplayRows, SiteWrapper, join_site, leaf_module, leave_site,
)
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


class LoraSite(SiteWrapper):
    """inner(x) + x A^T B^T with (A, B) chosen PER ROW — this FAMILY's link
    in the chain at a matched Linear.

    One wrapper serves every LoraState installed at this site; whose delta a
    row gets is a property of the batch (the row plan), not of the module
    tree. Chain mechanics (roster, nesting with other families) are
    SiteWrapper's; what is THIS family's is the math and the FAMILY FILTER: a
    row whose routed state at this path is not a LoraState — none, or another
    family's — passes through to `inner`, where its own wrapper (or the base
    Linear) is waiting. The per-tenant deltas deliberately do NOT register as
    parameters of the base: the learner owns them through
    `params.parameters()`.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A row's delta is the one its slot carries HERE — and a slot that
        carries none of THIS family here gets the inner module untouched."""
        rows = self.plan.rows
        one = rows.uniform()
        if one is not None:
            state = one.get(self.path)
            if not isinstance(state, LoraState):
                return self.inner(x)          # the transparent case
            delta = _whole_batch_delta(x, state, self.path)
        else:
            delta = _per_row_delta(x, rows, self.path)
        return self.inner(x) + delta.to(x.dtype)


def _whole_batch_delta(x: torch.Tensor, state: LoraState,
                       path: str) -> torch.Tensor:
    """Every row on one delta: (x A^T) B^T in one GEMM pair — the plain
    single-tenant expression, kept verbatim so a uniform microbatch is
    bit-identical whatever else shares the learner."""
    return (x.to(state.a[path].dtype) @ state.a[path].T) @ state.b[path].T


def _per_row_delta(x: torch.Tensor, rows: ReplayRows,
                   path: str) -> torch.Tensor:
    """Rows on different deltas: gather each row's (A, B) out of the slot
    stack, then two batched GEMMs — punica's shape, written in stock torch.

    The rows of `x` ARE the plan's rows, so a site whose activations are not
    [rows, tokens, in] cannot be routed per row and says so. Every slot of a
    MIXED forward must carry this path: the transparent case is uniform-only,
    because zero-padding one row's delta is the coalescer's admission rule, not
    a silent fallback here.
    """
    missing = [i for i, slot in enumerate(rows.slots)
               if not isinstance(slot.get(path), LoraState)]
    if missing:
        raise ValueError(
            f"site {path}: slots {missing} carry no delta here, so a mixed "
            f"forward cannot route this site (a uniform forward serves them "
            f"the base; admission is the coalescer's job)")
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


def install(model: torch.nn.Module, state: LoraState) -> None:
    """ADD this state to its family's wrapper at each path — joining the one
    that stands, or entering the chain (I8: additive, and nesting-safe beside
    another family's wrapper). Placement happens here — the delta moves to
    the device its site's weight lives on."""
    for path in state.a:
        parent, leaf = leaf_module(model, path)
        device = next(getattr(parent, leaf).parameters()).device
        state.a[path].data = state.a[path].data.to(device)
        state.b[path].data = state.b[path].data.to(device)
        join_site(model, path, LoraSite, state)


def uninstall(model: torch.nn.Module, state: LoraState) -> None:
    """install's exact inverse: drop this state, splicing this family's
    wrapper out of the chain when its last state leaves. The params object
    survives untouched (its tensors are held by reference), so re-install
    restores identical numerics."""
    for path in state.a:
        leave_site(model, path, LoraSite, state)


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
