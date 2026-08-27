"""LoRA's compute half — imported lazily by the Lora adapter's methods, so the
client library stays importable without torch (STYLE rule 7).

Replay lowering = verb 1 (replace a module): each matched site's Linear is
wrapped so forward returns inner(x) + x A^T B^T. Scaling is fixed at 1
(emitted alpha == r), so the engine's peft consumer applies the same math.
Init is B = 0: version 0 of every delta IS the base model, on both sides.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

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


class LoraLinear(torch.nn.Module):
    """inner(x) + x A^T B^T — the module that replaces a matched Linear."""

    def __init__(self, inner: torch.nn.Module, a: torch.nn.Parameter,
                 b: torch.nn.Parameter) -> None:
        super().__init__()
        self.inner = inner
        self.lora_a = a
        self.lora_b = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (x.to(self.lora_a.dtype) @ self.lora_a.T) @ self.lora_b.T
        return self.inner(x) + delta.to(x.dtype)


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
    """Verb 1 at each path: navigate the module tree, wrap the Linear."""
    for path in state.a:
        parent = model
        *walk, leaf = path.split(".")
        for step in walk:
            parent = getattr(parent, step)
        inner = getattr(parent, leaf)
        device = next(inner.parameters()).device
        state.a[path].data = state.a[path].data.to(device)
        state.b[path].data = state.b[path].data.to(device)
        setattr(parent, leaf, LoraLinear(inner, state.a[path], state.b[path]))


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
