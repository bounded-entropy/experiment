"""spectral's rollout lowering: punica, serving the materialized top-k.

The trainer already collapsed the spectrum: emit's payload carries an
ordinary rank-k peft pair per site (lora_A = V_S^T, lora_B = U_S diag(eff_S))
beside the dense gains, so this side only unwraps the container, writes the
peft half to disk, and registers a LoRARequest — lora_vllm's moves, restated
for spectral's container format. No factors artifact, no cas reader: the
frozen spectrum never leaves the trainer.

vLLM and torch are imported at module scope — this file loads only from
Spectral.rollout_lowering, never from the package root (STYLE rule 7).
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from typing import Any

import torch
from safetensors.torch import save as st_save
from vllm.lora.request import LoRARequest

from rlstack.policy.adapters import lora_torch, spectral_torch
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.rollout import (
    Alignment, BuildDemands, Levers, Request, RolloutLowering, ServingBuild,
)
from rlstack.policy.siteschema import SiteMeta

PEFT_KEY = "peft."


class SpectralRollout(RolloutLowering):
    """Top-k spectral gains, served as one ordinary punica adapter."""

    adapter_type = "spectral"
    mechanism = Mechanism.PUNICA
    claims = ("lora_request",)

    def __init__(self, build: ServingBuild) -> None:
        super().__init__(build)
        self._next_int_id = 1        # punica addresses its slots by int id

    def demands(self) -> BuildDemands:
        """Exactly lora's demands: one resident bundle is one adapter."""
        return BuildDemands(engine_args={
            "enable_lora": True,
            "max_loras": self.build.max_bundles,
            "max_lora_rank": self.build.max_rank})

    def reaches(self, meta: SiteMeta) -> bool:
        return meta.has_weight

    def attach(self, bundle_id: str,
               payloads: Mapping[str, bytes]) -> LoRARequest:
        """Unwrap each entry's container, keep the peft half, fuse and
        register — disjoint keys guaranteed by the one-delta-per-site rule."""
        merged: dict[str, torch.Tensor] = {}
        leaves: list[str] = []
        rank = 1
        for name in sorted(payloads):
            head, tensors = spectral_torch.unpack(payloads[name])
            rank = max(rank, int(head["k"]))
            for key, value in tensors.items():
                if not key.startswith(PEFT_KEY):
                    continue                      # the dense gains are resume's
                peft_key = key[len(PEFT_KEY):]
                if peft_key in merged:
                    raise ValueError(
                        f"payload {name!r} collides on {peft_key!r}")
                merged[peft_key] = value
                if peft_key.endswith(".lora_A.weight"):
                    leaves.append(
                        peft_key[: -len(".lora_A.weight")].rsplit(".", 1)[-1])
        adapter_dir = self.build.workdir / f"spectral-{bundle_id.replace(':', '_')}"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_model.safetensors").write_bytes(st_save(merged))
        (adapter_dir / "adapter_config.json").write_text(
            lora_torch.peft_config(self.build.base, rank, leaves))
        request = LoRARequest(lora_name=f"spectral-{bundle_id}",
                              lora_int_id=self._next_int_id,
                              lora_path=str(adapter_dir))
        self._next_int_id += 1
        return request

    def apply(self, attached: Any, request: Request) -> Levers:
        """One keyword, the whole mechanism — lora's rule verbatim."""
        return Levers(kwargs={"lora_request": attached})

    def align(self, attached: Any) -> Alignment:
        """A weight delta adds no positions."""
        return Alignment(0)

    def detach(self, attached: LoRARequest) -> None:
        """Delete the fused adapter; ids are monotone and never reused."""
        shutil.rmtree(attached.lora_path, ignore_errors=True)
