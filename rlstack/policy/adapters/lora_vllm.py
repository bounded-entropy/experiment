"""LoRA's rollout lowering: punica, the native multi-LoRA lever.

What this adapter type pays and what it buys: demands enable_lora plus its
sizing; attach merges the bank's fragments (disjoint keys, the
one-delta-per-site rule) into one peft adapter dir and registers it as a
LoRARequest, whose int id is per BUILD and allocated here because this lowering
is the only thing that hands them out; apply contributes the lora_request
keyword, which is what lets requests pinning different bundles batch together;
align contributes nothing, because a weight delta occupies no prompt positions.

vLLM and torch are imported at module scope — this file loads only from
Lora.rollout_lowering, never from the package root (STYLE rule 7).
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from typing import Any

from vllm.lora.request import LoRARequest

from rlstack.policy.adapters import lora_torch
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.rollout import (
    Alignment, BuildDemands, Levers, Request, RolloutLowering, ServingBuild,
)
from rlstack.policy.siteschema import SiteMeta


class LoraRollout(RolloutLowering):
    """Per-matrix low-rank deltas, served by vLLM's own punica kernels."""

    adapter_type = "lora"
    mechanism = Mechanism.PUNICA
    claims = ("lora_request",)

    def __init__(self, build: ServingBuild) -> None:
        super().__init__(build)
        self._next_int_id = 1        # punica addresses its slots by int id

    def demands(self) -> BuildDemands:
        """The lever plus its sizing: how many adapters may be resident and
        the widest rank the kernels are built for (uniform across a bundle —
        the merged adapter_config carries one r)."""
        return BuildDemands(engine_args={
            "enable_lora": True,
            "max_loras": self.build.max_bundles,
            "max_lora_rank": self.build.max_rank})

    def reaches(self, meta: SiteMeta) -> bool:
        """Punica reaches every weighted matrix of the served base."""
        return meta.has_weight

    def attach(self, bundle_id: str,
               payloads: Mapping[str, bytes]) -> LoRARequest:
        """The bank's fragments, fused into ONE adapter on disk and registered.

        peft is punica's native format, so the site knowledge crosses inside
        the payload (lora_torch.emit writes module paths as keys) and this side
        only has to say where the file is.
        """
        merged, leaves, rank = lora_torch.merge_fragments(dict(payloads))
        adapter_dir = self.build.workdir / bundle_id.replace(":", "_")
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_model.safetensors").write_bytes(merged)
        (adapter_dir / "adapter_config.json").write_text(
            lora_torch.peft_config(self.build.base, rank, leaves))
        request = LoRARequest(lora_name=bundle_id,
                              lora_int_id=self._next_int_id,
                              lora_path=str(adapter_dir))
        self._next_int_id += 1
        return request

    def apply(self, attached: Any, request: Request) -> Levers:
        """One keyword, and it is the whole mechanism: the request pins its own
        adapter, so requests carrying different bundles batch together."""
        return Levers(kwargs={"lora_request": attached})

    def align(self, attached: Any) -> Alignment:
        """A weight delta adds no positions — the prompt is exactly the tokens."""
        return Alignment(0)

    def detach(self, attached: LoRARequest) -> None:
        """Delete the fused adapter this bundle wrote, and forget its id.

        There is nothing to tell the engine: attach never registered anything
        with vLLM — it built a LoRARequest that each request carries — so
        releasing IS removing the directory that request pointed at. vLLM's own
        LoRA caches evict on their own schedule and reload from that path, so
        the one rule is that no request may pin this bundle again, which is what
        the engine's pin-aware eviction guarantees.

        Int ids are drawn monotonically and never reused, so a later bundle can
        never be confused with a stale cached copy of this one.
        """
        shutil.rmtree(attached.lora_path, ignore_errors=True)
