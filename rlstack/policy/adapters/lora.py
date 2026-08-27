"""Per-matrix low-rank delta; served natively by vLLM multi-LoRA (punica)."""

from __future__ import annotations

from rlstack.policy.adapters.base import Adapter, Mechanism, adapter
from rlstack.policy.siteschema import SiteMeta


@adapter("lora")
class Lora(Adapter):
    serving = Mechanism.PUNICA

    def site_ok(self, meta: SiteMeta) -> bool:
        return meta.has_weight

    # compute half — lora_torch imports torch, so it loads lazily (rule 7)

    def params(self, sites: tuple[SiteMeta, ...], init: dict):
        from rlstack.policy.adapters import lora_torch
        return lora_torch.build(sites, init)

    def install_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import lora_torch
        lora_torch.install(model, params)

    def uninstall_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import lora_torch
        lora_torch.uninstall(model, params)

    def emit(self, params) -> bytes:
        from rlstack.policy.adapters import lora_torch
        return lora_torch.emit(params)

    def load(self, params, payload: bytes) -> None:
        from rlstack.policy.adapters import lora_torch
        lora_torch.load(params, payload)
