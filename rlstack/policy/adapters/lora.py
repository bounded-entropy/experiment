"""A per-matrix low-rank delta, served through punica — the engine's own
multi-LoRA lever — and wrapped around a Linear on the replay side."""

from __future__ import annotations

from rlstack.policy.adapters.base import Adapter, Mechanism, adapter
from rlstack.policy.siteschema import SiteMeta


@adapter("lora")
class Lora(Adapter):
    serving = Mechanism.PUNICA

    def site_ok(self, meta: SiteMeta) -> bool:
        return meta.has_weight

    # compute halves — lora_torch imports torch and lora_vllm imports vLLM,
    # so both load lazily, from here only (rule 7)

    def rollout_lowering(self, build):
        from rlstack.policy.adapters import lora_vllm
        return lora_vllm.LoraRollout(build)

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
