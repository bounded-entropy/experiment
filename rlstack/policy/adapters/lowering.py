"""Both execution sides compile from the very same AdapterDefinition.

Only provider loading happens here; no adapter-specific dispatch table and
no torch/vLLM imports on a client. Providers retain native optimized math.
"""

from __future__ import annotations

from importlib import import_module
from dataclasses import dataclass

from rlstack.policy.adapters.definition import AdapterDefinition, ResolvedAdapter


@dataclass(frozen=True)
class ReplayLowering:
    definition: AdapterDefinition

    def provider(self):
        path = ("rlstack.policy.adapters.site_program_torch"
                if self.definition.program is not None else self.definition.replay.module)
        return import_module(path)

    def build(self, resolved: ResolvedAdapter):
        if resolved.definition != self.definition:
            raise ValueError("replay resolved a different adapter definition")
        if self.definition.program is not None:
            return self.provider().build(resolved)
        return self.provider().build(resolved.sites, resolved.settings)

    def install(self, model, state) -> None:
        self.provider().install(model, state)

    def uninstall(self, model, state) -> None:
        self.provider().uninstall(model, state)

    def emit(self, state) -> bytes:
        return self.provider().emit(state)

    def load(self, state, payload: bytes) -> None:
        self.provider().load(state, payload)


def compile_replay(definition: AdapterDefinition) -> ReplayLowering:
    return ReplayLowering(definition)


def compile_rollout(definition: AdapterDefinition, build, adapter_type: str):
    if definition.program is not None:
        from rlstack.policy.adapters.site_program_vllm import SiteRollout
        lowering = SiteRollout(build, adapter_type, definition)
    else:
        if definition.rollout is None:
            raise NotImplementedError(f"{adapter_type} is trainer-only")
        lowering = definition.rollout.factory.load()(build)
    if lowering.mechanism != definition.mechanism:
        raise ValueError("rollout provider mechanism disagrees with the adapter definition")
    # Both native and custom implementations carry their semantic contract.
    lowering.definition = definition
    lowering.adapter_type = adapter_type
    return lowering
