"""A per-matrix low-rank delta, served through punica — the engine's own
multi-LoRA lever — and wrapped around a Linear on the replay side."""

from __future__ import annotations

from rlstack.policy.adapters.definition import Operation, SiteKind, builtin
from rlstack.policy.adapters.base import AdapterType, Mechanism, adapter_type
from rlstack.policy.siteschema import SiteMeta


@adapter_type("lora")
class Lora(AdapterType):
    definition = builtin(Operation.LOW_RANK, SiteKind.LINEAR, 'lora',
                         'lora', 'LoraRollout', 'punica')
    serving = Mechanism.PUNICA

    def site_ok(self, meta: SiteMeta) -> bool:
        return meta.has_weight
