"""A trainer-only scalar head at a hidden boundary: it has no rollout lowering
at all, so a bundle pins its version like any other delta but never ships a
payload for it. What it contributes is `values`, a tensor the replay forward
provides."""

from __future__ import annotations

from rlstack.policy.adapters.base import Adapter, adapter
from rlstack.policy.siteschema import SiteMeta


@adapter("value_head")
class ValueHead(Adapter):
    serving = None
    provides = frozenset({"values"})

    def site_ok(self, meta: SiteMeta) -> bool:
        return meta.is_boundary and not meta.has_weight
