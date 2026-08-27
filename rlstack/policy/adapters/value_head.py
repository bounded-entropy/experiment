"""Trainer-only scalar head over a hidden boundary; never served — it versions
and stores like any delta, but the engine never hears about it."""

from __future__ import annotations

from rlstack.policy.adapters.base import Adapter, adapter
from rlstack.policy.siteschema import SiteMeta


@adapter("value_head")
class ValueHead(Adapter):
    serving = None
    provides = frozenset({"values"})

    def site_ok(self, meta: SiteMeta) -> bool:
        return meta.is_boundary and not meta.has_weight
