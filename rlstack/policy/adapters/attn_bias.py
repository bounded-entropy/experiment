"""Learned bias on an attention-score rectangle — the one adapter served by
OUR mechanism: stock kernels untouched, the biased prompt-segment attention
computed densely and merged by exact LSE arithmetic. Ships as an engine plugin
(rlstack_engine.side_attention, jointly consuming the soft prompt's rows) on
the rollout side, and a score-level patch on the replay side.

Its site (queries -> prompt[:n]) is not a base-model site: the soft prompt
EXPORTS it, so an attn_bias without a soft prompt in the bank dies at Phase 0
with site-no-match.
"""

from __future__ import annotations

from rlstack.policy.adapters.base import Adapter, Mechanism, adapter
from rlstack.policy.siteschema import SiteMeta


@adapter("attn_bias")
class AttnBias(Adapter):
    serving = Mechanism.SIDE_ATTENTION
    engine_plugin = "rlstack_engine.side_attention"

    def site_ok(self, meta: SiteMeta) -> bool:
        return not meta.has_weight
