"""Learned bias on an attention-score rectangle: one scalar per (head, prompt
row), over the sites a soft prompt exports.

HALF BUILT, ON PURPOSE. The replay lowering is real and proven; the rollout
lowering demands a PLUGIN whose seams the pinned build does not have — so no
build serves this adapter type, every engine honestly reports NONE for
SIDE_ATTENTION, and a spec carrying an attn_bias is refused at Phase 0 rather
than served wrong. That refusal is the feature: an adapter type is served when
its mechanism is proven, not when its class exists.

Its site (queries -> prompt[:n]) is not a base-model site, so an attn_bias
without a soft prompt in the bank dies at Phase 0 with site-no-match.
"""

from __future__ import annotations

from rlstack.policy.adapters.definition import Operation, SiteKind, builtin
from rlstack.policy.adapters.base import AdapterType, Mechanism, adapter_type
from rlstack.policy.siteschema import SiteMeta


@adapter_type("attn_bias")
class AttnBias(AdapterType):
    definition = builtin(Operation.ATTENTION_BIAS, SiteKind.ATTENTION, 'attn_bias',
                         'attn_bias', 'AttnBiasRollout', 'side_attention')
    serving = Mechanism.SIDE_ATTENTION
    engine_plugin = "rlstack_engine.side_attention"

    def site_ok(self, meta: SiteMeta) -> bool:
        return not meta.has_weight
