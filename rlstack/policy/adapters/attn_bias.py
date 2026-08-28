"""Learned bias on an attention-score rectangle — the one adapter served by
OUR mechanism: stock kernels untouched, the biased prompt-segment attention
computed densely and merged by exact LSE arithmetic. Ships as an engine plugin
(rlstack_engine.side_attention, jointly consuming the soft prompt's rows) on
the rollout side, and a score-level patch on the replay side.

Its site (queries -> prompt[:n]) is not a base-model site: the soft prompt
EXPORTS it, so an attn_bias without a soft prompt in the bank dies at Phase 0
with site-no-match.

HALF BUILT, ON PURPOSE (#46). The REPLAY lowering is real and proven
(attn_bias_torch: the bias rides the 4-D attention mask, which the stock
attention already adds to the scores). The ROLLOUT lowering exists as a
declaration only (attn_bias_vllm): its demands() name a PLUGIN, and the seams
that plugin was designed against do not exist on vllm 0.28.0 — so
rlstack_engine.side_attention still refuses at probe, no build serves this
kind, and every engine honestly reports NONE for SIDE_ATTENTION. This kind
therefore cannot pass Phase 0's reachability check and no run can use it yet.
That refusal is the feature: a kind is served when its mechanism is proven,
not when its class exists.
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

    # compute halves — attn_bias_torch imports torch, so it loads lazily
    # (rule 7); attn_bias_vllm is the rollout half this build cannot pay for
    def rollout_lowering(self, build):
        from rlstack.policy.adapters import attn_bias_vllm
        return attn_bias_vllm.AttnBiasRollout(build)

    def params(self, sites: tuple[SiteMeta, ...], init: dict):
        from rlstack.policy.adapters import attn_bias_torch
        return attn_bias_torch.build(sites, init)

    def install_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import attn_bias_torch
        attn_bias_torch.install(model, params)

    def uninstall_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import attn_bias_torch
        attn_bias_torch.uninstall(model, params)

    def emit(self, params) -> bytes:
        from rlstack.policy.adapters import attn_bias_torch
        return attn_bias_torch.emit(params)

    def load(self, params, payload: bytes) -> None:
        from rlstack.policy.adapters import attn_bias_torch
        attn_bias_torch.load(params, payload)
