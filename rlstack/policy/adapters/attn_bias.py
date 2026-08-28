"""Learned bias on an attention-score rectangle: one scalar per (head, prompt
row), over the sites a soft prompt exports.

HALF BUILT, ON PURPOSE. The replay lowering is real and proven; the rollout
lowering demands a PLUGIN whose seams the pinned build does not have — so no
build serves this kind, every engine honestly reports NONE for SIDE_ATTENTION,
and a spec carrying an attn_bias is refused at Phase 0 rather than served
wrong. That refusal is the feature: a kind is served when its mechanism is
proven, not when its class exists.

Its site (queries -> prompt[:n]) is not a base-model site, so an attn_bias
without a soft prompt in the bank dies at Phase 0 with site-no-match.
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
