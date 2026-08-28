"""attn_bias's ROLLOUT lowering: the one whose demands NO build here can pay
(#48, #46, I7).

The kind's replay half is real and proven (attn_bias_torch: an attention mask
already IS a score-level bias). Its rollout half is a PLUGIN — our own
mechanism, not a lever vLLM maintains — and the plugin is honestly blocked on
the pinned build: vllm 0.28.0 does not plumb return_softmax_lse through the
dense FlashAttention path, so "stock kernel + tiny partition attention + exact
LSE merge" has no seam short of forking the kernel dispatch (#46 names the
symbols, and rlstack_engine.side_attention's probe fails on the true one).

That is what this file says, in the contract's own vocabulary: a rollout
lowering whose demands() name a PLUGIN, which an engine that installs none
cannot pay. The consequences are the ones #46 already shipped, now reached
through the same door as every other kind:

  - no build serves this kind, so reachability reports NONE for the
    attention-score rectangle and a spec carrying an attn_bias is refused at
    Phase 0 rather than served wrong;
  - the plugin is named by STRING, never imported: rlstack_engine ships in the
    engine image and the import is one-way (rlstack never imports back).

A kind is served when its mechanism is proven, not when its class exists. The
route that would unblock it is FlexAttention's score_mod (#46), which is a
registered backend and a metadata builder — deliberately not built here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.rollout import (
    BuildDemands, Levers, Request, RolloutLowering,
)
from rlstack.policy.siteschema import SiteMeta

PLUGIN = "rlstack_engine.side_attention"   # by string: the import is one-way
ATTN_SCORES = "attn_scores"


class AttnBiasRollout(RolloutLowering):
    """Soft prompt + bias served jointly by the side-attention plugin."""

    kind = "attn_bias"
    mechanism = Mechanism.SIDE_ATTENTION
    claims = ()          # unreachable, so it claims nothing of any request yet

    def demands(self) -> BuildDemands:
        """A plugin in the engine image, installed and probed at boot. No build
        in this repo installs one, so asking to serve this kind refuses at
        construction — the honest form of #46's blocked mechanism."""
        return BuildDemands(plugin=PLUGIN)

    def reaches(self, meta: SiteMeta) -> bool:
        """What the payment would buy: the queries -> prompt[:n] rectangle the
        soft prompt exports."""
        return meta.path == ATTN_SCORES

    def attach(self, bundle_id: str, payloads: Mapping[str, bytes]) -> Any:
        raise NotImplementedError(
            f"{PLUGIN} is not installed on this build (#46: vllm 0.28.0 hands "
            f"back no dense LSE) — this kind is unreachable, not attachable")

    def apply(self, attached: Any, request: Request) -> Levers:
        raise NotImplementedError(
            f"{PLUGIN} is not installed on this build — nothing ever attaches, "
            f"so no request carries this kind's levers")
