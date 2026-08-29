"""attn_bias's rollout lowering: the one whose demands NO build here can pay.

Its mechanism is a PLUGIN — ours, not a lever vLLM maintains — and the pinned
build does not plumb the dense attention LSE the plugin was designed against,
so probe() fails for the true reason (rlstack_engine.side_attention names the
missing symbols). The consequences reach a spec through the same door as every
other adapter type: no build serves this adapter type, reachability reports
NONE for the attention-score rectangle, and a spec carrying an attn_bias is
refused at Phase 0 rather than served wrong.

The plugin is named by STRING and never imported — rlstack_engine ships in the
engine image and the import direction is one-way.
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
    """Soft prompt + bias, served jointly by the side-attention plugin — the
    lowering no build in this repo can pay for."""

    adapter_type = "attn_bias"
    mechanism = Mechanism.SIDE_ATTENTION
    claims = ()          # unreachable, so it claims nothing of any request yet

    def demands(self) -> BuildDemands:
        """A plugin in the engine image, installed and probed at boot. No build
        here installs one, so asking to serve this adapter type refuses at
        construction
        — the honest form of a blocked mechanism."""
        return BuildDemands(plugin=PLUGIN)

    def reaches(self, meta: SiteMeta) -> bool:
        """What the payment would buy: the queries -> prompt[:n] rectangle the
        soft prompt exports."""
        return meta.path == ATTN_SCORES

    def attach(self, bundle_id: str, payloads: Mapping[str, bytes]) -> Any:
        raise NotImplementedError(
            f"{PLUGIN} is not installed on this build (#46: vllm 0.28.0 hands "
            f"back no dense LSE) — this adapter type is unreachable, not "
            f"attachable")

    def apply(self, attached: Any, request: Request) -> Levers:
        raise NotImplementedError(
            f"{PLUGIN} is not installed on this build — nothing ever attaches, "
            f"so no request carries this adapter type's levers")

    def detach(self, attached: Any) -> None:
        """Nothing ever attached, so nothing is ever released — the honest
        inverse of an attach that refuses."""
