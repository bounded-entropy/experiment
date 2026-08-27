"""Learned virtual prompt rows; served natively via vLLM prompt_embeds.

The one builtin that EXPORTS sites: the base checkpoint has no prompt
positions — this entry creates them, so their names resolve only when a soft
prompt is in the bank. Alongside the positions themselves it exports the
queries -> prompt[:n] attention rectangle, the site an attn_bias attaches to
(served jointly with this entry by the side_attention plugin).
"""

from __future__ import annotations

from rlstack.policy.adapters.base import Adapter, Mechanism, adapter
from rlstack.policy.siteschema import SiteMeta
from rlstack.spec.specs import AdapterSpec


@adapter("soft_prompt")
class SoftPrompt(Adapter):
    serving = Mechanism.PROMPT_EMBEDS

    def site_ok(self, meta: SiteMeta) -> bool:
        return not meta.has_weight

    def exports(self, spec: AdapterSpec) -> tuple[SiteMeta, ...]:
        n = int(spec.init["n"])
        return (
            SiteMeta(name=f"prompt[:{n}]", path="model.embed_tokens",
                     has_weight=False, shape=None, is_boundary=True),
            SiteMeta(name=f"queries -> prompt[:{n}]", path="attn_scores",
                     has_weight=False, shape=None, is_boundary=False),
        )
