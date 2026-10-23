"""Learned virtual prompt rows; served natively via vLLM prompt_embeds.

The one builtin that EXPORTS sites: the base checkpoint has no prompt
positions, so this entry creates them — its own prompt[:n] positions, and the
queries -> prompt[:n] attention rectangle an attn_bias attaches to — and those
names resolve only when a soft prompt is in the bank. Both lowerings prepend
the same rows at the same place.
"""

from __future__ import annotations

from rlstack.policy.adapters.definition import Operation, SiteKind, builtin
from rlstack.policy.adapters.base import AdapterType, Mechanism, adapter_type
from rlstack.policy.siteschema import SiteMeta
from rlstack.spec.specs import AdapterSpec


@adapter_type("soft_prompt")
class SoftPrompt(AdapterType):
    definition = builtin(Operation.PROMPT, SiteKind.PROMPT, 'soft_prompt',
                         'soft_prompt', 'SoftPromptRollout', 'prompt_embeds')
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
