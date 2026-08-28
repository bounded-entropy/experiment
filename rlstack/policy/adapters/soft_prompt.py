"""Learned virtual prompt rows; served natively via vLLM prompt_embeds.

The one builtin that EXPORTS sites: the base checkpoint has no prompt
positions — this entry creates them, so their names resolve only when a soft
prompt is in the bank. Alongside the positions themselves it exports the
queries -> prompt[:n] attention rectangle, the site an attn_bias attaches to
(served jointly with this entry by the side_attention plugin).

Its two lowerings prepend the same rows at the same place. ROLLOUT: vLLM
0.28's mixed embeds prompt — the engine embeds the request's token ids from
its OWN table and takes only the learned rows from us
(soft_prompt_vllm.SoftPromptRollout). REPLAY: a boundary around the trainer's forward
that prepends the rows and cuts the positions back off the logits
(soft_prompt_torch.PromptBoundary), so the [len(batch)] alignment above it is
untouched.
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

    # compute halves — both import torch, so both load lazily, from here
    # only (rule 7)

    def rollout_lowering(self, build):
        from rlstack.policy.adapters import soft_prompt_vllm
        return soft_prompt_vllm.SoftPromptRollout(build)

    def params(self, sites: tuple[SiteMeta, ...], init: dict):
        from rlstack.policy.adapters import soft_prompt_torch
        return soft_prompt_torch.build(sites, init)

    def install_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import soft_prompt_torch
        soft_prompt_torch.install(model, params)

    def uninstall_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import soft_prompt_torch
        soft_prompt_torch.uninstall(model, params)

    def emit(self, params) -> bytes:
        from rlstack.policy.adapters import soft_prompt_torch
        return soft_prompt_torch.emit(params)

    def load(self, params, payload: bytes) -> None:
        from rlstack.policy.adapters import soft_prompt_torch
        soft_prompt_torch.load(params, payload)
