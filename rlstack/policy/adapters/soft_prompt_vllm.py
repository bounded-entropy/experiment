"""The soft prompt's ROLLOUT lowering: vLLM 0.28's MIXED embeds prompt
(#48, #46, #3).

The mirror of soft_prompt_torch, on the other side of the bridge (I2). The
replay half prepends the rows at the embedding boundary and cuts the positions
back off the logits; this half hands the same rows to the engine as the first
positions of the prompt — and then tells the engine how many positions it took,
because a virtual row is a position with NO token and every answer read off the
prompt has to start after them.

WHY THE MIXED FORM, and it is better than embedding the whole prompt ourselves
(verified in the pinned image, #46): EmbedsPrompt takes prompt_embeds AND
prompt_token_ids AND a per-position prompt_is_token_ids mask. The ENGINE embeds
every position marked True from its OWN table, so the learned rows are the only
thing we hand over and nothing depends on our copy of the embedding matrix
matching the served one. The real token ids stay in the request, so prefix-cache
block hashes, detokenization and prompt_logprobs indexing all see the tokens
they would see without a soft prompt — and the block hash digests the embeds
themselves, so two bundles' prefixes can never alias.

The four verbs:
  demands   enable_prompt_embeds (which on 0.28 also excludes the V2 model
            runner and moves the embedding layer outside the CUDA graph) — a
            build that was not asked for it reports NONE at the boundary and
            refuses a soft-prompt tenant at Phase 0 rather than serving it
            wrong (#25b, reachability as a build fact).
  attach    the bank's row blocks fused into ONE prefix, in BANK ORDER
            (merge_rows), in the served model's dtype.
  apply     the mixed prompt: rows, placeholder ids, the mask.
  align     n — the positions the rows occupy.

torch is imported at module scope, so this file loads only from the kind's
methods (SoftPrompt.rollout_lowering), never from the package root — the
soft_prompt_torch precedent, STYLE rule 7.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from rlstack.policy.adapters import soft_prompt_torch
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.rollout import (
    Alignment, BuildDemands, Levers, Request, RolloutLowering,
)
from rlstack.policy.siteschema import SiteMeta

EMBED_BOUNDARY = "model.embed_tokens"   # the path a soft prompt exports at


class SoftPromptRollout(RolloutLowering):
    """Learned rows served as the prompt's first positions."""

    kind = "soft_prompt"
    mechanism = Mechanism.PROMPT_EMBEDS
    claims = ("prompt",)               # it shapes the prompt FORM, so it owns it

    def demands(self) -> BuildDemands:
        return BuildDemands(engine_args={"enable_prompt_embeds": True})

    def reaches(self, meta: SiteMeta) -> bool:
        """One boundary, the one this kind exports at: the served model's
        embedding table. Nothing else is a place a row can enter."""
        return meta.path == EMBED_BOUNDARY

    def attach(self, bundle_id: str,
               payloads: Mapping[str, bytes]) -> torch.Tensor:
        """The bank's soft prompts, fused into ONE virtual prefix in the served
        model's dtype — precomputed once so every request pinning this bundle
        just concatenates. Two entries are two segments of one virtual prompt
        and merge in BANK ORDER (#48's composition rule, within a kind).

        A width that is not the base's hidden size cannot be an embedding row,
        so it dies at registration rather than inside the engine core (which the
        0.28 docs warn will simply crash on a wrong shape).
        """
        rows = soft_prompt_torch.merge_rows(payloads)
        width = int(self.build.config.hidden_size)
        if int(rows.shape[1]) != width:
            raise ValueError(
                f"bundle {bundle_id!r} carries {rows.shape[1]}-wide prompt "
                f"rows; {self.build.base} embeds at {width}")
        return rows.to(self._served_dtype())

    def apply(self, attached: torch.Tensor, request: Request) -> Levers:
        """The mixed prompt: our rows at the False positions, the request's own
        tokens at the True ones (the engine embeds those itself)."""
        rows = attached
        n, width = rows.shape
        token_ids = list(request.token_ids)
        embeds = torch.cat(
            [rows, torch.zeros(len(token_ids), width, dtype=rows.dtype)])
        return Levers(prompt={
            "prompt_embeds": embeds,
            "prompt_token_ids": [soft_prompt_torch.VIRTUAL_TOKEN] * n + token_ids,
            "prompt_is_token_ids": [False] * n + [True] * len(token_ids)})

    def align(self, attached: torch.Tensor) -> Alignment:
        """The rows are prompt positions with no token: everything read off the
        prompt starts n later."""
        return Alignment(int(attached.shape[0]))

    def _served_dtype(self) -> Any:
        """The dtype the served weights carry — a virtual row IS an embedding
        and must arrive in the same one."""
        declared = (getattr(self.build.config, "dtype", None)
                    or getattr(self.build.config, "torch_dtype", None))
        if isinstance(declared, str):
            return getattr(torch, declared)
        return declared or torch.bfloat16
