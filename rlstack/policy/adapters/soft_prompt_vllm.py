"""The soft prompt's rollout lowering: vLLM's MIXED embeds prompt.

demands enable_prompt_embeds — a build not asked for it reports NONE at the
embedding boundary, so a soft-prompt tenant is refused at Phase 0 rather than
served wrong; attach fuses the bank's row blocks into ONE prefix in BANK ORDER;
apply contributes the mixed prompt (rows, placeholder ids, the per-position
is_token_ids mask); align contributes n, so an answer read off the prompt is
found after the rows.

The mixed form is why nothing depends on our copy of the embedding matrix: the
engine embeds every position marked as a token id from its OWN table, so the
learned rows are the only thing handed over, and the real ids stay in the
request where prefix-cache hashing, detokenization and prompt_logprobs indexing
can all see them. torch is imported at module scope — this file loads only from
SoftPrompt.rollout_lowering (STYLE rule 7).
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

    adapter_type = "soft_prompt"
    mechanism = Mechanism.PROMPT_EMBEDS
    claims = ("prompt",)               # it shapes the prompt FORM, so it owns it

    def demands(self) -> BuildDemands:
        """One engine arg, and its stated cost: enable_prompt_embeds also gives
        up the V2 model runner and moves the embedding layer outside the CUDA
        graph."""
        return BuildDemands(engine_args={"enable_prompt_embeds": True})

    def reaches(self, meta: SiteMeta) -> bool:
        """One boundary, the one this adapter type exports at: the served
        model's embedding table. Nothing else is a place a row can enter."""
        return meta.path == EMBED_BOUNDARY

    def attach(self, bundle_id: str,
               payloads: Mapping[str, bytes]) -> torch.Tensor:
        """The bank's soft prompts, fused into ONE virtual prefix in the served
        model's dtype — precomputed once, so every request pinning this bundle
        just concatenates. Two entries are two segments of one virtual prompt
        and merge in BANK ORDER.

        A width that is not the base's hidden size cannot be an embedding row,
        so it dies here at attach rather than inside the engine core, which
        simply crashes on a wrong shape.
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
