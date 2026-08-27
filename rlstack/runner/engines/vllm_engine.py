"""VllmEngine: the Engine protocol over vLLM (Phase B3, v0).

The two enum-conditioned functions live here and nowhere else: add_bundle
routes payloads BY MECHANISM to their consumers (punica: merge peft fragments
→ LoRARequest; prompt_embeds: merge row blocks → a virtual prefix), and
sample_tokens turns a pinned bundle into request fields (prompt form,
lora_request, seed, stop). The multi-tenancy invariant holds by construction:
add_bundle only ever ADDS a registration, and every request carries its own
bundle's levers, batched by vLLM's own scheduler — including requests pinning
bundles of DIFFERENT kinds, which is the point of two consumers on one engine.

REACHABILITY IS A BUILD FACT (#25b), and here it is literally a constructor
argument: prompt_embeds costs an engine arg (`enable_prompt_embeds`, which on
0.28 also excludes the V2 model runner), so a build that was not asked for it
honestly reports NONE at the embedding boundary and a soft-prompt tenant is
refused at Phase 0 rather than served wrong.

v0 choices, stated: prompts are RAW token concatenations of the messages —
each message tokenized separately, exactly as flatten will re-tokenize it (no
chat template; template-faithful rendering is a logged open thread); uniform
LoRA rank across bank entries (the merged adapter_config carries one r); plugin
mechanisms are not installed, so SIDE_ATTENTION is reported NONE.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

from rlstack.data.trajectory import Message
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.compile import Bundle, group_by_mechanism
from rlstack.runner.interfaces import FinishEvent, TokenEvent
from rlstack.spec.specs import SamplingSpec

EMBED_BOUNDARY = "model.embed_tokens"   # the path a soft prompt exports at


class VllmEngine:
    def __init__(self, base: str, *, gpu_memory_utilization: float = 0.45,
                 max_model_len: int = 1024, max_loras: int = 8,
                 max_lora_rank: int = 32, enforce_eager: bool = True,
                 tp: int = 1, prompt_embeds: bool = False) -> None:
        from transformers import AutoConfig, AutoTokenizer

        self.base = base
        self.tp = tp                # build fact (#43): tensor-parallel width
        self.prompt_embeds = prompt_embeds   # build fact: the soft-prompt lever
        self._engine_args = dict(
            model=base, enable_lora=True, max_loras=max_loras,
            max_lora_rank=max_lora_rank, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tp, enable_prompt_embeds=prompt_embeds,
            enforce_eager=enforce_eager, disable_log_stats=True)
        self._tokenizer = AutoTokenizer.from_pretrained(base)
        self._config = AutoConfig.from_pretrained(base)
        self._llm = None                       # built inside the running loop
        self._known: set[str] = set()          # every registered bundle_id
        self._lora: dict[str, object] = {}     # bundle_id -> LoRARequest
        self._rows: dict[str, object] = {}     # bundle_id -> [n, d] prefix rows
        self._next_int_id = 1
        self._workdir = Path(tempfile.mkdtemp(prefix="rlstack-bundles-"))
        self._request_count = 0

    def _ensure_llm(self):
        """AsyncLLMEngine wants a running event loop; the runner has one by
        the first sample — so construction is deferred to that moment."""
        if self._llm is None:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            self._llm = AsyncLLMEngine.from_engine_args(
                AsyncEngineArgs(**self._engine_args))
        return self._llm

    # ---- Engine protocol ----------------------------------------------------

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(self._tokenizer.encode(text, add_special_tokens=False))

    def reachability(self, sites) -> Mapping[str, Mechanism]:
        """This build's self-reported inventory: punica reaches every weighted
        matrix, the embedding boundary is reachable only if this engine was
        BUILT with prompt_embeds, and no plugin is installed — so the
        side-attention rectangle is honestly NONE."""
        def reach(meta) -> Mechanism:
            if meta.has_weight:
                return Mechanism.PUNICA
            if meta.path == EMBED_BOUNDARY and self.prompt_embeds:
                return Mechanism.PROMPT_EMBEDS
            return Mechanism.NONE
        return {meta.name: reach(meta) for meta in sites}

    def add_bundle(self, bundle: Bundle) -> None:
        """THE mechanism dispatch (rollout side): every consumer this build
        has, fed the whole of its own mechanism's payloads.

        A bundle may reach more than one consumer — a bank of lora + soft
        prompt registers a LoRARequest AND a virtual prefix, and its requests
        carry both. Registration is additive and idempotent (the invariant).
        """
        if bundle.bundle_id in self._known:
            return
        for mechanism, payloads in group_by_mechanism(bundle).items():
            if mechanism is Mechanism.PUNICA:
                self._lora[bundle.bundle_id] = self._register_lora(
                    bundle.bundle_id, payloads)
            elif mechanism is Mechanism.PROMPT_EMBEDS:
                self._rows[bundle.bundle_id] = self._register_prompt_rows(
                    bundle.bundle_id, payloads)
            else:
                raise NotImplementedError(
                    f"no consumer for {mechanism} on this engine (this build "
                    f"serves punica and prompt_embeds)")
        self._known.add(bundle.bundle_id)   # no payloads: serve the bare base

    async def sample_tokens(
        self,
        messages: Sequence[Message],
        sampling: SamplingSpec,
        stop: tuple[str, ...],
        bundle_id: str,
        seed: int,
    ) -> AsyncIterator[TokenEvent | FinishEvent]:
        if bundle_id not in self._known:
            raise RuntimeError(f"bundle {bundle_id!r} was never registered")
        from vllm import SamplingParams

        # Each message tokenized separately, then concatenated — the EXACT
        # ids flatten reproduces for injected spans (no template drift).
        prompt_ids = [tid for m in messages for tid in self.tokenize(m.content)]
        params = SamplingParams(
            temperature=sampling.temperature, top_p=sampling.top_p,
            max_tokens=sampling.max_tokens, stop=list(stop), seed=seed,
            logprobs=0)
        self._request_count += 1
        request_id = f"rlstack-{self._request_count}-{seed}"

        emitted = 0
        text_len = 0
        final = None
        async for output in self._ensure_llm().generate(
                self._prompt_for(prompt_ids, bundle_id), params, request_id,
                lora_request=self._lora.get(bundle_id)):
            completion = output.outputs[0]
            new_ids = list(completion.token_ids)[emitted:]
            for offset, token_id in enumerate(new_ids):
                position = emitted + offset
                logprob = completion.logprobs[position][token_id].logprob
                # attribute the whole text delta to the last new token: the
                # concatenation (the Message content) stays exact either way
                last = offset == len(new_ids) - 1
                delta = completion.text[text_len:] if last else ""
                yield TokenEvent(token_id=int(token_id), logprob=float(logprob),
                                 text_delta=delta)
            emitted += len(new_ids)
            text_len = len(completion.text)
            final = completion
        yield _finish_event(final)

    async def score_tokens(self, messages: Sequence[Message],
                           token_ids: Sequence[int],
                           bundle_id: str) -> tuple[float, ...]:
        """ONE prefill over context + tokens with prompt_logprobs: vLLM
        returns each prompt position's logprob under the pinned bundle; we
        read off the scored suffix. max_tokens=1 because vLLM must generate
        something — the one decoded token is discarded. Scoring is prefill-
        shaped traffic: the natural tenant of a prefill-disaggregated pool."""
        if bundle_id not in self._known:
            raise RuntimeError(f"bundle {bundle_id!r} was never registered")
        if not token_ids:
            return ()
        from vllm import SamplingParams

        context_ids = [tid for m in messages for tid in self.tokenize(m.content)]
        full_ids = context_ids + [int(t) for t in token_ids]
        params = SamplingParams(max_tokens=1, temperature=0.0,
                                prompt_logprobs=0)
        self._request_count += 1
        request_id = f"rlstack-score-{self._request_count}"

        final = None
        async for output in self._ensure_llm().generate(
                self._prompt_for(full_ids, bundle_id), params, request_id,
                lora_request=self._lora.get(bundle_id)):
            final = output
        assert final is not None and final.prompt_logprobs is not None
        # the scored suffix sits after the context — and after the bundle's
        # virtual rows, which occupy prompt positions of their own
        start = self._virtual_rows(bundle_id) + len(context_ids)
        return tuple(
            float(final.prompt_logprobs[start + j][int(tok)].logprob)
            for j, tok in enumerate(token_ids))

    # ---- the request's prompt form ------------------------------------------

    def _virtual_rows(self, bundle_id: str) -> int:
        """How many prompt positions this bundle's soft prompts occupy."""
        rows = self._rows.get(bundle_id)
        return 0 if rows is None else int(rows.shape[0])

    def _prompt_for(self, prompt_ids: list[int], bundle_id: str):
        """The prompt in the form the pinned bundle needs (I8: the request is
        where a bundle becomes USED).

        No learned rows: the raw token ids. With rows: vLLM 0.28's MIXED embeds
        prompt — a full-length embeds tensor plus the token ids and a
        per-position `prompt_is_token_ids` mask. The ENGINE embeds every
        position marked True from its own table (so the rows are the only thing
        we hand over, and nothing depends on our copy of the embedding matrix
        matching the served one), while the False positions take our rows. The
        real token ids stay in the request, so prefix-cache block hashes,
        detokenization and prompt_logprobs indexing all see the tokens they
        would see without a soft prompt — and the block hash also digests the
        embeds themselves, so two bundles' prefixes can never alias.
        """
        rows = self._rows.get(bundle_id)
        if rows is None:
            from vllm import TokensPrompt
            return TokensPrompt(prompt_token_ids=prompt_ids)
        import torch

        from rlstack.policy.adapters.soft_prompt_torch import VIRTUAL_TOKEN

        n, width = rows.shape
        embeds = torch.cat(
            [rows, torch.zeros(len(prompt_ids), width, dtype=rows.dtype)])
        return {"prompt_embeds": embeds,
                "prompt_token_ids": [VIRTUAL_TOKEN] * n + list(prompt_ids),
                "prompt_is_token_ids": [False] * n + [True] * len(prompt_ids)}

    # ---- the punica consumer ------------------------------------------------

    def _register_lora(self, bundle_id: str, payloads: dict[str, bytes]):
        from vllm.lora.request import LoRARequest

        from rlstack.policy.adapters import lora_torch

        merged, leaves, rank = lora_torch.merge_fragments(payloads)
        adapter_dir = self._workdir / bundle_id.replace(":", "_")
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_model.safetensors").write_bytes(merged)
        (adapter_dir / "adapter_config.json").write_text(
            lora_torch.peft_config(self.base, rank, leaves))
        request = LoRARequest(lora_name=bundle_id,
                              lora_int_id=self._next_int_id,
                              lora_path=str(adapter_dir))
        self._next_int_id += 1
        return request

    # ---- the prompt_embeds consumer -----------------------------------------

    def _register_prompt_rows(self, bundle_id: str, payloads: dict[str, bytes]):
        """The bank's soft prompts, fused into ONE virtual prefix in the
        served model's dtype — precomputed here so every request pinning this
        bundle just concatenates.

        A width that is not the base's hidden size cannot be an embedding row,
        so it dies at registration rather than inside the engine core (which
        the 0.28 docs warn will simply crash on a wrong shape).
        """
        if not self.prompt_embeds:
            raise RuntimeError(
                f"bundle {bundle_id!r} carries soft-prompt rows, but this "
                f"engine was built without prompt_embeds — reachability said "
                f"so at Phase 0; serving it now would be a lie")
        from rlstack.policy.adapters import soft_prompt_torch

        rows = soft_prompt_torch.merge_rows(payloads)
        width = int(self._config.hidden_size)
        if int(rows.shape[1]) != width:
            raise ValueError(
                f"bundle {bundle_id!r} carries {rows.shape[1]}-wide prompt "
                f"rows; {self.base} embeds at {width}")
        return rows.to(self._model_dtype())

    def _model_dtype(self):
        """The dtype the served weights carry — a virtual row is an embedding
        and must arrive in the same one."""
        import torch

        declared = getattr(self._config, "dtype", None) or getattr(
            self._config, "torch_dtype", None)
        if isinstance(declared, str):
            return getattr(torch, declared)
        return declared or torch.bfloat16


def _finish_event(completion) -> FinishEvent:
    if completion is None:
        return FinishEvent("length")
    if completion.finish_reason == "length":
        return FinishEvent("length")
    if isinstance(completion.stop_reason, str):
        return FinishEvent("stop", stop_hit=completion.stop_reason)
    return FinishEvent("eos")
