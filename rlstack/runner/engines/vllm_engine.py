"""VllmEngine: the Engine protocol over vLLM (Phase B3, v0).

The two enum-conditioned functions live here and nowhere else: add_bundle
routes payloads BY MECHANISM to their consumers (punica: merge peft fragments
→ LoRARequest), and sample_tokens turns a pinned bundle into request fields
(lora_request, seed, stop). The multi-tenancy invariant holds by construction:
add_bundle only ever ADDS a registration, and every request carries its own
bundle's LoRARequest, batched by vLLM's own scheduler.

v0 choices, stated: prompts are RAW token concatenations of the messages —
each message tokenized separately, exactly as flatten will re-tokenize it (no
chat template; template-faithful rendering is a logged open thread); uniform
LoRA rank across bank entries (the merged adapter_config carries one r);
reachability reports punica on weighted sites and NONE elsewhere (no plugin
mechanisms are installed in v0).
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


class VllmEngine:
    def __init__(self, base: str, *, gpu_memory_utilization: float = 0.45,
                 max_model_len: int = 1024, max_loras: int = 8,
                 max_lora_rank: int = 32, enforce_eager: bool = True) -> None:
        from transformers import AutoTokenizer

        self.base = base
        self._engine_args = dict(
            model=base, enable_lora=True, max_loras=max_loras,
            max_lora_rank=max_lora_rank, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager, disable_log_stats=True)
        self._tokenizer = AutoTokenizer.from_pretrained(base)
        self._llm = None                       # built inside the running loop
        self._lora: dict[str, object] = {}     # bundle_id -> LoRARequest | None
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
        return {meta.name: Mechanism.PUNICA if meta.has_weight else Mechanism.NONE
                for meta in sites}

    def add_bundle(self, bundle: Bundle) -> None:
        """THE mechanism dispatch (rollout side)."""
        if bundle.bundle_id in self._lora:
            return  # additive and idempotent
        if not bundle.payloads:
            self._lora[bundle.bundle_id] = None          # serve the bare base
            return
        for mechanism, payloads in group_by_mechanism(bundle).items():
            if mechanism is Mechanism.PUNICA:
                self._lora[bundle.bundle_id] = self._register_lora(
                    bundle.bundle_id, payloads)
            else:
                raise NotImplementedError(
                    f"no consumer for {mechanism} on this engine (v0 serves "
                    f"punica only)")

    async def sample_tokens(
        self,
        messages: Sequence[Message],
        sampling: SamplingSpec,
        stop: tuple[str, ...],
        bundle_id: str,
        seed: int,
    ) -> AsyncIterator[TokenEvent | FinishEvent]:
        if bundle_id not in self._lora:
            raise RuntimeError(f"bundle {bundle_id!r} was never registered")
        from vllm import SamplingParams, TokensPrompt

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
                TokensPrompt(prompt_token_ids=prompt_ids), params, request_id,
                lora_request=self._lora[bundle_id]):
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
        if bundle_id not in self._lora:
            raise RuntimeError(f"bundle {bundle_id!r} was never registered")
        if not token_ids:
            return ()
        from vllm import SamplingParams, TokensPrompt

        context_ids = [tid for m in messages for tid in self.tokenize(m.content)]
        full_ids = context_ids + [int(t) for t in token_ids]
        params = SamplingParams(max_tokens=1, temperature=0.0,
                                prompt_logprobs=0)
        self._request_count += 1
        request_id = f"rlstack-score-{self._request_count}"

        final = None
        async for output in self._ensure_llm().generate(
                TokensPrompt(prompt_token_ids=full_ids), params, request_id,
                lora_request=self._lora[bundle_id]):
            final = output
        assert final is not None and final.prompt_logprobs is not None
        start = len(context_ids)
        return tuple(
            float(final.prompt_logprobs[start + j][int(tok)].logprob)
            for j, tok in enumerate(token_ids))

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


def _finish_event(completion) -> FinishEvent:
    if completion is None:
        return FinishEvent("length")
    if completion.finish_reason == "length":
        return FinishEvent("length")
    if isinstance(completion.stop_reason, str):
        return FinishEvent("stop", stop_hit=completion.stop_reason)
    return FinishEvent("eos")
