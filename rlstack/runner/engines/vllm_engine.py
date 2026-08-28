"""VllmEngine: the Engine protocol over vLLM — a BUS, not a consumer.

Every adapter type's serving knowledge lives in its own rollout lowering
(policy/adapters/<adapter_type>_vllm.py), so this file knows no mechanism at
all. What it knows is the shape of the work:

  BUILD        every served adapter type's demands(), folded into the engine
               args; reachability is the union of what those payments bought.
  add_bundle   each of the bundle's adapter types attach()es its own payloads,
               additively and idempotently — a bundle this engine cannot
               express is refused HERE, while it is still just an id.
  a request    every attached adapter type's apply(), merged into ONE unit of
               work. Requests pinning different bundles batch together in vLLM's
               own scheduler: the multi-tenancy invariant, held by construction.
  an answer    every attached adapter type's align(), SUMMED, because a scored
               suffix sits after everything the bundle put in front of it.

Two build facts off to the side. enable_sleep_mode buys the sleep()/wake() pair
an alternating host's arbiter hooks call to make a partition really hand the
device back — deliberately NOT on the Engine protocol, since a fake or a remote
pool has no device to give. And no engine plugin is installed here, so an
adapter type whose demands() name one cannot be served on this build.

v0 choice, stated: prompts are RAW token concatenations of the messages, each
tokenized exactly as flatten will re-tokenize it — no chat template.
"""

from __future__ import annotations

import tempfile
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

from rlstack.data.trajectory import Message
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.rollout import (
    Levers, Request, RolloutLowering, ServingBuild, check_levers_compose,
)
from rlstack.policy.compile import Bundle, group_by_adapter_type
from rlstack.registry import ADAPTER_TYPES
from rlstack.runner.interfaces import FinishEvent, TokenEvent
from rlstack.spec.specs import SamplingSpec


class VllmEngine:
    def __init__(self, base: str, *, gpu_memory_utilization: float = 0.45,
                 max_model_len: int = 1024, max_bundles: int = 8,
                 max_rank: int = 32, enforce_eager: bool = True,
                 tp: int = 1, serves: Sequence[str] = ("lora",),
                 enable_sleep_mode: bool = False) -> None:
        from transformers import AutoConfig, AutoTokenizer

        self.base = base
        self.tp = tp                # build fact: tensor-parallel width
        self.serves = tuple(serves)  # build fact: the ADAPTER TYPES this build pays for
        # build fact: whether this engine can hand the device back. Not an
        # adapter type's demand (no lowering asks for it) and not part of the
        # Engine protocol — a capability of THIS build, reached by the deploy
        # that owns the metal and wired into its host's arbiter hooks.
        self.sleeps = enable_sleep_mode
        self._tokenizer = AutoTokenizer.from_pretrained(base)
        self._config = AutoConfig.from_pretrained(base)
        self._workdir = Path(tempfile.mkdtemp(prefix="rlstack-bundles-"))
        # max_bundles / max_rank are plain capacity, spelled the way
        # ServingBuild spells them: how many bundles' state may be resident and
        # the widest delta rank served. Only the adapter type that SPENDS them
        # turns them into an engine arg (lora_vllm names vLLM's own max_loras /
        # max_lora_rank), so no mechanism-flavored word survives in this file.
        self._build = ServingBuild(base=base, config=self._config,
                                   workdir=self._workdir,
                                   max_bundles=max_bundles, max_rank=max_rank)
        self._lowerings = self._serving_adapter_types()
        self._engine_args = dict(
            model=base, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tp, enable_sleep_mode=enable_sleep_mode,
            enforce_eager=enforce_eager, disable_log_stats=True)
        for lowering in self._lowerings.values():
            self._engine_args.update(self._pay_demands(lowering))
        self._llm = None                       # built inside the running loop
        self._asleep = False                   # the sleep seam's own state
        self._known: set[str] = set()          # every registered bundle_id
        # id -> adapter type -> state
        self._attached: dict[str, dict[str, object]] = {}
        self._request_count = 0

    def _serving_adapter_types(self) -> dict[str, RolloutLowering]:
        """One rollout lowering per adapter type this build was asked to serve,
        built by the adapter type itself — the bus never names a mechanism, only
        the adapter types it pays for."""
        return {adapter_type:
                ADAPTER_TYPES.get(adapter_type).instance.rollout_lowering(self._build)
                for adapter_type in self.serves}

    def _pay_demands(self, lowering: RolloutLowering) -> Mapping[str, object]:
        """What this build owes one adapter type, or a loud refusal.

        Engine args it can pay by construction; a PLUGIN it cannot, since none
        ships installed here — so an adapter type demanding one is refused at
        build time (I7: die at boot, never mid-run) rather than reporting a
        reachability it could not honor.
        """
        demands = lowering.demands()
        if demands.plugin is not None:
            raise NotImplementedError(
                f"adapter type {lowering.adapter_type!r} demands the engine plugin "
                f"{demands.plugin!r}; this build installs none, so it cannot "
                f"serve it (reachability stays NONE and Phase 0 refuses)")
        return demands.engine_args

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
        """This build's inventory: the union of what its served adapter types
        reach, asked in build order.

        An adapter type this build did not pay for is not in the union at all,
        so its sites come back NONE and a tenant needing one is refused at Phase
        0 rather than served wrong.
        """
        def reach(meta) -> Mechanism:
            for lowering in self._lowerings.values():
                if lowering.reaches(meta):
                    return lowering.mechanism
            return Mechanism.NONE
        return {meta.name: reach(meta) for meta in sites}

    def add_bundle(self, bundle: Bundle) -> None:
        """THE bus loop: every adapter type in the bundle, made resident through
        its own lowering.

        A bank of two adapter types attaches two pieces of state and its requests
        carry both adapter types' levers. Registration is additive and idempotent
        (I8), and a bundle whose adapter types cannot compose into one request is
        refused here, before any request can pin it.
        """
        if bundle.bundle_id in self._known:
            return
        payloads = group_by_adapter_type(bundle)
        lowerings = [self._lowering_for(bundle.bundle_id, adapter_type)
                     for adapter_type in payloads]
        check_levers_compose(bundle.bundle_id, lowerings)
        self._attached[bundle.bundle_id] = {
            lowering.adapter_type: lowering.attach(
                bundle.bundle_id, payloads[lowering.adapter_type])
            for lowering in lowerings}
        self._known.add(bundle.bundle_id)   # no payloads: serve the bare base

    def _lowering_for(self, bundle_id: str, adapter_type: str) -> RolloutLowering:
        """The lowering that serves `adapter_type` here, or the honest refusal."""
        if adapter_type not in self._lowerings:
            raise NotImplementedError(
                f"bundle {bundle_id!r} carries {adapter_type!r} state, but this "
                f"build serves {list(self.serves)} — reachability said so at "
                f"Phase 0; serving it now would be a lie")
        return self._lowerings[adapter_type]

    def attachments(self, bundle_id: str) -> Mapping[str, object]:
        """What each adapter type made resident for this bundle (adapter type ->
        its state): the bus's own inventory, read by probes. Requests go through
        levers."""
        return self._attached.get(bundle_id, {})

    def residency(self) -> Mapping[str, int]:
        """How many registered bundles each served adapter type holds state for —
        the census of I8 on this engine, one number per adapter type."""
        counts = {adapter_type: 0 for adapter_type in self.serves}
        for attached in self._attached.values():
            for adapter_type in attached:
                counts[adapter_type] += 1
        return counts

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
        levers = self._levers_for(prompt_ids, bundle_id)

        emitted = 0
        text_len = 0
        final = None
        async for output in self._ensure_llm().generate(
                levers.prompt, params, request_id, **levers.kwargs):
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
        returns each prompt position's logprob under the pinned bundle, and we
        read off the scored suffix. max_tokens=1 because vLLM must generate
        something — the one decoded token is discarded."""
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
        levers = self._levers_for(full_ids, bundle_id)

        final = None
        async for output in self._ensure_llm().generate(
                levers.prompt, params, request_id, **levers.kwargs):
            final = output
        assert final is not None and final.prompt_logprobs is not None
        # the scored suffix sits after the context — and after everything the
        # bundle's adapter types put in front of it
        start = self._occupied(bundle_id) + len(context_ids)
        return tuple(
            float(final.prompt_logprobs[start + j][int(tok)].logprob)
            for j, tok in enumerate(token_ids))

    # ---- the sleep seam (a build capability, NOT the Engine protocol) -------

    async def sleep(self) -> None:
        """Hand the device back: vLLM's sleep(level=1) offloads the weights to
        host RAM and DISCARDS the KV cache. THE evict verb an alternating
        host's arbiter hook calls.

        Idempotent, and quiet on an engine that was never built: the vLLM
        build is lazy, so an evict can arrive before the first sample, and
        nothing resident is nothing to offload."""
        self.check_sleeps()
        if self._llm is None or self._asleep:
            return
        await self._llm.sleep(1)
        self._asleep = True

    async def wake(self) -> None:
        """The other half: wake_up() puts the weights back and re-allocates
        the KV cache. Idempotent, and the exact inverse of sleep() — the
        arbiter only switches when in-flight work is zero, so no request ever
        meets a half-woken engine."""
        self.check_sleeps()
        if self._llm is None or not self._asleep:
            return
        await self._llm.wake_up()
        self._asleep = False

    def check_sleeps(self) -> None:
        """Alternation is a BUILD fact: vLLM allocates its weights into a
        releasable memory pool only when built with enable_sleep_mode. A build
        that did not pay for it cannot sleep, and says so instead of
        pretending it did."""
        if not self.sleeps:
            raise RuntimeError(
                f"engine for {self.base!r} was built without "
                f"enable_sleep_mode: it cannot hand the device back. "
                f"Alternation is a build fact — build it "
                f"VllmEngine(..., enable_sleep_mode=True)")

    # ---- one unit of work ---------------------------------------------------

    def _levers_for(self, prompt_ids: list[int], bundle_id: str) -> Levers:
        """Every attached adapter type's apply(), merged into ONE request —
        where a bundle becomes USED (I8).

        The adapter types are asked in BANK ORDER, and one that shapes the prompt
        form replaces the plain token prompt the bus would otherwise send — which
        is unambiguous because add_bundle refused any bundle whose adapter types
        claim the same lever. Nothing attached: the raw token ids, as if no bank
        existed.
        """
        from vllm import TokensPrompt

        request = Request(token_ids=tuple(prompt_ids))
        levers = Levers(prompt=TokensPrompt(prompt_token_ids=prompt_ids))
        for adapter_type, attached in self._attached[bundle_id].items():
            levers = levers.merged_with(
                self._lowerings[adapter_type].apply(attached, request))
        return levers

    def _occupied(self, bundle_id: str) -> int:
        """Prompt positions this bundle's state occupies before the request's
        own tokens — SUMMED across its adapter types, because they all sit in
        front of the same first real token."""
        return sum(self._lowerings[adapter_type].align(attached).positions
                   for adapter_type, attached
                   in self._attached[bundle_id].items())


def _finish_event(completion) -> FinishEvent:
    if completion is None:
        return FinishEvent("length")
    if completion.finish_reason == "length":
        return FinishEvent("length")
    if isinstance(completion.stop_reason, str):
        return FinishEvent("stop", stop_hit=completion.stop_reason)
    return FinishEvent("eos")
