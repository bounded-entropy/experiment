"""VllmEngine: the Engine protocol over vLLM — a BUS, not a consumer.

Every kind's serving knowledge lives in its own rollout lowering
(policy/adapters/<kind>_vllm.py), so this file knows no mechanism at all. What
it knows is the shape of the work:

  BUILD        every served kind's demands(), folded into the engine args;
               reachability is the union of what those payments bought.
  add_bundle   each of the bundle's kinds attach()es its own payloads,
               additively and idempotently — a bundle this engine cannot
               express is refused HERE, while it is still just an id.
  a request    every attached kind's apply(), merged into ONE unit of work.
               Requests pinning different bundles batch together in vLLM's own
               scheduler: the multi-tenancy invariant, held by construction.
  an answer    every attached kind's align(), SUMMED, because a scored suffix
               sits after everything the bundle put in front of it.

Two build facts off to the side. enable_sleep_mode buys the sleep()/wake() pair
an alternating host's arbiter hooks call to make a partition really hand the
device back — deliberately NOT on the Engine protocol, since a fake or a remote
pool has no device to give. And no engine plugin is installed here, so a kind
whose demands() name one cannot be served on this build.

v0 choice, stated: prompts are RAW token concatenations of the messages, each
tokenized exactly as flatten will re-tokenize it — no chat template.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

from rlstack.data.trajectory import Message
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.rollout import (
    Levers, Request, RolloutLowering, ServingBuild, check_levers_compose,
)
from rlstack.policy.compile import Bundle, group_by_kind
from rlstack.registry import ADAPTERS
from rlstack.runner.interfaces import FinishEvent, TokenEvent
from rlstack.runner.meters import TrafficMeter
from rlstack.spec.specs import SamplingSpec


class VllmEngine:
    def __init__(self, base: str, *, gpu_memory_utilization: float = 0.45,
                 max_model_len: int = 1024, max_loras: int = 8,
                 max_lora_rank: int = 32, enforce_eager: bool = True,
                 tp: int = 1, serves: Sequence[str] = ("lora",),
                 enable_sleep_mode: bool = False) -> None:
        from transformers import AutoConfig, AutoTokenizer

        self.base = base
        self.tp = tp                # build fact: tensor-parallel width
        self.serves = tuple(serves)  # build fact: the KINDS this build pays for
        # build fact: whether this engine can hand the device back. Not a
        # kind's demand (no lowering asks for it) and not part of the Engine
        # protocol — a capability of THIS build, reached by the deploy that
        # owns the metal and wired into its host's arbiter hooks.
        self.sleeps = enable_sleep_mode
        self._tokenizer = AutoTokenizer.from_pretrained(base)
        self._config = AutoConfig.from_pretrained(base)
        self._workdir = Path(tempfile.mkdtemp(prefix="rlstack-bundles-"))
        # max_loras / max_lora_rank keep their vLLM-flavored names because
        # every call site already spells them that way; the bus passes them on
        # as plain capacity and only the kind that spends them names an engine
        # arg. They are the one mechanism-flavored word left in this file.
        self._build = ServingBuild(base=base, config=self._config,
                                   workdir=self._workdir,
                                   max_bundles=max_loras, max_rank=max_lora_rank)
        self._lowerings = self._serving_kinds()
        self._engine_args = dict(
            model=base, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tp, enable_sleep_mode=enable_sleep_mode,
            enforce_eager=enforce_eager, disable_log_stats=True)
        for lowering in self._lowerings.values():
            self._engine_args.update(self._pay_demands(lowering))
        self._llm = None                       # built inside the running loop
        self._asleep = False                   # the sleep seam's own state
        # traffic is counted HERE, at our own seam — never inside vLLM, whose
        # statistics are version-coupled. A host wires its own meter in.
        self.meter = TrafficMeter()
        self._known: set[str] = set()          # every registered bundle_id
        self._attached: dict[str, dict[str, object]] = {}  # id -> kind -> state
        self._request_count = 0

    def _serving_kinds(self) -> dict[str, RolloutLowering]:
        """One rollout lowering per kind this build was asked to serve, built
        by the kind itself — the bus never names a mechanism, only the kinds
        it pays for."""
        return {kind: ADAPTERS.get(kind).instance.rollout_lowering(self._build)
                for kind in self.serves}

    def _pay_demands(self, lowering: RolloutLowering) -> Mapping[str, object]:
        """What this build owes one kind, or a loud refusal.

        Engine args it can pay by construction; a PLUGIN it cannot, since none
        ships installed here — so a kind demanding one is refused at build
        time (I7: die at boot, never mid-run) rather than reporting a
        reachability it could not honor.
        """
        demands = lowering.demands()
        if demands.plugin is not None:
            raise NotImplementedError(
                f"kind {lowering.kind!r} demands the engine plugin "
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
        """This build's inventory: the union of what its served kinds reach,
        asked in build order.

        A kind this build did not pay for is not in the union at all, so its
        sites come back NONE and a tenant needing one is refused at Phase 0
        rather than served wrong.
        """
        def reach(meta) -> Mechanism:
            for lowering in self._lowerings.values():
                if lowering.reaches(meta):
                    return lowering.mechanism
            return Mechanism.NONE
        return {meta.name: reach(meta) for meta in sites}

    def add_bundle(self, bundle: Bundle) -> None:
        """THE bus loop: every kind in the bundle, made resident through its
        own lowering.

        A bank of two kinds attaches two pieces of state and its requests
        carry both kinds' levers. Registration is additive and idempotent
        (I8), and a bundle whose kinds cannot compose into one request is
        refused here, before any request can pin it.
        """
        if bundle.bundle_id in self._known:
            return
        payloads = group_by_kind(bundle)
        lowerings = [self._lowering_for(bundle.bundle_id, kind)
                     for kind in payloads]
        check_levers_compose(bundle.bundle_id, lowerings)
        self._attached[bundle.bundle_id] = {
            lowering.kind: lowering.attach(bundle.bundle_id,
                                           payloads[lowering.kind])
            for lowering in lowerings}
        self._known.add(bundle.bundle_id)   # no payloads: serve the bare base

    def _lowering_for(self, bundle_id: str, kind: str) -> RolloutLowering:
        """The lowering that serves `kind` here, or the honest refusal."""
        if kind not in self._lowerings:
            raise NotImplementedError(
                f"bundle {bundle_id!r} carries {kind!r} state, but this build "
                f"serves {list(self.serves)} — reachability said so at Phase 0; "
                f"serving it now would be a lie")
        return self._lowerings[kind]

    def attachments(self, bundle_id: str) -> Mapping[str, object]:
        """What each kind made resident for this bundle (kind -> its state):
        the bus's own inventory, read by probes. Requests go through levers."""
        return self._attached.get(bundle_id, {})

    def residency(self) -> Mapping[str, int]:
        """How many registered bundles each served kind holds state for — the
        census of I8 on this engine, one number per kind."""
        counts = {kind: 0 for kind in self.serves}
        for attached in self._attached.values():
            for kind in attached:
                counts[kind] += 1
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
        # the prefill is known BEFORE the request leaves, and the clock for
        # time-to-first-token starts on the same line
        self.meter.opened_request(len(prompt_ids))
        submitted = time.time()

        emitted = 0
        text_len = 0
        final = None
        async for output in self._ensure_llm().generate(
                levers.prompt, params, request_id, **levers.kwargs):
            completion = output.outputs[0]
            new_ids = list(completion.token_ids)[emitted:]
            if new_ids and not emitted:
                self.meter.first_token_after(time.time() - submitted)
            self.meter.decoded(len(new_ids))
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
        # one prefill of known length, no decode loop: it costs the metal a
        # prefill and the window says so
        self.meter.opened_request(len(full_ids))

        final = None
        async for output in self._ensure_llm().generate(
                levers.prompt, params, request_id, **levers.kwargs):
            final = output
        assert final is not None and final.prompt_logprobs is not None
        # the scored suffix sits after the context — and after everything the
        # bundle's kinds put in front of it
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
        """Every attached kind's apply(), merged into ONE request — where a
        bundle becomes USED (I8).

        The kinds are asked in BANK ORDER, and one that shapes the prompt form
        replaces the plain token prompt the bus would otherwise send — which is
        unambiguous because add_bundle refused any bundle whose kinds claim the
        same lever. Nothing attached: the raw token ids, as if no bank existed.
        """
        from vllm import TokensPrompt

        request = Request(token_ids=tuple(prompt_ids))
        levers = Levers(prompt=TokensPrompt(prompt_token_ids=prompt_ids))
        for kind, attached in self._attached[bundle_id].items():
            levers = levers.merged_with(
                self._lowerings[kind].apply(attached, request))
        return levers

    def _occupied(self, bundle_id: str) -> int:
        """Prompt positions this bundle's state occupies before the request's
        own tokens — SUMMED across its kinds, because they all sit in front of
        the same first real token."""
        return sum(self._lowerings[kind].align(attached).positions
                   for kind, attached in self._attached[bundle_id].items())


def _finish_event(completion) -> FinishEvent:
    if completion is None:
        return FinishEvent("length")
    if completion.finish_reason == "length":
        return FinishEvent("length")
    if isinstance(completion.stop_reason, str):
        return FinishEvent("stop", stop_hit=completion.stop_reason)
    return FinishEvent("eos")
