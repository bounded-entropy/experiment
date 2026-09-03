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

import os
import tempfile
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

from rlstack.data.trajectory import Message
from rlstack.policy.adapters.base import Directive, Mechanism
from rlstack.policy.adapters.rollout import (
    Levers, Request, RolloutLowering, ServingBuild, check_demand_fits,
    check_levers_compose,
)
from rlstack.policy.compile import Bundle, group_by_adapter_type
from rlstack.registry import ADAPTER_TYPES
from rlstack.runner.interfaces import FinishEvent, TokenEvent
from rlstack.runner.meters import TrafficMeter
from rlstack.runner.residency import BundleResidency
from rlstack.spec.specs import SamplingSpec


class VllmEngine:
    def __init__(self, base: str, *, gpu_memory_utilization: float = 0.45,
                 max_model_len: int = 1024, max_bundles: int = 8,
                 max_rank: int = 32, max_members: int = 0,
                 cas_get=None, enforce_eager: bool = True,
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
        # max_bundles / max_rank / max_members are plain capacity, spelled the
        # way ServingBuild spells them: how many bundles' state may be resident,
        # the widest delta rank served, and the widest ENSEMBLE one bundle may
        # be served as. Only the adapter type that SPENDS them turns them into
        # an engine arg (lora_vllm names vLLM's own max_loras / max_lora_rank),
        # so no mechanism-flavored word survives in this file. `cas_get` is the
        # same kind of build fact: how a lowering resolves an address its
        # payload carried, handed in by the deploy that owns the store.
        self._build = ServingBuild(base=base, config=self._config,
                                   workdir=self._workdir,
                                   max_bundles=max_bundles, max_rank=max_rank,
                                   max_members=max_members, cas=cas_get)
        self._lowerings = self._serving_adapter_types()
        self._engine_args = dict(
            model=base, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tp, enable_sleep_mode=enable_sleep_mode,
            enforce_eager=enforce_eager, disable_log_stats=True)
        # the environment the engine's process must carry when it is built —
        # paid demands, applied in _ensure_llm before the core is spawned
        self._engine_env: dict[str, str] = {}
        for lowering in self._lowerings.values():
            self._engine_args.update(self._pay_demands(lowering))
        self._llm = None                       # built inside the running loop
        self._asleep = False                   # the sleep seam's own state
        # traffic is counted HERE, at our own seam — never inside vLLM, whose
        # statistics are version-coupled. A host wires its own meter in.
        self.meter = TrafficMeter()
        # what this engine is holding, and the rule for letting go: bounded,
        # least-recently-used first, never a bundle a request is pinning.
        self._residency = BundleResidency(max_bundles, self._detach_all)
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
        check_demand_fits(lowering.adapter_type, demands, self._engine_args,
                          {**os.environ, **self._engine_env})
        self._engine_env.update(demands.env)
        return demands.engine_args

    def _ensure_llm(self):
        """AsyncLLMEngine wants a running event loop; the runner has one by
        the first sample — so construction is deferred to that moment."""
        if self._llm is None:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            # the paid environment demands, in THIS process: the engine core
            # and its workers are spawned from here and inherit them
            os.environ.update(self._engine_env)
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
        if self._residency.knows(bundle.bundle_id):
            return
        payloads = group_by_adapter_type(bundle)
        lowerings = [self._lowering_for(bundle.bundle_id, adapter_type)
                     for adapter_type in payloads]
        check_levers_compose(bundle.bundle_id, lowerings)
        self._residency.hold(bundle.bundle_id, {
            lowering.adapter_type: lowering.attach(
                bundle.bundle_id, payloads[lowering.adapter_type])
            for lowering in lowerings})       # no payloads: serve the bare base

    def knows_bundle(self, bundle_id: str) -> bool:
        """Is this bundle resident HERE, right now?

        The one question that makes restore demand-driven: a caller asks before
        reading the store, so the common path (still resident) costs nothing and
        the miss path costs one rebuild. It is a fact about this engine at this
        instant, never about the run — the ledger says what a version IS, this
        says whether we happen to be holding it.
        """
        return self._residency.knows(bundle_id)

    def _detach_all(self, attached: Mapping[str, object]) -> None:
        """Give one bundle's state back, adapter type by adapter type —
        the callback residency calls when it evicts."""
        for adapter_type, state in attached.items():
            self._lowerings[adapter_type].detach(state)

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
        return self._residency.attached(bundle_id)

    def residency(self) -> Mapping[str, int]:
        """How many registered bundles each served adapter type holds state for —
        the census of I8 on this engine, one number per adapter type."""
        counts = {adapter_type: 0 for adapter_type in self.serves}
        return {**counts, **self._residency.census()}

    async def sample_tokens(
        self,
        messages: Sequence[Message],
        sampling: SamplingSpec,
        stop: tuple[str, ...],
        bundle_id: str,
        seed: int,
        directives: Sequence[Directive] = (),
    ) -> AsyncIterator[TokenEvent | FinishEvent]:
        if not self._residency.knows(bundle_id):
            raise RuntimeError(
                f"bundle {bundle_id!r} is not resident on this engine; a"
                f" caller restores a committed version before pinning it"
                f" (runner/restore.py)")
        from vllm import SamplingParams

        # Each message tokenized separately, then concatenated — the EXACT
        # ids flatten reproduces for injected spans (no template drift).
        prompt_ids = [tid for m in messages for tid in self.tokenize(m.content)]
        self._request_count += 1
        request_id = f"rlstack-{self._request_count}-{seed}"
        with self._residency.pinned(bundle_id):
            levers = self._levers_for(prompt_ids, bundle_id, seed, directives)
            params = SamplingParams(
                temperature=sampling.temperature, top_p=sampling.top_p,
                max_tokens=sampling.max_tokens, stop=list(stop), seed=seed,
                logprobs=0, extra_args=dict(levers.extra_args) or None)
            # the prefill is known BEFORE the request leaves, and the clock for
            # time-to-first-token starts on the same line
            self.meter.opened_request(len(prompt_ids))
            submitted = time.time()

            emitted = 0
            text_len = 0
            final = None
            async for output in self._ensure_llm().generate(
                    _salted(levers), params, request_id, **levers.kwargs):
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
            # the merged levers' recorded facts leave with the stream's last
            # event: what the bundle's adapter types DREW for this request is
            # sampling-time truth, sealed into Turn.turn_extras (I6)
            yield _finish_event(final, levers.turn_extras)

    async def score_tokens(self, messages: Sequence[Message],
                           token_ids: Sequence[int],
                           bundle_id: str,
                           directives: Sequence[Directive] = (),
                           ) -> tuple[float, ...]:
        """ONE prefill over context + tokens with prompt_logprobs: vLLM
        returns each prompt position's logprob under the pinned bundle, and we
        read off the scored suffix. max_tokens=1 because vLLM must generate
        something — the one decoded token is discarded."""
        if not self._residency.knows(bundle_id):
            raise RuntimeError(
                f"bundle {bundle_id!r} is not resident on this engine; a"
                f" caller restores a committed version before pinning it"
                f" (runner/restore.py)")
        if not token_ids:
            return ()
        from vllm import SamplingParams

        context_ids = [tid for m in messages for tid in self.tokenize(m.content)]
        full_ids = context_ids + [int(t) for t in token_ids]
        self._request_count += 1
        request_id = f"rlstack-score-{self._request_count}"
        with self._residency.pinned(bundle_id):
            # score traffic is SEEDLESS by contract — it draws nothing and must
            # be deterministic, so an adapter type that chooses per request is
            # told there is no seed rather than handed one
            levers = self._levers_for(full_ids, bundle_id, None, directives)
            params = SamplingParams(max_tokens=1, temperature=0.0,
                                    prompt_logprobs=0,
                                    extra_args=dict(levers.extra_args) or None)
            # one prefill of known length, no decode loop: it costs the metal a
            # prefill and the window says so
            self.meter.opened_request(len(full_ids))

            final = None
            async for output in self._ensure_llm().generate(
                    _salted(levers), params, request_id, **levers.kwargs):
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

    def shutdown(self) -> None:
        """End the engine's background machinery, so the container can exit.

        vLLM v1 runs its EngineCore in a child process with a shared-memory
        queue. Nothing stops it when the owning process merely returns: the
        core keeps its loop, the async output handler wakes on a closed loop
        and raises, and the container is still tearing down when a venue's
        grace period expires — a finished run then exits looking failed
        (#61 measured that against Modal's 30s). This is the verb a container
        calls on its way out.

        Idempotent, and quiet on an engine that was never built, for the same
        reason sleep() is: the vLLM build is lazy, so a host may be told to
        shut down before it ever served a request.
        """
        if self._llm is None:
            return
        self._llm.shutdown()
        self._llm = None

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

    def _levers_for(self, prompt_ids: list[int], bundle_id: str,
                    seed: int | None,
                    directives: Sequence[Directive] = ()) -> Levers:
        """Every attached adapter type's apply(), merged into ONE request —
        where a bundle becomes USED (I8).

        The adapter types are asked in BANK ORDER, and one that shapes the prompt
        form replaces the plain token prompt the bus would otherwise send — which
        is unambiguous because add_bundle refused any bundle whose adapter types
        claim the same lever. Nothing attached: the raw token ids, as if no bank
        existed.

        The request's seed travels with it because an adapter type may have a
        per-request CHOICE to make; None says this is score traffic, which is
        seedless and deterministic by contract. So do the caller's directives,
        and the positions the bundle's adapter types occupy in front of the
        real tokens — so an adapter type placing itself by position speaks
        real-token coordinates and lands where the engine's slice has it.
        """
        from vllm import TokensPrompt

        request = Request(token_ids=tuple(prompt_ids), seed=seed,
                          occupied=self._occupied(bundle_id),
                          directives=tuple(directives))
        levers = Levers(prompt=TokensPrompt(prompt_token_ids=prompt_ids))
        for adapter_type, attached in self._residency.attached(bundle_id).items():
            levers = levers.merged_with(
                self._lowerings[adapter_type].apply(attached, request))
        return levers

    def _occupied(self, bundle_id: str) -> int:
        """Prompt positions this bundle's state occupies before the request's
        own tokens — SUMMED across its adapter types, because they all sit in
        front of the same first real token."""
        return sum(self._lowerings[adapter_type].align(attached).positions
                   for adapter_type, attached
                   in self._residency.attached(bundle_id).items())


def _salted(levers: Levers):
    """The request's prompt form, carrying its prefix-cache identity when an
    adapter type contributed one (Levers.cache_salt): vLLM folds `cache_salt`
    into the first block's hash and every later block chains on it, so reuse
    lives within one salt and dies across two. Every prompt form vLLM accepts
    is a dict, so the salt rides whichever form the bank produced."""
    if levers.cache_salt is None:
        return levers.prompt
    return {**levers.prompt, "cache_salt": levers.cache_salt}


def _finish_event(completion, turn_extras: Mapping[str, object]) -> FinishEvent:
    """Why generation stopped, plus whatever this request's adapter types
    recorded about it — one terminal event carries both."""
    extras = dict(turn_extras)
    if completion is None:
        return FinishEvent("length", turn_extras=extras)
    if completion.finish_reason == "length":
        return FinishEvent("length", turn_extras=extras)
    if isinstance(completion.stop_reason, str):
        return FinishEvent("stop", stop_hit=completion.stop_reason,
                           turn_extras=extras)
    return FinishEvent("eos", turn_extras=extras)
