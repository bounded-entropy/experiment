"""VllmEngine: the Engine protocol over vLLM — a BUS, not a consumer.

Every adapter type's serving knowledge lives in its own rollout lowering
(policy/adapters/<adapter_type>_vllm.py), so this file knows no mechanism at
all. What it knows is the shape of the work:

  BUILD        every served adapter type's demands(), folded into the engine
               args; reachability is the union of what those payments bought.
  add_bundle   each of the bundle's adapter types attach()es its own payloads,
               additively and idempotently — a bundle this engine cannot
               express is refused HERE, while it is still just an id.
  add_library  one NAMED payload into the build's Library (ADR 0019), bytes a
               role read from the store; what a route under that name is
               served as is the adapter type's business, at apply().
  a request    every attached adapter type's apply(), merged into ONE unit of
               work. Requests pinning different bundles batch together in vLLM's
               own scheduler: the multi-tenancy invariant, held by construction.
  an answer    every attached adapter type's align(), SUMMED, because a scored
               suffix sits after everything the bundle put in front of it.

CUDA GRAPHS BY DEFAULT. `enforce_eager=None` (the default) is AUTO: eager only
where a served adapter type DEMANDS it (steer's residual add cannot ride a
captured graph), CUDA graphs otherwise — decode on a hybrid (Gated DeltaNet)
base in eager mode is launch-bound, and graphs are compatible with sleep mode
and LoRA. If the graphed build fails, the AUTO engine is rebuilt eager, once,
and says so; an explicit True or False is a build fact and is never changed.

Two build facts off to the side. enable_sleep_mode buys the sleep()/wake() pair
an alternating host's arbiter hooks call to make a partition really hand the
device back — deliberately NOT on the Engine protocol, since a fake or a remote
pool has no device to give. And no engine plugin is installed here, so an
adapter type whose demands() name one cannot be served on this build.

v0 choice, stated: prompts are RAW token concatenations of the messages, each
tokenized exactly as flatten will re-tokenize it — no chat template.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import time
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path

from rlstack.data.trajectory import Message
from rlstack.policy.adapters.base import Directive, Mechanism, Reachability
from rlstack.policy.adapters.rollout import (
    Levers, Library, Request, RolloutLowering, ServingBuild, check_demand_fits,
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
                 max_library: int = 32, cas_get=None, enforce_eager: bool | None = None,
                 tp: int = 1, serves: Sequence[str] = ("lora",),
                 enable_sleep_mode: bool = False, log_stats: bool = False,
                 weight_transfer_host: str | None = None,
                 code: Mapping[str, str] | None = None,
                 enable_min_tokens_after: bool = False) -> None:
        from rlstack.code import verify_code, require_declared_definition
        code = dict(code or {})
        verify_code(code)
        for kind in serves:
            for module in ADAPTER_TYPES.get(kind).instance.code_modules():
                require_declared_definition(module, tuple(code))
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
        # THE LIBRARY (ADR 0019): named payloads `add_library` holds, at most
        # `max_library` of them, least-recently-used out first. The lowerings
        # read it through the build at request time, and are told what it
        # evicted so they release what they materialized from it.
        self._library = Library(max_library, self._forget_library)
        self._build = ServingBuild(base=base, config=self._config,
                                   workdir=self._workdir,
                                   max_bundles=max_bundles, max_rank=max_rank,
                                   max_members=max_members, cas=cas_get,
                                   library=self._library,
                                   code=code,
                                   site_types=tuple(name for name in serves
                                       if ADAPTER_TYPES.get(name).instance.serving == Mechanism.SITE))
        self._lowerings = self._serving_adapter_types()
        self._engine_args = dict(
            model=base, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tp, enable_sleep_mode=enable_sleep_mode,
            disable_log_stats=not log_stats)
        self.enable_min_tokens_after = enable_min_tokens_after
        if enable_min_tokens_after:
            self._engine_args["logits_processors"] = [
                "rlstack_engine.min_tokens_after:MinTokensAfterProcessor"]
            self._engine_args["logprobs_mode"] = "raw_logprobs"
        self.log_stats = log_stats
        self._stats_task: asyncio.Task | None = None
        if enforce_eager is not None:
            self._engine_args["enforce_eager"] = enforce_eager
        if getattr(self._config, "vision_config", None) is not None:
            # a vision-language checkpoint (Qwen3.5) served for text alone:
            # no image or video per prompt, so vLLM neither profiles nor
            # reserves its encoder cache (post 0010)
            self._engine_args["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        # the environment the engine's process must carry when it is built —
        # paid demands, applied in _ensure_llm before the core is spawned
        self._engine_env: dict[str, str] = {}
        for lowering in self._lowerings.values():
            self._engine_args.update(self._pay_demands(lowering))
        # AUTO: no build fact and no demand chose eager, so graphs
        self.eager_auto = "enforce_eager" not in self._engine_args
        self._engine_args.setdefault("enforce_eager", False)
        self.eager_fallback: str | None = None   # why an AUTO build fell back
        self._llm = None                       # built inside the running loop
        self._asleep = False                   # the sleep seam's own state
        # traffic is counted HERE, at our own seam — never inside vLLM, whose
        # statistics are version-coupled. A host wires its own meter in.
        self.meter = TrafficMeter()
        # what this engine is holding, and the rule for letting go: bounded,
        # least-recently-used first, never a bundle a request is pinning.
        self._residency = BundleResidency(max_bundles, self._detach_all)
        self._verified_code: set[tuple] = set()
        self._request_count = 0
        self._weight_receiver = None
        self._publication_lock = threading.RLock()
        self.weight_transfer_host = weight_transfer_host

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
            try:
                self._llm = AsyncLLMEngine.from_engine_args(
                    AsyncEngineArgs(**self._engine_args))
            except Exception as failure:
                if not (self.eager_auto and not self._engine_args["enforce_eager"]):
                    raise
                # the AUTO build's safe fallback: the same engine, eager
                self.eager_fallback = f"{type(failure).__name__}: {failure}"
                print(f"[vllm_engine] CUDA-graph build of {self.base!r} failed "
                      f"({self.eager_fallback}); rebuilding eager", flush=True)
                self._engine_args["enforce_eager"] = True
                self._llm = AsyncLLMEngine.from_engine_args(
                    AsyncEngineArgs(**self._engine_args))
            if self.log_stats:
                self._stats_task = asyncio.get_running_loop().create_task(
                    self._log_stats_forever())
        return self._llm

    async def _log_stats_forever(self, every_s: float = 10.0) -> None:
        """vLLM's own stats line (running / waiting requests, KV-cache use,
        throughput), asked every `every_s` — an AsyncLLM logs only when its
        owner asks, as vLLM's server does. Diagnosis only."""
        while self._llm is not None:
            await asyncio.sleep(every_s)
            if self._llm is not None and not self._asleep:
                await self._llm.do_log_stats()

    # ---- Engine protocol ----------------------------------------------------

    def tokenize(self, text: str) -> tuple[int, ...]:
        return tuple(self._tokenizer.encode(text, add_special_tokens=False))

    def reachability(self, sites) -> Reachability:
        """This build's inventory: the union of what its served adapter types
        reach, asked in build order.

        An adapter type this build did not pay for is not in the union at all,
        so its sites come back NONE and a tenant needing one is refused at Phase
        0 rather than served wrong.
        """
        def reach(meta):
            mechanisms = frozenset(lowering.mechanism for lowering in self._lowerings.values()
                                   if lowering.reaches(meta))
            return (Mechanism.NONE if not mechanisms else next(iter(mechanisms))
                    if len(mechanisms) == 1 else mechanisms)
        return {meta.name: reach(meta) for meta in sites}

    def add_bundle(self, bundle: Bundle) -> None:
        with self._publication_lock:
            self._add_bundle(bundle)

    def _add_bundle(self, bundle: Bundle) -> None:
        """THE bus loop: every adapter type in the bundle, made resident through
        its own lowering.

        A bank of two adapter types attaches two pieces of state and its requests
        carry both adapter types' levers. Registration is additive and idempotent
        (I8), and a bundle whose adapter types cannot compose into one request is
        refused here, before any request can pin it.
        """
        self.verify_bundle_code(bundle.code, bundle.adapter_types)
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

    def verify_bundle_code(self, code, adapter_types) -> None:
        from rlstack.code import verify_code, require_declared_definition
        # Source files are immutable for this resident's lifetime. Verify once
        # per manifest; repeated publication must not rescan a recipe tree.
        key = tuple(sorted(code.items()))
        if key not in self._verified_code:
            verify_code(code)
            self._verified_code.add(key)
        for kind in adapter_types.values():
            for module in ADAPTER_TYPES.get(kind).instance.code_modules():
                require_declared_definition(module, tuple(code))

    def add_library(self, name: str, payload: bytes) -> None:
        """Hold one named payload for routes to read (ADR 0019): idempotent,
        bounded by `max_library`, and nothing but a dict write — so it is safe
        under traffic. A name a request is reading is pinned against eviction
        (`Library.pinned`); what a lowering built from an evicted name goes
        with it (`_forget_library`)."""
        self._library.add(name, payload)

    def install_tensor_state(self, manifest, states) -> None:
        with self._publication_lock:
            self._install_tensor_state(manifest, states)

    def _install_tensor_state(self, manifest, states) -> None:
        """Adapter-neutral bus; lowerings own installation, not carriage."""
        self.verify_bundle_code(manifest.get("code", {}), manifest["adapter_types"])
        bundle_id = manifest["bundle_id"]
        if self._residency.knows(bundle_id):
            return
        grouped: dict[str, dict] = {}
        for name, state in states.items():
            grouped.setdefault(manifest["adapter_types"][name], {})[name] = state
        lowerings = [self._lowering_for(bundle_id, kind) for kind in grouped]
        check_levers_compose(bundle_id, lowerings)
        attached = {}
        try:
            for lowering in lowerings:
                attached[lowering.adapter_type] = lowering.attach_state(
                    bundle_id, grouped[lowering.adapter_type])
        except BaseException:
            self._detach_all(attached)
            raise
        self._residency.hold(bundle_id, attached)

    def prepare_transfer(self, manifest, transfer) -> dict:
        self.verify_bundle_code(manifest.get("code", {}), manifest["adapter_types"])
        from rlstack.runner.transports.tensor import TensorReceiver
        from dataclasses import replace
        if self._asleep:
            raise ValueError("NCCL publication requires an awake dedicated inference resident")
        if set(manifest["states"]) != set(manifest["adapter_types"]):
            raise ValueError("tensor publication must label every adapter state")
        check_levers_compose(manifest["bundle_id"], [
            self._lowering_for(manifest["bundle_id"], kind)
            for kind in sorted(set(manifest["adapter_types"].values()))])
        if self.weight_transfer_host is not None:
            transfer = replace(transfer, rendezvous_host=self.weight_transfer_host)
        with self._publication_lock:
            if self._weight_receiver is None:
                self._weight_receiver = TensorReceiver(self.install_tensor_state)
            return self._weight_receiver.prepare(manifest, transfer)

    def finish_transfer(self, source: str, token: str) -> dict:
        if self._weight_receiver is None:
            raise ValueError("no tensor transfer was prepared")
        return self._weight_receiver.finish(source, token)

    def abort_transfer(self, source: str, token: str) -> None:
        if self._weight_receiver is not None:
            self._weight_receiver.abort(source, token)

    def holds_library(self, name: str, digest: str) -> bool:
        return self._library.holds(name, digest)

    def _forget_library(self, name: str) -> None:
        """The Library's eviction callback: every lowering releases what it
        materialized from `name`."""
        for lowering in self._lowerings.values():
            lowering.forget_library(name)

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

    def first_contact(self) -> dict:
        """WHAT THIS BUILD ACTUALLY TOOK, once it has served something (ADR
        0008, F5) — empty before the first request, because before it there
        is no engine yet (the core is built inside the running loop, at the
        first sample) and nothing to measure.

        Three numbers, all read off vLLM's own config rather than guessed:
        the GB the weights occupy, how many tokens the KV cache actually got
        (blocks x block size — the number a context length is spent out of),
        and the fraction of the card this build was allowed. The host
        journals them beside the GB the partition was DECLARED at, which is
        the comparison nobody was making when 48 GB per card met a 46 GiB
        resident.

        Every read is defensive about vLLM's own shape ON PURPOSE, against
        this file's usual rule: these attributes are version-coupled
        internals and a measurement that crashes the engine it is measuring
        would be worse than one that says it could not look."""
        if self._llm is None or not self._request_count:
            return {}
        report: dict = {"requests": self._request_count,
                        "gpu_memory_utilization":
                            self._engine_args["gpu_memory_utilization"],
                        "max_model_len": self._engine_args["max_model_len"],
                        "enforce_eager": self._engine_args["enforce_eager"],
                        "eager_fallback": self.eager_fallback}
        try:
            config = self._llm.engine.vllm_config
            report["weights_gb"] = round(
                float(config.model_config.get_weights_bytes()) / 2 ** 30, 3)
            report["kv_tokens"] = (int(config.cache_config.num_gpu_blocks or 0)
                                   * int(config.cache_config.block_size or 0))
        except Exception as unreadable:          # noqa: BLE001 - see the docstring
            report["unreadable"] = str(unreadable)
        return report

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
        if sampling.min_tokens_after is not None:
            if not self.enable_min_tokens_after:
                raise ValueError("min_tokens_after requires an opted-in EngineBuild")
            if stop:
                raise ValueError("min_tokens_after is incompatible with string stops")
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        # Each message tokenized separately, then concatenated — the EXACT
        # ids flatten reproduces for injected spans (no template drift).
        prompt_ids = [tid for m in messages for tid in self.tokenize(m.content)]
        self._request_count += 1
        request_id = f"rlstack-{self._request_count}-{seed}"
        with self._held_for(prompt_ids, bundle_id, seed, directives) as levers:
            await self._prepare_bundle(bundle_id)
            # DELTA outputs: each step hands over only its new tokens, log-
            # probs and text, so a long completion costs O(n) here, not the
            # O(n^2) of re-reading the cumulative output every step
            controls = {"top_k": sampling.top_k, "min_p": sampling.min_p,
                        "presence_penalty": sampling.presence_penalty,
                        "repetition_penalty": sampling.repetition_penalty}
            overrides = {key: value for key, value in controls.items() if value is not None}
            if self.enable_min_tokens_after:
                from rlstack.sampling import CONTINUATION_KEY
                if CONTINUATION_KEY in levers.extra_args:
                    raise ValueError("min_tokens_after reserved extra_args key collision")
            params = SamplingParams(
                **overrides, temperature=sampling.temperature, top_p=sampling.top_p,
                max_tokens=sampling.max_tokens, stop=list(stop), seed=seed,
                logprobs=0, extra_args=dict(levers.extra_args) or None,
                output_kind=RequestOutputKind.DELTA)
            continuation = None
            if sampling.min_tokens_after is not None:
                from rlstack.sampling import CONTINUATION_KEY, ContinuationCounter
                llm = self._ensure_llm()
                # Resolve exactly as vLLM 0.28's InputProcessor does before
                # cloning the request. The GPU processor verifies the resolved
                # set again when the request enters the persistent batch.
                ip = llm.input_processor
                params.update_from_generation_config(
                    ip.generation_config_fields, ip.renderer.get_eos_token_id())
                terminal_ids = sorted(params.all_stop_token_ids)
                if not terminal_ids:
                    raise ValueError("min_tokens_after requires resolved terminal token IDs")
                floor = sampling.min_tokens_after
                available_output_budget = min(sampling.max_tokens,
                    self._engine_args["max_model_len"] - len(prompt_ids) -
                    self._occupied(bundle_id))
                if available_output_budget < len(floor.sequence) + floor.min_tokens:
                    raise ValueError("model window cannot accommodate min_tokens_after")
                if set(floor.sequence) & set(terminal_ids):
                    raise ValueError("activation sequence cannot contain terminal token IDs")
                config = {"sequence": list(floor.sequence), "min_tokens": floor.min_tokens,
                          "terminal_token_ids": terminal_ids}
                extra_args = dict(params.extra_args or {})
                if CONTINUATION_KEY in extra_args:
                    raise ValueError("min_tokens_after reserved extra_args key collision")
                extra_args[CONTINUATION_KEY] = config
                params.extra_args = extra_args
                continuation = ContinuationCounter(floor.sequence, tuple(terminal_ids))
            # the prefill is known BEFORE the request leaves, and the clock for
            # time-to-first-token starts on the same line
            self.meter.opened_request(len(prompt_ids))
            submitted = time.time()

            emitted = 0
            final = None
            cached = False
            async for output in self._ensure_llm().generate(
                    _salted(levers), params, request_id, **levers.kwargs):
                if not cached:
                    # what the prefix cache already held of this prompt,
                    # reported with the first output (ADR 0022)
                    self.meter.cached(int(getattr(output, "num_cached_tokens", 0) or 0))
                    cached = True
                completion = output.outputs[0]
                new_ids = completion.token_ids
                if continuation is not None:
                    continuation.extend(new_ids)
                if new_ids and not emitted:
                    self.meter.first_token_after(time.time() - submitted)
                self.meter.decoded(len(new_ids))
                for offset, token_id in enumerate(new_ids):
                    logprob = completion.logprobs[offset][token_id].logprob
                    # attribute the whole text delta to the last new token: the
                    # concatenation (the Message content) stays exact either way
                    last = offset == len(new_ids) - 1
                    yield TokenEvent(token_id=int(token_id), logprob=float(logprob),
                                     text_delta=completion.text if last else "")
                emitted += len(new_ids)
                final = completion
            # the merged levers' recorded facts leave with the stream's last
            # event: what the bundle's adapter types DREW for this request is
            # sampling-time truth, sealed into Turn.turn_extras (I6)
            recorded = dict(levers.turn_extras)
            if self.enable_min_tokens_after:
                recorded["sampling_logprobs"] = {
                    "mode": "raw_logprobs", "behavior_logprobs": False,
                    "note": "raw model likelihood before sampling constraints and transforms"}
            if continuation is not None:
                floor = sampling.min_tokens_after
                recorded["min_tokens_after"] = {
                    **config, **continuation.evidence(floor.min_tokens),
                    "evidence_source": "returned_output_token_ids",
                    "max_tokens": sampling.max_tokens,
                    "available_output_budget": available_output_budget,
                    "processor": "rlstack_engine.min_tokens_after:MinTokensAfterProcessor",
                    "logprobs_mode": "raw_logprobs",
                    "behavior_logprobs": False,
                    "finish_reason": final.finish_reason if final is not None else None,
                    "stop_reason": final.stop_reason if final is not None else None}
            if overrides:
                # Read back the constructed engine parameters, not the requested spec.
                recorded["sampling_parameters"] = {
                    "temperature": params.temperature, "top_p": params.top_p,
                    "max_tokens": params.max_tokens, "top_k": params.top_k,
                    "min_p": params.min_p, "presence_penalty": params.presence_penalty,
                    "repetition_penalty": params.repetition_penalty,
                    "frequency_penalty": params.frequency_penalty,
                    "seed": params.seed, "stop": params.stop}
            yield _finish_event(final, recorded)

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
        # score traffic is SEEDLESS by contract — it draws nothing and must be
        # deterministic, so an adapter type that chooses per request is told
        # there is no seed rather than handed one
        with self._held_for(full_ids, bundle_id, None, directives) as levers:
            await self._prepare_bundle(bundle_id)
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
            self.meter.cached(int(getattr(final, "num_cached_tokens", 0) or 0))
            # the scored suffix sits after the context — and after everything the
            # bundle's adapter types put in front of it
            start = self._occupied(bundle_id) + len(context_ids)
            return tuple(
                float(final.prompt_logprobs[start + j][int(tok)].logprob)
                for j, tok in enumerate(token_ids))

    # ---- the sleep seam (a build capability, NOT the Engine protocol) -------

    async def _prepare_bundle(self, bundle_id: str) -> None:
        engine = self._ensure_llm()
        for name, attached in self._residency.attached(bundle_id).items():
            await self._lowerings[name].prepare(engine, attached)

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
        if self._weight_receiver is not None:
            self._weight_receiver.close()
            self._weight_receiver = None
        if self._llm is None:
            return
        if self._stats_task is not None:
            self._stats_task.cancel()
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

    @contextlib.contextmanager
    def _held_for(self, prompt_ids: list[int], bundle_id: str,
                  seed: int | None, directives: Sequence[Directive]):
        """ONE REQUEST'S HOLDS, around its levers: the bundle it pins, and the
        library names its levers read (ADR 0019) — both immune from eviction
        until the last token is out. Nothing awaits between a lowering reading
        a name and the pin taken here; a name evicted in that gap all the same
        is refused by the pin, loudly, and never served."""
        with self._residency.pinned(bundle_id):
            levers = self._levers_for(prompt_ids, bundle_id, seed, directives)
            with self._library.pinned(levers.library):
                yield levers

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
