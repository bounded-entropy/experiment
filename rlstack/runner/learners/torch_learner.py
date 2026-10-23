"""TorchLearner: the Learner protocol on real metal, multi-tenant.

ONE frozen base shared by every tenant — the memory asymmetry that makes
tenancy affordable — with per-tenant params built and installed by each adapter
type's own compute half, and one optimizer per (tenant, entry) so optim blobs
map 1:1 onto the store's optim/<name>@v.

Tenancy is ADDITIVE INSTALL + ROW ROUTING, the trainer-side twin of the
engine's punica path (I8): a tenant's deltas stay wired for as long as it is
installed, and each row of a padded microbatch carries the slot whose delta
applies to it. Every verb pins one tenant, so today all rows of a forward carry
that tenant's slot — the degenerate one-slot case; rows carrying different
slots in ONE forward is the same mechanism with a mixed index.

v0 choices, stated: one padded forward per microbatch (documents left-aligned
and right-padded, causal attention), and determinism is best-effort — CUDA
kernels are not bit-stable, so byte-identical resume stays a fakes-suite
property and the real-metal invariant is the ledger's logprob_gap staying at
its floor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
import hashlib
import json
import math
import os
from typing import Any, Callable
import time
import threading

import torch
import torch.utils.checkpoint

from rlstack.client import GradientComparison, GradientDocumentStats
from rlstack.data.flatten import TokenBatch
from rlstack.policy.adapters.replay import ReplayRows, SiteWrapper, row_plan
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES, LOSSES
from rlstack.runner.learners.gradient_cache import GradientCache
from rlstack.runner.interfaces import (
    Emitted, Parameterization, TrainStats, check_lr_scales, lr_scale_of,
)
from rlstack.training.losses import PolicyOutputs
from rlstack.training.losses.factual_sft import sequence_sft


LOGPROB_CHUNK = 1024
"""Positions per fp32 log-softmax chunk in `chosen_logprobs`: 1024 × a 152k
vocab × 4 bytes is ~0.6 GiB, held twice at the peak of one chunk. The whole-
copy path this replaced held it for EVERY position at once."""

GRADIENT_MIN_NORM = 1e-12
"""Numerical guard in selected-token-mean loss coordinates, not a reward term."""


def serialized_learner_operation(operation: Callable[..., Any]) -> Callable[..., Any]:
    """One local resident operation owns its shared model and wrapper roster.

    Admission governs residency; it permits overlapping calls. The model's
    RowPlan and adapter installation require this separate, reentrant lock.
    """
    @wraps(operation)
    def serialized(self, *args, **kwargs):
        with self._operation_lock:
            return operation(self, *args, **kwargs)
    return serialized


def unscored_prefix(batch: TokenBatch, spans: list[tuple[int, int]]) -> int:
    """How many leading positions EVERY row of a padded forward leaves
    unscored: the shortest run of loss_mask=0 tokens that opens a document.

    No logit is needed there — nothing credits a prompt token — and a long
    prompt makes them the whole cost: a dreamer reads ten articles (~65k
    tokens) before it writes, and that prompt's logits over Qwen3.5's 248k
    vocabulary are ~32 GB in bf16 (post 0010). Position t scores token t from
    the hidden state at t-1, so the first scored token at index f needs
    logits from f-1 on; those positions' logprobs are reported as 0.0, the
    value a document's first position already has. Every scored number is
    the same arithmetic as before."""
    first = []
    for start, stop in spans:
        mask = batch.loss_mask[start:stop]
        scored = next((t for t, bit in enumerate(mask) if bit), len(mask))
        first.append(max(scored - 1, 0))
    return min(first, default=0)


def chosen_logprobs(logits: torch.Tensor, targets: torch.Tensor,
                    chunk: int = LOGPROB_CHUNK) -> torch.Tensor:
    """log p(target_t | prefix) at every position, in fp32, from the model's
    own-dtype logits [R, W, V] and targets [R, W] — CHUNK positions at a time,
    under recompute.

    The obvious `log_softmax(logits.float())` makes a four-byte copy of every
    logit and a second one for the softmax, and autograd keeps the second for
    backward: for a 4096-token microbatch over a 152k vocab that is one 8 GiB
    contiguous block on top of whatever the shards and the recompute state
    hold — and it was refused twice on the venue, at 48 and at 64 GiB per
    card, with the card itself not full (ADR 0005's arms). Chunked and
    checkpointed, the fp32 copy exists one chunk at a time in forward and
    again in backward, and what autograd retains is the model's own logits,
    which it retained anyway. Same numbers: fp32 softmax over the same
    values, chunk boundaries cut across positions and never across a vocab.
    """
    rows, width, _ = logits.shape
    if width == 0:          # every document is ONE token long: nothing has a prefix to be scored from
        return logits.new_zeros((rows, 0), dtype=torch.float32)
    out = []
    for row in range(rows):
        for start in range(0, width, chunk):
            piece = logits[row, start:start + chunk]        # [<=chunk, V], a view
            picked = targets[row, start:start + chunk, None]
            out.append(torch.utils.checkpoint.checkpoint(
                _chunk_logprob, piece, picked, use_reentrant=False))
    return torch.cat(out).reshape(rows, width)


def _chunk_logprob(piece: torch.Tensor, picked: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(piece.float(), dim=-1).gather(1, picked)[:, 0]


def head_logprobs(hidden: torch.Tensor, head, targets: torch.Tensor,
                  chunk: int = LOGPROB_CHUNK) -> torch.Tensor:
    """`chosen_logprobs` from the FINAL HIDDEN STATE [R, W, H] instead of the
    logits: the vocabulary projection itself runs CHUNK positions at a time
    under recompute, so no [R, W, V] logits tensor ever exists. A 10k-token
    dreamer episode over Qwen3.5's 248k vocabulary is a 5 GiB logits block
    and as much again for its gradient, and it was refused at 40 and at 46
    GiB (post 0011, grpo1/grpo2 update 1); here autograd keeps the hidden
    state, which it kept anyway. Same numbers: the head is the model's own
    `lm_head`, applied to the same rows."""
    rows, width, _ = hidden.shape
    if width == 0:
        return hidden.new_zeros((rows, 0), dtype=torch.float32)
    out = []
    for row in range(rows):
        for start in range(0, width, chunk):
            out.append(torch.utils.checkpoint.checkpoint(
                _head_chunk_logprob, head, hidden[row, start:start + chunk],
                targets[row, start:start + chunk, None], use_reentrant=False))
    return torch.cat(out).reshape(rows, width)


def _head_chunk_logprob(head, piece: torch.Tensor, picked: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(head(piece).float(), dim=-1).gather(1, picked)[:, 0]


def trunk_and_head(model):
    """(trunk, head) for a Hugging Face causal LM — `model.model` returns the
    final hidden state and `get_output_embeddings()` is the plain vocabulary
    projection — or None for anything else (the replay then reads logits)."""
    trunk = getattr(model, "model", None)
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    return (trunk, head) if trunk is not None and head is not None else None


@dataclass
class _Tenant:
    """One tenant's state on this learner: params, adapter types, optimizers,
    and the SLOT a forward's rows route to — its installed deltas keyed the way a
    site asks for them (site path -> the params holding it)."""

    loss_fn: object
    trainable: list[str]
    entries: list[str]                                  # install order
    params: dict[str, object] = field(default_factory=dict)
    adapter_types: dict[str, object] = field(default_factory=dict)
    sites: dict[str, tuple[SiteMeta, ...]] = field(default_factory=dict)
    optimizers: dict[str, torch.optim.Optimizer] = field(default_factory=dict)
    # entry -> its optimizer's groups IN ORDER, each (name, base lr): what
    # `optim_step(lr_scales=...)` scales from, fixed at optimizer build so a
    # scaled step can never compound into the next one
    base_lrs: dict[str, tuple[tuple[str, float], ...]] = field(default_factory=dict)
    slot: dict[str, Any] = field(default_factory=dict)
    routed_slots: dict[str, dict[str, Any]] = field(default_factory=dict)


class TorchLearner:
    def __init__(self, device: str | None = None,
                 dtype: torch.dtype = torch.bfloat16,
                 grad_clip: float = 1.0,
                 checkpoint_activations: bool = True) -> None:
        self._operation_lock = threading.RLock()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.grad_clip = grad_clip
        # build fact, mirrored from VllmEngine.sleeps: this learner can hand
        # its device back (sleep/wake below). A resident's hello reports it and
        # the host wires the alternation hooks only when it says so. The
        # SHARDED build probes its torch for the hook a shard's move needs and
        # may report false with a reason (fsdp_torch.probe_sharded_sleep).
        self.sleeps = True
        self.sleep_refusal = ""
        # how many forwards this learner has run: what makes `first_contact`
        # empty until there has BEEN a first forward (ADR 0008, F5)
        self.forwards = 0
        self._asleep = False
        self.checkpoint_activations = checkpoint_activations
        self.fsdp = 1               # build fact: this build is unsharded
        self.dp = 1                 # build fact: one group (ADR 0022)
        self._tokenizer = None
        self._base: str | None = None
        self._model: torch.nn.Module | None = None
        self._tenants: dict[str, _Tenant] = {}
        self._weight_sender = None
        self.gradient_cache = GradientCache(
            int(os.environ.get("RLSTACK_GRADIENT_CACHE_BYTES", "0")))

    # ---- Learner protocol ---------------------------------------------------

    @serialized_learner_operation
    def tokenize(self, tenant: str, text: str) -> tuple[int, ...]:
        """The same base tokenizer and encoding convention as VllmEngine.

        Loaded lazily on CPU, including on the lead FSDP rank; tokenization
        needs no model forward or collective operation.
        """
        self._tenants[tenant]  # only installed tenants have a bound base
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self._base)
        return tuple(self._tokenizer.encode(text, add_special_tokens=False))

    @serialized_learner_operation
    def install(self, tenant: str, parameterization: Parameterization) -> None:
        """The record is everything this learner may know of the experiment:
        the loss and every adapter type are resolved BY KEY against this
        process's own registries, the seeds arrive derived, and the sites
        arrive resolved (ADR 0002, Q2)."""
        self._ensure_base(parameterization.base)
        if tenant in self._tenants:            # Phase 1 re-runs on attach
            self._remove(tenant)

        state = _Tenant(
            loss_fn=LOSSES.get(parameterization.loss).fn,
            trainable=sorted(e.name for e in parameterization.entries
                             if e.trainable),
            entries=[e.name for e in parameterization.entries],
        )
        for entry in parameterization.entries:
            adapter_type = ADAPTER_TYPES.get(entry.adapter_type).instance
            # version 0 through the adapter type's OWN init function, so a
            # learner-built v0 and a learner-less run's (ADR 0006 Part B)
            # are the same bytes and the same bundle id
            params = adapter_type.initial_params(entry.sites, entry.init)
            state.params[entry.name] = params
            state.adapter_types[entry.name] = adapter_type
            state.sites[entry.name] = entry.sites
            adapter_type.install_replay(self._model, params, entry.sites)
            self._claim_slot(state, entry.sites, params, entry.route)
            if entry.name in state.trainable:
                state.optimizers[entry.name] = self._optimizer_for(
                    entry.name, params, adapter_type, parameterization.optim)
                state.base_lrs[entry.name] = self._base_lrs(
                    entry.name, params, adapter_type, parameterization.optim)
        self._tenants[tenant] = state

    def first_contact(self) -> dict:
        """WHAT THIS LEARNER ACTUALLY TOOK after its first forward (ADR 0008,
        F5) — empty before it, because nothing has been allocated to measure.

        `peak_gb` is `torch.cuda.max_memory_allocated` on this process's
        devices: the high-water mark of the allocator, which is the number an
        OOM is decided against and the one nobody had. On this fleet 48 GB
        per card "should fit" and met a 46.3 GiB resident plus one 8.2 GiB
        fp32 logits block; 64 met the same block with the allocator grown
        into it. Journaled by the host beside the GB the partition was
        DECLARED at, once, on every venue, so the next arm's sizing is read
        off a measurement instead of a guess."""
        if not self.forwards:
            return {}
        return {"peak_gb": self.peak_gb(), "forwards": self.forwards,
                "device": str(self.device)}

    def peak_gb(self) -> list[float]:
        """The allocator's high-water mark per device this process can see,
        in GiB — empty off CUDA, where there is no allocator to ask."""
        if not torch.cuda.is_available():
            return []
        return [round(torch.cuda.max_memory_allocated(index) / 2 ** 30, 3)
                for index in range(torch.cuda.device_count())]

    @serialized_learner_operation
    def uninstall(self, tenant: str) -> None:
        """Install's inverse as a VERB (ADR 0006 Part A): the tenancy ends, so
        the module tree goes back to the way install found it. Idempotent — a
        tenant nobody installed is already uninstalled, which is what lets a
        host uninstall in a `finally` without knowing how far its run got."""
        if self._weight_sender is not None:
            self._weight_sender.release_owner(tenant)
        if tenant in self._tenants:
            self._remove(tenant)

    def _optimizer_for(self, entry: str, params: object, adapter_type: object,
                       optim) -> torch.optim.Optimizer:
        """ONE AdamW per trainable entry, over the adapter type's NAMED param
        groups — so an entry stays one optim blob in the store however many
        groups it has.

        The groups come from the adapter type (AdapterType.param_groups); the
        settings come from OptimSpec, overridden by name. The grammar is dotted:
        `"pi"` reaches every group of entry `pi`, `"pi.mapper"` reaches one of
        them, and the dotted form wins where both apply — specific over general,
        the only reading under which writing both is not a contradiction.
        Groups are taken in sorted name order, which is what keeps a resumed
        optimizer's state_dict addressable by the same indices.
        """
        groups = adapter_type.param_groups(params)
        return torch.optim.AdamW(
            [{"params": list(groups[name]),
              **self._group_settings(optim, entry, name)}
             for name in sorted(groups)],
            lr=optim.lr, betas=optim.betas, weight_decay=optim.weight_decay)

    @staticmethod
    def _group_settings(optim, entry: str, group: str) -> dict:
        """This group's optimizer settings: the spec's defaults, then the
        entry-wide override, then the group's FAMILY's, then the group's own.
        The default group (name "") is the whole entry, so only the entry-wide
        form addresses it.

        A group named `family:member` (dream_bank's `memory:03`) belongs to
        the family before the colon, so `"pi.memory"` reaches every memory
        group the way it reached the one `memory` group before the groups
        became one per route (ADR 0019) — general before specific, again."""
        settings: dict = {"lr": optim.lr, "weight_decay": optim.weight_decay}
        settings.update(optim.overrides.get(entry, {}))
        if group:
            family = group.partition(":")[0]
            if family != group:
                settings.update(optim.overrides.get(f"{entry}.{family}", {}))
            settings.update(optim.overrides.get(f"{entry}.{group}", {}))
        return settings

    def _base_lrs(self, entry: str, params: object, adapter_type: object,
                  optim) -> tuple[tuple[str, float], ...]:
        """Each group's name and BASE learning rate, in the optimizer's own
        group order (`_optimizer_for` sorts by name; so does this)."""
        return tuple(
            (name, float(self._group_settings(optim, entry, name)["lr"]))
            for name in sorted(adapter_type.param_groups(params)))

    @serialized_learner_operation
    def forward_backward(self, tenant: str, batch: TokenBatch) -> TrainStats:
        state = self._tenant(tenant)
        self.forwards += 1
        spans = _doc_spans(batch)
        # the routing spans the BACKWARD too: with the blocks checkpointed the
        # forward is run again inside backward(), and a recomputed forward that
        # found no plan would be a forward with no deltas — the rows have to be
        # pinned for as long as the forward can RUN, not just until it returns
        with row_plan(self._model).route(self._rows_of(state, batch, len(spans))):
            logprobs = self._batched_logprobs(batch, spans)
            provided = self._provided(state)
            result = state.loss_fn(
                PolicyOutputs(logprobs=logprobs, provided=provided), batch)
            # a microbatch may hold NO target at all — one-token documents,
            # which a dream cut at blank lines produces (a `---` line is one
            # token) and a fork at batch 1 steps alone: its loss is a constant
            # 0 with no graph, and it accumulates nothing
            # Fully frozen replay measures loss only; its adapter tensors can
            # still require grad, but there is no trainable entry to accumulate.
            if state.trainable and result.loss.requires_grad:
                result.loss.backward()
        return TrainStats(loss=float(result.loss), mean_ratio=result.mean_ratio,
                          logprob_gap=result.logprob_gap,
                          grad_norm=self._grad_norm(state), tokens=len(batch),
                          provided=_summarize(provided), components=result.components)

    @serialized_learner_operation
    def accumulate(self, tenant: str,
                   batches) -> tuple[TrainStats, ...]:
        """Every microbatch of one update, in order (ADR 0022): one group,
        so the loop the Trainer always ran."""
        return tuple(self.forward_backward(tenant, batch) for batch in batches)

    def _provided(self, state: _Tenant) -> dict[str, Any]:
        """The bank's PROVIDED tensors for this forward, merged under their
        declared names — computed INSIDE the routed forward because they are
        part of this pass's graph and grad flows back through them.

        One owner per name, exactly as the post pipeline has one owner per
        column: two entries providing the same string would leave a loss
        reading whichever the bank order happened to put last, so it is refused
        here instead.
        """
        provided: dict[str, Any] = {}
        for entry in state.entries:
            for name, value in state.adapter_types[entry].provide(
                    state.params[entry]).items():
                if name in provided:
                    raise ValueError(
                        f"two bank entries provide {name!r} to the forward — a "
                        f"loss requiring it could not say whose it meant")
                provided[name] = value
        return provided

    @serialized_learner_operation
    def optim_step(self, tenant: str,
                   lr_scales: Mapping[str, float] | None = None) -> None:
        """One step of every entry that accumulated a gradient. `lr_scales` multiplies the named
        groups' BASE learning rates for THIS step (see `scaled_lrs`); with
        none given no group's lr is touched at all."""
        state = self._tenant(tenant)
        self.gradient_cache.invalidate(id(state))
        if lr_scales:
            check_lr_scales(lr_scales, {entry: [name for name, _ in groups]
                                        for entry, groups in state.base_lrs.items()})
        # Unrouted banks retain their shared norm. Explicitly routed banks
        # clip each entry independently; an absent entry never advances Adam.
        touched = [entry for entry in state.trainable
                   if any(p.grad is not None for p in state.params[entry].parameters())]
        together = [p for entry in touched
                    if not state.adapter_types[entry].clips_by_group
                    and not state.routed_slots
                    for p in state.params[entry].parameters()]
        if together:
            torch.nn.utils.clip_grad_norm_(together, self.grad_clip)
        for entry in touched:
            if state.adapter_types[entry].clips_by_group:
                for group in state.optimizers[entry].param_groups:
                    held = [p for p in group["params"] if p.grad is not None]
                    if held:
                        torch.nn.utils.clip_grad_norm_(held, self.grad_clip)
            elif state.routed_slots:
                torch.nn.utils.clip_grad_norm_(
                    state.params[entry].parameters(), self.grad_clip)
        for entry in touched:
            optimizer = state.optimizers[entry]
            with scaled_lrs(optimizer, entry, state.base_lrs[entry], lr_scales):
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for entry in touched:
            state.adapter_types[entry].after_step(state.params[entry])

    @serialized_learner_operation
    def forward(self, tenant: str, batch: TokenBatch) -> tuple[float, ...]:
        """Per-document mean NLL over the document's loss_mask tokens, with no
        gradient: the SAME routed padded forward `forward_backward` scores
        with (`_batched_logprobs`, rows routed by their own facts, the base in
        the eval mode it always is in), so a probe reads exactly the numbers
        the loss would have. A document with no masked token scores 0.0.
        No gradient accumulates and no provided tensor is computed; what a
        routed forward does at a site it still does (a `calibrate` dream_bank
        counts these activations too)."""
        state = self._tenant(tenant)
        self.forwards += 1
        spans = _doc_spans(batch)
        with torch.no_grad():
            with row_plan(self._model).route(self._rows_of(state, batch, len(spans))):
                logprobs = self._batched_logprobs(batch, spans)
        mask = torch.tensor(batch.loss_mask, dtype=logprobs.dtype,
                            device=logprobs.device)
        return tuple(
            float(-(logprobs[start:stop] * mask[start:stop]).sum()
                  / mask[start:stop].sum().clamp(min=1.0))
            for start, stop in spans)

    @serialized_learner_operation
    def gradient_cosines(
        self, tenant: str, entry: str,
        reference_batches: Sequence[TokenBatch],
        candidate_batches: Sequence[Sequence[TokenBatch]],
    ) -> tuple[GradientComparison, ...]:
        """Read ordinary SFT gradient alignment at one unchanged entry state.

        The base must already be in eval mode. No training flag or RNG state
        is changed/restored: doing so could overwrite another tenant's state.
        `enable_grad` is thread-local and restored on success and failure.
        Replayed stochastic draws, if any, must come from the batch's facts.
        """
        if self.fsdp != 1 or self.dp != 1:
            raise NotImplementedError("gradient_cosines requires fsdp=dp=1")
        if torch.is_inference_mode_enabled():
            raise RuntimeError("gradient_cosines cannot run in torch inference mode")
        state = self._tenant(tenant)
        if entry not in state.trainable:
            raise ValueError(f"gradient_cosines requires a trainable entry: {entry!r}")
        # Replay wrappers are installed after _ensure_base.eval(). Their own
        # default training flag does not control LoRA replay; the wrapped
        # base modules retain their eval flags. Do not toggle either tenant.
        if any(module.training and not isinstance(module, SiteWrapper)
               for module in self._model.modules()):
            raise RuntimeError("gradient_cosines requires the base already in eval mode")
        check_gradient_documents(reference_batches, "reference")
        if not candidate_batches:
            raise ValueError("gradient_cosines needs at least one candidate group")
        for index, batches in enumerate(candidate_batches):
            check_gradient_documents(batches, f"candidate[{index}]")
        parameters, schema = self._gradient_parameters(state, entry)
        parameter_count = sum(parameter.numel() for parameter in parameters)
        comparisons = []
        # Tensor versions change on optimizer steps and payload loads. A fresh
        # installation has a new state object; device moves also change context.
        context = (id(state), entry, schema, tuple(
            (id(p), p._version, str(p.device), str(p.dtype)) for p in parameters))
        cache = self.gradient_cache
        cache.select(context)
        reference_key = cache.key(reference_batches)
        with torch.enable_grad():
            started = time.perf_counter()
            saved = cache.reference(reference_key)
            reference_hit = saved is not None
            if saved is None:
                reference, reference_documents, reference_loss_reason = self._document_gradients(
                    state, parameters, reference_batches)
                reference_norm, reference_reason = measured_gradient_norm(reference, "reference")
                cache.save_reference(reference_key, reference, reference_documents,
                                     reference_loss_reason, reference_norm, reference_reason)
            else:
                (reference, reference_documents, reference_loss_reason,
                 reference_norm, reference_reason) = saved
            reference_seconds = time.perf_counter() - started
            for batches in candidate_batches:
                started = time.perf_counter()
                candidate_key = (reference_key, cache.key(batches))
                saved_comparison = cache.comparison(candidate_key)
                if saved_comparison is not None:
                    comparisons.append(replace(saved_comparison,
                        reference_seconds=reference_seconds,
                        candidate_seconds=time.perf_counter() - started,
                        reference_cache_hit=reference_hit, candidate_cache_hit=True))
                    continue
                candidate, candidate_documents, candidate_loss_reason = self._document_gradients(
                    state, parameters, batches)
                candidate_norm, candidate_reason = measured_gradient_norm(candidate, "candidate")
                reason = (reference_loss_reason or candidate_loss_reason
                          or reference_reason or candidate_reason)
                dot = measured_gradient_dot(reference, candidate)
                cosine = None
                if reason is None:
                    cosine = dot / (reference_norm * candidate_norm)
                    if not math.isfinite(cosine) or abs(cosine) > 1 + 1e-6:
                        reason = "invalid_cosine_numerics"
                        cosine = None
                    else:
                        # Bound roundoff only; no objective clipping or norm penalty.
                        cosine = max(-1.0, min(1.0, cosine))
                comparisons.append(GradientComparison(
                    cosine=cosine, dot=dot, reference_norm=reference_norm,
                    candidate_norm=candidate_norm, valid=reason is None, reason=reason,
                    reference_documents=reference_documents,
                    candidate_documents=candidate_documents,
                    reference_seconds=reference_seconds,
                    candidate_seconds=time.perf_counter() - started,
                    parameter_count=parameter_count, parameter_schema_sha256=schema,
                    min_norm=GRADIENT_MIN_NORM,
                    reference_cache_hit=reference_hit))
                cache.save_comparison(candidate_key, comparisons[-1])
        return tuple(comparisons)

    def _gradient_parameters(self, state: _Tenant, entry: str):
        """One stable coordinate order: sorted groups, then their parameter order."""
        groups = state.adapter_types[entry].param_groups(state.params[entry])
        parameters = []
        coordinates = []
        seen = set()
        for group in sorted(groups):
            for index, parameter in enumerate(groups[group]):
                if not parameter.requires_grad:
                    continue
                if id(parameter) in seen:
                    raise ValueError("gradient_cosines parameter appears in two groups")
                seen.add(id(parameter))
                parameters.append(parameter)
                coordinates.append((group, index, list(parameter.shape), str(parameter.dtype)))
        if not parameters:
            raise ValueError("gradient_cosines entry has no differentiable parameters")
        layout = {"sites": [site.path for site in state.sites[entry]],
                  "coordinates": coordinates}
        schema = hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()
        return tuple(parameters), schema

    def _document_gradients(self, state: _Tenant, parameters,
                            batches: Sequence[TokenBatch]):
        """Use sequence_sft's exact reductions, accumulating into private fp32 tensors."""
        gradients = [torch.zeros_like(parameter, dtype=torch.float32,
                                      memory_format=torch.preserve_format)
                     for parameter in parameters]
        documents = []
        loss_reason = None
        for batch in batches:
            self.forwards += 1
            spans = _doc_spans(batch)
            with row_plan(self._model).route(self._rows_of(state, batch, len(spans))):
                logprobs = self._batched_logprobs(batch, spans)
                mask = torch.tensor(batch.loss_mask, dtype=torch.bool, device=logprobs.device)
                # Zero weight means absent, even if an unused score is NaN.
                # Undefined derivatives inside the model still invalidate the
                # measured gradient; masking cannot repair those derivatives.
                logprobs = torch.where(mask, logprobs, torch.zeros_like(logprobs))
                result = sequence_sft(PolicyOutputs(logprobs=logprobs), batch)
                for index in range(len(spans)):
                    nll = float(result.components[f"document_{index}_nll_sum"])
                    if not math.isfinite(nll):
                        loss_reason = "non_finite_document_nll"
                        nll = None
                    documents.append(GradientDocumentStats(
                        nll_sum=nll,
                        scored_tokens=int(result.components[f"document_{index}_tokens"])))
                if result.loss.requires_grad:
                    local = torch.autograd.grad(result.loss, parameters, allow_unused=True)
                    for accumulated, gradient in zip(gradients, local, strict=True):
                        if gradient is not None:
                            accumulated.add_(gradient.detach().to(dtype=torch.float32))
        return tuple(gradients), tuple(documents), loss_reason

    # ---- named sets (ADR 0019): one route of one entry at a time ------------

    @serialized_learner_operation
    def load_set(self, tenant: str, entry: str, route: str,
                 payload: bytes | None) -> None:
        """Start one set over — from a named payload, or from its init when
        None — and reset its moments in that entry's optimizer. A `lib:<name>`
        route installs a frozen library set outside every optimizer group.
        The adapter type owns what a set IS; this picks the entry."""
        state = self._tenant(tenant)
        self.gradient_cache.invalidate(id(state))
        self._adapter_type(state, entry).load_set(
            state.params[entry], route, payload, state.optimizers.get(entry))

    @serialized_learner_operation
    def drop_set(self, tenant: str, entry: str, route: str) -> None:
        """Forget one library set of `entry` (library sets only)."""
        state = self._tenant(tenant)
        self.gradient_cache.invalidate(id(state))
        self._adapter_type(state, entry).drop_set(state.params[entry], route)

    @serialized_learner_operation
    def emit_set(self, tenant: str, entry: str, route: str) -> bytes:
        """One set of `entry` as a named payload."""
        state = self._tenant(tenant)
        return self._adapter_type(state, entry).emit_set(state.params[entry], route)

    @staticmethod
    def _adapter_type(state: _Tenant, entry: str):
        if entry not in state.adapter_types:
            raise KeyError(f"this tenant has no bank entry {entry!r} "
                           f"(entries: {state.entries})")
        return state.adapter_types[entry]

    @serialized_learner_operation
    def emit(self, tenant: str) -> Emitted:
        return Emitted(adapters=self.emit_adapters(tenant), optim=self.emit_optim(tenant))

    @serialized_learner_operation
    def bind_versions(self, tenant: str, versions: Mapping[str, int]) -> None:
        state = self._tenant(tenant)
        for entry, params in state.params.items():
            params.bind_version(versions[entry])

    @serialized_learner_operation
    def emit_initial(self, tenant: str, versions: Mapping[str, int]) -> Emitted:
        self.bind_versions(tenant, versions)
        return self.emit(tenant)

    @serialized_learner_operation
    def emit_adapters(self, tenant: str, versions: Mapping[str, int] | None = None) -> Mapping[str, bytes]:
        state = self._tenant(tenant)
        if versions is None:       # legacy initial export, including its draw counter
            return {entry: state.adapter_types[entry].emit(state.params[entry])
                    for entry in state.params}
        return {entry: state.adapter_types[entry].export_state(
                    state.params[entry], versions[entry]).encode()
                for entry in state.params}

    @serialized_learner_operation
    def export_state(self, tenant: str, versions: Mapping[str, int], names: Sequence[str]):
        state = self._tenant(tenant)
        return {entry: state.adapter_types[entry].export_state(
                    state.params[entry], versions[entry])
                for entry in names}

    @serialized_learner_operation
    def policy_pin(self, tenant, versions, names, adapter_types):
        """Content identity without shipping weights or an optimizer archive."""
        from rlstack.policy.compile import compile_tensor_pin
        states = self.export_state(tenant, versions, names)
        return compile_tensor_pin({name: snapshot.layout() for name, snapshot in states.items()},
                                  versions, adapter_types)

    @serialized_learner_operation
    def prepare_transfer(self, tenant, versions, names, adapter_types) -> dict:
        from rlstack.runner.transports.tensor import TensorSender
        if self._weight_sender is None:
            self._weight_sender = TensorSender(self.device)
        started = time.monotonic()
        states = self.export_state(tenant, versions, names)
        materialized = time.monotonic()
        manifest = self._weight_sender.capture(states, versions, adapter_types, owner=tenant)
        manifest["materialize_s"] = materialized - started
        return manifest

    @serialized_learner_operation
    def send_transfer(self, token, endpoint, timeout_s) -> dict:
        if self._weight_sender is None:
            raise ValueError("no tensor snapshot was captured")
        return self._weight_sender.send(token, endpoint, timeout_s)

    @serialized_learner_operation
    def release_transfer(self, token) -> None:
        if self._weight_sender is not None:
            self._weight_sender.release(token)

    @serialized_learner_operation
    def emit_optim(self, tenant: str) -> Mapping[str, bytes]:
        state = self._tenant(tenant)
        return {entry: _state_bytes(state.optimizers[entry]) for entry in state.trainable}

    @serialized_learner_operation
    def load(self, tenant: str, adapters: Mapping[str, bytes],
             optim: Mapping[str, bytes] | None) -> None:
        """Restore in place. Install already placed this tenant's params on
        the base's device, so restored moments land beside them (AdamW's
        load_state_dict casts to each param's device) — placement is settled
        before any load, never after it."""
        state = self._tenant(tenant)
        self.gradient_cache.invalidate(id(state))
        for entry, payload in adapters.items():
            state.adapter_types[entry].load(state.params[entry], payload)
        if optim:
            for entry, payload in optim.items():
                state.optimizers[entry].load_state_dict(_state_from(payload))
        else:
            for optimizer in state.optimizers.values():
                optimizer.state.clear()

    # ---- the sleep seam: a door verb, the learner's half of alternation -----

    async def sleep(self) -> None:
        """Hand the device back: the frozen base, every tenant's params and
        moments go to host RAM and the allocator's cache is returned — the
        learner's twin of vLLM's level-1 sleep (ADR 0002, Q8a). On an
        alternating host this is what keeps ONE base copy resident at a time.

        ASYNC because the arbiter's evict/wake hooks are (ADR 0002, Q8), and
        for no other reason: the move itself is synchronous, and lives in
        `hand_the_device_back` so that the sharded build's chorus — whose
        followers hear verbs on a blocking wire, not on a loop — can run the
        SAME body rank 0 runs."""
        self.hand_the_device_back()

    async def wake(self) -> None:
        """The inverse. The arbiter switches only at zero in-flight work, so
        no forward meets a half-woken learner."""
        self.take_the_device_back()

    @serialized_learner_operation
    def hand_the_device_back(self) -> None:
        """sleep's body, on whichever rank runs it. Idempotent, and quiet on a
        learner that holds no base yet — which is what lets a host evict a
        learner it has not yet installed a tenant on."""
        if self._model is None or self._asleep:
            return
        self.gradient_cache.clear()
        self._move_everything("cpu")
        torch.cuda.empty_cache()
        self._asleep = True

    @serialized_learner_operation
    def take_the_device_back(self) -> None:
        """wake's body, on whichever rank runs it. Idempotent, and quiet on a
        learner that never slept."""
        if self._model is None or not self._asleep:
            return
        self._move_everything(self.device)
        self._asleep = False

    @serialized_learner_operation
    def shutdown(self) -> None:
        """Close publication groups; the sharded build also ends its rank chorus."""
        if self._weight_sender is not None:
            self._weight_sender.close()
            self._weight_sender = None

    def _move_everything(self, device: str) -> None:
        self.move_the_base(device)
        self.move_the_tenants(device)

    @serialized_learner_operation
    def move_the_base(self, device: str) -> None:
        """The frozen base and its buffers across the bus, in one `Module.to`.

        Nothing trainable rides along, by design: a tenant's deltas hang off
        the site wrappers in a PLAIN LIST and are never registered as the
        base's own parameters (`soft_prompt_torch.PromptBoundary` states the
        rule for its boundary; `replay.SiteWrapper.installed` is the same list
        for every module-replacing family). So this moves the base alone and
        `move_the_tenants` moves the rest.

        The SHARDED build overrides this: a shard's storage may not be swapped
        without re-aliasing FSDP's own view of it (fsdp_torch.move_the_base).
        """
        self._model.to(device)

    @serialized_learner_operation
    def move_the_tenants(self, device: str) -> None:
        """Every installed tenant's deltas, their grads, and their optimizer
        moments — PER TENSOR and in place (`parameter.data = ...`), never
        `Module.to`, because the optimizer holds these Parameter objects by
        identity and a rebind would leave it stepping tensors nobody reads.

        Best-effort on what it reaches: each params object's `parameters()`
        and each optimizer's state tensors. An adapter type holding device
        tensors outside those two is a stated gap, proven only on metal.
        """
        for state in self._tenants.values():
            for params in state.params.values():
                for parameter in params.parameters():
                    parameter.data = parameter.data.to(device)
                    if parameter.grad is not None:
                        parameter.grad = parameter.grad.to(device)
            for optimizer in state.optimizers.values():
                for slot in optimizer.state.values():
                    for key, value in list(slot.items()):
                        if torch.is_tensor(value):
                            slot[key] = value.to(device)

    # ---- tenancy ------------------------------------------------------------

    def _ensure_base(self, base: str) -> None:
        """One learner, one base: the first install loads it, later tenants
        must agree (a different base needs its own learner — its own metal)."""
        if self._model is None:
            from transformers import AutoModelForCausalLM

            self._model = AutoModelForCausalLM.from_pretrained(
                base, torch_dtype=self.dtype).to(self.device)
            self._model.requires_grad_(False)
            self._model.eval()   # replay is exact recompute: no dropout, ever
            self._base = base
            self.checkpoint_the_blocks()
        elif self._base != base:
            raise ValueError(
                f"this learner holds base {self._base!r}; tenant wants "
                f"{base!r} — one learner serves one base")

    def decoder_blocks(self) -> list[torch.nn.Module]:
        """The base's decoder blocks, named `model.layers` in an HF causal LM —
        the same path the site schema resolves against."""
        return list(self._model.model.layers)

    def checkpoint_the_blocks(self) -> None:
        """Trade compute for memory: recompute each block's interior in the
        backward instead of keeping it from the forward.

        WHY THIS IS NOT OPTIONAL AT SCALE. A wave's documents are packed into
        forwards under `microbatch_tokens`, but pack never SPLITS a document
        (flatten.py), so the floor on one forward is the longest document — and
        a competition-math answer is ~1500 tokens. Storing every layer's
        interior for one such document costs ~14 GiB on a 14B base (#61
        measured it), which no batch knob can reduce, because the knob cannot
        go below one document. Checkpointing keeps only each block's INPUT
        (~15 MB) and pays about a third more backward compute.

        The forward is replaced in place rather than by wrapping the module:
        adapter installation and the site schema both address blocks by their
        module path (`model.layers.7.self_attn.q_proj`), and a wrapper would
        rename every one of them. use_reentrant=False because the blocks take
        keyword arguments and because it is the form that composes with FSDP's
        re-gather in the backward.

        Exactness is unaffected: the base is frozen and in eval, so a recomputed
        forward is the same arithmetic on the same weights — this buys memory,
        never a different number.
        """
        if not self.checkpoint_activations:
            return
        from torch.utils.checkpoint import checkpoint

        for block in self.decoder_blocks():
            block.forward = _recomputed(block.forward, checkpoint)

    def _claim_slot(self, state: _Tenant, sites: tuple[SiteMeta, ...],
                    params: object, route: str | None = None) -> None:
        """One delta per site and row route reaches its replay slot.

        Unrouted entries accompany every route, so their sites must be
        disjoint from every explicit route's sites.
        """
        slot = state.slot if route is None else state.routed_slots.setdefault(route, {})
        for meta in sites:
            conflicts = (slot, *state.routed_slots.values()) if route is None else (slot, state.slot)
            if any(meta.path in claimed for claimed in conflicts):
                raise ValueError(
                    f"two bank entries claim the replay path {meta.path!r} — "
                    f"a site carries at most one delta per row route")
            slot[meta.path] = params

    def _remove(self, tenant: str) -> None:
        """Unwire a tenant: every adapter type's uninstall_replay, install order
        reversed. Its params objects survive untouched — what leaves the tree
        is the routability of its deltas, not the deltas."""
        self.gradient_cache.clear()
        state = self._tenants.pop(tenant)
        for entry in reversed(state.entries):
            state.adapter_types[entry].uninstall_replay(
                self._model, state.params[entry], state.sites[entry])

    def _tenant(self, tenant: str) -> _Tenant:
        if tenant not in self._tenants:
            raise KeyError(f"tenant {tenant!r} was never installed")
        return self._tenants[tenant]

    # ---- the forward --------------------------------------------------------

    def _rows_of(self, state: _Tenant, batch: TokenBatch,
                 rows: int) -> ReplayRows:
        """Select the tenant's declared bank route for each document.

        Only routes present in this microbatch enter its slot table. Keeping
        absent parameters out of the graph preserves grad=None and Adam's
        inactive-state semantics. Without explicit bank routes, recorded
        facts remain entirely adapter-owned, including legacy internal routes.
        """
        if not state.routed_slots:
            return ReplayRows(slots=(state.slot,),
                              index=torch.zeros(rows, dtype=torch.long,
                                                device=self.device),
                              facts=batch.doc_turn_extras or None)
        facts = batch.doc_turn_extras or ((),) * rows
        selected: dict[str | None, int] = {}
        slots = []
        index = []
        for turns in facts:
            routes = {turn["route"] for turn in turns if turn.get("route") is not None}
            if len(routes) > 1:
                raise ValueError("one replay document must carry one adapter route")
            route = next(iter(routes), None)
            if route is not None and route not in state.routed_slots:
                raise ValueError(f"unknown adapter route {route!r}")
            if route not in selected:
                selected[route] = len(slots)
                slots.append({**state.slot, **state.routed_slots.get(route, {})})
            index.append(selected[route])
        return ReplayRows(slots=tuple(slots),
                          index=torch.tensor(index, dtype=torch.long, device=self.device),
                          facts=batch.doc_turn_extras or None)

    def _batched_logprobs(self, batch: TokenBatch,
                          spans: list[tuple[int, int]]) -> torch.Tensor:
        """[len(batch)] logprobs from ONE padded forward: position t scores
        token t given tokens < t.

        Row d is document d, LEFT-ALIGNED and right-padded, so causal
        attention over the padding mask scores each real position exactly as a
        document-at-a-time forward would — and the rows are exactly the unit
        the row plan routes. Position 0 of a document has no prefix; its
        logprob is 0.0, and flatten guarantees a doc never starts with a
        trainable token.

        What padding costs: the logits are rows × LONGEST document, while
        pack() bounds a microbatch by its token SUM. A wave of near-equal
        documents pays nothing; a very ragged one pays that ratio in logit
        memory. Length-bucketed sub-forwards are the fix — logged, not built.
        """
        width = max(stop - start for start, stop in spans)
        ids = torch.zeros((len(spans), width), dtype=torch.long,
                          device=self.device)
        attention = torch.zeros((len(spans), width), dtype=torch.long,
                                device=self.device)
        for row, (start, stop) in enumerate(spans):
            ids[row, :stop - start] = torch.tensor(
                batch.token_ids[start:stop], dtype=torch.long,
                device=self.device)
            attention[row, :stop - start] = 1
        # use_cache=False is load-bearing under checkpoint_the_blocks: a
        # checkpointed block runs AGAIN inside backward(), and a block that
        # appended K/V to a DynamicCache on the first pass would append a
        # second copy on the recompute — the replay forward decodes nothing,
        # so there is no cache to want
        start = unscored_prefix(batch, spans)
        split = trunk_and_head(self._model)
        if split is not None:
            trunk, head = split
            hidden = trunk(input_ids=ids, attention_mask=attention,
                           use_cache=False).last_hidden_state
            scored = head_logprobs(hidden[:, start:-1], head, ids[:, start + 1:],
                                   chunk=LOGPROB_CHUNK)
        else:
            keep = {} if start == 0 else {"logits_to_keep": width - start}
            logits = self._model(input_ids=ids, attention_mask=attention,
                                 use_cache=False, **keep).logits
            scored = chosen_logprobs(logits[:, :-1], ids[:, start + 1:])
        chosen = torch.cat([                                         # [R, W-1]
            torch.zeros((len(spans), start), dtype=torch.float32, device=self.device),
            scored], dim=1)
        zero = torch.zeros(1, dtype=chosen.dtype, device=self.device)
        return torch.cat([torch.cat([zero, chosen[row, :stop - start - 1]])
                          for row, (start, stop) in enumerate(spans)])

    def _grad_norm(self, state: _Tenant) -> float:
        total = 0.0
        for entry in state.trainable:
            for p in state.params[entry].parameters():
                if p.grad is not None:
                    total += float(p.grad.detach().pow(2).sum())
        return total ** 0.5


@contextmanager
def scaled_lrs(optimizer: torch.optim.Optimizer, entry: str,
               base_lrs: tuple[tuple[str, float], ...],
               lr_scales: Mapping[str, float] | None):
    """For the span of ONE step, each named group's lr is its BASE lr times
    its scale; afterwards every group has the lr it had before. Scaling from
    the base recorded at optimizer build — never from the lr a previous step
    left — is what makes a schedule a pure function of the step, and restoring
    in `finally` is what keeps a failed step from leaking a scaled rate. With
    no scales this touches nothing."""
    if not lr_scales:
        yield
        return
    before = [group["lr"] for group in optimizer.param_groups]
    try:
        for group, (name, base) in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base * lr_scale_of(lr_scales, entry, name)
        yield
    finally:
        for group, lr in zip(optimizer.param_groups, before):
            group["lr"] = lr


def _summarize(provided: Mapping[str, Any]) -> dict[str, float]:
    """Each provided tensor as ONE float, so it can be journaled: a scalar is
    itself, anything else is its mean.

    The summary is what makes `provides` an observability channel and not only a
    loss-input channel — every declared name lands in the ledger per update
    whether or not any loss requires it, for free and with no per-adapter
    plumbing. Detached, because this number is a report, not a gradient path.
    """
    return {name: float(value.detach().reshape(-1).mean())
            for name, value in provided.items()}


def check_gradient_documents(batches: Sequence[TokenBatch], role: str) -> None:
    """Every independent document has a nonempty binary mask and its own mean."""
    documents = sum(len(batch.doc_starts) for batch in batches)
    if not batches or not documents:
        raise ValueError(f"{role}: gradient measurement needs documents")
    for batch in batches:
        if not batch.doc_starts or batch.doc_starts[0] != 0:
            raise ValueError(f"{role}: document starts must begin at zero")
        if batch.documents_in_update != documents:
            raise ValueError(f"{role}: documents_in_update must equal {documents}")
        if any(bit not in (0, 1) for bit in batch.loss_mask):
            raise ValueError(f"{role}: gradient loss_mask must be binary")
        for start, stop in _doc_spans(batch):
            if start < 0 or stop > len(batch) or stop <= start:
                raise ValueError(f"{role}: invalid independent document span")
            if batch.loss_mask[start]:
                raise ValueError(f"{role}: a document's first token has no causal prediction")
            if not any(batch.loss_mask[start:stop]):
                raise ValueError(f"{role}: every document needs scored tokens")


def measured_gradient_norm(gradients, role: str) -> tuple[float | None, str | None]:
    """FP64 reduction of private fp32 gradients, with explicit invalid outcomes."""
    squared = torch.stack([gradient.to(dtype=torch.float64).square().sum()
                           for gradient in gradients]).sum()
    norm = float(squared.sqrt())
    if not math.isfinite(norm):
        return None, f"{role}_non_finite_gradient"
    if norm == 0:
        return norm, f"{role}_zero_gradient"
    if norm <= GRADIENT_MIN_NORM:
        return norm, f"{role}_tiny_gradient"
    return norm, None


def measured_gradient_dot(reference, candidate) -> float | None:
    """Only one scalar leaves the device, never an exported gradient vector."""
    dot = float(torch.stack([
        (left.to(dtype=torch.float64) * right.to(dtype=torch.float64)).sum()
        for left, right in zip(reference, candidate, strict=True)]).sum())
    return dot if math.isfinite(dot) else None


def _doc_spans(batch: TokenBatch) -> list[tuple[int, int]]:
    starts = list(batch.doc_starts)
    return list(zip(starts, starts[1:] + [len(batch)]))


def _state_bytes(optimizer: torch.optim.Optimizer) -> bytes:
    import io
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    return buffer.getvalue()


def _state_from(payload: bytes) -> dict:
    import io
    return torch.load(io.BytesIO(payload), map_location="cpu",
                      weights_only=False)


def _recomputed(forward, checkpoint):
    """One block's forward, run again in the backward instead of remembered.

    A closure over the ORIGINAL bound method, so the module tree is untouched
    and every site path still resolves. Under no_grad there is nothing to
    recompute for, and checkpoint would only add bookkeeping, so the plain
    forward runs — which is what keeps a scoring pass as cheap as it was.
    """
    def run(*args, **kwargs):
        if not torch.is_grad_enabled():
            return forward(*args, **kwargs)
        return checkpoint(forward, *args, use_reentrant=False, **kwargs)
    return run
