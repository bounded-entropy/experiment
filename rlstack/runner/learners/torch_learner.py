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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from rlstack.data.flatten import TokenBatch
from rlstack.policy.adapters.replay import ReplayRows, row_plan
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES, LOSSES
from rlstack.runner.interfaces import Emitted, Parameterization, TrainStats
from rlstack.training.losses import PolicyOutputs


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
    slot: dict[str, Any] = field(default_factory=dict)


class TorchLearner:
    def __init__(self, device: str | None = None,
                 dtype: torch.dtype = torch.bfloat16,
                 grad_clip: float = 1.0,
                 checkpoint_activations: bool = True) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.grad_clip = grad_clip
        # build fact, mirrored from VllmEngine.sleeps: an unsharded learner can
        # hand its device back (sleep/wake below). A resident's hello reports
        # it and the host wires the alternation hooks only when it says so.
        self.sleeps = True
        self._asleep = False
        self.checkpoint_activations = checkpoint_activations
        self.fsdp = 1               # build fact: this build is unsharded
        self._base: str | None = None
        self._model: torch.nn.Module | None = None
        self._tenants: dict[str, _Tenant] = {}

    # ---- Learner protocol ---------------------------------------------------

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
            self._claim_slot(state, entry.sites, params)
            if entry.name in state.trainable:
                state.optimizers[entry.name] = self._optimizer_for(
                    entry.name, params, adapter_type, parameterization.optim)
        self._tenants[tenant] = state

    def uninstall(self, tenant: str) -> None:
        """Install's inverse as a VERB (ADR 0006 Part A): the tenancy ends, so
        the module tree goes back to the way install found it. Idempotent — a
        tenant nobody installed is already uninstalled, which is what lets a
        host uninstall in a `finally` without knowing how far its run got."""
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
        entry-wide override, then the group's own. The default group (name "")
        is the whole entry, so only the entry-wide form addresses it."""
        settings: dict = {"lr": optim.lr, "weight_decay": optim.weight_decay}
        settings.update(optim.overrides.get(entry, {}))
        if group:
            settings.update(optim.overrides.get(f"{entry}.{group}", {}))
        return settings

    def forward_backward(self, tenant: str, batch: TokenBatch) -> TrainStats:
        state = self._tenant(tenant)
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
            result.loss.backward()
        return TrainStats(loss=float(result.loss), mean_ratio=result.mean_ratio,
                          logprob_gap=result.logprob_gap,
                          grad_norm=self._grad_norm(state), tokens=len(batch),
                          provided=_summarize(provided))

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

    def optim_step(self, tenant: str) -> None:
        state = self._tenant(tenant)
        parameters = [p for entry in state.trainable
                      for p in state.params[entry].parameters()]
        torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        for optimizer in state.optimizers.values():
            optimizer.step()
            optimizer.zero_grad()

    def emit(self, tenant: str) -> Emitted:
        state = self._tenant(tenant)
        adapters = {entry: state.adapter_types[entry].emit(state.params[entry])
                    for entry in state.params}
        optim = {entry: _state_bytes(state.optimizers[entry])
                 for entry in state.trainable}
        return Emitted(adapters=adapters, optim=optim)

    def load(self, tenant: str, adapters: Mapping[str, bytes],
             optim: Mapping[str, bytes] | None) -> None:
        """Restore in place. Install already placed this tenant's params on
        the base's device, so restored moments land beside them (AdamW's
        load_state_dict casts to each param's device) — placement is settled
        before any load, never after it."""
        state = self._tenant(tenant)
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
        Idempotent, and quiet on a learner that holds no base yet.

        Best-effort on what it moves: the module tree (which every installed
        replay half is wired into), each params object's `parameters()`, and
        each optimizer's state tensors. An adapter type holding device tensors
        outside those three is a stated gap, proven only on metal."""
        if self._model is None or self._asleep:
            return
        self._move_everything("cpu")
        torch.cuda.empty_cache()
        self._asleep = True

    async def wake(self) -> None:
        """The inverse. The arbiter switches only at zero in-flight work, so
        no forward meets a half-woken learner."""
        if self._model is None or not self._asleep:
            return
        self._move_everything(self.device)
        self._asleep = False

    def shutdown(self) -> None:
        """A resident's last verb. An unsharded learner holds no process of
        its own, so there is nothing to end; the sharded build overrides."""

    def _move_everything(self, device: str) -> None:
        self._model.to(device)
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
                    params: object) -> None:
        """One delta per site (the bank rule) reaches the forward as one
        params object per site PATH — the slot rows route to."""
        for meta in sites:
            if meta.path in state.slot:
                raise ValueError(
                    f"two bank entries claim the replay path {meta.path!r} — "
                    f"a site carries at most one delta")
            state.slot[meta.path] = params

    def _remove(self, tenant: str) -> None:
        """Unwire a tenant: every adapter type's uninstall_replay, install order
        reversed. Its params objects survive untouched — what leaves the tree
        is the routability of its deltas, not the deltas."""
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
        """Every row of a microbatch pins the verb's tenant: ONE slot, index
        all zeros. The replay lowering is per-row either way, so a coalesced
        microbatch is this same record with more slots and a mixed index —
        the sites need no change to serve it.

        The batch's per-document turn extras ride along as the rows' FACTS: the
        rows of a padded forward ARE the documents, so row r's facts are
        document r's. Adapter-blind by construction — this passes the mappings
        through and never reads a key.
        """
        return ReplayRows(slots=(state.slot,),
                          index=torch.zeros(rows, dtype=torch.long,
                                            device=self.device),
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
        logits = self._model(input_ids=ids, attention_mask=attention,
                             use_cache=False).logits
        given_prefix = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        chosen = given_prefix.gather(2, ids[:, 1:, None])[..., 0]   # [R, W-1]
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
