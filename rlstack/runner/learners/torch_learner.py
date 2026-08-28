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

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from rlstack.data.flatten import TokenBatch
from rlstack.policy.adapters.replay import ReplayRows, row_plan
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES, LOSSES
from rlstack.runner.interfaces import Emitted, TrainStats
from rlstack.spec.specs import ExperimentSpec
from rlstack.training.losses import PolicyOutputs


def _init_seed(master: int, entry: str) -> int:
    digest = hashlib.sha256(f"{master}:init:{entry}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


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
                 grad_clip: float = 1.0) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.grad_clip = grad_clip
        self.fsdp = 1               # build fact: this build is unsharded
        self._base: str | None = None
        self._model: torch.nn.Module | None = None
        self._tenants: dict[str, _Tenant] = {}

    # ---- Learner protocol ---------------------------------------------------

    def install(self, tenant: str, spec: ExperimentSpec,
                resolved_sites: Mapping[str, tuple[SiteMeta, ...]]) -> None:
        self._ensure_base(spec.policy.base)
        if tenant in self._tenants:            # Phase 1 re-runs on attach
            self._remove(tenant)

        state = _Tenant(
            loss_fn=LOSSES.get(spec.algo.loss).fn,
            trainable=sorted(n for n, a in spec.policy.bank.items()
                             if a.trainable),
            entries=list(spec.policy.bank),
        )
        for entry, adapter_spec in spec.policy.bank.items():
            adapter_type = ADAPTER_TYPES.get(adapter_spec.adapter_type).instance
            init = dict(adapter_spec.init)
            init.setdefault("seed", _init_seed(spec.seeds.master, entry))
            params = adapter_type.params(resolved_sites[entry], init)
            state.params[entry] = params
            state.adapter_types[entry] = adapter_type
            state.sites[entry] = resolved_sites[entry]
            adapter_type.install_replay(self._model, params, resolved_sites[entry])
            self._claim_slot(state, resolved_sites[entry], params)
            if entry in state.trainable:
                overrides = dict(spec.algo.optim.overrides.get(entry, {}))
                state.optimizers[entry] = torch.optim.AdamW(
                    params.parameters(),
                    lr=float(overrides.get("lr", spec.algo.optim.lr)),
                    betas=spec.algo.optim.betas,
                    weight_decay=spec.algo.optim.weight_decay)
        self._tenants[tenant] = state

    def forward_backward(self, tenant: str, batch: TokenBatch) -> TrainStats:
        state = self._tenant(tenant)
        spans = _doc_spans(batch)
        with row_plan(self._model).route(self._rows_of(state, len(spans))):
            logprobs = self._batched_logprobs(batch, spans)
        result = state.loss_fn(PolicyOutputs(logprobs=logprobs), batch)
        result.loss.backward()
        return TrainStats(loss=float(result.loss), mean_ratio=result.mean_ratio,
                          logprob_gap=result.logprob_gap,
                          grad_norm=self._grad_norm(state), tokens=len(batch))

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
        elif self._base != base:
            raise ValueError(
                f"this learner holds base {self._base!r}; tenant wants "
                f"{base!r} — one learner serves one base")

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

    def _rows_of(self, state: _Tenant, rows: int) -> ReplayRows:
        """Every row of a microbatch pins the verb's tenant: ONE slot, index
        all zeros. The replay lowering is per-row either way, so a coalesced
        microbatch is this same record with more slots and a mixed index —
        the sites need no change to serve it."""
        return ReplayRows(slots=(state.slot,),
                          index=torch.zeros(rows, dtype=torch.long,
                                            device=self.device))

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
        logits = self._model(input_ids=ids, attention_mask=attention).logits
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
