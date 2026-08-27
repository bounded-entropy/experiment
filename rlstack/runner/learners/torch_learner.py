"""TorchLearner: the Learner protocol on real metal (Phase B2, multi-tenant).

Owns ONE frozen HF base shared by every tenant (the memory asymmetry that
capped tenancy — CONTEXT #34), per-tenant adapter params built and installed
by each kind's OWN compute half, and one optimizer per (tenant, entry) so
optim blobs map 1:1 onto the store's optim/<name>@v.

Tenancy is swap-install: exactly one tenant's adapters are wired into the
module tree at a time; `_ensure_active` uninstalls the previous tenant's
kinds (their exact inverse, module rebinds only — no weight copies) and
installs the requester's. Params objects survive deactivation untouched, so
switching is numerics-exact and costs microseconds.

v0 choices, stated: docs run one at a time (no cross-doc packing in the
forward — correct first, fast later); attention is the stock HF sdpa path;
determinism is best-effort (CUDA kernels are not bit-stable — byte-identical
resume stays a fakes-suite property; the real-metal invariant is the ledger's
logprob_gap staying small).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from rlstack.data.flatten import TokenBatch
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTERS, LOSSES
from rlstack.runner.interfaces import Emitted, TrainStats
from rlstack.spec.specs import ExperimentSpec
from rlstack.training.losses import PolicyOutputs


def _init_seed(master: int, entry: str) -> int:
    digest = hashlib.sha256(f"{master}:init:{entry}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class _Tenant:
    """One experiment's state on this learner: params, kinds, optimizers."""

    loss_fn: object
    trainable: list[str]
    entries: list[str]                                  # install order
    params: dict[str, object] = field(default_factory=dict)
    kinds: dict[str, object] = field(default_factory=dict)
    sites: dict[str, tuple[SiteMeta, ...]] = field(default_factory=dict)
    optimizers: dict[str, torch.optim.Optimizer] = field(default_factory=dict)


class TorchLearner:
    def __init__(self, device: str | None = None,
                 dtype: torch.dtype = torch.bfloat16,
                 grad_clip: float = 1.0) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.grad_clip = grad_clip
        self.fsdp = 1               # build fact (#43): this build is unsharded
        self._base: str | None = None
        self._model: torch.nn.Module | None = None
        self._tenants: dict[str, _Tenant] = {}
        self._active: str | None = None

    # ---- Learner protocol ---------------------------------------------------

    def install(self, tenant: str, spec: ExperimentSpec,
                resolved_sites: Mapping[str, tuple[SiteMeta, ...]]) -> None:
        self._ensure_base(spec.policy.base)
        if tenant in self._tenants:            # Phase 1 re-runs on attach
            if self._active == tenant:
                self._deactivate()
            del self._tenants[tenant]

        state = _Tenant(
            loss_fn=LOSSES.get(spec.algo.loss).fn,
            trainable=sorted(n for n, a in spec.policy.bank.items()
                             if a.trainable),
            entries=list(spec.policy.bank),
        )
        for entry, adapter_spec in spec.policy.bank.items():
            kind = ADAPTERS.get(adapter_spec.kind).instance
            init = dict(adapter_spec.init)
            init.setdefault("seed", _init_seed(spec.seeds.master, entry))
            params = kind.params(resolved_sites[entry], init)
            state.params[entry] = params
            state.kinds[entry] = kind
            state.sites[entry] = resolved_sites[entry]
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
        self._ensure_active(tenant)
        logprobs = self._batched_logprobs(batch)
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
        adapters = {entry: state.kinds[entry].emit(state.params[entry])
                    for entry in state.params}
        optim = {entry: _state_bytes(state.optimizers[entry])
                 for entry in state.trainable}
        return Emitted(adapters=adapters, optim=optim)

    def load(self, tenant: str, adapters: Mapping[str, bytes],
             optim: Mapping[str, bytes] | None) -> None:
        state = self._tenant(tenant)
        for entry, payload in adapters.items():
            state.kinds[entry].load(state.params[entry], payload)
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

    def _ensure_active(self, tenant: str) -> None:
        """Swap-install: wire `tenant`'s adapters into the module tree,
        unwinding the previous tenant's first. Rebinds only, never copies."""
        if self._active == tenant:
            return
        self._deactivate()
        state = self._tenants[tenant]
        for entry in state.entries:
            state.kinds[entry].install_replay(
                self._model, state.params[entry], state.sites[entry])
        for optimizer in state.optimizers.values():
            _colocate_optim_state(optimizer)
        self._active = tenant

    def _deactivate(self) -> None:
        if self._active is None:
            return
        state = self._tenants[self._active]
        for entry in reversed(state.entries):
            state.kinds[entry].uninstall_replay(
                self._model, state.params[entry], state.sites[entry])
        self._active = None

    def _tenant(self, tenant: str) -> _Tenant:
        if tenant not in self._tenants:
            raise KeyError(f"tenant {tenant!r} was never installed")
        return self._tenants[tenant]

    # ---- the forward --------------------------------------------------------

    def _batched_logprobs(self, batch: TokenBatch) -> torch.Tensor:
        """Per-token logprobs for every doc, ONE padded forward per
        microbatch: position t scores token t given tokens < t within its
        own doc.

        Numerics are identical to the per-doc [1, L] forward it replaced:
        docs are left-aligned and attention is causal, so a real position
        never attends to the right padding behind it, and padded positions
        never enter the gather. Position 0 has no prefix; its logprob is
        0.0 — flatten guarantees a doc never starts with a trainable token
        (prompts come first)."""
        spans = _doc_spans(batch)
        docs = [batch.token_ids[start:stop] for start, stop in spans]
        longest = max(len(doc) for doc in docs)
        padded = torch.zeros((len(docs), longest), dtype=torch.long,
                             device=self.device)
        for row, doc in enumerate(docs):
            padded[row, :len(doc)] = torch.tensor(doc, dtype=torch.long,
                                                  device=self.device)
        logits = self._model(padded).logits                 # [D, longest, V]
        zero = torch.zeros(1, dtype=torch.float32, device=self.device)
        per_doc = []
        for row, doc in enumerate(docs):
            length = len(doc)
            given_prefix = torch.log_softmax(
                logits[row, :length - 1].float(), dim=-1)   # [L-1, V]
            chosen = given_prefix.gather(
                1, padded[row, 1:length, None])[:, 0]       # [L-1]
            per_doc.append(torch.cat([zero, chosen]))
        return torch.cat(per_doc)

    def _grad_norm(self, state: _Tenant) -> float:
        total = 0.0
        for entry in state.trainable:
            for p in state.params[entry].parameters():
                if p.grad is not None:
                    total += float(p.grad.detach().pow(2).sum())
        return total ** 0.5


def _colocate_optim_state(optimizer: torch.optim.Optimizer) -> None:
    """Optimizer moments live WHERE THEIR PARAMS LIVE — the rule activation
    enforces. A tenant `load`ed before its first activation has CPU moments
    (load_state_dict casts to the params' device, and params move to the GPU
    only when the kind's install_replay wires them in), so the first
    optim_step after a resume would mix devices. Activation is the placement
    moment of truth; after the first pass this is a no-op scan."""
    for group in optimizer.param_groups:
        for param in group["params"]:
            moments = optimizer.state.get(param)
            if not moments:
                continue
            for key, value in moments.items():
                if torch.is_tensor(value) and value.device != param.device:
                    moments[key] = value.to(param.device)


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
