"""TorchLearner: the Learner protocol on real metal (Phase B2, v0).

Owns the frozen HF base, per-entry adapter params (built and installed by each
kind's OWN compute half — the learner never knows what a LoRA is), one
optimizer per entry (so optim blobs map 1:1 onto the store's optim/<name>@v),
and the forward that turns a TokenBatch into PolicyOutputs for the registered
loss.

v0 choices, stated: docs run one at a time (no cross-doc packing in the
forward — correct first, fast later); attention is the stock HF sdpa path;
determinism is best-effort (CUDA kernels are not bit-stable — byte-identical
resume stays a fakes-suite property; the real-metal invariant is the ledger's
logprob_gap staying small).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

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


class TorchLearner:
    def __init__(self, device: str | None = None,
                 dtype: torch.dtype = torch.bfloat16,
                 grad_clip: float = 1.0) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.grad_clip = grad_clip
        self._model: torch.nn.Module | None = None
        self._params: dict[str, object] = {}          # entry -> kind's state
        self._kinds: dict[str, object] = {}           # entry -> Adapter instance
        self._optimizers: dict[str, torch.optim.Optimizer] = {}
        self._trainable: list[str] = []
        self._loss_fn = None

    # ---- Learner protocol ---------------------------------------------------

    def install(self, spec: ExperimentSpec,
                resolved_sites: Mapping[str, tuple[SiteMeta, ...]]) -> None:
        from transformers import AutoModelForCausalLM

        self._model = AutoModelForCausalLM.from_pretrained(
            spec.policy.base, torch_dtype=self.dtype).to(self.device)
        self._model.requires_grad_(False)
        self._model.eval()   # replay is exact recompute: no dropout, ever

        self._loss_fn = LOSSES.get(spec.algo.loss).fn
        self._trainable = sorted(name for name, a in spec.policy.bank.items()
                                 if a.trainable)
        for entry, adapter_spec in spec.policy.bank.items():
            kind = ADAPTERS.get(adapter_spec.kind).instance
            init = dict(adapter_spec.init)
            init.setdefault("seed", _init_seed(spec.seeds.master, entry))
            params = kind.params(resolved_sites[entry], init)
            kind.install_replay(self._model, params, resolved_sites[entry])
            self._params[entry] = params
            self._kinds[entry] = kind
            if entry in self._trainable:
                overrides = dict(spec.algo.optim.overrides.get(entry, {}))
                self._optimizers[entry] = torch.optim.AdamW(
                    params.parameters(),
                    lr=float(overrides.get("lr", spec.algo.optim.lr)),
                    betas=spec.algo.optim.betas,
                    weight_decay=spec.algo.optim.weight_decay)

    def forward_backward(self, batch: TokenBatch) -> TrainStats:
        logprobs = torch.cat([self._doc_logprobs(batch, start, stop)
                              for start, stop in _doc_spans(batch)])
        result = self._loss_fn(PolicyOutputs(logprobs=logprobs), batch)
        result.loss.backward()
        return TrainStats(loss=float(result.loss), mean_ratio=result.mean_ratio,
                          logprob_gap=result.logprob_gap,
                          grad_norm=self._grad_norm(), tokens=len(batch))

    def optim_step(self) -> None:
        parameters = [p for entry in self._trainable
                      for p in self._params[entry].parameters()]
        torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
        for optimizer in self._optimizers.values():
            optimizer.step()
            optimizer.zero_grad()

    def emit(self) -> Emitted:
        adapters = {entry: self._kinds[entry].emit(self._params[entry])
                    for entry in self._params}
        optim = {entry: _state_bytes(self._optimizers[entry])
                 for entry in self._trainable}
        return Emitted(adapters=adapters, optim=optim)

    def load(self, adapters: Mapping[str, bytes],
             optim: Mapping[str, bytes] | None) -> None:
        for entry, payload in adapters.items():
            self._kinds[entry].load(self._params[entry], payload)
        if optim:
            for entry, payload in optim.items():
                self._optimizers[entry].load_state_dict(_state_from(payload))
        else:
            for optimizer in self._optimizers.values():
                optimizer.state.clear()

    # ---- the forward --------------------------------------------------------

    def _doc_logprobs(self, batch: TokenBatch, start: int, stop: int) -> torch.Tensor:
        """[stop-start] logprobs: position t scores token t given tokens < t.

        Position 0 has no prefix; its logprob is 0.0 — flatten guarantees a
        doc never starts with a trainable token (prompts come first).
        """
        ids = torch.tensor(batch.token_ids[start:stop], dtype=torch.long,
                           device=self.device)
        logits = self._model(ids[None]).logits[0]           # [L, V]
        given_prefix = torch.log_softmax(logits[:-1].float(), dim=-1)
        chosen = given_prefix.gather(1, ids[1:, None])[:, 0]  # [L-1]
        zero = torch.zeros(1, dtype=chosen.dtype, device=self.device)
        return torch.cat([zero, chosen])

    def _grad_norm(self) -> float:
        total = 0.0
        for entry in self._trainable:
            for p in self._params[entry].parameters():
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
