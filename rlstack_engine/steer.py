"""SteerPlugin: the RESIDUAL mechanism — a per-slot vector added to the
residual stream at a boundary, per token, inside one fused batch (ADR 0004).

The lifecycle is the plugin contract's (probe / install / load / evict /
cache_salt); the per-forward verb is `add`. What is re-earned here, named:

  per-request selection   a request names its bundle's steer FILE in its
                          extra_args (the adapter type's Levers.extra_args);
                          `slot_of` turns that into a bank slot, loading the
                          file on first sight — the LoRARequest precedent, a
                          path the worker loads and caches under a bound.
  the window              the request's resolved [start, end) rides beside
                          the file; `routing` masks each token by its own
                          absolute position, so "every position" and
                          "completion only" and "constantly on decode" are one
                          rule evaluated per token, at prefill and at every
                          decode step.
  multi-tenancy           tokens of different requests in one forward gather
                          from their own slots (BatchView.token_slot); a
                          request naming no file adds nothing — a lora-only
                          tenant rides the same batch untouched.
  refusal                 a file the bank cannot load RAISES (the row plan's
                          rule on the engine side: an unrouted forward is a
                          wiring bug, never a fallback), and so does a bank
                          that would have to evict a slot the batch in flight
                          still reads.

torch and safetensors at module scope: this ships in the engine image only,
named by string from rlstack (STYLE rule 8's one-way import). The vLLM seam
that drives it lives beside it in steer_worker.py, so this file is
exercisable on a CPU with a BatchView and no engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from safetensors.torch import load as st_load

from rlstack import Mechanism
from rlstack.policy.adapters.steer import STEER_END, STEER_FILE, STEER_START

from rlstack_engine.batch_view import BatchView
from rlstack_engine.plugin import EnginePlugin
from rlstack_engine.slots import SlotTable, SlotsFull

NO_SLOT = -1


@dataclass(frozen=True)
class SteerRouting:
    """One forward's steer routing, as tensors on the device: for every token,
    the bank slot it gathers from and whether its position is inside its
    request's window. Built once per forward (the model's pre-hook), read at
    every boundary."""

    slot: torch.Tensor            # [tokens] long, NO_SLOT for none
    inside: torch.Tensor          # [tokens] bool
    slots: tuple[int, ...]        # the distinct slots this forward reads

    @property
    def steers(self) -> bool:
        return bool(self.slots)


class SteerPlugin(EnginePlugin):
    mechanism = Mechanism.RESIDUAL
    consumes = ("steer", "nsteer")
    # The seams this mechanism stands on, on the pinned build (vllm 0.28.0):
    # the worker class whose boot we extend, the forward context the batch
    # geometry is read from, the V1 runner's request table and input batch
    # (request ids in batch order — the V2 runner the build boots by default
    # has neither, so the adapter type demands V1), and the per-request
    # extra_args.
    required_symbols = frozenset({
        "vllm.v1.worker.gpu_worker.Worker.compile_or_warm_up_model",
        "vllm.forward_context.get_forward_context",
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner.input_batch",
        "vllm.v1.worker.gpu_model_runner.GPUModelRunner.requests",
        "vllm.sampling_params.SamplingParams.extra_args",
    })

    def __init__(self, max_slots: int, device: Any, dtype: Any) -> None:
        self.table = SlotTable(max_slots)
        self.device = device
        self.dtype = dtype
        self._banks: dict[int, dict[str, torch.Tensor]] = {}   # slot -> path -> [d]
        self._alpha: dict[int, float | None] = {}   # per slot: the fraction, or a plain steer
        self._order: list[int] = []                             # least recently seen first
        self._in_flight: frozenset[int] = frozenset()

    # ---- bundle lifecycle (the contract) -------------------------------------

    def load(self, slot: int, bundle_id: str, payloads: Mapping[str, bytes]) -> None:
        """One safetensors payload, {path: vector}, into the banks at `slot`,
        in the served dtype on the device."""
        if slot in self._banks:
            raise ValueError(
                f"slot {slot} already holds a steer bank; evict first")
        (payload,) = payloads.values()
        alpha = _payload_alpha(payload)
        vectors = st_load(payload)
        if alpha is not None:
            # a norm-scaled bank holds UNIT directions: the magnitude is
            # alpha times the token's own norm, applied at add time
            vectors = {path: v.float() / (v.float().norm() + 1e-12)
                       for path, v in vectors.items()}
        self._banks[slot] = {path: vector.to(self.device, self.dtype)
                             for path, vector in vectors.items()}
        self._alpha[slot] = alpha

    def evict(self, slot: int) -> None:
        del self._banks[slot]
        self._alpha.pop(slot, None)
        if slot in self._order:
            self._order.remove(slot)

    def bank(self, slot: int) -> Mapping[str, torch.Tensor]:
        """The vectors at `slot`, by boundary path — the one door to them."""
        return self._banks[slot]

    # ---- per-request selection ------------------------------------------------

    def slot_of(self, extra_args: Mapping[str, Any]) -> int:
        """The bank slot a request's extra_args name — loading the file on
        first sight, evicting the least recently seen bank the batch in flight
        does not read when the table is full. NO_SLOT for a request that
        steers nothing."""
        file = extra_args.get(STEER_FILE)
        if file is None:
            return NO_SLOT
        if file in self.table.resident():
            slot = self.table.slot_of(file)
        else:
            slot = self._acquire(file)
            self.load(slot, file, {"steer": _read(file)})
        self._order = [s for s in self._order if s != slot] + [slot]
        return slot

    def _acquire(self, file: str) -> int:
        try:
            return self.table.acquire(file)
        except SlotsFull:
            idle = [s for s in self._order if s not in self._in_flight]
            if not idle:
                raise RuntimeError(
                    f"the steer bank holds {self.table.max_slots} files and "
                    f"every one is read by the batch in flight; loading "
                    f"{file!r} would evict a live one")
            victim = idle[0]
            (victim_file,) = [f for f, s in self.table.resident().items()
                              if s == victim]
            self.table.release(victim_file)
            self.evict(victim)
            return self.table.acquire(file)

    # ---- the per-forward verb --------------------------------------------------

    def routing(self, view: BatchView) -> SteerRouting:
        """This forward's routing, from the view: each token's slot and
        whether its absolute position lies inside its request's window."""
        inside = []
        for token in range(len(view)):
            extra = view.request_extra[view.token_request[token]]
            start = int(extra.get(STEER_START, 0))
            end = extra.get(STEER_END)
            pos = view.position[token]
            inside.append(view.token_slot[token] != NO_SLOT and pos >= start
                          and (end is None or pos < int(end)))
        slots = tuple(sorted({s for s in view.token_slot if s != NO_SLOT}))
        self._in_flight = frozenset(slots)
        return SteerRouting(
            slot=torch.tensor(view.token_slot, dtype=torch.long,
                              device=self.device),
            inside=torch.tensor(inside, dtype=torch.bool, device=self.device),
            slots=slots)

    def add(self, routing: SteerRouting, path: str,
            hidden: torch.Tensor) -> None:
        """hidden[t] += bank[slot_t][path] for every token inside its window —
        in place, on the first len(routing.slot) rows (the actual tokens; a
        padded row past them is left alone). A slot without a vector at this
        path adds zero: that bundle steers elsewhere, and zero is the
        identity."""
        if not routing.steers:
            return
        width = int(hidden.shape[-1])
        zero = torch.zeros(width, device=hidden.device, dtype=hidden.dtype)
        table = torch.stack([zero] + [
            self._banks[slot].get(path, zero).to(hidden.dtype)
            for slot in routing.slots])                            # [1 + S, d]
        index = torch.zeros_like(routing.slot)
        for row, slot in enumerate(routing.slots, start=1):
            index[routing.slot == slot] = row
        n = int(routing.slot.shape[0])
        delta = table[index] * routing.inside.unsqueeze(-1).to(hidden.dtype)
        alphas = torch.tensor(
            [0.0] + [float(self._alpha.get(slot) or 0.0) for slot in routing.slots],
            device=hidden.device, dtype=torch.float32)[index]      # [tokens]
        if bool((alphas > 0).any()):
            # the norm-scaled add: alpha times THIS token's live residual norm,
            # along the bank's unit direction (a plain steer's alpha is 0 -> x1)
            norms = hidden[:n].float().norm(dim=-1)                  # [tokens]
            scale = torch.where(alphas > 0, alphas * norms,
                                torch.ones_like(norms))
            delta = (delta.float() * scale.unsqueeze(-1)).to(hidden.dtype)
        hidden[:n].add_(delta)


def _read(file: str) -> bytes:
    try:
        with open(file, "rb") as handle:
            return handle.read()
    except FileNotFoundError:
        raise RuntimeError(
            f"a request names steer file {file!r}, which this worker cannot "
            f"read: the bundle was detached under a request, or never "
            f"attached here") from None



def _payload_alpha(payload: bytes) -> float | None:
    """The fraction the fused file declares (safetensors metadata written by
    the lowering's attach), or None for a plain steer. Read here rather than
    imported: the engine image carries no rlstack training code."""
    import json
    import struct

    n = struct.unpack("<Q", payload[:8])[0]
    meta = (json.loads(payload[8:8 + n]).get("__metadata__") or {})
    return None if "rlstack_alpha" not in meta else float(meta["rlstack_alpha"])
