"""Custom operations share one rollout lowering and tensor installation path.

Only small IPC handles cross the local vLLM process boundary. Tensor state
uses ADR 0027 publication; there is no per-adapter archive or file consumer.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

import torch
from safetensors.torch import load as load_tensors
from torch.multiprocessing.reductions import reduce_storage, reduce_tensor

from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.rollout import Alignment, BuildDemands, Levers, RolloutLowering
from rlstack.policy.adapters.state_torch import TensorState
from rlstack.policy.adapters.definition import ProgramManifest
from rlstack.registry import ADAPTER_TYPES


@dataclass
class TensorHandle:
    rebuild: object
    args: tuple
    dtype: object = None
    offset: int = 0
    shape: tuple = ()
    stride: tuple = ()

    @classmethod
    def of(cls, tensor):
        value = tensor.detach().contiguous()
        if value.is_cuda:
            rebuild, args = reduce_tensor(value)
            return cls(rebuild, args)
        if value.device.type != "cpu":
            raise ValueError("site installation supports CPU or CUDA tensors")
        value.share_memory_()
        rebuild, args = reduce_storage(value.untyped_storage())
        return cls(rebuild, args, value.dtype, value.storage_offset(), tuple(value.shape), tuple(value.stride()))

    def tensor(self):
        result = self.rebuild(*self.args)
        if self.dtype is None:
            return result
        return torch.empty(0, dtype=self.dtype).set_(result, self.offset, self.shape, self.stride)


@dataclass
class SiteResident:
    bundle_id: str
    states: dict[str, TensorState]


class SiteRollout(RolloutLowering):
    mechanism = Mechanism.SITE

    def __init__(self, build, adapter_type, definition):
        super().__init__(build)
        self.adapter_type, self.definition = adapter_type, definition
        self.key = f"rlstack_site:{adapter_type}"
        self.claims = (f"extra_args.{self.key}",)
        self._prepared = set()
        self._retired = []
        self._prepare_lock = None
        self._retirement_lock = threading.Lock()

    def demands(self):
        types = self.build.site_types or (self.adapter_type,)
        modules = sorted({ADAPTER_TYPES.get(name).cls.__module__ for name in types})
        return BuildDemands(
            engine_args={"worker_cls": "rlstack_engine.site_worker.SiteWorker",
                         "enable_sleep_mode": False},
            env={"VLLM_USE_V2_MODEL_RUNNER": "0",
                 "RLSTACK_SITE_TYPES": json.dumps(types),
                 "RLSTACK_SITE_MODULES": json.dumps(modules),
                 "RLSTACK_SITE_SLOTS": str(self.build.max_bundles)})

    def reaches(self, meta):
        # Embeddings and vocabulary heads are different operation contracts.
        return (self.definition.accepts(meta)
                and meta.path.rsplit(".", 1)[-1] not in ("embed_tokens", "lm_head"))

    def attach(self, bundle_id, payloads):
        states = {}
        for name, payload in payloads.items():
            size = int.from_bytes(payload[:8], "little")
            metadata = json.loads(payload[8:8 + size]).get("__metadata__")
            states[name] = TensorState(load_tensors(payload), metadata=metadata)
        return self.attach_state(bundle_id, states)

    def attach_state(self, bundle_id, states):
        expected = self.definition.source_hash()
        paths = set()
        for state in states.values():
            if state.prefix or state.metadata is None:
                raise ValueError("site publication definition differs from this engine build")
            manifest = ProgramManifest.from_metadata(state.metadata)
            if manifest.program_hash != expected:
                raise ValueError("site publication definition differs from this engine build")
            current = set(manifest.sites)
            if paths & current:
                raise ValueError("two site entries overlap in one bundle")
            paths |= current
        return SiteResident(bundle_id, dict(states))

    async def prepare(self, engine, attached):
        import asyncio
        if self._prepare_lock is None:
            self._prepare_lock = asyncio.Lock()
        async with self._prepare_lock:
            # Retirement happens after host request holds expire and before a
            # subsequent load can reuse worker slots. Every rank acknowledges.
            with self._retirement_lock:
                retired, self._retired = self._retired, []
            try:
                for bundle_id in retired:
                    await engine.collective_rpc("release_site_state", timeout=60,
                                                args=(self.adapter_type, bundle_id))
                    self._prepared.discard(bundle_id)
            except BaseException:
                with self._retirement_lock:
                    self._retired[:0] = retired
                raise
            if attached.bundle_id in self._prepared:
                return
            states = [(state.metadata, {name: TensorHandle.of(tensor)
                       for name, tensor in state.tensors.items()}) for state in attached.states.values()]
            acknowledgements = await engine.collective_rpc("load_site_state", timeout=60, args=(
                self.adapter_type, attached.bundle_id, states))
            if not acknowledgements or any(ack != {
                    "adapter_type": self.adapter_type, "bundle_id": attached.bundle_id,
                    "ready": True} for ack in acknowledgements):
                raise RuntimeError("site workers did not acknowledge the exact parameter version")
            self._prepared.add(attached.bundle_id)

    def apply(self, attached, request):
        return Levers(extra_args={self.key: attached.bundle_id},
                      cache_salt=f"site:{attached.bundle_id}")

    def align(self, attached):
        return Alignment(0)

    def detach(self, attached):
        # A cancelled/uncertain load may have reached the worker without an
        # acknowledgement. Retire it too; worker release is idempotent.
        with self._retirement_lock:
            self._retired.append(attached.bundle_id)
