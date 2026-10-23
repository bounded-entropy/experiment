"""One adapter's exported tensors, before any byte serialization.

The adapter owns the keys and optional header. Shared publication owns the
device, copies, content manifest and wire backend. Checkpoints still encode
the same safetensors payloads; NCCL never calls encode on the sending side.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from collections.abc import Mapping

import torch
from safetensors.torch import save

from rlstack.policy.state import StateLayout, TensorLayout


@dataclass(frozen=True)
class TensorState:
    tensors: Mapping[str, torch.Tensor]
    prefix: bytes = b""
    metadata: Mapping[str, str] | None = None

    def encode(self) -> bytes:
        return self.prefix + save({name: tensor.detach().cpu().contiguous()
                                   for name, tensor in self.tensors.items()},
                                  metadata=self.metadata)

    def layout(self) -> StateLayout:
        """Content hashing uses raw bytes, never Python lists or base64.

        A host read for the hash is still a cost; keep it separate from the
        NCCL timing. This is not a claim of end-to-end zero-copy publication.
        """
        tensors = []
        for name, tensor in sorted(self.tensors.items()):
            raw = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
            digest = hashlib.sha256(memoryview(raw.numpy())).hexdigest()
            tensors.append(TensorLayout(name, str(tensor.dtype).removeprefix("torch."),
                                        tuple(tensor.shape), digest))
        return StateLayout(self.prefix.hex(), self.metadata, tuple(tensors))
