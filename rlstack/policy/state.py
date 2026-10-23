"""Adapter-neutral layout of a tensor publication; no GPU dependency.

Tensor bundles have their own content-id namespace. Legacy byte bundles and
checkpoint payloads keep their existing codecs. The same manifest can be
derived from resident tensors or a checkpoint, so restoration proves the id.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from typing import Protocol, Any


class AdapterState(Protocol):
    """A compute-side adapter state exposes tensors without choosing carriage."""

    def export_state(self, version: int) -> Any: ...

    def bind_version(self, version: int) -> None:
        """Bind checkpoint-relative publication counters; static states need none."""


@dataclass(frozen=True)
class TensorLayout:
    name: str
    dtype: str
    shape: tuple[int, ...]
    digest: str


@dataclass(frozen=True)
class StateLayout:
    prefix: str                         # opaque adapter header, hex on control RPC
    metadata: Mapping[str, str] | None
    tensors: tuple[TensorLayout, ...]

    def row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping) -> StateLayout:
        return cls(row["prefix"], row["metadata"], tuple(
            TensorLayout(t["name"], t["dtype"], tuple(t["shape"]), t["digest"])
            for t in row["tensors"]))


DTYPES = {"BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16",
          "I32": "int32", "I64": "int64", "F16": "float16",
          "BF16": "bfloat16", "F32": "float32", "F64": "float64"}


def payload_layout(payload: bytes) -> StateLayout:
    """Rebuild a tensor manifest from either existing safetensors container.

    The optional adapter-owned prefix is length-prefixed big-endian JSON;
    safetensors itself uses a little-endian header length. Neither requires
    importing torch, decoding tensors, or knowing an adapter's registered name.
    """
    offset = 0
    size = int.from_bytes(payload[:8], "little")
    if size > len(payload) - 8:
        size = int.from_bytes(payload[:8], "big")
        offset = 8 + size
    size = int.from_bytes(payload[offset:offset + 8], "little")
    end = offset + 8 + size
    header = json.loads(payload[offset + 8:end])
    body = memoryview(payload)[end:]
    tensors = []
    for name, spec in sorted(header.items()):
        if name == "__metadata__":
            continue
        first, last = spec["data_offsets"]
        tensors.append(TensorLayout(name, DTYPES[spec["dtype"]],
                                    tuple(spec["shape"]),
                                    hashlib.sha256(body[first:last]).hexdigest()))
    return StateLayout(payload[:offset].hex(), header.get("__metadata__"),
                       tuple(tensors))


def tensor_bundle_id(versions: Mapping[str, int],
                     layouts: Mapping[str, StateLayout]) -> str:
    row = {"versions": dict(versions),
           "states": {name: layout.row() for name, layout in sorted(layouts.items())}}
    data = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    return "bundle:tensor:" + hashlib.sha256(data).hexdigest()[:24]
