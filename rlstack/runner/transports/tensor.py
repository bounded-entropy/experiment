"""Resident-to-resident tensors over a dedicated, persistent process group.

No model, adapter name, base64 codec or checkpoint serializer lives here.
Control RPCs carry layouts and rendezvous facts. GPU buffers travel on NCCL;
Gloo runs the identical tensor protocol in CPU integration tests. These groups
never initialize or join the learner's default dp/fsdp process group.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from collections.abc import Callable, Mapping

import torch
import torch.distributed as dist

from rlstack.policy.adapters.state_torch import TensorState
from rlstack.policy.state import StateLayout, TensorLayout, tensor_bundle_id
from rlstack.runner.weight_transfer import WeightTransfer


DTYPES = {str(dtype).removeprefix("torch."): dtype for dtype in (
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    torch.float16, torch.bfloat16, torch.float32, torch.float64)}


def check_backend(backend: str) -> None:
    if backend == "nccl":
        if not torch.cuda.is_available() or not dist.is_nccl_available():
            raise ValueError("NCCL weight transfer requires CUDA and an NCCL-enabled torch build")
    elif backend != "gloo":
        raise ValueError(f"unsupported tensor transport {backend!r}")


def group_for(store, rank: int, backend: str, timeout_s: float):
    timeout = timedelta(seconds=timeout_s)
    if backend == "gloo":
        return dist.ProcessGroupGloo(store, rank, 2, timeout)
    options = dist.ProcessGroupNCCL.Options()
    options._timeout = timeout
    return dist.ProcessGroupNCCL(store, rank, 2, options)


def layouts_of(manifest: Mapping) -> dict[str, StateLayout]:
    return {name: StateLayout.from_row(row) for name, row in manifest["states"].items()}


def bucket_specs(layouts: Mapping[str, StateLayout]) -> dict[str, list]:
    """Coalesce all entries by dtype; Python never iterates tensor elements."""
    buckets: dict[str, list] = {}
    for entry, layout in sorted(layouts.items()):
        for tensor in layout.tensors:
            if tensor.dtype not in DTYPES:
                raise ValueError(f"tensor transport has no dtype {tensor.dtype!r}")
            buckets.setdefault(tensor.dtype, []).append((entry, tensor))
    return dict(sorted(buckets.items()))


def content_layouts(layouts, buffers) -> dict[str, StateLayout]:
    """One host read per dtype bucket for content hashes, not per tensor."""
    digests: dict[str, dict] = {entry: {} for entry in layouts}
    for dtype, specs in bucket_specs(layouts).items():
        flat = buffers[dtype]
        raw = memoryview(flat.detach().reshape(-1).view(torch.uint8).cpu().numpy())
        offset = 0
        for entry, tensor in specs:
            size = math.prod(tensor.shape) * flat.element_size()
            digests[entry][tensor.name] = hashlib.sha256(raw[offset:offset + size]).hexdigest()
            offset += size
    return {entry: StateLayout(layout.prefix, layout.metadata, tuple(
        TensorLayout(t.name, t.dtype, t.shape, digests[entry][t.name]) for t in layout.tensors))
        for entry, layout in layouts.items()}


def frozen_buckets(states, device):
    """One immutable GPU allocation per dtype, reused for every destination."""
    layouts = {entry: StateLayout(state.prefix.hex(), state.metadata, tuple(
        TensorLayout(name, str(t.dtype).removeprefix("torch."), tuple(t.shape), "")
        for name, t in sorted(state.tensors.items()))) for entry, state in states.items()}
    tensors: dict[str, dict] = {entry: {} for entry in states}
    buffers = {}
    for dtype, specs in bucket_specs(layouts).items():
        flat = torch.cat([states[entry].tensors[t.name].detach().reshape(-1).to(device)
                          for entry, t in specs])
        buffers[dtype] = flat
        offset = 0
        for entry, t in specs:
            size = math.prod(t.shape)
            tensors[entry][t.name] = flat[offset:offset + size].reshape(t.shape)
            offset += size
    frozen = {entry: TensorState(tensors[entry], state.prefix,
              None if state.metadata is None else dict(state.metadata))
              for entry, state in states.items()}
    return frozen, buffers, layouts


class TensorSender:
    def __init__(self, device, backend: str = "nccl"):
        check_backend(backend)
        self.device, self.backend = torch.device(device), backend
        if backend == "nccl" and self.device.type != "cuda":
            raise ValueError("NCCL sender buffers must reside on CUDA")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.identity = uuid.uuid4().hex
        self.snapshots: dict[str, dict[str, TensorState]] = {}
        self.layouts: dict[str, dict[str, StateLayout]] = {}
        self.owners: dict[str, str] = {}
        self.buffers: dict[str, dict] = {}
        self.groups: OrderedDict[str, tuple] = OrderedDict()

    def capture(self, states: Mapping[str, TensorState], versions, adapter_types, owner="", code=None) -> dict:
        if len(self.snapshots) >= 2:
            raise RuntimeError("unacknowledged tensor snapshots must be released before capture")
        started = time.monotonic()
        states, buffers, shapes = frozen_buckets(states, self.device)
        frozen_at = time.monotonic()
        layouts = content_layouts(shapes, buffers)
        token = uuid.uuid4().hex
        self.snapshots[token] = dict(states)
        self.layouts[token] = layouts
        self.owners[token] = owner
        self.buffers[token] = buffers
        return {"token": token, "source": self.identity,
                "versions": dict(versions), "adapter_types": dict(adapter_types),
                "states": {name: layout.row() for name, layout in layouts.items()},
                "bundle_id": tensor_bundle_id(versions, layouts, code),
                **({"code": dict(code)} if code else {}),
                "capture_s": frozen_at - started,
                "hash_s": time.monotonic() - frozen_at}

    def send(self, token: str, endpoint: Mapping, timeout_s: float) -> dict:
        self.snapshots[token]       # a released snapshot cannot be sent again
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        key = endpoint["channel"]
        connect_started = time.monotonic()
        if key not in self.groups:
            if len(self.groups) >= 32:
                _, (_, retired) = self.groups.popitem(last=False)
                retired.abort()
            store = dist.TCPStore(endpoint["host"], endpoint["port"], 2, False,
                                  timedelta(seconds=timeout_s))
            self.groups[key] = (store, group_for(store, 1, self.backend, timeout_s))
        _, group = self.groups[key]
        self.groups.move_to_end(key)
        began = time.monotonic()
        count = 0
        try:
            for flat in self.buffers[token].values():
                if flat.numel():
                    group.send([flat], 0, 0).wait(timedelta(seconds=timeout_s))
                count += flat.numel() * flat.element_size()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except BaseException:
            self.groups.pop(key, None)
            group.abort()
            raise
        return {"bytes": count, "connect_s": began - connect_started,
                "transfer_s": time.monotonic() - began}

    def release(self, token: str) -> None:
        self.snapshots.pop(token, None)
        self.layouts.pop(token, None)
        self.owners.pop(token, None)
        self.buffers.pop(token, None)

    def release_owner(self, owner: str) -> None:
        for token, held_by in tuple(self.owners.items()):
            if held_by == owner:
                self.release(token)

    def close(self) -> None:
        for _, group in self.groups.values():
            group.shutdown()
        self.groups.clear()
        self.snapshots.clear()
        self.layouts.clear()
        self.owners.clear()
        self.buffers.clear()


class _ReceiverPeer:
    def __init__(self, host: str, timeout_s: float, backend: str):
        self.store = dist.TCPStore(host, 0, 2, True, timedelta(seconds=timeout_s),
                                   wait_for_workers=False)
        self.endpoint = {"host": host, "port": self.store.port,
                         "channel": uuid.uuid4().hex}
        self.backend, self.timeout_s = backend, timeout_s
        self.group = None
        self.pending: Future | None = None
        self.token: str | None = None
        self.cancelled = False

    def receive(self, manifest, device, install: Callable) -> dict:
        began = time.monotonic()
        if device.type == "cuda":
            torch.cuda.set_device(device)
        if self.group is None:
            self.group = group_for(self.store, 0, self.backend, self.timeout_s)
        layouts = layouts_of(manifest)
        tensors: dict[str, dict] = {entry: {} for entry in layouts}
        buffers = {}
        count = 0
        for dtype, specs in bucket_specs(layouts).items():
            sizes = [math.prod(t.shape) for _, t in specs]
            flat = torch.empty(sum(sizes), dtype=DTYPES[dtype], device=device)
            buffers[dtype] = flat
            if flat.numel():
                self.group.recv([flat], 1, 0).wait(timedelta(seconds=self.timeout_s))
            offset = 0
            for (entry, tensor), size in zip(specs, sizes):
                tensors[entry][tensor.name] = flat[offset:offset + size].reshape(tensor.shape)
                offset += size
            count += flat.numel() * flat.element_size()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        received = time.monotonic()
        states = {entry: TensorState(tensors[entry], bytes.fromhex(layout.prefix),
                                     layout.metadata) for entry, layout in layouts.items()}
        if content_layouts(layouts, buffers) != layouts:
            raise ValueError("tensor publication content mismatch")
        if tensor_bundle_id(manifest["versions"], layouts, manifest.get("code")) != manifest["bundle_id"]:
            raise ValueError("tensor publication bundle id does not match its state")
        checked = time.monotonic()
        if self.cancelled:
            raise RuntimeError("tensor publication was cancelled before installation")
        install(manifest, states)
        return {"bundle_id": manifest["bundle_id"], "bytes": count,
                "receive_s": received - began, "verify_s": checked - received,
                "install_s": time.monotonic() - checked}

    def close(self, abort: bool = False):
        self.cancelled = abort
        if self.group is not None:
            if abort:
                self.group.abort()
            else:
                self.group.shutdown()


class TensorReceiver:
    def __init__(self, install: Callable, device="cuda:0", backend="nccl"):
        check_backend(backend)
        self.install, self.device = install, torch.device(device)
        self.backend = backend
        self.peers: dict[str, _ReceiverPeer] = {}
        self.receipts: OrderedDict[str, dict] = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weight-receive")
        self.lock = threading.Lock()

    def prepare(self, manifest, transfer: WeightTransfer) -> dict:
        bucket_specs(layouts_of(manifest))  # refuse unsupported layouts before any collective
        source, token = manifest["source"], manifest["token"]
        with self.lock:
            if token in self.receipts:
                return {"complete": self.receipts[token]}
            peer = self.peers.get(source)
            if peer is not None and peer.pending is None and (
                    peer.timeout_s != transfer.timeout_s or
                    peer.endpoint["host"] != transfer.rendezvous_host):
                peer.close()
                self.peers.pop(source)
                peer = None
            if peer is None:
                if len(self.peers) >= 32:
                    idle = next((key for key, held in self.peers.items()
                                 if held.pending is None), None)
                    if idle is None:
                        raise RuntimeError("tensor peer limit reached with all peers in flight")
                    self.peers.pop(idle).close()
                peer = self.peers[source] = _ReceiverPeer(
                    transfer.rendezvous_host, transfer.timeout_s, self.backend)
            if peer.pending is not None:
                if peer.token != token:
                    raise RuntimeError("a source has an unacknowledged tensor transfer")
                return dict(peer.endpoint)
            peer.token = token
            peer.pending = self.executor.submit(peer.receive, manifest, self.device, self.install)
            return dict(peer.endpoint)

    def finish(self, source: str, token: str) -> dict:
        with self.lock:
            if token in self.receipts:
                return self.receipts[token]
            peer = self.peers[source]
            if peer.token != token or peer.pending is None:
                raise ValueError("no matching tensor transfer to acknowledge")
            pending = peer.pending
        try:
            receipt = pending.result(timeout=peer.timeout_s)
        except BaseException:
            peer.close(abort=True)
            with self.lock:
                self.peers.pop(source, None)
            raise
        with self.lock:
            self.receipts[token] = receipt
            if len(self.receipts) > 64:
                self.receipts.popitem(last=False)
            peer.pending, peer.token = None, None
        return receipt

    def close(self):
        for peer in self.peers.values():
            peer.close(abort=peer.pending is not None and not peer.pending.done())
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.peers.clear()

    def abort(self, source: str, token: str):
        with self.lock:
            peer = self.peers.get(source)
            if peer is None or peer.token != token:
                return
            self.peers.pop(source)
        peer.close(abort=True)
