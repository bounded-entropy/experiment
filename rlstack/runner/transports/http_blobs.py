"""Private, finite-lived HTTP transfer files; never scientific Store artifacts."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
import time
from typing import BinaryIO

CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class HttpLimits:
    """One runner's finite transfer budget; raising it requires enough RAM/disk."""
    inline_bytes: int = 1024 * 1024
    max_blob_bytes: int = 2 * 1024**3
    spool_bytes: int = 8 * 1024**3
    max_blobs: int = 128
    max_connections: int = 64
    ttl_s: float = 300
    io_timeout_s: float = 30

    @classmethod
    def from_environment(cls) -> "HttpLimits":
        """The gateway, clients and workers share explicit operator byte budgets."""
        return cls(max_blob_bytes=int(os.environ.get("RLSTACK_HTTP_MAX_BLOB_BYTES", 2 * 1024**3)),
                   spool_bytes=int(os.environ.get("RLSTACK_HTTP_SPOOL_BYTES", 8 * 1024**3)))

    def __post_init__(self) -> None:
        import math
        if not 0 < self.inline_bytes <= self.max_blob_bytes <= self.spool_bytes:
            raise ValueError("Expected inline <= one blob <= spool byte limits")
        if self.max_blobs < 1 or self.max_connections < 1:
            raise ValueError("HTTP connection and blob limits must be positive")
        if any(not math.isfinite(n) or n <= 0 for n in (self.ttl_s, self.io_timeout_s)):
            raise ValueError("HTTP expiry and I/O timeout must be positive and finite")


@dataclass(frozen=True)
class BlobRef:
    id: str
    size: int
    sha256: str

    @classmethod
    def decode(cls, value: dict, maximum: int) -> "BlobRef":
        """References name opaque local ids, never caller-selected paths or URLs."""
        if not isinstance(value, dict) or set(value) != {"id", "size", "sha256"}:
            raise ValueError("Expected blob id, size and SHA-256")
        ref = cls(**value)
        if (not isinstance(ref.id, str) or re.fullmatch(r"[a-f0-9]{32}", ref.id) is None
                or type(ref.size) is not int or not 0 < ref.size <= maximum
                or not isinstance(ref.sha256, str)
                or re.fullmatch(r"[a-f0-9]{64}", ref.sha256) is None):
            raise ValueError("Invalid blob reference")
        return ref

    def row(self) -> dict:
        return {"id": self.id, "size": self.size, "sha256": self.sha256}


class JsonFile:
    """Encode incrementally onto disk, with an explicit total byte limit."""
    def __init__(self, maximum: int, *, output: BinaryIO | None = None,
                 account: Callable[[int], None] | None = None) -> None:
        self.file = output or tempfile.TemporaryFile(mode="w+b")
        self.maximum, self.account = maximum, account
        self.size = 0
        self.digest = hashlib.sha256()

    def write(self, text: str) -> None:
        self.append(text.encode("utf-8"))

    def append(self, data: bytes) -> None:
        """Charge a bounded byte block before it reaches the spool."""
        if self.size + len(data) > self.maximum:
            raise ValueError("HTTP payload exceeds the configured blob limit before dispatch/transfer")
        if self.account is not None:
            self.account(len(data))
        self.file.write(data)
        self.size += len(data)
        self.digest.update(data)

    def encode(self, value: dict) -> "JsonFile":
        json.dump(value, self, separators=(",", ":"))
        self.file.flush()
        self.file.seek(0)
        return self

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "JsonFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_CLIENT_SPOOL_LOCK = threading.Lock()
_CLIENT_SPOOL_BYTES = 0


class ClientFile(JsonFile):
    """Concurrent transports in one process share a finite outgoing/incoming spool."""
    def __init__(self, limits: HttpLimits) -> None:
        self.charged = 0
        def account(count: int) -> None:
            global _CLIENT_SPOOL_BYTES
            with _CLIENT_SPOOL_LOCK:
                if _CLIENT_SPOOL_BYTES + count > limits.spool_bytes:
                    raise ValueError("HTTP client transfer spool quota exhausted")
                _CLIENT_SPOOL_BYTES += count
                self.charged += count
        super().__init__(limits.max_blob_bytes, account=account)

    def close(self) -> None:
        global _CLIENT_SPOOL_BYTES
        try:
            super().close()
        finally:
            with _CLIENT_SPOOL_LOCK:
                _CLIENT_SPOOL_BYTES -= self.charged
                self.charged = 0


@dataclass
class Blob:
    ref: BlobRef
    kind: str
    expires: float
    ready: bool = False
    active: int = 0
    consumed: bool = False
    deleting: bool = False


class BlobStore:
    """Bound all uploaded and produced bytes, including abandoned reservations."""
    def __init__(self, limits: HttpLimits) -> None:
        self.limits = limits
        self.directory = tempfile.TemporaryDirectory(prefix="rlstack-http-")
        self.root = Path(self.directory.name)
        self.lock = threading.RLock()
        self.blobs: dict[str, Blob] = {}
        self.used = 0
        self.closed = False

    def _path(self, blob_id: str) -> Path:
        # Only ids allocated by reserve or validated by BlobRef reach this path.
        return self.root / blob_id

    def reserve(self, size: int, digest: str, kind: str = "request") -> BlobRef:
        with self.lock:
            self.collect()
            if self.closed:
                raise RuntimeError("HTTP transfer store closed")
            if len(self.blobs) >= self.limits.max_blobs or self.used + size > self.limits.spool_bytes:
                raise ValueError("HTTP transfer spool quota exhausted")
            ref = BlobRef(secrets.token_hex(16), size, digest)
            self.blobs[ref.id] = Blob(ref, kind, time.monotonic() + self.limits.ttl_s)
            self.used += size
            return ref

    def _blob(self, ref: BlobRef) -> Blob:
        self.collect()
        blob = self.blobs.get(ref.id)
        if blob is None or blob.ref != ref or blob.deleting:
            raise ValueError("Unknown, expired or mismatched blob")
        return blob

    def status(self, ref: BlobRef) -> bool:
        with self.lock:
            return self._blob(ref).ready

    @contextmanager
    def upload(self, ref: BlobRef) -> Iterator[BinaryIO]:
        """Only one complete, checksum-verified upload can become a request."""
        with self.lock:
            blob = self._blob(ref)
            if blob.kind != "request" or blob.ready or blob.active:
                raise ValueError("Blob is immutable or already being uploaded")
            blob.active += 1
        try:
            with self._path(ref.id).open("xb") as output:
                yield output
            with self.lock:
                blob.ready = True
                blob.expires = time.monotonic() + self.limits.ttl_s
        except BaseException:
            self.discard(ref.id)
            raise
        finally:
            with self.lock:
                blob.active -= 1
                self.collect()

    def produce(self, value: dict) -> BlobRef:
        """Charge serialized response bytes before writing, including concurrent results."""
        ref = self.reserve(0, "", "result")
        with self.lock:
            blob = self.blobs[ref.id]
            blob.active = 1
        def account(count: int) -> None:
            with self.lock:
                if self.used + count > self.limits.spool_bytes:
                    raise ValueError("HTTP transfer spool quota exhausted after execution")
                self.used += count
                blob.ref = BlobRef(ref.id, blob.ref.size + count, "")
        try:
            with self._path(ref.id).open("xb") as output:
                writer = JsonFile(self.limits.max_blob_bytes, output=output, account=account)
                writer.encode(value)
                complete = BlobRef(ref.id, writer.size, writer.digest.hexdigest())
            with self.lock:
                blob.ref, blob.ready = complete, True
                blob.expires = time.monotonic() + self.limits.ttl_s
            return complete
        except BaseException:
            self.discard(ref.id)
            raise
        finally:
            with self.lock:
                blob.active -= 1
                self.collect()

    @contextmanager
    def read(self, ref: BlobRef, *, consume: bool = False) -> Iterator[BinaryIO]:
        """A request reference is one-use; result downloads never dispatch work."""
        with self.lock:
            blob = self._blob(ref)
            expected = "request" if consume else "result"
            if not blob.ready or blob.kind != expected or blob.consumed:
                raise ValueError("Blob is incomplete, consumed or the wrong kind")
            blob.active += 1
            if consume:
                blob.consumed = True
        try:
            with self._path(ref.id).open("rb") as source:
                yield source
        finally:
            with self.lock:
                blob.active -= 1
                if consume:
                    blob.deleting = True
                self.collect()

    def discard(self, blob_id: str) -> None:
        with self.lock:
            blob = self.blobs.get(blob_id)
            if blob is not None:
                blob.deleting = True
            self.collect()

    def collect(self) -> None:
        """Expired idle blobs release both disk and reservation quota."""
        with self.lock:
            now = time.monotonic()
            for key, blob in list(self.blobs.items()):
                if not blob.active and (self.closed or blob.deleting or blob.expires <= now):
                    self._path(key).unlink(missing_ok=True)
                    self.used -= blob.ref.size
                    del self.blobs[key]
            if self.closed and not self.blobs:
                self.directory.cleanup()

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.collect()
            if not self.blobs:
                self.directory.cleanup()
