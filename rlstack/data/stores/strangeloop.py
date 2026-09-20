"""One Strange Loop scratch tree, read through HTTP and optionally written mounted.

Scratch mounts do not refresh when another client writes. Mutable reads and
stats use the authoritative API. Hashed payloads may use a verified local
cache or already-visible mount, with HTTP fallback and no reload (ADR 0020). A mounted
writer publishes each mutation with checked `sync /scratch`; an API writer
replaces a whole journal under its existing single-owner custody (ADR 0015).
Neither the lock here nor successful readback supplies cross-process ownership
or the provider's missing conditional-write/operation-status contract.
"""

from __future__ import annotations

import io
import fcntl
import http.client
import json
import math
import os
import queue
import re
import subprocess
import threading
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

from rlstack.data.stores.base import Store, StoreAddress, StoreError, _canonical, check_subdir
from rlstack.data.stores.local import LocalStore

DEFAULT_API_BASE = "https://api.strangeloopresearch.com/api/v1"
CHUNK_BYTES = 1024 * 1024
# A publication's readback is retried this many times, this far apart, before
# a transient read failure is allowed to stop the owner (see _readback).
READBACK_ATTEMPTS = 4
READBACK_PAUSE_S = 2.0
WRITE_ATTEMPTS = 3                   # tries a write gets when the API REJECTS it outright
WRITE_PAUSE_S = 2.0


class ScratchRejected(StoreError):
    """The API answered a write with a server error: the request was seen
    and refused, so nothing is in flight — unlike a lost acknowledgement,
    which may still land. The one write failure that is safe to retry."""


@dataclass(frozen=True)
class ScratchCredentials:
    """Runtime authorization; never serialized into a StoreAddress."""

    token: str = field(repr=False)
    api_base: str
    profile: str


def credential_file(credentials: ScratchCredentials) -> bytes:
    """The bytes of a token file: what a pod's processes resolve their
    credentials from (SL_API_TOKEN_FILE) and what a reauth rewrites."""
    return json.dumps({"token": credentials.token, "api_base": credentials.api_base}).encode()


def _token_file() -> dict:
    """The token file SL_API_TOKEN_FILE names, or nothing when it is unset
    or unreadable — an absent file falls through to the profile, and the
    resolver's own error names every source when none has a token."""
    path = os.environ.get("SL_API_TOKEN_FILE")
    if not path:
        return {}
    try:
        row = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return row if isinstance(row, dict) else {}


def resolve_scratch_credentials(*, token: str | None = None,
                                api_base: str | None = None,
                                profile: str | None = None) -> ScratchCredentials:
    """Follow the CLI's explicit, environment, profile, default precedence —
    with the TOKEN FILE between the environment and the profile: a pod has
    no profile, and a token baked into its environment cannot be renewed
    (2026-09-17: a browser session's token expired ten hours in, the pods'
    copies with it, and every run died at its next store call). A file the
    desk can rewrite, re-read on every 401, can."""
    name = profile or os.environ.get("SL_PROFILE") or "default"
    config_dir = os.environ.get("STRANGELOOP_CONFIG_DIR")
    root = (Path(config_dir).expanduser() if config_dir else
            Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
            / "strangeloop")
    try:
        with (root / "config.toml").open("rb") as source:
            profiles = tomllib.load(source).get("profile", {})
    except FileNotFoundError:
        profiles = {}
    stored = profiles.get(name)
    if stored is None:  # The CLI permits dots in unquoted profile names.
        stored = profiles
        for part in name.split("."):
            stored = stored.get(part, {})
    filed = _token_file()
    resolved = token or os.environ.get("SL_API_TOKEN") or filed.get("token") or stored.get("token")
    if not resolved:
        raise StoreError("scratch needs SL_API_TOKEN, a readable SL_API_TOKEN_FILE, "
                         "or a logged-in Strange Loop profile")
    base = (api_base or os.environ.get("SL_API_BASE") or filed.get("api_base")
            or stored.get("api_base"))
    return ScratchCredentials(resolved, _api_base(base or DEFAULT_API_BASE), name)


def _api_base(base: str) -> str:
    """A public API origin/path, never credentials disguised as a locator."""
    parsed = urllib.parse.urlsplit(base.rstrip("/"))
    local = parsed.hostname in ("localhost", "127.0.0.1", "::1")
    if (parsed.scheme != "https" and not (parsed.scheme == "http" and local)
            or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment):
        raise StoreError("scratch API base must be HTTPS (HTTP is allowed for localhost tests)")
    clean = urllib.parse.urlunsplit(parsed)
    return clean if clean.endswith("/api/v1") else clean + "/api/v1"


def _relative(path: str, *, empty: bool = False) -> str:
    """A scratch key stays beneath its declared prefix."""
    if empty and not path.strip("/"):
        return ""
    if path.startswith("/") or "\\" in path or any(
            part in ("", ".", "..") for part in path.rstrip("/").split("/")):
        raise StoreError(f"not a scratch-relative key: {path!r}")
    return path.rstrip("/")


@dataclass(frozen=True)
class ScratchEntry:
    """One API file or directory, with a store-relative path."""

    path: str
    type: str
    size: int


def scratch_path(key: str) -> str:
    """Encode logical Store keys within Scratch's restricted filename alphabet.

    Ordinary keys keep their spelling. Escape @ itself so a literal @3a and
    a colon cannot name the same file; slash remains the directory separator.
    """
    key = _relative(key, empty=True)
    safe = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-+/"
    return "".join(chr(byte) if byte in safe else f"@{byte:02x}" for byte in key.encode("utf-8"))


def scratch_key(path: str) -> str:
    """Decode only canonical filenames produced by scratch_path."""
    try:
        key = urllib.parse.unquote_to_bytes(path.replace("@", "%")).decode("utf-8")
    except UnicodeDecodeError:
        raise StoreError("scratch entry has an invalid encoded filename") from None
    if scratch_path(key) != path:
        raise StoreError("scratch entry has a noncanonical encoded filename")
    return key


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise StoreError("scratch API redirected; configure its final API base explicitly")


UPLOAD_FLOOR_BYTES_PER_S = 100_000   # the slowest transfer the deadlines plan for, both ways
READ_ATTEMPTS = 5                    # a read's tries against an API that is UP, each with the full timeout
READ_BACKOFF_S = 1.0                 # the pause before the second try; doubles after
OUTAGE_CEILING_S = 8 * 3600.0        # how long one request waits out an API that is down, or a refused token
OUTAGE_PAUSE_MAX_S = 60.0            # the longest pause between tries while waiting one out
OUTAGE_NOTE_EVERY_S = 600.0          # how often a waiting client says so


class ScratchClient:
    """The documented scratch file routes, scoped to one verified account."""

    def __init__(self, volume: str, prefix: str = "rlstack", *,
                 token: str | None = None, api_base: str | None = None,
                 profile: str | None = None, timeout: float = 30.0) -> None:
        if not re.fullmatch(r"sl-scratch-[A-Za-z0-9-]+", volume):
            raise StoreError("scratch volume must name sl-scratch-<user-id>")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("scratch timeout must be finite and positive")
        self.volume = volume
        self.prefix = _relative(prefix)
        # where the credentials came from, so a 401 can ask there again
        self._source = {"token": token, "api_base": api_base, "profile": profile}
        self.credentials = resolve_scratch_credentials(**self._source)
        self.timeout = timeout
        self.outage_ceiling_s = OUTAGE_CEILING_S
        self._clock = self._sleep = None      # a test's clock; the wall clock otherwise
        self._opener = urllib.request.build_opener(_NoRedirect())
        identity = self._json("GET", "/auth/me")
        if volume != "sl-scratch-" + str(identity["user_id"]):
            raise StoreError("scratch account differs from the saved store's account")
        clusters = identity["clusters"]
        if clusters is not None and "modal" not in clusters:
            raise StoreError("this account does not provide Modal scratch storage")

    @property
    def locator(self) -> str:
        locator = f"strangeloop://{self.volume}/{urllib.parse.quote(self.prefix, safe='/')}"
        if self.credentials.api_base != DEFAULT_API_BASE:
            locator += "?" + urllib.parse.urlencode({"api_base": self.credentials.api_base})
        return locator

    @classmethod
    def from_locator(cls, locator: str) -> "ScratchClient":
        parsed = urllib.parse.urlsplit(locator)
        query = urllib.parse.parse_qs(parsed.query, strict_parsing=True)
        if (parsed.scheme != "strangeloop" or parsed.username or parsed.password
                or parsed.port or parsed.fragment or set(query) - {"api_base"}
                or any(len(values) != 1 for values in query.values())):
            raise StoreError("invalid scratch store locator")
        return cls(parsed.netloc, urllib.parse.unquote(parsed.path.lstrip("/")),
                   api_base=query.get("api_base", [DEFAULT_API_BASE])[0])

    def _path(self, key: str) -> str:
        relative = scratch_path(key)
        return self.prefix + ("/" + relative if relative else "")

    def refresh_credentials(self) -> bool:
        """Ask again where the credentials came from — the profile a login
        rewrites, the token file a reauth rewrites, the environment — and
        say whether the token changed. Credentials handed to the constructor
        outright have nowhere to be renewed from. The API base is kept: it
        is part of this client's locator."""
        if self._source["token"] is not None:
            return False
        try:
            fresh = resolve_scratch_credentials(**self._source)
        except StoreError:
            return False
        changed = fresh.token != self.credentials.token
        self.credentials = ScratchCredentials(fresh.token, self.credentials.api_base, fresh.profile)
        return changed

    def _authorize(self, request) -> None:
        request.remove_header("Authorization")
        request.add_header("Authorization", "Bearer " + self.credentials.token)

    def _api_up(self) -> bool:
        """Does the API answer at all? One GET of `/auth/me` within the
        client's timeout: any status is an answer (a 401 says the token is
        the problem, not the API); a 5xx, a dropped connection or a stall
        is not."""
        request = urllib.request.Request(self.credentials.api_base + "/auth/me", method="GET",
                                         headers={"User-Agent": "rlstack-scratch"})
        self._authorize(request)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                response.read()
            return True
        except urllib.error.HTTPError as error:
            error.close()
            return error.code < 500
        except (OSError, http.client.HTTPException):
            return False

    def _note(self, message: str) -> None:
        print(message, flush=True)

    def _now(self) -> float:
        return (self._clock or time.monotonic)()

    def _pause(self, seconds: float) -> None:
        (self._sleep or time.sleep)(seconds)

    def _open(self, request):
        """One request against an API that may be up, down, or refusing the token.

        THE SIZE RULE: a request's deadline is the client's timeout plus the
        bytes it uploads at UPLOAD_FLOOR_BYTES_PER_S — a 14 MB plan took
        longer than a flat 30 s over the scratch API and stopped an owner
        for an "uncertain" write that was merely slow (measured 2026-09-17).

        THE OUTAGE RULE (2026-09-17, after twenty minutes of the API
        answering 500 and timing out ended four 7B runs and the desk's own
        owner): what the API cannot answer for now is WAITED OUT, never
        given up on. A 5xx or a dropped connection is judged by one cheap
        question — does `/auth/me` answer? — and an API that is DOWN is
        retried, pausing up to OUTAGE_PAUSE_MAX_S between tries, for
        `outage_ceiling_s`; a run paused in a store call costs its lease
        nothing more than a dead one would, and resumes the moment the API
        does. A 401 re-resolves the credentials from where they came: a
        renewed token retries at once, an unchanged one waits the same way,
        since only a person can renew it. An API that is UP keeps the older
        rules: a read owns READ_ATTEMPTS tries, each with the full timeout
        (a 5xx or a stall against a live API is the API's own answer — a
        path under a missing parent is a 500 — and one 30 s stall must not
        reach a runner role); a write is tried once — a lost acknowledgement
        is never replayed (ADR 0015) — while a REFUSED write (a 5xx, a 401)
        has nothing in flight and is retried to the ceiling whether or not
        the API answers its health question, since a live API refuses
        journal appends with 500 on its bad minutes."""
        body = request.data or b""
        budget = self.timeout + len(body) / UPLOAD_FLOOR_BYTES_PER_S
        method = request.get_method()
        reading = method == "GET"
        tries = 0                   # tries answered by an API that was up
        waited_since = None         # when this request first found the API down
        noted_at = None
        pause = READ_BACKOFF_S
        while True:
            try:
                response = self._opener.open(request, timeout=budget)
            except urllib.error.HTTPError as error:
                if error.code == 401:
                    error.close()
                    if self.refresh_credentials():
                        self._authorize(request)
                        continue                    # renewed: not an outage
                    reason = "credentials refused (HTTP 401): a login or reauth is needed"
                elif error.code == 429:
                    # RATE-LIMITED IS NOT REFUSED (2026-09-19): the API answers
                    # 429 when many hosts talk at once, nothing of the request
                    # was taken, and the cure is to come back slower — waited
                    # out like an outage, for reads and writes alike. Twelve
                    # writers from one laptop met it within seconds.
                    error.close()
                    reason = "HTTP 429: the API is rate-limiting this client"
                elif 500 <= error.code < 600:
                    if reading and self._api_up():
                        # a live API's 5xx on a read is its own answer (a
                        # path under a missing parent is a 500): bounded tries
                        tries += 1
                        if tries >= READ_ATTEMPTS:
                            raise
                        error.close()
                        self._pause(READ_BACKOFF_S * (2 ** (tries - 1)))
                        continue
                    error.close()
                    # a REFUSED write is never in flight, and the API refuses
                    # journal appends with 500 on its bad minutes (three such
                    # minutes stopped the desk's owner on 2026-09-17, twice):
                    # waited out like an outage, whatever the health question says
                    reason = (f"HTTP {error.code} while the API is down" if reading
                              else f"HTTP {error.code} refused the write")
                else:
                    raise
            except (OSError, http.client.HTTPException) as error:
                failure = f"scratch {method} connection failed: {type(error).__name__}"
                if not reading:
                    raise StoreError(failure) from error
                if self._api_up():
                    tries += 1
                    if tries >= READ_ATTEMPTS:
                        raise StoreError(failure) from error
                    self._pause(READ_BACKOFF_S * (2 ** (tries - 1)))
                    continue
                reason = f"{type(error).__name__} while the API is down"
            else:
                if waited_since is not None:
                    self._note(f"scratch {method} recovered after "
                               f"{self._now() - waited_since:.0f} s: {reason}")
                return response
            now = self._now()
            waited_since = now if waited_since is None else waited_since
            if now - waited_since >= self.outage_ceiling_s:
                raise StoreError(f"scratch {method} gave up after "
                                 f"{self.outage_ceiling_s:.0f} s: {reason}")
            if noted_at is None or now - noted_at >= OUTAGE_NOTE_EVERY_S:
                self._note(f"scratch {method} waiting: {reason}; retrying for up to "
                           f"{self.outage_ceiling_s:.0f} s")
                noted_at = now
            self._pause(pause)
            pause = min(pause * 2, OUTAGE_PAUSE_MAX_S)

    def _request(self, method: str, route: str, *, key: str | None = None,
                 data: bytes | None = None, recursive: bool = False):
        params = {"path": self._path(key)} if key is not None else {}
        if route == "/scratch/files" and method == "GET":
            params["recursive"] = "true" if recursive else "false"
        url = self.credentials.api_base + route
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, method=method, data=data, headers={
            "User-Agent": "rlstack-scratch",
            "Content-Type": "application/octet-stream",
        })
        self._authorize(request)
        try:
            return self._open(request)
        except urllib.error.HTTPError as error:
            status = error.code
            error.close()
            if status == 404 and method == "GET" and key is not None:
                raise FileNotFoundError(key) from None
            if status == 500 and method == "GET" and route == "/scratch/files" and key is not None:
                # Listing an absent directory currently returns 500. Only
                # its metadata's authoritative absence makes that an empty list.
                try:
                    self.stat(key)
                except FileNotFoundError:
                    raise FileNotFoundError(key) from None
            if (status == 500 and method == "GET" and key
                    and route in ("/scratch/files/raw", "/scratch/files/stat")):
                # The live API returns 500 for a path under a missing parent.
                # A parent's authoritative 404 proves this child absent;
                # an existing or unreadable parent leaves the 500 an error.
                try:
                    self.stat(key.rpartition("/")[0])
                except FileNotFoundError:
                    raise FileNotFoundError(key) from None
            if status >= 500 and method != "GET":
                raise ScratchRejected(
                    f"scratch {method} {route} rejected (HTTP {status}, key={key!r})") from None
            raise StoreError(f"scratch {method} {route} failed (HTTP {status}, key={key!r})") from None

    def _copy(self, response, target: BinaryIO) -> None:
        """Stream the body. THE SIZE RULE again, on the read side: the
        deadline is the timeout plus the declared length at the floor rate
        (a 12 MB task set exceeded a flat 30 s over the scratch API and
        killed a generator, 2026-09-17), and a body of unknown length may
        take as long as its chunks keep arriving within one timeout."""
        expected = response.headers.get("Content-Length")
        declared = int(expected) if expected is not None else 0
        deadline = time.monotonic() + self.timeout + declared / UPLOAD_FLOOR_BYTES_PER_S
        received = 0
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("scratch response exceeded its read deadline")
            chunk = response.read1(CHUNK_BYTES)
            if chunk and expected is None:
                deadline = max(deadline, time.monotonic() + self.timeout)
            if not chunk:
                if expected is not None and received != int(expected):
                    raise StoreError("scratch response ended before its declared byte count")
                return
            target.write(chunk)
            received += len(chunk)

    def _json(self, method: str, route: str, *, key: str | None = None,
              data: bytes | None = None, recursive: bool = False) -> dict:
        buffer = io.BytesIO()
        with self._request(method, route, key=key, data=data,
                           recursive=recursive) as response:
            self._copy(response, buffer)
        return json.loads(buffer.getvalue())

    def read(self, key: str) -> bytes:
        buffer = io.BytesIO()
        with self._request("GET", "/scratch/files/raw", key=key) as response:
            self._copy(response, buffer)
        return buffer.getvalue()

    def download(self, key: str, destination: str | Path) -> None:
        """Stream to a private file; the destination appears only when complete."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with self._request("GET", "/scratch/files/raw", key=key) as response:
                with temporary.open("wb") as target:
                    self._copy(response, target)
                    target.flush()
                    os.fsync(target.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def write(self, key: str, data: bytes) -> None:
        _relative(key)
        self._json("PUT", "/scratch/files/raw", key=key, data=data)

    def delete(self, key: str) -> None:
        _relative(key)
        self._json("DELETE", "/scratch/files", key=key)

    def _entry(self, row: dict) -> ScratchEntry:
        path = str(row["path"]).lstrip("/").rstrip("/")
        if path != self.prefix and not path.startswith(self.prefix + "/"):
            raise StoreError("scratch API returned an entry outside this store")
        kind = row["type"]
        if kind not in ("file", "dir"):
            raise StoreError(f"unknown scratch entry type: {kind!r}")
        return ScratchEntry(scratch_key(path.removeprefix(self.prefix).lstrip("/")), kind,
                            int(row["size"] or 0) if kind == "file" else 0)

    def stat(self, key: str) -> ScratchEntry:
        return self._entry(self._json("GET", "/scratch/files/stat", key=key))

    def list(self, prefix: str, *, recursive: bool = False) -> list[ScratchEntry]:
        try:
            rows = self._json("GET", "/scratch/files", key=prefix,
                              recursive=recursive)["entries"]
        except FileNotFoundError:
            return []
        return sorted((self._entry(row) for row in rows), key=lambda entry: entry.path)


@dataclass(frozen=True)
class HashedReads:
    """Pod-local tuning, inherited by residents; never part of run identity."""

    blob_cache_dir: Path | None = None
    blob_cache_bytes: int = 0
    mount_reads: bool = False

    def __post_init__(self) -> None:
        if type(self.blob_cache_bytes) is not int or type(self.mount_reads) is not bool:
            raise ValueError("blob cache bytes must be an integer and mount_reads a boolean")
        if self.blob_cache_bytes < 0 or bool(self.blob_cache_dir) != (self.blob_cache_bytes > 0):
            raise ValueError("blob cache needs both a directory and a positive byte bound, or neither")
        if self.blob_cache_dir is not None:
            path = self.blob_cache_dir.resolve()
            if not self.blob_cache_dir.is_absolute() or any(
                    path.is_relative_to(root.resolve()) for root in (Path("/scratch"), Path("/persist"))):
                raise ValueError("blob cache must be an absolute path on local disk, outside shared mounts")

    @classmethod
    def from_environment(cls) -> HashedReads:
        directory = os.environ.get("RLSTACK_BLOB_CACHE_DIR", "")
        mounted = os.environ.get("RLSTACK_MOUNT_READS", "0")
        if mounted not in ("0", "1"):
            raise ValueError("RLSTACK_MOUNT_READS must be 0 or 1")
        return cls(Path(directory) if directory else None,
                   int(os.environ.get("RLSTACK_BLOB_CACHE_BYTES", "0")), mounted == "1")

    def environment(self) -> dict[str, str]:
        return {"RLSTACK_BLOB_CACHE_DIR": str(self.blob_cache_dir or ""),
                "RLSTACK_BLOB_CACHE_BYTES": str(self.blob_cache_bytes),
                "RLSTACK_MOUNT_READS": "1" if self.mount_reads else "0"}


class BlobCache:
    """Verified content retained under a byte bound, shared across processes.

    Locks are never unlinked: replacing a lock's inode would admit two owners.
    256 stripes bound their count; a collision only serializes unrelated misses.
    Maintenance never waits on a download. Its lock also protects temporary
    writes, so startup cleanup cannot unlink another process's active insertion.
    """

    def __init__(self, root: Path, capacity: int, note: Callable[[str], None]) -> None:
        self.root, self.capacity, self.note = root, capacity, note
        (root / ".locks").mkdir(parents=True, exist_ok=True)
        with self._lock("maintenance"):
            for temporary in root.glob("*/*.tmp"):
                temporary.unlink(missing_ok=True)
            self._evict(0)

    @contextmanager
    def _lock(self, name: str) -> Iterator[None]:
        with (self.root / ".locks" / name).open("a+b") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    @contextmanager
    def claim(self, sha256: str) -> Iterator[None]:
        """One fetch per stripe; an unavailable cache never prevents a read."""
        with ExitStack() as stack:
            try:
                stack.enter_context(self._lock(sha256[:2]))
            except OSError as error:
                self.note(f"scratch cache lock unavailable: {error}")
            yield

    def path(self, sha256: str) -> Path:
        return self.root / sha256[:2] / sha256

    def get(self, sha256: str) -> bytes | None:
        """An open file survives eviction; touching its descriptor cannot recreate it."""
        try:
            with self.path(sha256).open("rb") as source:
                data = source.read()
                os.utime(source.fileno(), None)
            return data
        except FileNotFoundError:
            return None
        except OSError as error:
            self.note(f"scratch cache read unavailable: {error}")
            return None

    def discard(self, sha256: str) -> None:
        try:
            with self._lock("maintenance"):
                self.path(sha256).unlink(missing_ok=True)
        except OSError as error:
            self.note(f"scratch cache discard unavailable: {error}")

    def _evict(self, incoming: int) -> None:
        """Called under maintenance; explicit access times work with noatime mounts."""
        entries = [(path.stat(), path) for path in self.root.glob("*/*")
                   if re.fullmatch(r"[0-9a-f]{64}", path.name)]
        size = sum(stat.st_size for stat, _ in entries)
        for stat, path in sorted(entries, key=lambda entry: (entry[0].st_atime_ns, str(entry[1]))):
            if size + incoming <= self.capacity:
                break
            path.unlink()
            size -= stat.st_size

    def put(self, sha256: str, data: bytes) -> None:
        """Only complete verified bytes become visible; oversized objects bypass retention."""
        if len(data) > self.capacity:
            return
        try:
            with self._lock("maintenance"):
                path = self.path(sha256)
                path.parent.mkdir(parents=True, exist_ok=True)
                self._evict(len(data))
                temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    temporary.write_bytes(data)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        except OSError as error:
            self.note(f"scratch cache write unavailable: {error}")


class StrangeLoopStore(Store):
    """The authoritative HTTP store; a failed mutation stops this owner."""

    def __init__(self, scratch: ScratchClient, *, read_only: bool = False,
                 hashed_reads: HashedReads | None = None,
                 mount_root: str | Path | None = None) -> None:
        self.scratch = scratch
        self.hashed_reads = hashed_reads if hashed_reads is not None else HashedReads.from_environment()
        self._mount_root = Path(mount_root) if mount_root else None
        self._blob_cache: BlobCache | None = None
        settings = self.hashed_reads
        if settings.blob_cache_dir is not None:
            if self._mount_root is not None and settings.blob_cache_dir.resolve().is_relative_to(
                    self._mount_root.resolve()):
                raise ValueError("blob cache must be outside the mounted store")
            try:
                self._blob_cache = BlobCache(settings.blob_cache_dir, settings.blob_cache_bytes,
                                             scratch._note)
            except OSError as error:
                scratch._note(f"scratch cache unavailable: {error}")
        self.read_only = read_only
        self._write_lock = threading.RLock()
        self._write_failure: str | None = None
        self._journal: queue.Queue[str] | None = None     # the fleet journal's writer, started on first use

    # ---- the fleet journal, written off the caller's thread ------------------

    def append_fleet_event(self, entry: dict) -> None:
        """One journal line, ENQUEUED: the fleet journal is observability, never
        read for correctness (base.py), and the desk appends it from its event
        loop at every placement step — so a refused append that is waited out
        (an hour of the API's bad minutes) must not hold the loop that answers
        every host. One writer thread appends in order; a reader or a persist
        drains it first; the write that finally fails stops this owner, and the
        next append says so."""
        self._check_writer()
        self._journal_queue().put(_canonical(entry))

    def _journal_queue(self) -> "queue.Queue[str]":
        with self._write_lock:
            if self._journal is None:
                self._journal = queue.Queue()
                threading.Thread(target=self._write_journal, name="fleet-journal",
                                 daemon=True).start()
            return self._journal

    def _write_journal(self) -> None:
        assert self._journal is not None
        while True:
            line = self._journal.get()
            try:
                if self._write_failure is None:
                    self._append_line("fleet/log.jsonl", line)
            except Exception as error:          # _publish recorded the owner's stop
                if self._write_failure is None:
                    self._write_failure = f"write 'fleet/log.jsonl': {error}"
            finally:
                self._journal.task_done()

    def _drain_journal(self) -> None:
        """Every enqueued line landed (or the owner stopped trying)."""
        if self._journal is not None:
            self._journal.join()

    def read_fleet_log(self) -> list[dict]:
        self._drain_journal()
        return super().read_fleet_log()

    def describe(self) -> str:
        return self.scratch.locator

    def address(self) -> StoreAddress:
        return StoreAddress("strangeloop", "", self.describe())

    def _check_writer(self) -> None:
        if self.read_only:
            raise StoreError("this scratch store is read-only")
        if self._write_failure is not None:
            raise StoreError("scratch owner stopped after uncertain mutation: " + self._write_failure)

    def _read(self, key: str) -> bytes:
        return self.scratch.read(key)

    def _read_hashed(self, key: str, sha256: str) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise StoreError(f"invalid sha256 for {key!r}: {sha256!r}")
        cache = self._blob_cache
        if cache is None:
            return self._read_hashed_source(key, sha256)
        with cache.claim(sha256):
            data = cache.get(sha256)
            if data is not None:
                if self.fingerprint(data) == sha256:
                    self.scratch._note(f"scratch hashed read cache: {key}")
                    return data
                self.scratch._note(f"scratch rejected corrupt cache: {key}")
                cache.discard(sha256)
            data = self._read_hashed_source(key, sha256)
            cache.put(sha256, data)
            return data

    def _read_hashed_source(self, key: str, sha256: str) -> bytes:
        """Use only matching mounted bytes; never reload or repair the mount."""
        if self.hashed_reads.mount_reads and self._mount_root is not None:
            try:
                data = (self._mount_root / scratch_path(key)).read_bytes()
            except FileNotFoundError:
                pass
            except OSError as error:
                self.scratch._note(f"scratch mount read unavailable for {key}: {error}")
            else:
                if self.fingerprint(data) == sha256:
                    self.scratch._note(f"scratch hashed read mount: {key}")
                    return data
                self.scratch._note(f"scratch rejected corrupt mount: {key}")
        data = self._verify_hash(key, self._read(key), sha256)
        self.scratch._note(f"scratch hashed read API: {key}")
        return data

    def _exists(self, key: str) -> bool:
        try:
            self.scratch.stat(key)
            return True
        except FileNotFoundError:
            return False

    def _size(self, key: str) -> int:
        return self.scratch.stat(key).size

    def _list(self, prefix: str) -> list[str]:
        return [entry.path for entry in self.scratch.list(prefix, recursive=True)
                if entry.type == "file"]

    def run_children(self, subdir: str = "") -> list[dict[str, str]]:
        folder = check_subdir(subdir) if subdir else ""
        prefix = "runs/" + folder
        if folder and self._exists(prefix + "/manifest.json"):
            return []
        return [{"name": entry.path.rsplit("/", 1)[-1],
                 "kind": "run" if self._exists(entry.path + "/manifest.json") else "folder",
                 "path": entry.path.removeprefix("runs/")}
                for entry in self.scratch.list(prefix) if entry.type == "dir"]

    def _run_directories(self) -> dict[str, str]:
        homes: dict[str, str] = {}
        def visit(prefix: str) -> None:
            entries = self.scratch.list(prefix)
            if any(entry.path == prefix + "/manifest.json" for entry in entries):
                homes[prefix.removeprefix("runs/")] = prefix
                return
            for entry in entries:
                if entry.type == "dir":
                    visit(entry.path)
        visit("runs")
        return homes

    def _readback(self, key: str) -> bytes:
        """The observation half of a publication, retried on its own.

        A write is attempted once (ADR 0015: uncertainty never triggers a
        retry). READING the result back is idempotent, so a transient
        transport failure on the readback — a response that ends before
        its declared byte count, a read deadline, a 5xx — is retried a
        bounded number of times before the owner concludes it cannot
        establish what it published. Only a readback that returns different
        bytes, or one that fails every attempt, stops this owner."""
        last: Exception | None = None
        for attempt in range(READBACK_ATTEMPTS):
            try:
                return self._read(key)
            except (StoreError, TimeoutError, OSError) as error:
                last = error
                if attempt + 1 < READBACK_ATTEMPTS:
                    time.sleep(READBACK_PAUSE_S)
        assert last is not None
        raise last

    def _still_there(self, key: str) -> bool:
        """`_exists`, retried like `_readback`: the observation after a
        delete is idempotent, so a transient stat failure is retried before
        the owner concludes it cannot establish what it deleted (the lamp
        pipeline's finding #18: one wire hiccup on the verifying stat
        quarantined the owner for good)."""
        last: Exception | None = None
        for attempt in range(READBACK_ATTEMPTS):
            try:
                return self._exists(key)
            except (StoreError, TimeoutError, OSError) as error:
                last = error
                if attempt + 1 < READBACK_ATTEMPTS:
                    time.sleep(READBACK_PAUSE_S)
        assert last is not None
        raise last

    def _published(self, key: str, data: bytes, *, hashed: bool) -> bool:
        """Observe publication without replaying it; hashed bytes need only a file size.

        Size confirmation deliberately does not prove content. Every hashed
        reader checks the digest; mutable writes still compare full readback.
        """
        if not hashed:
            return self._readback(key) == data
        for attempt in range(READBACK_ATTEMPTS):
            try:
                entry = self.scratch.stat(key)
                return entry.type == "file" and entry.size == len(data)
            except (StoreError, OSError):
                if attempt + 1 == READBACK_ATTEMPTS:
                    raise
                time.sleep(READBACK_PAUSE_S)
        return False

    def _publish(self, key: str, data: bytes, *, hashed: bool = False) -> None:
        """A write, then the readback that establishes what was published.

        Three outcomes of the write itself (ADR 0015, refined 2026-09-17
        after the desk's owner stopped on fleet-journal appends the API
        answered 500): the API ACCEPTED it — the readback must return the
        offered bytes or the owner stops; the API REJECTED it with a server
        error — nothing is in flight, so it is tried again, WRITE_ATTEMPTS
        times; the acknowledgement was LOST — the write may still land, so
        it is never replayed, but it is read back: the offered bytes mean it
        landed, anything else stops the owner. An uncertain write never
        triggers a retry; a refused one is not uncertain."""
        for attempt in range(WRITE_ATTEMPTS):
            try:
                self.scratch.write(key, data)
            except ScratchRejected:
                if attempt + 1 < WRITE_ATTEMPTS:
                    time.sleep(WRITE_PAUSE_S)
                    continue
                self._write_failure = f"write {key!r} rejected {WRITE_ATTEMPTS} times; establish completion before replacing this owner"
                raise
            except Exception:
                try:
                    landed = self._published(key, data, hashed=hashed)
                except Exception:
                    landed = False
                if landed:
                    return
                self._write_failure = f"write {key!r}; establish completion before replacing this owner"
                raise
            try:
                if not self._published(key, data, hashed=hashed):
                    raise StoreError("scratch write confirmation differs from offered "
                                     + ("length" if hashed else "bytes"))
            except Exception:
                self._write_failure = f"write {key!r}; establish completion before replacing this owner"
                raise
            return

    def _write(self, key: str, data: bytes) -> None:
        with self._write_lock:
            self._check_writer()
            self._publish(key, data)

    def _write_hashed(self, key: str, data: bytes, sha256: str) -> None:
        with self._write_lock:
            self._check_writer()
            self._verify_hash(key, data, sha256)
            self._publish(key, data, hashed=True)

    def _append_line(self, key: str, line: str) -> None:
        with self._write_lock:
            self._check_writer()
            try:
                previous = self._read(key)
            except FileNotFoundError:
                previous = b""
            self._publish(key, previous + ((line + "\n").encode("utf-8") if line else b""))

    def _delete(self, key: str) -> None:
        with self._write_lock:
            self._check_writer()
            self.scratch.stat(key)  # A definite missing file has not mutated anything.
            try:
                self.scratch.delete(key)
                if self._still_there(key):
                    raise StoreError("scratch delete readback still finds the file")
            except Exception:
                self._write_failure = f"delete {key!r}; establish completion before replacing this owner"
                raise

    def _persist(self) -> None:
        self._drain_journal()
        with self._write_lock:
            self._check_writer()

    def _sweep_partial(self, prefix: str) -> None:
        with self._write_lock:
            self._check_writer()
            for key in self._list(prefix):
                if key.endswith(".tmp"):
                    self._delete(key)


class StrangeLoopLocalStore(StrangeLoopStore, LocalStore):
    """Mounted mutations with per-write publication and verified hashed reads.

    Call verify_publication once during worker bootstrap, before offering it
    work. A reopened resident may read hashed mounted bytes but cannot mutate.
    """

    def __init__(self, root: str | Path, scratch: ScratchClient, *,
                 mountpoint: str | Path = "/scratch", commit_timeout: float = 30.0,
                 read_only: bool = False, hashed_reads: HashedReads | None = None) -> None:
        if not math.isfinite(commit_timeout) or commit_timeout <= 0:
            raise ValueError("scratch commit timeout must be finite and positive")
        self.mountpoint = Path(mountpoint)
        if not self.mountpoint.is_dir():
            raise StoreError("scratch mountpoint is absent")
        expected = self.mountpoint / scratch.prefix
        if Path(root).resolve() != expected.resolve():
            raise StoreError("mounted root differs from the scratch API prefix")
        settings = hashed_reads if hashed_reads is not None else HashedReads.from_environment()
        if (settings.blob_cache_dir is not None
                and settings.blob_cache_dir.resolve().is_relative_to(self.mountpoint.resolve())):
            raise ValueError("blob cache must be outside the shared mount")
        StrangeLoopStore.__init__(self, scratch, read_only=read_only,
                                 hashed_reads=settings, mount_root=root)
        self.root = Path(root)
        self.commit_timeout = commit_timeout
        self._publication_verified = False

    def address(self) -> StoreAddress:
        return StoreAddress("strangeloop", str(self.root), self.describe())

    def path_of(self, key: str) -> Path:
        """Mounted writes and API reads resolve the same encoded physical key."""
        return self.root / scratch_path(key)

    def _check_writer(self) -> None:
        super()._check_writer()
        if not self._publication_verified:
            raise StoreError("scratch mounted publication has not passed verify_publication()")

    def _sync(self) -> None:
        subprocess.run(["sync", str(self.mountpoint)], check=True,
                       timeout=self.commit_timeout, capture_output=True)

    def verify_publication(self) -> None:
        """Refuse a platform where mount sync does not publish API-readable bytes."""
        with self._write_lock:
            StrangeLoopStore._check_writer(self)
            key = ".publication/" + uuid.uuid4().hex
            payload = uuid.uuid4().bytes
            try:
                LocalStore._write(self, key, payload)
                self._sync()
                if self._read(key) != payload:
                    raise StoreError("scratch mount publication did not return the marker bytes")
                LocalStore._delete(self, key)
                self._sync()
                # Explicit API deletion also covers a mount that failed to publish removal.
                if self._still_there(key):
                    self.scratch.delete(key)
                if self._still_there(key):
                    raise StoreError("scratch publication marker deletion did not become visible")
                self._publication_verified = True
            except Exception:
                self._write_failure = "mounted publication probe failed"
                raise

    def _publish(self, key: str, data: bytes, *, hashed: bool = False) -> None:
        phase = "local write"
        try:
            LocalStore._write(self, key, data)
            phase = "mount sync"
            self._sync()
            phase = "publication readback"
            if not self._published(key, data, hashed=hashed):
                raise StoreError("scratch mount sync did not publish the offered "
                                 + ("length" if hashed else "bytes"))
        except Exception as error:
            # A concurrent operation may report the fence instead of this
            # exception. Keep the first failure visible through that path.
            self._write_failure = (f"mounted write {key!r} failed during {phase}: "
                                   f"{type(error).__name__}: {error}; "
                                   "stop this owner before recovery")
            raise

    def _append_line(self, key: str, line: str) -> None:
        """Hydrate committed history before touching an append target, even on resume.

        Atomic local replacement avoids publishing an intermediate hydrated
        prefix before the new line. Its bytes are exactly LocalStore's append.
        Native provider append remains required follow-up (ADR 0015).
        """
        super()._append_line(key, line)

    def _delete(self, key: str) -> None:
        with self._write_lock:
            self._check_writer()
            self.scratch.stat(key)
            try:
                # Discard this owner's stale copy before another sync can publish it.
                self.path_of(key).unlink(missing_ok=True)
                self._sync()
                if self._still_there(key):
                    self.scratch.delete(key)
                if self._still_there(key):
                    raise StoreError("scratch deletion remained visible")
            except Exception:
                self._write_failure = f"mounted delete {key!r}; stop this owner before recovery"
                raise

    def _persist(self) -> None:
        with self._write_lock:
            self._check_writer()
            try:
                self._sync()
            except Exception:
                self._write_failure = "mount sync failed; stop this owner before recovery"
                raise

    def _sweep_partial(self, prefix: str) -> None:
        with self._write_lock:
            self._check_writer()
            # Remove local-only debris first; do not trust the mounted tree for discovery.
            LocalStore._sweep_partial(self, prefix)
            self._persist()
            super()._sweep_partial(prefix)
