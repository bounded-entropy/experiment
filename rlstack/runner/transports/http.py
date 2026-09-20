"""The two Service doors over HTTP, with transparent temporary bulk transfers.

Large JSON is spooled and verified before dispatch, without changing model APIs
or scientific Store bytes. A mutation is sent once. Result blob GETs may retry;
a lost RPC acknowledgement still means unknown completion, never permission to
replay. Bind to loopback/SSH or terminate authenticated TLS at ingress.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import ContextVar
import hashlib
import hmac
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from rlstack.runner.remote import (
    BUILD_DEADLINE_S, DEADLINE_S, LocalTransport, Service, Unreachable,
    WrongEpoch, bounded, stamped,
)
from rlstack.runner.transports.http_blobs import (
    BlobRef, BlobStore, CHUNK_BYTES, ClientFile, HttpLimits, JsonFile,
)

# Legacy inline framing remains accepted; new peers offload at 1 MiB by default.
MAX_FRAME_BYTES = 64 * 1024 * 1024
PROTOCOL = "blobs-v1"
SERVING_UNTIL: ContextVar[float | None] = ContextVar("http_serving_until", default=None)


def forwarded_deadline_s(maximum: float = BUILD_DEADLINE_S) -> float:
    """Relays inherit the caller's remaining deadline across async/thread bridges."""
    until = SERVING_UNTIL.get()
    remaining = maximum if until is None else min(maximum, until - time.monotonic())
    if remaining <= 0:
        raise Unreachable("HTTP deadline elapsed before forwarding")
    return remaining


class RemoteError(RuntimeError):
    """A service refused a frame with a type outside the wire's known errors."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # never replay a model call or disclose its token elsewhere


class Budget:
    """Serialization, uploads, execution and downloads spend one caller budget."""
    def __init__(self, seconds: float, cancelled: threading.Event | None = None) -> None:
        self.until = time.monotonic() + seconds
        self.cancelled = cancelled or threading.Event()

    def remaining(self) -> float:
        remaining = self.until - time.monotonic()
        if self.cancelled.is_set() or remaining <= 0:
            raise Unreachable("HTTP deadline elapsed; operation completion may be unknown")
        return remaining

    def chunks(self, source, length: int, *, network: bool = False) -> Iterator[bytes]:
        while length:
            self.remaining()
            count = min(CHUNK_BYTES, length)
            # Buffered read1 returns available network bytes so a slow trickle
            # cannot hide the total deadline inside a full-chunk read.
            data = source.read1(count) if network else source.read(count)
            if not data:
                if network:
                    raise Unreachable("Truncated HTTP transfer; completion may be unknown")
                raise ValueError("Truncated HTTP blob")
            length -= len(data)
            yield data


ASK_RETRY_S = 600.0            # how long an idempotent ask is retried through a dropped door
ASK_RETRY_PAUSE_S = 2.0
ASK_RETRY_PAUSE_MAX_S = 30.0


def transient(error: Unreachable) -> Unreachable:
    """Mark an Unreachable the TRANSPORT raised (a connection that dropped or
    was refused) apart from one the far side answered with or a deadline
    that expired: only the former is worth asking again."""
    error.transient = True
    return error


class HttpTransport:
    """One logical RPC, with independent authenticated byte transfers when needed."""
    def __init__(self, endpoint: str, host: str = "", epoch: str = "", *,
                 token: str = "", limits: HttpLimits | None = None) -> None:
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query
                or parsed.fragment or parsed.path not in ("", "/")):
            raise ValueError("HTTP transport needs an http(s)://host:port endpoint")
        _ = parsed.port
        self.endpoint = endpoint.rstrip("/")
        self.host, self.epoch, self.token = host, epoch, token
        self.limits = limits or HttpLimits.from_environment()

    async def call(self, verb: str, payload: dict, *, deadline_s: float = DEADLINE_S) -> dict:
        return await self.frame("call", verb, payload, deadline_s)

    async def ask(self, verb: str, payload: dict, *, deadline_s: float = DEADLINE_S) -> dict:
        """An ask is idempotent by contract, so a TRANSIENT failure of the
        door — the connection dropped or refused: a desk container replaced
        while a run's judge call relayed through it, a tunnel reopening — is
        asked again, pausing ASK_RETRY_PAUSE_S doubling to ASK_RETRY_PAUSE_MAX_S,
        for ASK_RETRY_S (2026-09-17: every arm dialing a judge on another
        node died at one desk redeploy). A deadline that expired, or an
        Unreachable the far side answered with, is raised as before."""
        started = time.monotonic()
        pause = ASK_RETRY_PAUSE_S
        while True:
            try:
                return await self.frame("ask", verb, payload, deadline_s)
            except Unreachable as error:
                if not getattr(error, "transient", False) or time.monotonic() - started >= ASK_RETRY_S:
                    raise
                await asyncio.sleep(pause)
                pause = min(pause * 2, ASK_RETRY_PAUSE_MAX_S)

    async def frame(self, door: str, verb: str, payload: dict, deadline_s: float) -> dict:
        if not math.isfinite(deadline_s) or deadline_s <= 0:
            raise ValueError("deadline_s must be positive and finite")
        cancelled = threading.Event()
        budget = Budget(deadline_s, cancelled)
        try:
            return await bounded(asyncio.to_thread(
                self.request, door, verb, stamped(payload, self.epoch), budget),
                deadline_s, f"{self.endpoint}#{self.host}::{verb}")
        finally:
            cancelled.set()  # an abandoned upload must not dispatch a later RPC

    def open(self, method: str, path: str, budget: Budget, *, data=None,
             length: int | None = None, ref: BlobRef | None = None):
        headers = {"Content-Type": "application/json", "X-Rlstack-Protocol": PROTOCOL,
                   "X-Transfer-Timeout": str(budget.remaining())}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if length is not None:
            headers["Content-Length"] = str(length)
        if ref is not None:
            headers.update({"X-Blob-Size": str(ref.size), "X-Blob-Sha256": ref.sha256})
        opener = build_opener(ProxyHandler({}), NoRedirect())
        request = Request(self.endpoint + path, data=data, headers=headers, method=method)
        try:
            timeout = budget.remaining() if path in ("/call", "/ask") else min(budget.remaining(), self.limits.io_timeout_s)
            return opener.open(request, timeout=timeout)
        except HTTPError as exc:
            exc.close()
            raise RemoteError(f"HTTP {exc.code} from {self.endpoint}{path}; no RPC replay") from exc
        except (OSError, URLError, IncompleteRead) as exc:
            raise transient(Unreachable(f"{self.endpoint}{path}: {exc}; completion may be unknown")) from exc

    def read_control_response(self, response, budget: Budget) -> dict:
        """A finite control reply cannot hold a cancelled client thread indefinitely."""
        length = int(response.headers.get("Content-Length", "-1"))
        if not 0 < length <= MAX_FRAME_BYTES:
            raise RemoteError("HTTP control response lacks a bounded length")
        raw = b"".join(budget.chunks(response, length, network=True))
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RemoteError("Expected a JSON object response")
        return value

    def control(self, method: str, path: str, budget: Budget, value: dict | None = None,
                *, ref: BlobRef | None = None) -> dict:
        body = None if value is None else json.dumps(value, separators=(",", ":")).encode()
        if body is not None and len(body) > MAX_FRAME_BYTES:
            raise ValueError("HTTP control frame exceeds the limit before dispatch")
        try:
            with self.open(method, path, budget, data=body,
                           length=None if body is None else len(body), ref=ref) as response:
                value = self.read_control_response(response, budget)
            budget.remaining()
            return value
        except (OSError, URLError, IncompleteRead) as exc:
            raise Unreachable(f"HTTP reply lost; completion may be unknown: {exc}") from exc
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise RemoteError("Invalid HTTP JSON response; completion may be unknown") from exc

    def discard(self, ref: BlobRef) -> None:
        """Best-effort release never overrides a completed RPC; expiry is the fallback."""
        try:
            self.control("DELETE", "/blobs/" + ref.id, Budget(1), ref=ref)
        except (RemoteError, Unreachable, ValueError):
            pass

    def upload(self, body: JsonFile, budget: Budget) -> BlobRef:
        row = self.control("POST", "/blobs", budget,
                           {"size": body.size, "sha256": body.digest.hexdigest()})
        ref = BlobRef.decode(row["blob"], self.limits.max_blob_bytes)
        if (ref.size, ref.sha256) != (body.size, body.digest.hexdigest()):
            raise RemoteError("Upload reservation changed the bytes")
        try:
            with self.open("PUT", "/blobs/" + ref.id, budget,
                           data=budget.chunks(body.file, body.size), length=body.size, ref=ref) as response:
                reply = self.read_control_response(response, budget)
                if reply.get("ready") is not True:
                    raise RemoteError("Blob upload was not acknowledged")
            return ref
        except BaseException:
            self.discard(ref)
            raise

    def download(self, ref: BlobRef, budget: Budget) -> dict:
        """Retry only an immutable result GET, never the operation that made it."""
        try:
            for attempt in range(2):
                try:
                    with ClientFile(self.limits) as body:
                        with self.open("GET", "/blobs/" + ref.id, budget, ref=ref) as response:
                            if int(response.headers.get("Content-Length", "-1")) != ref.size:
                                raise RemoteError("Result blob length disagrees with its reference")
                            digest = hashlib.sha256()
                            for data in budget.chunks(response, ref.size, network=True):
                                body.append(data)
                                digest.update(data)
                        if digest.hexdigest() != ref.sha256:
                            raise RemoteError("Result blob checksum mismatch")
                        body.file.seek(0)
                        value = json.load(body.file)
                        budget.remaining()
                        return value
                except (OSError, URLError, IncompleteRead, Unreachable, ValueError) as exc:
                    budget.remaining()
                    if attempt:
                        raise Unreachable("Result download failed; RPC was not replayed") from exc
            raise AssertionError("unreachable")
        finally:
            self.discard(ref)

    def request(self, door: str, verb: str, payload: dict, budget: Budget) -> dict:
        ref = None
        try:
            with ClientFile(self.limits) as body:
                body.encode(payload)
                budget.remaining()
                frame = {"host": self.host, "verb": verb}
                if body.size <= min(self.limits.inline_bytes, MAX_FRAME_BYTES // 2):
                    frame["payload"] = json.load(body.file)
                else:
                    ref = self.upload(body, budget)
                    frame["payload_blob"] = ref.row()
                frame["deadline_s"] = budget.remaining()
                envelope = self.control("POST", "/" + door, budget, frame)
            if "result_blob" in envelope:
                result_ref = BlobRef.decode(envelope["result_blob"], self.limits.max_blob_bytes)
                envelope = self.download(result_ref, budget)
            if "error" in envelope:
                error = envelope["error"]
                kinds = {"WrongEpoch": WrongEpoch, "Unreachable": Unreachable,
                         "ValueError": ValueError, "KeyError": KeyError, "RuntimeError": RuntimeError}
                raise kinds.get(error["type"], RemoteError)(error["message"])
            result = envelope["result"]
            if not isinstance(result, dict):
                raise RemoteError("Expected a dict service result")
            return result
        except (OSError, URLError, IncompleteRead) as exc:
            raise transient(Unreachable("HTTP transfer failed; operation completion may be unknown")) from exc
        finally:
            if ref is not None:
                self.discard(ref)


class BoundedServer(ThreadingHTTPServer):
    """Connection pressure is refused before allocating another handler thread."""
    daemon_threads = True
    def __init__(self, address, handler, maximum: int):
        self.slots = threading.BoundedSemaphore(maximum)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class HttpServer:
    """Serve persistent Service objects; transfer storage lives only with this runner."""
    def __init__(self, service_for: Callable[[str], Service], *, host: str = "127.0.0.1",
                 port: int = 8000, token: str = "", max_deadline_s: float = BUILD_DEADLINE_S,
                 limits: HttpLimits | None = None) -> None:
        if host not in ("127.0.0.1", "localhost") and not token:
            raise ValueError("A non-loopback HTTP server requires a bearer token and TLS at ingress")
        if not math.isfinite(max_deadline_s) or max_deadline_s <= 0:
            raise ValueError("max_deadline_s must be positive and finite")
        self.service_for = service_for
        self.host, self.port, self.token = host, port, token
        self.max_deadline_s = max_deadline_s
        self.limits = limits or HttpLimits.from_environment()
        self._server: BoundedServer | None = None
        self._thread: threading.Thread | None = None
        self.blobs: BlobStore | None = None
        self._stop = threading.Event()
        self._collector: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        if self._server is None:
            raise RuntimeError("HTTP service has not started")
        return f"http://{self.host}:{self._server.server_port}"

    async def dispatch(self, door: str, frame: dict) -> dict:
        if self._stop.is_set():
            raise Unreachable("HTTP service is stopping; frame was not dispatched")
        context = SERVING_UNTIL.set(time.monotonic() + frame["deadline_s"])
        try:
            transport = LocalTransport(self.service_for(frame["host"]))
            method = transport.call if door == "call" else transport.ask
            return await method(frame["verb"], frame["payload"], deadline_s=frame["deadline_s"])
        finally:
            SERVING_UNTIL.reset(context)

    async def __aenter__(self) -> "HttpServer":
        owner, loop = self, asyncio.get_running_loop()
        self._stop.clear()
        self.blobs = blobs = BlobStore(self.limits)

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(owner.limits.io_timeout_s)

            def log_message(self, *args):
                pass

            def authorized(self) -> bool:
                return not owner.token or hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + owner.token)

            def respond(self, status: int, value: dict) -> None:
                data = json.dumps(value, separators=(",", ":")).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def length(self, maximum: int) -> int:
                values = self.headers.get_all("Content-Length", [])
                if len(values) != 1 or self.headers.get("Transfer-Encoding"):
                    raise ValueError("Expected one bounded Content-Length")
                length = int(values[0])
                if not 0 < length <= maximum:
                    raise ValueError("HTTP body exceeds its configured limit")
                return length

            def budget(self) -> Budget:
                seconds = float(self.headers.get("X-Transfer-Timeout", owner.max_deadline_s))
                if not math.isfinite(seconds) or not 0 < seconds <= owner.max_deadline_s:
                    raise ValueError("Invalid transfer deadline")
                return Budget(seconds)

            def read_control(self) -> dict:
                length = self.length(MAX_FRAME_BYTES)
                budget = self.budget()
                self.connection.settimeout(min(owner.limits.io_timeout_s, budget.remaining()))
                raw = b"".join(budget.chunks(self.rfile, length, network=True))
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("Expected a JSON object")
                return value

            def reference(self) -> BlobRef:
                return BlobRef.decode({"id": self.path.removeprefix("/blobs/"),
                                       "size": int(self.headers.get("X-Blob-Size", "0")),
                                       "sha256": self.headers.get("X-Blob-Sha256", "")},
                                      owner.limits.max_blob_bytes)

            def transfer(self, method: str) -> None:
                budget = self.budget()
                if method == "POST" and self.path == "/blobs":
                    row = self.read_control()
                    if set(row) != {"size", "sha256"}:
                        raise ValueError("Expected upload size and SHA-256")
                    checked = BlobRef.decode({"id": "0" * 32, **row}, owner.limits.max_blob_bytes)
                    ref = blobs.reserve(checked.size, checked.sha256)
                    self.respond(200, {"blob": ref.row()})
                    return
                ref = self.reference()
                if method == "DELETE":
                    # Validate the full reference before deletion; ids are not paths.
                    blobs.status(ref)
                    blobs.discard(ref.id)
                    self.respond(200, {"deleted": True})
                elif method == "PUT":
                    if self.length(owner.limits.max_blob_bytes) != ref.size:
                        raise ValueError("Upload length disagrees with reservation")
                    digest = hashlib.sha256()
                    with blobs.upload(ref) as output:
                        self.connection.settimeout(min(owner.limits.io_timeout_s, budget.remaining()))
                        for data in budget.chunks(self.rfile, ref.size, network=True):
                            self.connection.settimeout(min(owner.limits.io_timeout_s, budget.remaining()))
                            output.write(data)
                            digest.update(data)
                        if digest.hexdigest() != ref.sha256:
                            raise ValueError("Upload checksum mismatch")
                    self.respond(200, {"ready": True})
                elif method == "GET":
                    with blobs.read(ref) as source:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/octet-stream")
                        self.send_header("Content-Length", str(ref.size))
                        self.end_headers()
                        for data in budget.chunks(source, ref.size):
                            self.connection.settimeout(min(owner.limits.io_timeout_s, budget.remaining()))
                            self.wfile.write(data)
                else:
                    self.respond(404, {"error": "unknown transfer endpoint"})

            def rpc(self) -> None:
                received = time.monotonic()
                frame = self.read_control()
                if not isinstance(frame.get("host"), str) or not isinstance(frame.get("verb"), str):
                    raise ValueError("Expected host and verb strings")
                deadline = float(frame.get("deadline_s", DEADLINE_S))
                if not math.isfinite(deadline) or not 0 < deadline <= owner.max_deadline_s:
                    raise ValueError("Request deadline exceeds the service's finite bound")
                if ("payload" in frame) == ("payload_blob" in frame):
                    raise ValueError("Expected exactly one inline or blob payload")
                if "payload_blob" in frame:
                    ref = BlobRef.decode(frame.pop("payload_blob"), owner.limits.max_blob_bytes)
                    with blobs.read(ref, consume=True) as source:
                        digest = hashlib.sha256()
                        for data in Budget(deadline).chunks(source, ref.size):
                            digest.update(data)
                        if digest.hexdigest() != ref.sha256:
                            raise ValueError("Stored request checksum mismatch")
                        source.seek(0)
                        frame["payload"] = json.load(source)
                if not isinstance(frame["payload"], dict):
                    raise ValueError("Expected dict payload")
                deadline -= time.monotonic() - received
                if deadline <= 0:
                    raise ValueError("Request deadline elapsed before dispatch")
                frame["deadline_s"] = deadline
                pending = asyncio.run_coroutine_threadsafe(owner.dispatch(self.path[1:], frame), loop)
                try:
                    result = pending.result(timeout=deadline)
                    envelope = {"result": result}
                except FutureTimeout:
                    pending.cancel()
                    envelope = {"error": {"type": "Unreachable", "message": "Service deadline elapsed"}}
                except Exception as exc:
                    envelope = {"error": {"type": type(exc).__name__, "message": str(exc)}}
                try:
                    ref = blobs.produce(envelope)
                    inline_limit = (min(owner.limits.inline_bytes, MAX_FRAME_BYTES)
                                    if self.headers.get("X-Rlstack-Protocol") == PROTOCOL else MAX_FRAME_BYTES)
                    if ref.size <= inline_limit:
                        with blobs.read(ref) as source:
                            value = json.load(source)
                        blobs.discard(ref.id)
                        self.respond(200, value)
                    elif self.headers.get("X-Rlstack-Protocol") == PROTOCOL:
                        self.respond(200, {"result_blob": ref.row()})
                    else:
                        blobs.discard(ref.id)
                        self.respond(200, {"error": {"type": "RemoteError", "message":
                            "Result needs blob-capable client; operation already executed; do not replay"}})
                except (ValueError, OSError, RuntimeError) as exc:
                    self.respond(200, {"error": {"type": "RemoteError", "message":
                        f"Result transfer failed after execution; do not replay: {exc}"}})

            def route(self, method: str) -> None:
                try:
                    if not self.authorized():
                        self.respond(401, {"error": "unauthorized"})
                    elif method == "GET" and self.path == "/health":
                        self.respond(200, {"ready": True, "protocol": PROTOCOL,
                                           "max_blob_bytes": owner.limits.max_blob_bytes})
                    elif self.path == "/blobs" or self.path.startswith("/blobs/"):
                        self.transfer(method)
                    elif method == "POST" and self.path in ("/call", "/ask"):
                        self.rpc()
                    else:
                        self.respond(404, {"error": "unknown endpoint"})
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    pass  # lost connection is never a reason to replay
                except (ValueError, TypeError, KeyError, UnicodeError, Unreachable) as exc:
                    try:
                        self.respond(400, {"error": str(exc)})
                    except (OSError, TimeoutError):
                        pass

            def do_GET(self): self.route("GET")
            def do_POST(self): self.route("POST")
            def do_PUT(self): self.route("PUT")
            def do_DELETE(self): self.route("DELETE")

        try:
            self._server = BoundedServer((self.host, self.port), Handler, self.limits.max_connections)
        except BaseException:
            blobs.close()
            raise
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        def collect() -> None:
            while not self._stop.wait(min(1, owner.limits.ttl_s / 2)):
                blobs.collect()
        self._collector = threading.Thread(target=collect, daemon=True)
        self._collector.start()
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._server is not None and self._thread is not None and self.blobs is not None
        self._stop.set()  # frames still parsing cannot enter a closing runtime
        await asyncio.to_thread(self._server.shutdown)
        self._server.server_close()
        await asyncio.to_thread(self._thread.join)
        self._stop.set()
        await asyncio.to_thread(self._collector.join)
        self.blobs.close()
        self._server = self._thread = None
