"""The two service doors over HTTP, for resident processes outside Modal.

The HTTP server lives on the host's existing event loop. All admitted work
still goes through Service.serve, and synchronous answers leave that loop
through LocalTransport. No request constructs a model or retries a mutation.

Bind to loopback and use an SSH tunnel (for example `strangeloop gpu forward`)
or terminate authenticated TLS in front of it. This module adds no dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeout
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from rlstack.runner.remote import (
    BUILD_DEADLINE_S, DEADLINE_S, LocalTransport, Service, Unreachable,
    WrongEpoch, bounded, stamped,
)

MAX_FRAME_BYTES = 64 * 1024 * 1024


class RemoteError(RuntimeError):
    """A service refused a frame with a type outside the wire's known errors."""


class NoRedirect(HTTPRedirectHandler):
    """A model call is sent to exactly one door, never replayed at a redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpTransport:
    """A JSON frame per call; independent requests may be in flight together.

    Each request has its own connection so the synchronous Blocking bridge and
    async sampling loops can share this object. The daemon, not a connection,
    owns the state. A timed-out mutation may have executed: callers must not
    retry it blindly. Optional bearer auth is supplied explicitly, not in URLs.
    """

    def __init__(self, endpoint: str, host: str = "", epoch: str = "", *,
                 token: str = "") -> None:
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username or parsed.password or parsed.query
                or parsed.fragment or parsed.path not in ("", "/")):
            raise ValueError("HTTP transport needs an http(s)://host:port endpoint")
        self.endpoint = endpoint.rstrip("/")
        self.host, self.epoch, self.token = host, epoch, token

    async def call(self, verb: str, payload: dict, *,
                   deadline_s: float = DEADLINE_S) -> dict:
        return await self.frame("call", verb, payload, deadline_s)

    async def ask(self, verb: str, payload: dict, *,
                  deadline_s: float = DEADLINE_S) -> dict:
        return await self.frame("ask", verb, payload, deadline_s)

    async def frame(self, door: str, verb: str, payload: dict,
                    deadline_s: float) -> dict:
        if not math.isfinite(deadline_s) or deadline_s <= 0:
            raise ValueError("deadline_s must be positive and finite")
        return await bounded(asyncio.to_thread(
            self.request, door, verb, stamped(payload, self.epoch), deadline_s),
            deadline_s, f"{self.endpoint}#{self.host}::{verb}")

    def request(self, door: str, verb: str, payload: dict,
                deadline_s: float) -> dict:
        body = json.dumps({"host": self.host, "verb": verb, "payload": payload,
                           "deadline_s": deadline_s}).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        request = Request(f"{self.endpoint}/{door}", data=body, headers=headers)
        # Explicit URLs must not route through an inherited HTTP_PROXY: a local
        # SSH endpoint, in particular, must remain local.
        opener = build_opener(ProxyHandler({}), NoRedirect())
        try:
            with opener.open(request, timeout=deadline_s) as response:
                raw = response.read(MAX_FRAME_BYTES + 1)
        except HTTPError as exc:
            exc.close()
            raise RemoteError(f"HTTP {exc.code} from {self.endpoint}/{door}") from exc
        except (OSError, URLError) as exc:
            raise Unreachable(f"{self.endpoint}/{door}: {exc}") from exc
        if len(raw) > MAX_FRAME_BYTES:
            raise RemoteError("HTTP response exceeds the frame limit")
        envelope = json.loads(raw)
        if "error" in envelope:
            error = envelope["error"]
            kinds = {"WrongEpoch": WrongEpoch, "Unreachable": Unreachable,
                     "ValueError": ValueError, "KeyError": KeyError,
                     "RuntimeError": RuntimeError}
            raise kinds.get(error["type"], RemoteError)(error["message"])
        return envelope["result"]


class HttpServer:
    """Serve existing Service objects on ONE owning loop until context exit.

    `service_for` resolves a host fragment, including the empty fragment for a
    desk or metal plane. A venue supplies this routing; HTTP knows no model,
    placement rule or lifecycle policy. A finite caller deadline bounds each
    door. Network timeouts do not promise cancellation of synchronous GPU work.
    """

    def __init__(self, service_for: Callable[[str], Service], *,
                 host: str = "127.0.0.1", port: int = 8000, token: str = "",
                 max_deadline_s: float = BUILD_DEADLINE_S) -> None:
        if host not in ("127.0.0.1", "localhost") and not token:
            raise ValueError("A non-loopback HTTP server requires a bearer token and TLS at ingress")
        if not math.isfinite(max_deadline_s) or max_deadline_s <= 0:
            raise ValueError("max_deadline_s must be positive and finite")
        self.service_for = service_for
        self.host, self.port, self.token = host, port, token
        self.max_deadline_s = max_deadline_s
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        if self._server is None:
            raise RuntimeError("HTTP service has not started")
        return f"http://{self.host}:{self._server.server_port}"

    async def dispatch(self, door: str, frame: dict) -> dict:
        transport = LocalTransport(self.service_for(frame["host"]))
        if door == "call":
            return await transport.call(frame["verb"], frame["payload"],
                                        deadline_s=frame["deadline_s"])
        return await transport.ask(frame["verb"], frame["payload"],
                                   deadline_s=frame["deadline_s"])

    async def __aenter__(self) -> "HttpServer":
        owner, loop = self, asyncio.get_running_loop()

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(30)

            def log_message(self, *args):
                pass

            def authorized(self) -> bool:
                return not owner.token or hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + owner.token)

            def respond(self, status: int, value: dict) -> None:
                data = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # client deadline elapsed; do not replay its work

            def do_GET(self):
                if not self.authorized():
                    self.respond(401, {"error": "unauthorized"})
                elif self.path == "/health":
                    self.respond(200, {"ready": True})
                else:
                    self.respond(404, {"error": "unknown endpoint"})

            def do_POST(self):
                if not self.authorized():
                    self.respond(401, {"error": "unauthorized"})
                    return
                if self.path not in ("/call", "/ask"):
                    self.respond(404, {"error": "unknown door"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if (not 0 < length <= MAX_FRAME_BYTES
                            or self.headers.get("Transfer-Encoding")):
                        raise ValueError("Expected a bounded Content-Length JSON frame")
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise ValueError("Incomplete HTTP frame")
                    frame = json.loads(raw)
                    if (not isinstance(frame["host"], str)
                            or not isinstance(frame["verb"], str)
                            or not isinstance(frame["payload"], dict)):
                        raise ValueError("Expected host, verb and dict payload")
                    deadline = float(frame.get("deadline_s", DEADLINE_S))
                    if not math.isfinite(deadline) or not 0 < deadline <= owner.max_deadline_s:
                        raise ValueError("Request deadline exceeds the service's finite bound")
                    frame["deadline_s"] = deadline
                except (ValueError, TypeError, KeyError) as exc:
                    self.respond(400, {"error": str(exc)})
                    return
                pending = asyncio.run_coroutine_threadsafe(
                    owner.dispatch(self.path[1:], frame), loop)
                try:
                    result = pending.result(timeout=deadline + 1)
                    envelope = {"result": result}
                except FutureTimeout:
                    pending.cancel()
                    envelope = {"error": {"type": "Unreachable", "message": "Service deadline elapsed"}}
                except Exception as exc:
                    envelope = {"error": {"type": type(exc).__name__, "message": str(exc)}}
                self.respond(200, envelope)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._server is not None and self._thread is not None
        await asyncio.to_thread(self._server.shutdown)
        self._server.server_close()
        await asyncio.to_thread(self._thread.join)
        self._server = self._thread = None
