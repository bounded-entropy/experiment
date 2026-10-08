"""What a desk or a metal served over HTTP needs, whichever provider runs it.

A provider that stands its desk up as a process of its own (Strange Loop,
RunPod) needs the same few things around the shared runtimes: one owner per
state directory, a bearer token that survives restarts, a read-only observer
guarded by that token, a health probe, and a metal process that listens
before it registers and ends when its shift does. They live here so that no
provider folder imports another (ADR 0017).
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hmac
from http.cookies import SimpleCookie
import json
import os
from pathlib import Path
import secrets
import signal
import socket
from socketserver import ThreadingMixIn
import threading
from typing import Callable, Iterator
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import ProxyHandler, Request, build_opener
from wsgiref.simple_server import WSGIServer, make_server

from rlstack.data.stores.base import Store
from rlstack.observe.ui import ui_app
from rlstack.runner.venues.runtime import MetalRuntime

LOOPBACK = "127.0.0.1"
EVERY_INTERFACE = "0.0.0.0"
"""A desk or metal reached from another machine binds every interface: an
ingress proxy or a private network reaches the port from outside loopback,
and the bearer token (plus TLS at any public ingress) is what stands between
that network and the door."""

OBSERVER_COOKIE = "rlstack_observer"


def runtime_token(state_dir: Path) -> str:
    """A local secret survives gateway restarts without entering store addresses."""
    path = state_dir / "http-token"
    if not path.exists():
        state_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "x", opener=lambda file, flags: os.open(file, flags, 0o600)) as output:
            output.write(secrets.token_urlsafe(32))
    return path.read_text().strip()


def check_origin(endpoint: str, field: str) -> None:
    """An endpoint is a bare http(s) origin: a route (`#metal/host`) is the
    address's to add, and a token never rides in a URL."""
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError(f"{field} must be a plain http(s) origin such as "
                         f"https://<host> or http://<host>:<port>")


def health(endpoint: str, *, token: str = "", timeout: float = 5.0) -> bool:
    headers = {"Authorization": "Bearer " + token} if token else {}
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(endpoint + "/health", headers=headers), timeout=timeout) as response:
        return json.load(response).get("ready") is True


# ---------------------------------------------------------------------------
# one desk per state directory
# ---------------------------------------------------------------------------

BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")


def boot_id() -> str:
    """The kernel's boot id, fresh for every Modal container (each runs its
    own kernel), and empty where there is none to read (the laptop)."""
    try:
        return BOOT_ID_PATH.read_text().strip()
    except OSError:
        return ""


@dataclass(frozen=True)
class LockHolder:
    """Who took the desk lock: the host name, the pid and — where the kernel
    offers one — its boot id, written into the lock file."""
    host: str
    pid: int
    boot: str = ""

    @classmethod
    def mine(cls) -> "LockHolder":
        return cls(socket.gethostname(), os.getpid(), boot_id())

    def line(self) -> str:
        return f"{self.host} {self.pid}" + (f" {self.boot}" if self.boot else "") + "\n"

    def alive(self) -> bool:
        """A holder is live only on this very machine and only while its pid
        exists. A record another machine wrote — a container that died, its
        hostname with it — is stale by definition: the pid means nothing
        here. The host name alone cannot tell one Modal container from its
        successor (found on the first deploy, 2026-09-17: every container
        is host `modal` with the desk at pid 2, so a died container's record
        would read as this very process), so a record that names a boot id
        is live only under that boot."""
        if self.host != socket.gethostname():
            return False
        if self.boot and self.boot != boot_id():
            return False
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


def lock_holder(path: Path) -> LockHolder | None:
    """The record in a lock file; None for a missing, cleared or unreadable one."""
    try:
        fields = path.read_text().split()
    except FileNotFoundError:
        return None
    if len(fields) < 2 or not fields[1].isdigit():
        return None
    return LockHolder(fields[0], int(fields[1]), fields[2] if len(fields) > 2 else "")


@contextmanager
def own_local_desk(path: Path) -> Iterator[None]:
    """One desk process owns a fleet journal. The fence is flock where the
    filesystem fences (a laptop disk: the kernel drops a dead owner's lock)
    and the holder record where it does not (a mounted volume that refuses
    flock): a record whose process is gone, or that another machine wrote,
    is stale and a fresh desk takes it — so a `desk.lock` left on the volume
    by a container that died never bricks the next start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("a local desk already owns this state directory") from exc
        except OSError:
            holder = lock_holder(path)
            if holder is not None and holder.alive():
                raise RuntimeError("a local desk already owns this state directory "
                                   f"(pid {holder.pid} on {holder.host})")
        lock.seek(0)
        lock.truncate()
        lock.write(LockHolder.mine().line())
        lock.flush()
        try:
            yield
        finally:
            lock.seek(0)
            lock.truncate()
            lock.flush()
    # This lock does not fence another machine that can flock. Do not start
    # a second desk against this journal until the previous owner has
    # stopped (one owner per journal, ADR 0015).


# ---------------------------------------------------------------------------
# the observer beside the desk
# ---------------------------------------------------------------------------

class ObserverServer(ThreadingMixIn, WSGIServer):
    """An idle browser connection cannot block other reads or shutdown."""
    daemon_threads = True


def observer_guard(app: Callable, token: str) -> Callable:
    """An exposed observer answers the runtime token only: as the bearer
    header the gateway takes, or once as `?token=` — which sets the cookie a
    browser then carries — and nothing else, since a read-only page behind a
    public URL still shows every run."""
    def guarded(environ, start_response):
        cookies = SimpleCookie(environ.get("HTTP_COOKIE", ""))
        carried = cookies[OBSERVER_COOKIE].value if OBSERVER_COOKIE in cookies else ""
        bearer = environ.get("HTTP_AUTHORIZATION", "")
        if (hmac.compare_digest(bearer, "Bearer " + token)
                or hmac.compare_digest(carried, token)):
            return app(environ, start_response)
        query = parse_qs(environ.get("QUERY_STRING", ""))
        offered = query.pop("token", [""])[0]
        if hmac.compare_digest(offered, token):
            rest = urlencode(query, doseq=True)
            location = environ.get("PATH_INFO", "/") + ("?" + rest if rest else "")
            start_response("303 See Other", [
                ("Location", location),
                ("Set-Cookie", f"{OBSERVER_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict"),
                ("Content-Length", "0")])
            return [b""]
        start_response("401 Unauthorized", [("Content-Type", "text/plain"),
                                            ("Content-Length", "0")])
        return [b""]
    return guarded


@contextmanager
def observer(store: Store, port: int, *, host: str = LOOPBACK,
             token: str = "") -> Iterator[None]:
    """Serve the existing read-only observer against fresh store reads;
    off loopback it is guarded by the runtime token."""
    if host != LOOPBACK and not token:
        raise ValueError("an observer off loopback requires the runtime token")
    app = observer_guard(ui_app([store]), token) if token else ui_app([store])
    server = make_server(host, port, app, server_class=ObserverServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# the metal process
# ---------------------------------------------------------------------------

async def serve_metal(runtime: MetalRuntime, *, port: int, token: str,
                      host: str = LOOPBACK, ready: dict | None = None) -> None:
    """Listen, run the shared metal, and end on a signal or on the release.

    The door opens before the registration task can deliver a parked run
    back to this metal; both share this one owning event loop. The `ready`
    line, printed once registration has started, is the process log's proof
    of which epoch and which code this process is."""
    from rlstack.runner.transports.http import HttpServer

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(number, stop.set)
    async with HttpServer(runtime.service_for, host=host, port=port, token=token):
        async with runtime:
            print(json.dumps({"ready": True, "metal": runtime.service.metal.name,
                              "epoch": runtime.service.epoch, "idle_s": runtime.idle_s,
                              "pid": os.getpid(), **(ready or {})}), flush=True)
            stopped = asyncio.create_task(stop.wait())
            released = asyncio.create_task(runtime.wait())
            try:
                await asyncio.wait((stopped, released), return_when=asyncio.FIRST_COMPLETED)
            finally:
                for pending in (stopped, released):
                    pending.cancel()
                await asyncio.gather(stopped, released, return_exceptions=True)
