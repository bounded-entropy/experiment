"""Strange Loop's lease and SSH wiring around the shared Desk and Metal.

The laptop owns one gateway. Every worker reverse-forwards that gateway onto
its own loopback, so a recorded address means the same thing on every caller.
The gateway forwards to each worker's existing service; it never places work.
Provider operations use the official CLI, including its conversation capture.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Mapping
from typing import Callable, Iterator
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
import uuid
import zipfile

from rlstack.runner.remote import Service, Transport, Unreachable
from rlstack.runner.residents import Builds
from rlstack.data.stores.strangeloop import HashedReads


@dataclass(frozen=True)
class MetalConfig:
    """An explicitly authorized allocation slot, never an inferred GPU budget."""
    name: str
    gpu: str
    image: str
    local_port: int
    devices: int = 1
    hours: float = 4.0
    idle_s: float = 90.0
    python: str = "/opt/rlstack/bin/python"
    builds: Builds | None = None
    environment: dict[str, str] | None = None
    blob_cache_dir: str | None = None
    blob_cache_bytes: int = 0
    mount_reads: bool = False

    def __post_init__(self) -> None:
        self.hashed_reads()
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", self.name):
            raise ValueError("metal name must contain only letters, digits, '_' and '-'")
        if not self.image:
            raise ValueError("each metal requires a prebuilt compatible image")
        if not 1 <= self.local_port <= 65535 or self.devices < 1:
            raise ValueError("metal needs a valid forward port and positive device count")
        if not math.isfinite(self.idle_s) or self.idle_s <= 0:
            raise ValueError("automatic GPU idle release must be finite and positive")
        if not 0.25 <= self.hours <= 24:
            raise ValueError("lease hours must be between 0.25 and 24")
        if self.environment is not None and not all(
                isinstance(key, str) and isinstance(value, str) for key, value in self.environment.items()):
            raise ValueError("worker environment names and values must be strings")

    def hashed_reads(self) -> HashedReads:
        """Store tuning belongs to the pod, not to an experiment's spec."""
        return HashedReads(Path(self.blob_cache_dir) if self.blob_cache_dir else None,
                           self.blob_cache_bytes, self.mount_reads)


@dataclass(frozen=True)
class DeskConfig:
    """The one local service, shared scratch identity, and allowed metal slots."""
    store: str
    metals: tuple[MetalConfig, ...]
    state_dir: Path
    gateway_port: int = 18760
    observer_port: int = 8321
    worker_port: int = 8000
    idle_s: float = 90.0
    gpu_ceiling: int = 1
    automatic_recovery: bool = False
    operator_endpoint: str = ""
    """Where the operator's verbs reach a desk that is NOT on this machine
    (a hosted container's public URL). Empty means the laptop-run desk: the
    loopback gateway. Never part of a recorded address — every worker keeps
    reaching the gateway at its own loopback through the reverse tunnel."""

    def __post_init__(self) -> None:
        ports = [self.gateway_port, self.observer_port,
                 *(metal.local_port for metal in self.metals)]
        if len(set(ports)) != len(ports) or any(not 1 <= p <= 65535 for p in ports):
            raise ValueError("gateway, observer and each local forward need distinct valid ports")
        if not 1 <= self.worker_port <= 65535 or self.worker_port == self.gateway_port:
            raise ValueError("worker service port must differ from reverse gateway port")
        if len({m.name for m in self.metals}) != len(self.metals):
            raise ValueError("metal names must be unique")
        if not math.isfinite(self.idle_s) or self.idle_s <= 0:
            raise ValueError("automatic GPU idle release must remain finite")
        if self.gpu_ceiling < 1:
            raise ValueError("gpu_ceiling must explicitly allow at least one GPU")
        if sum(metal.devices for metal in self.metals) > self.gpu_ceiling:
            raise ValueError("configured allocation slots exceed gpu_ceiling")
        if not self.store.startswith("strangeloop://"):
            raise ValueError("Strange Loop desk requires its explicit scratch locator")
        if self.operator_endpoint:
            check_operator_endpoint(self.operator_endpoint)

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.gateway_port}"

    @property
    def operator_door(self) -> str:
        """Where an operator verb knocks: the hosted desk's URL when one is
        configured, else the same loopback gateway the workers see."""
        return self.operator_endpoint or self.endpoint

    def address(self, metal: str, host: str = "", epoch: str = "") -> str:
        route = metal + ("/" + host if host else "")
        return self.endpoint + "#" + route + ("@" + epoch if epoch else "")

    @classmethod
    def read(cls, path: str | Path, *, state_dir: Path | None = None) -> "DeskConfig":
        """The file's config; `state_dir` replaces the file's own for a desk
        whose state lives elsewhere than the file says (a hosted container
        keeps it on its volume, while the file names the laptop's)."""
        row = json.loads(Path(path).read_text())
        metals = []
        for raw in row.pop("metals"):
            raw = dict(raw)
            if raw.get("builds") is not None:
                raw["builds"] = Builds.from_row(raw["builds"])
            metals.append(MetalConfig(**raw))
        row["state_dir"] = (state_dir if state_dir is not None
                            else Path(row["state_dir"]).expanduser().resolve())
        return cls(metals=tuple(metals), **row)


def check_operator_endpoint(endpoint: str) -> None:
    """An operator endpoint is a bare http(s) origin: the route (`#metal/host`)
    is the gateway's to add, and a token never rides in a URL."""
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("operator_endpoint must be a plain http(s) origin such as "
                         "https://<workspace>--rlstack-sl-desk-gateway.modal.run")


ENDED_LEASE_STATUSES = frozenset({"released", "failed"})


def lease_ended(row: Mapping) -> bool:
    """Is this lease over, by the platform's own word: released by us, or
    failed and closed by it (a status that never becomes `released`)."""
    return row.get("status") in ENDED_LEASE_STATUSES


SSH_METADATA_WAIT_S = 300.0   # a fresh pod's SSH metadata lags its ready status


@dataclass
class LeaseState:
    """Provider identity retained before/after a request, independent of the fleet journal."""
    label: str
    lease_id: str = ""
    boot_started: bool = False


class StrangeLoopCLI:
    """Bounded official commands; ambiguous mutations are never automatically replayed."""
    def __init__(self, executable: str | None = None) -> None:
        installed = Path.home() / ".local" / "bin" / "strangeloop"
        selected = executable or os.environ.get("STRANGELOOP_CLI") or shutil.which("strangeloop")
        if selected is None and installed.is_file():
            selected = str(installed)
        if selected is None:
            raise RuntimeError("Strange Loop CLI is not installed; set STRANGELOOP_CLI to its executable")
        self.executable = selected

    def run(self, *args: str, timeout: float = 180.0) -> dict:
        result = subprocess.run([self.executable, "--json", *args],
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"Strange Loop {' '.join(args[:2])} failed "
                               f"({result.returncode}): {result.stderr.strip()}")
        return json.loads(result.stdout)

    def status(self, lease_id: str) -> dict:
        return self.run("gpu", "status", lease_id)

    def down(self, lease_id: str) -> bool:
        """Confirm the exact lease ended, including a lost deletion response.

        A successful DELETE is not the confirmation: bounded status reads
        must report ENDED. Two statuses are ended on this platform: `released`,
        and `failed` — a lease the platform closed itself ("the pod backing
        this lease is gone ... Closed so it stops billing"), which never
        becomes `released` and never appears among the active leases; taking
        it for live parked two metal slots for good on 2026-09-17. Unknown
        status never permits replacement. The provider's physical termination
        guarantee still needs a live drill.
        """
        deadline = time.monotonic() + 180.0
        try:
            if lease_ended(self.run("gpu", "status", lease_id, timeout=15.0)):
                return True
        except (RuntimeError, subprocess.TimeoutExpired):
            pass
        try:
            self.run("gpu", "down", lease_id, timeout=min(120.0, deadline - time.monotonic()))
        except (RuntimeError, subprocess.TimeoutExpired):
            # The delete may have succeeded. Read its state; never substitute
            # a failed SSH probe or a 'failed' provisioning state for release.
            pass
        for attempt in range(6):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                row = self.run("gpu", "status", lease_id, timeout=min(15.0, remaining))
                if lease_ended(row):
                    return True
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
            if attempt < 5:
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        return False


class RelayService:
    """Forward service frames unchanged; admission and state stay on the GPU host."""
    def __init__(self, transport: Transport) -> None:
        self.transport = transport

    async def serve(self, verb: str, payload: dict) -> dict:
        from rlstack.runner.transports.http import forwarded_deadline_s
        return await self.transport.call(verb, payload, deadline_s=forwarded_deadline_s())

    def answer(self, verb: str, payload: dict) -> dict:
        # HttpServer calls synchronous answers on its worker thread. Outer
        # caller bounds still apply; cancellation never triggers a replay.
        from rlstack.runner.transports.http import forwarded_deadline_s
        return asyncio.run(self.transport.ask(verb, payload,
                                             deadline_s=forwarded_deadline_s()))


class Gateway:
    """Stable canonical addresses routed through the laptop's SSH forwards."""
    def __init__(self, config: DeskConfig, *, token: str = "") -> None:
        self.config, self.token = config, token
        self.desk_service: Service | None = None
        self.metals = {metal.name: metal for metal in config.metals}

    def service_for(self, host: str) -> Service:
        if not host:
            if self.desk_service is None:
                raise Unreachable("desk has not started")
            return self.desk_service
        name, _, member = host.partition("/")
        metal = self.metals.get(name)
        if metal is None:
            raise ValueError(f"unknown metal {name!r}")
        from rlstack.runner.transports.http import HttpTransport

        return RelayService(HttpTransport(f"http://127.0.0.1:{metal.local_port}",
                                          member, token=self.token))


def source_bundle(root: Path) -> bytes:
    """Ship the exact working Python and engine plugin, including uncommitted work."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for package in ("rlstack", "rlstack_engine"):
            for path in sorted((root / package).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    bundle.write(path, str(path.relative_to(root)))
    return buffer.getvalue()


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
    """One desk process owns a scratch fleet journal. The fence is flock where
    the filesystem fences (a laptop disk: the kernel drops a dead owner's
    lock) and the holder record where it does not (a mounted volume that
    refuses flock): a record whose process is gone, or that another machine
    wrote, is stale and a fresh desk takes it — so a `desk.lock` left on the
    volume by a container that died never bricks the next start."""
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
    # a second desk against this scratch prefix until the previous owner has
    # stopped (one owner per journal, ADR 0015).


def ssh_forward_argv(argv: list[str], gateway_port: int) -> list[str]:
    """Add a reverse route before SSH's destination, never as a remote command."""
    if not argv or argv[0] != "ssh" or "-N" not in argv or len(argv) < 3:
        raise ValueError("Strange Loop did not return its expected SSH argument array")
    return [*argv[:-1], "-R", f"127.0.0.1:{gateway_port}:127.0.0.1:{gateway_port}",
            argv[-1]]


def ssh_command_argv(forward: list[str], command: str) -> list[str]:
    """Reuse the CLI-installed identity without forwarding or exposing its key."""
    config = forward[forward.index("-F") + 1]
    return ["ssh", "-F", config, "-T", "-o", "BatchMode=yes", "-o",
            "ConnectTimeout=15", forward[-1], command]


def upload_over_ssh(forward: list[str], path: str, content: bytes) -> None:
    """Send code or secret configuration over SSH stdin, never command arguments."""
    command = "umask 077; cat > " + shlex.quote(path)
    subprocess.run(ssh_command_argv(forward, command), input=content,
                   capture_output=True, check=True, timeout=180)


def credentials_path(lease_id: str) -> str:
    """Where a pod keeps its scratch token: written at boot, rewritten by
    `reauth`, read by every process there through SL_API_TOKEN_FILE."""
    return "/tmp/rlstack-" + lease_id + "-credentials.json"


def reverse_forwarding_config(original: str, port: int) -> str:
    """Permit this desk's loopback reverse route on the lease's OpenSSH server."""
    if not 1 <= port <= 65535:
        raise ValueError("reverse gateway port must be valid")
    lines = original.splitlines()
    if any(line.strip().lower().startswith("match ") for line in lines):
        raise RuntimeError("cannot amend a scoped provider SSH configuration")
    if "GatewayPorts no" not in lines or not any(
            line in ("AllowTcpForwarding local", "AllowTcpForwarding yes") for line in lines):
        raise RuntimeError("provider SSH configuration differs from the validated runner")
    lines = [line for line in lines if not line.startswith(("AllowTcpForwarding ", "PermitListen "))]
    return "\n".join([*lines, "AllowTcpForwarding yes", f"PermitListen 127.0.0.1:{port}"]) + "\n"


def configure_reverse_forwarding(forward: list[str], port: int) -> None:
    """The current runner enables only -L; validate and reload a narrowly scoped -R permission.

    This is lease-local provider wiring. Public listening stays disabled and
    authentication is unchanged. The official forward command can regenerate
    the server configuration, so check after obtaining its SSH argument array.
    """
    path = "/run/strangeloop-ssh/sshd_config"
    read = subprocess.run(ssh_command_argv(forward, "cat " + path),
                          capture_output=True, text=True, check=True, timeout=30)
    revised = reverse_forwarding_config(read.stdout, port)
    if revised == read.stdout:
        return
    temporary = path + ".rlstack-next"
    upload_over_ssh(forward, temporary, revised.encode())
    command = (f"/usr/sbin/sshd -t -f {temporary} && "
               f"cp -n {path} {path}.rlstack-before && mv {temporary} {path} && "
               "kill -HUP $(cat /run/strangeloop-ssh/sshd.pid)")
    subprocess.run(ssh_command_argv(forward, command),
                   capture_output=True, check=True, timeout=30)


def health(endpoint: str, *, token: str = "", timeout: float = 5.0) -> bool:
    headers = {"Authorization": "Bearer " + token} if token else {}
    opener = build_opener(ProxyHandler({}))
    with opener.open(Request(endpoint + "/health", headers=headers), timeout=timeout) as response:
        return json.load(response).get("ready") is True


class StrangeLoopVenue:
    """Allocation, forwarding and bootstrap; the shared Desk retains placement policy."""
    def __init__(self, config: DeskConfig, source_root: Path, *,
                 cli: StrangeLoopCLI | None = None, token: str = "") -> None:
        self.config, self.source_root, self.token = config, source_root, token
        self.cli = cli or StrangeLoopCLI()
        self.metals = {m.name: m for m in config.metals}
        self.processes: dict[str, subprocess.Popen] = {}
        self.forward_leases: dict[str, str] = {}
        self.registered: Callable[[str, str, str], bool] | None = None
        self.locks = {name: threading.Lock() for name in self.metals}
        self.forward_locks = {name: threading.RLock() for name in self.metals}
        self.allocation_lock = threading.Lock()
        self.config.state_dir.mkdir(parents=True, exist_ok=True)

    def state_path(self, name: str) -> Path:
        return self.config.state_dir / (name + ".json")

    def state(self, name: str) -> LeaseState | None:
        path = self.state_path(name)
        return LeaseState(**json.loads(path.read_text())) if path.exists() else None

    def save(self, name: str, state: LeaseState) -> None:
        path = self.state_path(name)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(asdict(state), stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def allocate(self, metal: MetalConfig) -> tuple[LeaseState, dict]:
        """Reserve capacity serially across slots, including simultaneous operator calls."""
        with self.allocation_lock:
            return self.allocate_reserved(metal)

    def allocate_reserved(self, metal: MetalConfig) -> tuple[LeaseState, dict]:
        """Reuse an active lease; replace only a provider-confirmed ended allocation."""
        state = self.state(metal.name)
        if state and state.lease_id:
            row = self.cli.status(state.lease_id)
            if row["status"] == "failed":
                if not self.cli.down(state.lease_id):
                    raise RuntimeError("failed old lease has not been confirmed released")
            elif row["status"] != "released":
                return state, row
            state = None
        rows = self.cli.run("gpu", "list")["gpus"]
        active = [row for row in rows if row["status"] != "released"]
        if state is not None:
            matching = [row for row in active if row.get("label") == state.label]
            if matching:
                if len(matching) != 1:
                    raise RuntimeError("provider returned duplicate active allocations for one label")
                state.lease_id = matching[0]["id"]
                self.save(metal.name, state)
                return state, matching[0]
        image = self.cli.run("image", "status", metal.image)
        if image["status"] != "ready":
            raise RuntimeError("build the configured compatible image before taking a GPU")
        occupied = sum(int(row.get("gpu_count") or 1) for row in active)
        labels = {row.get("label") for row in active}
        # A lost launch response can precede provider visibility. Unresolved
        # other slots reserve their GPUs until explicitly reconciled.
        for other in self.config.metals:
            pending = self.state(other.name)
            if (other.name != metal.name and pending and not pending.lease_id
                    and pending.label not in labels):
                occupied += other.devices
        if occupied + metal.devices > self.config.gpu_ceiling:
            raise RuntimeError("the account GPU inventory would exceed gpu_ceiling")
        if state is None:
            state = LeaseState(f"rlstack-{metal.name}-{uuid.uuid4().hex[:12]}")
            self.save(metal.name, state)
        # The label is persisted before requesting. After an uncertain CLI
        # timeout, a later explicit boot reuses it rather than buying twice.
        row = self.cli.run("gpu", "up", "--gpu", metal.gpu,
                           "--count", str(metal.devices), "--hours", str(metal.hours),
                           "--image", metal.image, "--label", state.label, "--no-wait")
        state.lease_id = row["id"]
        self.save(metal.name, state)
        return state, row

    def open_forward(self, metal: MetalConfig, lease_id: str) -> list[str]:
        """Startup reconnection and an operator boot share one owned tunnel."""
        with self.forward_locks[metal.name]:
            return self.open_forward_owned(metal, lease_id)

    def open_forward_owned(self, metal: MetalConfig, lease_id: str) -> list[str]:
        if self.forward_leases.get(metal.name) != lease_id:
            self.close_forward(metal.name)
        old = self.processes.get(metal.name)
        args = self.cli.run("gpu", "forward", lease_id,
                            f"{metal.local_port}:{self.config.worker_port}",
                            "--print-command")["argv"]
        if old is None or old.poll() is not None:
            configure_reverse_forwarding(args, self.config.gateway_port)
            process = subprocess.Popen(ssh_forward_argv(args, self.config.gateway_port),
                                       stdin=subprocess.DEVNULL)
            self.processes[metal.name] = process
            self.forward_leases[metal.name] = lease_id
        return args

    def boot(self, name: str) -> None:
        """Start one explicit slot; a timeout never starts a second worker process."""
        metal = self.metals[name]
        with self.locks[name]:
            state, row = self.allocate(metal)
            if row["status"] != "ready":
                row = self.cli.run("gpu", "status", state.lease_id, "--wait",
                                   "--timeout", "900", timeout=930)
            # Ready can precede publication of SSH metadata by minutes on a
            # fresh pod (measured: two boots in a row missed a 30 s wait). A
            # null value is pending; an explicit available=false remains an
            # unsupported lease. The lease is saved already, so a timeout
            # here never leaks it: the next boot reuses it.
            deadline = time.monotonic() + SSH_METADATA_WAIT_S
            while row["status"] == "ready" and row.get("ssh") is None and time.monotonic() < deadline:
                row = self.cli.run("gpu", "status", state.lease_id, timeout=15)
                if row.get("ssh") is None:
                    time.sleep(2.0)
            if row["status"] != "ready" or not (row.get("ssh") or {}).get("available"):
                raise RuntimeError(f"lease {state.lease_id} needs a ready SSH-capable pod")
            forward = self.open_forward(metal, state.lease_id)
            if state.boot_started:
                # Reattach an existing process, never repeat nohup after an
                # uncertain exec outcome. Killing the old lease is explicit.
                self.wait_worker(metal)
                return
            self.bootstrap(metal, state, row, forward)
            self.wait_worker(metal)

    def bootstrap(self, metal: MetalConfig, state: LeaseState,
                  lease: dict, forward: list[str]) -> None:
        from rlstack.data.stores.strangeloop import ScratchClient, credential_file

        from rlstack.runner.transports.http_blobs import HttpLimits

        http_limits = HttpLimits.from_environment()
        client = ScratchClient.from_locator(self.config.store)
        bundle = source_bundle(self.source_root)
        digest = hashlib.sha256(bundle).hexdigest()
        prefix = "/tmp/rlstack-" + state.lease_id
        upload_over_ssh(forward, prefix + ".zip", bundle)
        # The scratch token lives in a file of its own on the pod, not in
        # the worker's environment: every process there resolves it from
        # SL_API_TOKEN_FILE and asks the file again on a 401, so `reauth`
        # can renew a rotated token under running runs (2026-09-17).
        upload_over_ssh(forward, credentials_path(state.lease_id),
                        credential_file(client.credentials))
        worker = {"name": metal.name, "store": self.config.store,
                  "desk": self.config.endpoint, "address": self.config.address(metal.name),
                  "port": self.config.worker_port, "idle_s": metal.idle_s,
                  "lease_id": state.lease_id, "artifact_dir": lease["artifact_dir"],
                  "builds": None if metal.builds is None else metal.builds.row(),
                  "environment": {**(metal.environment or {}),
                                  **metal.hashed_reads().environment(),
                                  "SL_API_TOKEN_FILE": credentials_path(state.lease_id),
                                  "SL_API_BASE": client.credentials.api_base,
                                  "RLSTACK_HTTP_TOKEN": self.token,
                                  "RLSTACK_HTTP_MAX_BLOB_BYTES": str(http_limits.max_blob_bytes),
                                  "RLSTACK_HTTP_SPOOL_BYTES": str(http_limits.spool_bytes)},
                  "source_sha256": digest}
        upload_over_ssh(forward, prefix + ".json", json.dumps(worker).encode())
        source = prefix + "-source"
        # Verify interpreter before backgrounding anything. Default runner
        # Python 3.10 is insufficient; image builds install this interpreter.
        verify = "import sys; assert sys.version_info >= (3, 11), sys.version"
        subprocess.run(ssh_command_argv(forward, shlex.join([metal.python, "-c", verify])),
                       capture_output=True, check=True, timeout=30)
        extract = "import zipfile; zipfile.ZipFile(" + repr(prefix + ".zip") + ").extractall(" + repr(source) + ")"
        subprocess.run(ssh_command_argv(forward, shlex.join([metal.python, "-c", extract])),
                       capture_output=True, check=True, timeout=90)
        launch = [metal.python, "-m", "rlstack.runner.venues.strangeloop.worker", "--config", prefix + ".json"]
        # artifact_dir contains the literal $PERSIST_DIR returned by SL. Resolve
        # that one platform variable remotely; never interpolate arbitrary shell.
        shell = ("cd " + shlex.quote(source) + "; "
                 "mkdir -p \"$PERSIST_DIR/runs/" + state.lease_id + "\"; "
                 "nohup " + shlex.join(launch) + " </dev/null >\"$PERSIST_DIR/runs/" +
                 state.lease_id + "/rlstack-service.log\" 2>&1 &")
        state.boot_started = True
        self.save(metal.name, state)
        self.cli.run("gpu", "exec", state.lease_id, "--timeout", "30", "--",
                     "bash", "-c", shell, timeout=60)

    def wait_worker(self, metal: MetalConfig, timeout: float = 120.0) -> None:
        """Both SSH directions and storage initialization must work before registration."""
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            process = self.processes[metal.name]
            if process.poll() is not None:
                raise RuntimeError("SSH forwarding exited; reverse gateway support is required")
            try:
                state = self.state(metal.name)
                healthy = health(f"http://127.0.0.1:{metal.local_port}", token=self.token)
                if healthy and state and self.registered is not None:
                    from rlstack.runner.transports.http import HttpTransport

                    transport = HttpTransport(f"http://127.0.0.1:{metal.local_port}", token=self.token)
                    described = asyncio.run(transport.ask("describe", {}, deadline_s=5))
                    if self.registered(metal.name, state.lease_id, described["epoch"]):
                        return
            except (OSError, ValueError, Unreachable) as exc:
                last = exc
            time.sleep(0.25)
        raise RuntimeError(f"worker health unavailable; lease remains allocated: {last}")

    def reauth(self, name: str) -> dict:
        """THE TOKEN ROTATED: write the profile's current credentials to the
        pod's token file, which every process there re-reads on its next
        401 (2026-09-17: a browser session's token expired ten hours in,
        the pods' baked-in copies with it, and every run died at its next
        store call). The lease, the worker and its runs stay as they are —
        a run paused in a store call resumes where it waits."""
        from rlstack.data.stores.strangeloop import ScratchClient, credential_file

        metal = self.metals[name]
        state = self.state(name)
        if state is None or not state.lease_id or not state.boot_started:
            raise RuntimeError("a booted worker is required before its credentials can be renewed")
        if self.cli.status(state.lease_id)["status"] != "ready":
            raise RuntimeError("the lease is no longer ready")
        client = ScratchClient.from_locator(self.config.store)
        forward = self.open_forward(metal, state.lease_id)
        upload_over_ssh(forward, credentials_path(state.lease_id),
                        credential_file(client.credentials))
        return {"metal": name, "lease_id": state.lease_id, "reauthorized": True}

    def export(self, name: str, run_ref: str, objective: str) -> dict:
        """Declare a run, then start one independent committed-ledger exporter."""
        from rlstack.data.stores.strangeloop import ScratchClient

        metal = self.metals[name]
        state = self.state(name)
        if state is None or not state.lease_id or not state.boot_started:
            raise RuntimeError("a running worker is required for its offline W&B export")
        lease = self.cli.status(state.lease_id)
        if lease["status"] != "ready":
            raise RuntimeError("the run's lease is no longer ready")
        export_id = hashlib.sha256(run_ref.encode()).hexdigest()[:16]
        prefix = "/tmp/rlstack-export-" + state.lease_id + "-" + export_id
        ready = prefix + "-ready.json"
        record = self.config.state_dir / ("export-" + state.lease_id + "-" + export_id + ".json")
        forward = self.cli.run("gpu", "forward", state.lease_id,
                               f"{metal.local_port}:{self.config.worker_port}",
                               "--print-command")["argv"]
        self.cli.run("gpu", "declare", state.lease_id, "--name", run_ref,
                     "--description", "rlstack committed ledger; objective=" + objective)
        if record.exists():
            prior = json.loads(record.read_text())
            ready = prior["ready_file"]
            read = subprocess.run(ssh_command_argv(forward, "cat " + shlex.quote(ready)),
                                  capture_output=True, text=True, timeout=20)
            # A new explicit export request may replace a terminal failed
            # attempt. Missing readiness is uncertainty, never permission to
            # launch another worker. Keep the old record, log and bundle.
            previous_state = json.loads(read.stdout).get("state") if read.returncode == 0 else None
            changed_objective = prior["objective"] != objective
            if changed_objective and previous_state not in ("failed", "finished"):
                raise RuntimeError("stop the existing exporter before changing its objective; "
                                   "its state is not terminal")
            if previous_state == "failed" or (previous_state == "finished" and changed_objective):
                attempt = uuid.uuid4().hex[:12]
                record.rename(record.with_name(record.stem + "-" + previous_state + "-" + attempt + ".json"))
                prefix += "-" + attempt
                ready = prefix + "-ready.json"
        if not record.exists():
            client = ScratchClient.from_locator(self.config.store)
            row = {"store": self.config.store, "run_ref": run_ref,
                   "objective": objective, "lease_id": state.lease_id,
                   "directory": "$PERSIST_DIR/runs/" + state.lease_id + "/rlstack-wandb/" + export_id,
                   "ready_file": ready,
                   "environment": {"SL_API_TOKEN_FILE": credentials_path(state.lease_id),
                                   "SL_API_BASE": client.credentials.api_base}}
            upload_over_ssh(forward, prefix + ".json", json.dumps(row).encode())
            # Retain uncertainty rather than spawning a second exporter after
            # a lost exec reply. A later call only reads its existing ready file.
            with record.open("x") as stream:
                json.dump({"run_ref": run_ref, "objective": objective, "ready_file": ready}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            source = "/tmp/rlstack-" + state.lease_id + "-source"
            command = [metal.python, "-m", "rlstack.runner.exporters.wandb", "--config", prefix + ".json"]
            shell = ("cd " + shlex.quote(source) + "; nohup " + shlex.join(command) +
                     " </dev/null >" + shlex.quote(prefix + ".log") + " 2>&1 &")
            self.cli.run("gpu", "exec", state.lease_id, "--timeout", "30", "--",
                         "bash", "-c", shell, timeout=60)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            read = subprocess.run(ssh_command_argv(forward, "cat " + shlex.quote(ready)),
                                  capture_output=True, text=True, timeout=20)
            if read.returncode == 0:
                output = json.loads(read.stdout)
                if output.get("state") == "failed":
                    raise RuntimeError("offline W&B exporter failed: " + output.get("error", "unknown"))
                self.cli.run("gpu", "declare", state.lease_id, "--name", run_ref,
                             "--wandb-path", output["wandb_path"])
                return output
            time.sleep(1)
        raise RuntimeError("exporter did not publish its path; declared run needs attention: " + run_ref)

    async def terminate(self, lease_id: str) -> bool:
        ended = await asyncio.to_thread(self.cli.down, lease_id)
        if ended:
            for name in self.metals:
                state = self.state(name)
                if state and state.lease_id == lease_id:
                    await asyncio.to_thread(self.close_forward, name)
        return ended

    def close_forward(self, name: str) -> None:
        with self.forward_locks[name]:
            self.close_forward_owned(name)

    def close_forward_owned(self, name: str) -> None:
        process = self.processes.pop(name, None)
        self.forward_leases.pop(name, None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    def close(self) -> None:
        """Closing routes does not claim remote services or leases have stopped."""
        for name in list(self.processes):
            self.close_forward(name)
