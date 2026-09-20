"""THE STRANGE LOOP DESK, HOSTED: the laptop's standing process in one Modal container.

    modal secret create --force strangeloop-token SL_API_TOKEN="$(python3 -c \\
        'import tomllib, pathlib; print(tomllib.load(open(pathlib.Path.home() \\
        / ".config/strangeloop/config.toml", "rb"))["profile"]["default"]["token"])')"
                            # the profile's token, never a placeholder by hand
    modal volume put rlstack-strangeloop-desk-state ~/.config/strangeloop/ssh strangeloop/ssh
    modal volume put rlstack-strangeloop-desk-state \\
        ~/.local/state/rlstack-strangeloop desk                       # the leases it reconnects
    modal deploy deploy/strangeloop_desk.py                                  # prints the two URLs
    modal volume get rlstack-strangeloop-desk-state desk/http-token -        # the operator's bearer

The process is `serve(config, source_root)` — the same one
`python -m deploy.strangeloop --config ... serve` runs on the laptop — on a
thread and event loop of its own (`DeskThread`: Modal's user loop runs only
while a hook or an input runs), and nothing about placement, leases or
addresses changes: the desk still opens
one outbound ssh per lease, the pod still reaches the gateway at its own
loopback through that ssh's reverse route, and every recorded address is
still `http://127.0.0.1:<gateway_port>#<metal>/<host>@<epoch>`. What the
container supplies is what the laptop supplied: outbound ssh
(`openssh-client`), the CLI (the published 0.8.1 wheel — the package is not
on PyPI), its token (`SL_API_TOKEN` from the secret, which the CLI honours
on every command, so no `login` runs here), the repo files (`source_bundle`
reads them under /root), and a state directory that outlives the container:
the volume, `/state/desk` for the lease files, the runtime token and the
lock, `/state/strangeloop` for the CLI's own config dir — its machine key,
which every pod it boots trusts, and which the laptop's pods trust only if
the laptop's key is seeded there.

Two things reach in from outside: the gateway and the observer, each a Modal
web endpoint over TLS. The gateway checks its bearer as it always did; the
observer is guarded by the same token (`observer_guard`: `?token=` once in a
browser). The operator's verbs stay on the laptop, pointed at the gateway by
the config's `operator_endpoint`, carrying the token in `RLSTACK_HTTP_TOKEN`
(examples/strangeloop-desk.md, "The desk hosted").

ONE OWNER PER JOURNAL (ADR 0015): stop the laptop desk before the hosted one
starts against the same scratch prefix. The lock on the volume is stale-safe
(`own_local_desk`): a container that died leaves a record the next takes.
"""
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import modal

from modal_venue import require_workspace

from rlstack.runner.venues.strangeloop.desk import serve
from rlstack.runner.venues.strangeloop.provider import DeskConfig

require_workspace()

APP = "rlstack-strangeloop-desk"
CONFIG = os.environ.get("RLSTACK_DESK_CONFIG", "examples/strangeloop-hosted.json")
"""Which desk config the container serves: a repo-relative path, read at
deploy time and shipped into the image at the same path under /root."""

STATE_MOUNT = "/state"
STATE_DIR = Path(STATE_MOUNT) / "desk"
CLI_CONFIG_DIR = Path(STATE_MOUNT) / "strangeloop"
SOURCE_ROOT = Path("/root")
ROOT = SOURCE_ROOT if not modal.is_local() else Path(__file__).resolve().parents[1]
"""The repo tree: the checkout on the laptop at deploy time, /root in the
container (the two package trees and the config are copied there)."""

CLI_WHEEL = "https://cli.strangeloopresearch.com/strangeloop_cli-0.8.1-py3-none-any.whl"
GATEWAY_LABEL = "rlstack-sl-desk-gateway"
OBSERVER_LABEL = "rlstack-sl-desk-observer"
STARTUP_TIMEOUT_S = 900.0
"""How long the ingress waits for the ports. The desk reconnects every saved
lease BEFORE its servers start (an ssh to each pod, a token renewal), and a
pod whose SSH metadata lags can hold that for minutes (SSH_METADATA_WAIT_S)."""

app = modal.App(APP)
state_volume = modal.Volume.from_name("rlstack-strangeloop-desk-state", create_if_missing=True)
token_secret = modal.Secret.from_name("strangeloop-token", required_keys=["SL_API_TOKEN"])

BYTECODE = ["**/__pycache__", "*.pyc"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("openssh-client")
    .pip_install("safetensors", "numpy", CLI_WHEEL)
    .env({"STRANGELOOP_CONFIG_DIR": str(CLI_CONFIG_DIR), "HOME": "/root",
          "RLSTACK_DESK_CONFIG": CONFIG})
    # the whole package trees, not only .py: `source_bundle` ships every file
    # under rlstack/ and rlstack_engine/ to a pod (the observer's assets among
    # them), so the container must hold the same tree the laptop does
    .add_local_dir(ROOT / "rlstack", str(SOURCE_ROOT / "rlstack"), copy=True, ignore=BYTECODE)
    .add_local_dir(ROOT / "rlstack_engine", str(SOURCE_ROOT / "rlstack_engine"), copy=True,
                   ignore=BYTECODE)
    .add_local_file(ROOT / CONFIG, str(SOURCE_ROOT / CONFIG), copy=True)
    # the one mount-only step comes LAST: Modal refuses a `copy=True` step
    # after an `add_local_*` mount (found on the first deploy, 2026-09-17)
    .add_local_python_source("modal_venue")
)


def hosted_config(root: Path) -> DeskConfig:
    """The config file as the container serves it: the file's store, metals
    and ports, with the state dir on the volume instead of the laptop's."""
    return DeskConfig.read(root / CONFIG, state_dir=STATE_DIR)


config = hosted_config(ROOT)


class DeskThread:
    """THE DESK RUNS ON A LOOP OF ITS OWN. Modal's container entrypoint drives
    its user event loop only while a hook or an input runs, and right after
    the enter hook its main thread blocks in the web server's port check —
    so a task created on that loop gets one step and freezes at its first
    await (the first deploy, 2026-09-17: py-spy showed the main thread in
    `wait_for_web_server`, the CLI's first `status` answered on an idle
    executor thread, and no frame of `serve` anywhere). `serve` therefore
    runs under `asyncio.run` on a daemon thread of its own; `bring_down`
    sets its stop event through that loop. A desk that stops says why in
    the container log; the ingress then fails its port check or its next
    request, which is how the operator learns. Nothing restarts it in
    place: a lease's tunnels belong to the process that opened them, and
    the next container reconnects them."""

    def __init__(self, serving, *args, **kwargs) -> None:
        self.serving, self.args, self.kwargs = serving, args, kwargs
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop: asyncio.Event | None = None
        self.thread = threading.Thread(target=self.run, name="strangeloop-desk", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def run(self) -> None:
        asyncio.run(self.main())

    async def main(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.stop = asyncio.Event()
        try:
            await self.serving(*self.args, **self.kwargs, stop=self.stop)
        except BaseException as error:
            print(f"desk: stopped: {error!r}", flush=True)
            raise
        print("desk: stopped", flush=True)

    def request_stop(self, timeout: float = 60.0) -> None:
        """Set the desk's stop event on the desk's own loop and wait for the
        thread; a desk that never started its loop has nothing to stop."""
        if self.loop is not None and self.stop is not None:
            self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(timeout=timeout)


DESK_CPU = float(os.environ.get("RLSTACK_DESK_CPU", "4"))
DESK_MEMORY_MB = int(os.environ.get("RLSTACK_DESK_MEMORY_MB", "8192"))
"""WHAT THE STANDING DESK RUNS ON (2026-09-19). With no request it got Modal's
default sliver of a core on PREEMPTIBLE capacity: with eighteen hosts listed,
placement's probes timed out against every metal at once, stop verbs outlived
the ingress, and Modal preempted the container twice in one night — each
restart a fresh idle clock over live hosts. A desk is small but it is the one
process everything else reaches through: it asks for cores and is not
preempted."""


@app.cls(image=image, volumes={STATE_MOUNT: state_volume}, secrets=[token_secret],
         timeout=24 * 3600, min_containers=1, max_containers=1, scaledown_window=1200,
         cpu=DESK_CPU, memory=DESK_MEMORY_MB, nonpreemptible=True)
@modal.concurrent(max_inputs=32)
class HostedDesk:
    """The laptop's `serve`, kept standing: one container, one journal owner,
    the process on a thread of its own from enter to exit (`DeskThread`)."""

    @modal.enter()
    def bring_up(self) -> None:
        self.desk = DeskThread(serve, config, ROOT, exposed=True)
        self.desk.start()

    @modal.web_server(config.gateway_port, startup_timeout=STARTUP_TIMEOUT_S, label=GATEWAY_LABEL)
    def gateway(self) -> None:
        """The desk's door at the config's gateway port; the bearer is the
        runtime token on the volume (`desk/http-token`)."""

    @modal.web_server(config.observer_port, startup_timeout=STARTUP_TIMEOUT_S,
                      label=OBSERVER_LABEL)
    def observer(self) -> None:
        """The read-only observer, guarded by the same token."""

    @modal.exit()
    def bring_down(self) -> None:
        self.desk.request_stop(timeout=60.0)
