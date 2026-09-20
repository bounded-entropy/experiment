"""Modal deployment declarations and compatibility helpers for experiment venues.

App, image, volume and card declarations stay here. The provider adapters in
`runner/venues/modal/` instantiate the shared Desk and Metal runtimes; common
submission, readiness and observer reads live in `runner/venues/client.py`.
Existing experiment files keep their imports and scientific spec builders.
"""

from __future__ import annotations

import json
import os
import time

import modal

try:                      # a released container STOPS FETCHING where it can
    from modal.experimental import stop_fetching_inputs
except ImportError:       # the idle scaledown is the backstop
    stop_fetching_inputs = None

WORKSPACE = "yu-masala-workspace"
"""THE ONE WORKSPACE (Samarth, 2026-09-05): every rlstack deploy, door and
volume lives in this Modal profile and no other. The personal profile is
retired for runs — a day's GPU went there by accident on 2026-09-04 and the
store split across two volumes. Enforced at import, below, so a `modal
deploy` or `modal run` under any other profile refuses BY NAME instead of
quietly standing a second fleet up."""


def require_workspace() -> None:
    """Refuse to run against any Modal profile but WORKSPACE.

    Reads the active profile off the SDK's own config, ON THE CLIENT ONLY:
    inside a container `modal.is_local()` is false and the SDK reports the
    profile 'default', and a check there crash-looped every container of
    the first deploy (found on the venue). Under the test stand-in
    (tests/venue_stub.py) the profile is not a string and the check stands
    down — the fakes suite deploys nothing. Set
    `MODAL_PROFILE=yu-masala-workspace` on the command; never `modal profile
    activate`, which changes the machine's default silently."""
    from modal import config as modal_config

    if not modal.is_local():
        return                # inside a container the SDK reports no profile
    profile = getattr(modal_config, "_profile", None)
    if isinstance(profile, str) and profile != WORKSPACE:
        raise SystemExit(
            f"rlstack runs in the Modal workspace {WORKSPACE!r} only; this "
            f"command is running under {profile!r}. Prefix it with "
            f"MODAL_PROFILE={WORKSPACE}.")


require_workspace()
HEARTBEAT_S = 20.0
"""THE CADENCE A METAL RENEWS ITS LEASE AT (ADR 0008, Q1) — the fallback only.
The desk owns the number and hands it back in the registration reply, because
two clocks that can drift apart is one clock too many; this is what a
container uses until it has been told."""

DESK_APP = "rlstack-desk"
DESK_CLS = "Desk"
DESK_ADDRESS = f"modal://{DESK_APP}/{DESK_CLS}"
"""THE ONE DESK, by address (Q2). Every venue's metal registers here and every
campaign submits here, which is what makes two campaigns serialize through one
placement ladder instead of double-reading one residual (#68, stated in 2026
and built by ADR 0007)."""

STORE = "modal://rlstack-store"
STORE_MOUNT = "/store"

OBSERVER = os.environ.get("RLSTACK_OBSERVER", "")
"""THE OBSERVER'S BASE URL — where a campaign door reads its run's progress
(ADR 0008, F6). `deploy/ui.py` deploys under the label `rlstack-ui`, so the
URL is `https://<workspace>--rlstack-ui.modal.run`; the workspace is not
something this file can know, so it is an environment variable and an unset
one is a loud refusal at the first poll rather than a silent fallback."""

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlstack-hf-cache", create_if_missing=True)

SOURCES = ("rlstack", "rlstack_engine", "modal_venue")
"""The local Python every rlstack container needs: the library, the engine
plugin package, and THIS module — a metal container runs `metal_class`'s own
bring-up inside the image, so the chassis ships with it."""


def cpu_image_for(*pip: str) -> modal.Image:
    """The driver/desk image: no CUDA, the store's two dependencies, and the
    sources. `pip` adds a venue's own (a corpus builder's tokenizer)."""
    return (modal.Image.debian_slim(python_version="3.12")
            .pip_install("safetensors", "numpy", *pip)
            .add_local_python_source(*SOURCES))


def gpu_image_for(env: dict | None = None,
                  with_tests: bool = False) -> modal.Image:
    """THE PINNED GPU IMAGE — vLLM, torch and transformers at the versions
    every number in this repo was measured against, plus the environment the
    engine needs to behave (no FlashInfer sampler, spawn for the multiproc
    workers, one OMP thread, expandable segments).

    `env` is the venue's own — a condition read at deploy time and carried
    into the container. `with_tests` ships `tests/` so `run_suite` can run the
    fakes suite in the image, where the torch-gated cases actually execute.
    The observer's static assets are not Python sources and the suite in the
    image asserts they ship, so they always do."""
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("vllm==0.28.0", "torch==2.13.0",
                          "transformers==5.16.1", "safetensors", "numpy")
             .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
                   "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                   "OMP_NUM_THREADS": "1",
                   "HF_HOME": "/hf",
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   **(env or {})})
             .add_local_python_source(*SOURCES)
             .add_local_dir("rlstack/observe/web",
                            remote_path="/root/rlstack/observe/web"))
    if not with_tests:
        return image
    # the suite's venue tests import deploy/*.py by path (tests/test_venues.py),
    # so the venue files ship beside the tests — found in the image: three
    # fixture errors, "No such file or directory: /root/deploy/concept_steer.py"
    return (image.add_local_dir("tests", remote_path="/root/tests")
            .add_local_dir("deploy", remote_path="/root/deploy"))


def a_store():
    """The one store on the shared volume, with the DEFAULT fleet journal.

    Every per-venue `Store` subclass that re-keyed `fleet/<venue>.jsonl` is
    gone with the per-venue desks (Q2): one desk writes one journal, which is
    I10 said plainly, and the old files stay on the volume as history nothing
    reads."""
    from rlstack import ModalVolumeStore

    return ModalVolumeStore(STORE_MOUNT, volume=store_volume, locator=STORE)


# ---------------------------------------------------------------------------
# the addresses: the grammar, filled in for this venue
# ---------------------------------------------------------------------------

def metal_address(app_name: str, cls: str = "MetalS") -> str:
    from rlstack.runner.venues.modal.provider import metal_address as address
    return address(app_name, cls)


def host_address(app_name: str, host: str, cls: str = "MetalS", epoch: str = "") -> str:
    from rlstack.runner.venues.modal.provider import host_address as address
    return address(app_name, host, cls, epoch)


# ---------------------------------------------------------------------------
# the desk, from a venue's side
# ---------------------------------------------------------------------------

def desk():
    """A handle on THE desk — `transport_for` reads the address and builds the
    Modal transport, the same call the desk itself makes to reach metal."""
    from rlstack.runner.remote import RemoteDesk, transport_for

    return RemoteDesk(transport_for(DESK_ADDRESS))


def metal_handle(app_name: str, cls: str = "MetalS"):
    from rlstack.runner.venues.modal.provider import metal_handle as handle
    return handle(app_name, cls)


def boot_by_spawn(app_name: str, cls: str = "MetalS"):
    from rlstack.runner.venues.modal.provider import boot_by_spawn as boot
    return boot(app_name, cls)


def fleet() -> dict:
    """The desk's inventory, from a driver's synchronous entrypoint.

    NEVER A CLIENT-SIDE TIMEOUT ON A DESK CALL (ADR 0008, Q4): cancelling an
    input of a concurrent container is what shut the desk down three times on
    2026-09-04, so a helper here waits patiently and the DEADLINE is the
    desk's own, server-side. `asyncio.run` drives one call to completion and
    cancels nothing."""
    import asyncio

    return asyncio.run(desk().status())


def wait_for_metal(name: str, timeout_s: float = 900.0) -> dict:
    import asyncio
    from rlstack.runner.venues.client import VenueClient
    return asyncio.run(VenueClient(desk(), OBSERVER).wait_for_metal(name, timeout_s))


def guarded_release(names, reason: str) -> dict:
    import asyncio
    from rlstack.runner.venues.client import VenueClient
    return asyncio.run(VenueClient(desk(), OBSERVER).guarded_release(names, reason))


def mine_are_released(names) -> bool:
    import asyncio
    from rlstack.runner.venues.client import VenueClient
    return asyncio.run(VenueClient(desk(), OBSERVER).mine_are_released(names))


def take_down(names, call, reason: str) -> dict:
    """A check venue's ending, whole: release my metal, watch the keepalive
    return because the DESK said so (ADR 0003's promise), and assert my
    metals are released. A campaign door never calls this — idle metal is the
    desk's to collect (Q6)."""
    verdict: dict = {"released": guarded_release(names, reason)}
    if call is not None:
        started = time.time()
        try:
            verdict["keepalive_returned"] = call.get(timeout=600)
            print(f"[shift] the keepalive returned "
                  f"{json.dumps(verdict['keepalive_returned'])} "
                  f"{time.time() - started:.1f}s after the release", flush=True)
        except Exception as still:
            verdict["keepalive_returned"] = f"NOT within 600s: {still}"
    verdict["mine_released"] = mine_are_released(names)
    print(json.dumps(verdict, indent=1, default=str), flush=True)
    if not verdict["mine_released"]:
        raise SystemExit(f"metal left standing after the door: {sorted(names)}")
    return verdict


# ---------------------------------------------------------------------------
# the campaign helpers: submit, and follow to the extent
# ---------------------------------------------------------------------------

def canonical_row(build, borrow=()):
    import asyncio
    from rlstack.runner.venues.client import VenueClient
    return asyncio.run(VenueClient(desk(), OBSERVER).canonical_row(build, borrow))


def submit_spec(row: dict, subdir: str, anchor: str | None = None,
                solo: bool = False, resume: bool = False, *,
                every: int, delivery: str = "wire") -> dict:
    """One canonical spec row through THE desk. The reply is the placement:
    accepted, the run id, the anchor host and every pool's address. `every`
    is the run's checkpoint cadence in updates and `delivery` how its policy
    reaches its pools (ADR 0014) — a door says both, nothing defaults them."""
    import asyncio

    from rlstack.runner.checkpointing import Checkpointing
    from rlstack.runner.venues.client import VenueClient

    reply = asyncio.run(VenueClient(desk(), OBSERVER).submit(
        row, subdir, anchor=anchor, solo=solo, resume=resume,
        checkpointing=Checkpointing(every=int(every), delivery=delivery)))
    print(f"[submit] {json.dumps(reply, default=str)[:400]}", flush=True)
    if not reply.get("accepted"):
        raise SystemExit(f"not accepted: {reply}")
    return reply


def progress(run_id: str, folder: str = "", tail: int = 3) -> dict:
    from rlstack.runner.venues.client import VenueClient
    return VenueClient(desk(), OBSERVER).progress(run_id, folder, tail)


def ledgers(run_ids, folder: str = "", tail: int = 8) -> dict:
    from rlstack.runner.venues.client import VenueClient
    return VenueClient(desk(), OBSERVER).ledgers(run_ids, folder, tail)


def follow(run_id: str, timeout_s: float, every_s: float = 60.0,
           folder: str = "") -> str:
    import asyncio
    from rlstack.runner.venues.client import VenueClient
    return asyncio.run(VenueClient(desk(), OBSERVER).follow(run_id, timeout_s, every_s, folder))


def submit_and_follow(row: dict, subdir: str, timeout_s: float,
                      anchor: str | None = None, *,
                      every: int, delivery: str = "wire") -> str:
    import asyncio
    from rlstack.runner.checkpointing import Checkpointing
    from rlstack.runner.venues.client import VenueClient
    return asyncio.run(VenueClient(desk(), OBSERVER).submit_and_follow(
        row, subdir, timeout_s, anchor,
        checkpointing=Checkpointing(every=int(every), delivery=delivery)))


def export_blob(store, run_id: str, blob: str, key: str) -> dict:
    """One of a run's blobs copied out to a plain volume path, for a harness
    that is not rlstack. Bytes in, bytes out; the run is not touched."""
    data = store._read(f"{store.run_prefix(run_id)}/{blob}")
    store._write(key, data)
    store_volume.commit()
    return {"run_id": run_id, "bytes": len(data), "from": blob,
            "path": f"{STORE_MOUNT}/{key}", "uri": f"{STORE}/{key}"}


# ---------------------------------------------------------------------------
# the metal container: one class, built for a venue
# ---------------------------------------------------------------------------

def metal_class(app, app_name: str, metal: str, gpu, image, *, module: str,
                idle_s: float = 1800.0, recipe=None, cls: str = "MetalS",
                secrets=(), max_containers: int = 1):
    """Bind this deployment's resources to the shared Modal worker adapter."""
    from rlstack.runner.venues.modal.worker import metal_class as worker_class

    return worker_class(
        app, app_name, metal, gpu, image, module=module, store_for=a_store,
        desk_address=DESK_ADDRESS, volumes={STORE_MOUNT: store_volume, "/hf": hf_cache},
        idle_s=idle_s, recipe=recipe, cls=cls, secrets=secrets,
        max_containers=max_containers, heartbeat_s=HEARTBEAT_S,
        stop_fetching=stop_fetching_inputs)


def smoke_function(app, image, *, module: str, name: str = "smoke",
                   imports=(), exercise=None):
    """THE IMAGE, EXERCISED BEFORE ANY METAL IS BOOKED (ADR 0008, F5).

    A build is a declaration until something runs in it. On 2026-09-04 that
    cost three deploys: the corpus image was a pip list that turned out to be
    missing jinja2, and `apply_chat_template` found out at first use; ADR
    0007's observer rewrite was "reasoned, not measured" and every container
    died at construction while the old one served a week-old view. Both would
    have taken seconds to catch INSIDE the image, on no GPU.

    So a venue wires one of these and RUNS IT BEFORE `modal deploy`: it
    imports the modules its science actually needs and, where the venue gives
    one, `exercise` builds its specs — the same client-side path a submit
    takes — against a temporary store. No metal, no volume, no desk: just the
    image, asked whether it can do the thing it was built for.

    `imports` are dotted module names; `exercise` is a callable returning
    anything JSON-safe. A failure raises INSIDE the container, which is the
    whole point — a smoke that passes is not a promise the venue works, only
    that its image is not missing a dependency or a file."""
    def smoke() -> dict:
        import importlib

        report: dict = {"imported": []}
        for dotted in imports:
            importlib.import_module(dotted)
            report["imported"].append(dotted)
        if exercise is not None:
            report["exercised"] = exercise()
        print(json.dumps(report, indent=1, default=str), flush=True)
        return report

    smoke.__name__ = smoke.__qualname__ = name
    smoke.__module__ = module       # the container imports it from the VENUE
    return app.function(image=image, timeout=900)(smoke)


def run_suite() -> str:
    """The fakes suite inside a venue's GPU image, where the torch-gated cases
    actually run. A venue wires this to a `@app.function` of its own."""
    import subprocess

    out = subprocess.run(["python", "-m", "unittest", "discover", "-s", "tests"],
                         cwd="/root", capture_output=True, text=True)
    tail = out.stderr[-4000:]
    print(tail)
    if out.returncode != 0:
        raise SystemExit("the suite is red in the image")
    return tail
