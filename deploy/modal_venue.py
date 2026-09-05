"""The Modal venue chassis: every line the three venue files used to copy.

A VENUE OWNS THREE THINGS AND NOTHING ELSE (I5): its app, its images and its
cards; its constants; and its science — the specs, the plans, the
measurements. Everything between those and rlstack — the metal container's
bring-up and announce and keepalive, the door that follows a run to its
extent, the desk handle, the release — is the same code in every venue, and
it lives here once (ADR 0007).

WHAT IS NOT HERE. No `Transport` class: the wire substrates are
`rlstack/runner/transports/` and `transport_for` is the one factory (Q1, Q3).
No `Desk` container: there is ONE desk, `deploy/desk.py`, and every venue's
metal registers with it (Q2). No recipe: a metal boots BARE and the desk
declares what it builds (Q4) — a venue may PROPOSE one, and `metal_class`
takes it as exactly that.

NOTHING HERE IS SEMANTICS-BEARING. This module constructs specs it is handed
and never builds one; it names no loss, no adapter type, no plan.
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

DESK_APP = "rlstack-desk"
DESK_CLS = "Desk"
DESK_ADDRESS = f"modal://{DESK_APP}/{DESK_CLS}"
"""THE ONE DESK, by address (Q2). Every venue's metal registers here and every
campaign submits here, which is what makes two campaigns serialize through one
placement ladder instead of double-reading one residual (#68, stated in 2026
and built by ADR 0007)."""

STORE = "modal://rlstack-store"
STORE_MOUNT = "/store"

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
    """A metal container's OWN PLANE — carve, decarve, residual, release."""
    return f"modal://{app_name}/{cls}"


def host_address(app_name: str, host: str, cls: str = "MetalS") -> str:
    """ONE HOST inside a metal container, by the address's `#host` fragment.

    The venue is IN the address (Q3), which is what lets the one desk command
    metal deployed in this app and in the next one — the journal's `address`
    fields are self-describing and a desk rebuilt from them reaches every
    metal it ever registered."""
    return f"modal://{app_name}/{cls}#{host}"


# ---------------------------------------------------------------------------
# the desk, from a venue's side
# ---------------------------------------------------------------------------

def desk():
    """A handle on THE desk — `transport_for` reads the address and builds the
    Modal transport, the same call the desk itself makes to reach metal."""
    from rlstack.runner.remote import RemoteDesk, transport_for

    return RemoteDesk(transport_for(DESK_ADDRESS))


def metal_handle(app_name: str, cls: str = "MetalS"):
    """This venue's deployed metal class, for the one thing a transport does
    not do: SPAWN the keepalive. Booting a container is not a frame."""
    return modal.Cls.from_name(app_name, cls)()


def boot_by_spawn(app_name: str, cls: str = "MetalS"):
    """THE KNOCK, MADE EXPLICIT (ADR 0007, Q5): a `boot_for` the desk can call
    to wake a metal it released or reaped. Spawning the keepalive is exactly
    what `::up` does by hand, and handing it to the desk is what turns the
    reap -> knock -> re-register -> reroute loop from a Modal accident (a call
    to a lazy container happens to boot it) into a thing the venue said."""
    def boot(name: str):
        call = metal_handle(app_name, cls).serve.spawn()
        print(f"[knock] {name}: spawned keepalive {call.object_id}", flush=True)
        return call.object_id
    return boot


def wait_for_metal(name: str, timeout_s: float = 900.0) -> dict:
    """Block until `name` is registered with the desk AND reachable on the
    metal plane — the container's own announce is what says so, never the
    spawn's return."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        held = desk().status().get("metal", {})
        if name in held and held[name].get("plane"):
            return held[name]
        time.sleep(10)
    raise SystemExit(f"{name} did not register within {timeout_s:.0f}s")


def guarded_release(names, reason: str) -> dict:
    """THIS DOOR'S OWN METAL, handed back — and nobody else's (ADR 0007, Q6).

    Under one desk the plane may hold another experiment's metal, so a door
    releases the metals IT registered, by name, and the desk refuses even
    those while another tenancy is running on them. A refusal is reported and
    is not an error here: the door's work is done either way, and the idle
    rule will collect what the guard spared. `force` is not sent — it is the
    operator's verb at `deploy/desk.py::release`.
    """
    import asyncio

    held = desk().status().get("metal", {})
    out: dict[str, dict] = {}
    for name in sorted(names):
        if not held.get(name, {}).get("plane"):
            continue
        told = asyncio.run(desk().release(name, reason=reason))
        print(f"[release] {name}: {json.dumps(told)}", flush=True)
        out[name] = told
    return out


def mine_are_released(names) -> bool:
    """The check venues' promise, SCOPED (Q2): the metals this door acquired
    are released. Not "the plane is empty" — under one desk the plane may
    legitimately hold metal this venue never registered."""
    held = desk().status().get("metal", {})
    standing = [name for name in sorted(names)
                if held.get(name, {}).get("plane")]
    print(f"[plane] mine still standing: {standing or 'none'}", flush=True)
    return not standing


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

def submit_spec(row: dict, subdir: str, anchor: str | None = None) -> dict:
    """One canonical spec row through THE desk. The reply is the placement:
    accepted, the run id, the anchor host and every pool's address."""
    import asyncio

    from rlstack.runner.remote import spec_from_json

    reply = asyncio.run(desk().submit(spec_from_json(row), subdir=subdir,
                                      anchor=anchor))
    print(f"[submit] {json.dumps(reply, default=str)[:400]}", flush=True)
    if not reply.get("accepted"):
        raise SystemExit(f"not accepted: {reply}")
    return reply


def follow(progress_fn, run_id: str, timeout_s: float,
           every_s: float = 60.0) -> str:
    """A run followed TO ITS EXTENT — the one predicate for both kinds of run
    (ADR 0006 Part B: a generation-only run's extent is its rollout plan).
    `progress_fn` is the venue's own volume-side reader, because reading a
    ledger needs the mount and this driver has none."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        told = progress_fn.remote([run_id])[run_id]
        print(f"[{run_id[:12]}] {told['completed']}/{told['planned']} "
              f"{told['extent']} — {json.dumps(told.get('train', []))}",
              flush=True)
        if told["done"]:
            return run_id
        time.sleep(every_s)
    raise SystemExit(f"{run_id} did not finish within {timeout_s:.0f}s")


def progress_function(app, image, *, module: str, name: str = "progress",
                      tail: int = 3):
    """THE EXTENT READER, one copy: a venue's `@app.function` that reports how
    far each run has got, plus the tail of its train blocks.

    It lives on the volume because a ledger is a store read and the driver has
    no mount, and it lives HERE because `run_progress` is the ONE predicate
    for both kinds of run (ADR 0006 Part B: a generation-only run's extent is
    its rollout plan) and three venues asking it three ways was three chances
    to ask it wrong. `tail` is how many of the run's train blocks come back —
    a check venue that reports the rail per update asks for all of them."""
    def progress(run_ids: list[str]) -> dict:
        from rlstack import run_progress

        store_volume.reload()
        store = a_store()
        out = {}
        for run_id in run_ids:
            told = run_progress(store, run_id)
            entries = store.peek_ledger(run_id)
            out[run_id] = {
                "extent": told.extent, "completed": told.completed,
                "planned": told.planned, "done": told.done,
                "committed": int(entries[-1]["update"]) if entries else 0,
                "train": [dict(e.get("train", {})) for e in entries[-tail:]]}
        return out

    progress.__name__ = progress.__qualname__ = name
    progress.__module__ = module    # the container imports it from the VENUE
    return app.function(image=image, volumes={STORE_MOUNT: store_volume},
                        timeout=600)(progress)


def submit_and_follow(progress_fn, row: dict, subdir: str, timeout_s: float,
                      anchor: str | None = None) -> str:
    """A CAMPAIGN DOOR, whole: submit and follow. IT NEVER RELEASES (Q6) —
    a campaign of three arms that tore its metal down after each one would
    boot the base three times, and under one desk it would tear down whatever
    else had joined. Idle metal is the desk's to collect (ADR 0003)."""
    return follow(progress_fn, submit_spec(row, subdir, anchor)["run_id"],
                  timeout_s)


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
    """THE METAL CONTAINER, built for one venue — what every venue file used
    to hold 145 lines of.

    `app` is the venue's `modal.App` (the class must be declared on it), and
    `app_name`/`cls` are the same names as an ADDRESS, which is how the desk
    reaches back. `module` is the VENUE's `__name__`, and the venue binds the
    returned class under the name `cls`: a Modal container boots by importing
    the class's `__module__` and reading `cls` off it, so a class built here
    but left stamped `modal_venue` fails at the container's first breath
    (found on the venue: "module 'modal_venue' has no attribute 'MetalS'"). `recipe` is a `Builds` this venue PROPOSES — journaled by
    the desk as its own `recipe` event and overwritten by the desk's door
    (Q4); None is a metal that boots bare and waits to be told.

    What the class does, in the order it does it: MEASURE the card (never
    typed — ADR 0001 Q6, and the deploy may name several cards), mount the
    store, publish its own hosts on the in-process switchboard through
    `MetalService.route` (so a sibling pool is a LocalTransport and never a
    Modal self-call — #77); ANNOUNCE itself to the one desk with its idle
    limit declared; keep its hosts' stats and commit the volume; hold the
    shift open on `serve` until the desk releases it (ADR 0003, Q3); stand a
    FRESH metal up if a door reaches a container the desk already released
    (Q4); and end every resident on the way out.
    """
    def bring_up_metal():
        """This container's books and router. Bare unless the venue proposed
        a recipe — and even then the desk's row is what the carve carries."""
        from rlstack.policy.siteschema import hf_schema
        from rlstack.runner.desk import MetalService
        from rlstack.runner.remote import transport_for

        service = MetalService(
            MetalService.measure(metal), store=a_store(), builds=recipe,
            address_of=lambda host: host_address(app_name, host, cls),
            schema_for=hf_schema, transport_for=transport_for)
        print(f"[{metal}] up, bare: {service.metal.gpu} "
              f"x{service.metal.devices} at {service.metal.vram_gb:g} GB; "
              f"residual {service.residual()}", flush=True)
        return service

    async def announce(service) -> None:
        """The metal REGISTERS ITSELF the moment it exists (ADR 0001, Q5a),
        at its plane address, with its idle limit declared — and with the
        venue's recipe as a PROPOSAL when it has one (ADR 0007, Q4)."""
        from rlstack.runner.remote import RemoteDesk, transport_for

        card = service.metal
        told = await RemoteDesk(transport_for(DESK_ADDRESS)).register_metal(
            card.name, card.gpu, card.devices, card.vram_gb,
            metal_address(app_name, cls),
            builds=None if recipe is None else recipe.row(), idle_s=idle_s,
            # the container id Modal gave this metal: what the desk's release
            # terminates, so a released metal is not merely deaf but gone
            container=os.environ.get("MODAL_TASK_ID"))
        print(f"[{metal}] registered with the desk: {json.dumps(told)}",
              flush=True)

    async def metal_duties(service) -> None:
        """Announce, then follow every carved host's stats and commit the
        volume — until the desk releases this metal, at which point the
        duties end with the shift."""
        import asyncio

        try:
            await announce(service)
        except Exception as refused:
            print(f"[{metal}] REGISTRATION REFUSED: {refused}", flush=True)
        stats: dict[str, asyncio.Task] = {}
        tick = 0
        try:
            while not service.released.is_set():
                for host_service in list(service.services.values()):
                    host = host_service.host
                    if host.name not in stats:
                        stats[host.name] = asyncio.create_task(host.run_stats())
                await asyncio.sleep(30)
                tick += 1
                if tick % 2 == 0:
                    await store_volume.commit.aio()
        finally:
            for task in stats.values():
                task.cancel()
            await store_volume.commit.aio()

    class MetalS:
        @modal.enter()
        async def bring_up(self) -> None:
            self.stand_up()

        def stand_up(self) -> None:
            """A fresh metal on this container: books, router, duties."""
            import asyncio

            self.born = time.time()
            self.metal_service = bring_up_metal()
            self.duties = asyncio.create_task(metal_duties(self.metal_service))

        def live(self):
            """The metal every door answers through.

            A released service is done: books empty, duties ended, shift
            latch set. What happens to an input that reaches this container
            AFTER that depends on whether the container will keep fetching
            inputs. Where `serve` can tell Modal to stop, it is
            REFUSED, loudly: found on the venue, a keepalive spawned in the
            second between the release and the stop landed here, the old
            rule stood a fresh metal up and REGISTERED it, and the desk then
            carved for an hour into a container that fetched nothing — a
            registered, deaf metal, and every desk call queued behind it.
            A refusal fails that input by name; the next knock boots a fresh
            container. Only where the stop is unavailable does the container
            stay reachable, and only then is rebirth what a knock asks for
            (ADR 0003, Q4)."""
            if self.metal_service.released.is_set():
                if stop_fetching_inputs is not None:
                    raise RuntimeError(
                        f"[{metal}] retired: released, and this container "
                        f"stops fetching inputs — this input cannot be served "
                        f"here; the next knock boots a fresh container")
                print(f"[{metal}] reborn on a released container", flush=True)
                self.stand_up()
            return self.metal_service

        @modal.method()
        async def door(self, host: str, verb: str, payload: dict) -> dict:
            """THE ADMITTED VERBS, host-addressed. An empty `host` is this
            container's own metal plane; a name is one host inside it — the
            two doors every rlstack Modal container wears, which is why ONE
            transport class reaches all of them (ADR 0007, Q1)."""
            live = self.live()
            if not host:
                return await live.serve(verb, payload)
            return await live.service_for(
                host_address(app_name, host, cls)).serve(verb, payload)

        @modal.method()
        def door_ask(self, host: str, verb: str, payload: dict) -> dict:
            """The admission-free verbs, same addressing."""
            live = self.live()
            if not host:
                return live.answer(verb, payload)
            return live.service_for(
                host_address(app_name, host, cls)).answer(verb, payload)

        @modal.method()
        async def serve(self) -> dict:
            """THE KEEPALIVE, AS THE SHIFT (ADR 0003, Q3): the input in flight
            is what keeps this container from scaling down while its hosts
            carry work, and it RETURNS when the desk releases this metal — so
            the venue reclaims the container as a consequence of the desk's
            decision, never of its own timer (which is the backstop, set no
            shorter). After the shift the container stops taking inputs, so
            the venue reclaims it NOW rather than at the idle scaledown, and a
            later knock boots a fresh one."""
            service = self.live()
            await service.until_released()
            if stop_fetching_inputs is not None:
                stop_fetching_inputs()
            return {"released": True, "metal": metal,
                    "shift_s": round(time.time() - self.born, 1)}

        @modal.exit()
        def bring_down(self) -> None:
            self.duties.cancel()
            for teardown in self.metal_service.shutdown():
                if not teardown.graceful:
                    print(teardown.line(), flush=True)

    # DECORATED BY HAND, and named FIRST: Modal registers a class under the
    # `__name__` it carries when `app.cls` runs, and that name is the one an
    # address says (`modal://<app>/<cls>`). A `@app.cls` line above the class
    # body would freeze it as this function's local name instead.
    MetalS.__name__ = MetalS.__qualname__ = cls
    MetalS.__module__ = module      # where the container will look it up
    return app.cls(
        image=image, gpu=gpu,
        volumes={STORE_MOUNT: store_volume, "/hf": hf_cache},
        secrets=list(secrets), timeout=86400,
        scaledown_window=int(idle_s), max_containers=max_containers,
    )(modal.concurrent(max_inputs=64)(MetalS))


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
