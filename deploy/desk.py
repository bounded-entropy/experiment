"""THE DESK: one standing service, one journal, every venue's metal.

    modal deploy deploy/desk.py                       # the desk, standing
    modal run deploy/desk.py::status                  # every listing and every metal
    modal run deploy/desk.py::recipe --metal concept-a100 \
        --engine max_model_len=4096,serves=steer,enforce_eager,enable_sleep_mode \
        --learner checkpoint_activations                # what that metal builds
    modal run deploy/desk.py::release --metal steer-l4  # hand one back (guarded)
    modal run deploy/desk.py::release --metal steer-l4 --force
    modal run deploy/desk.py::sweep                   # release every metal it holds
    modal run deploy/desk.py::reap                    # probe the listings, reap the dead
                                                      #   (the `reaper` cron does this every 15 min)

WHY ONE (ADR 0007, Q2). Every desk-shaped venue used to stand up its own Desk
container and its own fleet journal, so three fleets shared one volume without
seeing each other: a campaign could only place onto metal that had registered
with ITS app, and #68's own design statement — "concurrent campaigns serialize
through one desk instead of double-reading one residual" — was recorded in
2026 and never built. It is built here. Metal in any venue's app registers at
this address; a campaign in any venue submits to it; the placement ladder,
the residual and the idle clock are ONE.

Addresses are what make that possible: `modal://<app>/<cls>[#<host>]` carries
the venue, so `transport_for` reaches metal in an app this container has never
heard of (Q3). The old per-venue journals stay on the volume as history and
are never read.

WHAT THIS DESK DECLARES. Since ADR 0007 the RECIPE — what a metal builds when
it is carved on — is the desk's, not the container's: a metal boots bare and
the `recipe` door below says what it is for, journaled, replayed after a
kill -9, and carried by every carve. A venue may propose one at registration;
the last word said is the one that rides.
"""

from __future__ import annotations

import json
import os

import modal

from modal_venue import (
    DESK_APP, a_store, boot_by_spawn, cpu_image_for, desk, smoke_function, store_volume,
)

app = modal.App(DESK_APP)

cpu_image = cpu_image_for()

BOOTABLE_METALS = (None if "RLSTACK_BOOTABLE_METALS" not in os.environ else
                   frozenset(name.strip() for name in
                             os.environ["RLSTACK_BOOTABLE_METALS"].split(",")
                             if name.strip()))
"""Optional comma-separated allowlist for automatic acquisition.
Unset uses the desk's inventory; an empty value disables automatic boots.
Allocation choices belong to deployment configuration, not the framework.
"""

RECOVERY_GENERATION = os.environ.get("RLSTACK_RECOVERY_GENERATION", "")
"""Optional recovery generation, held stable across service restarts.
When set, only explicit submissions in this generation recover automatically.
An empty value retains the library's recovery behavior.
"""

IDLE_S = 90.0
"""THE FLEET'S CLOCK (ADR 0003): metal nothing has been busy on for this long
is released. A metal may declare its own at registration, and None there pins
it forever. NINETY SECONDS since 2026-09-05 (Samarth: "if no process is
running on a metal, it literally stops after a minute or smth") — busy is
a running tenancy, work in flight, or admitted traffic that moved
(`listing_busy`), so a run mid-load is never idle; only truly empty metal
is, and it goes within IDLE_TICK_S of its limit, not on the reaper's cron."""

IDLE_TICK_S = 30.0
"""How often the standing desk reads its own idle clock — inside the
container, on its loop, because a ninety-second limit read every fifteen
minutes by the reaper cron would still leave metal standing a quarter hour."""


async def terminate_container(container_id: str) -> bool:
    """THE HAND THAT ENDS A METAL'S CONTAINER: what `modal container stop`
    does, from inside the desk. A released metal stops taking inputs, but a
    container that takes no inputs still stands (billed) until the venue's
    scaledown, and one whose residents wedged never heard the release at all
    — so the desk terminates it by the id the metal registered with. Already
    finished is success."""
    from modal.client import _Client
    from modal_proto import api_pb2

    client = await _Client.from_env()
    info = await client.stub.TaskGetInfo(
        api_pb2.TaskGetInfoRequest(task_id=container_id))
    if info.info.finished_at:
        return True
    await client.stub.ContainerStop(
        api_pb2.ContainerStopRequest(task_id=container_id, graceful=False))
    print(f"[desk] terminated container {container_id}", flush=True)
    return True


@app.cls(image=cpu_image, volumes={"/store": store_volume},
         timeout=3600, min_containers=1, max_containers=1,
         scaledown_window=1200)
@modal.concurrent(max_inputs=32)
class Desk:
    """The one container. `min_containers=1` because the desk is a STANDING
    service — a metal announcing itself must find someone home — and
    `max_containers=1` because two desks replaying one journal would each
    believe they own the fleet."""

    @modal.enter()
    def bring_up(self) -> None:
        from rlstack.runner.campaign import Campaigns
        from rlstack.runner.desk import Desk as TheDesk
        from rlstack.runner.remote import (
            RemoteHost, RemoteMetal, transport_for,
        )

        self.desk = TheDesk.from_journal(
            a_store(),
            host_for=lambda address: RemoteHost(transport_for(address)),
            metal_for=lambda address: RemoteMetal(transport_for(address)),
            boot_for=self.boot,
            bootable_metals=BOOTABLE_METALS,
            recovery_generation=RECOVERY_GENERATION,
            idle_s=IDLE_S,
            # Busy hosts can answer slowly. Keep probes bounded, concurrent,
            # and outside placement locks; an expired read remains unknown.
            probe_deadline_s=90.0,
            residual_deadline_s=30.0,
            terminate_for=terminate_container)
        # This deployment never pins GPUs: restore finite automatic shutdown
        # even when its historical journal contains a manual infinite limit.
        # Changing the existing policy here does not register metal or retry
        # parked work, and finite venue-specific limits stay unchanged.
        for name in self.desk.metal:
            if self.desk.idle_limit(name) is None:
                self.desk.declare_idle(name, 300.0)
        self.desk.require_finite_idle = True
        self.campaigns = Campaigns(self.desk)
        self.idle_ticker = None      # started by the first async door: enter runs with no loop
        print(f"[desk] rebuilt from journal: {sorted(self.desk.listings)} "
              f"/ metal plane: {sorted(self.desk.metal_remotes)} "
              f"/ released: {sorted(self.desk.released)} "
              f"/ recipes: {sorted(self.desk.metal_builds)}", flush=True)

    def ensure_idle_ticker(self) -> None:
        """The idle ticker starts on the first ASYNC door call, because
        `bring_up` is a synchronous enter with no running loop (found on the
        venue: a create_task there crashed every desk container)."""
        import asyncio

        if self.idle_ticker is None or self.idle_ticker.done():
            self.idle_ticker = asyncio.create_task(self.tick_idle())

    async def tick_idle(self) -> None:
        """The idle clock, read every IDLE_TICK_S: observe every metal's
        listings, then release what has sat past its limit. The reaper cron
        still does the same on its own tick (and reaps the dead); this is
        what makes a ninety-second limit mean ninety seconds."""
        import asyncio
        import time

        while True:
            await asyncio.sleep(IDLE_TICK_S)
            try:
                now = time.time()
                await self.desk.observe_idle(now)
                released = await self.desk.release_idle(now)
                if released:
                    print(f"[desk] idle: released {released}", flush=True)
            except Exception as refused:
                print(f"[desk] idle tick: {refused}", flush=True)

    def boot(self, name: str):
        """THE KNOCK (ADR 0007, Q5): the metal's OWN app is in the address
        this desk journaled, so the desk can spawn that app's keepalive
        without knowing the venue — which is what makes the knock explicit
        instead of an accident of Modal's lazy boot."""
        from rlstack.runner.remote import parse_address

        parsed = parse_address(self.desk.metal_addresses.get(name) or "")
        return boot_by_spawn(parsed.app, parsed.cls)(name)

    @modal.method()
    async def door(self, host: str, verb: str, payload: dict) -> dict:
        """The admitted verbs, through the spec-aware sidecar. `host` is
        ignored — a desk IS its own plane — and present because every rlstack
        Modal container wears the SAME two doors, which is why one transport
        class reaches all of them (ADR 0007, Q1)."""
        self.ensure_idle_ticker()
        return await self.campaigns.serve(verb, payload)

    @modal.method()
    async def door_ask(self, host: str, verb: str, payload: dict) -> dict:
        """The admission-free verbs (status, placements) — ASYNC like its
        twin since ADR 0008 (Q4): a cancelled input of a synchronous method
        on THIS container, which is `@modal.concurrent(max_inputs=32)` on one
        process, shuts the whole desk down. That happened three times on
        2026-09-04. The answer itself reads memory and a journal, so it runs
        on a thread and this container's loop keeps turning."""
        import asyncio

        return await asyncio.to_thread(self.campaigns.answer, verb, payload)


@app.function(image=cpu_image, schedule=modal.Period(minutes=15),
              timeout=3600)
async def reaper() -> None:
    """THE SUPERVISION TICK, on a clock (ADR 0001 Q5, ADR 0003): probe every
    listing, reap the ones that no longer answer, knock their metal back,
    retry the parked queue — and release the metal nothing has been busy on
    for its idle limit. `reap` is idempotent, so a tick that finds nothing to
    do does nothing, and `::reap` below is the same pass by hand.

    ONE clock for one fleet: this lives with the desk and not with any venue,
    because the campaign venues used to carry a reaper cron each, and under
    one desk that would tick the whole plane once per venue."""
    print(json.dumps(await desk().reap(probes=3, wait=30.0)), flush=True)


# ---------------------------------------------------------------------------
# the operator's doors
# ---------------------------------------------------------------------------

doors = modal.App(f"{DESK_APP}-doors")
"""THE OPERATOR'S APP, WHICH HOLDS NOTHING — and that is the whole point.

`modal run <file>::<door>` stands up an EPHEMERAL instance of the app the door
belongs to, and `Desk` above is `min_containers=1`: a door on `app` therefore
booted a SECOND desk, replaying the journal beside the standing one, on every
`::status`. Three of them were found standing at once, because a door that
blocks on a stalled desk holds its ephemeral app open for as long as it waits.

The doors below never touch the local class: `desk()` is a `Cls.from_name`
lookup that resolves to the DEPLOYED app. This operator app declares no Desk;
its optional image smoke runs only a CPU check. `modal deploy deploy/desk.py`
is unchanged: Modal deploys the variable named `app`."""

smoke = smoke_function(doors, cpu_image, module=__name__,
                       imports=("rlstack.runner.desk", "rlstack.data.stores.modal_volume"))


def parse_build(text: str) -> dict:
    """`max_model_len=4096,serves=steer,enforce_eager` as a build's kwargs.

    A bare word is a flag set true; `k=v` is a value; `serves` repeats to make
    a tuple. Ints and floats are read as such so a recipe's row is the same
    whether it came from this door or from a venue's constant."""
    out: dict = {}
    serves: list[str] = []
    for piece in [p.strip() for p in text.split(",") if p.strip()]:
        key, sep, value = piece.partition("=")
        if not sep:
            out[key] = True
            continue
        if key == "serves":
            serves.append(value)
            continue
        out[key] = read_scalar(value)
    if serves:
        out["serves"] = tuple(serves)
    return out


def read_scalar(value: str):
    """A flag's value in its narrowest honest type: int, then float, then the
    string as given."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def declared(engine: str, learner: str):
    """The two flag lines as a `Builds` record — what a carve is built from.

    Constructed on the CLIENT, because `modal run` runs a local entrypoint
    with this repo on its path and a Builds is an rlstack record like any
    other; only its row crosses. The write itself goes through the desk's own
    `recipe` verb, never into the journal beside it, because the fleet journal
    has ONE writer (I10)."""
    from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

    return Builds(engine=EngineBuild(**parse_build(engine)),
                  learner=LearnerBuild(**parse_build(learner)))


@doors.local_entrypoint()
def status() -> None:
    """Everything this desk knows, no wire calls into the metal."""
    import asyncio

    told = asyncio.run(desk().status())
    print(json.dumps({"listings": told["listings"], "metal": told["metal"],
                      "liveness": asyncio.run(desk().liveness())}, indent=2))


@doors.local_entrypoint()
def recipe(metal: str = "", engine: str = "", learner: str = "") -> None:
    """WHAT THIS METAL BUILDS (ADR 0007, Q4). Journaled, so it survives the
    desk; carried by every carve, so a reborn container is rebuilt from it and
    never from its own constants; and required, because a bare metal refuses a
    carve it cannot describe."""
    import asyncio

    if not metal:
        raise SystemExit("--metal <name> [--engine k=v,flag,...] "
                         "[--learner k=v,...]")
    told = asyncio.run(desk().recipe(metal, declared(engine, learner).row()))
    print(json.dumps(told, indent=1))


@doors.local_entrypoint()
def release(metal: str = "", reason: str = "released by hand",
            force: bool = False) -> None:
    """Hand one metal back. GUARDED (Q6): refused, with the running work
    named, while another tenancy is live on it. `--force` is this door's
    alone — no venue's door may send it."""
    import asyncio

    if not metal:
        raise SystemExit("--metal <name> [--reason ...] [--force]")
    told = asyncio.run(desk().release(metal, reason=reason, force=force))
    print(json.dumps(told, indent=1))
    if not told.get("released"):
        raise SystemExit(told.get("error", "refused"))


@doors.local_entrypoint()
def sweep(reason: str = "sweep", force: bool = False) -> None:
    """THE OPERATOR'S BACKSTOP: every metal on the plane released. Guarded by
    default — a sweep that silently killed a running campaign would be worse
    than metal left standing — so `--force` is how you mean it."""
    import asyncio

    held = asyncio.run(desk().status()).get("metal", {})
    for name, row in sorted(held.items()):
        if not row.get("plane"):
            continue
        told = asyncio.run(desk().release(name, reason=reason, force=force))
        print(f"[sweep] {name}: {json.dumps(told)}")
    print(json.dumps(asyncio.run(desk().status())["metal"], indent=1))


@doors.local_entrypoint()
def reap(probes: int = 3, wait: float = 0.0) -> None:
    """Probe every listing and reap the ones that no longer answer — the
    supervision pass, by hand."""
    import asyncio

    print(json.dumps(asyncio.run(desk().reap(probes=probes, wait=wait)),
                     indent=1))
