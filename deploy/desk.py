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
                                                      #   (DeskRuntime does this every 15 min)

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

from rlstack.runner.venues.modal.desk import desk_class

from modal_venue import (
    DESK_APP, a_store, cpu_image_for, desk, smoke_function, store_volume,
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

IDLE_S = 90.0
"""THE FLEET'S CLOCK (ADR 0003): metal nothing has been busy on for this long
is released. A metal may declare its own finite limit at registration. NINETY SECONDS since 2026-09-05 (Samarth: "if no process is
running on a metal, it literally stops after a minute or smth") — busy is
a running tenancy, work in flight, or admitted traffic that moved
(`listing_busy`), so a run mid-load is never idle; only truly empty metal
is, and it goes within IDLE_TICK_S of its limit, independently of recovery probes."""

IDLE_TICK_S = 30.0
"""How often the standing desk reads its own idle clock — inside the
container, on its loop, because a ninety-second limit read every fifteen
minutes by recovery probes would still leave metal standing a quarter hour."""


Desk = desk_class(
    app, cpu_image, module=__name__, store_for=lambda: a_store(),
    volumes={"/store": store_volume}, bootable_metals=BOOTABLE_METALS,
    idle_s=IDLE_S,
    idle_tick_s=IDLE_TICK_S)


# Recovery and idle clocks are owned by the shared DeskRuntime.


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
def stop(run: str = "", subdir: str = "", reason: str = "stopped by hand",
         no_drain: bool = False) -> None:
    """A DELIBERATE STOP (ADR 0014, Part C): one run by id, or every
    unfinished run filed under a subdir. Drained by default — the Trainer
    checkpoints its last commit before the run ends — and journaled
    `stopped`, which nothing automatic revives; resubmit to move it again.

        modal run deploy/desk.py::stop --run <run_id>
        modal run deploy/desk.py::stop --subdir six-family/composition --reason "scope cut"
    """
    import asyncio

    if bool(run) == bool(subdir):
        raise SystemExit("say exactly one of --run <run_id> or --subdir <subdir>")
    if run:
        reply = asyncio.run(desk().stop(run, reason, drain=not no_drain))
    else:
        reply = asyncio.run(desk().stop_subdir(subdir, reason, drain=not no_drain))
    print(json.dumps(reply, indent=2, default=str), flush=True)


@doors.local_entrypoint()
def dispositions() -> None:
    """What is parked, what was stopped by hand, what failed of its own —
    the desk's standing disposition per run, with reasons and errors."""
    import asyncio

    print(json.dumps(asyncio.run(desk().dispositions()), indent=2, default=str), flush=True)


@doors.local_entrypoint()
def reap(probes: int = 3, wait: float = 0.0) -> None:
    """Probe every listing and reap the ones that no longer answer — the
    supervision pass, by hand."""
    import asyncio

    print(json.dumps(asyncio.run(desk().reap(probes=probes, wait=wait)),
                     indent=1))
