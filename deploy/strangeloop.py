"""Thin local Strange Loop deployment. See examples/strangeloop-desk.md."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rlstack.runner.venues.strangeloop.provider import DeskConfig
from rlstack.runner.venues.strangeloop.desk import desk, serve
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.venues.client import VenueClient


async def execute(args) -> None:
    config = DeskConfig.read(args.config)
    root = Path(__file__).resolve().parents[1]
    if args.command == "serve":
        await serve(config, root)
    elif args.command == "up":
        # Acquisition is an operator command, not an experiment submission.
        desk(config)  # loads the shared runtime token
        # The standing process must own the tunnels. Its existing knock door
        # uses the configured boot callback, including for an initial slot.
        from rlstack.runner.remote import transport_for, BUILD_DEADLINE_S
        reply = await transport_for(config.operator_door).call("boot", {"metal": args.metal},
                                                         deadline_s=BUILD_DEADLINE_S)
        print(json.dumps(reply, indent=2))
    elif args.command == "reauth":
        # The token rotated (a login after it expired): hand every booted
        # pod, or the one named, the profile's current credentials.
        desk(config)
        from rlstack.runner.remote import transport_for, BUILD_DEADLINE_S
        for name in [args.metal] if args.metal else renewable(config):
            try:
                reply = await transport_for(config.operator_door).call(
                    "reauth", {"metal": name}, deadline_s=BUILD_DEADLINE_S)
            except Exception as exc:  # one pod's failure must not skip the rest
                reply = {"metal": name, "reauthorized": False, "error": str(exc)}
            print(json.dumps(reply, indent=2))
    elif args.command == "status":
        print(json.dumps(await desk(config).status(), indent=2))
    elif args.command == "release":
        print(json.dumps(await desk(config).release(args.metal), indent=2))
    elif args.command == "stop":
        remote = desk(config)
        if args.subdir:
            reply = await remote.stop_subdir(args.subdir, args.reason, drain=not args.no_drain)
        else:
            reply = await remote.stop(args.run, args.reason, drain=not args.no_drain)
        print(json.dumps(reply, indent=2, default=str))
    elif args.command == "dispositions":
        print(json.dumps(await desk(config).dispositions(), indent=2, default=str))
    elif args.command == "submit":
        row = json.loads(Path(args.spec).read_text())
        remote = desk(config)
        reply = await VenueClient(remote).submit(
            row, args.subdir, resume=args.resume, solo=args.solo,
            checkpointing=Checkpointing(every=args.every, delivery=args.delivery))
        print(json.dumps(reply, indent=2))
        if reply.get("accepted"):
            from rlstack.runner.remote import transport_for, BUILD_DEADLINE_S

            inventory = await remote.status()
            metal = inventory["listings"][reply["host"]]["metal"]
            run_ref = args.subdir.strip("/") + "/" + reply["run_id"] if args.subdir else reply["run_id"]
            exported = await transport_for(config.operator_door).call(
                "export", {"metal": metal, "run_ref": run_ref, "objective": args.objective},
                deadline_s=BUILD_DEADLINE_S)
            print(json.dumps({"export": exported}, indent=2))


def renewable(config: DeskConfig) -> list[str]:
    """Which slots `reauth` hands renewed credentials when none is named: the
    ones whose saved lease booted, read from the local state dir. A hosted
    desk's state is on its volume, so every slot is asked and an unbooted
    one answers with the desk's own refusal."""
    if config.operator_endpoint:
        return [metal.name for metal in config.metals]
    return [metal.name for metal in config.metals if booted(config, metal.name)]


def booted(config: DeskConfig, name: str) -> bool:
    """Whether this slot's saved lease has a worker to hand credentials to."""
    path = config.state_dir / (name + ".json")
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    return bool(row.get("lease_id")) and bool(row.get("boot_started"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve")
    commands.add_parser("status")
    for name in ("up", "release"):
        commands.add_parser(name).add_argument("metal")
    reauth = commands.add_parser("reauth", help="after a login: renew every booted pod's scratch token")
    reauth.add_argument("metal", nargs="?", help="one slot; every booted slot when omitted")
    stop = commands.add_parser("stop", help="a deliberate stop: drained, journaled, never revived (ADR 0014)")
    which = stop.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", help="one run id")
    which.add_argument("--subdir", help="every unfinished run filed under this subdir")
    stop.add_argument("--reason", default="stopped by hand")
    stop.add_argument("--no-drain", action="store_true", help="cancel at once instead of checkpointing first")
    commands.add_parser("dispositions", help="what is parked, stopped by hand, or failed")
    submit = commands.add_parser("submit")
    submit.add_argument("spec")
    submit.add_argument("--subdir", default="")
    submit.add_argument("--resume", action="store_true")
    submit.add_argument("--every", type=int, required=True,
                        help="checkpoint cadence in updates (ADR 0014) — a deliberate decision, no default")
    submit.add_argument("--delivery", choices=("wire", "store"), default="wire",
                        help="how the policy reaches its pools: over the wire (default) or from the store")
    submit.add_argument("--solo", action="store_true",
                        help="carve every host on a metal with nothing standing and refuse joins: "
                             "one run per metal, for runs whose training or serving must not share a GPU")
    submit.add_argument("--objective", required=True, help="actual committed metric, e.g. train/loss")
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
