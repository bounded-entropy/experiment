"""Thin RunPod deployment: the desk pod's process and the operator's verbs.

Pods are started and stopped by hand (ADR 0021); there is no `up`. See
examples/runpod-desk.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.venues.client import VenueClient
from rlstack.runner.venues.runpod.desk import desk, serve
from rlstack.runner.venues.runpod.provider import DeskConfig

STOP_BY_HAND = ("the desk released {metal}, but rlstack does not stop RunPod pods: "
                "stop the pod at {address} by hand in the RunPod console; it bills until then")


async def execute(args) -> None:
    config = DeskConfig.read(args.config)
    if args.command == "serve":
        await serve(config)
    elif args.command == "status":
        print(json.dumps(await desk(config).status(), indent=2))
    elif args.command == "release":
        remote = desk(config)
        reply = await remote.release(args.metal)
        print(json.dumps(reply, indent=2))
        if not reply.get("terminated"):
            print(STOP_BY_HAND.format(metal=args.metal,
                                      address=config.metal(args.metal).address), flush=True)
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
        reply = await VenueClient(desk(config)).submit(
            row, args.subdir, resume=args.resume, solo=args.solo,
            checkpointing=Checkpointing(every=args.every, delivery=args.delivery))
        print(json.dumps(reply, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="the desk itself, on the desk pod")
    commands.add_parser("status")
    commands.add_parser("release", help="release one metal; its pod is then stopped by hand"
                        ).add_argument("metal")
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
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
