"""The CLI: thin argparse over the observer (rlstack/observe/).

    python -m rlstack {hosts,runs,gpu} <top-dir> [<top-dir> ...]
    python -m rlstack runs <top-dir> [--grep <substring>]
    python -m rlstack ui <top-dir> ... [--port N] [--panels panels.json]
    python -m rlstack tag <store-root> <run_id> [--name N] [--tag T ...] [--note ...]
    python -m rlstack sweep <store-root> <run_id>

A TOP DIRECTORY, not a store (#58): the views discover every store root
beneath it — a directory holding runs/, hosts/, fleet/ or annotations.jsonl —
and a top that is itself a root is the degenerate single-store case, which is
what every earlier invocation was. Locators resolve through observe.store_for:
paths and file:// resolve where mounted; modal:// tells you to run the reader
beside the volume. With none given, $RLSTACK_STORES (colon-separated) is used.

Every VIEW is peek-only: this entry point reads journals and manifests and
never attaches a run. `tag` puts annotations beside runs/; `sweep` reaches
inside a run directory and is the
one that ATTACHES — so it is for a run that has stopped (attaching discards
unsealed work, I10), never for one that is training right now.
"""

from __future__ import annotations

import argparse
import os

from rlstack.data.stores.base import Store
from rlstack.data.stores.retention import DEFAULT_RETENTION, RetentionPolicy
from rlstack.observe import (
    render_gpu, render_hosts, render_runs, roots_for, store_for,
)
from rlstack.observe.ui import serve as serve_ui

VIEWS = {"hosts": render_hosts, "runs": render_runs, "gpu": render_gpu}

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlstack")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in VIEWS:
        p = sub.add_parser(name)
        p.add_argument("store_locators", nargs="*",
                       help="top directories (default: $RLSTACK_STORES, "
                            "colon-separated)")
        if name == "runs":
            p.add_argument("--grep", default="",
                           help="case-insensitive substring over name, tags, "
                                "note and run_id")
    ui = sub.add_parser("ui", help="the graphs, in a browser (local wandb)")
    ui.add_argument("store_locators", nargs="*")
    ui.add_argument("--port", type=int, default=8321)
    ui.add_argument("--panels", default=None,
                    help="panels.json of derived graphs (default: the "
                         "store's own panels.json)")
    tag = sub.add_parser("tag", help="annotate a run (flavortext, never hashed)")
    tag.add_argument("store_root", help="THE run's own store root — the "
                                        "folder it was born in")
    tag.add_argument("run_id")
    tag.add_argument("--name", default=None, help="the run's human name")
    tag.add_argument("--tag", action="append", dest="tags", default=None,
                     help="repeatable; a later `tag` call replaces the list")
    tag.add_argument("--note", default=None)
    sweep = sub.add_parser("sweep", help="free what a finished run's store "
                                         "will never read again")
    sweep.add_argument("store_root", help="THE run's own store root — the "
                                          "folder it was born in")
    sweep.add_argument("run_id")
    fleet = sub.add_parser("fleet", help="talk to a standing desk, on any venue")
    fleet.add_argument("--desk", required=True,
                       help="the desk's address (modal://app/Desk, http://host:port, ...)")
    fleet_verbs = fleet.add_subparsers(dest="fleet_verb", required=True)
    stop = fleet_verbs.add_parser("stop", help="a deliberate stop: drained, "
                                              "journaled, never revived (ADR 0014)")
    which = stop.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", help="one run id")
    which.add_argument("--subdir", help="every unfinished run filed under this subdir")
    stop.add_argument("--reason", default="stopped by hand")
    stop.add_argument("--no-drain", action="store_true",
                      help="cancel at once instead of checkpointing first")
    fleet_verbs.add_parser("dispositions", help="what is parked, stopped by hand, or failed")
    fleet_verbs.add_parser("status", help="the desk's inventory")
    args = parser.parse_args(argv)

    if args.command == "fleet":
        fleet_verb(args)
        return

    if args.command == "tag":
        annotate(args)
        return
    if args.command == "sweep":
        sweep_run(store_for(args.store_root), args.run_id, DEFAULT_RETENTION)
        return
    locators = args.store_locators or [
        r for r in os.environ.get("RLSTACK_STORES", "").split(":") if r]
    if not locators:
        parser.error("no store locators given and RLSTACK_STORES is unset")
    roots = roots_for(locators)
    if args.command == "ui":
        serve_ui(roots, port=args.port, panels_path=args.panels)
        return
    if args.command == "runs":
        print(render_runs(roots, grep=args.grep), end="")
        return
    print(VIEWS[args.command](roots), end="")


def fleet_verb(args) -> None:
    """One desk verb over whatever transport its address names — the
    venue-neutral operator door (ADR 0014, Part C)."""
    import asyncio
    import json

    from rlstack.runner.remote import RemoteDesk, transport_for

    desk = RemoteDesk(transport_for(args.desk))
    if args.fleet_verb == "stop":
        if args.subdir:
            reply = asyncio.run(desk.stop_subdir(args.subdir, args.reason,
                                                 drain=not args.no_drain))
        else:
            reply = asyncio.run(desk.stop(args.run, args.reason, drain=not args.no_drain))
    elif args.fleet_verb == "dispositions":
        reply = asyncio.run(desk.dispositions())
    else:
        reply = asyncio.run(desk.status())
    print(json.dumps(reply, indent=2, default=str))


def annotate(args) -> None:
    """`tag` appends ONE row to <store root>/annotations.jsonl — the run
    directory is not touched, and only the flags actually given are written
    (the rest keep whatever an earlier row said)."""
    store = store_for(args.store_root)
    store.annotate_run(args.run_id, name=args.name, tags=args.tags,
                       note=args.note)
    merged = store.read_annotations().get(args.run_id, {})
    print(f"{args.run_id} in {store.describe()}\n"
          f"  name : {merged.get('name') or '—'}\n"
          f"  tags : {', '.join(merged.get('tags') or []) or '—'}\n"
          f"  note : {merged.get('note') or '—'}")


def sweep_run(store: Store, run_id: str, policy: RetentionPolicy) -> None:
    """`sweep` applies a retention policy to ONE run: it prints the blob
    versions the policy says nothing can read again, frees them, and reports
    the bytes recovered.

    The operator's half of retention — the Trainer sweeps at every commit, and
    this is for a run that has already stopped (a campaign that finished, or
    one interrupted before its last sweep). It ATTACHES, which is what makes it
    the one verb here that must never be pointed at a live run: attach discards
    work no ledger line committed, and a running trainer's unsealed update is
    exactly that.
    """
    run = store.open_run(run_id)
    named = list(policy.expendable(run.read_ledger()))
    print(f"{run_id} in {store.describe()}\n"
          f"  {type(policy).__name__}: {len(named)} blob version(s) expendable")
    swept = run.sweep(policy)
    for section, name, version in swept.blobs:
        print(f"    freed {section}/{name}@{version}")
    print(f"  {len(swept.blobs)} blob(s) deleted, {human_bytes(swept.freed)} "
          f"recovered")


def human_bytes(count: int) -> str:
    """Bytes as an operator reads them: binary units, one decimal."""
    size = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


if __name__ == "__main__":
    main()
