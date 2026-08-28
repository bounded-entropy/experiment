"""The CLI: thin argparse over the observer (rlstack/observe/).

    python -m rlstack {hosts,runs,gpu} <top-dir> [<top-dir> ...]
    python -m rlstack runs <top-dir> [--grep <substring>]
    python -m rlstack ui <top-dir> ... [--port N] [--panels panels.json]
    python -m rlstack tag <store-root> <run_id> [--name N] [--tag T ...] [--note ...]

A TOP DIRECTORY, not a store (#58): the views discover every store root
beneath it — a directory holding runs/, hosts/, fleet/ or annotations.jsonl —
and a top that is itself a root is the degenerate single-store case, which is
what every earlier invocation was. Locators resolve through observe.store_for:
paths and file:// resolve where mounted; modal:// tells you to run the reader
beside the volume. With none given, $RLSTACK_STORES (colon-separated) is used.

Every VIEW is peek-only: this entry point reads journals and manifests and
never attaches a run. `tag` is the one verb that writes, and it writes
flavortext beside runs/ — never inside a run directory, never into identity.
"""

from __future__ import annotations

import argparse
import os

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
    args = parser.parse_args(argv)

    if args.command == "tag":
        annotate(args)
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


if __name__ == "__main__":
    main()
