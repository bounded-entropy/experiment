"""The CLI: thin argparse over the observer (rlstack/observe/).

    python -m rlstack {hosts,runs,gpu} <top-dir> [<top-dir> ...]
    python -m rlstack runs <top-dir> [--grep <substring>]
    python -m rlstack ui <top-dir> ... [--port N] [--panels panels.json]
    python -m rlstack tag <store-root> <run_id> [--name N] [--tag T ...] [--note ...]
    python -m rlstack tasks <dataset> --store <root> [--split name=frac ...]
    python -m rlstack sweep <store-root> <run_id>

A TOP DIRECTORY, not a store (#58): the views discover every store root
beneath it — a directory holding runs/, hosts/, fleet/ or annotations.jsonl —
and a top that is itself a root is the degenerate single-store case, which is
what every earlier invocation was. Locators resolve through observe.store_for:
paths and file:// resolve where mounted; modal:// tells you to run the reader
beside the volume. With none given, $RLSTACK_STORES (colon-separated) is used.

Every VIEW is peek-only: this entry point reads journals and manifests and
never attaches a run. Three verbs write. `tag` puts flavortext beside runs/
and `tasks` puts content-addressed task sets in the store's cas/, neither
inside a run directory; `sweep` is the one that reaches inside one, and the
one that ATTACHES — so it is for a run that has stopped (attaching discards
unsealed work, I10), never for one that is training right now.
"""

from __future__ import annotations

import argparse
import os

from rlstack.data.stores.base import Store
from rlstack.data.stores.retention import DEFAULT_RETENTION, RetentionPolicy
from rlstack.data.tasks import (
    concept_prompt_tasks, dapo_math_tasks, split_tasks, write_tasks,
)
from rlstack.observe import (
    render_gpu, render_hosts, render_runs, roots_for, store_for,
)
from rlstack.observe.ui import serve as serve_ui

VIEWS = {"hosts": render_hosts, "runs": render_runs, "gpu": render_gpu}

# dataset name -> the one function turning that dataset into Task rows
BUILDERS = {"concept_prompts": concept_prompt_tasks,
            "dapo_math": dapo_math_tasks}


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
    tasks = sub.add_parser("tasks", help="build a dataset into task sets")
    tasks.add_argument("dataset", choices=sorted(BUILDERS))
    tasks.add_argument("--store", required=True,
                       help="the store root the sets are written into")
    tasks.add_argument("--split", action="append", dest="splits", default=None,
                       metavar="NAME=FRACTION", help="repeatable; fractions "
                       "must sum to 1 (default: train=0.98 eval=0.02)")
    tasks.add_argument("--seed", type=int, default=17,
                       help="the split draw's seed (a task's split is a "
                            "function of this and its id alone)")
    sweep = sub.add_parser("sweep", help="free what a finished run's store "
                                         "will never read again")
    sweep.add_argument("store_root", help="THE run's own store root — the "
                                          "folder it was born in")
    sweep.add_argument("run_id")
    args = parser.parse_args(argv)

    if args.command == "tag":
        annotate(args)
        return
    if args.command == "sweep":
        sweep_run(store_for(args.store_root), args.run_id, DEFAULT_RETENTION)
        return
    if args.command == "tasks":
        build_task_sets(store_for(args.store), args.dataset,
                        fractions(args.splits), args.seed)
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


def fractions(splits: list[str] | None) -> dict[str, float]:
    """`--split train=0.98` pairs as a mapping, in the order given (which is
    the order split_tasks lays the intervals out in)."""
    if not splits:
        return {"train": 0.98, "eval": 0.02}
    return {name: float(value)
            for name, _, value in (s.partition("=") for s in splits)}


def build_task_sets(store: Store, dataset: str, splits: dict[str, float],
                    seed: int) -> dict[str, str]:
    """Build one dataset, split it, write each split — the whole task-set path.

    Prints and returns {split name: cas uri}: those uris are what a spec pins,
    and the printing is why this lives in the CLI rather than under data/.
    """
    tasks = BUILDERS[dataset]()
    print(f"{dataset}: {len(tasks)} tasks, split {splits} at seed {seed}")
    uris = {}
    for name, members in split_tasks(tasks, splits, seed).items():
        uris[name] = write_tasks(store, members)
        print(f"  {name:<8} {len(members):>7}  {uris[name]}")
    return uris


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
