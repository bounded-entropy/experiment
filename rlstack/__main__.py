"""rlstack CLI: operational observability over one or more stores.

    python -m rlstack hosts <store-root> [<store-root> ...]
    python -m rlstack runs  <store-root> [<store-root> ...]
    python -m rlstack gpu   <store-root> [<store-root> ...]

Three views over the host journals (hosts/<name>/log.jsonl, written by
runner/host.py; correctness never reads them):

    hosts   per-host: engines' bases, the store it binds, first/last seen,
            tenancy counts by status
    runs    per-experiment: which host(s) it attached to, status, committed
            progress, and WHERE ITS DATA LIVES (the store the host declared)
    gpu     per-host metal statistics from the journaled stats samples:
            utilization, memory, sample coverage, downtime gaps

Deliberately operational only — run identity, placement, status, progress.
No experiment CONTENT (rewards, losses, curves) is rendered here: that is a
UI's job, reading the same stores. Store roots may also come from the
RLSTACK_STORES environment variable (colon-separated) when none are given.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence

from rlstack.data.stores.base import Store
from rlstack.data.stores.local import LocalStore


def _when(t: float | None) -> str:
    return time.strftime("%m-%d %H:%M:%S", time.localtime(t)) if t else "?"


def _events_by_host(stores: Sequence[Store]) -> list[tuple[Store, str, list[dict]]]:
    out = []
    for store in stores:
        for host in store.list_hosts():
            out.append((store, host, store.read_host_log(host)))
    return out


def _progress(store: Store, run_id: str) -> str:
    """Committed updates vs the spec's target — status, not content. Uses
    the store's read-only peeks: an observer never attaches a live run."""
    entries = store.peek_ledger(run_id)
    committed = int(entries[-1]["update"]) if entries else 0
    manifest = store.peek_manifest(run_id)
    try:
        target = json.loads(manifest["spec"])["algo"]["schedule"]["n_updates"]
    except (TypeError, KeyError, json.JSONDecodeError):
        target = "?"
    return f"{committed}/{target}"


# ---------------------------------------------------------------------------
# the three views
# ---------------------------------------------------------------------------

def render_hosts(stores: Sequence[Store]) -> str:
    lines = []
    for store, host, events in _events_by_host(stores):
        ups = [e for e in events if e.get("event") == "host-up"]
        attached = {e["run_id"]: e for e in events
                    if e.get("event") == "attach"}
        detached = {e["run_id"]: e for e in events
                    if e.get("event") == "detach"}
        running = len(attached.keys() - detached.keys())
        done = sum(1 for e in detached.values() if e.get("status") == "done")
        failed = sum(1 for e in detached.values() if e.get("status") == "failed")
        engines = ups[-1].get("engines", []) if ups else []
        bound = ups[-1].get("store", store.describe()) if ups else store.describe()
        lines.append(f"host {host}")
        lines.append(f"  engines : {', '.join(engines) or '?'}")
        lines.append(f"  store   : {bound}")
        lines.append(f"  seen    : first {_when(events[0].get('t') if events else None)}"
                     f"  last {_when(events[-1].get('t') if events else None)}"
                     f"  ({len(ups)} boot{'s' if len(ups) != 1 else ''})")
        lines.append(f"  tenants : {running} running, {done} done, {failed} failed")
        lines.append("")
    return "\n".join(lines) if lines else "no hosts journaled in these stores\n"


def render_runs(stores: Sequence[Store]) -> str:
    rows: dict[str, dict] = {}
    for store, host, events in _events_by_host(stores):
        for event in events:
            run_id = event.get("run_id")
            if not run_id or event.get("event") not in ("attach", "detach"):
                continue
            row = rows.setdefault(run_id, {
                "hosts": [], "status": "running", "t": 0.0,
                "store": event.get("store", store.describe()),
                "progress_store": store})
            if host not in row["hosts"]:
                row["hosts"].append(host)
            if event.get("event") == "detach":
                row["status"] = event.get("status", "?")
            row["t"] = max(row["t"], event.get("t", 0.0))
    if not rows:
        return "no experiments journaled in these stores\n"
    lines = [f"{'run':<14} {'status':<8} {'committed':>9}  "
             f"{'host(s)':<20} {'last event':<15} store"]
    for run_id, row in sorted(rows.items(), key=lambda kv: kv[1]["t"]):
        lines.append(
            f"{run_id:<14} {row['status']:<8} "
            f"{_progress(row['progress_store'], run_id):>9}  "
            f"{'+'.join(row['hosts']):<20} {_when(row['t']):<15} {row['store']}")
    return "\n".join(lines) + "\n"


def render_gpu(stores: Sequence[Store]) -> str:
    lines = []
    for _, host, events in _events_by_host(stores):
        samples = [e for e in events if e.get("event") == "stats"]
        lines.append(f"host {host}")
        if not samples:
            lines.append("  no gpu samples journaled (run_stats not running "
                         "or no NVIDIA runtime)\n")
            continue
        times = [s["t"] for s in samples]
        span = times[-1] - times[0]
        utils = [max(g["util"] for g in s["gpus"]) for s in samples]
        last = samples[-1]["gpus"]
        deltas = sorted(b - a for a, b in zip(times, times[1:]))
        median = deltas[len(deltas) // 2] if deltas else 0.0
        dark = [d for d in deltas if median and d > 3 * median]
        lines.append(f"  samples : {len(samples)} over {span/60:.1f} min "
                     f"(every ~{median:.0f}s)")
        lines.append(f"  util    : mean {sum(utils)/len(utils):.0f}%  "
                     f"max {max(utils)}%  last {utils[-1]}%")
        lines.append("  memory  : " + "  ".join(
            f"gpu{i} {g['mem_used']}/{g['mem_total']} MiB"
            for i, g in enumerate(last)))
        lines.append(f"  downtime: {len(dark)} gap(s), "
                     f"{sum(dark)/60:.1f} min dark\n")
    return "\n".join(lines) if lines else "no hosts journaled in these stores\n"


# ---------------------------------------------------------------------------

VIEWS = {"hosts": render_hosts, "runs": render_runs, "gpu": render_gpu}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlstack")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in VIEWS.items():
        p = sub.add_parser(name, help=fn.__doc__)
        p.add_argument("store_roots", nargs="*",
                       help="store roots (default: $RLSTACK_STORES, colon-separated)")
    args = parser.parse_args(argv)
    roots = args.store_roots or [
        r for r in os.environ.get("RLSTACK_STORES", "").split(":") if r]
    if not roots:
        parser.error("no store roots given and RLSTACK_STORES is unset")
    stores = [LocalStore(root) for root in roots]
    print(VIEWS[args.command](stores), end="")


if __name__ == "__main__":
    main()
