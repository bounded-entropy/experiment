"""rlstack CLI: observability over a store.

    python -m rlstack hosts <store-root>

Reads the host journals (hosts/<name>/log.jsonl — written by runner/host.py,
never consulted by correctness) joined with each run's ledger, and prints per
host: when it came up, what bases its engines serve, which experiments
attached (their pools, declared target, committed progress), and what state
they ended in. A host whose last event is old may simply be idle — the
journal records events, not liveness.
"""

from __future__ import annotations

import argparse
import json
import time

from rlstack.data.stores.base import Store
from rlstack.data.stores.local import LocalStore


def _when(t: float | None) -> str:
    if not t:
        return "?"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def _progress(store: Store, run_id: str) -> str:
    try:
        run = store.open_run(run_id)
    except Exception:
        return "no run dir"
    tail = run.ledger_tail()
    committed = int(tail["update"]) if tail else 0
    try:
        spec = json.loads(run.manifest["spec"])
        target = spec["algo"]["schedule"]["n_updates"]
    except Exception:
        target = "?"
    return f"{committed}/{target} committed"


def render_hosts(store: Store) -> str:
    lines: list[str] = []
    for host in store.list_hosts():
        events = store.read_host_log(host)
        ups = [e for e in events if e.get("event") == "host-up"]
        last = events[-1] if events else {}
        engines = ups[-1].get("engines", []) if ups else []
        lines.append(f"host {host}  up {len(ups)}x, last up "
                     f"{_when(ups[-1].get('t') if ups else None)}  "
                     f"engines: {', '.join(engines) or '?'}  "
                     f"last event {_when(last.get('t'))}")
        latest: dict[str, dict] = {}
        for event in events:
            if event.get("event") in ("attach", "detach"):
                run_id = event.get("run_id", "?")
                latest.setdefault(run_id, {}).update(event)
        for run_id, event in sorted(latest.items(),
                                    key=lambda kv: kv[1].get("t", 0)):
            status = event.get("status",
                               "running" if event.get("event") == "attach"
                               else "?")
            pools = ",".join(event.get("pools", [])) or "?"
            lines.append(f"  run {run_id}  pools {pools}  status {status}  "
                         f"{_progress(store, run_id)}")
        lines.append("")
    return "\n".join(lines) if lines else "no hosts have journaled here\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rlstack")
    sub = parser.add_subparsers(dest="command", required=True)
    hosts = sub.add_parser("hosts", help="what ran (or runs) on which metal")
    hosts.add_argument("store_root", help="store root (the dir holding runs/)")
    args = parser.parse_args(argv)
    if args.command == "hosts":
        print(render_hosts(LocalStore(args.store_root)), end="")


if __name__ == "__main__":
    main()
