"""The three views: hosts / runs / gpu.

Each comes in two layers over the same store bytes — a *_data function
returning plain structures (what a UI serializes) and a render_* function
formatting that as terminal text. Pure functions of peeks and journals, so the
region's never-attach, never-write rule holds. Operational facts only:
identity, placement, status, progress, metal statistics. Experiment CONTENT is
a UI's business, driven by each run's own dictionary.json.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

from rlstack.data.stores.base import Store


def _when(t: float | None) -> str:
    return time.strftime("%m-%d %H:%M:%S", time.localtime(t)) if t else "?"


def _events_by_host(stores: Sequence[Store]) -> list[tuple[Store, str, list[dict]]]:
    out = []
    for store in stores:
        for host in store.list_hosts():
            out.append((store, host, store.read_host_log(host)))
    return out


def _progress(store: Store, run_id: str) -> tuple[int, object]:
    """(committed, target) from read-only peeks — an observer never attaches."""
    entries = store.peek_ledger(run_id)
    committed = int(entries[-1]["update"]) if entries else 0
    manifest = store.peek_manifest(run_id)
    try:
        target = json.loads(manifest["spec"])["algo"]["schedule"]["n_updates"]
    except (TypeError, KeyError, json.JSONDecodeError):
        target = "?"
    return committed, target


# ---------------------------------------------------------------------------
# hosts
# ---------------------------------------------------------------------------

def partition_metal(partition: dict) -> str:
    """The Metal name off a journaled partition, old spelling or new.

    Partition.metal was `gpuset` before the #55 rename, and journals on the
    volume are append-only history: an observer that only knew the new key
    would render every pre-rename host blank. Reading both is the whole cost of
    keeping that history legible."""
    return partition.get("metal") or partition.get("gpuset") or "?"


def _metal(partition: dict | None) -> str:
    """One journaled partition as the line an operator reads: how much of
    what, and where. A partition carries its gpu kind as a birth fact (#49) —
    without it a fraction cannot tell half an L4 from half an H100."""
    if not partition:
        return "unpartitioned"
    return (f"{partition.get('gpu') or '?'} {partition_metal(partition)}"
            f"[{','.join(str(d) for d in partition['devices'])}]"
            f" @ {partition['memory']:.2f}")


def hosts_data(stores: Sequence[Store]) -> list[dict]:
    out = []
    for store, host, events in _events_by_host(stores):
        ups = [e for e in events if e.get("event") == "host-up"]
        attached = {e["run_id"] for e in events if e.get("event") == "attach"}
        detach = {e["run_id"]: e.get("status", "?") for e in events
                  if e.get("event") == "detach"}
        out.append({
            "host": host,
            "journal_store": store.describe(),
            "engines": ups[-1].get("engines", []) if ups else [],
            "partition": ups[-1].get("partition") if ups else None,
            "boots": len(ups),
            "first_seen": events[0].get("t") if events else None,
            "last_seen": events[-1].get("t") if events else None,
            "running": sorted(attached - detach.keys()),
            "done": sum(1 for s in detach.values() if s == "done"),
            "failed": sum(1 for s in detach.values() if s == "failed"),
        })
    return out


def render_hosts(stores: Sequence[Store]) -> str:
    lines = []
    for h in hosts_data(stores):
        lines.append(f"host {h['host']}")
        lines.append(f"  metal   : {_metal(h['partition'])}")
        lines.append(f"  engines : {', '.join(h['engines']) or '?'}")
        lines.append(f"  journal : {h['journal_store']}")
        lines.append(f"  seen    : first {_when(h['first_seen'])}  last "
                     f"{_when(h['last_seen'])}  ({h['boots']} boot"
                     f"{'s' if h['boots'] != 1 else ''})")
        lines.append(f"  tenants : {len(h['running'])} running, {h['done']} "
                     f"done, {h['failed']} failed")
        lines.append("")
    return "\n".join(lines) if lines else "no hosts journaled in these stores\n"


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def runs_data(stores: Sequence[Store]) -> list[dict]:
    rows: dict[str, dict] = {}
    for store, host, events in _events_by_host(stores):
        for event in events:
            run_id = event.get("run_id")
            if not run_id or event.get("event") not in ("attach", "detach"):
                continue
            row = rows.setdefault(run_id, {
                "run_id": run_id, "hosts": [], "status": "running", "t": 0.0,
                "store": event.get("store", store.describe())})
            if host not in row["hosts"]:
                row["hosts"].append(host)
            if event.get("event") == "detach":
                row["status"] = event.get("status", "?")
            row["t"] = max(row["t"], event.get("t", 0.0))

    for run_id, row in rows.items():
        # I10, checked where every store is visible: one experiment, one store
        # — the same run_id held by two of them is a silently forked history,
        # which the observer flags rather than prevents
        holding = [s for s in stores if s.peek_manifest(run_id) is not None]
        row["in_stores"] = [s.describe() for s in holding]
        row["forked"] = len(holding) > 1
        committed, target = (_progress(holding[0], run_id) if holding
                             else (0, "?"))
        row["committed"], row["target"] = committed, target
    return sorted(rows.values(), key=lambda r: r["t"])


def render_runs(stores: Sequence[Store]) -> str:
    rows = runs_data(stores)
    if not rows:
        return "no experiments journaled in these stores\n"
    lines = [f"{'run':<14} {'status':<8} {'committed':>9}  "
             f"{'host(s)':<20} {'last event':<15} store"]
    for row in rows:
        status = row["status"] + (" ⚠FORK" if row["forked"] else "")
        progress = f"{row['committed']}/{row['target']}"
        lines.append(
            f"{row['run_id']:<14} {status:<8} {progress:>9}  "
            f"{'+'.join(row['hosts']):<20} {_when(row['t']):<15} "
            f"{row['store']}"
            + (f"  (also in: {', '.join(row['in_stores'][1:])})"
               if row["forked"] else ""))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# gpu
# ---------------------------------------------------------------------------

def gpu_data(stores: Sequence[Store]) -> list[dict]:
    out = []
    for _, host, events in _events_by_host(stores):
        samples = [e for e in events if e.get("event") == "stats"]
        entry: dict = {"host": host, "samples": len(samples)}
        if samples:
            times = [s["t"] for s in samples]
            deltas = sorted(b - a for a, b in zip(times, times[1:]))
            median = deltas[len(deltas) // 2] if deltas else 0.0
            utils = [max(g["util"] for g in s["gpus"]) for s in samples]
            entry.update({
                "span_seconds": times[-1] - times[0],
                "cadence_seconds": median,
                "util_mean": sum(utils) / len(utils),
                "util_max": max(utils),
                "util_last": utils[-1],
                "memory_last": samples[-1]["gpus"],
                "series": [{"t": s["t"], "gpus": s["gpus"]} for s in samples],
                "dark_gaps": [d for d in deltas if median and d > 3 * median],
            })
        out.append(entry)
    return out


def render_gpu(stores: Sequence[Store]) -> str:
    lines = []
    for g in gpu_data(stores):
        lines.append(f"host {g['host']}")
        if not g["samples"]:
            lines.append("  no gpu samples journaled (run_stats not running "
                         "or no NVIDIA runtime)\n")
            continue
        lines.append(f"  samples : {g['samples']} over "
                     f"{g['span_seconds']/60:.1f} min "
                     f"(every ~{g['cadence_seconds']:.0f}s)")
        lines.append(f"  util    : mean {g['util_mean']:.0f}%  "
                     f"max {g['util_max']}%  last {g['util_last']}%")
        lines.append("  memory  : " + "  ".join(
            f"gpu{i} {m['mem_used']}/{m['mem_total']} MiB"
            for i, m in enumerate(g["memory_last"])))
        lines.append(f"  downtime: {len(g['dark_gaps'])} gap(s), "
                     f"{sum(g['dark_gaps'])/60:.1f} min dark\n")
    return "\n".join(lines) if lines else "no hosts journaled in these stores\n"
