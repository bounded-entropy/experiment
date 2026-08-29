"""The three views: hosts / runs / gpu.

Each comes in two layers over the same store bytes — a *_data function
returning plain structures (what a UI serializes) and a render_* function
formatting that as terminal text. Pure functions of peeks and journals, so the
region's never-attach, never-write rule holds. Operational facts only:
identity, placement, status, progress, metal statistics. Experiment CONTENT is
a UI's business, driven by each run's own dictionary.json.

Every view reads ROOTS (#58), not bare stores: a row carries the FOLDER it was
found in — the store root's path relative to the top directory — because two
folders may legitimately hold the same run_id (the same spec submitted twice)
and the same host name. A bare Store still works and means folder "".
Annotations ride along on the runs view, read from the row's own root: they
are flavortext, rendered here and consulted nowhere else.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from rlstack.data.plan import wave_count
from rlstack.data.stores.base import ANNOTATION_FIELDS, Store
from rlstack.observe.locate import Root, rooted


def _when(t: float | None) -> str:
    return time.strftime("%m-%d %H:%M:%S", time.localtime(t)) if t else "?"


def qualified(folder: str, name: str) -> str:
    """A name inside its folder — how the observer says WHICH one when two
    roots hold the same host or run name. Bare in the degenerate case."""
    return f"{folder}/{name}" if folder else name


def _events_by_host(roots: Sequence[Store | Root]) -> list[tuple[Root, str, list[dict]]]:
    out = []
    for root in rooted(roots):
        for host in root.store.list_hosts():
            out.append((root, host, root.store.read_host_log(host)))
    return out


def _progress(store: Store, run_id: str) -> tuple[int, object]:
    """(committed, target) from read-only peeks — an observer never attaches.

    The target is the TRAIN PLAN's length: one wave is one gradient update, so
    a run is done when its plan is exhausted (#59). "?" for a run whose store
    holds no train plan — a pre-#59 run, or one still being created."""
    entries = store.peek_ledger(run_id)
    committed = int(entries[-1]["update"]) if entries else 0
    plan = store.peek_plan(run_id, "train")
    return committed, "?" if plan is None else wave_count(plan)


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


def hosts_data(roots: Sequence[Store | Root]) -> list[dict]:
    out = []
    for root, host, events in _events_by_host(roots):
        ups = [e for e in events if e.get("event") == "host-up"]
        attached = {e["run_id"] for e in events if e.get("event") == "attach"}
        detach = {e["run_id"]: e.get("status", "?") for e in events
                  if e.get("event") == "detach"}
        out.append({
            "host": host,
            "folder": root.folder,
            "journal_store": root.store.describe(),
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


def render_hosts(roots: Sequence[Store | Root]) -> str:
    lines = []
    for h in hosts_data(roots):
        lines.append(f"host {qualified(h['folder'], h['host'])}")
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

def runs_data(roots: Sequence[Store | Root]) -> list[dict]:
    """One row per (FOLDER, run_id) journaled in these roots, oldest first.

    Keyed by the pair because a run's address is the pair: the same spec
    submitted under two folders is the same run_id twice, two experiments
    with two histories. Each row carries the annotations its own root holds —
    read here, written by the CLI, and consulted by no experiment."""
    known = rooted(roots)
    annotations = [root.store.read_annotations() for root in known]
    rows: dict[tuple[str, str], dict] = {}
    for root, host, events in _events_by_host(known):
        for event in events:
            run_id = event.get("run_id")
            if not run_id or event.get("event") not in ("attach", "detach"):
                continue
            row = rows.setdefault((root.folder, run_id), {
                "run_id": run_id, "folder": root.folder, "hosts": [],
                "status": "running", "t": 0.0,
                "store": event.get("store", root.store.describe())})
            if host not in row["hosts"]:
                row["hosts"].append(host)
            if event.get("event") == "detach":
                row["status"] = event.get("status", "?")
            row["t"] = max(row["t"], event.get("t", 0.0))

    for (folder, run_id), row in rows.items():
        # the same run_id in more than one root: legitimate under #58 (one
        # spec, two folders), and the reason every link the observer emits
        # carries its folder — a bare /run/<id> must never silently pick one
        holding = [root for root in known
                   if root.store.peek_manifest(run_id) is not None]
        row["in_stores"] = [root.store.describe() for root in holding]
        row["in_folders"] = [root.folder for root in holding]
        row["forked"] = len(holding) > 1
        own = next((root for root in holding if root.folder == folder),
                   holding[0] if holding else None)
        committed, target = (_progress(own.store, run_id) if own else (0, "?"))
        row["committed"], row["target"] = committed, target
        for index, root in enumerate(known):
            if root.folder == folder:
                row.update(annotated(annotations[index].get(run_id)))
                break
    return sorted(rows.values(), key=lambda r: r["t"])


def annotated(fields: dict | None) -> dict:
    """A run's annotation as the views spell it: always all three keys, so a
    page never branches on presence. Absent is empty, never missing."""
    fields = fields or {}
    return {"name": str(fields.get("name") or ""),
            "tags": [str(tag) for tag in fields.get("tags") or []],
            "note": str(fields.get("note") or "")}


def matches(row: dict, needle: str) -> bool:
    """THE SEARCH, and it is nothing fancier: a case-insensitive substring
    over name + tags + note + run_id. An empty needle matches everything.

    The UI filters client-side with the same rule; this is its statement in
    Python, and the CLI's --grep."""
    if not needle:
        return True
    fields = annotated({field: row.get(field) for field in ANNOTATION_FIELDS})
    hay = " ".join([row.get("run_id", ""), fields["name"], fields["note"],
                    *fields["tags"]]).lower()
    return needle.lower() in hay


def render_runs(roots: Sequence[Store | Root], grep: str = "") -> str:
    """The runs table, grouped under its folders. In the degenerate case
    (one root, folder "") there is one nameless group and the table reads
    exactly as it always has, two annotation columns wider."""
    rows = [row for row in runs_data(roots) if matches(row, grep)]
    if not rows:
        return (f"no experiment matching {grep!r} in these stores\n" if grep
                else "no experiments journaled in these stores\n")
    lines = []
    for folder in sorted({row["folder"] for row in rows}):
        if folder:
            lines.append(f"folder {folder}")
        lines.append(f"{'run':<14} {'name':<18} {'tags':<16} {'status':<8} "
                     f"{'committed':>9}  {'host(s)':<20} {'last event':<15} store")
        for row in rows:
            if row["folder"] != folder:
                continue
            status = row["status"] + (" ⚠FORK" if row["forked"] else "")
            progress = f"{row['committed']}/{row['target']}"
            # the same id elsewhere: the other FOLDERS when they differ, and
            # the other stores when two roots share one address
            others = ([f or "(top)" for f in row["in_folders"] if f != folder]
                      or [s for s in row["in_stores"] if s != row["store"]])
            lines.append(
                f"{row['run_id']:<14} {row['name'][:18]:<18} "
                f"{','.join(row['tags'])[:16]:<16} {status:<8} {progress:>9}  "
                f"{'+'.join(row['hosts']):<20} {_when(row['t']):<15} "
                f"{row['store']}"
                + (f"  (also in: {', '.join(others)})" if others else ""))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# gpu
# ---------------------------------------------------------------------------

def gpu_data(roots: Sequence[Store | Root]) -> list[dict]:
    out = []
    for root, host, events in _events_by_host(roots):
        samples = [e for e in events if e.get("event") == "stats"]
        entry: dict = {"host": host, "folder": root.folder,
                       "samples": len(samples)}
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


def render_gpu(roots: Sequence[Store | Root]) -> str:
    lines = []
    for g in gpu_data(roots):
        lines.append(f"host {qualified(g['folder'], g['host'])}")
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
