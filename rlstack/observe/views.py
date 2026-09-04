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

from rlstack.data.stores.base import (
    ANNOTATION_FIELDS, RunProgress, Store, run_progress,
)
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


def _progress(store: Store, run_id: str) -> RunProgress:
    """How far the run got, from read-only peeks — an observer never attaches.

    The one predicate the desk's reaper reads (data/stores/base.py): the
    target is the EXTENT plan's length — the train plan's where a run trains,
    since one wave is one gradient update (#59), the rollout plan's where it
    only generates (ADR 0006 Part B). `planned` is None for a run whose store
    holds neither — a pre-#59 run, or one still being created."""
    return run_progress(store, run_id)


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
        # per-run last word, then the ledger's: an attach after a detach is a
        # resume (running again), and commits reaching the plan mean done even
        # when the final detach died with its container
        status_by_run: dict[str, str] = {}
        for e in events:
            rid = e.get("run_id")
            if not rid:
                continue
            if e.get("event") == "attach":
                status_by_run[rid] = "running"
            elif e.get("event") == "detach":
                status_by_run[rid] = e.get("status", "?")
        for rid, status in list(status_by_run.items()):
            if status != "done" and _progress(root.store, rid).done:
                status_by_run[rid] = "done"
        out.append({
            "host": host,
            "folder": root.folder,
            "journal_store": root.store.describe(),
            "engines": ups[-1].get("engines", []) if ups else [],
            "partition": ups[-1].get("partition") if ups else None,
            # the processes the host was born with (ADR 0002): label + pid
            # off the last host-up, so "which residents are living" reads
            # off the journal without a probe
            "residents": ups[-1].get("residents", []) if ups else [],
            "boots": len(ups),
            "first_seen": events[0].get("t") if events else None,
            "last_seen": events[-1].get("t") if events else None,
            "running": sorted(r for r, s in status_by_run.items()
                              if s == "running"),
            "done": sum(1 for s in status_by_run.values() if s == "done"),
            "failed": sum(1 for s in status_by_run.values() if s == "failed"),
        })
    return out


def render_hosts(roots: Sequence[Store | Root]) -> str:
    lines = []
    for h in hosts_data(roots):
        lines.append(f"host {qualified(h['folder'], h['host'])}")
        lines.append(f"  metal   : {_metal(h['partition'])}")
        lines.append(f"  engines : {', '.join(h['engines']) or '?'}")
        if h["residents"]:
            lines.append("  residents: " + ", ".join(
                f"{r.get('label', '?')} (pid {r.get('pid', '?')})"
                for r in h["residents"]))
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
    filings = [root.store.run_subdirs() for root in known]
    rows: dict[tuple[str, str], dict] = {}
    for root, host, events in _events_by_host(known):
        for event in events:
            run_id = event.get("run_id")
            if not run_id or event.get("event") not in ("attach", "detach"):
                continue
            row = rows.setdefault((root.folder, run_id), {
                "run_id": run_id, "folder": root.folder, "hosts": [],
                "status": "running", "t": 0.0, "_status_t": -1.0,
                "store": event.get("store", root.store.describe())})
            if host not in row["hosts"]:
                row["hosts"].append(host)
            # residency per (run, host): a journal is chronological, so the
            # host's last word decides — attach leaves it OPEN, detach CLOSES
            # it. Stalling reads open residencies only: a venue the run left
            # (died on and resumed elsewhere) is provenance, not presence.
            row.setdefault("_open", {})[host] = (
                event.get("event") == "attach")
            # THE NEWEST EVENT SPEAKS FOR THE RUN, whichever host journal it
            # lives in: a run hops hosts across generations (resubmission is
            # resume), and journals are walked per host, so without the time
            # gate an old host's dying detach — iterated after the live
            # host's re-attach — would call a healthy second attempt by its
            # first attempt's death (observed live: six running arms shown
            # failed by a dead venue's journal).
            when = float(event.get("t", 0.0))
            if when >= row["_status_t"]:
                row["_status_t"] = when
                if event.get("event") == "detach":
                    row["status"] = event.get("status", "?")
                else:
                    row["status"] = "running"
            # filing from the attach itself (newest wins): the run directory's
            # manifest lands moments AFTER the attach line, so a snapshot can
            # hold the event and not the directory — without this, a newborn
            # files at the store's top for one refresh, then hops into its
            # subdir (observed live on the gsm arms)
            if (event.get("event") == "attach" and "subdir" in event
                    and when >= row.get("_subdir_t", -1.0)):
                row["_subdir_t"] = when
                row["_attach_subdir"] = event.get("subdir") or ""
            row["t"] = max(row["t"], when)

    for (folder, run_id), row in rows.items():
        row.pop("_status_t", None)   # ordering scratch, not a view field
        row["open_hosts"] = sorted(
            host for host, open_ in row.pop("_open", {}).items() if open_)
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
        progress = (_progress(own.store, run_id) if own
                    else RunProgress("", 0, None))
        # `committed` and `target` are the page's own words for the extent's
        # two numbers, and `extent` says which they count: a training run
        # commits updates, a generation-only run seals rollouts
        row["extent"] = progress.extent
        row["committed"] = progress.completed
        row["target"] = "?" if progress.planned is None else progress.planned
        # THE PLAN IS TRUTH: a run whose extent is complete is done, whatever
        # the journal's tail says — a crashed container loses its detach
        # events, and observability must not let that read as failure
        if progress.done:
            row["status"] = "done"
        # the SUBDIR the run's directory was filed under at birth ("" at the
        # top): the directory scan is truth once the manifest exists; the
        # attach event's word covers the birth window before it does
        attach_subdir = row.pop("_attach_subdir", None)
        row.pop("_subdir_t", None)
        for index, root in enumerate(known):
            if root.folder == folder:
                row.update(annotated(annotations[index].get(run_id)))
                filed = filings[index].get(run_id)
                row["subdir"] = (filed if filed is not None
                                 else attach_subdir or "")
                break
        else:
            row["subdir"] = attach_subdir or ""
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
        # "done/planned" of whatever the run's EXTENT counts: updates
        # committed, or rollouts sealed for a run that only generates
        lines.append(f"{'run':<14} {'name':<18} {'tags':<16} {'status':<8} "
                     f"{'progress':>9}  {'host(s)':<20} {'last event':<15} store")
        mine = [row for row in rows if row["folder"] == folder]
        seen_dir: str | None = None
        for row in sorted(mine, key=lambda r: (r.get("subdir", ""), r["t"])):
            subdir = row.get("subdir", "")
            if subdir != seen_dir:
                if subdir:
                    lines.append(f"  dir {subdir}/")
                seen_dir = subdir
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
