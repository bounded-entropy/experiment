"""Per-host series assembly: one host's journal read as the fleet sees it.

series.py is the per-EXPERIMENT reading; this is the per-HOST one, and its only
input is hosts/<name>/log.jsonl — which correctness never reads. Six named
readings, one per kind of fact a journal carries: boot_facts (the birth
attestation — engines, and for newer hosts the Partition and Regimes; older
journals carry neither and the page SAYS so rather than inventing one),
tenancy_lanes (attach/detach paired into residencies), gpu_channels (stats as
per-device util and memory series), traffic_channels (the host's own load, six
channels off the windows its meter drains), run_timing (one run's update clock,
across every host it ran on), and metric_series — which serves the traffic
channels first and then the OPEN SLOT, any numeric field an event carries that
no named reading claims.

The journal row IS the contract between the emitters (runner/meters.py) and
this file; the observer imports nothing from the runner, so the shape is
carried by the bytes and by nothing else.

fleet_data is the GLOBAL reading: hosts_data and runs_data joined with each
host's lanes and utilization under one shared time window, because placement
and load are fleet facts while progress and curves stay per-run facts.
"""

from __future__ import annotations

from collections.abc import Sequence

from rlstack.data.stores.base import Store
from rlstack.observe.locate import Root, rooted
from rlstack.observe.views import hosts_data, runs_data

# What the named readings above already render. Everything else numeric an
# event carries falls through to the open slot inside metric_series.
CLAIMED_FIELDS = {
    "host-up": ("engines", "partition", "regimes", "store"),
    "attach": ("pools", "remotes", "n_updates", "store"),
    "detach": ("status", "updates_completed"),
    "stats": ("gpus",),
    "traffic": ("window_s", "prefill_tokens", "decode_tokens", "requests",
                "ttft_ms_mean", "admit_wait_ms_mean", "admit_wait_ms_max",
                "inflight"),
    "update": ("update", "seconds", "phases"),
}
ALWAYS_CLAIMED = ("event", "t", "run_id")

# The two ways a traffic window becomes a channel. A RATE is a window count
# divided by the window it was counted over; a LEVEL is the window's own
# summary, plotted as journaled. Channel names are the contract the UI reads.
TRAFFIC_RATES = (("prefill_tok_s", "prefill_tokens"),
                 ("decode_tok_s", "decode_tokens"),
                 ("requests_s", "requests"))
TRAFFIC_LEVELS = (("ttft_ms", "ttft_ms_mean"),
                  ("admit_wait_ms", "admit_wait_ms_mean"),
                  ("inflight", "inflight"))
TRAFFIC_CHANNELS = tuple(name for name, _ in TRAFFIC_RATES + TRAFFIC_LEVELS)

UPDATE_PHASES = ("collect", "post", "train", "seal")

FLEET_POINTS = 600   # the fleet plot is a shape; the host page carries it all


def journals_for(roots: Sequence[Store | Root], host: str) -> list[tuple[Root, list[dict]]]:
    """Every root that journals this host, with its events. A host name is
    unique per store, not globally, so the observer shows all of them — and
    the folder is what tells two same-named hosts apart."""
    out = []
    for root in rooted(roots):
        events = root.store.read_host_log(host)
        if events:
            out.append((root, events))
    return out


def windowed(events: Sequence[dict], since: float | None) -> list[dict]:
    """Events at or after `since`; the whole history when since is None. The
    window bounds SERIES — identity facts (boots, first/last seen) always
    read the full journal, because a host does not stop being what it is
    when its birth scrolls out of the last 24 hours."""
    if since is None:
        return list(events)
    return [e for e in events if (e.get("t") or 0.0) >= since]


def lanes_overlapping(lanes: list[dict], since: float | None) -> list[dict]:
    """The residencies a window shows: still open, or closed inside it. A
    lane that both began and ended before the window is history the ALL
    range still tells whole."""
    if since is None:
        return lanes
    return [lane for lane in lanes
            if lane.get("detached") is None
            or (lane.get("detached") or 0.0) >= since]


def host_series(roots: Sequence[Store | Root], host: str,
                since: float | None = None) -> dict | None:
    """One host's page, from its journal alone. None when no root in this
    set has ever journaled that host; when several do, their events are read
    as one history and the page names every folder it drew from. `since`
    windows the series (lanes, gpu, metrics), never the identity facts."""
    journals = journals_for(roots, host)
    if not journals:
        return None
    events = sorted((event for _, evs in journals for event in evs),
                    key=lambda e: e.get("t") or 0.0)
    recent = windowed(events, since)
    boots = boot_facts(events)
    latest = boots[-1] if boots else {}
    return {
        "host": host,
        "folders": [root.folder for root, _ in journals],
        "journal_stores": [root.store.describe() for root, _ in journals],
        "events": len(events),
        "boots": boots,
        "engines": latest.get("engines", []),
        "partition": latest.get("partition"),
        "regimes": latest.get("regimes", []),
        "first_seen": events[0].get("t"),
        "last_seen": events[-1].get("t"),
        "tenancy": lanes_overlapping(tenancy_lanes(events), since),
        "gpus": gpu_channels(recent),
        "metrics": metric_series(recent),
    }


def boot_facts(events: Sequence[dict]) -> list[dict]:
    """Each host-up as the host attested itself at birth. A Partition and its
    Regimes are what a host IS (I12); a boot journaled before they were
    carries None, and the page says so rather than inventing one."""
    return [{"t": event.get("t"),
             "engines": list(event.get("engines", [])),
             "partition": event.get("partition"),
             "regimes": list(event.get("regimes", [])),
             "store": event.get("store")}
            for event in events if event.get("event") == "host-up"]


def tenancy_lanes(events: Sequence[dict]) -> list[dict]:
    """attach/detach paired into residencies, in attach order. A tenant that
    attached twice (resume, an alternating host) is two residencies; an
    attach still open renders detached=None — it is on the host now."""
    lanes: list[dict] = []
    open_lane: dict[str, dict] = {}
    for event in events:
        run_id = event.get("run_id")
        kind = event.get("event")
        if not run_id or kind not in ("attach", "detach"):
            continue
        if kind == "attach":
            lane = {"run_id": run_id, "store": event.get("store"),
                    "pools": list(event.get("pools", [])),
                    "remotes": list(event.get("remotes", [])),
                    "n_updates": event.get("n_updates"),
                    "attached": event.get("t"), "detached": None,
                    "status": "running", "updates_completed": None}
            lanes.append(lane)
            open_lane[run_id] = lane
        else:
            lane = open_lane.pop(run_id, None)
            if lane is None:      # a detach whose attach this store never saw
                lane = {"run_id": run_id, "store": event.get("store"),
                        "pools": [], "remotes": [], "n_updates": None,
                        "attached": None}
                lanes.append(lane)
            lane["detached"] = event.get("t")
            lane["status"] = event.get("status", "?")
            lane["updates_completed"] = event.get("updates_completed")
    return lanes


def gpu_channels(events: Sequence[dict]) -> list[dict]:
    """The stats events as one channel per device: utilization % and memory
    MiB over wall time, with the device's total as the ceiling. A gap in the
    samples IS host downtime (the gpu view names it); the page draws it."""
    channels: list[dict] = []
    for event in events:
        if event.get("event") != "stats" or not isinstance(event.get("gpus"), list):
            continue
        when = event.get("t")
        if not isinstance(when, (int, float)):
            continue
        for device, gpu in enumerate(event["gpus"]):
            if not isinstance(gpu, dict):
                continue
            while len(channels) <= device:
                channels.append({"device": len(channels), "util": [],
                                 "mem_used": [], "mem_total": None})
            channel = channels[device]
            if isinstance(gpu.get("util"), (int, float)):
                channel["util"].append([when, gpu["util"]])
            if isinstance(gpu.get("mem_used"), (int, float)):
                channel["mem_used"].append([when, gpu["mem_used"]])
            if isinstance(gpu.get("mem_total"), (int, float)):
                channel["mem_total"] = gpu["mem_total"]
    return channels


def traffic_channels(events: Sequence[dict]) -> list[dict]:
    """THE THROUGHPUT READING: each `traffic` event is one drained window of
    the host's own load, and these are its six channels — prefill_tok_s,
    decode_tok_s, requests_s (counts over the window they were counted in),
    ttft_ms, admit_wait_ms (the window's own means), inflight (the gauge at
    the tick).

    All six appear together as soon as a host has ever journaled a window, so
    a polling page never watches cards appear and vanish; a latency channel
    stays EMPTY while nothing measured one, because a window that served only
    scoring prefills has no time to first token and a fabricated 0ms would be
    a lie. A host that predates the emission plane gets no channels at all.
    """
    points: dict[str, list] = {name: [] for name in TRAFFIC_CHANNELS}
    windows = 0
    for event in events:
        when = event.get("t")
        if event.get("event") != "traffic" or not isinstance(when, (int, float)):
            continue
        windows += 1
        span = event.get("window_s")
        if isinstance(span, (int, float)) and not isinstance(span, bool) and span > 0:
            for channel, field in TRAFFIC_RATES:
                for _, count in numbers_under(channel, event.get(field)):
                    points[channel].append([when, count / span])
        for channel, field in TRAFFIC_LEVELS:
            for _, value in numbers_under(channel, event.get(field)):
                points[channel].append([when, value])
    if not windows:
        return []
    return [{"key": name, "event": "traffic", "points": points[name]}
            for name in TRAFFIC_CHANNELS]


def run_timing(store: Store, run_id: str) -> dict:
    """One run's UPDATE CLOCK: every `update` event this store's hosts
    journaled for it, raw and sorted by update.

    A run's placement is discoverable from the journals and a run may have
    moved (resume onto other metal), so every host in the store is scanned and
    the rows are joined — the run, not the host, is what the reader asked
    about. Nothing is derived here: steps/s and per-step bars are the page's
    arithmetic over these rows, and a row is exactly what the Trainer measured.

    One row per update, LAST line wins (by t): a journal is append-only
    history, and the same content-addressed run_id resubmitted over a wiped
    store legitimately journals the same update numbers again — the newest
    measurement is the one describing the run the reader is looking at.
    """
    updates: dict[int, dict] = {}
    for host in store.list_hosts():
        for event in store.read_host_log(host):
            if event.get("event") != "update" or event.get("run_id") != run_id:
                continue
            update = event.get("update")
            when = event.get("t")
            if (not isinstance(update, int) or isinstance(update, bool)
                    or not isinstance(when, (int, float))):
                continue
            if update in updates and updates[update]["t"] >= float(when):
                continue
            phases = event.get("phases")
            phases = phases if isinstance(phases, dict) else {}
            updates[update] = {
                "update": update, "t": float(when),
                "seconds": float(event.get("seconds") or 0.0),
                "phases": {phase: float(phases.get(phase) or 0.0)
                           for phase in UPDATE_PHASES}}
    return {"updates": [updates[number] for number in sorted(updates)]}


def metric_series(events: Sequence[dict]) -> list[dict]:
    """The host's numeric series: the six named traffic channels first, then
    THE OPEN SLOT — every numeric field an event carries that no named
    reading claims, keyed <event>.<field>. One rule, no schema — a host that
    starts journaling a new numeric fact gets it plotted without the observer
    learning its name first."""
    return traffic_channels(events) + open_slot(events)


def open_slot(events: Sequence[dict]) -> list[dict]:
    """The unclaimed half of metric_series, keyed <event>.<field>."""
    points: dict[str, list] = {}
    for event in events:
        kind = str(event.get("event", "event"))
        when = event.get("t")
        if not isinstance(when, (int, float)):
            continue
        claimed = set(ALWAYS_CLAIMED) | set(CLAIMED_FIELDS.get(kind, ()))
        for field, value in sorted(event.items()):
            if field in claimed:
                continue
            for name, number in numbers_under(str(field), value):
                points.setdefault(f"{kind}.{name}", []).append([when, number])
    return [{"key": key, "event": key.split(".")[0], "points": series}
            for key, series in sorted(points.items())]


def numbers_under(prefix: str, value) -> list[tuple[str, float]]:
    """(dotted field, float) for every number under a journaled value.
    Bools are flags and lists are facets, not series: neither is plotted."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [(prefix, float(value))]
    if isinstance(value, dict):
        out: list[tuple[str, float]] = []
        for field, inner in sorted(value.items()):
            out.extend(numbers_under(f"{prefix}.{field}", inner))
        return out
    return []


# ---------------------------------------------------------------------------
# the global reading
# ---------------------------------------------------------------------------

def fleet_data(roots: Sequence[Store | Root],
               since: float | None = None,
               now: float | None = None) -> dict:
    """What is true of the FLEET rather than of one experiment: every host as
    the hosts view renders it, joined with its residencies and its utilization
    shape, plus the runs view's placement. One window spans every timeline, so
    lanes on one page are comparable.

    The fleet spans EVERY discovered root — placement is a fact about metal,
    not about folders — but a row is per (folder, host), read from that
    root's journal alone: two folders may journal the same host name, and
    merging their lanes would invent a residency neither recorded."""
    known = rooted(roots)
    hosts = []
    for root in known:
        for row in hosts_data([root]):
            events = sorted(root.store.read_host_log(row["host"]),
                            key=lambda e: e.get("t") or 0.0)
            boots = boot_facts(events)
            latest = boots[-1] if boots else {}
            host = dict(row)                  # whatever the hosts view names
            host["tenancy"] = lanes_overlapping(tenancy_lanes(events), since)
            host["util"] = thin(busiest_device(windowed(events, since)),
                                FLEET_POINTS)
            host["regimes"] = latest.get("regimes", [])
            host["partition"] = latest.get("partition")
            hosts.append(host)
    # the axis is ANCHORED TO THE CLOCK when the reader asked for a window:
    # "past hour" ends now, not at the last event — a chart that stops hours
    # ago must show the emptiness, because the emptiness is the reading
    win = fleet_window(hosts)
    if win is not None and since is not None:
        win = [since, now if now is not None else win[1]]
    return {
        "hosts": hosts,
        "runs": runs_data(known),
        "stores": [root.store.describe() for root in known],
        "folders": [root.folder for root in known],
        "window": win,
    }


def busiest_device(events: Sequence[dict]) -> list[list[float]]:
    """One utilization line per host: the busiest device per sample. The
    fleet page asks whether a host is working; the host page asks which
    of its GPUs was."""
    out = []
    for event in events:
        if event.get("event") != "stats" or not isinstance(event.get("gpus"), list):
            continue
        when = event.get("t")
        utils = [gpu["util"] for gpu in event["gpus"]
                 if isinstance(gpu, dict) and isinstance(gpu.get("util"), (int, float))]
        if utils and isinstance(when, (int, float)):
            out.append([when, max(utils)])
    return out


def thin(points: list, limit: int) -> list:
    """Even striding down to `limit` points, endpoints kept — a shape, not a
    record. The host page reads the same journal unthinned."""
    if len(points) <= limit:
        return points
    step = len(points) / limit
    kept = [points[int(index * step)] for index in range(limit)]
    kept[-1] = points[-1]
    return kept


def fleet_window(hosts: Sequence[dict]) -> list[float] | None:
    """[first, last] over every host's journal — one time axis for the page."""
    times = [t for host in hosts
             for t in (host.get("first_seen"), host.get("last_seen"))
             if isinstance(t, (int, float))]
    return [min(times), max(times)] if times else None
