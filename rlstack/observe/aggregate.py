"""The readings over the journals' NUMERIC plane that the per-host one does not
own: a serving host's six rails, one run's clock, the fleet's two sums, and the
moments a timeline marks.

series.py reads one EXPERIMENT and host_series.py reads one HOST; the fleet
reading here is true of neither alone — what the whole fleet is producing. Two
aggregates, one per kind of partition: inference partitions summed as tokens
per second (every serving host's traffic), training partitions summed as
updates per second (every run's completed updates, wherever they ran).

Both are derived from the host journals' windowed emission:

    {"event": "traffic", "t", "window_s", "prefill_tokens", "decode_tokens",
     "requests", "ttft_ms_mean", "admit_wait_ms_mean", "admit_wait_ms_max",
     "inflight"}                                        one per stats tick
    {"event": "update", "t", "run_id", "update", "seconds",
     "phases": {"collect", "post", "train", "seal"}}    one per committed update

THE BUCKET RULE: hosts sample on their own clocks, so a sum across hosts is
only honest inside a window. Each bucket takes, per host, the MEAN of that
host's own rates inside it (every traffic event divides by its OWN window_s),
then sums those means across hosts — a host that sampled twice in a bucket does
not count twice. Buckets with no sample are omitted rather than drawn as zero:
an unjournaled host is silent, not idle, and the page's gap rule says so.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rlstack.data.stores.base import Store

FLEET_BUCKETS = 180     # the aggregate is a shape over the fleet's whole window

# The six rails a serving host draws. Rates are per-second (each traffic event
# over its own window); latencies and depth are journaled as they are.
TRAFFIC_CHANNELS = ("prefill_tok_s", "decode_tok_s", "requests_s",
                    "ttft_ms", "admit_wait_ms", "inflight")

_RATE_FIELDS = (("prefill_tok_s", "prefill_tokens"),
                ("decode_tok_s", "decode_tokens"),
                ("requests_s", "requests"))
_LEVEL_FIELDS = (("ttft_ms", "ttft_ms_mean"),
                 ("admit_wait_ms", "admit_wait_ms_mean"),
                 ("inflight", "inflight"))
PHASES = ("collect", "post", "train", "seal")


# merge seam: replaced by host_series.metric_series's named channels at merge —
# one line, `{m["key"]: m["points"] for m in metric_series(events)
#             if m["key"] in TRAFFIC_CHANNELS}`.
def traffic_channels(events: Sequence[dict]) -> dict[str, list]:
    """One serving host's six rails, from its windowed traffic events. A rate
    divides by the window the host itself declared, so a host that changes its
    stats cadence mid-run stays comparable with itself."""
    channels: dict[str, list] = {name: [] for name in TRAFFIC_CHANNELS}
    for event in events:
        if event.get("event") != "traffic":
            continue
        when, window = event.get("t"), event.get("window_s")
        if not isinstance(when, (int, float)):
            continue
        if isinstance(window, (int, float)) and window > 0:
            for channel, field in _RATE_FIELDS:
                count = event.get(field)
                if _numeric(count):
                    channels[channel].append([when, count / window])
        for channel, field in _LEVEL_FIELDS:
            value = event.get(field)
            if _numeric(value):
                channels[channel].append([when, float(value)])
    return {name: points for name, points in channels.items() if points}


# merge seam: replaced by host_series.run_timing at merge — one line,
# `from rlstack.observe.host_series import run_timing`.
def run_timing(store: Store, run_id: str) -> dict:
    """One run's completed updates as its hosts journaled them: wall seconds
    per update, decomposed into the four phases. Sorted by update; an update
    journaled twice (a resume re-running it) keeps the last line, which is the
    one that committed."""
    updates: dict[int, dict] = {}
    for host in store.list_hosts():
        for event in store.read_host_log(host):
            if event.get("event") != "update" or event.get("run_id") != run_id:
                continue
            number, seconds = event.get("update"), event.get("seconds")
            if not isinstance(number, int) or isinstance(number, bool):
                continue
            phases = event.get("phases") if isinstance(event.get("phases"), dict) else {}
            updates[number] = {
                "update": number,
                "t": event.get("t"),
                "seconds": float(seconds) if _numeric(seconds) else None,
                "phases": {phase: float(phases[phase]) for phase in PHASES
                           if _numeric(phases.get(phase))},
            }
    return {"updates": [updates[number] for number in sorted(updates)]}


# ---------------------------------------------------------------------------
# the fleet aggregate
# ---------------------------------------------------------------------------

def fleet_throughput(stores: Sequence[Store],
                     buckets: int = FLEET_BUCKETS) -> dict:
    """What the fleet is producing, summed across partitions: inference
    tokens/s and requests/s over every serving host, training updates/s over
    every run. One window, one bucket width, so the two charts read against
    the same clock."""
    traffic: list[tuple[float, str, dict]] = []
    updates: list[tuple[float, dict]] = []
    for store in stores:
        for host in store.list_hosts():
            for event in store.read_host_log(host):
                when = event.get("t")
                if not isinstance(when, (int, float)):
                    continue
                if event.get("event") == "traffic":
                    traffic.append((float(when), host, event))
                elif event.get("event") == "update":
                    updates.append((float(when), event))

    times = [when for when, _, _ in traffic] + [when for when, _ in updates]
    if not times:
        return {"window": None, "bucket_s": None, "inference": [],
                "training": [], "hosts": [], "runs": [], "totals": {}}
    window = [min(times), max(times)]
    width = max(1.0, (window[1] - window[0]) / buckets)

    return {
        "window": window,
        "bucket_s": width,
        "inference": inference_buckets(traffic, window[0], width),
        "training": training_buckets(updates, window[0], width),
        "hosts": sorted({host for _, host, _ in traffic}),
        "runs": sorted({str(event.get("run_id")) for _, event in updates}),
        "totals": {
            "prefill_tokens": _total(traffic, "prefill_tokens"),
            "decode_tokens": _total(traffic, "decode_tokens"),
            "requests": _total(traffic, "requests"),
            "updates": len(updates),
            "update_seconds": math.fsum(
                float(event["seconds"]) for _, event in updates
                if _numeric(event.get("seconds"))),
        },
    }


def inference_buckets(traffic: list[tuple[float, str, dict]],
                      start: float, width: float) -> list[dict]:
    """Tokens and requests per second summed over INFERENCE partitions: per
    bucket, each host's mean of its own per-window rates, summed across hosts.
    The raw journaled counts ride along, so the hover reads them."""
    binned: dict[int, dict[str, dict[str, list]]] = {}
    counts: dict[int, dict[str, float]] = {}
    for when, host, event in traffic:
        window = event.get("window_s")
        index = int((when - start) / width)
        per_host = binned.setdefault(index, {}).setdefault(
            host, {channel: [] for channel, _ in _RATE_FIELDS})
        raw = counts.setdefault(index, {field: 0.0 for _, field in _RATE_FIELDS})
        for channel, field in _RATE_FIELDS:
            value = event.get(field)
            if not _numeric(value):
                continue
            raw[field] += float(value)
            if isinstance(window, (int, float)) and window > 0:
                per_host[channel].append(value / window)
    out = []
    for index in sorted(binned):
        hosts = binned[index]
        point = {"t": start + (index + 0.5) * width,
                 "hosts": len(hosts),
                 "samples": sum(len(rates["prefill_tok_s"]) for rates in hosts.values())}
        for channel, field in _RATE_FIELDS:
            point[channel] = math.fsum(_mean(rates[channel]) or 0.0
                                       for rates in hosts.values())
            point[field] = counts[index][field]
        out.append(point)
    return out


def training_buckets(updates: list[tuple[float, dict]],
                     start: float, width: float) -> list[dict]:
    """Updates per second summed over TRAINING partitions — every run's
    committed updates, wherever they ran — with the mean wall seconds and the
    mean phase split inside the bucket, because an updates/s that halved is
    only a question until you can see which phase took the time."""
    binned: dict[int, list[dict]] = {}
    for when, event in updates:
        binned.setdefault(int((when - start) / width), []).append(event)
    out = []
    for index in sorted(binned):
        events = binned[index]
        seconds = [float(e["seconds"]) for e in events if _numeric(e.get("seconds"))]
        out.append({
            "t": start + (index + 0.5) * width,
            "updates": len(events),
            "updates_s": len(events) / width,
            "runs": len({str(e.get("run_id")) for e in events}),
            "seconds_mean": _mean(seconds),
            "phases": {phase: _mean([float(e["phases"][phase]) for e in events
                                     if isinstance(e.get("phases"), dict)
                                     and _numeric(e["phases"].get(phase))])
                       for phase in PHASES},
        })
    return out


# ---------------------------------------------------------------------------
# the moments a timeline marks
# ---------------------------------------------------------------------------

SAMPLED_EVENTS = ("stats", "traffic", "update")


def moments(events: Sequence[dict]) -> list[dict]:
    """Every journaled event that is NOT a sample: a boot, an attach, a detach
    — and, the day the arbiter journals them, a sleep or a wake. The gpu rails
    mark them on their own time axis, because a memory curve that falls off a
    cliff is only explained by the moment standing beside it."""
    out = []
    for event in events:
        kind = str(event.get("event", ""))
        when = event.get("t")
        if not kind or kind in SAMPLED_EVENTS or not _numeric(when):
            continue
        out.append({"t": float(when), "event": kind,
                    "run_id": event.get("run_id"),
                    "status": event.get("status")})
    return out


def _total(traffic: list[tuple[float, str, dict]], field: str) -> float:
    return math.fsum(float(event[field]) for _, _, event in traffic
                     if _numeric(event.get(field)))


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _numeric(value) -> bool:
    """A journaled number, not a flag: bools are flags (host_series' rule)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
