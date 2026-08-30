"""Liveness: is the process behind a journal still breathing?

The store records history; it cannot by itself say whether a container is up
RIGHT NOW — a journal goes quiet both when a host is idle and when it is
dead. Two readings answer it, layered:

  heartbeat — journal-only, always available: a host that has been emitting
      events at some cadence and then stops for several cadences is PRESUMED
      down. Honest about its nature: it is a presumption, labeled as such.
  desk — authoritative when a fleet service is connected: the desk probes its
      listings over the wire (Listing.alive()), and where it speaks it wins.

The venue hands the desk in as a PLAIN CALLABLE returning {host: alive} —
the observer imports nothing from the runner and knows no venue word, so the
same UI runs against any store and any fleet service (or none).

STALLED is the reading experiments need: a run the journal says is running,
on at least one host that is not alive, is not running in any useful sense —
it is stalled, and that is the row an operator debugs first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

# A host with no cadence to speak of still gets a floor: silence longer than
# this is presumed-down even for slow emitters, and anything quieter than its
# own cadence times the slack is too.
FLOOR_SECONDS = 120.0
CADENCE_SLACK = 4.0

DeskLiveness = Callable[[], Mapping[str, bool]]


def heartbeat(events: Sequence[dict], now: float) -> dict:
    """One host's pulse, from its journal alone.

    {"last": t, "age": s, "cadence": s | None, "live": bool | None} — `live`
    None means the journal is too thin to presume anything (fewer than two
    timed events), and the page says "unknown" rather than guessing.
    """
    times = sorted(float(e["t"]) for e in events
                   if isinstance(e.get("t"), (int, float)))
    if not times:
        return {"last": None, "age": None, "cadence": None, "live": None}
    last = times[-1]
    age = max(0.0, now - last)
    if len(times) < 2:
        return {"last": last, "age": age, "cadence": None, "live": None}
    tail = times[-33:]
    gaps = sorted(b - a for a, b in zip(tail, tail[1:]) if b > a)
    cadence = gaps[len(gaps) // 2] if gaps else None
    quiet_after = max(FLOOR_SECONDS,
                      CADENCE_SLACK * cadence if cadence else 0.0)
    return {"last": last, "age": age, "cadence": cadence,
            "live": age <= quiet_after}


def liveness_by_host(events_by_host: Sequence[tuple[str, Sequence[dict]]],
                     now: float,
                     desk: DeskLiveness | None = None) -> dict[str, dict]:
    """Every host's pulse, desk-corrected where a desk speaks.

    The desk's answer is a PROBE (the container answered a call just now);
    the heartbeat is a PRESUMPTION (the journal went quiet). Each row names
    its source so the page can say which kind of truth it is showing.
    """
    answered: Mapping[str, bool] = {}
    if desk is not None:
        try:
            answered = desk() or {}
        except Exception:
            answered = {}          # a dead desk is a journal-only day
    out: dict[str, dict] = {}
    for host, events in events_by_host:
        pulse = heartbeat(events, now)
        pulse["source"] = "heartbeat"
        if host in answered:
            pulse["live"] = bool(answered[host])
            pulse["source"] = "desk"
        out[host] = pulse
    return out


def stall_runs(rows: Sequence[dict], pulses: Mapping[str, dict]) -> None:
    """The STALLED reading, in place: a running run on any host presumed or
    known down. Unknown pulses stall nothing — a presumption of death needs
    at least a silence, and "failed"/"done" already say what they say."""
    for row in rows:
        if row.get("status") != "running":
            continue
        down = [host for host in row.get("hosts", [])
                if pulses.get(host, {}).get("live") is False]
        if down:
            row["status"] = "stalled"
            row["stalled_hosts"] = down
