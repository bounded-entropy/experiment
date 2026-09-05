"""Liveness: is the process behind a journal still breathing?

The store records history; it cannot by itself say whether a container is up
RIGHT NOW — a journal goes quiet both when a host is idle and when it is
dead. Two readings answer it, layered:

  heartbeat — journal-only, always available: a host that has been emitting
      events at some cadence and then stops for several cadences is PRESUMED
      down. Honest about its nature: it is a presumption, labeled as such.
  desk — authoritative when a fleet service is connected: the desk probes its
      listings over the wire (Listing.alive()), and where it speaks it wins.

The venue hands the desk in as a PLAIN CALLABLE returning, per host, either
`alive` (a bool) or a PULSE — {"alive": bool, "running": [run_id, ...]}, the
roster read off the same probe. The observer imports nothing from the runner
and knows no venue word, so the same UI runs against any store and any fleet
service (or none).

STALLED is the reading experiments need: a run the journal says is running,
on at least one host that is not alive, is not running in any useful sense —
it is stalled, and that is the row an operator debugs first.

LOST is its twin, and it needs the roster: a run the journal leaves open on
a host that IS alive and does NOT carry it. A carve name recurs per container
generation (the counter is the container's), so one host journal outlives
every generation, and a crashed generation's attach — never detached, because
a killed container writes nothing — reads as "running" for as long as any
later generation keeps the name alive. The journal cannot tell; the host can,
and the desk asks it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

# A host with no cadence to speak of still gets a floor: silence longer than
# this is presumed-down even for slow emitters, and anything quieter than its
# own cadence times the slack is too.
FLOOR_SECONDS = 120.0
CADENCE_SLACK = 4.0

DeskLiveness = Callable[[], Mapping[str, "bool | Mapping"]]


def bounded(desk: DeskLiveness, seconds: float) -> DeskLiveness:
    """A desk probe that ANSWERS WITHIN `seconds` or answers nothing — {}, a
    journal-only reading — because a page must never hang behind one probe.
    The probe runs on a thread of its own; one that outlives the deadline is
    abandoned there (it holds no state) and its late answer is dropped. What
    the venue's own wire deadline (transports) does for one verb, this does
    for the whole pulse from the page's side."""
    import concurrent.futures

    def probe() -> Mapping[str, "bool | Mapping"]:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(desk)
        pool.shutdown(wait=False)
        try:
            return future.result(timeout=seconds) or {}
        except concurrent.futures.TimeoutError:
            return {}
    return probe


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
    its source so the page can say which kind of truth it is showing. A desk
    that answers with a roster (`{"alive", "running"}`) puts `running` on
    the pulse, and only then can `stall_runs` call a run lost.
    """
    answered: Mapping[str, bool | Mapping] = {}
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
            answer = answered[host]
            if isinstance(answer, Mapping):
                pulse["live"] = bool(answer.get("alive"))
                pulse["running"] = sorted(answer.get("running", []))
            else:
                pulse["live"] = bool(answer)
            pulse["source"] = "desk"
        out[host] = pulse
    return out


def stall_runs(rows: Sequence[dict], pulses: Mapping[str, dict]) -> None:
    """The STALLED reading, in place: a running run on a down host it still
    RESIDES on. Residency is `open_hosts` — journals whose last word for the
    run is an attach; a venue the run detached from (died on, resumed
    elsewhere) is history and its death stalls nothing, while a crashed host
    that lost its detach stays open and rightly stalls. Rows without an
    `open_hosts` reading fall back to every host named. Unknown pulses stall
    nothing — a presumption of death needs at least a silence, and
    "failed"/"done" already say what they say.

    THE LOST reading, second: a running run resident on a host that is ALIVE
    and, asked, does not carry it. Only a pulse that brought a roster can say
    so (a bool-only desk marks nothing lost), and a host that is down is
    stalled first — a corpse cannot be asked what it carries."""
    for row in rows:
        if row.get("status") != "running":
            continue
        resident = row.get("open_hosts", row.get("hosts", []))
        down = [host for host in resident
                if pulses.get(host, {}).get("live") is False]
        if down:
            row["status"] = "stalled"
            row["stalled_hosts"] = down
            continue
        lost = [host for host in resident
                if pulses.get(host, {}).get("live") is True
                and "running" in pulses[host]
                and row.get("run_id") not in pulses[host]["running"]]
        if lost:
            row["status"] = "lost"
            row["lost_hosts"] = lost
