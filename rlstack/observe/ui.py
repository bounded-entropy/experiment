"""The observer UI: the routes over the store, with zero core interference.

One stdlib WSGI app — no dependencies, no build step — served either locally or
beside a remote store. Three families of response: the DOCUMENT (index.html,
returned for every page route, because the page routes on location.pathname),
the ASSETS (/web/<file>: the stylesheet and the ES modules, read out of the
package by page.py), and the JSON API polled every few seconds, so live
monitoring is just committed state read again.

Three readings, three families of route. Per EXPERIMENT, where panel priority
IS the run's own dictionary.json with the loss walkback first (I11), and whose
sealed waves are readable one click deep (#57). Per HOST, off the journal
alone. Per FLEET, the aggregate no single host or run knows.

    /  /run/<id>  /run/<id>/wave/<n>  /hosts  /host/<name>
    /api/runs  /api/hosts  /api/fleet  /api/host/<name>
    /api/run/<id>  /api/run/<id>/timing  /api/run/<id>/waves
                                         /api/run/<id>/wave/<n>

THE FOLDER RIDES IN THE QUERY (#58). The app reads ROOTS, and a run_id names
one run PER ROOT — the same spec submitted under two folders is the same
run_id twice. So every run and host route takes ?root=<folder>; a route
without one resolves against all roots and serves the unique match, and when
several roots hold the id it answers with the AMBIGUITY — the folders, for the
page to list — rather than silently picking one. A bare Store still means the
degenerate single root (folder ""), which is exactly how the deployed
observer hands in its volume: no query, one root, unchanged behavior.

The UI never re-derives a declaration, never attaches, and never writes.
Annotations appear in /api/runs because the runs view read them; nothing here
writes one, and there is no POST route to write one with.
"""

from __future__ import annotations

import hashlib
import time

import json
from collections.abc import Callable, Sequence
from urllib.parse import parse_qs, unquote

from rlstack.data.stores.base import Store
from rlstack.observe.aggregate import (
    fleet_throughput, moments, run_timing, traffic_channels,
)
from rlstack.observe.host_series import (
    fleet_data, host_series, journals_for, windowed,
)
from rlstack.observe.liveness import (
    DeskLiveness, liveness_by_host, stall_runs,
)
from rlstack.observe.locate import Root, rooted
from rlstack.observe.select import metric_names, overlay
from rlstack.observe.page import WEB_PREFIX, asset, document
from rlstack.observe.series import run_series
from rlstack.observe.views import fleet_notes, note_the_fleet, run_row, runs_data
from rlstack.observe.waves import wave_detail, wave_list

NOT_FOUND = "404 Not Found"


def page_bytes(start_response, environ, body: bytes, content_type: str):
    """One page file — the document or one asset — served WITH A VALIDATOR.

    The rule: a browser must never pair a held document with fetched modules
    (index.html names the elements the modules write into), so every page file
    says no-cache and is revalidated. But "revalidate" without a validator
    means "re-download": there was no ETag and no Last-Modified, so a
    conditional request answered 200 with the whole body, and every click
    re-fetched the entire module graph before a line of the page ran. The
    ETag makes the same revalidation a 304 with no body.
    """
    tag = '"' + hashlib.sha256(body).hexdigest()[:16] + '"'
    validators = [("ETag", tag), ("Cache-Control", "no-cache")]
    if environ.get("HTTP_IF_NONE_MATCH") == tag:
        start_response("304 Not Modified", validators)
        return [b""]
    start_response("200 OK", [("Content-Type", content_type)] + validators)
    return [body]


def ui_app(roots: Sequence[Store | Root],
           refresh: Callable[[], None] | None = None,
           panels: Callable[[], list[dict]] | None = None,
           desk: DeskLiveness | None = None):
    """The WSGI app over one top directory's roots (or one bare store, the
    degenerate case). `refresh` runs before each API read (a volume-backed
    store may need it; None for local). `panels` supplies derived-graph
    declarations (None → each store's own panels.json, re-read per request so
    edits appear live). `desk` is the OPTIONAL liveness probe — a plain
    callable returning {host: alive}, constructed by the venue around
    whatever fleet service it runs (the observer never learns which); None
    means journal heartbeats are the only pulse.

    Two speed layers, both honesty-preserving: every root reads through a
    CachedReadStore (immutable bytes cached forever, journals revalidated on
    size — observe/cache.py), and one MEMO holds each API payload for a poll
    tick, so N viewers polling every 3s cost one computation, not N."""
    import threading

    from rlstack.observe.cache import CachedReadStore

    known = [Root(root.folder, CachedReadStore(root.store))
             for root in rooted(roots)]
    memo: dict[str, tuple[float, object, str]] = {}
    hold = threading.Lock()
    memo_ttl = 2.5

    def app(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        if path.startswith(WEB_PREFIX):
            found = asset(unquote(path[len(WEB_PREFIX):]))
            if found is None:
                start_response(NOT_FOUND, [("Content-Type", "text/plain")])
                return [b"no such asset"]
            body, content_type = found
            return page_bytes(start_response, environ, body, content_type)
        if path.startswith("/api/"):
            query = environ.get("QUERY_STRING", "")
            ticket = path + "?" + query
            with hold:
                held = memo.get(ticket)
            if held is not None and time.time() - held[0] < memo_ttl:
                return _json(start_response, held[1], held[2])
            try:
                if refresh is not None:
                    refresh()
                hours = asked_hours(query)
                now = time.time()
                payload, status = api(
                    known, [unquote(part) for part in path.split("/") if part],
                    asked_folder(query),
                    panels() if panels is not None else None,
                    since=None if hours is None else now - hours * 3600.0,
                    now=now, desk=desk,
                    params=parse_qs(query, keep_blank_values=True))
            except Exception as racing:
                # an observer read can RACE the store it reads (a volume
                # reload swapping files mid-scan): that is a 503 saying try
                # again — never a dead worker, and never a lying 404 the page
                # would mistake for "no such run"
                payload, status = ({"error": "transient read failure",
                                    "detail": str(racing)},
                                   "503 Service Unavailable")
            if status == "200 OK":     # a failure is retried, never served stale
                with hold:
                    memo[ticket] = (time.time(), payload, status)
                    if len(memo) > 256:
                        del memo[min(memo, key=lambda k: memo[k][0])]
            return _json(start_response, payload, status)
        # every page is the same document; the modules route on the pathname
        return page_bytes(start_response, environ, document(),
                          "text/html; charset=utf-8")

    return app


def asked_hours(query: str) -> float | None:
    """?hours=N — how far back the fleet pages read. Absent or 0 means
    everything (the server stays timeless; the WEB CLIENT asks for one day by
    default, because a fleet chart spanning its whole journal is a smear, not
    a reading). Run pages never window: an experiment's curve is its whole
    story by definition."""
    params = parse_qs(query, keep_blank_values=True)
    raw = params.get("hours", ["0"])[0]
    try:
        hours = float(raw)
    except ValueError:
        hours = 24.0
    return None if hours <= 0 else hours


def asked_folder(query: str) -> str | None:
    """?root=<folder> — the folder the link carried, or None when it carried
    none. "" is a folder (the top itself), so absence is not emptiness."""
    params = parse_qs(query, keep_blank_values=True)
    return params["root"][0] if "root" in params else None


def pulses_for(roots: Sequence[Root], now: float,
               desk: DeskLiveness | None) -> dict[str, dict]:
    """Every journaled host's pulse (heartbeat, desk-corrected), keyed by
    bare host name — the join runs_data and fleet_data rows use."""
    pairs = []
    for root in rooted(roots):
        for host in root.store.list_hosts():
            pairs.append((host, root.store.read_host_log(host)))
    return liveness_by_host(pairs, now, desk)


def api(roots: Sequence[Root], route: list[str], folder: str | None,
        panels: list[dict] | None,
        since: float | None = None, now: float | None = None,
        desk: DeskLiveness | None = None,
        params: dict | None = None) -> tuple[object, str]:
    """One route → (payload, status). Route is the path already split and
    unquoted: ["api", "run", <id>, "wave", <n>] and its shorter kin; `folder`
    is the ?root= the link carried."""
    now = time.time() if now is None else now
    match route:
        case ["api", "runs"]:
            rows = runs_data(roots)
            stall_runs(rows, pulses_for(roots, now, desk))
            notes = fleet_notes(roots)
            note_the_fleet(rows, notes, "run_id")
            return {"now": now, "runs": rows}, "200 OK"
        case ["api", "hosts"]:
            return fleet_pulsed(roots, since, now, desk), "200 OK"
        case ["api", "fleet", "page"]:
            return fleet_page(roots, since, now, desk), "200 OK"
        case ["api", "charts", "page"]:
            asked = params or {}
            return charts_page(roots, (asked.get("metric") or ["reward"])[0],
                               (asked.get("q") or [""])[0], now), "200 OK"
        case ["api", "run", run_id, "page"]:
            return run_route(roots, run_id, folder, "unknown run",
                             lambda root: run_page(roots, root, run_id,
                                                   panels, now))
        case ["api", "metrics"]:
            return {"metrics": metric_names(roots)}, "200 OK"
        case ["api", "series"]:
            asked = params or {}
            metric = (asked.get("metric") or ["reward"])[0]
            expr = (asked.get("q") or [""])[0]
            payload = overlay(roots, metric, expr)
            payload["now"] = now
            return payload, "200 OK"
        case ["api", "fleet"]:
            return fleet_throughput([root.store for root in roots],
                                    since=since), "200 OK"
        case ["api", "host", host]:
            page = host_page(in_folder(roots, folder), host, since)
            if page is not None:
                page["pulse"] = pulses_for(roots, now, desk).get(host)
                page["now"] = now
            return _found(page, "unknown host")
        case ["api", "run", run_id]:
            return run_route(roots, run_id, folder, "unknown run",
                             lambda root: run_series(root.store, run_id,
                                                     panels=panels))
        case ["api", "run", run_id, "timing"]:
            return run_route(
                roots, run_id, folder, "unknown run",
                lambda root: (run_timing(root.store, run_id)
                              if root.store.peek_manifest(run_id) is not None
                              else None))
        case ["api", "run", run_id, "waves"]:
            return run_route(roots, run_id, folder, "unknown run",
                             lambda root: wave_list(root.store, run_id))
        case ["api", "run", run_id, "wave", update] if update.isdigit():
            return run_route(roots, run_id, folder, "no such sealed wave",
                             lambda root: wave_detail(root.store, run_id,
                                                      int(update)))
    return {"error": "unknown route"}, NOT_FOUND


def pulses_of_lanes(host: dict, pulses: dict[str, dict]) -> dict[str, dict]:
    """A tenancy lane's only host is the row it sits on: stalling a lane
    consults exactly this host's pulse (the lane rows carry no hosts list,
    so one is synthesized for the join)."""
    for lane in host["tenancy"]:
        lane.setdefault("hosts", [host["host"]])
    return pulses


def holders(roots: Sequence[Root], run_id: str) -> list[Root]:
    """Every root whose store holds this run — a peek, never an attach."""
    return [root for root in roots
            if root.store.peek_manifest(run_id) is not None]


def run_route(roots: Sequence[Root], run_id: str, folder: str | None,
              missing: str, read: Callable[[Root], object | None]):
    """A run route resolved to ONE root, or refused.

    With a folder, that root and no other. Without one, the unique root
    holding the id — and when several hold it, the AMBIGUITY itself: the
    folders, so the page can list them as links. Never a silent pick."""
    if folder is not None:
        chosen = [root for root in roots if root.folder == folder]
        return (_found(read(chosen[0]), missing) if chosen
                else ({"error": "unknown folder"}, NOT_FOUND))
    found = holders(roots, run_id)
    if len(found) > 1:
        return ({"run_id": run_id,
                 "ambiguous": [root.folder for root in found]}, "200 OK")
    return (_found(read(found[0]), missing) if found
            else ({"error": missing}, NOT_FOUND))


def in_folder(roots: Sequence[Root], folder: str | None) -> list[Root]:
    """The roots a host route reads: the named folder's, or all of them —
    a host name is unique per store, not per fleet."""
    return (list(roots) if folder is None
            else [root for root in roots if root.folder == folder])


def host_page(roots: Sequence[Root], host: str,
              since: float | None = None) -> dict | None:
    """One host's journal as its page reads it: the four named readings, plus
    the traffic rails and the moments a timeline marks. `since` windows every
    series, never the identity facts."""
    series = host_series(roots, host, since=since)
    if series is None:
        return None
    events = sorted((event for _, evs in journals_for(roots, host)
                     for event in evs), key=lambda e: e.get("t") or 0.0)
    events = windowed(events, since)
    series["channels"] = traffic_channels(events)
    series["moments"] = moments(events)
    return series


def fleet_pulsed(roots: Sequence[Root], since: float | None, now: float,
                 desk: DeskLiveness | None) -> dict:
    """The fleet listing as /api/hosts answers it: fleet_data with every host
    pulsed, and every run and every lane stalled against those pulses."""
    data = fleet_data(roots, since=since, now=now)
    pulses = pulses_for(roots, now, desk)
    for host in data["hosts"]:
        host["pulse"] = pulses.get(host["host"])
    stall_runs(data["runs"], pulses)
    for host in data["hosts"]:
        stall_runs(host["tenancy"], pulses_of_lanes(host, pulses))
    # what the DESK saw and no host journal could say (ADR 0008, F6): a run
    # parked with what it wants, a row that missed its deadline
    notes = fleet_notes(roots)
    note_the_fleet(data["runs"], notes, "run_id")
    note_the_fleet(data["hosts"], notes, "host")
    for host in data["hosts"]:
        note_the_fleet(host["tenancy"], notes, "run_id")
    data["parked"] = notes["parked"]
    data["unreachable"] = notes["unreachable"]
    data["now"] = now
    return data


# ONE REQUEST PER PAGE. Each page used to assemble itself from two or three
# routes, every one paying its own dispatch, its own pass through the venue's
# reload barrier and its own slot in the container's queue — found live: a
# page ticking three requests every 3 s filled the 32 slots, and the reader's
# own click waited a minute behind them. The single routes stay for anything
# that wants one reading; a page asks for its page.

def fleet_page(roots: Sequence[Root], since: float | None, now: float,
               desk: DeskLiveness | None) -> dict:
    """/hosts in one answer: the listing beside the two aggregates."""
    return {"fleet": fleet_pulsed(roots, since, now, desk),
            "flow": fleet_throughput([root.store for root in roots],
                                     since=since),
            "now": now}


def charts_page(roots: Sequence[Root], metric: str, expr: str,
                now: float) -> dict:
    """/charts in one answer: the metric names, the metric actually drawn
    (the asked one when some run carries it, else the first that exists),
    and its overlay."""
    names = metric_names(roots)
    drawn = metric if metric in names else (names[0] if names else metric)
    series = overlay(roots, drawn, expr)
    series["now"] = now
    return {"metrics": names, "metric": drawn, "series": series}


def run_page(roots: Sequence[Root], root: Root, run_id: str,
             panels: list[dict] | None, now: float) -> dict | None:
    """/run/<id> in one answer: the series, the step timing, the sealed tail,
    and the run's own index ROW (name, tags, folder, hosts), read by peeking
    this run alone rather than walking the index."""
    series = run_series(root.store, run_id, panels=panels)
    if series is None:
        return None
    return {"run": series, "timing": run_timing(root.store, run_id),
            "waves": wave_list(root.store, run_id),
            "row": run_row(roots, run_id, root.folder), "now": now}


def _found(payload: object | None, missing: str) -> tuple[object, str]:
    return ((payload, "200 OK") if payload is not None
            else ({"error": missing}, NOT_FOUND))


def _json(start_response, payload, status: str = "200 OK"):
    body = json.dumps(payload).encode("utf-8")
    start_response(status, [("Content-Type", "application/json"),
                            ("Cache-Control", "no-store")])
    return [body]


def serve(roots: Sequence[Store | Root], port: int = 8321,
          panels_path: str | None = None) -> None:
    """Local serving: python -m rlstack ui <top> [--port N]
    [--panels panels.json] — the file re-reads per refresh, so editing a
    panel and saving shows up on the next poll."""
    import json as _json_mod
    from wsgiref.simple_server import make_server

    def panels():
        try:
            return _json_mod.loads(open(panels_path).read())
        except (OSError, ValueError):
            return []

    app = ui_app(roots, panels=panels if panels_path else None)
    with make_server("127.0.0.1", port, app) as httpd:
        print(f"rlstack ui: http://127.0.0.1:{port}  (ctrl-c to stop)")
        httpd.serve_forever()
