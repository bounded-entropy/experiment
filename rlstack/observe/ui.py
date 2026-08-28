"""The observer UI: the routes over the store, with zero core interference.

One stdlib WSGI app — no dependencies, no build step — served either locally or
beside a remote store. Three families of response: the DOCUMENT (index.html,
returned for every page route, because the page routes on location.pathname),
the ASSETS (/web/<file>: the stylesheet and the ES modules, read out of the
package by page.py), and the JSON API polled every few seconds, so live
monitoring is just committed state read again.

Three readings, three families of route. Per EXPERIMENT, where panel priority
IS the run's own dictionary.json with the loss walkback first (I11), and whose
sealed waves are readable one click deep (#56). Per HOST, off the journal
alone. Per FLEET, the aggregate no single host or run knows.

    /  /run/<id>  /run/<id>/wave/<n>  /hosts  /host/<name>
    /api/runs  /api/hosts  /api/fleet  /api/host/<name>
    /api/run/<id>  /api/run/<id>/timing  /api/run/<id>/waves
                                         /api/run/<id>/wave/<n>

The UI never re-derives a declaration, never attaches, and never writes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from urllib.parse import unquote

from rlstack.data.stores.base import Store
from rlstack.observe.aggregate import (
    fleet_throughput, moments, run_timing, traffic_channels,
)
from rlstack.observe.host_series import fleet_data, host_series, journals_for
from rlstack.observe.page import WEB_PREFIX, asset, document
from rlstack.observe.series import run_series
from rlstack.observe.views import runs_data
from rlstack.observe.waves import wave_detail, wave_list

NOT_FOUND = "404 Not Found"


def ui_app(stores: Sequence[Store],
           refresh: Callable[[], None] | None = None,
           panels: Callable[[], list[dict]] | None = None):
    """The WSGI app. `refresh` runs before each API read (a Modal volume
    needs .reload() to see commits from other containers; None for local).
    `panels` supplies derived-graph declarations (None → each store's own
    panels.json, re-read per request so edits appear live)."""

    def app(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        if path.startswith(WEB_PREFIX):
            found = asset(unquote(path[len(WEB_PREFIX):]))
            if found is None:
                start_response(NOT_FOUND, [("Content-Type", "text/plain")])
                return [b"no such asset"]
            body, content_type = found
            start_response("200 OK", [("Content-Type", content_type),
                                      ("Cache-Control", "no-cache")])
            return [body]
        if path.startswith("/api/"):
            if refresh is not None:
                refresh()
            payload, status = api(
                stores, [unquote(part) for part in path.split("/") if part],
                panels() if panels is not None else None)
            return _json(start_response, payload, status)
        # every page is the same document; the modules route on the pathname
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [document()]

    return app


def api(stores: Sequence[Store], route: list[str],
        panels: list[dict] | None) -> tuple[object, str]:
    """One route → (payload, status). Route is the path already split and
    unquoted: ["api", "run", <id>, "wave", <n>] and its shorter kin."""
    match route:
        case ["api", "runs"]:
            return runs_data(stores), "200 OK"
        case ["api", "hosts"]:
            return fleet_data(stores), "200 OK"
        case ["api", "fleet"]:
            return fleet_throughput(stores), "200 OK"
        case ["api", "host", host]:
            return _found(host_page(stores, host), "unknown host")
        case ["api", "run", run_id]:
            return _found(first(stores, lambda store: run_series(
                store, run_id, panels=panels)), "unknown run")
        case ["api", "run", run_id, "timing"]:
            return _found(first(stores, lambda store: (
                run_timing(store, run_id)
                if store.peek_manifest(run_id) is not None else None)),
                "unknown run")
        case ["api", "run", run_id, "waves"]:
            return _found(first(stores, lambda store: wave_list(store, run_id)),
                          "unknown run")
        case ["api", "run", run_id, "wave", update] if update.isdigit():
            return _found(first(stores, lambda store: wave_detail(
                store, run_id, int(update))), "no such sealed wave")
    return {"error": "unknown route"}, NOT_FOUND


def host_page(stores: Sequence[Store], host: str) -> dict | None:
    """One host's journal as its page reads it: the four named readings, plus
    the traffic rails and the moments a timeline marks."""
    series = host_series(stores, host)
    if series is None:
        return None
    events = sorted((event for _, evs in journals_for(stores, host)
                     for event in evs), key=lambda e: e.get("t") or 0.0)
    series["channels"] = traffic_channels(events)
    series["moments"] = moments(events)
    return series


def first(stores: Sequence[Store], read: Callable[[Store], object | None]):
    """The first store that has the thing. One experiment, one store (I10) —
    a run in two stores is a fork the runs view already flags."""
    for store in stores:
        found = read(store)
        if found is not None:
            return found
    return None


def _found(payload: object | None, missing: str) -> tuple[object, str]:
    return ((payload, "200 OK") if payload is not None
            else ({"error": missing}, NOT_FOUND))


def _json(start_response, payload, status: str = "200 OK"):
    body = json.dumps(payload).encode("utf-8")
    start_response(status, [("Content-Type", "application/json"),
                            ("Cache-Control", "no-store")])
    return [body]


def serve(stores: Sequence[Store], port: int = 8321,
          panels_path: str | None = None) -> None:
    """Local serving: python -m rlstack ui <store> [--port N]
    [--panels panels.json] — the file re-reads per refresh, so editing a
    panel and saving shows up on the next poll."""
    import json as _json_mod
    from wsgiref.simple_server import make_server

    def panels():
        try:
            return _json_mod.loads(open(panels_path).read())
        except (OSError, ValueError):
            return []

    app = ui_app(stores, panels=panels if panels_path else None)
    with make_server("127.0.0.1", port, app) as httpd:
        print(f"rlstack ui: http://127.0.0.1:{port}  (ctrl-c to stop)")
        httpd.serve_forever()
