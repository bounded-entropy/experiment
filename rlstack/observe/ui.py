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

import json
from collections.abc import Callable, Sequence
from urllib.parse import parse_qs, unquote

from rlstack.data.stores.base import Store
from rlstack.observe.aggregate import (
    fleet_throughput, moments, run_timing, traffic_channels,
)
from rlstack.observe.host_series import fleet_data, host_series, journals_for
from rlstack.observe.locate import Root, rooted
from rlstack.observe.page import WEB_PREFIX, asset, document
from rlstack.observe.series import run_series
from rlstack.observe.views import runs_data
from rlstack.observe.waves import wave_detail, wave_list

NOT_FOUND = "404 Not Found"


def ui_app(roots: Sequence[Store | Root],
           refresh: Callable[[], None] | None = None,
           panels: Callable[[], list[dict]] | None = None):
    """The WSGI app over one top directory's roots (or one bare store, the
    degenerate case). `refresh` runs before each API read (a Modal volume
    needs .reload() to see commits from other containers; None for local).
    `panels` supplies derived-graph declarations (None → each store's own
    panels.json, re-read per request so edits appear live)."""
    known = rooted(roots)

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
                known, [unquote(part) for part in path.split("/") if part],
                asked_folder(environ.get("QUERY_STRING", "")),
                panels() if panels is not None else None)
            return _json(start_response, payload, status)
        # every page is the same document; the modules route on the pathname
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [document()]

    return app


def asked_folder(query: str) -> str | None:
    """?root=<folder> — the folder the link carried, or None when it carried
    none. "" is a folder (the top itself), so absence is not emptiness."""
    params = parse_qs(query, keep_blank_values=True)
    return params["root"][0] if "root" in params else None


def api(roots: Sequence[Root], route: list[str], folder: str | None,
        panels: list[dict] | None) -> tuple[object, str]:
    """One route → (payload, status). Route is the path already split and
    unquoted: ["api", "run", <id>, "wave", <n>] and its shorter kin; `folder`
    is the ?root= the link carried."""
    match route:
        case ["api", "runs"]:
            return runs_data(roots), "200 OK"
        case ["api", "hosts"]:
            return fleet_data(roots), "200 OK"
        case ["api", "fleet"]:
            return fleet_throughput([root.store for root in roots]), "200 OK"
        case ["api", "host", host]:
            return _found(host_page(in_folder(roots, folder), host),
                          "unknown host")
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


def host_page(roots: Sequence[Root], host: str) -> dict | None:
    """One host's journal as its page reads it: the four named readings, plus
    the traffic rails and the moments a timeline marks."""
    series = host_series(roots, host)
    if series is None:
        return None
    events = sorted((event for _, evs in journals_for(roots, host)
                     for event in evs), key=lambda e: e.get("t") or 0.0)
    series["channels"] = traffic_channels(events)
    series["moments"] = moments(events)
    return series


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
