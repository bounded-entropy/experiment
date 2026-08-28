"""The observer UI: the routes over the store, with zero core interference.

One stdlib WSGI app — no dependencies, no build step — served either locally or
beside a remote store, with page.py as the one document every route returns and
the JSON API polled every few seconds, so live monitoring is just committed
state read again. Two readings, two families of route: per EXPERIMENT, where
panel priority IS the run's own dictionary.json with the loss walkback first
(I11); and per HOST, off the journal alone. The UI never re-derives a
declaration and never reads anything but peeks and journals.

    /  /run/<id>  /hosts  /host/<name>       and /api/ beside each
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from urllib.parse import unquote

from rlstack.data.stores.base import Store
from rlstack.observe.host_series import fleet_data, host_series
from rlstack.observe.page import PAGE
from rlstack.observe.series import run_series
from rlstack.observe.views import runs_data


def ui_app(stores: Sequence[Store],
           refresh: Callable[[], None] | None = None,
           panels: Callable[[], list[dict]] | None = None):
    """The WSGI app. `refresh` runs before each API read (a Modal volume
    needs .reload() to see commits from other containers; None for local).
    `panels` supplies derived-graph declarations (None → each store's own
    panels.json, re-read per request so edits appear live)."""

    def app(environ, start_response):
        path = environ.get("PATH_INFO", "/")
        if path.startswith("/api/"):
            if refresh is not None:
                refresh()
            if path == "/api/runs":
                return _json(start_response, runs_data(stores))
            if path == "/api/hosts":
                return _json(start_response, fleet_data(stores))
            if path.startswith("/api/run/"):
                run_id = unquote(path[len("/api/run/"):])
                for store in stores:
                    series = run_series(
                        store, run_id,
                        panels=panels() if panels is not None else None)
                    if series is not None:
                        return _json(start_response, series)
                return _json(start_response, {"error": "unknown run"}, "404 Not Found")
            if path.startswith("/api/host/"):
                series = host_series(stores, unquote(path[len("/api/host/"):]))
                if series is not None:
                    return _json(start_response, series)
                return _json(start_response, {"error": "unknown host"}, "404 Not Found")
            return _json(start_response, {"error": "unknown route"}, "404 Not Found")
        # every page is the same document; JS routes on location.pathname
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [PAGE.encode("utf-8")]

    return app


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
