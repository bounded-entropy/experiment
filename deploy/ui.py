"""The observer UI beside the store volume.

    modal deploy deploy/ui.py          # -> https://<workspace>--rlstack-ui.modal.run

Deployment only (I5): an image, a volume MOUNT, and the SAME stdlib WSGI app
`python -m rlstack ui <store-root>` serves locally, over the SAME `LocalStore`
— because a mounted volume IS a POSIX tree, and a store over one is
`data/stores/local.py` with a locator (ADR 0007, Q7). The second Modal store
backend that used to live in this file is retired: a complete `Store`
implementation outside `data/stores/`, outside `open_store` and outside the
fakes suite is a backend nothing tests.

THE ONE VENUE-SPECIFIC THING is freshness. Readers of a Modal volume see a
snapshot until they reload; the store deliberately knows nothing about that
(it commits on write and never reads stale — only an OUTSIDE observer can).

AND THE ONE MEASURED TRAP, kept because the reasoning is the artifact: the
retired SDK-backed store was written after a mount was observed to misbehave
under a RELOAD TAKEN MID-SCAN — runs hopped root -> subdir on click-out, and
reads queued behind a slow reload until /api/runs timed out at 60 s, both
live. So the reload here is taken at ONE point only: just before a cache entry
is rebuilt, under the same lock that already serializes rebuilds, and never
while a response is being assembled. A page therefore reads one consistent
snapshot, at most TTL seconds old.

UNPROVEN: this file has not been deployed since the change (ADR 0007 runs no
metal). If the observer reads stale or hops, the reload placement here is the
first thing to look at, and the SDK-backed reader is in git at 90bbc8e.
"""

import modal

app = modal.App("rlstack-ui")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    # copy=True on BOTH: add_local_python_source ships only .py files, so
    # the web assets ride as their own layer — and a runtime MOUNT of the
    # package tree would shadow any dir added under it, so both go in as
    # image layers, assets after sources (later layer wins). Measured both
    # ways: without the assets every page is a 200 with no body; with a
    # runtime mount the asset route 404s.
    .add_local_python_source("rlstack", copy=True)
    .add_local_dir("rlstack/observe/web", "/root/rlstack/observe/web",
                   copy=True)
)

STORE_MOUNT = "/store"
STORE = "modal://rlstack-store"
TTL = 12.0
"""Seconds a GET's response is served from memory. One reader pays the walk;
every poll inside the window answers instantly (measured before the rewrite:
17 s -> instant)."""


@app.function(image=image, volumes={STORE_MOUNT: store_volume},
              scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=32)
@modal.wsgi_app(label="rlstack-ui")
def ui():
    """The CLI's own app over the mount, behind a response cache."""
    import threading
    import time

    from rlstack.data.stores.modal_volume import ModalVolumeStore
    from rlstack.observe.ui import ui_app

    # The store the metal writes through, read here and never written: the
    # LocalStore over the mount, named by the fleet's locator so `in_stores`
    # says modal://rlstack-store the way every host journal does. Found on
    # the venue: a bare LocalStore takes no locator, every container of the
    # first mount deploy died on this line, and Modal kept serving the old
    # SDK reader in its place — which is why the run page never learned
    # about extents.
    inner = ui_app([ModalVolumeStore(STORE_MOUNT, volume=store_volume,
                                     locator=STORE)])
    cache: dict[str, tuple[float, str, list, bytes]] = {}
    building = threading.Lock()

    def call_inner(key: str, environ: dict) -> tuple[str, list, bytes]:
        """One response built from a FRESHLY RELOADED mount. The reload is
        here and nowhere else: taken before the scan, never during one."""
        caught: dict = {}

        def catch(status, headers, exc_info=None):
            caught["status"], caught["headers"] = status, list(headers)

        try:
            store_volume.reload()
        except Exception:
            pass            # a reload that fails serves the snapshot we have
        body = b"".join(inner(environ, catch))
        cache[key] = (time.time(), caught["status"], caught["headers"], body)
        return caught["status"], caught["headers"], body

    def cached_app(environ, start_response):
        """GET responses cached for TTL seconds; non-GETs and misses pass
        through. One rebuild at a time — late arrivals reuse it."""
        key = (environ.get("PATH_INFO", "") + "?"
               + environ.get("QUERY_STRING", ""))
        if environ.get("REQUEST_METHOD") != "GET":
            return inner(environ, start_response)
        held = cache.get(key)
        if held is not None and time.time() - held[0] < TTL:
            _, status, headers, body = held
        else:
            with building:
                held = cache.get(key)
                if held is not None and time.time() - held[0] < TTL:
                    _, status, headers, body = held
                else:
                    status, headers, body = call_inner(key, environ)
        start_response(status, headers)
        return [body]

    def prewarm() -> None:
        """The hot endpoints never go cold: the walker pays, readers do not."""
        while True:
            for path in ("/api/runs", "/api/hosts", "/api/fleet"):
                try:
                    with building:
                        call_inner(path + "?", {
                            "REQUEST_METHOD": "GET", "PATH_INFO": path,
                            "QUERY_STRING": "", "SERVER_NAME": "prewarm",
                            "SERVER_PORT": "80", "wsgi.url_scheme": "http",
                            "wsgi.input": None, "wsgi.errors": None})
                except Exception:
                    pass
            time.sleep(10.0)

    threading.Thread(target=prewarm, daemon=True).start()
    return cached_app
