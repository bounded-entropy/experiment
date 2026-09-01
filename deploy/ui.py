"""The observer UI beside the store volume.

    modal deploy deploy/ui.py          # -> https://<workspace>--rlstack-ui.modal.run

Deployment only (I5): an image, a volume, and the SAME stdlib WSGI app
`python -m rlstack ui <store-root>` serves locally, handed the volume's store
instead. The one venue-specific line is the per-request volume.reload():
readers of a Modal volume see a snapshot until they reload, the store
deliberately knows nothing about that (it commits on write and never reads
stale — only an OUTSIDE observer can), so freshness is this file's job.
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


@app.function(image=image, volumes={"/store": store_volume},
              scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=32)
@modal.wsgi_app(label="rlstack-ui")
def ui():
    from rlstack.data.stores.base import Store, StoreError
    from rlstack.observe.ui import ui_app

    class VolumeReadStore(Store):
        """The observer's store with NO MOUNT AT ALL: every byte verb rides
        the volume SDK's committed view, which is always fresh — no reload
        tick (a reload half-invalidates a mount mid-scan: runs hopped
        root -> subdir on click-out), no gate (reads queued behind a slow
        reload until /api/runs timed out at 60s, both observed live).
        CachedReadStore above makes repeat reads free: immutable bytes are
        fetched once ever, journals revalidate on the size a directory
        listing already carries."""

        def __init__(self) -> None:
            super().__init__()
            self._entry_sizes: dict[str, int] = {}

        def describe(self) -> str:
            return "modal://rlstack-store"

        def _read(self, key: str) -> bytes:
            try:
                return b"".join(store_volume.read_file(key))
            except Exception:
                raise FileNotFoundError(key) from None

        def _list(self, prefix: str) -> list[str]:
            out: list[str] = []
            try:
                entries = store_volume.listdir(prefix.rstrip("/"),
                                               recursive=True)
            except Exception:
                return out
            for entry in entries:
                path = entry.path.lstrip("/")
                size = getattr(entry, "size", None)
                if size is not None:
                    self._entry_sizes[path] = int(size)
                out.append(path)
            return sorted(out)

        def _exists(self, key: str) -> bool:
            try:
                self._size(key)
                return True
            except FileNotFoundError:
                return False

        def _size(self, key: str) -> int:
            parent = key.rsplit("/", 1)[0] if "/" in key else ""
            try:
                for entry in store_volume.listdir(parent):
                    path = entry.path.lstrip("/")
                    size = getattr(entry, "size", None)
                    if size is not None:
                        self._entry_sizes[path] = int(size)
            except Exception:
                raise FileNotFoundError(key) from None
            if key in self._entry_sizes:
                return self._entry_sizes[key]
            raise FileNotFoundError(key)

        def _write(self, key: str, data: bytes) -> None:
            raise StoreError("the observer writes nothing")

        def _append_line(self, key: str, line: str) -> None:
            raise StoreError("the observer writes nothing")

        def _delete(self, key: str) -> None:
            raise StoreError("the observer deletes nothing")

    import threading
    import time

    inner = ui_app([VolumeReadStore()])
    cache: dict[str, tuple[float, str, list, bytes]] = {}
    TTL = 12.0
    building = threading.Lock()

    def call_inner(key: str, environ: dict) -> tuple[str, list, bytes]:
        caught: dict = {}

        def catch(status, headers, exc_info=None):
            caught["status"], caught["headers"] = status, list(headers)

        body = b"".join(inner(environ, catch))
        cache[key] = (time.time(), caught["status"], caught["headers"], body)
        return caught["status"], caught["headers"], body

    def cached_app(environ, start_response):
        """GET responses cached for TTL seconds: one reader pays the SDK
        walk, every poll inside the window answers from memory (measured:
        17s -> instant). Non-GETs and cache misses pass through."""
        key = (environ.get("PATH_INFO", "") + "?"
               + environ.get("QUERY_STRING", ""))
        if environ.get("REQUEST_METHOD") != "GET":
            return inner(environ, start_response)
        held = cache.get(key)
        if held is not None and time.time() - held[0] < TTL:
            _, status, headers, body = held
        else:
            with building:      # one rebuild at a time; late arrivals reuse
                held = cache.get(key)
                if held is not None and time.time() - held[0] < TTL:
                    _, status, headers, body = held
                else:
                    status, headers, body = call_inner(key, environ)
        start_response(status, headers)
        return [body]

    def prewarm() -> None:
        # the hot endpoint never goes cold: the walker pays, readers don't
        while True:
            try:
                call_inner("/api/runs?", {
                    "REQUEST_METHOD": "GET", "PATH_INFO": "/api/runs",
                    "QUERY_STRING": "", "SERVER_NAME": "prewarm",
                    "SERVER_PORT": "80", "wsgi.url_scheme": "http",
                    "wsgi.input": None, "wsgi.errors": None})
            except Exception:
                pass
            time.sleep(10.0)

    threading.Thread(target=prewarm, daemon=True).start()
    return cached_app
