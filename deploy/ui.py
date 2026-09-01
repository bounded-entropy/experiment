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
    import threading
    import time

    from rlstack import ModalVolumeStore
    from rlstack.observe.ui import ui_app

    gate = threading.Lock()

    class GatedStore(ModalVolumeStore):
        """Every read holds the gate the refresher's reload takes: a reload
        landing DURING a scan half-invalidates the mount under it, and the
        runs list then files runs at the store's top for one response
        (observed live: a run hopping root -> subdir on click-out). Reads
        gate one call at a time — the wide window (a whole directory scan)
        is closed; the between-calls sliver is covered by the attach event's
        own subdir."""

        def _read(self, key):
            with gate:
                return super()._read(key)

        def _list(self, prefix):
            with gate:
                return super()._list(prefix)

        def _exists(self, key):
            with gate:
                return super()._exists(key)

    store = GatedStore("/store", volume=store_volume,
                       locator="modal://rlstack-store")

    def refresher() -> None:
        # Freshness rides a BACKGROUND tick, never the request path — and
        # the tick takes the same gate the reads hold, so a reload waits for
        # the in-flight call instead of invalidating the mount under it.
        # The observer serves a snapshot at most ~15s stale, which its
        # stale-tail tolerance already licenses.
        while True:
            time.sleep(15.0)
            try:
                with gate:
                    store_volume.reload()
            except Exception:
                pass

    threading.Thread(target=refresher, daemon=True).start()
    return ui_app([store])
