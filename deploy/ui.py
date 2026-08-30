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

    store = ModalVolumeStore("/store", volume=store_volume,
                             locator="modal://rlstack-store")

    def refresher() -> None:
        # Freshness rides a BACKGROUND tick, never the request path: a reload
        # invalidates the very handles an in-flight scan holds open, so
        # per-request reloads made the runs list drop random rows (measured —
        # 8 of 40 resolving, then 39). A tick that lands between requests
        # succeeds; one that collides fails harmlessly and the next tries
        # again. The observer serves a snapshot at most ~15s stale, which its
        # stale-tail tolerance already licenses.
        while True:
            time.sleep(15.0)
            try:
                store_volume.reload()
            except Exception:
                pass

    threading.Thread(target=refresher, daemon=True).start()
    return ui_app([store])
