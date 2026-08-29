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
    .add_local_python_source("rlstack")
)


@app.function(image=image, volumes={"/store": store_volume},
              scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=32)
@modal.wsgi_app(label="rlstack-ui")
def ui():
    from rlstack import ModalVolumeStore
    from rlstack.observe.ui import ui_app

    store = ModalVolumeStore("/store", volume=store_volume,
                             locator="modal://rlstack-store")
    inner = ui_app([store])

    def fresh(environ, start_response):
        try:
            store_volume.reload()      # see every commit since the last request
        except Exception:
            # a CONCURRENT request's read holds a file open, and Modal refuses
            # to reload under one. Serve the current snapshot instead: the
            # observer tolerates stale tails by charter, and the next quiet
            # request refreshes.
            pass
        return inner(environ, start_response)

    return fresh
