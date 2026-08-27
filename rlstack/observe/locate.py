"""store_for(locator): the principle is that a store is NAMED by a locator,
and the reader must run somewhere that locator resolves.

    /path, file:///path   LocalStore — resolves wherever that filesystem is
                          mounted (a Modal volume IS one, inside a container)
    s3://bucket/prefix    an S3Store subclass, when it lands — resolves
                          anywhere (network API)
    modal://volume        does NOT resolve locally, on purpose: run the
                          reader beside the volume (deploy's hosts hook /
                          an ASGI observer) instead of pretending to fetch it
"""

from __future__ import annotations

from rlstack.data.stores.base import Store
from rlstack.data.stores.local import LocalStore


def store_for(locator: str) -> Store:
    if locator.startswith("file://"):
        return LocalStore(locator[len("file://"):])
    if locator.startswith("s3://"):
        raise NotImplementedError(
            f"{locator}: no S3 store backend yet (the Store ABC's byte verbs "
            f"are where it lands — data/stores/, one file per backend)")
    if locator.startswith("modal://"):
        raise NotImplementedError(
            f"{locator} does not resolve outside a container. Run the reader "
            f"beside the volume: `modal run deploy/modal_app.py::hosts`, or "
            f"an observer deployed with the volume mounted.")
    return LocalStore(locator)
