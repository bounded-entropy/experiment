"""open_store: a StoreAddress back into a Store — the backends, by name.

A resident (runner/residents.py) is a child process of the metal and must read
the same store the metal reads — an engine's bundle payloads come off the cas —
without being handed a live object, because nothing live crosses a spawn. The
metal ships `store.address()` (a StoreAddress, JSON-safe) and the child calls
`open_store` on it. "One file per backend" already implied an index of them;
this is that index, and the one place a backend name is spelled.
"""

from __future__ import annotations

from rlstack.data.stores.base import Store, StoreAddress
from rlstack.data.stores.local import LocalStore
from rlstack.data.stores.modal_volume import ModalVolumeStore


def open_store(address: StoreAddress) -> Store:
    """The store at `address`, reopened here. A volume-backed store reopens
    as a mount-only view (no volume handle, so nothing it writes would
    persist) — which is exactly a resident's relation to the store: it reads
    cas blobs and journals nothing."""
    if address.backend == "local":
        return LocalStore(address.root)
    if address.backend == "modal_volume":
        return ModalVolumeStore(address.root, volume=None,
                                locator=address.locator)
    raise ValueError(
        f"unknown store backend {address.backend!r}: the backends a resident "
        f"may reopen are spelled in data/stores/address.py, one per file")
