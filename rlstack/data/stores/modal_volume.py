"""ModalVolumeStore: the run store on a mounted Modal Volume.

A Modal Volume mounted in a container behaves like a local filesystem whose
writes are STAGED until `volume.commit()` persists them — which maps exactly
onto our commit protocol: everything an update writes (blobs, waves,
postdata, the ledger line) is staged on the mount, and committing right after
the ledger append persists all of it together. A crash before the commit
loses only work the ledger never sealed, which is precisely what attach-time
recovery (`_discard_unsealed`) assumes.

The volume object is passed in (duck-typed: anything with .commit()), so this
module needs no modal import and the fakes suite can exercise the commit
discipline with a recorder.
"""

from __future__ import annotations

from rlstack.data.stores.local import LocalStore


class ModalVolumeStore(LocalStore):
    """LocalStore over the volume mount + commit at the durable points."""

    def __init__(self, root, volume=None) -> None:
        super().__init__(root)
        self._volume = volume

    def _persist(self) -> None:
        if self._volume is not None:
            self._volume.commit()

    def _write(self, key: str, data: bytes) -> None:
        super()._write(key, data)
        # a run must exist durably once created; eval output is firewalled
        # measurement, persisted as it lands (never part of crash recovery)
        if key.endswith("manifest.json") or "/eval/" in key:
            self._persist()

    def _append_line(self, key: str, line: str) -> None:
        super()._append_line(key, line)
        if key.endswith("ledger.jsonl"):
            self._persist()   # THE commit point: seals the whole update
        elif key.startswith("hosts/"):
            self._persist()   # observability should survive the container
