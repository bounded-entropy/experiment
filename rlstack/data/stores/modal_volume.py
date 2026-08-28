"""ModalVolumeStore: the store on a mounted Modal Volume.

A mounted Volume behaves like a local filesystem whose writes are STAGED until
`volume.commit()` persists them, which maps onto the commit point exactly: an
update's blobs, wave, postdata and ledger line all stage on the mount, and
committing right after the ledger append persists them together. A crash
before that loses only work no ledger line sealed — precisely what attach-time
recovery (`_discard_unsealed`) already assumes.

The volume object is passed in and only `.commit()` is called on it, so this
module needs no modal import and the fakes suite can exercise the commit
discipline with a recorder.
"""

from __future__ import annotations

from rlstack.data.stores.local import LocalStore


class ModalVolumeStore(LocalStore):
    """LocalStore over the volume mount, committing at the durable points."""

    def __init__(self, root, volume=None, locator: str | None = None) -> None:
        super().__init__(root)
        self._volume = volume
        self._locator = locator

    def describe(self) -> str:
        """The volume's LOCATOR (modal://<name>), not the mount path — a
        mount path only resolves inside this container; journals must name
        something an outside reader can act on."""
        return self._locator or str(self.root)

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
        elif key.startswith(("hosts/", "fleet/")) or key == "annotations.jsonl":
            self._persist()   # observability and flavortext should survive
                              # the container that wrote them
