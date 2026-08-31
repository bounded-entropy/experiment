"""ModalVolumeStore: the store on a mounted Modal Volume.

A mounted Volume behaves like a local filesystem whose writes are STAGED until
`volume.commit()` persists them, which maps onto the commit point exactly: an
update's blobs, wave, postdata and ledger line all stage on the mount, and
committing right after the ledger append persists them together. A crash
before that loses only work no ledger line sealed — precisely what attach-time
recovery (`_discard_unsealed`) already assumes.

A DELETION STAGES EXACTLY LIKE A WRITE, which is what makes retention work
here at all: a sweep's unlinks are invisible to the volume until a commit, so
without one the freed bytes come back with the next container. `_persist` is
the Store's durability hook and this class's whole answer to it — the sweep
calls it once, after the batch, and it commits the same way the ledger line
does. Committing there is safe for the same reason it is safe at the ledger:
it persists whatever else is staged, and the sweep runs immediately after a
commit, so there is nothing else.

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
        """The durability hook: stage -> volume. Called at every durable point
        below, and once by a sweep, whose deletions stage like writes."""
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
        elif (key.startswith(("hosts/", "fleet/", "measurements/"))
              or key == "annotations.jsonl"):
            self._persist()   # observability, measurement and flavortext
                              # should survive the container that wrote them
