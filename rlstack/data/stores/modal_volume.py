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

The volume object is passed in and only `.commit()` and `.read_file()` are
called on it, so this module needs no modal import and the fakes suite can
exercise the commit discipline with a recorder.

READS FALL THROUGH TO THE VOLUME, NEVER RELOAD THE MOUNT: a mounted Volume is
a boot-time snapshot, so a blob committed by another container (the desk's
sliced plan, a learner's sealed bundle head) may be missing from this mount.
`volume.reload()` would surface it — by invalidating the whole mount for a
moment, which shoots every OTHER tenant's in-flight write on this container
(the 12-arm wave died of exactly that: each later adopt's reload killed the
earlier arms' writers mid-mkdir). `_read` instead falls through to
`volume.read_file()` — an RPC against the committed view that touches nothing
mounted — so a miss on the snapshot is served fresh and running tenants never
feel another tenant's arrival.
"""

from __future__ import annotations

import concurrent.futures
import threading

from rlstack.data.stores.base import StoreAddress
from rlstack.data.stores.local import LocalStore


class ModalVolumeStore(LocalStore):
    """LocalStore over the volume mount, committing at the durable points.

    EVERY VOLUME RPC RUNS ON THIS STORE'S OWN WORKER THREAD, never the
    caller's: a caller on an event loop (a host's async serve path
    journaling, an async adopt sealing) that blocks in modal's client
    deadlocks the very loop modal needs to finish the call — the metals'
    journal-freeze wedge. The detour costs one thread hop from sync
    callers and, being a single worker, also serializes volume requests.
    Writers waiting before a commit starts share that commit; a writer
    arriving during it waits for the next one.
    """

    def __init__(self, root, volume=None, locator: str | None = None) -> None:
        super().__init__(root)
        self._volume = volume
        self._locator = locator
        self._volume_thread = concurrent.futures.ThreadPoolExecutor(
            max_workers=1)
        self._commit_lock = threading.Lock()
        self._pending_commit: concurrent.futures.Future[None] | None = None

    def _on_the_volume_thread(self, fn):
        """The one door to the volume's RPCs (see the class docstring)."""
        return self._volume_thread.submit(fn).result()

    def describe(self) -> str:
        """The volume's LOCATOR (modal://<name>), not the mount path — a
        mount path only resolves inside this container; journals must name
        something an outside reader can act on."""
        return self._locator or str(self.root)

    def address(self) -> StoreAddress:
        """Reopened by a resident as a MOUNT-ONLY view (open_store hands no
        volume handle): a resident reads cas blobs off the mount and journals
        nothing — the metal process is the writer, and its commits are what
        make the mount current."""
        return StoreAddress("modal_volume", str(self.root), self.describe())

    def _persist(self) -> None:
        """Wait for a commit that starts after this writer staged its work.

        Several writers queued behind a volume read need one commit, not
        one each. Joining a commit already in progress would acknowledge
        later writes too early, so its group closes when the worker starts.
        """
        if self._volume is not None:
            with self._commit_lock:
                if self._pending_commit is None:
                    self._pending_commit = self._volume_thread.submit(
                        self._commit_waiting_writers)
                pending = self._pending_commit
            pending.result()

    def _commit_waiting_writers(self) -> None:
        """Close this group before the RPC; subsequent writers form a new one."""
        with self._commit_lock:
            self._pending_commit = None
        self._volume.commit()

    def open_run(self, run_id, manifest=None, subdir=None, *, create=None):
        """An adopting writer starts from the committed ledger, never its mount.

        The runner offers the manifest on both creation and resume. Before
        resuming an existing run, copy its committed ledger into this mount:
        reading through a missing file alone is insufficient because the next
        local append would otherwise create a ledger containing only that line.
        The desk must have ended the old writer's custody before adoption.
        """
        if self._volume is not None and manifest is not None:
            home = self.run_prefix(run_id, subdir)
            if self._exists(f"{home}/manifest.json"):
                key = f"{home}/ledger.jsonl"

                def committed_ledger():
                    entries = self._volume.iterdir(home, recursive=False)
                    if not any(entry.path.lstrip("/") == key for entry in entries):
                        return b""  # creation can crash before its first append
                    return b"".join(self._volume.read_file(key))

                # Observation failures propagate: unknown history must never
                # be replaced by a stale local prefix or an empty ledger.
                data = self._on_the_volume_thread(committed_ledger)
                super()._write(key, data)
        return super().open_run(run_id, manifest=manifest, subdir=subdir, create=create)

    def _read(self, key: str) -> bytes:
        """Mount first; on a snapshot miss, the volume's committed view via
        read_file — the reload-free answer to cross-container blobs (see the
        module docstring for why reload is forbidden here)."""
        try:
            return super()._read(key)
        except FileNotFoundError:
            if self._volume is None:
                raise
            try:
                return self._on_the_volume_thread(
                    lambda: b"".join(self._volume.read_file(key)))
            except Exception:
                raise FileNotFoundError(
                    f"{key!r}: on neither the mount snapshot nor the "
                    f"volume's committed view") from None

    def _exists(self, key: str) -> bool:
        """A missing mounted file may already exist in the committed volume."""
        if super()._exists(key):
            return True
        try:
            self._read(key)
            return True
        except FileNotFoundError:
            return False

    def _run_directories(self) -> dict[str, str]:
        """Discover parents committed after this container mounted its snapshot."""
        homes = super()._run_directories()
        if self._volume is not None:
            def committed_entries():
                roots = self._volume.iterdir("/", recursive=False)
                if not any(entry.path.strip("/") == "runs" for entry in roots):
                    return []
                return list(self._volume.iterdir("runs", recursive=True))

            entries = self._on_the_volume_thread(committed_entries)
            for entry in entries:
                if entry.path.endswith("/manifest.json"):
                    home = entry.path.removesuffix("/manifest.json").lstrip("/")
                    homes[home.removeprefix("runs/")] = home
        return homes

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
