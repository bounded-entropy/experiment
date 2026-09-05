"""Read caching for the observer: polls stop re-reading the world.

The observer's cost model broke on remote mounts: every poll re-read every
manifest, ledger and journal over a network filesystem, so /api/runs grew
linear in (runs x file latency) and a 3s poll could take minutes. The fix is
the store's own write discipline, read back. A store's files are either
IMMUTABLE once present (manifests, dictionaries, sealed waves, postdata, cas
blobs — the ledger's commit is what seals them, I10) or APPEND-ONLY journals
(the ledger, host logs, the fleet log, annotations — single writers, one
line at a time). So immutable bytes cache forever, append-only bytes
revalidate on SIZE alone (`_size`, the stat-shaped byte verb: an append
grows a file and nothing else may touch it — a file that SHRANK is a
rewritten store and drops its entry), a miss holds briefly (half-born runs
settle), and directory listings hold a short TTL so a new run appears on
the next tick.

CachedReadStore is an OBSERVER'S store: every write verb refuses (I11 — the
observer interferes with nothing). It caches only the byte primitives and
inherits the whole reading grammar from Store, so every peek answers
byte-identically to the uncached store — just never slower than the files
actually change.
"""

from __future__ import annotations

import time

from rlstack.data.stores.base import Store, StoreError

LIST_TTL = 2.0      # seconds a directory listing may serve stale
ABSENT_TTL = 2.0    # seconds a miss may serve stale


def immutable(key: str) -> bool:
    """Sealed-once keys, cached forever: cas blobs by content address, the
    rest by the store's write-once rule — a committed artifact is never
    rewritten, so its first bytes are its only bytes."""
    return (key.startswith("cas/")
            or (key.endswith(("manifest.json", "dictionary.json"))
                and not deletable(key))
            or "/waves/" in key or "/postdata/" in key)


def deletable(key: str) -> bool:
    """The ONE deletable tree: measurements are observations, superseded or
    removed at will (#70) — so for them, and only them, absence is believed
    (after ABSENT_TTL) instead of outranked by held bytes."""
    return key.startswith("measurements/")


class CachedReadStore(Store):
    """The reading grammar over cached byte verbs, wrapping a real store."""

    def __init__(self, inner: Store) -> None:
        self.inner = inner
        self._bytes: dict[str, tuple[int, bytes]] = {}    # key -> (size, data)
        self._absent: dict[str, float] = {}               # key -> noticed at
        self._lists: dict[str, tuple[float, list[str]]] = {}
        self._homes: tuple[float, dict[str, str]] | None = None

    def describe(self) -> str:
        return self.inner.describe()

    def _run_directories(self) -> dict[str, str]:
        """THE INNER STORE'S OWN WALK, memoized like a listing. The base
        answer walks _list("runs/"), and through this wrapper that re-listed
        the whole tree every time the LIST_TTL memo lapsed — found live:
        25,131 stats, 9 s, on every index request more than two seconds
        after the last, with LocalStore's manifest-pruned walk sitting unused
        underneath. The reload blink is honoured as for listings: a walk
        that finds no run right after one that found many stands one more
        tick."""
        held = self._homes
        if held is not None and time.time() - held[0] < LIST_TTL:
            return held[1]
        homes = self.inner._run_directories()
        if not homes and held is not None and held[1]:
            self._homes = (time.time(), held[1])
            return held[1]
        self._homes = (time.time(), homes)
        return homes

    # ---- cached byte verbs --------------------------------------------------

    def _read(self, key: str) -> bytes:
        held = self._bytes.get(key)
        if held is not None:
            if immutable(key):
                return held[1]
            try:
                if self.inner._size(key) == held[0]:
                    return held[1]
            except FileNotFoundError:
                if deletable(key):
                    self._bytes.pop(key, None)
                    self._absent[key] = time.time()
                    raise
                # THE BLINK: a mount reload makes the whole tree transiently
                # absent, and nothing else in this store ever deletes — so
                # bytes we HELD outrank a momentary 404, and the next
                # changed size re-reads honestly
                return held[1]
        noticed = self._absent.get(key)
        if noticed is not None and time.time() - noticed < ABSENT_TTL:
            raise FileNotFoundError(key)
        try:
            data = self.inner._read(key)
        except FileNotFoundError:
            self._absent[key] = time.time()
            raise
        self._absent.pop(key, None)
        self._bytes[key] = (len(data), data)
        return data

    def _exists(self, key: str) -> bool:
        if immutable(key) and key in self._bytes:
            return True
        return self.inner._exists(key)

    def _size(self, key: str) -> int:
        return self.inner._size(key)

    def _list(self, prefix: str) -> list[str]:
        held = self._lists.get(prefix)
        if held is not None and time.time() - held[0] < LIST_TTL:
            return held[1]
        keys = self.inner._list(prefix)
        if (not keys and held is not None and held[1]
                and not deletable(prefix)):
            # a populated tree does not empty itself (runs and journals are
            # never deleted; measurements/ is the exception and skips this):
            # a blank listing right after a full one is the reload blink, so
            # the last real listing stands one more tick
            self._lists[prefix] = (time.time(), held[1])
            return held[1]
        self._lists[prefix] = (time.time(), keys)
        return keys

    # ---- the observer never writes ------------------------------------------

    def _write(self, key: str, data: bytes) -> None:
        raise StoreError("the observer's store never writes")

    def _append_line(self, key: str, line: str) -> None:
        raise StoreError("the observer's store never writes")

    def _delete(self, key: str) -> None:
        raise StoreError("the observer's store never writes")
