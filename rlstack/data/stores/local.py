"""LocalStore: the key tree on a local filesystem, with real durability.

The store's seven byte verbs under the tmp+fsync+rename discipline: readers see
old bytes or new bytes, never a tear; renames are recorded in the parent
directory; ledger appends are flushed and fsynced. Every verb lands durably as
it is called, so the durability hook (`_persist`) has nothing to do here.
"""

from __future__ import annotations

import os
from pathlib import Path

from rlstack.data.stores.base import Store, StoreAddress


class LocalStore(Store):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        return str(self.root)

    def address(self) -> StoreAddress:
        return StoreAddress("local", str(self.root), self.describe())

    def path_of(self, key: str) -> Path:
        """The on-disk path for a key (for tools and tests)."""
        return self.root / key

    # ---- the verbs ----------------------------------------------------------

    def _read(self, key: str) -> bytes:
        return self.path_of(key).read_bytes()

    def _write(self, key: str, data: bytes) -> None:
        path = self.path_of(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            with open(tmp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            _fsync_dir(path.parent)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _append_line(self, key: str, line: str) -> None:
        path = self.path_of(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            if line:
                handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _exists(self, key: str) -> bool:
        return self.path_of(key).exists()

    def _list(self, prefix: str) -> list[str]:
        root = self.path_of(prefix)
        if not root.exists():
            return []
        return sorted(str(path.relative_to(self.root)).replace(os.sep, "/")
                      for path in root.rglob("*") if path.is_file())

    def _run_directories(self) -> dict[str, str]:
        """A RUN DIRECTORY IS A LEAF: the walk turns back the moment it sees a
        manifest, so a store's waves, rollouts and adapters are never listed
        to find its runs. Measured on the venue before this: rglob stat'd
        23,078 entries, 12 s, to find 109 manifests — on every index request."""
        top = self.path_of("runs")
        if not top.exists():
            return {}
        out: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(top):
            if "manifest.json" in filenames:
                home = str(Path(dirpath).relative_to(self.root)).replace(os.sep, "/")
                out[home.rsplit("/", 1)[-1]] = home
                dirnames[:] = []          # nothing beneath a run is a run
        return out

    def _delete(self, key: str) -> None:
        self.path_of(key).unlink()

    def _size(self, key: str) -> int:
        return self.path_of(key).stat().st_size

    def _sweep_partial(self, prefix: str) -> None:
        """Stray *.tmp files from interrupted atomic writes."""
        root = self.path_of(prefix)
        if root.exists():
            for tmp in root.rglob("*.tmp"):
                tmp.unlink()


def _fsync_dir(path: Path) -> None:
    """Durably record a rename in the parent directory."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
