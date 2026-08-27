"""SPEC.md §2D store: the membrane's durable side.

    runs/<run_id>/{manifest.json, ledger.jsonl, rollouts/<update>.jsonl.gz,
                   adapters/<name>@<v>.bin, optim/<name>@<v>.bin, eval/<v>/}
    cas/<sha256>/blob

Invariants: identity is computed, never typed (I3); every write is atomic
(tmp + os.replace + fsync); the ledger is append-only and strictly increasing;
resume = attach + ledger tail; work not committed by the ledger is UNSEALED and is
discarded on open. Optimizer moments are a store primitive committed in lockstep
with the ledger (decision #14), hence the "optim" blob section.

Stdlib only; Phase B swaps the .bin blob payloads for safetensors and the gzipped
jsonl rollouts for parquet without changing this surface.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BLOB_SECTIONS = ("adapters", "optim")


class StoreError(RuntimeError):
    """Store invariant violation."""


class ManifestMismatch(StoreError):
    """An existing run's manifest differs from the one offered on attach (I3)."""


class LedgerError(StoreError, ValueError):
    """Malformed or non-monotonic ledger entry (the ledger is append-only)."""


def _canonical(obj: Any) -> str:
    """Canonical JSON: sorted keys, tight separators, real UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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


def _atomic_write(path: Path, data: bytes) -> None:
    """tmp + fsync + os.replace: readers see the old bytes or the new ones, never a tear."""
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


def _gzip_jsonl(rows: list[dict[str, Any]]) -> bytes:
    """Rows -> deterministic gzip of canonical jsonl."""
    raw = "".join(_canonical(row) + "\n" for row in rows).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        gz.write(raw)
    return buffer.getvalue()


class Store:
    """A run store rooted at `root`: runs/ plus a content-addressed cas/."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.cas_dir = self.root / "cas"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.cas_dir.mkdir(parents=True, exist_ok=True)

    # ---------- runs ----------

    def open_run(self, run_id: str, manifest: dict[str, Any] | None = None) -> RunHandle:
        """Create runs/<run_id>/ (manifest required) or attach to it (manifest checked).

        Attaching also discards unsealed work: rollout shards and blob versions that
        the ledger never committed.
        """
        path = self.runs_dir / run_id
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            if manifest is None:
                raise StoreError(f"run {run_id!r} does not exist; a manifest is required to create it")
            for section in ("rollouts", "adapters", "optim", "eval"):
                (path / section).mkdir(parents=True, exist_ok=True)
            _atomic_write(manifest_path, _canonical(manifest).encode("utf-8"))
            (path / "ledger.jsonl").touch()
            _fsync_dir(path)
            return RunHandle(self, run_id, path, _canonical(manifest))

        stored = manifest_path.read_text(encoding="utf-8")
        if manifest is not None and _canonical(manifest) != _canonical(json.loads(stored)):
            raise ManifestMismatch(
                f"run {run_id!r} exists with a different manifest (identity is computed, I3)"
            )
        handle = RunHandle(self, run_id, path, stored)
        handle._discard_unsealed()
        return handle

    def list_runs(self) -> list[str]:
        """Run ids present in the store."""
        return sorted(p.name for p in self.runs_dir.iterdir() if (p / "manifest.json").exists())

    # ---------- content-addressed storage ----------

    def fingerprint(self, data: bytes) -> str:
        """sha256 hex of `data` -- the store's only identity function (I3)."""
        return hashlib.sha256(data).hexdigest()

    def cas_put(self, data: bytes) -> str:
        """Store `data` under its own hash; identical bytes dedupe. Returns cas://<sha>."""
        digest = self.fingerprint(data)
        blob = self.cas_dir / digest / "blob"
        if not blob.exists():
            _atomic_write(blob, data)
        return f"cas://{digest}"

    def cas_get(self, uri: str) -> bytes:
        """Read back a cas://<sha>[/label] object (the label is cosmetic)."""
        if not uri.startswith("cas://"):
            raise ValueError(f"not a cas uri: {uri!r}")
        digest = uri[len("cas://") :].strip("/").split("/")[0]
        blob = self.cas_dir / digest / "blob"
        if not blob.exists():
            raise FileNotFoundError(f"cas object not found: {uri}")
        return blob.read_bytes()


@dataclass
class RunHandle:
    """Handle on runs/<run_id>/: manifest, ledger, rollouts, blobs, eval dirs."""

    store: Store
    run_id: str
    path: Path
    _manifest_json: str

    # ---------- manifest ----------

    @property
    def manifest(self) -> dict[str, Any]:
        """The run's manifest (a fresh copy; the stored one is immutable)."""
        return json.loads(self._manifest_json)

    # ---------- ledger (append-only, strictly increasing update) ----------

    @property
    def ledger_path(self) -> Path:
        return self.path / "ledger.jsonl"

    def read_ledger(self) -> list[dict[str, Any]]:
        """All committed entries. A torn final line (kill -9 mid-append) is dropped."""
        if not self.ledger_path.exists():
            return []
        lines = [ln for ln in self.ledger_path.read_text(encoding="utf-8").split("\n") if ln]
        entries: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break  # torn tail: uncommitted
                raise LedgerError(f"corrupt ledger line {index} in {self.ledger_path}") from None
        return entries

    def ledger_tail(self) -> dict[str, Any] | None:
        """The last committed entry, or None on a fresh run."""
        entries = self.read_ledger()
        return entries[-1] if entries else None

    def _repair_ledger(self) -> int:
        """Truncate bytes after the last valid entry (a torn append is uncommitted)."""
        if not self.ledger_path.exists():
            return 0
        raw = self.ledger_path.read_bytes()
        keep = 0
        while True:
            newline = raw.find(b"\n", keep)
            if newline == -1:
                break
            try:
                json.loads(raw[keep:newline].decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                break
            keep = newline + 1
        if keep != len(raw):
            with open(self.ledger_path, "r+b") as handle:
                handle.truncate(keep)
                handle.flush()
                os.fsync(handle.fileno())
        return len(raw) - keep

    def append_ledger(self, entry: dict[str, Any]) -> None:
        """Commit one entry: canonical json line, flushed and fsynced."""
        self._repair_ledger()
        if "update" not in entry:
            raise LedgerError("ledger entry must carry an integer 'update'")
        update = entry["update"]
        if isinstance(update, bool) or not isinstance(update, int):
            raise LedgerError(f"ledger 'update' must be an int, got {update!r}")
        tail = self.ledger_tail()
        if tail is not None and update <= int(tail["update"]):
            raise LedgerError(
                f"ledger update must strictly increase: {update} <= {tail['update']}"
            )
        with open(self.ledger_path, "a", encoding="utf-8") as handle:
            handle.write(_canonical(entry) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # ---------- rollouts ----------

    def _rollouts_path(self, update: int) -> Path:
        return self.path / "rollouts" / f"{update:06d}.jsonl.gz"

    def write_rollouts(self, update: int, rows: list[dict[str, Any]]) -> Path:
        """Write one sealed wave's rows atomically (gzip of canonical jsonl)."""
        tail = self.ledger_tail()
        path = self._rollouts_path(update)
        if path.exists() and tail is not None and update <= int(tail["update"]):
            raise StoreError(f"rollouts {update} are committed by the ledger; the store is append-only")
        _atomic_write(path, _gzip_jsonl(rows))
        return path

    def read_rollouts(self, update: int) -> list[dict[str, Any]]:
        """Read back one wave's rows."""
        path = self._rollouts_path(update)
        if not path.exists():
            raise FileNotFoundError(f"no rollouts for update {update}: {path}")
        with gzip.open(path, "rb") as handle:
            raw = handle.read().decode("utf-8")
        return [json.loads(line) for line in raw.split("\n") if line]

    def list_updates(self) -> list[int]:
        """Rollout updates present on disk, sorted."""
        updates: list[int] = []
        for path in (self.path / "rollouts").glob("*.jsonl.gz"):
            number = _parse_update(path)
            if number is not None:
                updates.append(number)
        return sorted(updates)

    # ---------- blobs: adapters/ and optim/ (decision #14) ----------

    def _blob_path(self, section: str, name: str, version: int) -> Path:
        if section not in BLOB_SECTIONS:
            raise ValueError(f"section must be one of {sorted(BLOB_SECTIONS)}, got {section!r}")
        return self.path / section / f"{name}@{version}.bin"

    def write_blob(self, section: str, name: str, version: int, data: bytes) -> Path:
        """Write adapters/<name>@<v>.bin or optim/<name>@<v>.bin atomically."""
        path = self._blob_path(section, name, version)
        _atomic_write(path, data)
        return path

    def read_blob(self, section: str, name: str, version: int) -> bytes:
        """Read one delta's tensors or its optimizer moments at a pinned version."""
        path = self._blob_path(section, name, version)
        if not path.exists():
            raise FileNotFoundError(f"no {section} blob {name}@{version}: {path}")
        return path.read_bytes()

    # ---------- eval ----------

    def eval_dir(self, version: int) -> Path:
        """runs/<id>/eval/<version>/ -- firewalled measurement output, created on demand."""
        path = self.path / "eval" / str(version)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---------- crash recovery (runs on every attach) ----------

    def _discard_unsealed(self) -> None:
        """Drop everything the ledger never committed.

        The ledger is the commit record: work is sealed by its ledger entry and
        nothing else. So on attach — a torn final ledger line is truncated (that
        append never committed), rollout shards beyond the ledger tail are
        deleted, blob versions above each delta's committed version are deleted,
        and stray *.tmp files from interrupted atomic writes are swept.
        """
        self._repair_ledger()
        entries = self.read_ledger()
        self._drop_rollouts_beyond(int(entries[-1]["update"]) if entries else -1)
        self._drop_blobs_beyond(_committed_versions(entries))
        for tmp in self.path.rglob("*.tmp"):
            tmp.unlink()

    def _drop_rollouts_beyond(self, tail_update: int) -> None:
        for path in (self.path / "rollouts").glob("*.jsonl.gz"):
            update = _parse_update(path)
            if update is not None and update > tail_update:
                path.unlink()

    def _drop_blobs_beyond(self, committed: dict[str, int] | None) -> None:
        """`committed` is each delta's highest ledger-committed version.

        None means the ledger records no versions at all — nothing to compare
        against, keep every blob. A delta the ledger never mentioned has
        committed version 0, so all of its blobs are uncommitted and dropped.
        """
        if committed is None:
            return
        for section in BLOB_SECTIONS:
            for path in (self.path / section).glob("*@*.bin"):
                name, version = _parse_blob(path)
                if version is not None and version > committed.get(name, 0):
                    path.unlink()


def _committed_versions(entries: list[dict[str, Any]]) -> dict[str, int] | None:
    """Per-delta maxima over the entries' {"versions": {name: v}} maps.

    Returns None when no entry carries a versions map at all.
    """
    found = False
    out: dict[str, int] = {}
    for entry in entries:
        versions = entry.get("versions")
        if isinstance(versions, dict):
            found = True
            for name, version in versions.items():
                out[str(name)] = max(out.get(str(name), 0), int(version))
    return out if found else None


def _parse_update(path: Path) -> int | None:
    """<update>.jsonl.gz -> update, or None if the name is not ours."""
    try:
        return int(path.name.split(".", 1)[0])
    except ValueError:
        return None


def _parse_blob(path: Path) -> tuple[str, int | None]:
    """<name>@<version>.bin -> (name, version), version None if the name is not ours."""
    name, _, version = path.name[: -len(".bin")].rpartition("@")
    try:
        return name, int(version)
    except ValueError:
        return name, None


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------

def bump(policy_version: Mapping[str, int], names: Iterable[str]) -> dict[str, int]:
    """A new PolicyVersion with +1 on each named delta; unnamed deltas ride along."""
    out = {k: int(v) for k, v in policy_version.items()}
    for name in names:
        out[name] = out.get(name, 0) + 1
    return out
