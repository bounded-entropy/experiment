"""Run storage: ONE layout, many backends.

The layout, its invariants, and all orchestration (attach-or-create, the
append-only ledger, crash recovery) are universal and live HERE, written
against six abstract byte verbs. A backend — local disk, S3/R2, anything —
subclasses Store and implements only the verbs (data/stores/local.py is the
reference). Every backend stores the same key tree:

    runs/<run_id>/manifest.json                identity (I3), written once
                  dictionary.json              the run's self-description
                                               (derived, spec/flow.py)
                  ledger.jsonl                 the commit record, append-only
                  waves/<update>.jsonl.gz      sealed waves (trajectory rows)
                  postdata/<update>.json       the pipeline's columns per wave
                  adapters/<name>@<v>.bin      delta payloads
                  optim/<name>@<v>.bin         optimizer moments (lockstep)
                  eval/<update>/...            firewalled measurement output
    cas/<sha256>/blob                          content-addressed objects
    hosts/<name>/log.jsonl                     host observability journal
                                               (correctness never reads it)

Invariants: identity is computed, never typed (I3); writes are atomic; the
ledger is append-only and strictly increasing; resume = attach + ledger tail;
work not committed by the ledger is UNSEALED and is discarded on attach.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

BLOB_SECTIONS = ("adapters", "optim")

# Sections holding one artifact per update, committed by the ledger.
UPDATE_SECTIONS = (("waves", ".jsonl.gz"), ("postdata", ".json"))


class StoreError(RuntimeError):
    """Store invariant violation."""


class ManifestMismatch(StoreError):
    """An existing run's manifest differs from the one offered on attach (I3)."""


class LedgerError(StoreError, ValueError):
    """Malformed or non-monotonic ledger entry (the ledger is append-only)."""


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _gzip_jsonl(rows: list[dict[str, Any]]) -> bytes:
    """Rows -> deterministic gzip of canonical jsonl."""
    raw = "".join(_canonical(row) + "\n" for row in rows).encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        gz.write(raw)
    return buffer.getvalue()


class Store(ABC):
    """The universal layout over a backend's byte verbs."""

    # ---- the byte verbs a backend must provide ------------------------------

    @abstractmethod
    def _read(self, key: str) -> bytes:
        """Bytes at key; FileNotFoundError if absent."""

    @abstractmethod
    def _write(self, key: str, data: bytes) -> None:
        """Atomic, durable write: readers see old bytes or new, never a tear."""

    @abstractmethod
    def _append_line(self, key: str, line: str) -> None:
        """Durable append of one line (creates the key if absent)."""

    @abstractmethod
    def _exists(self, key: str) -> bool: ...

    @abstractmethod
    def _list(self, prefix: str) -> list[str]:
        """All keys under prefix, sorted."""

    @abstractmethod
    def _delete(self, key: str) -> None: ...

    def _sweep_partial(self, prefix: str) -> None:
        """Remove backend-specific write debris (default: nothing)."""

    # ---- runs ---------------------------------------------------------------

    def open_run(self, run_id: str, manifest: dict[str, Any] | None = None) -> "RunHandle":
        """Create runs/<run_id>/ (manifest required) or attach (manifest checked).

        Attaching also discards unsealed work: per-update artifacts and blob
        versions the ledger never committed.
        """
        manifest_key = f"runs/{run_id}/manifest.json"
        if not self._exists(manifest_key):
            if manifest is None:
                raise StoreError(
                    f"run {run_id!r} does not exist; a manifest is required to create it")
            self._write(manifest_key, _canonical(manifest).encode("utf-8"))
            self._append_line(f"runs/{run_id}/ledger.jsonl", "")
            return RunHandle(self, run_id, _canonical(manifest))

        stored = self._read(manifest_key).decode("utf-8")
        if manifest is not None and _canonical(manifest) != _canonical(json.loads(stored)):
            raise ManifestMismatch(
                f"run {run_id!r} exists with a different manifest (identity is computed, I3)")
        handle = RunHandle(self, run_id, stored)
        handle._discard_unsealed()
        return handle

    def list_runs(self) -> list[str]:
        """Run ids present in the store."""
        return sorted({key.split("/")[1] for key in self._list("runs/")
                       if key.endswith("/manifest.json")})

    # ---- content-addressed storage ------------------------------------------

    def fingerprint(self, data: bytes) -> str:
        """sha256 hex of `data` — the store's only identity function (I3)."""
        return hashlib.sha256(data).hexdigest()

    def cas_put(self, data: bytes) -> str:
        """Store `data` under its own hash; identical bytes dedupe."""
        digest = self.fingerprint(data)
        key = f"cas/{digest}/blob"
        if not self._exists(key):
            self._write(key, data)
        return f"cas://{digest}"

    def cas_get(self, uri: str) -> bytes:
        """Read back a cas://<sha>[/label] object (the label is cosmetic)."""
        if not uri.startswith("cas://"):
            raise ValueError(f"not a cas uri: {uri!r}")
        digest = uri[len("cas://"):].strip("/").split("/")[0]
        key = f"cas/{digest}/blob"
        if not self._exists(key):
            raise FileNotFoundError(f"cas object not found: {uri}")
        return self._read(key)

    def describe(self) -> str:
        """Where this store's data lives, for journals and CLIs — a path,
        a bucket, a mount. Backends override; the class name is the floor."""
        return type(self).__name__

    # ---- read-only peeks (for observers: never attach, never mutate) --------

    def peek_dictionary(self, run_id: str) -> dict[str, Any] | None:
        """The run's self-description (dictionary.json) WITHOUT attaching:
        what its store contains and why — a UI renders from this."""
        try:
            return json.loads(
                self._read(f"runs/{run_id}/dictionary.json").decode("utf-8"))
        except FileNotFoundError:
            return None

    def peek_manifest(self, run_id: str) -> dict[str, Any] | None:
        """A run's manifest WITHOUT attaching (open_run sweeps unsealed
        work — an observer must never do that to a live run)."""
        try:
            return json.loads(
                self._read(f"runs/{run_id}/manifest.json").decode("utf-8"))
        except FileNotFoundError:
            return None

    def peek_ledger(self, run_id: str) -> list[dict[str, Any]]:
        """A run's committed entries WITHOUT attaching; torn or corrupt
        lines are skipped, not repaired — peeking never writes."""
        try:
            text = self._read(f"runs/{run_id}/ledger.jsonl").decode("utf-8")
        except FileNotFoundError:
            return []
        out = []
        for line in text.split("\n"):
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    # ---- host journal (observability ONLY; correctness never reads it) ------

    def append_host_event(self, host: str, entry: dict[str, Any]) -> None:
        """One event line in hosts/<host>/log.jsonl (host-up/attach/detach).
        Outside every run directory, outside identity, outside recovery."""
        self._append_line(f"hosts/{host}/log.jsonl", _canonical(entry))

    def read_host_log(self, host: str) -> list[dict[str, Any]]:
        """Every parseable event for one host; a torn tail is tolerated —
        this is observability, not a commit record."""
        try:
            text = self._read(f"hosts/{host}/log.jsonl").decode("utf-8")
        except FileNotFoundError:
            return []
        out = []
        for line in text.split("\n"):
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def list_hosts(self) -> list[str]:
        """Host names that have ever journaled to this store."""
        return sorted({key.split("/")[1] for key in self._list("hosts/")
                       if key.endswith("/log.jsonl")})



@dataclass
class RunHandle:
    """Handle on runs/<run_id>/ — backend-agnostic; all IO via the store's verbs."""

    store: Store
    run_id: str
    _manifest_json: str

    def _key(self, *parts: str) -> str:
        return "/".join(("runs", self.run_id, *parts))

    @property
    def manifest(self) -> dict[str, Any]:
        """The run's manifest (a fresh copy; the stored one is immutable)."""
        return json.loads(self._manifest_json)

    def write_dictionary(self, dictionary: dict[str, Any]) -> None:
        """The run's self-description, beside the manifest. DERIVED from the
        spec (never part of identity), deterministic (resume rewrites the
        same bytes), and the store neither reads nor validates it — the
        membrane stays dumb; the runner supplies the content."""
        self.store._write(self._key("dictionary.json"),
                          _canonical(dictionary).encode("utf-8"))

    # ---- ledger (append-only, strictly increasing update) -------------------

    @property
    def ledger_key(self) -> str:
        return self._key("ledger.jsonl")

    def read_ledger(self) -> list[dict[str, Any]]:
        """All committed entries. A torn final line (kill -9 mid-append) is dropped."""
        try:
            text = self.store._read(self.ledger_key).decode("utf-8")
        except FileNotFoundError:
            return []
        lines = [line for line in text.split("\n") if line]
        entries: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    break  # torn tail: uncommitted
                raise LedgerError(
                    f"corrupt ledger line {index} in {self.ledger_key}") from None
        return entries

    def ledger_tail(self) -> dict[str, Any] | None:
        """The last committed entry, or None on a fresh run."""
        entries = self.read_ledger()
        return entries[-1] if entries else None

    def _repair_ledger(self) -> None:
        """Rewrite the ledger without any torn tail (a torn append never committed)."""
        try:
            raw = self.store._read(self.ledger_key)
        except FileNotFoundError:
            return
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
            self.store._write(self.ledger_key, raw[:keep])

    def append_ledger(self, entry: dict[str, Any]) -> None:
        """Commit one entry: canonical json line, durable."""
        self._repair_ledger()
        if "update" not in entry:
            raise LedgerError("ledger entry must carry an integer 'update'")
        update = entry["update"]
        if isinstance(update, bool) or not isinstance(update, int):
            raise LedgerError(f"ledger 'update' must be an int, got {update!r}")
        tail = self.ledger_tail()
        if tail is not None and update <= int(tail["update"]):
            raise LedgerError(
                f"ledger update must strictly increase: {update} <= {tail['update']}")
        self.store._append_line(self.ledger_key, _canonical(entry))

    # ---- per-update artifacts: waves + postdata --------------------------

    def _wave_key(self, update: int) -> str:
        return self._key("waves", f"{update:06d}.jsonl.gz")

    def write_wave(self, update: int, rows: list[dict[str, Any]]) -> None:
        """Write one sealed wave's rows atomically (gzip of canonical jsonl)."""
        self._refuse_committed_overwrite(update, "waves")
        self.store._write(self._wave_key(update), _gzip_jsonl(rows))

    def read_wave(self, update: int) -> list[dict[str, Any]]:
        key = self._wave_key(update)
        if not self.store._exists(key):
            raise FileNotFoundError(f"no wave for update {update}: {key}")
        raw = gzip.decompress(self.store._read(key)).decode("utf-8")
        return [json.loads(line) for line in raw.split("\n") if line]

    def _postdata_key(self, update: int) -> str:
        return self._key("postdata", f"{update:06d}.json")

    def write_postdata(self, update: int, columns: Mapping[str, list[float]]) -> None:
        """The pipeline's per-trajectory columns for one wave, in wave order."""
        self._refuse_committed_overwrite(update, "postdata")
        self.store._write(self._postdata_key(update),
                          _canonical({"columns": dict(columns)}).encode("utf-8"))

    def read_postdata(self, update: int) -> dict[str, list[float]]:
        key = self._postdata_key(update)
        if not self.store._exists(key):
            raise FileNotFoundError(f"no postdata for update {update}: {key}")
        return json.loads(self.store._read(key).decode("utf-8"))["columns"]

    def _refuse_committed_overwrite(self, update: int, section: str) -> None:
        tail = self.ledger_tail()
        if tail is not None and update <= int(tail["update"]):
            raise StoreError(
                f"{section} {update} are committed by the ledger; the store is append-only")

    def list_updates(self) -> list[int]:
        """Wave updates present, sorted."""
        out = []
        for key in self.store._list(self._key("waves")):
            update = _parse_update(key, ".jsonl.gz")
            if update is not None:
                out.append(update)
        return sorted(out)

    # ---- blobs: adapters/ and optim/ ----------------------------------------

    def _blob_key(self, section: str, name: str, version: int) -> str:
        if section not in BLOB_SECTIONS:
            raise ValueError(
                f"section must be one of {sorted(BLOB_SECTIONS)}, got {section!r}")
        return self._key(section, f"{name}@{version}.bin")

    def write_blob(self, section: str, name: str, version: int, data: bytes) -> None:
        self.store._write(self._blob_key(section, name, version), data)

    def read_blob(self, section: str, name: str, version: int) -> bytes:
        key = self._blob_key(section, name, version)
        if not self.store._exists(key):
            raise FileNotFoundError(f"no {section} blob {name}@{version}: {key}")
        return self.store._read(key)

    # ---- eval ---------------------------------------------------------------

    def write_eval(self, update: int, filename: str, text: str) -> None:
        """Firewalled measurement output; never consulted by crash recovery."""
        self.store._write(self._key("eval", str(update), filename),
                          text.encode("utf-8"))

    def has_eval(self, update: int, filename: str = "summary.json") -> bool:
        """Whether `update`'s eval completed (the summary is written last)."""
        return self.store._exists(self._key("eval", str(update), filename))

    def read_eval(self, update: int, filename: str) -> str:
        return self.store._read(self._key("eval", str(update), filename)).decode("utf-8")

    # ---- crash recovery (runs on every attach) ------------------------------

    def _discard_unsealed(self) -> None:
        """Drop everything the ledger never committed.

        The ledger is the commit record: work is sealed by its ledger entry and
        nothing else. A torn final ledger line is repaired away; per-update
        artifacts (waves, postdata) beyond the tail are deleted; blob
        versions above each delta's committed version are deleted; backend
        write debris is swept.
        """
        self._repair_ledger()
        entries = self.read_ledger()
        tail_update = int(entries[-1]["update"]) if entries else -1

        for section, suffix in UPDATE_SECTIONS:
            for key in self.store._list(self._key(section)):
                update = _parse_update(key, suffix)
                if update is not None and update > tail_update:
                    self.store._delete(key)

        committed = _committed_versions(entries)
        if committed is not None:
            for section in BLOB_SECTIONS:
                for key in self.store._list(self._key(section)):
                    name, version = _parse_blob(key)
                    if version is not None and version > committed.get(name, 0):
                        self.store._delete(key)

        self.store._sweep_partial(self._key())


def _committed_versions(entries: list[dict[str, Any]]) -> dict[str, int] | None:
    """Per-delta maxima over the entries' {"versions": {name: v}} maps.

    None when no entry carries a versions map at all (nothing to compare
    against — keep every blob). A delta the ledger never mentioned has
    committed version 0, so all of its blobs are uncommitted.
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


def _parse_update(key: str, suffix: str) -> int | None:
    """.../<update><suffix> -> update, or None if the name is not ours."""
    name = key.rsplit("/", 1)[-1]
    if not name.endswith(suffix):
        return None
    try:
        return int(name[: -len(suffix)])
    except ValueError:
        return None


def _parse_blob(key: str) -> tuple[str, int | None]:
    """.../<name>@<version>.bin -> (name, version); version None if not ours."""
    name = key.rsplit("/", 1)[-1]
    if not name.endswith(".bin"):
        return name, None
    stem, _, version = name[: -len(".bin")].rpartition("@")
    try:
        return stem, int(version)
    except ValueError:
        return stem, None


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------

def bump(policy_version: Mapping[str, int], names: Iterable[str]) -> dict[str, int]:
    """A new PolicyVersion with +1 on each named delta; unnamed deltas ride along."""
    out = {k: int(v) for k, v in policy_version.items()}
    for name in names:
        out[name] = out.get(name, 0) + 1
    return out
