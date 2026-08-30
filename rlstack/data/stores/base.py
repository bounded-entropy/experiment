"""The store: ONE key tree, many backends.

The tree and all orchestration — attach-or-create, the append-only ledger,
crash recovery, retention — live HERE, written against seven abstract byte
verbs; a backend subclasses Store and implements only the verbs (local.py is
the reference):

    runs/<run_id>/manifest.json                identity (I3), written once
                  dictionary.json              the run's self-description
                                               (derived, spec/flow.py)
                  ledger.jsonl                 the commit record, append-only
                  waves/<update>.jsonl.gz      sealed waves (trajectory rows)
                  postdata/<update>.json       the pipeline's columns per wave
                  postdata/<update>.<who>.json ONE producer's columns (the
                                               Scorer's half), merged into the
                                               file above at the commit
                  adapters/<name>@<v>.bin      delta payloads
                  optim/<name>@<v>.bin         optimizer moments (lockstep)
                  eval/<update>/...            firewalled measurement output
    cas/<sha256>/blob                          content-addressed objects
    hosts/<name>/log.jsonl                     the host and fleet journals:
    fleet/log.jsonl                            observability only (correctness
                                               never reads them)
    panels.json                                user-defined derived graphs
                                               (observer reads; never identity)
    annotations.jsonl                          names, tags and notes a human
                                               attached to runs — flavortext,
                                               beside runs/ and never inside it

Writes are atomic; the ledger is append-only, strictly increasing, and the
commit bit; resume is attach plus the ledger tail, and work no ledger line
committed is UNSEALED and is discarded on attach.

Deletion has exactly two meanings, and they are the same rule read twice:
attach deletes what the ledger NEVER COMMITTED, and a sweep (retention.py)
deletes what the ledger has MOVED PAST. Neither can reach a byte the run's
identity or its commit record is made of.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rlstack.data.stores.retention import RetentionPolicy, Swept

BLOB_SECTIONS = ("adapters", "optim")

# Sections holding one artifact per update, committed by the ledger.
UPDATE_SECTIONS = (("waves", ".jsonl.gz"), ("postdata", ".json"))

# The three things a human may say about a run. Each merges independently:
# a row carrying only a note leaves an earlier name standing.
ANNOTATION_FIELDS = ("name", "tags", "note")


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


def check_subdir(subdir: str) -> str:
    """A run's filing path, validated: "/"-joined name segments, each of
    [A-Za-z0-9._-]+ and never "." or "..". The subdir is WHERE a run's
    directory spawns (runs/<subdir>/<run_id>) — organization, not identity:
    it never hashes, and the same spec filed differently is the same run."""
    segments = [seg for seg in str(subdir).strip("/").split("/") if seg]
    if not segments:
        raise StoreError(f"subdir {subdir!r} names no path segment")
    for seg in segments:
        if seg in (".", "..") or not re.fullmatch(r"[A-Za-z0-9._-]+", seg):
            raise StoreError(
                f"subdir segment {seg!r} is not a plain name segment "
                f"([A-Za-z0-9._-]+, never '.' or '..')")
    return "/".join(segments)


def wave_key(run_dir: str, update: int) -> str:
    """Where one update's sealed wave lives, under the run's DIRECTORY key
    (runs/<run_id>, or runs/<subdir>/<run_id> where the run was filed). One
    tree: the handle writes it and the observer's peek reads it through the
    same name."""
    return f"{run_dir}/waves/{update:06d}.jsonl.gz"


def rollout_key(run_dir: str, index: int) -> str:
    """Where one GENERATED wave lives, before any update consumes it.

    Separate from waves/ because the generator's output and the trainer's input
    stopped being the same thing (#59): a plan may train on rollout 7 at update
    9, on two rollouts at once, or on none at all."""
    return f"{run_dir}/rollouts/{index:06d}.jsonl.gz"


def plan_key(run_dir: str, kind: str) -> str:
    """Where a run's copy of one plan lives. The spec pins the plan by cas uri;
    this copy is what makes the run self-describing (I11) and what an observer
    reads without resolving anything."""
    return f"{run_dir}/plans/{kind}.jsonl"


def postdata_key(run_dir: str, update: int) -> str:
    """Where one update's postprocessor columns live, beside its wave."""
    return f"{run_dir}/postdata/{update:06d}.json"


def postdata_part_key(run_dir: str, update: int, producer: str) -> str:
    """Where ONE producer's postdata columns live: beside the merged file,
    named by who wrote them.

    The producer is one NAME segment, and the assertion is the same rule the
    host journal's key states: a "/" would shear the key, and a "." would shear
    the update out of it — `_parse_update` reads this name to decide what
    attach discards.
    """
    assert "/" not in producer and "." not in producer, (
        f"postdata producer {producer!r} must be one dot-free name segment")
    return f"{run_dir}/postdata/{update:06d}.{producer}.json"


class Store(ABC):
    """The key tree and its orchestration over a backend's six byte verbs."""

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

    @abstractmethod
    def _size(self, key: str) -> int:
        """Bytes at key WITHOUT reading them; FileNotFoundError if absent.

        The seventh verb, and deletion's: a sweep has to report what it freed,
        and reading a 160 MiB optimizer blob to measure it would cost more than
        leaving it where it is."""

    def _sweep_partial(self, prefix: str) -> None:
        """Remove backend-specific write debris (default: nothing)."""

    def _persist(self) -> None:
        """Make staged verb calls durable (default: nothing — a backend whose
        verbs are durable as they land has nothing to do here).

        A backend that STAGES (a Modal Volume mount) persists on commit, and a
        deletion stages exactly like a write: without this, a sweep's freed
        bytes come back when the container ends."""

    # ---- runs ---------------------------------------------------------------

    def run_prefix(self, run_id: str) -> str:
        """The key of this run's DIRECTORY: runs/<run_id> at the top, or
        runs/<subdir>/<run_id> wherever open_run filed it at birth. One run,
        one home, found by its manifest and cached; a run that exists nowhere
        resolves to the top spelling, so absence still reads as absence. The
        cache never goes stale because a run NEVER MOVES — its home is fixed
        the moment the manifest is written."""
        homes = self.__dict__.setdefault("_run_homes", {})
        cached = homes.get(run_id)
        if cached is not None:
            return cached
        suffix = f"/{run_id}/manifest.json"
        for key in self._list("runs/"):
            if key.endswith(suffix):
                homes[run_id] = key[: -len("/manifest.json")]
                return homes[run_id]
        return f"runs/{run_id}"

    def run_subdirs(self) -> dict[str, str]:
        """{run_id: subdir} for every run in the store ("" at the top) — the
        observer's one question about filing."""
        out: dict[str, str] = {}
        for key in self._list("runs/"):
            if not key.endswith("/manifest.json"):
                continue
            parts = key.split("/")
            out[parts[-2]] = "/".join(parts[1:-2])
        return out

    def open_run(self, run_id: str, manifest: dict[str, Any] | None = None,
                 subdir: str | None = None) -> "RunHandle":
        """Attach-or-create: create the run's directory (manifest required) or
        attach to it (the manifest must match — identity is computed, I3).

        `subdir` says WHERE a NEW run's directory spawns
        (runs/<subdir>/<run_id>); it is filing, never identity. A run's home
        is fixed at birth: attaching finds the run wherever it lives, and a
        different subdir asked later is ignored — resubmission is resume, not
        a move (I10's "for life" includes the address).

        Attaching discards unsealed work: per-update artifacts and blob versions
        no ledger line committed. Observers must peek instead (I10).
        """
        home = self.run_prefix(run_id)
        manifest_key = f"{home}/manifest.json"
        if not self._exists(manifest_key):
            if manifest is None:
                raise StoreError(
                    f"run {run_id!r} does not exist; a manifest is required to create it")
            if subdir is not None:
                home = f"runs/{check_subdir(subdir)}/{run_id}"
                manifest_key = f"{home}/manifest.json"
            self._write(manifest_key, _canonical(manifest).encode("utf-8"))
            self._append_line(f"{home}/ledger.jsonl", "")
            self.__dict__.setdefault("_run_homes", {})[run_id] = home
            return RunHandle(self, run_id, _canonical(manifest), home)

        stored = self._read(manifest_key).decode("utf-8")
        if manifest is not None and _canonical(manifest) != _canonical(json.loads(stored)):
            raise ManifestMismatch(
                f"run {run_id!r} exists with a different manifest (identity is computed, I3)")
        handle = RunHandle(self, run_id, stored, home)
        handle._discard_unsealed()
        return handle

    def list_runs(self) -> list[str]:
        """Run ids present in the store, wherever they are filed."""
        return sorted({key.split("/")[-2] for key in self._list("runs/")
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
                self._read(f"{self.run_prefix(run_id)}/dictionary.json")
                .decode("utf-8"))
        except FileNotFoundError:
            return None

    def peek_manifest(self, run_id: str) -> dict[str, Any] | None:
        """A run's manifest WITHOUT attaching (open_run sweeps unsealed
        work — an observer must never do that to a live run)."""
        try:
            return json.loads(
                self._read(f"{self.run_prefix(run_id)}/manifest.json")
                .decode("utf-8"))
        except FileNotFoundError:
            return None

    def peek_ledger(self, run_id: str) -> list[dict[str, Any]]:
        """A run's committed entries WITHOUT attaching; torn or corrupt
        lines are skipped, not repaired — peeking never writes."""
        try:
            text = self._read(
                f"{self.run_prefix(run_id)}/ledger.jsonl").decode("utf-8")
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

    def peek_wave(self, run_id: str, update: int) -> list[dict[str, Any]] | None:
        """One sealed wave's trajectory rows WITHOUT attaching. The observer's
        one door onto experiment CONTENT (#57): SEALED artifacts only — an
        update the ledger committed can never be rewritten, so reading it is
        as safe as reading the ledger. None when that update has no wave."""
        try:
            raw = gzip.decompress(
                self._read(wave_key(self.run_prefix(run_id), update)))
        except FileNotFoundError:
            return None
        return [json.loads(line) for line in raw.decode("utf-8").split("\n") if line]

    def peek_postdata(self, run_id: str, update: int) -> dict[str, list] | None:
        """The pipeline's columns for one sealed wave, in wave order, WITHOUT
        attaching. None when that update has no postdata."""
        try:
            payload = json.loads(
                self._read(postdata_key(self.run_prefix(run_id),
                                        update)).decode("utf-8"))
        except FileNotFoundError:
            return None
        return payload["columns"]

    def peek_plan(self, run_id: str, kind: str) -> bytes | None:
        """One of the plans the run was created with, WITHOUT attaching: the
        run's SHAPE, copied in verbatim at creation and never rewritten, so
        reading it is as safe as reading the ledger. This is where a run's
        length lives (#59) — an observer counts its waves to know what the
        committed updates are counting up to. None when the run declared no
        plan of that kind."""
        try:
            return self._read(plan_key(self.run_prefix(run_id), kind))
        except FileNotFoundError:
            return None

    def peek_eval_summaries(self, run_id: str) -> list[dict[str, Any]]:
        """Every completed eval summary for a run, WITHOUT attaching —
        the observer's held-out series. Unparseable files are skipped."""
        out = []
        for key in self._list(f"{self.run_prefix(run_id)}/eval"):
            if not key.endswith("/summary.json"):
                continue
            try:
                out.append(json.loads(self._read(key).decode("utf-8")))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        return sorted(out, key=lambda s: s.get("update", 0))

    def read_panels(self) -> list[dict[str, Any]]:
        """User-defined derived-graph declarations (panels.json at the store
        root): [{"name", "expr"}, ...]. The user writes them, the observer only
        reads them; malformed or absent → empty."""
        try:
            payload = json.loads(self._read("panels.json").decode("utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        return [p for p in payload if isinstance(p, dict)
                and "name" in p and "expr" in p] if isinstance(payload, list) else []

    # ---- host journal (observability ONLY; correctness never reads it) ------

    def append_host_event(self, host: str, entry: dict[str, Any]) -> None:
        """One journal line in hosts/<host>/log.jsonl (host-up, attach, detach,
        gpu samples): observability, outside every run directory, outside
        identity, outside recovery.

        The host name is ONE path segment: a "/" would shear the key and the
        event would land where list_hosts() (which recovers the name with
        split("/")[1]) never looks. The Host attests this at birth; this is the
        same rule at the layer that owns the key."""
        assert "/" not in host, (
            f"host name {host!r} contains '/': it is one journal path segment")
        self._append_line(f"hosts/{host}/log.jsonl", _canonical(entry))

    def read_host_log(self, host: str) -> list[dict[str, Any]]:
        """Every parseable event for one host; a torn tail is tolerated —
        this is observability, not a commit record."""
        return self._read_log(f"hosts/{host}/log.jsonl")

    def list_hosts(self) -> list[str]:
        """Host names that have ever journaled to this store."""
        return sorted({key.split("/")[1] for key in self._list("hosts/")
                       if key.endswith("/log.jsonl")})

    # ---- fleet journal (observability ONLY; correctness never reads it) -----

    def append_fleet_event(self, entry: dict[str, Any]) -> None:
        """One journal line in fleet/log.jsonl (place / carve): the fleet's
        decision record. Carving is automatic BECAUSE it is journaled —
        legibility by record, not by approval. Outside every run directory,
        outside identity, outside recovery."""
        self._append_line("fleet/log.jsonl", _canonical(entry))

    def read_fleet_log(self) -> list[dict[str, Any]]:
        """Every parseable fleet event; a torn tail is tolerated."""
        return self._read_log("fleet/log.jsonl")

    # ---- annotations (FLAVORTEXT: never hashed, never read by experiments) --

    def annotations_key(self) -> str:
        """annotations.jsonl AT THE STORE ROOT — beside runs/, never inside a
        run directory.

        A run directory is identity plus the commit record, and
        resume-equivalence compares its bytes; a name a human typed belongs to
        neither, so it lives outside. Annotating a run cannot change one byte
        of it."""
        return "annotations.jsonl"

    def annotate_run(self, run_id: str, *, name: str | None = None,
                     tags: Sequence[str] | None = None,
                     note: str | None = None) -> None:
        """Append ONE annotation row: {"t", "run_id", + only the fields passed
        here}. Append-only — an annotation is never edited and never deleted,
        only superseded by a later row.

        Flavortext, and only flavortext: nothing here is hashed into identity,
        no experiment ever reads it, and the runner never writes it. A call
        naming no field is a mistake, not a no-op."""
        row: dict[str, Any] = {"t": time.time(), "run_id": run_id}
        if name is not None:
            row["name"] = str(name)
        if tags is not None:
            row["tags"] = [str(tag) for tag in tags]
        if note is not None:
            row["note"] = str(note)
        if not set(row) & set(ANNOTATION_FIELDS):
            raise StoreError(
                f"annotate_run({run_id!r}) named none of {ANNOTATION_FIELDS}")
        self._append_line(self.annotations_key(), _canonical(row))

    def read_annotations(self) -> dict[str, dict[str, Any]]:
        """{run_id: merged fields} — LATEST WINS PER FIELD, in append order.

        A later row's "tags" replaces the whole list (a tag is never removed
        one at a time); a row carrying only a note leaves an earlier name
        standing. Unparseable lines are skipped: this is flavortext, not a
        commit record, and a torn tail must never stop a page rendering."""
        merged: dict[str, dict[str, Any]] = {}
        for row in self._read_log(self.annotations_key()):
            run_id = row.get("run_id")
            if not isinstance(run_id, str):
                continue
            fields = merged.setdefault(run_id, {})
            for field in ANNOTATION_FIELDS:
                if field in row:
                    fields[field] = row[field]
        return merged

    def _read_log(self, key: str) -> list[dict[str, Any]]:
        try:
            text = self._read(key).decode("utf-8")
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



@dataclass
class RunHandle:
    """Handle on the run's directory (runs/<run_id>, or runs/<subdir>/<run_id>
    where it was filed) — backend-agnostic; all IO via the store's verbs."""

    store: Store
    run_id: str
    _manifest_json: str
    run_dir: str = ""

    def _key(self, *parts: str) -> str:
        return "/".join((self.run_dir or f"runs/{self.run_id}", *parts))

    @property
    def manifest(self) -> dict[str, Any]:
        """The run's manifest (a fresh copy; the stored one is immutable)."""
        return json.loads(self._manifest_json)

    def write_dictionary(self, dictionary: dict[str, Any]) -> None:
        """The run's self-description beside the manifest (I11): the flow graph
        serialized, DERIVED and never identity, deterministic (resume rewrites
        the same bytes). The store neither reads nor validates it — the membrane
        stays dumb; the runner supplies the content."""
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
        """THE commit point: one canonical json line, durable, and the Trainer's
        alone. `update` must strictly increase — the ledger is append-only."""
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
        return wave_key(self._key(), update)

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

    def _rollout_key(self, index: int) -> str:
        return rollout_key(self._key(), index)

    def write_rollout(self, index: int, rows: list[dict[str, Any]]) -> None:
        """Seal one generated wave. Never refused against the ledger: a rollout
        belongs to whichever updates their plan says, so it has no update of
        its own to be committed under."""
        self.store._write(self._rollout_key(index), _gzip_jsonl(rows))

    def read_rollout(self, index: int) -> list[dict[str, Any]]:
        key = self._rollout_key(index)
        if not self.store._exists(key):
            raise FileNotFoundError(f"no rollout {index}: {key}")
        raw = gzip.decompress(self.store._read(key)).decode("utf-8")
        return [json.loads(line) for line in raw.split("\n") if line]

    def write_plan(self, kind: str, data: bytes) -> None:
        """Copy one plan into the run, verbatim: the bytes the spec's cas uri
        addresses, so the run holds the shape it actually ran."""
        self.store._write(plan_key(self._key(), kind), data)

    def read_plan(self, kind: str) -> bytes:
        key = plan_key(self._key(), kind)
        if not self.store._exists(key):
            raise FileNotFoundError(f"no {kind} plan: {key}")
        return self.store._read(key)

    def _postdata_key(self, update: int) -> str:
        return postdata_key(self._key(), update)

    def write_postdata(self, update: int, columns: Mapping[str, list[float]]) -> None:
        """The postprocessor pipeline's columns for one wave, in wave order —
        stored beside the sealed wave, never inside it."""
        self._refuse_committed_overwrite(update, "postdata")
        self.store._write(self._postdata_key(update),
                          _canonical({"columns": dict(columns)}).encode("utf-8"))

    def read_postdata(self, update: int) -> dict[str, list[float]]:
        key = self._postdata_key(update)
        if not self.store._exists(key):
            raise FileNotFoundError(f"no postdata for update {update}: {key}")
        return json.loads(self.store._read(key).decode("utf-8"))["columns"]

    def write_postdata_part(self, update: int, producer: str,
                            columns: Mapping[str, list[float]]) -> None:
        """One producer's share of a wave's columns, written BEFORE the merged
        file — the handshake between the Scorer and the Trainer.

        Same discipline as the merged write: atomic, and refused once the
        ledger has committed the update, because a committed update's postdata
        is immutable in every one of its pieces. The merge stays the Trainer's
        job, so `read_postdata`, flatten, the observer and the wave browser
        keep reading exactly one file per update.
        """
        self._refuse_committed_overwrite(update, "postdata")
        self.store._write(postdata_part_key(self._key(), update, producer),
                          _canonical({"columns": dict(columns)}).encode("utf-8"))

    def read_postdata_part(self, update: int,
                           producer: str) -> dict[str, list[float]] | None:
        """One producer's columns, or None while it has not written them.

        None rather than FileNotFoundError because this read is an AWAIT
        PREDICATE: the Trainer blocks on exactly this absence, and a daemon
        waiting on a store predicate should be reading a value, not catching an
        exception.
        """
        key = postdata_part_key(self._key(), update, producer)
        if not self.store._exists(key):
            return None
        return json.loads(self.store._read(key).decode("utf-8"))["columns"]

    def _refuse_committed_overwrite(self, update: int, section: str) -> None:
        """An update the ledger committed is immutable; only unsealed work
        may be rewritten."""
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

    def has_blob(self, section: str, name: str, version: int) -> bool:
        """Is this exact version's blob already written? Blobs are immutable
        per version, so this is what lets a once-only write stay idempotent
        across resumes."""
        return self.store._exists(self._blob_key(section, name, version))

    def read_blob(self, section: str, name: str, version: int) -> bytes:
        key = self._blob_key(section, name, version)
        if not self.store._exists(key):
            raise FileNotFoundError(f"no {section} blob {name}@{version}: {key}")
        return self.store._read(key)

    # ---- retention (the second meaning of deletion) --------------------------

    def sweep(self, policy: RetentionPolicy) -> Swept:
        """Free what `policy` says nothing will ever read again; report it.

        Attach sweeps what the ledger never committed; this sweeps what the
        ledger has moved past. Every triple is addressed through `_blob_key`,
        so a policy can name nothing but a versioned blob under adapters/ or
        optim/ — the ledger, the manifest, a sealed wave and its postdata are
        unreachable from here and the append-only guards stand untouched.

        The whole batch is checked BEFORE the first delete, so a policy that
        names live state frees nothing at all rather than half of what it
        asked for. A blob already gone is skipped, which is what makes
        sweeping twice free nothing the second time.
        """
        tail = self.ledger_tail()
        named = []
        for section, name, version in policy.expendable(self.read_ledger()):
            self._refuse_live_version(tail, section, name, version)
            named.append(((section, name, version),
                          self._blob_key(section, name, version)))

        freed, gone = 0, []
        for triple, key in named:
            if not self.store._exists(key):
                continue
            freed += self.store._size(key)
            self.store._delete(key)
            gone.append(triple)
        if gone:
            # a deletion STAGES like a write: a backend that needs a commit to
            # make one durable gets exactly one, and only if bytes moved
            self.store._persist()
        return Swept(tuple(gone), freed)

    def _refuse_live_version(self, tail: dict[str, Any] | None, section: str,
                             name: str, version: int) -> None:
        """The version the ledger tail names is the run's LIVE state — resume
        reads exactly it — so no policy may free it, whatever it says.

        The floor under every retention policy, enforced here and not there: a
        swept run still resumes.
        """
        if tail is None:
            return
        live = tail.get("versions")
        if isinstance(live, dict) and name in live and int(version) == int(live[name]):
            raise StoreError(
                f"retention named {section}/{name}@{version}, the version the "
                f"ledger tail commits: the tail is what resume reads")

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
        """Drop everything no ledger line committed — work is sealed by its
        ledger entry and nothing else.

        A torn final ledger line is repaired away; per-update artifacts (waves,
        postdata, and the postdata PARTS a Scorer wrote ahead of the Trainer)
        beyond the tail are deleted; blob versions above each delta's committed
        version are deleted; backend write debris is swept.

        Sweeping a part costs nothing to recover: scoring is deterministic
        seedless prefill at a pinned bundle, so the daemon that wrote it writes
        the same bytes again on the next attach.
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
    """.../<update><suffix> -> update, or None if the name is not ours.

    A postdata PART (.../<update>.<producer><suffix>) parses to the same
    update, which is the whole reason the producer may not carry a dot: a part
    is a per-update artifact like any other, so attach discards it by exactly
    the rule that discards the merged file, and neither survives an update the
    ledger never committed.
    """
    name = key.rsplit("/", 1)[-1]
    if not name.endswith(suffix):
        return None
    try:
        return int(name[: -len(suffix)].split(".")[0])
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
