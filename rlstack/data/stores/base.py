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
                  eval/<update>/...            PRE-#70 in-run eval (legacy read)
    runs/<subdir>/names/<name>.bin             NAMED ADAPTERS (ADR 0019): one
                         <name>.json           LoRA set's payload, written once
                         <name>.promise        under the experiment's subdir;
                                               the meta seals it, the promise
                                               says which run will write it
    measurements/<run_id>/<name>/manifest.json observation OUTSIDE the run
                                 points.jsonl  (#70): not identity, not
                                               resume-equivalence, deletable —
                                               supersede by NAME
    cas/<sha256>/blob                          content-addressed objects
    hosts/<name>/log.jsonl                     the host and fleet journals:
    fleet/log.jsonl                            observability only (no run's
                                               bytes ever read them; a wait on
                                               a named adapter reads the
                                               fleet's stopped/failed rows to
                                               REFUSE instead of waiting)
    panels.json                                user-defined derived graphs
                                               (observer reads; never identity)
    annotations.jsonl                          names, tags and notes a human
                                               attached to runs — flavortext,
                                               beside runs/ and never inside it

Writes are atomic; the ledger is append-only, strictly increasing, and the
COMMIT record — one line per update. The CHECKPOINT record (ADR 0014) is its
sibling, `checkpoints.jsonl`: one line per update at which every trainable
entry's blobs are durable, at a cadence the submission declared. Resume is
attach plus the CHECKPOINT tail: attach REWINDS to it — ledger lines, waves,
postdata past its update and rollouts pinned past its versions are UNSEALED
and discarded, and the Trainer redoes them. A run with no checkpoint record
at all is a pre-0014 run, where every ledger line was a checkpoint.

Deletion has exactly two meanings, and they are the same rule read twice:
attach deletes what the checkpoint record NEVER SEALED, and a sweep
(retention.py) deletes what it has MOVED PAST. Neither can reach a byte the
run's identity or its commit record is made of.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import time
from collections import OrderedDict
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rlstack.data.stores.retention import RetentionPolicy, Swept

BLOB_SECTIONS = ("adapters", "optim")

CHECKPOINTS = "checkpoints.jsonl"
"""The checkpoint record's file name (ADR 0014), beside `ledger.jsonl`."""

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


class NamedAdapterConflict(StoreError):
    """A name that already holds bytes was offered DIFFERENT bytes (ADR 0019):
    a named adapter is written once, and a changed adapter is a new name."""


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


PROMISES_FOLDER = "_promises"
"""ONE PROMISE FILE PER WRITER (2026-09-19): names/_promises/<digest>.json holds
{"writer", "names"}. The first fit runs on metal promised their 200 names one
file each and spent half an hour of ten-second scratch writes before their
first step; a writer now says everything it will write in one write."""

NAMED_PAYLOADS_HELD = 16
"""How many named payloads one store object keeps after reading them."""

NAMED_PAYLOAD, NAMED_META, NAMED_PROMISE = ".bin", ".json", ".promise"
"""The three files one named adapter may have (ADR 0019): its bytes, the meta
that SEALS them (written second, so a name is present exactly when its meta
is), and the promise its writer left at birth."""

PRESENT, PROMISED, ORPHANED, UNKNOWN = "present", "promised", "orphaned", "unknown"
"""What `Store.named_state` answers — the four things a consumer can learn
about a name it needs."""


def check_name(name: str) -> str:
    """An adapter's NAME, validated (ADR 0019): `[A-Za-z0-9._/-]+`, no leading
    slash, no `..`. A "/" files names into folders, so every segment must name
    something (no empty segment, no trailing slash). The last segment is never
    `manifest`: a name's meta is `<name>.json`, and a `manifest.json` anywhere
    under runs/ is what marks a RUN directory. `+` and `:` are outside the
    grammar on purpose — they are what a route (`lib:<name>+dreamer`) is cut
    on."""
    text = str(name)
    if (not re.fullmatch(r"[A-Za-z0-9._/-]+", text) or text.startswith("/")
            or ".." in text or "" in text.split("/")):
        raise StoreError(
            f"adapter name {name!r} is not a name ([A-Za-z0-9._/-]+, no "
            f"leading or trailing slash, no empty segment, no '..')")
    if text.split("/", 1)[0] == PROMISES_FOLDER:
        raise StoreError(
            f"adapter name {name!r} begins with {PROMISES_FOLDER!r}: that folder "
            f"holds the writers' promise files")
    if text.rsplit("/", 1)[-1] == "manifest":
        raise StoreError(
            f"adapter name {name!r} ends in 'manifest': its meta would be a "
            f"manifest.json, which is what marks a run directory")
    return text


def named_key(subdir: str, name: str, suffix: str) -> str:
    """Where one file of a named adapter lives: under the experiment's subdir,
    beside the runs filed there (runs/<subdir>/names/<name><suffix>). One
    function, so the writer, every waiting consumer and the observer agree."""
    return f"runs/{check_subdir(subdir)}/names/{check_name(name)}{suffix}"


def promises_key(subdir: str, writer: str) -> str:
    """The one file in which `writer` says every name it will write."""
    digest = hashlib.sha256(writer.encode("utf-8")).hexdigest()[:16]
    return f"runs/{check_subdir(subdir)}/names/{PROMISES_FOLDER}/{digest}.json"


def run_reference(run_id: str, subdir: str | None = None) -> str:
    """The explicit path below runs/, shared by callers and store readers."""
    name = check_subdir(run_id)
    if not subdir:
        return name
    if "/" in name:
        raise StoreError("supply either a qualified run reference or a subdir, not both")
    return f"{check_subdir(subdir)}/{name}"


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


@dataclass(frozen=True)
class StoreAddress:
    """Where a store IS, as a value: which backend, which root, which locator.

    A resident (runner/residents.py) is a child process of the metal and must
    open the store the metal opened without being handed a live object —
    nothing live crosses a spawn. This is the JSON-safe handle it opens from;
    `open_store` (address.py) is the one place backends are known by name.
    """

    backend: str                # "local" | "modal_volume" — one file per backend
    root: str                   # the directory or mount
    locator: str                # how the store describes itself to outsiders

    def row(self) -> dict[str, str]:
        return {"backend": self.backend, "root": self.root,
                "locator": self.locator}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "StoreAddress":
        return cls(backend=str(row["backend"]), root=str(row["root"]),
                   locator=str(row["locator"]))


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

    def _run_directories(self) -> dict[str, str]:
        """{run_id: key of its directory} for every run that has a manifest —
        the ONE question anything asks of the runs/ tree. Answered here by
        walking _list; a backend that can find manifests without listing
        every wave, rollout and adapter beneath them overrides (LocalStore)."""
        out: dict[str, str] = {}
        for key in self._list("runs/"):
            if key.endswith("/manifest.json"):
                home = key[: -len("/manifest.json")]
                out[home.removeprefix("runs/")] = home
        return out

    def run_children(self, subdir: str = "") -> list[dict[str, str]]:
        """Browse one folder. Mounted stores override with a shallow listing."""
        prefix = "runs/" + (check_subdir(subdir) + "/" if subdir else "")
        names = sorted({key[len(prefix):].split("/")[0]
                        for key in self._list(prefix) if key.startswith(prefix)})
        return [{"name": name, "kind": "run" if self._exists(
                    prefix + name + "/manifest.json") else "folder",
                 "path": (subdir + "/" if subdir else "") + name}
                for name in names]

    def run_prefix(self, run_id: str, subdir: str | None = None) -> str:
        """Resolve an explicit run reference, without IO or discovery.

        A reference is `subdir/run_id`, or just `run_id` at the root. A
        separately supplied subdir qualifies a bare ID; it is never a hint.
        """
        return f"runs/{run_reference(run_id, subdir)}"

    def run_subdirs(self) -> dict[str, str]:
        """{run_reference: subdir} for every run ("" at the top) — the
        observer's one question about filing."""
        return {run_id: "/".join(home.split("/")[1:-1])
                for run_id, home in self._run_directories().items()}

    def open_run(self, run_id: str, manifest: dict[str, Any] | None = None,
                 subdir: str | None = None, *, create: bool | None = None) -> "RunHandle":
        """Open exactly the named directory; never look in another folder.

        Offering a manifest permits idempotent creation. `create=False`
        requests resume only, validating that manifest without creating a
        missing run. Without a manifest the run must already exist.
        Attaching discards unsealed work; observers must peek instead (I10).
        """
        home = self.run_prefix(run_id, subdir)
        run_id = home.rsplit("/", 1)[-1]
        manifest_key = f"{home}/manifest.json"
        if not self._exists(manifest_key):
            if create is False or manifest is None:
                raise StoreError(
                    f"run does not exist at {home!r}; resume requires its exact directory")
            self._write(manifest_key, _canonical(manifest).encode("utf-8"))
            self._append_line(f"{home}/ledger.jsonl", "")
            # the checkpoint record is born with the run: its ABSENCE is what
            # marks a pre-0014 directory, so a new run always has one
            self._append_line(f"{home}/{CHECKPOINTS}", "")
            return RunHandle(self, run_id, _canonical(manifest), home)

        stored = self._read(manifest_key).decode("utf-8")
        if manifest is not None and _canonical(manifest) != _canonical(json.loads(stored)):
            raise ManifestMismatch(
                f"run {run_id!r} exists with a different manifest (identity is computed, I3)")
        handle = RunHandle(self, run_id, stored, home)
        handle._discard_unsealed()
        return handle

    def list_runs(self) -> list[str]:
        """Explicit discovery of run references, preserving their directories."""
        return sorted(self._run_directories())

    # ---- content-addressed storage ------------------------------------------

    def fingerprint(self, data: bytes) -> str:
        """sha256 hex of `data` — the store's only identity function (I3)."""
        return hashlib.sha256(data).hexdigest()

    def _verify_hash(self, key: str, data: bytes, sha256: str) -> bytes:
        """Never return hashed bytes that disagree with their key or seal."""
        actual = self.fingerprint(data)
        if actual != sha256:
            raise StoreError(f"hash mismatch for {key!r}: expected {sha256}, actual {actual}")
        return data

    def _read_hashed(self, key: str, sha256: str) -> bytes:
        """Read through the backend's miss fallback, then verify (ADR 0020)."""
        return self._verify_hash(key, self._read(key), sha256)

    def _write_hashed(self, key: str, data: bytes, sha256: str) -> None:
        """Publish bytes whose expected hash every consumer will check."""
        self._write(key, data)

    def cas_put(self, data: bytes) -> str:
        """Store `data` under its own hash; identical bytes dedupe."""
        digest = self.fingerprint(data)
        key = f"cas/{digest}/blob"
        if not self._exists(key):
            self._write_hashed(key, data, digest)
        return f"cas://{digest}"

    def cas_get(self, uri: str) -> bytes:
        """Read back a cas://<sha>[/label] object (the label is cosmetic).

        Reads THROUGH `_read`, never an existence pre-check: a backend's
        miss fall-through (ModalVolumeStore answers a mount miss from the
        volume's committed view) must serve cas blobs too — an `_exists`
        gate on the mount alone killed forty adoptions whose plans another
        container had committed moments earlier (observed live)."""
        if not uri.startswith("cas://"):
            raise ValueError(f"not a cas uri: {uri!r}")
        digest = uri[len("cas://"):].strip("/").split("/")[0]
        try:
            return self._read_hashed(f"cas/{digest}/blob", digest)
        except FileNotFoundError:
            raise FileNotFoundError(f"cas object not found: {uri}") from None

    # ---- named adapters (ADR 0019) -------------------------------------------
    #
    # runs/<subdir>/names/<name>.bin + .json + .promise. A name is WRITTEN
    # ONCE and is present exactly when its meta is: the payload lands first,
    # the meta seals it, and a payload with no meta is debris a rerun of the
    # same fit overwrites. Reads go THROUGH `_read` (never an existence
    # pre-check on a mount), for the reason `cas_get` states: the consumer of
    # a name is usually another container than its writer.

    def write_named(self, subdir: str, name: str, payload: bytes,
                    meta: Mapping[str, Any]) -> None:
        """WRITE-ONCE: a name that is present and holds these same bytes is a
        no-op (a resumed fit run rewriting what it already wrote); one that
        holds different bytes raises NamedAdapterConflict — a changed adapter
        is a new name. The stored meta is the caller's plus `sha256`, the
        payload's fingerprint, which is what the comparison reads, so a
        rewrite never pulls the old payload back. Persisted at once: a
        consumer on other metal is waiting on exactly this write."""
        digest = self.fingerprint(payload)
        sealed = self.named_meta(subdir, name)
        if sealed is not None:
            if sealed.get("sha256") != digest:
                raise NamedAdapterConflict(
                    f"adapter {name!r} under {subdir!r} is already written "
                    f"({str(sealed.get('sha256'))[:12]}) and was offered "
                    f"different bytes ({digest[:12]}): a name is written "
                    f"once — a changed adapter is a new name")
            return
        self._write_hashed(named_key(subdir, name, NAMED_PAYLOAD), payload, digest)
        self._write(named_key(subdir, name, NAMED_META),
                    _canonical({**dict(meta), "sha256": digest}).encode("utf-8"))
        self._persist()

    def read_named(self, subdir: str, name: str) -> bytes | None:
        """A present name's payload — exactly the bytes `lora_torch.emit`
        produces for one LoRA set — or None while the name is not present.
        None rather than FileNotFoundError because this read is an await
        predicate, like `read_postdata_part`."""
        key = named_key(subdir, name, NAMED_PAYLOAD)
        held = self._named_payloads()
        if key in held:
            digest, payload = held[key]
            if self.fingerprint(payload) == digest:
                held.move_to_end(key)
                return payload
            del held[key]
        meta = self.named_meta(subdir, name)
        if meta is None:
            return None
        digest = meta.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise StoreError(f"named adapter {key!r} has an invalid sha256 seal")
        try:
            payload = self._read_hashed(key, digest)
        except FileNotFoundError as error:
            raise StoreError(f"sealed named adapter {key!r} has no payload") from error
        held[key] = (digest, payload)
        while len(held) > NAMED_PAYLOADS_HELD:
            held.popitem(last=False)
        return payload

    def _named_payloads(self) -> "OrderedDict[str, tuple[str, bytes]]":
        """PRESENT NAMES ARE WRITE-ONCE, SO A PAYLOAD READ ONCE IS READ
        (2026-09-19): a run that stacks its dreamer on a memory hands the same
        ~60 MB to its engine, its learner and its fork fits every update, and
        the scratch store moves about five megabytes a second. The last
        NAMED_PAYLOADS_HELD payloads stay in this process."""
        return self.__dict__.setdefault("_named_payload_cache", OrderedDict())

    def named_meta(self, subdir: str, name: str) -> dict[str, Any] | None:
        """A present name's meta (what its writer said of it, plus `sha256`),
        or None while the name is not present. The meta is the seal, so this
        is also the presence question, at the cost of one small read."""
        try:
            return json.loads(
                self._read(named_key(subdir, name, NAMED_META)).decode("utf-8"))
        except FileNotFoundError:
            return None

    def _promise_files(self) -> dict[str, tuple[str, frozenset[str]]]:
        """Promise files already read, by key: a writer's file never changes
        (a rewrite is this store's own and evicts its entry), so each is read
        once per store object and a waiting consumer's poll costs one listing."""
        return self.__dict__.setdefault("_promise_file_cache", {})

    def promise_named(self, subdir: str, names: Sequence[str], writer: str) -> None:
        """`writer` (a run reference) WILL write these names — said at the
        writer's birth, before its first job, so a consumer that finds a name
        absent can tell "still coming" from "never coming". ONE write, however
        many names (PROMISES_FOLDER). A resubmitted fit run is a new writer
        with its own file, and a name several writers promised is still coming
        while ANY of them is still work; a resumed run restates its own file
        for free."""
        reference = run_reference(writer)
        stated = _canonical({"writer": reference,
                             "names": sorted(check_name(name) for name in names)}).encode("utf-8")
        key = promises_key(subdir, reference)
        try:
            if self._read(key) == stated:
                return
        except FileNotFoundError:
            pass
        self._write(key, stated)
        self._promise_files().pop(key, None)
        self._persist()

    def named_writers(self, subdir: str, name: str) -> tuple[str, ...]:
        """Every run that promised this name: the writers' files, read once
        each (a writer's file never changes), and the one-file-per-name
        promise the first fit runs left."""
        found: list[str] = []
        try:
            raw = self._read(named_key(subdir, name, NAMED_PROMISE))
            found.append(str(json.loads(raw.decode("utf-8"))["writer"]))
        except FileNotFoundError:
            pass
        folder = f"runs/{check_subdir(subdir)}/names/{PROMISES_FOLDER}/"
        for key in sorted(self._list(folder)):
            key = key if key.startswith("runs/") else folder + key.rsplit("/", 1)[-1]
            if key not in self._promise_files():
                try:
                    row = json.loads(self._read(key).decode("utf-8"))
                except FileNotFoundError:
                    continue
                self._promise_files()[key] = (str(row["writer"]), frozenset(row["names"]))
            promiser, promised = self._promise_files()[key]
            if name in promised and promiser not in found:
                found.append(promiser)
        return tuple(found)

    def named_writer(self, subdir: str, name: str) -> str | None:
        """The run that promised this name — one that is still work when
        several did — or None when nobody has."""
        writers = self.named_writers(subdir, name)
        for writer in writers:
            if not run_ended(self, writer):
                return writer
        return writers[-1] if writers else None

    def named_state(self, subdir: str, name: str) -> str:
        """What a consumer can know about a name: `present` (its bytes are
        sealed), `promised` (a run that is still work said it will write it),
        `orphaned` (the run that promised it is done, stopped or failed and
        the bytes are absent — it is never coming), `unknown` (no bytes, no
        promise: its writer may simply not be born yet).

        Presence is asked AGAIN after the writer is found ended: a writer
        that sealed its last name and finished between the two reads left a
        present name, not an orphan."""
        if self.named_meta(subdir, name) is not None:
            return PRESENT
        writer = self.named_writer(subdir, name)
        if writer is None:
            return UNKNOWN
        if not run_ended(self, writer):
            return PROMISED
        return PRESENT if self.named_meta(subdir, name) is not None else ORPHANED

    def describe(self) -> str:
        """Where this store's data lives, for journals and CLIs — a path,
        a bucket, a mount. Backends override; the class name is the floor."""
        return type(self).__name__

    def address(self) -> StoreAddress:
        """This store as a value a child process can reopen it from. Each
        backend answers with its own name; the base refuses, so a backend that
        forgot cannot be reopened by accident."""
        raise NotImplementedError(
            f"{type(self).__name__} has no StoreAddress: a backend a resident "
            f"may reopen must say how (data/stores/address.py)")

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

    def peek_checkpoints(self, run_id: str) -> list[dict[str, Any]]:
        """The checkpoint record, read-only: every durable point, oldest
        first, `{update, versions}`. A directory with no record is a pre-0014
        run, where every ledger line was a checkpoint, so its ledger IS its
        record."""
        home = self.run_prefix(run_id)
        if not self._exists(f"{home}/{CHECKPOINTS}"):
            return self.peek_ledger(run_id)
        return _parse_record(self._read(f"{home}/{CHECKPOINTS}"))

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

    def peek_rollout(self, run_id: str, index: int) -> list[dict[str, Any]] | None:
        """One SEALED generated wave's trajectory rows WITHOUT attaching — the
        rollouts twin of peek_wave, and the door another run's replay reads a
        generation-only run's output through (ADR 0006 Part B). A rollout is
        sealed or absent, never half (write_rollout is atomic), so a peek is
        as safe as reading the ledger. None when that rollout is not sealed."""
        try:
            raw = gzip.decompress(
                self._read(rollout_key(self.run_prefix(run_id), index)))
        except FileNotFoundError:
            return None
        return [json.loads(line) for line in raw.decode("utf-8").split("\n") if line]

    def peek_rollouts_sealed(self, run_id: str) -> int:
        """How many generated waves this run has sealed, WITHOUT attaching.

        The rollouts half of "how far along is this run": a Generator makes
        indices 1..n in order and each write is atomic, so counting the keys
        IS the count and the count IS the progress. Nothing ever sweeps
        rollouts/ (they belong to whichever updates a plan says, so they have
        no update of their own to be uncommitted under)."""
        return sum(1 for key in self._list(f"{self.run_prefix(run_id)}/rollouts")
                   if key.endswith(".jsonl.gz"))

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
        """Every completed eval summary for a PRE-#70 run, WITHOUT attaching —
        the legacy in-run eval's held-out series. New runs measure OUTSIDE the
        run dir (measurements/); this stays so history renders."""
        out = []
        for key in self._list(f"{self.run_prefix(run_id)}/eval"):
            if not key.endswith("/summary.json"):
                continue
            try:
                out.append(json.loads(self._read(key).decode("utf-8")))
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        return sorted(out, key=lambda s: s.get("update", 0))

    # ---- measurements: observation OUTSIDE the run (#70) ---------------------
    #
    # measurements/<run_id>/<name>/manifest.json + points.jsonl. Not identity,
    # not the run dir, not resume-equivalence: a Measurement is an observation
    # OF a run, configured by its own manifest, appended point by point, and
    # deletable — supersede by NAME rather than rewriting one. One writer per
    # (run_id, name); the observer reads them beside the legacy eval/.

    def open_measurement(self, run_id: str, name: str,
                         manifest: Mapping[str, Any]) -> None:
        """Write-once config: a second open with the SAME manifest is a no-op
        (the measurer's restart), a different one is refused — a changed
        observation is a NEW name, so no points file ever mixes configs."""
        key = f"measurements/{run_id}/{name}/manifest.json"
        stated = _canonical(dict(manifest))
        if self._exists(key):
            if self._read(key).decode("utf-8") != stated:
                raise StoreError(
                    f"measurement {name!r} of {run_id!r} exists with a "
                    f"different manifest — a changed observation is a new "
                    f"name, never a rewrite")
            return
        self._write(key, stated.encode("utf-8"))

    def append_measurement_point(self, run_id: str, name: str,
                                 row: Mapping[str, Any]) -> None:
        self._append_line(
            f"measurements/{run_id}/{name}/points.jsonl",
            json.dumps(dict(row), sort_keys=True, separators=(",", ":")))

    def measured_updates(self, run_id: str, name: str) -> set[int]:
        """The updates this measurement already holds — the idempotence key
        a measuring pass skips by."""
        return {int(row["update"])
                for row in self._read_log(
                    f"measurements/{run_id}/{name}/points.jsonl")
                if "update" in row}

    def read_measurements(self, run_id: str) -> dict[str, dict[str, Any]]:
        """{name: {"manifest": ..., "points": [...]}} for one run — the
        observer's read, beside the legacy eval summaries."""
        out: dict[str, dict[str, Any]] = {}
        for key in self._list(f"measurements/{run_id}/"):
            if not key.endswith("/manifest.json"):
                continue
            name = key.split("/")[-2]
            try:
                manifest = json.loads(self._read(key).decode("utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            points = sorted(
                self._read_log(f"measurements/{run_id}/{name}/points.jsonl"),
                key=lambda r: r.get("update", 0))
            out[name] = {"manifest": manifest, "points": points}
        return out

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


# ---------------------------------------------------------------------------
# how far a run got, for every kind of run — read off the store, by everyone
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunProgress:
    """HOW FAR A RUN GOT, against the plan that is its EXTENT (ADR 0006 Part
    B): updates committed for a run that trains, rollouts sealed for one that
    only generates.

    `extent` names which plan the numbers count ("" when the run has neither
    on record — a directory still being created), and `planned` is None for
    the same reason, which is why a reader that renders "?" can tell "not yet
    known" from "none yet done".
    """

    extent: str
    completed: int
    planned: int | None
    checkpointed: int | None = None

    @property
    def done(self) -> bool:
        """A run is done when its EXTENT is: the ledger reached the train
        plan's length AND the checkpoint record sealed that last update (ADR
        0014 — a run whose final blobs never landed rewinds on attach and is
        still work), or the rollout plan's last wave is sealed. A run with
        no plan on record is still work — the one predicate the desk's
        reaper, the observer and the host all read, so no copy of it can
        drift (Q3)."""
        if self.planned is None or self.completed < self.planned:
            return False
        return self.checkpointed is None or self.checkpointed >= self.planned


def run_progress(store: "Store", run_id: str) -> RunProgress:
    """One run's progress from read-only peeks — an observer never attaches.

    The train plan is the extent where a run has one (one wave is one gradient
    update, so the ledger is the count); otherwise the rollout plan is, and
    the sealed rollouts are the count. No Sealer and no ledger is needed for
    the second: `write_rollout` is atomic and nothing ever sweeps rollouts/.
    """
    # imported here, not at module scope: rlstack.data's package __init__
    # imports stores, so a top-level import back into the package would be a
    # cycle. Still inside the membrane — data/ imports no other package.
    from rlstack.data.plan import wave_count

    train = store.peek_plan(run_id, "train")
    if train is not None:
        entries = store.peek_ledger(run_id)
        sealed = store.peek_checkpoints(run_id)
        return RunProgress("train",
                           int(entries[-1]["update"]) if entries else 0,
                           wave_count(train),
                           checkpointed=int(sealed[-1]["update"]) if sealed else 0)
    fit = store.peek_plan(run_id, "fit")
    if fit is not None:
        # a FIT RUN (ADR 0019): one ledger line per fit job, so the ledger is
        # the count and the fit plan's length the extent, sealed like a
        # training run's by its checkpoint record
        entries = store.peek_ledger(run_id)
        sealed = store.peek_checkpoints(run_id)
        return RunProgress("fit",
                           int(entries[-1]["update"]) if entries else 0,
                           wave_count(fit),
                           checkpointed=int(sealed[-1]["update"]) if sealed else 0)
    rollout = store.peek_plan(run_id, "rollout")
    if rollout is None:
        return RunProgress("", 0, None)
    return RunProgress("rollout", store.peek_rollouts_sealed(run_id),
                       wave_count(rollout))


def run_done(store: "Store", run_id: str) -> bool:
    """Is this run still WORK? Its extent, read off the store, because the
    store is the run (I10) — what the desk's reaper asks of a run whose host
    died, and what the observer's "done" means."""
    return run_progress(store, run_id).done


def standing_disposition(store: "Store", run_id: str) -> str | None:
    """`stopped` or `failed` when that is the desk's standing word on this
    run, else None — the fleet journal's fold (ADR 0014, Part C) read for ONE
    run: a deliberate stop or the run's own death stands until a later
    DELIVERED placement supersedes it, because resubmitting is how a run
    moves again. `parked` is not an answer here: a parked run is retried.
    The journal names a run by its bare id, wherever it was filed.

    The journal is observability and no byte of any run is computed from it;
    this read can only turn a consumer's WAIT into a REFUSAL (`run_ended`),
    which is the one thing a stop or a failure is recorded nowhere else to
    say."""
    bare = run_id.rsplit("/", 1)[-1]
    standing: str | None = None
    for event in store.read_fleet_log():
        if event.get("run_id") != bare:
            continue
        kind = event.get("event")
        if kind in ("stopped", "failed"):
            standing = kind
        elif kind == "place" and event.get("delivered") and event.get("accepted"):
            standing = None
    return standing


def run_ended(store: "Store", run_id: str) -> bool:
    """Will this run never write again? Its extent is done, or the desk's
    standing word on it is `stopped` or `failed`. What turns a promised name
    into an orphan (ADR 0019): a consumer waits on a writer that is still
    work — parked, placed, or not yet born — and refuses one that ended."""
    return run_done(store, run_id) or standing_disposition(store, run_id) is not None


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
        self._repair_record(self.ledger_key)

    def _repair_record(self, key: str) -> None:
        """Rewrite one append-only jsonl record without its torn tail."""
        try:
            raw = self.store._read(key)
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
            self.store._write(key, raw[:keep])

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

    # ---- the checkpoint record (ADR 0014) -----------------------------------

    @property
    def checkpoints_key(self) -> str:
        return self._key(CHECKPOINTS)

    def read_checkpoints(self) -> list[dict[str, Any]]:
        """Every durable point, oldest first: `{update, versions}`, the last
        being what resume restores. A torn final line is dropped. A run with
        no record at all is a pre-0014 directory: every ledger line was a
        checkpoint, so the ledger is returned as the record."""
        try:
            raw = self.store._read(self.checkpoints_key)
        except FileNotFoundError:
            return self.read_ledger()
        return _parse_record(raw)

    def checkpoint_tail(self) -> dict[str, Any] | None:
        """The last durable point — what resume restores and attach rewinds
        to — or None on a run that has never checkpointed."""
        entries = self.read_checkpoints()
        return entries[-1] if entries else None

    def append_checkpoint(self, update: int, versions: Mapping[str, int]) -> None:
        """THE DURABLE POINT: written AFTER every trainable entry's blobs at
        `versions` are on the store, and the Trainer's alone. `update` must
        strictly increase and may not run ahead of the ledger (0 is the
        initial blobs, before any line); the versions must be the ones the
        ledger line at that update committed."""
        self._repair_record(self.checkpoints_key)
        if isinstance(update, bool) or not isinstance(update, int) or update < 0:
            raise LedgerError(f"checkpoint 'update' must be an int >= 0, got {update!r}")
        try:
            previous = _parse_record(self.store._read(self.checkpoints_key))
        except FileNotFoundError:
            previous = []
        if previous and update <= int(previous[-1]["update"]):
            raise LedgerError(
                f"checkpoint update must strictly increase: {update} <= {previous[-1]['update']}")
        tail = self.ledger_tail()
        committed = int(tail["update"]) if tail is not None else 0
        if update > committed:
            raise LedgerError(
                f"checkpoint {update} runs ahead of the ledger ({committed}): a "
                f"checkpoint seals a committed update")
        if tail is not None and int(tail["update"]) == update \
                and isinstance(tail.get("versions"), dict) \
                and {k: int(v) for k, v in tail["versions"].items()} != \
                {k: int(v) for k, v in versions.items()}:
            raise LedgerError(
                f"checkpoint {update} names versions {dict(versions)} but the ledger "
                f"line at {update} committed {tail['versions']}")
        self.store._append_line(self.checkpoints_key, _canonical(
            {"update": update, "versions": {k: int(v) for k, v in sorted(versions.items())}}))

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
        PREDICATE: the Trainer blocks on exactly this absence, and a runner
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
        tail = self.checkpoint_tail()
        named = []
        for section, name, version in policy.expendable(self.read_checkpoints()):
            self._refuse_live_version(tail, section, name, version)
            named.append(((section, name, version),
                          self._blob_key(section, name, version)))

        # Delete only files this store actually lists. On a mounted volume,
        # _exists also asks the committed remote view after a local miss:
        # checking every already-swept version that way makes U updates cost
        # O(U^2) network lookups. A listed blob is also one this writer can
        # size and unlink; remote-only blobs belong to the other mount.
        present = set()
        for section in {triple[0] for triple, _ in named}:
            present.update(self.store._list(self._key(section)))
        freed, gone = 0, []
        for triple, key in named:
            if key not in present:
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
        """The version the CHECKPOINT tail names is the run's restorable
        state — resume reads exactly it — so no policy may free it, whatever
        it says.

        The floor under every retention policy, enforced here and not there: a
        swept run still resumes.
        """
        if tail is None:
            return
        live = tail.get("versions")
        if isinstance(live, dict) and name in live and int(version) == int(live[name]):
            raise StoreError(
                f"retention named {section}/{name}@{version}, the version the "
                f"checkpoint tail seals: the tail is what resume reads")

    # ---- eval ---------------------------------------------------------------

    def read_eval(self, update: int, filename: str) -> str:
        """A PRE-#70 run's in-run eval output — legacy read, nothing writes."""
        return self.store._read(self._key("eval", str(update), filename)).decode("utf-8")

    # ---- crash recovery (runs on every attach) ------------------------------

    def _discard_unsealed(self) -> None:
        """THE REWIND (ADR 0014): drop everything the checkpoint record never
        sealed — work is durable at its checkpoint and nowhere else.

        Torn tails of both records are repaired away. With the checkpoint
        tail at update c and versions V_c: ledger lines above c are cut (the
        learner state that produced them is gone, so the Trainer redoes them
        and writes the same lines again); per-update artifacts (waves,
        postdata, and the postdata PARTS a Scorer wrote ahead of the Trainer)
        above c are deleted; blob versions above V_c are deleted; rollouts
        pinned to a version above V_c are deleted, newest first, because a
        rollout sampled at a version that no longer exists would pin a bundle
        no store can rebuild; backend write debris is swept. A pre-0014
        directory has no record and every ledger line was a checkpoint, so
        for it this is exactly the old rule.

        Redoing costs nothing but compute: every discarded byte is a pure
        function of (spec, code, data) from the checkpoint on, which is what
        `tests/test_resume.py` proves at every cadence.
        """
        self._repair_ledger()
        self._repair_record(self.checkpoints_key)
        sealed = self.read_checkpoints()
        # no checkpoint at all: NOTHING is sealed, update 0's artifacts included
        tail_update = int(sealed[-1]["update"]) if sealed else -1
        self._cut_ledger_above(tail_update)

        for section, suffix in UPDATE_SECTIONS:
            for key in self.store._list(self._key(section)):
                update = _parse_update(key, suffix)
                if update is not None and update > tail_update:
                    self.store._delete(key)

        committed = _committed_versions(sealed)
        if committed is not None:
            for section in BLOB_SECTIONS:
                for key in self.store._list(self._key(section)):
                    name, version = _parse_blob(key)
                    if version is not None and version > committed.get(name, 0):
                        self.store._delete(key)
            self._discard_rollouts_above(committed)

        self.store._sweep_partial(self._key())

    def _cut_ledger_above(self, update: int) -> None:
        """Rewrite the ledger with every line above `update` removed — the
        provisional tail past the checkpoint, cut exactly as a torn line is.
        The kept bytes are the kept lines, untouched."""
        try:
            raw = self.store._read(self.ledger_key)
        except FileNotFoundError:
            return
        keep = 0
        for line in raw.split(b"\n"):
            if not line:
                break
            if int(json.loads(line.decode("utf-8"))["update"]) > update:
                break
            keep += len(line) + 1
        if keep < len(raw):
            self.store._write(self.ledger_key, raw[:keep])

    def _discard_rollouts_above(self, committed: Mapping[str, int]) -> None:
        """Delete every sealed rollout whose turns pin a version above the
        checkpoint's, NEWEST FIRST and stopping at the first that does not:
        the Generator samples in index order at a version that only ever
        rises, so the discarded set is a suffix and the scan costs one read
        per discarded rollout plus one."""
        indexed = []
        for key in self.store._list(self._key("rollouts")):
            index = _parse_update(key, ".jsonl.gz")
            if index is not None:
                indexed.append((index, key))
        for index, key in sorted(indexed, reverse=True):
            raw = gzip.decompress(self.store._read(key)).decode("utf-8")
            rows = [json.loads(line) for line in raw.split("\n") if line]
            if not _pinned_above(rows, committed):
                break
            self.store._delete(key)


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


def _pinned_above(rows: Sequence[Mapping[str, Any]],
                  committed: Mapping[str, int]) -> bool:
    """Does any turn in these sealed rows pin a delta version above the
    checkpoint's? A delta the checkpoint never named has sealed version 0."""
    for row in rows:
        for turn in row.get("turns", ()):
            for name, version in (turn.get("policy_version") or {}).items():
                if int(version) > int(committed.get(str(name), 0)):
                    return True
    return False


def _parse_record(raw: bytes) -> list[dict[str, Any]]:
    """One append-only jsonl record's entries; a torn final line is dropped,
    any other bad line is a corrupt record."""
    lines = [line for line in raw.decode("utf-8").split("\n") if line]
    entries: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise LedgerError(f"corrupt record line {index}") from None
    return entries


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
