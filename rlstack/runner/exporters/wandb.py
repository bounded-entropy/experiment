"""Read committed metrics into one offline W&B export attempt.

Training reads its ledger; generation-only runs read atomically sealed rollout
waves. Generation metrics count actual completions, never fabricated losses.

This worker never attaches or repairs a run. Its cursor is in memory: repeated
polls export each committed update once in this process. A restart creates a
fresh bundle and replays history, because W&B offline resume cannot atomically
share a durable cursor with our ledger. Re-declare the same Strange Loop run
name with ONLY the replacement bundle's path; previous attempts remain evidence.
No objective is manufactured when the explicitly selected metric is absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from rlstack.data.stores.base import LedgerError, Store, StoreError, run_done, run_reference
from rlstack.data.stores.local import LocalStore


@dataclass(frozen=True)
class ExportProgress:
    logged: int
    last_update: int | None
    objective_points: int
    missing_objective_updates: tuple[int, ...]
    wandb_path: str
    attempt_id: str


def committed_entries(store: Store, run_ref: str) -> list[dict]:
    """Only complete ledger lines count; corrupt complete history is an error."""
    try:
        raw = store._read(store.run_prefix(run_ref) + "/ledger.jsonl")
    except FileNotFoundError:
        return []
    entries = []
    previous = -1
    for line in raw.split(b"\n")[:-1]:
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            raise LedgerError("export encountered a corrupt complete ledger line") from None
        update = entry.get("update") if isinstance(entry, dict) else None
        if isinstance(update, bool) or not isinstance(update, int) or update <= previous:
            raise LedgerError("export requires nonnegative, strictly increasing ledger updates")
        entries.append(entry)
        previous = update
    return entries


def ledger_metrics(entry: dict, objective: str) -> dict[str, int | float]:
    """Namespace real numeric ledger values; omit missing or nonfinite values."""
    metrics: dict[str, int | float] = {"update": entry["update"]}
    for section in ("post", "train", "wave"):
        for name, value in entry.get(section, {}).items():
            if (not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(value)):
                metrics[f"{section}/{name}"] = value
    if objective in metrics:
        metrics["objective"] = metrics[objective]
    return metrics


def rollout_metrics(index: int, rows: list[dict]) -> dict:
    """A sealed generation wave's measured counts; update here means wave index."""
    return {"update": index, "wave": {
        "trajectories": len(rows),
        "completion_tokens": sum(len(turn["token_ids"]) for row in rows for turn in row["turns"]),
        "truncated_trajectories": sum(any(turn["finish"] == "length" for turn in row["turns"])
                                      for row in rows)}}


class OfflineWandbRun:
    """The only W&B dependency; initialization stays out of package imports."""

    def __init__(self, *, store: str, run_ref: str, objective: str,
                 directory: Path, lease_id: str, project: str, attempt_id: str) -> None:
        import wandb

        directory.mkdir(parents=True, exist_ok=True)
        group = hashlib.sha256((store + "\n" + run_ref).encode()).hexdigest()[:32]
        self._run = wandb.init(
            mode="offline", project=project, dir=str(directory), id=attempt_id,
            name=f"{run_ref} / {lease_id} / export {attempt_id[:8]}", group=group,
            job_type="ledger-export", reinit="create_new", save_code=False,
            config={"store": store, "run_ref": run_ref, "lease_id": lease_id,
                    "objective_metric": objective, "export_attempt": attempt_id,
                    "restart_policy": "new bundle, full committed history"},
        )
        self.path = str(Path(self._run.dir).parent)

    def log(self, metrics: dict[str, int | float], *, step: int) -> None:
        self._run.log(metrics, step=step, commit=True)

    def finish(self) -> None:
        self._run.finish()


class LedgerWandbExporter:
    """One exact run's committed observations, independent of its trainer."""

    def __init__(self, store: Store, run_ref: str, *, objective: str,
                 directory: str | Path, lease_id: str, project: str = "rlstack") -> None:
        section, separator, name = objective.partition("/")
        if section not in ("post", "train", "wave") or not separator or not name:
            raise ValueError("objective must explicitly name a ledger metric, such as train/loss")
        self.run_ref = run_reference(run_ref)
        self.store = store
        manifest = store.peek_manifest(self.run_ref)
        if manifest is None:
            raise StoreError(f"export run does not exist at {self.run_ref!r}")
        spec = json.loads(manifest["spec"]) if "spec" in manifest else {}
        self._generation_only = bool(spec) and spec.get("algo") is None
        self._rollout_entries: list[dict] = []
        target = Path(directory).resolve()
        if isinstance(store, LocalStore) and target.is_relative_to(store.root.resolve()):
            raise StoreError("W&B export directory must be outside the authoritative store")
        self.objective = objective
        self.attempt_id = uuid.uuid4().hex
        self._wandb = OfflineWandbRun(
            store=store.describe(), run_ref=self.run_ref, objective=objective,
            directory=target, lease_id=lease_id, project=project, attempt_id=self.attempt_id)
        self.wandb_path = self._wandb.path
        self._history: list[str] = []
        self._last_update: int | None = None
        self._objective_points = 0
        self._failed = False
        self._finished = False
        self._lock = threading.Lock()

    def poll(self) -> ExportProgress:
        """Log new commits once in this attempt; fail closed after uncertain logging."""
        with self._lock:
            if self._failed or self._finished:
                raise StoreError("this export attempt stopped; restart with a fresh offline bundle")
            if self._generation_only:
                # Sealed rollouts are immutable and never rewound. Read each
                # large text artifact once, retaining only its measured counts.
                while True:
                    index = len(self._rollout_entries) + 1
                    rows = self.store.peek_rollout(self.run_ref, index)
                    if rows is None:
                        break
                    self._rollout_entries.append(rollout_metrics(index, rows))
                entries = self._rollout_entries
            else:
                entries = committed_entries(self.store, self.run_ref)
            signatures = [hashlib.sha256(json.dumps(entry, sort_keys=True,
                                                   separators=(",", ":")).encode()).hexdigest()
                          for entry in entries]
            if signatures[:len(self._history)] != self._history:
                self._failed = True
                raise LedgerError("previously exported committed history changed or disappeared")
            logged = 0
            missing = []
            for entry, signature in zip(entries[len(self._history):], signatures[len(self._history):]):
                metrics = ledger_metrics(entry, self.objective)
                try:
                    self._wandb.log(metrics, step=entry["update"])
                except Exception:
                    self._failed = True
                    raise
                self._history.append(signature)
                self._last_update = entry["update"]
                self._objective_points += int("objective" in metrics)
                if "objective" not in metrics:
                    missing.append(entry["update"])
                logged += 1
            return ExportProgress(logged, self._last_update, self._objective_points,
                                  tuple(missing), self.wandb_path, self.attempt_id)

    def finish(self) -> None:
        """Flush this bundle without acknowledging or changing any source commit."""
        with self._lock:
            if not self._finished:
                try:
                    self._wandb.finish()
                finally:
                    self._finished = True


def _ready(path: Path, value: dict) -> None:
    """Publish exporter state atomically outside the scientific store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as target:
        json.dump(value, target)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def await_manifest(store: Store, run_ref: str, timeout_s: float = 45.0) -> None:
    """Adoption can precede manifest publication; wait only for that exact run.

    No run is created, repaired or rediscovered. Corruption and transport
    errors still propagate; only an absent manifest is retried, boundedly.
    """
    deadline = time.monotonic() + timeout_s
    while store.peek_manifest(run_ref) is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StoreError(f"export run does not exist at {run_ref!r} after publication wait")
        time.sleep(min(5.0, remaining))


def main(argv: list[str] | None = None) -> int:
    """A separately supervised worker; config credentials never enter arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--every", type=float, default=10.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.every) or args.every <= 0:
        parser.error("--every must be finite and positive")
    try:
        config = json.loads(args.config.read_text())
    finally:
        args.config.unlink(missing_ok=True)
    environment = config.get("environment", {})
    if set(environment) - {"SL_API_TOKEN", "SL_API_TOKEN_FILE", "SL_API_BASE"}:
        raise ValueError("export config accepts only scratch credential environment keys")
    os.environ.update(environment)
    from rlstack.data.stores.strangeloop import ScratchClient, StrangeLoopStore

    ready_file = Path(os.path.expandvars(config["ready_file"]))
    directory = Path(os.path.expandvars(config["directory"]))
    if "$" in str(ready_file) or "$" in str(directory):
        raise ValueError("export paths refer to an unavailable environment variable")
    if ready_file.resolve().is_relative_to(Path("/scratch").resolve()):
        raise StoreError("export readiness state must live outside scratch")
    try:
        store = StrangeLoopStore(ScratchClient.from_locator(config["store"]), read_only=True)
        source_root = (Path("/scratch") / store.scratch.prefix).resolve()
        if any(path.resolve().is_relative_to(source_root) for path in (ready_file, directory)):
            raise StoreError("export output and state must stay outside the scratch store")
        await_manifest(store, config["run_ref"])
        exporter = LedgerWandbExporter(
            store, config["run_ref"], objective=config["objective"], directory=directory,
            lease_id=config["lease_id"], project=config.get("project", "rlstack"))
    except Exception as error:
        _ready(ready_file, {"state": "failed", "error": type(error).__name__})
        raise
    state = {"wandb_path": exporter.wandb_path, "attempt_id": exporter.attempt_id,
             "state": "running", "run_ref": exporter.run_ref, "pid": os.getpid()}
    stop = threading.Event()
    def request_stop(signum, frame):
        stop.set()
    previous = {number: signal.signal(number, request_stop)
                for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        _ready(ready_file, state)
        while True:
            progress = exporter.poll()
            state.update(asdict(progress))
            _ready(ready_file, state)
            if run_done(store, exporter.run_ref) or stop.wait(args.every):
                # A commit can land between the last poll and observing done/stop.
                progress = exporter.poll()
                state.update(asdict(progress))
                break
        exporter.finish()
        state["state"] = "finished"
        _ready(ready_file, state)
    except Exception as error:
        state.update(state="failed", error=type(error).__name__)
        _ready(ready_file, state)
        raise
    finally:
        try:
            exporter.finish()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
