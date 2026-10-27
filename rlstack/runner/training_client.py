"""Compose ordinary Trainer runs and measure isolated adapter states.

This client never drives optimizer steps. A processor supplies a full child
spec with sealed Replay data; the normal experiment loop owns every update,
checkpoint, stop and resume. Measurements use a separate installed tenant.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from functools import lru_cache

from rlstack.code import load_code
from rlstack.client import GradientRow, TrainingResult, TrainingSnapshot
from rlstack.data.flatten import flatten, pack
from rlstack.data.stores.base import NamedAdapterConflict, RunHandle
from rlstack.data.trajectory import trajectory_from_row
from rlstack.policy.siteschema import SiteSchema, resolve
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.gradient_measurement import (
    finish_thread_call, gradient_batches, gradient_row_record,
)
from rlstack.runner.interfaces import EntryInstall, Learner, OptimSettings, Parameterization
from rlstack.runner.meters import HostJournal
from rlstack.runner.names import names_subdir
from rlstack.runner.refs import RefReader
from rlstack.runner.signals import store_work
from rlstack.runner.weight_transfer import WeightTransfer
from rlstack.spec.canonical import canonical_json
from rlstack.spec.specs import AdapterSpec, ExperimentSpec


class TrainingClient:
    def __init__(self, learner: Learner, arbiter: Arbiter, *, tenant: str,
                 refs: RefReader, schema: SiteSchema, seed: int,
                 microbatch_tokens: int, journal: HostJournal | None = None,
                 weight_transfer: WeightTransfer = WeightTransfer(),
                 code: tuple[str, ...] = ()) -> None:
        self.learner, self.arbiter = learner, arbiter
        self.tenant, self.refs, self.schema = tenant, refs, schema
        self.base, self.seed = schema.base, seed
        self.code = load_code(code)
        self.microbatch_tokens = microbatch_tokens
        self.journal, self.weight_transfer = journal, weight_transfer
        self.update = 0
        self.lock = asyncio.Lock()
        self.measurement_tenant = tenant + ":measurement"
        self.measurement_key: tuple[str, str | None] | None = None
        self.tokenizer = lru_cache(maxsize=4096)(
            lambda text: tuple(learner.tokenize(tenant, text)))

    async def run(self, spec: ExperimentSpec, *, checkpoint_every: int,
                  save_state: bool = True) -> TrainingResult:
        """Run one ordinary child spec on the learner already held by this run."""
        from rlstack.runner.loop import experiment_identity, run_experiment_async
        from rlstack.runner.roles.base import StopRequest

        if spec.policy.base != self.base:
            raise ValueError("a child run must use its parent's resident base")
        if spec.plans.rollout is not None or spec.gen is not None:
            raise ValueError("child training consumes sealed Replay data; it cannot generate")
        from rlstack.registry import POST
        if spec.algo is None or any(POST.get(name).training or POST.get(name).pools
                                   for name in spec.algo.post):
            raise ValueError("child supervision declares a loss and only arithmetic postprocessing")
        subdir = self.child_subdir()
        spec = replace(spec, parent_run=(self.refs.run.run_dir or f"runs/{self.tenant}").removeprefix("runs/"))
        child_id = experiment_identity(spec, self.schema)
        checkpointing = Checkpointing(every=checkpoint_every,
                                      state="save" if save_state else "recompute")
        async with self.lock:
            stop = StopRequest()
            operation = asyncio.create_task(run_experiment_async(
                spec, self.schema, self.refs.store, {}, self.learner,
                arbiter=self.arbiter, journal=self.journal, subdir=subdir,
                checkpointing=checkpointing,
                weight_transfer=self.weight_transfer, stop=stop))
            try:
                try:
                    report = await asyncio.shield(operation)
                except asyncio.CancelledError:
                    # Cancelling a coroutine cannot stop synchronous learner
                    # work. Drain through the normal Trainer before releasing
                    # this client's lock or uninstalling its child tenant.
                    stop.request("parent supervision call cancelled")
                    while not operation.done():
                        try:
                            await asyncio.shield(operation)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if not operation.cancelled():
                        operation.exception()
                    raise
                run_ref = f"{subdir}/{report.run_id}"
                checkpoints = await store_work(self.refs.store.peek_checkpoints, run_ref)
                ledger = await store_work(self.refs.store.peek_ledger, run_ref)
                versions = dict((checkpoints or ledger)[-1]["versions"]) if checkpoints or ledger else {}
            finally:
                # Failure, cancellation and successful completion all release
                # the resident state; committed checkpoints remain resumable.
                async with self.arbiter.admit(self.learner):
                    await finish_thread_call(lambda: self.learner.uninstall(child_id))
        if self.journal is not None:
            await store_work(self.journal.append, {
                "event": "child-training", "t": time.time(),
                "run_id": self.tenant, "update": self.update,
                "child_run_ref": run_ref, "completed": report.completed,
                "checkpoint_every": checkpoint_every, "save_state": save_state})
        return TrainingResult(report.run_id, run_ref, report.completed, versions, report.stopped)

    def child_subdir(self) -> str:
        # RunHandle.prefix includes the backend's run directory; names_subdir
        # returns the portable experiment folder rather than a filesystem path.
        folder = "/".join((self.refs.run.run_dir or f"runs/{self.tenant}").split("/")[1:-1])
        return f"{folder}/children/{self.tenant}" if folder else f"children/{self.tenant}"

    async def snapshot(self, run_ref: str) -> TrainingSnapshot:
        """Read the exact checkpoint tail without attaching or sweeping it."""
        store = self.refs.store
        manifest = await store_work(store.peek_manifest, run_ref)
        if manifest is None:
            raise FileNotFoundError(run_ref)
        handle = RunHandle(store, run_ref.rsplit("/", 1)[-1], json.dumps(manifest),
                           store.run_prefix(run_ref))
        tail = await store_work(handle.checkpoint_tail)
        if tail is None:
            raise ValueError(f"child run {run_ref!r} has no checkpoint")
        versions = {name: int(version) for name, version in tail["versions"].items()}
        adapters, optim = {}, {}
        for name, version in versions.items():
            adapters[name] = await store_work(handle.read_blob, "adapters", name, version)
            if handle.has_blob("optim", name, version):
                optim[name] = await store_work(handle.read_blob, "optim", name, version)
        return TrainingSnapshot(adapters, optim, versions)

    def ensure_measurement(self, adapter: AdapterSpec, payload: bytes | None) -> None:
        """An unchanged measurement state retains its bounded gradient cache."""
        configured = replace(adapter, route=None, trainable=True,
                             init={"seed": self.seed, **adapter.init})
        key = (canonical_json(configured),
               hashlib.sha256(payload).hexdigest() if payload is not None else None)
        if key == self.measurement_key:
            return
        if self.measurement_key is not None:
            self.learner.uninstall(self.measurement_tenant)
            self.measurement_key = None
        sites = resolve(self.schema.sites, configured.site)
        if not sites:
            raise ValueError(f"measurement adapter site {configured.site!r} matches nothing")
        self.learner.install(self.measurement_tenant, Parameterization(
            base=self.base, loss="sequence_sft", code=self.code,
            entries=(EntryInstall(name="adapter", adapter_type=configured.adapter_type,
                                  sites=tuple(sites), init=dict(configured.init), trainable=True),),
            optim=OptimSettings("adamw", 1e-4, (0.9, 0.95), 0.0, {})))
        try:
            if payload is not None:
                self.learner.load(self.measurement_tenant, {"adapter": payload}, None)
        except BaseException:
            self.learner.uninstall(self.measurement_tenant)
            raise
        self.measurement_key = key

    async def evaluate(self, adapter: AdapterSpec, payload: bytes | None,
                       rows: Sequence[Mapping]) -> tuple[float, ...]:
        """Per-document token-mean NLL, with no optimizer operation."""
        async with self.lock:
            async with self.arbiter.admit(self.learner):
                def measure():
                    self.ensure_measurement(adapter, payload)
                    docs = [(flatten(trajectory_from_row(row), self.tokenizer), {}) for row in rows]
                    return tuple(value for batch in pack(docs, self.microbatch_tokens)
                                 for value in self.learner.forward(self.measurement_tenant, batch))
                return await finish_thread_call(measure)

    async def gradient_cosines(self, start_payload: bytes | None,
                               reference_rows: Sequence[GradientRow],
                               candidate_rows: Sequence[Sequence[GradientRow]], *,
                               adapter: AdapterSpec, teacher_prefix_ids: tuple[int, ...] | None = None):
        """Compare exact masked documents at an explicitly declared adapter."""
        if not candidate_rows:
            return ()
        reference = gradient_batches(reference_rows, None, self.microbatch_tokens)
        candidates = tuple(gradient_batches(rows, None, self.microbatch_tokens)
                           for rows in candidate_rows)
        requested = time.monotonic()
        async with self.lock:
            admitted = time.monotonic()
            async with self.arbiter.admit(self.learner):
                def measure():
                    self.ensure_measurement(adapter, start_payload)
                    return self.learner.gradient_cosines(
                        self.measurement_tenant, "adapter", reference, candidates,
                        **({"teacher_prefix_ids": teacher_prefix_ids}
                           if teacher_prefix_ids is not None else {}))
                results = tuple(await finish_thread_call(measure))
            if len(results) != len(candidate_rows):
                raise RuntimeError("gradient result count differs from candidate group")
            if self.journal is not None:
                await store_work(self.journal.append, {
                    "event": "gradient_comparison", "t": time.time(),
                    "run_id": self.tenant, "update": self.update,
                    "student_tenant": self.measurement_tenant,
                    "cadence": "every candidate group",
                    "seconds": time.monotonic() - requested,
                    "queue_seconds": admitted - requested,
                    "student_initialization": {
                        "base": self.base, "seed": self.seed,
                        "adapter": json.loads(canonical_json(adapter)),
                        "start_payload_sha256": self.measurement_key[1]},
                    "references": [gradient_row_record(row) for row in reference_rows],
                    "candidates": [[gradient_row_record(row) for row in rows]
                                   for rows in candidate_rows],
                    "results": [asdict(result) for result in results],
                    "candidate_objective": "sft" if teacher_prefix_ids is None else "teacher_kl",
                    "teacher_prefix_ids": teacher_prefix_ids,
                    "reduction": "reference NLL; candidate " + ("NLL" if teacher_prefix_ids is None else "KL(teacher||student)") + "; mean selected tokens then mean documents; global cosine"})
            return results

    async def tokenize(self, text: str) -> tuple[int, ...]:
        async with self.arbiter.admit(self.learner):
            return await finish_thread_call(lambda: self.tokenizer(text))

    async def rows(self, uri: str) -> list[dict]:
        file, _, span = uri.partition("#")
        if not file.startswith("cas://"):
            raise ValueError("sealed supervision rows must be content-addressed")
        rows = await store_work(self.refs.rows, file)
        if rows is None:
            raise FileNotFoundError(file)
        if not span:
            return rows
        first, colon, last = span.partition(":")
        if not colon or not first.isdigit() or not last.isdigit() or not 0 <= int(first) <= int(last) <= len(rows):
            raise ValueError(f"invalid sealed row span {uri!r}")
        return rows[int(first):int(last)]

    async def put(self, payload: bytes) -> str:
        return await store_work(self.refs.store.cas_put, payload)

    async def publish(self, name: str, payload: bytes, meta: Mapping) -> None:
        """A recipe's named output uses the store's existing write-once rule."""
        subdir = names_subdir(self.refs.run)
        previous = await store_work(self.refs.store.named_meta, subdir, name)
        if previous is not None and canonical_json({
                key: value for key, value in previous.items() if key != "sha256"
                }) != canonical_json(dict(meta)):
            raise NamedAdapterConflict(
                f"adapter {name!r} under {subdir!r} has different scientific provenance")
        await store_work(self.refs.store.write_named, subdir,
                         name, payload, dict(meta))

    async def probes(self, run_ref: str, update: int) -> dict:
        manifest = await store_work(self.refs.store.peek_manifest, run_ref)
        if manifest is None:
            raise FileNotFoundError(run_ref)
        handle = RunHandle(self.refs.store, run_ref.rsplit("/", 1)[-1], json.dumps(manifest),
                           self.refs.store.run_prefix(run_ref))
        return await store_work(handle.read_postdata_part, update, "probe") or {}

    async def close(self) -> None:
        async with self.lock:
            if self.measurement_key is not None:
                async with self.arbiter.admit(self.learner):
                    await finish_thread_call(lambda: self.learner.uninstall(self.measurement_tenant))
                self.measurement_key = None
