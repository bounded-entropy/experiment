"""The fit loop: many small LoRA fits as ONE client loop over the learner's
verbs (ADR 0019).

A FIT is `load_set` → steps of `forward_backward` + `optim_step` → `forward`
(the probe) → `emit_set`, driven from here and never from inside the learner:
the learner stays the engine's backprop twin, and a fit is its client. One
tenant's `dream_bank` entry holds K memory sets, and each is a LANE: a job
occupies one lane from its `load_set` to its `emit_set`, every active lane
contributes its next batch to each step, and the lanes' rows share the
forwards. A lane that finishes takes the next job; a lane with no rows in a
step gets no gradient and is not stepped.

    Stage       rows × epochs at one batch size and one peak lr, decayed
                (declared in rlstack/client.py: a processor hands one to
                `client.fit.forks`; re-exported here)
    FitJob      a start (a name, or None = a fresh set), stages, a probe
    FitResult   the steps taken and one probe vector per probe point
    run_fits    THE loop — the Fitter's and the fit client's alike
    FitClient   K forks of one start, for an inline postprocessor

THE ARITHMETIC IS SEAL'S (HF Trainer defaults): a stage is
epochs × ceil(n / batch) optimizer steps, each epoch a fresh seeded shuffle,
the last batch of an epoch short rather than dropped, and the learning rate
falling linearly from the stage's `lr` at its first step to lr/T at its last.
The per-step lr reaches the learner as `lr_scales[lane_route] =
(stage.lr / base_lr) × decay`, so each lane keeps its own schedule inside one
optimizer.

LANES DO NOT TOUCH EACH OTHER'S GRADIENT when the tenant's loss is
`sequence_sft` (FIT_LOSS): that loss weighs each document by
1 / documents_in_update, and the loop stamps that count with the number of
documents the document's OWN lane contributed this step — so a lane's step is
the mean over its own batch, whoever shared the forward. Lanes contributing
different counts are packed apart for exactly that reason. Under plain `sft`
(one token mean per microbatch) a lane's weight would follow its neighbours'
token counts, which is fine for one lane and wrong for many.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from rlstack.client import Stage
from rlstack.data.flatten import Doc, TokenBatch, flatten, pack
from rlstack.data.stores.base import StoreError, check_name
from rlstack.data.trajectory import trajectory_from_row
from rlstack.policy.adapters.dream_bank import memory_route
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.assemble import stamped
from rlstack.runner.interfaces import (
    EntryInstall, Learner, OptimSettings, Parameterization,
)
from rlstack.runner.refs import RefReader
from rlstack.runner.meters import HostJournal
from rlstack.runner.signals import store_work

if TYPE_CHECKING:       # roles/ imports this module; the type alone comes back
    from rlstack.runner.roles.base import StopRequest

FIT_LOSS = "sequence_sft"
"""The loss under which lanes are independent (see the module docstring);
what the fit client installs its fork tenant with. A fit RUN declares its own
(spec/validate.py: FIT_LOSSES)."""

DECAYS = ("linear", "constant")

RowsOf = Callable[[str], Awaitable[Sequence[Mapping[str, Any]]]]
PayloadOf = Callable[[str], Awaitable[bytes | None]]
OnDone = Callable[[int, "FitResult", bytes | None], Awaitable[None]]


class FitPlanError(ValueError):
    """A fit job, or a plan of them, that cannot be run as written."""


@dataclass
class FitTimings:
    """Wall seconds at the existing fit boundaries, outside deterministic results."""
    prepare: float = 0.0
    train: float = 0.0
    probe: float = 0.0


@dataclass
class FitTrace:
    """Bounded-cadence loss samples, flushed at fit completion outside run bytes."""
    every: int = 10
    samples: list[dict] = field(default_factory=list)

    def take(self) -> list[dict]:
        samples, self.samples = self.samples, []
        return samples


# ---------------------------------------------------------------------------
# the records
# ---------------------------------------------------------------------------

# `Stage` is declared on the neutral ground (rlstack/client.py) because it is
# an argument of `client.fit.forks`, which the training world calls; it is
# re-exported here, where the ADR files it and where everything that runs one
# lives.

@dataclass(frozen=True)
class FitJob:
    """One fit: from `start` (a named adapter, or None for a fresh set)
    through `stages` in order, scored on `probe` (a jsonl of trajectory rows)
    before the first stage and after every one, written as `out` (None = the
    caller wants the probes and not the adapter)."""

    out: str | None
    start: str | None
    stages: tuple[Stage, ...]
    probe: str | None = None


def stage_steps(stage: Stage, n_rows: int) -> int:
    """Optimizer steps a stage takes over `n_rows` rows: every epoch is
    ceil(n / batch) steps, the short last batch kept (HF's drop_last=False)."""
    return stage.epochs * math.ceil(n_rows / stage.batch)


@dataclass(frozen=True)
class FitResult:
    """What one fit did: the optimizer steps it took and one NLL vector per
    PROBE POINT — the first before any training, then one per stage end —
    each holding the per-document mean NLL of the probe rows. Empty when the
    job names no probe."""

    out: str | None
    steps: int
    probes: tuple[tuple[float, ...], ...]


def decay_scale(decay: str, step: int, steps: int) -> float:
    """The schedule's multiplier at 0-based `step` of `steps`: linear is HF
    Trainer's default with no warmup — 1 at the first step, 1/steps at the
    last, and it would reach 0 at the step that is never taken."""
    if decay == "constant":
        return 1.0
    return (steps - step) / steps


def epoch_batches(n_rows: int, batch: int, seed_key: str, stage: int,
                  epoch: int) -> list[list[int]]:
    """One epoch's batches as row indices: a shuffle seeded by the job, the
    stage and the epoch — never a global RNG, so a rerun of the job draws the
    same order whatever ran beside it — cut into `batch`-sized pieces, the
    last one short."""
    order = list(range(n_rows))
    random.Random(f"fit:{seed_key}:{stage}:{epoch}").shuffle(order)
    return [order[i:i + batch] for i in range(0, n_rows, batch)]


FORK_SEED_KEY = "fork"


def seed_key_of(job: FitJob) -> str:
    """What a job's shuffles are seeded by: its `out` name, which is what
    makes a resumed or re-ordered plan draw the same orders. A job with no
    name is a FORK of the fit client's, and every fork shares ONE key: forks
    are compared with each other, so two forks handed the same rows draw the
    same orders and train alike — a fork whose dream is empty IS the control
    rather than a control plus ordering noise."""
    return job.out if job.out is not None else FORK_SEED_KEY


# ---------------------------------------------------------------------------
# the plan: a jsonl of job rows
# ---------------------------------------------------------------------------

def job_row(job: FitJob) -> dict[str, Any]:
    """One job as its plan row."""
    return {"out": job.out, "start": job.start, "probe": job.probe,
            "stages": [stage_row(stage) for stage in job.stages]}


def stage_row(stage: Stage) -> dict[str, Any]:
    return {"rows": stage.rows, "epochs": stage.epochs, "batch": stage.batch,
            "lr": stage.lr, "decay": stage.decay}


def job_from_row(row: Mapping[str, Any]) -> FitJob:
    return FitJob(
        out=row.get("out"), start=row.get("start"), probe=row.get("probe"),
        stages=tuple(Stage(rows=s["rows"], epochs=s["epochs"], batch=s["batch"],
                           lr=s["lr"], decay=s.get("decay", "linear"))
                     for s in row["stages"]))


def encode_jobs(jobs: Sequence[FitJob]) -> bytes:
    """A fit plan's bytes: one canonical json line per job, in plan order."""
    return "".join(json.dumps(job_row(job), sort_keys=True, separators=(",", ":")) + "\n"
                   for job in jobs).encode("utf-8")


def decode_jobs(data: bytes) -> tuple[FitJob, ...]:
    """A fit plan back from its bytes, every job checked and the plan checked
    as a whole — the fit run's twin of `data.plan.decode`."""
    jobs = tuple(job_from_row(json.loads(line))
                 for line in data.decode("utf-8").splitlines() if line)
    check_plan(jobs)
    return jobs


def check_job_name(name: str, what: str) -> None:
    """A job's `out` and `start` are adapter NAMES — the store's grammar
    (data/stores/base.py: check_name), refused here as a plan error."""
    try:
        check_name(name)
    except StoreError as refusal:
        raise FitPlanError(f"{what}: {refusal}") from None


def check_job(job: FitJob, where: str = "job") -> None:
    """One job is runnable as written: legal names, at least one stage, every
    stage a cas file of rows with positive epochs, batch and lr and a known
    decay, the probe a cas file or absent."""
    if job.out is not None:
        check_job_name(job.out, f"{where}: out")
    if job.start is not None:
        check_job_name(job.start, f"{where}: start")
    if not job.stages:
        raise FitPlanError(f"{where}: a fit job has at least one stage")
    for number, stage in enumerate(job.stages):
        at = f"{where}: stage {number}"
        if not isinstance(stage.rows, str) or not stage.rows:
            raise FitPlanError(f"{at}: rows names a file of trajectory rows, got {stage.rows!r}")
        for name, value in (("epochs", stage.epochs), ("batch", stage.batch)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise FitPlanError(f"{at}: {name} is an int >= 1, got {value!r}")
        if isinstance(stage.lr, bool) or not isinstance(stage.lr, (int, float)) \
                or not stage.lr > 0:
            raise FitPlanError(f"{at}: lr is a positive number, got {stage.lr!r}")
        if stage.decay not in DECAYS:
            raise FitPlanError(f"{at}: decay is one of {DECAYS}, got {stage.decay!r}")


def check_plan(jobs: Sequence[FitJob]) -> None:
    """A FIT PLAN is a list of jobs a Fitter can finish: every job legal and
    NAMED (a fit run exists to write names), every name written once, every
    file a cas uri, and a `start` that this same plan produces produced by an
    EARLIER job — lanes take jobs in plan order, so a later producer is a
    wait nothing would ever end."""
    if not jobs:
        raise FitPlanError("a fit plan lists at least one job")
    position: dict[str, int] = {}
    for index, job in enumerate(jobs):
        where = f"fit job {index}"
        check_job(job, where)
        if job.out is None:
            raise FitPlanError(f"{where}: a fit run's job names its out")
        if job.out in position:
            raise FitPlanError(
                f"{where}: out {job.out!r} is already written by job "
                f"{position[job.out]} — a name is written once")
        position[job.out] = index
        for uri in [stage.rows for stage in job.stages] + ([job.probe] if job.probe else []):
            try:
                rows_slice(uri)
            except FitPlanError as error:
                raise FitPlanError(f"{where}: {error}") from None
    for index, job in enumerate(jobs):
        if job.start is not None and position.get(job.start, -1) >= index:
            raise FitPlanError(
                f"fit job {index} starts from {job.start!r}, which job "
                f"{position[job.start]} of this plan writes — a start is "
                f"produced by an earlier job, or by another run")


def rows_slice(uri: str) -> tuple[str, int | None, int | None]:
    """A ROWS URI: `cas://<sha>` is a whole rows file, `cas://<sha>#<a>:<b>`
    rows [a, b) of it. ONE FILE, MANY STAGES (2026-09-19): a bank's first
    shard wrote a thousand small files, the scratch API takes seconds per
    object and answers 429 to a dozen writers, so a shard packs its rows into
    one file and each stage names its span. Returns (file uri, a, b)."""
    file, _, span = uri.partition("#")
    if not file.startswith("cas://") or not file[len("cas://"):]:
        raise FitPlanError(f"fit rows live in the cas ('cas://<sha>[#a:b]'), got {uri!r}")
    if not span:
        return file, None, None
    first, colon, last = span.partition(":")
    if not (colon and first.isdigit() and last.isdigit() and int(first) <= int(last)):
        raise FitPlanError(f"{uri!r}: a span is '#<a>:<b>' with a <= b")
    return file, int(first), int(last)


async def sealed_rows(refs: RefReader, uri: str) -> list[dict[str, Any]]:
    """A rows file or a span of one, read off the loop through the run's ref
    reader (which caches the FILE, so every stage of a packed shard costs one
    read): what the Fitter's `rows_of` and the fit client's `rows` both are. A
    file of fit rows is content-addressed, so it is never "not yet"."""
    file, first, last = rows_slice(uri)
    rows = await store_work(refs.rows, file)
    if rows is None:
        raise FileNotFoundError(f"fit rows {file!r} are not in this store")
    if first is None:
        return rows
    if last > len(rows):
        raise FitPlanError(f"{uri!r} names rows up to {last} of a file that holds {len(rows)}")
    return rows[first:last]


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

@dataclass
class _Lane:
    """One memory set's seat in the loop: idle, WAITING on its job's start
    name, or FITTING the job stage by stage."""

    route: str
    index: int = -1
    job: FitJob | None = None
    waiting: asyncio.Task | None = None          # the start payload, still awaited
    stage_docs: list[list[Doc]] = field(default_factory=list)
    probe_docs: list[Doc] = field(default_factory=list)
    stage: int = 0
    step: int = 0                                # within the stage
    steps: int = 0                               # across the job
    batches: deque = field(default_factory=deque)  # the current epoch's remainder
    epoch: int = 0
    probes: list[tuple[float, ...]] = field(default_factory=list)

    @property
    def fitting(self) -> bool:
        return self.job is not None and self.waiting is None

    def stage_steps(self) -> int:
        return stage_steps(self.job.stages[self.stage], len(self.stage_docs[self.stage]))

    def next_docs(self) -> list[Doc]:
        """This step's documents: the next batch of the current epoch, a new
        epoch's shuffle drawn when the last one is spent."""
        if not self.batches:
            stage = self.job.stages[self.stage]
            self.batches = deque(epoch_batches(
                len(self.stage_docs[self.stage]), stage.batch,
                seed_key_of(self.job), self.stage, self.epoch))
            self.epoch += 1
        docs = self.stage_docs[self.stage]
        return [docs[i] for i in self.batches.popleft()]

    def scale(self, base_lr: float) -> float:
        stage = self.job.stages[self.stage]
        return (stage.lr / base_lr) * decay_scale(stage.decay, self.step, self.stage_steps())

    def clear(self) -> None:
        self.index, self.job, self.waiting = -1, None, None
        self.stage_docs, self.probe_docs, self.probes = [], [], []
        self.stage = self.step = self.steps = self.epoch = 0
        self.batches = deque()


async def run_fits(learner: Learner, tenant: str, entry: str,
                   jobs: Sequence[FitJob], rows_of: RowsOf, payload_of: PayloadOf,
                   lanes: int, *, arbiter: Arbiter, microbatch_tokens: int,
                   base_lr: float, on_done: OnDone | None = None,
                   stop: StopRequest | None = None,
                   timings: FitTimings | None = None,
                   trace: FitTrace | None = None,
                   poll_s: float = 0.5) -> list[FitResult]:
    """Run `jobs` through `lanes` memory sets of `tenant`'s `entry`; return
    the finished jobs' results in JOB order.

    `rows_of(uri)` yields a file's trajectory rows and `payload_of(name)` a
    start's named payload — awaited in the background, because a name may
    still be being fitted (by another run, or by an earlier job of this
    call), and a lane waiting for one must not hold the others' steps.
    `base_lr` is the lr the tenant's optimizer was installed with: what
    `lr_scales` multiplies. `on_done(index, result, payload)` is awaited as
    each job ends — the Fitter writes the name there, which is also what lets
    a later job of the same call start from it; `payload` is None for a job
    with no `out`, whose set is never emitted.

    A requested `stop` starts no further job; the lanes already fitting run
    to their ends, so every job either finished or never began — and the
    result simply omits the ones that never did.

    Every learner verb is asked from a thread under the arbiter's admission,
    exactly as the Trainer asks them.
    """
    if lanes < 1:
        raise ValueError(f"run_fits needs at least one lane, got {lanes}")
    if not base_lr > 0:
        raise ValueError(
            f"run_fits scales each lane's lr against the optimizer's base lr, "
            f"which must be positive; got {base_lr!r}")
    for index, job in enumerate(jobs):
        check_job(job, f"fit job {index}")
    if trace is not None and trace.every < 1:
        raise ValueError("fit trace cadence must be positive")
    doc_ids: dict[int, str] = {}
    optimizer_steps = 0

    async def ask(verb, *args, **kwargs):
        async with arbiter.admit(learner):
            return await asyncio.to_thread(verb, *args, **kwargs)

    async def docs_of(uri: str, route: str) -> list[Doc]:
        """A file's rows as packable documents, stamped with the lane's
        route the way `realize` stamps a leaf's role."""
        rows = await rows_of(uri)
        trajectories = [trajectory_from_row(stamped(dict(row), route)) for row in rows]

        def tokenize(text: str) -> tuple[int, ...]:
            return learner.tokenize(tenant, text)

        flats = await ask(lambda: [flatten(t, tokenize) for t in trajectories])
        if trace is not None:
            doc_ids.update({id(flat): row["task"]["id"] for flat, row in zip(flats, rows)})
        return [(flat, {}) for flat in flats]

    async def probe(group: Sequence[_Lane]) -> None:
        """One probe point for every lane of `group` that has a probe: their
        probe rows share the forwards, each under its own lane's route."""
        probed = [lane for lane in group if lane.probe_docs]
        if not probed:
            return
        started = time.monotonic()
        docs = [doc for lane in probed for doc in lane.probe_docs]
        nll: list[float] = []
        for batch in pack(docs, microbatch_tokens):
            nll.extend(await ask(learner.forward, tenant, batch))
        if len(nll) != len(docs):
            raise RuntimeError(
                f"learner.forward answered {len(nll)} NLLs for {len(docs)} probe documents")
        cursor = 0
        for lane in probed:
            width = len(lane.probe_docs)
            lane.probes.append(tuple(float(v) for v in nll[cursor:cursor + width]))
            cursor += width
        if timings is not None:
            timings.probe += time.monotonic() - started

    async def begin(lane: _Lane, payload: bytes | None) -> None:
        """The job's first breath on its lane: rows read and flattened, the
        set loaded (None = fresh; either way its moments reset). The probe
        before any training follows, shared with whoever else began."""
        job = lane.job
        started = time.monotonic()
        lane.waiting = None
        lane.stage_docs = [await docs_of(stage.rows, lane.route) for stage in job.stages]
        for number, docs in enumerate(lane.stage_docs):
            if not docs:
                raise FitPlanError(
                    f"fit job {lane.index} ({job.out!r}): stage {number}'s rows "
                    f"{job.stages[number].rows!r} hold no trajectory")
        lane.probe_docs = await docs_of(job.probe, lane.route) if job.probe else []
        await ask(learner.load_set, tenant, entry, lane.route, payload)
        began.append(lane)
        if timings is not None:
            timings.prepare += time.monotonic() - started

    async def step(active: Sequence[_Lane]) -> None:
        """ONE OPTIMIZER STEP: every fitting lane's next batch through
        forward_backward, then one optim_step carrying each lane's own lr.
        Lanes contributing the same number of documents share microbatches,
        stamped with that number (see the module docstring)."""
        nonlocal optimizer_steps
        started = time.monotonic()
        optimizer_steps += 1
        sample = trace is not None and (optimizer_steps % trace.every == 0 or any(
            lane.step == 0 or lane.step + 1 == lane.stage_steps() for lane in active))
        by_count: dict[int, list[Doc]] = {}
        scales: dict[str, float] = {}
        for lane in active:
            docs = lane.next_docs()
            by_count.setdefault(len(docs), []).extend(docs)
            scales[lane.route] = lane.scale(base_lr)
        batches: list[TokenBatch] = [
            replace(batch, documents_in_update=count)
            for count in sorted(by_count)
            for batch in pack(by_count[count], microbatch_tokens)]
        ids = ([doc_ids[id(flat)] for count in sorted(by_count) for flat, _ in by_count[count]]
               if sample else [])
        sampled, cursor = [], 0
        async with arbiter.admit(learner):
            for batch in batches:
                stats = await asyncio.to_thread(
                    learner.forward_backward, tenant,
                    replace(batch, microbatches_in_update=len(batches)))
                if sample:
                    width = len(batch.doc_starts)
                    sampled.append({"rows": ids[cursor:cursor + width],
                                    "routes": [turns[0]["route"] for turns in batch.doc_turn_extras],
                                    "loss": stats.loss, "components": dict(stats.components),
                                    "documents_in_update": batch.documents_in_update,
                                    "scored_tokens": sum(batch.loss_mask)})
                    cursor += width
            await asyncio.to_thread(learner.optim_step, tenant, lr_scales=scales)
        if sample:
            trace.samples.append({"t": time.time(), "optimizer_step": optimizer_steps,
                "seconds": time.monotonic() - started,
                "lanes": [{"route": lane.route, "job": lane.job.out, "index": lane.index,
                           "stage": lane.stage, "step": lane.steps + 1, "stage_step": lane.step + 1,
                           "lr": base_lr * scales[lane.route]} for lane in active],
                "microbatches": sampled})
        for lane in active:
            lane.step += 1
            lane.steps += 1
        if timings is not None:
            timings.train += time.monotonic() - started

    async def finish(lane: _Lane) -> None:
        job, index = lane.job, lane.index
        payload = (await ask(learner.emit_set, tenant, entry, lane.route)
                   if job.out is not None else None)
        result = FitResult(out=job.out, steps=lane.steps, probes=tuple(lane.probes))
        results[index] = result
        lane.clear()
        if on_done is not None:
            await on_done(index, result, payload)

    queue = deque(enumerate(jobs))
    seats = [_Lane(memory_route(i)) for i in range(lanes)]
    results: dict[int, FitResult] = {}
    began: list[_Lane] = []          # lanes loaded since the last probe point
    try:
        while True:
            stopping = stop is not None and stop.requested
            for lane in seats:
                if stopping and lane.waiting is not None:
                    lane.waiting.cancel()       # a job that never began
                    lane.clear()
                if lane.job is None and queue and not stopping:
                    lane.index, lane.job = queue.popleft()
                    if lane.job.start is None:
                        await begin(lane, None)
                    else:
                        lane.waiting = asyncio.create_task(payload_of(lane.job.start))
                if lane.waiting is not None and lane.waiting.done():
                    payload = lane.waiting.result()
                    if payload is None:
                        # the wait ended without the name, which only a stop
                        # asked meanwhile may do: a job with a start NEVER
                        # begins fresh, so it never begins
                        if stop is None or not stop.requested:
                            raise RuntimeError(
                                f"fit job {lane.index} starts from "
                                f"{lane.job.start!r} and was handed no payload")
                        lane.clear()
                    else:
                        await begin(lane, payload)
            await probe(began)                  # the point BEFORE any training
            began.clear()
            active = [lane for lane in seats if lane.fitting]
            if not active:
                waiting = [lane.waiting for lane in seats if lane.waiting is not None]
                if not waiting:
                    break
                # nothing to step: sleep until a start arrives (or a beat
                # passes, so a stop asked meanwhile is read)
                await asyncio.wait(waiting, timeout=poll_s,
                                   return_when=asyncio.FIRST_COMPLETED)
                continue
            await step(active)
            ended = [lane for lane in active if lane.step == lane.stage_steps()]
            await probe(ended)
            for lane in ended:
                lane.stage, lane.step, lane.epoch = lane.stage + 1, 0, 0
                lane.batches = deque()
                if lane.stage == len(lane.job.stages):
                    await finish(lane)
    finally:
        for lane in seats:
            if lane.waiting is not None:
                lane.waiting.cancel()
    return [results[index] for index in sorted(results)]


# ---------------------------------------------------------------------------
# the fit client: K forks of one start, for an inline postprocessor
# ---------------------------------------------------------------------------

FORK_ENTRY = "forks"
FORK_BASE_LR = 1e-4
"""What the fork tenant's optimizer is installed at. Only a reference point:
every step's lr is `stage.lr`, reached through lr_scales."""


def fork_parameterization(base: str, sites: tuple, r: int, forks: int,
                          seed: int) -> Parameterization:
    """THE FORK TENANT: the run's base, one `dream_bank` entry of `forks`
    memory sets at the rank and sites of the run's own, trained by FIT_LOSS
    under HF-default AdamW with no weight decay."""
    return Parameterization(
        base=base, loss=FIT_LOSS,
        entries=(EntryInstall(
            name=FORK_ENTRY, adapter_type="dream_bank", trainable=True, sites=sites,
            init={"r": r, "memories": forks, "lam": 0.0, "mode": "stream",
                  "seed": seed}),),
        optim=OptimSettings(name="adamw", lr=FORK_BASE_LR, betas=(0.9, 0.999),
                            weight_decay=0.0, overrides={}))


class FitClient:
    """Fits for an INLINE postprocessor, on the run's own learner (ADR 0019).

    `forks` trains K copies of one start on K row sets and answers each
    fork's final probe vector; `tokenize` and `rows` are what a processor
    needs to BUILD those row sets — the base's tokenizer (asked of the run's
    own tenant, so it works before any fork exists) and the run's cas. The
    copies are K lanes of a SECOND tenant —
    `<run tenant>:forks` — installed the first time it is needed (and again,
    wider, if a later call forks more ways): throwaway state that never
    reaches the store and dies with the learner's tenant table. Calls are
    serialized by one lock, because every group's processor runs concurrently
    and they share that one tenant.
    """

    def __init__(self, learner: Learner, arbiter: Arbiter, *, tenant: str,
                 refs: RefReader, base: str, sites: tuple, r: int, seed: int,
                 microbatch_tokens: int, journal: HostJournal | None = None) -> None:
        self.learner = learner
        self.arbiter = arbiter
        self.run_tenant = tenant
        self.tenant = f"{tenant}:forks"
        self.refs = refs
        self.base, self.sites, self.r, self.seed = base, sites, r, seed
        self.microbatch_tokens = microbatch_tokens
        self.installed = 0                      # lanes the fork tenant holds
        self.lock = asyncio.Lock()
        self.journal = journal
        self.update: int | None = None

    async def forks(self, start_payload: bytes | None,
                    fork_rows: Sequence[Sequence[Mapping[str, Any]]],
                    probe_rows: Sequence[Mapping[str, Any]],
                    stage: Stage) -> list[tuple[float, ...]]:
        """Fork `start_payload` (None = a fresh set) once per entry of
        `fork_rows`, train fork k on `fork_rows[k]` under `stage`'s epochs,
        batch, lr and decay (its `rows` field is ignored — the rows are
        handed in), and return each fork's probe vector AFTER training: the
        per-document mean NLL of `probe_rows`."""
        if not fork_rows:
            return []
        files = {f"fork://{k}": rows for k, rows in enumerate(fork_rows)}
        files["fork://probe"] = probe_rows
        jobs = [FitJob(out=None, start="fork/start" if start_payload is not None else None,
                       stages=(replace(stage, rows=f"fork://{k}"),),
                       probe="fork://probe")
                for k in range(len(fork_rows))]

        async def rows_of(uri: str):
            return files[uri]

        async def payload_of(name: str) -> bytes:
            return start_payload

        requested = time.monotonic()
        timings = FitTimings()
        trace = FitTrace() if self.journal is not None else None
        async with self.lock:
            admitted = time.monotonic()
            await self.ensure_lanes(len(jobs))
            results = await run_fits(
                self.learner, self.tenant, FORK_ENTRY, jobs, rows_of, payload_of,
                lanes=len(jobs), arbiter=self.arbiter,
                microbatch_tokens=self.microbatch_tokens, base_lr=FORK_BASE_LR, timings=timings, trace=trace)
        if self.journal is not None:
            # Full probe vectors remain recoverable without fitting again.
            # Stable training/probe row IDs join this event to the sealed wave
            # and task metadata (memory, use, topic and low-water reference).
            await store_work(self.journal.append, {
                "event": "forks", "t": time.time(), "run_id": self.run_tenant,
                "update": self.update, "cadence": "every fork group",
                "seconds": time.monotonic() - requested,
                "phases": {"queue": admitted - requested, "prepare": timings.prepare,
                           "train": timings.train, "probe": timings.probe},
                "stage": stage_row(stage),
                "fit_step_cadence": "stage first/last and every 10 shared optimizer steps",
                "fit_steps": trace.samples,
                "probe_rows": [row["task"]["id"] for row in probe_rows],
                "probe_tokens": [sum(len(turn["token_ids"]) for turn in row["turns"])
                                 for row in probe_rows],
                "lanes": [{"rows": [row["task"]["id"] for row in rows],
                           "steps": result.steps, "probes": result.probes}
                          for rows, result in zip(fork_rows, results)],
                "reduction": "probe: mean scored-token NLL per document; "
                             "training: mean document NLL within each lane"})
        return [result.probes[-1] for result in results]

    async def tokenize(self, text: str) -> tuple[int, ...]:
        """The learner's tokenizer, under the RUN's tenant — a CPU verb, asked
        from a thread under admission like every other."""
        async with self.arbiter.admit(self.learner):
            return tuple(await asyncio.to_thread(
                self.learner.tokenize, self.run_tenant, text))

    async def rows(self, uri: str) -> list[dict[str, Any]]:
        return await sealed_rows(self.refs, uri)

    async def close(self) -> None:
        """The run is over: the fork tenant leaves the learner with it, so a
        standing learner does not collect the forks of every run it served.
        Idempotent, as `uninstall` is."""
        if self.installed:
            async with self.arbiter.admit(self.learner):
                await asyncio.to_thread(self.learner.uninstall, self.tenant)
            self.installed = 0

    async def ensure_lanes(self, forks: int) -> None:
        """The fork tenant holds at least `forks` lanes — installed on first
        use, rebuilt wider when a call needs more (install resets a tenant,
        and a fork tenant has no state worth keeping between calls)."""
        if forks <= self.installed:
            return
        parameterization = fork_parameterization(
            self.base, self.sites, self.r, forks, self.seed)
        async with self.arbiter.admit(self.learner):
            await asyncio.to_thread(self.learner.install, self.tenant, parameterization)
        self.installed = forks
