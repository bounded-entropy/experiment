"""The Fitter: a plan of fit jobs becomes NAMED ADAPTERS (ADR 0019).

A fit run is a learner and this runner and nothing else. Its plan is a list of
jobs (runner/fit.py); its bank entry is a `dream_bank` whose memory sets are
the LANES the jobs fit in; its algo declares the loss, the optimizer and the
microbatch budget, and no pipeline. One job:

    PROMISE every `out` at birth (a consumer in another run waits on a
    promise, and refuses one whose writer ended without keeping it) →
    await the job's `start` name, if it has one → fit it on a lane
    (`run_fits`) → WRITE THE NAME (write-once: payload + meta) →
    COMMIT (one ledger line per job, in PLAN order) →
    CHECKPOINT (the checkpoint line, at the declared cadence) → notify

THE NAME IS THE DURABLE POINT. It is written once, at the job's end, so a
Fitter killed mid-job leaves no name and a live promise, and RESUME IS
"skip the jobs whose name is present": nothing else of a fit run needs
restoring, because a job starts from `load_set` and carries nothing over.

THE LEDGER IS IN PLAN ORDER — line u is job u — whatever order the lanes
finished in: a job that ends early has its name written at once and its line
held until every earlier job has one. A line says only what its name's meta
says, so one that the attach rewind cut, or that a crash fell between the
name and, is rebuilt from the meta to the same bytes. (What a FRESH job's
bytes are depends on the lane it sat in — a learner seeds each set off its
route — so a resumed run writes the same NAMES as an uninterrupted one, and
the same bytes only where its jobs found the same lanes.)

`Checkpointing.every` COUNTS JOBS, and there is nothing else to checkpoint:
the policy bank of a fit run never moves off version 0 (its sets are lanes,
not a policy), so a checkpoint here is the one line that lets `run_done` call
the last job sealed — no blob is written and none is swept.

A deliberate stop starts no further job, lets the lanes already fitting
finish and write their names, seals what was committed, and drains — the
Trainer's rule, at the Fitter's boundary.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

from rlstack.data.stores.base import PRESENT, RunHandle
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.fit import FitJob, FitResult, FitTrace, run_fits, sealed_rows, stage_row
from rlstack.runner.meters import HostJournal
from rlstack.runner.interfaces import Learner
from rlstack.runner.names import names_ready, names_subdir
from rlstack.runner.refs import RefReader
from rlstack.runner.roles.base import Runner, StopRequest
from rlstack.runner.signals import RunSignals, store_work
from rlstack.spec.specs import ExperimentSpec
from rlstack.spec.validate import FIT_ADAPTER_TYPE


def fit_entry(spec: ExperimentSpec) -> str:
    """The bank entry whose memory sets are a fit run's lanes: its one
    `dream_bank` entry (the gate — check_a_fit_run_is_a_learner_and_lanes —
    has already refused a fit run without exactly one)."""
    return next(name for name, adapter in sorted(spec.policy.bank.items())
                if adapter.adapter_type == FIT_ADAPTER_TYPE)


class Fitter(Runner):
    def __init__(self, signals: RunSignals, arbiter: Arbiter, run: RunHandle, *,
                 spec: ExperimentSpec, jobs: Sequence[FitJob],
                 refs: RefReader, learner: Learner,
                 tenant: str, initial_version: Mapping[str, int],
                 checkpointing: Checkpointing,
                 journal: HostJournal | None = None,
                 stop: StopRequest | None = None) -> None:
        super().__init__(signals, arbiter, run)
        self.spec = spec
        self.jobs = tuple(jobs)
        self.store = run.store
        # names live under the EXPERIMENT's subdir, beside the runs filed
        # there (a fit run filed at the store's root is refused by name), and
        # a promise names its writer by the QUALIFIED reference
        self.subdir = names_subdir(run)
        self.run_ref = f"{self.subdir}/{run.run_id}"
        self.refs = refs
        self.learner = learner
        self.tenant = tenant
        self.version = dict(initial_version)
        self.checkpointing = checkpointing
        self.stop = stop
        self.journal = journal
        self.entry = fit_entry(spec)
        self.lanes = int(spec.policy.bank[self.entry].init["memories"])
        # what each finished job's ledger line says, by `out` — filled from
        # the names already present at birth, then as jobs end
        self.facts: dict[str, dict[str, Any]] = {}
        self.last_update = 0
        self.checkpointed = 0

    # ---- the runner ---------------------------------------------------------

    async def run_forever(self) -> None:
        self.last_update = await store_work(self.committed)
        tail = await store_work(self.run.checkpoint_tail)
        self.checkpointed = int(tail["update"]) if tail else 0

        states = {job.out: await store_work(self.store.named_state, self.subdir, job.out)
                  for job in self.jobs}
        owed = [job.out for job in self.jobs if states[job.out] != PRESENT]
        if owed:
            await store_work(self.store.promise_named, self.subdir, owed, self.run_ref)
        for job in self.jobs:
            if states[job.out] == PRESENT:          # THAT IS RESUME
                meta = await store_work(self.store.named_meta, self.subdir, job.out)
                self.facts[job.out] = facts_of_meta(meta or {})
        await self.commit_in_plan_order()

        remaining = [job for job in self.jobs if states[job.out] != PRESENT]
        trace = FitTrace() if self.journal is not None else None

        async def on_done(position: int, result: FitResult, payload: bytes | None) -> None:
            await self.write_name(remaining[position], result, payload)
            await self.commit_in_plan_order()
            await self.signals.notify()
            if self.journal is not None and trace.samples:
                await store_work(self.journal.append, {
                    "event": "fit_steps", "t": time.time(), "run_id": self.tenant, "run_ref": self.run_ref,
                    "completed_job": remaining[position].out, "committed_update": self.last_update,
                    "cadence": "stage first/last and every 10 shared optimizer steps",
                    "loss": self.spec.algo.loss,
                    "reduction": "sequence_sft: sum document token-mean NLL / lane batch size; "
                                 "sft: token mean per microbatch; sum microbatches; no KL in either fit loss. "
                                 "Join sampled job names to committed ledger.",
                    "samples": trace.take()})

        await run_fits(
            self.learner, self.tenant, self.entry, remaining,
            self.rows_of, self.payload_of, self.lanes, arbiter=self.arbiter,
            microbatch_tokens=self.spec.algo.schedule.microbatch_tokens,
            base_lr=self.spec.algo.optim.lr, on_done=on_done, stop=self.stop, trace=trace)
        if self.stop is not None and self.stop.requested:
            await self.drain()

    # ---- what the loop reads ------------------------------------------------

    async def rows_of(self, uri: str) -> list[dict]:
        return await sealed_rows(self.refs, uri)

    async def payload_of(self, name: str) -> bytes:
        """A job's START, awaited until it exists: `names_ready` returns when
        the name is present, keeps waiting while it is promised or unknown,
        and raises when its promised writer ended without writing it — a fit
        never starts from init because its parent went missing. A stop asked
        meanwhile ends the wait; the loop then drops the job unstarted, so
        the missing payload is never loaded."""
        await names_ready(self.store, self.subdir, [name], self.signals,
                          stopped=lambda: self.stop is not None and self.stop.requested)
        return await store_work(self.store.read_named, self.subdir, name)

    # ---- the three steps, named ---------------------------------------------

    async def write_name(self, job: FitJob, result: FitResult, payload: bytes) -> None:
        """THE DURABLE POINT: the payload under its name, once, with the meta
        that says what it is and how it was made — enough to rebuild its
        ledger line, and enough for a reader to refuse a mismatched base."""
        adapter = self.spec.policy.bank[self.entry]
        meta = {
            "base": self.spec.policy.base, "r": int(adapter.init["r"]),
            "site": adapter.site, "loss": self.spec.algo.loss,
            "parent": job.start, "stages": [stage_row(stage) for stage in job.stages],
            "probe": job.probe, "probes": [list(vector) for vector in result.probes],
            "steps": result.steps, "run_ref": self.run_ref,
        }
        await store_work(self.store.write_named, self.subdir, job.out, payload, meta)
        self.facts[job.out] = facts_of_meta(meta)

    async def commit_in_plan_order(self) -> None:
        """THE COMMIT: a ledger line for every job whose name exists and whose
        every predecessor has a line — so line u is job u — and a checkpoint
        line wherever the cadence, or the extent, asks for one."""
        while self.last_update < len(self.jobs):
            job = self.jobs[self.last_update]
            if job.out not in self.facts:
                return
            update = self.last_update + 1
            await store_work(self.run.append_ledger,
                             ledger_line(update, job, self.facts[job.out], self.version))
            self.last_update = update
            if self.checkpointing.due(update, len(self.jobs)):
                await self.checkpoint(update)

    async def checkpoint(self, update: int) -> None:
        """One checkpoint line and no blob: the names are already durable,
        and the bank never moved (see the module docstring)."""
        if update <= self.checkpointed:
            return
        await store_work(self.run.append_checkpoint, update, dict(self.version))
        self.checkpointed = update

    async def drain(self) -> None:
        """A DELIBERATE STOP LOSES NOTHING: seal the last committed job, and
        let the run end."""
        await self.checkpoint(self.last_update)
        self.stop.drained = True


def facts_of_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """What a ledger line says about a finished job, read off its name's
    meta — the one source, so a line written as the job ended and a line
    rebuilt on resume are the same bytes."""
    return {"start": meta.get("parent"), "steps": int(meta.get("steps", 0)),
            "probes": [list(vector) for vector in meta.get("probes", [])]}


def ledger_line(update: int, job: FitJob, facts: Mapping[str, Any],
                versions: Mapping[str, int]) -> dict[str, Any]:
    """ONE JOB'S LEDGER LINE. `train` holds the scalars an observer plots —
    the steps taken and the probe's mean NLL before training and after the
    last stage; `probes` is every probe point in full."""
    probes = facts["probes"]
    train: dict[str, float] = {"steps": facts["steps"]}
    if probes and probes[0]:
        train["probe_first"] = math.fsum(probes[0]) / len(probes[0])
        train["probe_last"] = math.fsum(probes[-1]) / len(probes[-1])
    return {"update": update, "out": job.out, "start": facts["start"],
            "versions": dict(versions), "train": train, "probes": probes}
