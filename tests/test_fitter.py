"""A fit run end to end on fakes (ADR 0019): the plan kind, needs_of, the gate,
and the Fitter — promises at birth, names written once, one ledger line per
job in plan order, resume by skipping what is present, the wait on a start
and the refusal of an orphaned one, a drained stop."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace

from rlstack import run_experiment
from rlstack.data.stores.base import PRESENT, PROMISED, run_done, run_progress
from rlstack.data.stores.local import LocalStore
from rlstack.policy.siteschema import fake_qwen_schema
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.fit import FitJob, FitPlanError, Stage, encode_jobs
from rlstack.runner.loop import (
    data_fingerprint, experiment_identity, needs_of, run_experiment_async,
)
from rlstack.runner.names import NamedAdapterOrphaned
from rlstack.runner.remote import spec_from_json
from rlstack.runner.roles import Fitter, Trainer
from rlstack.runner.roles.base import StopRequest
from rlstack.spec.canonical import canonical_json
from rlstack.spec.specs import (
    AlgoSpec, ExperimentSpec, HostSpec, OptimSpec, Plans, PolicySpec, Schedule,
    Seeds, Topology, dream_bank, learner, lora, pool,
)
from rlstack.spec.validate import validate
from test_fit import PROBE, SEVEN, LaneLearner

BASE = "Qwen/Qwen2.5-0.5B"
SCHEMA = fake_qwen_schema(2, base=BASE)
SUBDIR = "dreams/g1"


def jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()


def fit_spec(plan: str, *, lanes: int = 2, **algo) -> ExperimentSpec:
    """A FIT RUN'S SPEC, whole: one dream_bank entry whose memories are the
    lanes, a learner and no pool, the fit plan and no other, and an algo that
    is the fit's loss, optimizer and microbatch budget."""
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank={"pi": dream_bank(
            "layers.0-1.self_attn.q_proj", 4, lanes)}),
        gen=None,
        plans=Plans(fit=plan),
        algo=AlgoSpec(**{"loss": "sequence_sft", "post": (),
                         "optim": OptimSpec("adamw", lr=1e-4, betas=(0.9, 0.999)),
                         "schedule": Schedule(microbatch_tokens=4096, max_policy_lag=0),
                         **algo}),
        topology=Topology(hosts=(HostSpec((learner(),)),)),
        seeds=Seeds(master=0))


class FitRunCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = LocalStore(self.tmp.name)
        self.rows = self.store.cas_put(jsonl(SEVEN))
        self.probe = self.store.cas_put(jsonl(PROBE))

    def job(self, out: str, start: str | None = None, epochs: int = 1) -> FitJob:
        return FitJob(out, start, (Stage(self.rows, epochs, 7, 1e-3),), self.probe)

    def spec_of(self, jobs: list[FitJob], store=None, **kwargs) -> ExperimentSpec:
        return fit_spec((store or self.store).cas_put(encode_jobs(jobs)), **kwargs)

    def run_fit(self, spec, learner=None, store=None, every: int = 2, resume: bool = False):
        return run_experiment(spec, SCHEMA, store or self.store, {}, learner or LaneLearner(),
                              subdir=SUBDIR, resume=resume,
                              checkpointing=Checkpointing(every=every))


class PlanKindTest(FitRunCase):
    def test_a_spec_without_a_fit_plan_is_the_identity_it_always_was(self) -> None:
        plans = Plans(train="cas://t", rollout="cas://r")
        self.assertEqual(canonical_json(plans),
                         '{"__type__":"Plans","rollout":"cas://r","train":"cas://t"}')
        self.assertEqual(plans.extent, "train")
        self.assertEqual(Plans(rollout="cas://r").extent, "rollout")

    def test_a_fit_plan_is_in_the_identity_and_is_the_extent(self) -> None:
        plans = Plans(fit="cas://f")
        self.assertIn('"fit":"cas://f"', canonical_json(plans))
        self.assertEqual(plans.extent, "fit")
        one, two = fit_spec("cas://" + "a" * 64), fit_spec("cas://" + "b" * 64)
        self.assertNotEqual(experiment_identity(one, SCHEMA), experiment_identity(two, SCHEMA))
        self.assertTrue(data_fingerprint(one).endswith("|fit:cas://" + "a" * 64))
        self.assertEqual(data_fingerprint(replace(one, plans=Plans(train="cas://t"))), "cas://t|-")

    def test_both_kinds_of_spec_cross_the_wire_whole(self) -> None:
        for spec in (fit_spec("cas://" + "a" * 64),
                     replace(fit_spec("cas://x"), plans=Plans(train="cas://t"))):
            self.assertEqual(spec_from_json(json.loads(canonical_json(spec))), spec)

    def test_needs_of_a_fit_run_is_a_fitter_and_no_trainer(self) -> None:
        (need,) = needs_of(fit_spec("cas://" + "a" * 64))
        self.assertIs(need.runner, Fitter)
        self.assertEqual((need.plan, need.learner, need.pools), ("fit", True, ()))
        trains = replace(fit_spec("cas://x"), plans=Plans(train="cas://t"))
        self.assertEqual([n.runner for n in needs_of(trains)], [Trainer])


class GateTest(FitRunCase):
    def codes(self, spec) -> set[str]:
        return {issue.code for issue in validate(spec, SCHEMA)}

    def paths(self, spec) -> set[str]:
        return {issue.path for issue in validate(spec, SCHEMA) if issue.code == "fit-run-shape"}

    def test_a_fit_run_as_written_passes(self) -> None:
        self.assertEqual(validate(fit_spec("cas://" + "a" * 64), SCHEMA), [])

    def test_what_a_fit_run_may_not_be(self) -> None:
        good = fit_spec("cas://" + "a" * 64)
        self.assertEqual(self.paths(replace(good, plans=Plans(fit="cas://f", train="cas://t"))),
                         {"plans.train"})
        self.assertEqual(self.paths(replace(good, plans=Plans(fit="jobs.jsonl"))), {"plans.fit"})
        self.assertEqual(self.paths(replace(good, algo=None)), {"algo"})
        self.assertEqual(self.paths(fit_spec("cas://f", loss="grpo")), {"algo.loss"})
        self.assertEqual(self.paths(fit_spec("cas://f", post=("grpo_advantage",))), {"algo.post"})
        self.assertEqual(self.paths(fit_spec("cas://f", optim=OptimSpec("adamw", lr=0.0))),
                         {"algo.optim.lr"})
        self.assertEqual(self.paths(fit_spec("cas://f", optim=OptimSpec(
            "adamw", lr=1e-4, overrides={"pi.memory": {"lr": 1e-3}}))),
            {"algo.optim.overrides.pi.memory"})
        self.assertEqual(self.paths(fit_spec("cas://f", lanes=0)), {"policy.bank.pi"})
        no_bank = replace(good, policy=PolicySpec(base=BASE, bank={"pi": lora("layers.0-1.self_attn.q_proj", 4)}))
        self.assertEqual(self.paths(no_bank), {"policy.bank"})
        no_learner = replace(good, topology=Topology(hosts=(HostSpec((pool("main"),)),)))
        self.assertIn("topology", self.paths(no_learner))

    def test_sft_is_a_legal_fit_loss_too(self) -> None:
        self.assertEqual(validate(fit_spec("cas://" + "a" * 64, loss="sft"), SCHEMA), [])

    def test_an_illegal_plan_file_is_refused_as_it_is_decoded(self) -> None:
        bad = self.store.cas_put(encode_jobs([self.job("m/0"), self.job("m/0")]))
        with self.assertRaises(FitPlanError):
            self.run_fit(fit_spec(bad))


class FitterTest(FitRunCase):
    def test_loss_trace_has_job_identity_and_bounded_cadence_outside_run_bytes(self):
        from rlstack.runner.meters import HostJournal
        jobs = [self.job(f"trace/m-{i}", epochs=12) for i in range(2)]
        report = asyncio.run(run_experiment_async(
            self.spec_of(jobs), SCHEMA, self.store, {}, LaneLearner(),
            subdir=SUBDIR, journal=HostJournal(self.store, "fit-trace"),
            checkpointing=Checkpointing(every=2)))
        events = [e for e in self.store.read_host_log("fit-trace") if e["event"] == "fit_steps"]
        samples = [s for e in events for s in e["samples"]]
        self.assertEqual([s["optimizer_step"] for s in samples], [1, 10, 12])
        self.assertEqual({l["job"] for s in samples for l in s["lanes"]}, {j.out for j in jobs})
        self.assertTrue(all(s["t"] > 0 for s in samples))
        self.assertTrue(all(e["run_ref"] == f"{SUBDIR}/{report.run_id}" for e in events))
        self.assertTrue(all("fit_steps" not in line for line in self.store.peek_ledger(f"{SUBDIR}/{report.run_id}")))

    def test_a_fit_run_writes_every_name_once_and_a_line_per_job(self) -> None:
        jobs = [self.job(f"bank/m-{i}", epochs=1 + i % 2) for i in range(5)]
        learner = LaneLearner()
        report = self.run_fit(self.spec_of(jobs), learner)
        self.assertEqual((report.extent, report.completed, report.stopped), ("fit", 5, False))
        ref = f"{SUBDIR}/{report.run_id}"
        self.assertTrue(run_done(self.store, ref))
        self.assertEqual(run_progress(self.store, ref).planned, 5)
        for job in jobs:
            self.assertEqual(self.store.named_state(SUBDIR, job.out), PRESENT)
            self.assertEqual(self.store.named_writer(SUBDIR, job.out), ref)
            meta = self.store.named_meta(SUBDIR, job.out)
            self.assertEqual((meta["base"], meta["r"], meta["site"], meta["loss"]),
                             (BASE, 4, "layers.0-1.self_attn.q_proj", "sequence_sft"))
            self.assertEqual((meta["parent"], meta["run_ref"], meta["probe"]),
                             (None, ref, self.probe))
            self.assertEqual(meta["stages"], [{"rows": self.rows, "epochs": job.stages[0].epochs,
                                               "batch": 7, "lr": 1e-3, "decay": "linear"}])
            self.assertEqual(meta["steps"], job.stages[0].epochs)
            self.assertEqual(len(meta["probes"]), 2)
        emitted = [call[1] for call in learner.calls if call[0] == "emit_set"]
        self.assertEqual(len(emitted), 5)                # each name emitted once
        # ONE LEDGER LINE PER JOB, IN PLAN ORDER, whatever order lanes finished in
        run = self.store.open_run(report.run_id, subdir=SUBDIR, create=False)
        ledger = run.read_ledger()
        self.assertEqual([line["update"] for line in ledger], [1, 2, 3, 4, 5])
        self.assertEqual([line["out"] for line in ledger], [job.out for job in jobs])
        first = ledger[0]
        self.assertEqual(sorted(first), ["out", "probes", "start", "train", "update", "versions"])
        self.assertEqual(first["versions"], {"pi": 0})
        self.assertEqual(first["train"]["steps"], 1)
        self.assertEqual(first["probes"], self.store.named_meta(SUBDIR, "bank/m-0")["probes"])
        self.assertAlmostEqual(first["train"]["probe_last"], sum(first["probes"][-1]) / 3)
        # every=2 over five jobs: 0 (the initial blobs), 2, 4 and the extent
        self.assertEqual([c["update"] for c in run.read_checkpoints()], [0, 2, 4, 5])
        self.assertEqual(run.read_plan("fit"), encode_jobs(jobs))

    def test_a_second_attach_fits_nothing(self) -> None:
        jobs = [self.job(f"bank/m-{i}") for i in range(3)]
        spec = self.spec_of(jobs)
        self.run_fit(spec)
        again = LaneLearner()
        report = self.run_fit(spec, again, resume=True)
        self.assertEqual(report.completed, 3)
        self.assertEqual(again.loads(), [])

    def test_resume_skips_the_names_that_are_present(self) -> None:
        """Killed while fitting the fourth job: three names exist, and the
        rerun fits exactly the other two — and (on one lane, where a job's
        fresh init cannot depend on which lane was free) its ledger and its
        names are the bytes an uninterrupted run writes."""
        for lanes in (1, 2):
            with self.subTest(lanes=lanes):
                self.setUp()
                self.killed_and_resumed(lanes)

    def killed_and_resumed(self, lanes: int) -> None:
        jobs = [self.job(f"bank/m-{i}", epochs=1 + i) for i in range(5)]

        class Dies(LaneLearner):
            def emit_set(self, tenant, entry, route):
                if sum(1 for call in self.calls if call[0] == "emit_set") == 3:
                    raise RuntimeError("killed")
                return super().emit_set(tenant, entry, route)

        spec = self.spec_of(jobs, lanes=lanes)
        with self.assertRaises(RuntimeError):
            self.run_fit(spec, Dies(), every=4)
        present = [job.out for job in jobs
                   if self.store.named_state(SUBDIR, job.out) == PRESENT]
        self.assertEqual(len(present), 3)
        self.assertTrue(all(self.store.named_state(SUBDIR, job.out) == PROMISED
                            for job in jobs if job.out not in present))
        before = {name: self.store.read_named(SUBDIR, name) for name in present}

        resumed = LaneLearner()
        report = self.run_fit(spec, resumed, every=4, resume=True)
        self.assertEqual(report.completed, 5)
        self.assertEqual(len(resumed.loads()), 2)
        self.assertEqual({name: self.store.read_named(SUBDIR, name) for name in present}, before)
        run = self.store.open_run(report.run_id, subdir=SUBDIR, create=False)
        self.assertEqual([line["out"] for line in run.read_ledger()], [job.out for job in jobs])
        self.assertEqual([c["update"] for c in run.read_checkpoints()], [0, 4, 5])
        for line in run.read_ledger():      # every line is its name's own meta
            self.assertEqual(line["probes"], self.store.named_meta(SUBDIR, line["out"])["probes"])
        if lanes > 1:
            return

        with tempfile.TemporaryDirectory() as other:
            clean = LocalStore(other)
            clean.cas_put(jsonl(SEVEN)), clean.cas_put(jsonl(PROBE))
            straight = self.run_fit(self.spec_of(jobs, clean, lanes=lanes), store=clean, every=4)
            self.assertEqual(straight.run_id, report.run_id)
            for name in ("ledger.jsonl", "checkpoints.jsonl"):
                key = f"runs/{SUBDIR}/{report.run_id}/{name}"
                self.assertEqual(clean._read(key), self.store._read(key), name)
            for job in jobs:
                self.assertEqual(clean.read_named(SUBDIR, job.out),
                                 self.store.read_named(SUBDIR, job.out))

    def test_a_chain_inside_one_plan_starts_each_job_from_the_one_before(self) -> None:
        jobs = [self.job("chain/0"), self.job("chain/1", start="chain/0"),
                self.job("chain/2", start="chain/1")]
        learner = LaneLearner()
        report = self.run_fit(self.spec_of(jobs, lanes=3), learner)
        self.assertEqual(report.completed, 3)
        loaded = [payload for _, payload in learner.loads()]
        self.assertEqual(loaded, [None, self.store.read_named(SUBDIR, "chain/0"),
                                  self.store.read_named(SUBDIR, "chain/1")])
        self.assertEqual(self.store.named_meta(SUBDIR, "chain/2")["parent"], "chain/1")

    def test_a_start_another_run_is_still_writing_is_waited_for(self) -> None:
        spec = self.spec_of([self.job("child/0", start="parent/0")])
        learner = LaneLearner()

        async def go():
            async def parent_arrives():
                await asyncio.sleep(0.2)
                self.assertEqual(learner.loads(), [])       # nothing started from init
                self.store.write_named(SUBDIR, "parent/0", b"parent-bytes", {"r": 4})
            report, _ = await asyncio.gather(
                run_experiment_async(spec, SCHEMA, self.store, {}, learner, subdir=SUBDIR,
                                     checkpointing=Checkpointing(every=1)),
                parent_arrives())
            return report

        report = asyncio.run(go())
        self.assertEqual(report.completed, 1)
        self.assertEqual([payload for _, payload in learner.loads()], [b"parent-bytes"])

    def test_an_orphaned_start_refuses_the_run(self) -> None:
        self.store.promise_named(SUBDIR, ["parent/0"], f"{SUBDIR}/deadrun")
        self.store.append_fleet_event({"event": "failed", "run_id": "deadrun"})
        learner = LaneLearner()
        with self.assertRaises(NamedAdapterOrphaned):
            self.run_fit(self.spec_of([self.job("child/0", start="parent/0")]), learner)
        self.assertEqual(learner.loads(), [])
        self.assertEqual(self.store.named_state(SUBDIR, "child/0"), PROMISED)

    def test_a_fit_run_filed_at_the_store_s_root_is_refused(self) -> None:
        from rlstack.data.plan import PlanError
        with self.assertRaises(PlanError):
            run_experiment(self.spec_of([self.job("m/0")]), SCHEMA, self.store, {}, LaneLearner(),
                           checkpointing=Checkpointing(every=1))

    def test_a_drained_stop_finishes_the_fitting_jobs_and_seals_them(self) -> None:
        jobs = [self.job(f"bank/m-{i}") for i in range(6)]
        spec = self.spec_of(jobs)
        stop = StopRequest()

        class StopsAfterTwo(LaneLearner):
            def emit_set(self, tenant, entry, route):
                if sum(1 for call in self.calls if call[0] == "emit_set") == 1:
                    stop.request("test")
                return super().emit_set(tenant, entry, route)

        report = asyncio.run(run_experiment_async(
            spec, SCHEMA, self.store, {}, StopsAfterTwo(), subdir=SUBDIR,
            checkpointing=Checkpointing(every=100), stop=stop))
        self.assertTrue(report.stopped)
        self.assertTrue(stop.drained)
        self.assertEqual(report.completed, 2)
        run = self.store.open_run(report.run_id, subdir=SUBDIR, create=False)
        self.assertEqual([c["update"] for c in run.read_checkpoints()], [0, 2])
        self.assertEqual(len(run.read_ledger()), 2)      # the attach kept both lines
        resumed = self.run_fit(spec, every=100, resume=True)
        self.assertEqual(resumed.completed, 6)
        self.assertTrue(run_done(self.store, f"{SUBDIR}/{report.run_id}"))


if __name__ == "__main__":
    unittest.main()
