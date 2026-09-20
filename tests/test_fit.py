"""The fit loop (ADR 0019, runner/fit.py): SEAL's arithmetic, lanes, probes.

Run against the REAL fake: FakeLearner keeps one digest per named set, folding
only that set's own documents, in order, its step count and its lr scale — so
"K lanes give the same per-lane result as K separate tenants" is an exact
equality here. `LaneLearner` only RECORDS the verbs on their way through.
"""

from __future__ import annotations

import asyncio
import unittest
from collections.abc import Mapping

from rlstack import FakeLearner
from rlstack.data.flatten import TokenBatch
from rlstack.data.trajectory import Message, Role, Task, Trajectory, Turn, trajectory_to_row
from rlstack.policy.adapters.dream_bank import memory_route
from rlstack.policy.siteschema import fake_qwen_schema, resolve
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.fit import (
    FitJob, FitPlanError, FitResult, Stage, check_plan, decay_scale, decode_jobs,
    encode_jobs, epoch_batches, rows_slice, run_fits, sealed_rows, stage_steps,
)
from rlstack.runner.interfaces import EntryInstall, OptimSettings, Parameterization
from rlstack.runner.roles.base import StopRequest

TENANT, ENTRY = "fit-tenant", "pi"
SCHEMA = fake_qwen_schema(2, base="fake/base")


def row(text: str, prompt: str = "recall") -> dict:
    """One supervised trajectory row: `text`'s characters are its tokens."""
    tokens = tuple(ord(c) for c in text)
    context, response = Message(Role.USER, prompt), Message(Role.ASSISTANT, text)
    turn = Turn(response, tokens, (0.0,) * len(tokens), "stop", None, "supervised", {}, 0)
    return trajectory_to_row(Trajectory(Task(id=f"t/{text}", prompt=prompt, meta={}),
                                        (context, response), (turn,), {}))


def route_of(batch: TokenBatch, doc: int) -> str:
    return batch.doc_turn_extras[doc][0].get("route", "dreamer")


def doc_tokens(batch: TokenBatch, doc: int) -> tuple[int, ...]:
    starts = (*batch.doc_starts, len(batch))
    return batch.token_ids[starts[doc]:starts[doc + 1]]


class LaneLearner(FakeLearner):
    """The real FakeLearner, with every verb the fit loop asks written down."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []
        self.installs: list[tuple] = []

    def install(self, tenant, parameterization) -> None:
        self.installs.append((tenant, parameterization))
        super().install(tenant, parameterization)

    def load_set(self, tenant: str, entry: str, route: str, payload: bytes | None) -> None:
        self.calls.append(("load_set", route, payload))
        super().load_set(tenant, entry, route, payload)

    def forward_backward(self, tenant: str, batch: TokenBatch):
        self.calls.append(("forward_backward", batch))
        return super().forward_backward(tenant, batch)

    def optim_step(self, tenant: str, lr_scales: Mapping[str, float] | None = None) -> None:
        self.calls.append(("optim_step", dict(lr_scales or {})))
        super().optim_step(tenant, lr_scales)

    def forward(self, tenant: str, batch: TokenBatch) -> tuple[float, ...]:
        self.calls.append(("forward", batch))
        return super().forward(tenant, batch)

    def emit_set(self, tenant: str, entry: str, route: str) -> bytes:
        self.calls.append(("emit_set", route))
        return super().emit_set(tenant, entry, route)

    # what the tests read

    def scales(self) -> list[dict[str, float]]:
        return [call[1] for call in self.calls if call[0] == "optim_step"]

    def loads(self) -> list[tuple[str, bytes | None]]:
        return [(call[1], call[2]) for call in self.calls if call[0] == "load_set"]


def installed(learner: LaneLearner, tenant: str = TENANT, memories: int = 8) -> LaneLearner:
    """A tenant whose one entry is a dream_bank of `memories` lanes."""
    learner.install(tenant, Parameterization(
        base="fake/base", loss="sequence_sft",
        entries=(EntryInstall(
            name=ENTRY, adapter_type="dream_bank", trainable=True,
            init={"r": 4, "memories": memories, "seed": 1},
            sites=tuple(resolve(SCHEMA.sites, "layers.0-1.self_attn.q_proj"))),),
        optim=OptimSettings("adamw", 1e-4, (0.9, 0.999), 0.0, {})))
    learner.installs.clear()
    return learner


class Fits:
    """One run_fits call over in-memory files, recording what finished."""

    def __init__(self, files: Mapping[str, list[dict]], *, lanes: int,
                 learner: LaneLearner | None = None, base_lr: float = 1e-4,
                 microbatch_tokens: int = 4096, names: Mapping[str, bytes] | None = None,
                 stop: StopRequest | None = None, stop_after: int | None = None) -> None:
        self.files, self.lanes, self.base_lr = dict(files), lanes, base_lr
        self.learner = learner or installed(LaneLearner())
        self.microbatch_tokens = microbatch_tokens
        self.names = dict(names or {})
        self.stop, self.stop_after = stop, stop_after
        self.done: list[tuple[int, FitResult, bytes | None]] = []

    async def rows_of(self, uri: str):
        return self.files[uri]

    async def payload_of(self, name: str) -> bytes:
        while name not in self.names:
            await asyncio.sleep(0)
        return self.names[name]

    async def on_done(self, index: int, result: FitResult, payload: bytes | None) -> None:
        self.done.append((index, result, payload))
        if result.out is not None:
            self.names[result.out] = payload
        if self.stop_after is not None and len(self.done) >= self.stop_after:
            self.stop.request("test")

    def run(self, jobs: list[FitJob]) -> list[FitResult]:
        async def go():
            arbiter = Arbiter()
            arbiter.attach(self.learner, label="learner")
            return await run_fits(
                self.learner, TENANT, ENTRY, jobs, self.rows_of, self.payload_of,
                self.lanes, arbiter=arbiter, microbatch_tokens=self.microbatch_tokens,
                base_lr=self.base_lr, on_done=self.on_done, stop=self.stop)
        return asyncio.run(go())


SEVEN = [row(f"fact {i} is remembered") for i in range(7)]
PROBE = [row("what is fact 1"), row("what is fact 2"), row("what is fact 3")]


class SealArithmeticTest(unittest.TestCase):
    def test_seven_rows_at_batch_one_for_ten_epochs_is_seventy_steps(self) -> None:
        fits = Fits({"cas://seven": SEVEN}, lanes=1, base_lr=1e-4)
        (result,) = fits.run([FitJob("m/0", None, (Stage("cas://seven", 10, 1, 1e-3),))])
        self.assertEqual(result.steps, 70)
        self.assertEqual(stage_steps(Stage("x", 10, 1, 1e-3), 7), 70)
        scales = [step[memory_route(0)] for step in fits.learner.scales()]
        self.assertEqual(len(scales), 70)
        # the first step runs at the stage's lr, the last at lr/T, linearly
        for step, scale in enumerate(scales):
            self.assertAlmostEqual(scale, (1e-3 / 1e-4) * (70 - step) / 70, places=12)
        self.assertAlmostEqual(scales[0], 10.0)
        self.assertAlmostEqual(scales[-1], 10.0 / 70)

    def test_a_short_last_batch_is_kept_so_steps_are_the_ceiling(self) -> None:
        fits = Fits({"cas://seven": SEVEN}, lanes=1)
        (result,) = fits.run([FitJob("m/0", None, (Stage("cas://seven", 3, 2, 1e-4),))])
        self.assertEqual(result.steps, 3 * 4)
        batches = [call[1] for call in fits.learner.calls if call[0] == "forward_backward"]
        self.assertEqual([len(b.doc_starts) for b in batches], [2, 2, 2, 1] * 3)
        # the normalizer a sequence loss reads is the lane's OWN batch
        self.assertEqual([b.documents_in_update for b in batches], [2, 2, 2, 1] * 3)

    def test_a_constant_stage_does_not_decay(self) -> None:
        self.assertEqual(decay_scale("constant", 5, 9), 1.0)
        self.assertEqual(decay_scale("linear", 0, 4), 1.0)
        self.assertEqual(decay_scale("linear", 3, 4), 0.25)

    def test_stages_run_in_order_each_with_its_own_schedule(self) -> None:
        fits = Fits({"cas://seven": SEVEN, "cas://probe": PROBE}, lanes=1)
        job = FitJob("m/0", None, (Stage("cas://seven", 1, 7, 2e-4),
                                   Stage("cas://probe", 2, 3, 1e-4, "constant")))
        (result,) = fits.run([job])
        self.assertEqual(result.steps, 1 + 2)
        scales = [step[memory_route(0)] for step in fits.learner.scales()]
        self.assertEqual(scales, [2.0, 1.0, 1.0])


class RowsSpanTest(unittest.TestCase):
    """ONE FILE, MANY STAGES: a bank shard packs its rows into one store
    object and every stage names its span (2026-09-19: the scratch API takes
    seconds per object and rate-limits a dozen writers)."""

    def test_a_rows_uri_is_a_file_or_a_span_of_one(self):
        self.assertEqual(rows_slice("cas://abc"), ("cas://abc", None, None))
        self.assertEqual(rows_slice("cas://abc#3:7"), ("cas://abc", 3, 7))
        for bad in ("store://x", "cas://", "cas://abc#3", "cas://abc#7:3", "cas://abc#a:b"):
            with self.assertRaises(FitPlanError):
                rows_slice(bad)

    def test_a_span_reads_its_rows_and_the_file_is_asked_for_by_its_own_uri(self):
        asked = []

        class Refs:
            def rows(self, uri):
                asked.append(uri)
                return [{"i": i} for i in range(10)]
        self.assertEqual(asyncio.run(sealed_rows(Refs(), "cas://abc#2:5")), [{"i": 2}, {"i": 3}, {"i": 4}])
        self.assertEqual(len(asyncio.run(sealed_rows(Refs(), "cas://abc"))), 10)
        self.assertEqual(asked, ["cas://abc", "cas://abc"])
        with self.assertRaises(FitPlanError):
            asyncio.run(sealed_rows(Refs(), "cas://abc#2:50"))


class ShuffleTest(unittest.TestCase):
    def test_an_epoch_is_a_seeded_permutation_cut_into_batches(self) -> None:
        first = epoch_batches(7, 2, "m/0", 0, 0)
        self.assertEqual(first, epoch_batches(7, 2, "m/0", 0, 0))
        self.assertEqual(sorted(i for batch in first for i in batch), list(range(7)))
        self.assertEqual([len(batch) for batch in first], [2, 2, 2, 1])
        self.assertNotEqual(first, epoch_batches(7, 2, "m/0", 0, 1))
        self.assertNotEqual(first, epoch_batches(7, 2, "m/1", 0, 0))

    def test_two_runs_of_one_job_train_the_same_documents_in_the_same_order(self) -> None:
        def order() -> list[tuple[int, ...]]:
            fits = Fits({"cas://seven": SEVEN}, lanes=1)
            fits.run([FitJob("m/0", None, (Stage("cas://seven", 3, 1, 1e-4),))])
            return [doc_tokens(call[1], 0) for call in fits.learner.calls
                    if call[0] == "forward_backward"]
        first = order()
        self.assertEqual(first, order())
        self.assertEqual(len(first), 21)
        for epoch in range(3):      # every epoch sees every row once
            self.assertEqual(len(set(first[epoch * 7:(epoch + 1) * 7])), 7)
        self.assertNotEqual(first[:7], first[7:14])


class LanesTest(unittest.TestCase):
    FILES = {"cas://three": SEVEN[:3], "cas://five": SEVEN[2:], "cas://probe": PROBE}
    SHORT = FitJob("m/short", None, (Stage("cas://three", 2, 1, 1e-4),), "cas://probe")   # 6 steps
    LONG = FitJob("m/long", None, (Stage("cas://five", 3, 2, 3e-4),), "cas://probe")      # 9 steps

    def test_two_lanes_of_different_lengths_share_steps_until_one_ends(self) -> None:
        fits = Fits(self.FILES, lanes=2)
        short, long = fits.run([self.SHORT, self.LONG])
        self.assertEqual((short.steps, long.steps), (6, 9))
        scales = fits.learner.scales()
        self.assertEqual(len(scales), 9)                 # ONE optim_step per shared step
        a, b = memory_route(0), memory_route(1)
        self.assertEqual([sorted(step) for step in scales], [[a, b]] * 6 + [[b]] * 3)
        for step in range(6):
            self.assertAlmostEqual(scales[step][a], (6 - step) / 6)
        for step in range(9):
            self.assertAlmostEqual(scales[step][b], 3.0 * (9 - step) / 9)

    def test_rows_are_stamped_with_their_lane_s_route(self) -> None:
        fits = Fits(self.FILES, lanes=2)
        fits.run([self.SHORT, self.LONG])
        three = {tuple(ord(c) for c in r["turns"][0]["content"]) for r in SEVEN[:2]}
        for call in fits.learner.calls:
            if call[0] not in ("forward_backward", "forward"):
                continue
            batch = call[1]
            for doc in range(len(batch.doc_starts)):
                facts = batch.doc_turn_extras[doc][0]
                self.assertEqual(facts["role"], facts["route"])
                if call[0] == "forward_backward" and any(
                        doc_tokens(batch, doc)[-len(t):] == t for t in three):
                    self.assertEqual(facts["route"], memory_route(0))

    def test_a_microbatch_holds_only_lanes_contributing_the_same_count(self) -> None:
        """SHORT gives 1 document a step, LONG gives 2, 2, then the epoch's
        short last 1: a forward holds lanes of ONE count, and says it."""
        from collections import Counter
        fits = Fits(self.FILES, lanes=2)
        fits.run([self.SHORT, self.LONG])
        shared = 0
        for call in fits.learner.calls:
            if call[0] == "forward_backward":
                batch = call[1]
                per_route = Counter(route_of(batch, d) for d in range(len(batch.doc_starts)))
                self.assertEqual(set(per_route.values()), {batch.documents_in_update})
                shared += len(per_route) == 2
        self.assertGreater(shared, 0)        # LONG's short batch rode with SHORT's

    def test_lanes_of_one_batch_size_share_their_forwards(self) -> None:
        jobs = [FitJob(f"m/{i}", None, (Stage("cas://three", 2, 1, 1e-4),)) for i in range(3)]
        fits = Fits(self.FILES, lanes=3)
        fits.run(jobs)
        batches = [call[1] for call in fits.learner.calls if call[0] == "forward_backward"]
        self.assertEqual(len(batches), 6)             # 3 lanes, ONE forward per step
        for batch in batches:
            self.assertEqual({route_of(batch, d) for d in range(3)},
                             {memory_route(i) for i in range(3)})
            self.assertEqual(batch.documents_in_update, 1)   # each lane's own batch

    def test_the_microbatch_budget_bounds_every_forward(self) -> None:
        jobs = [FitJob(f"m/{i}", None, (Stage("cas://three", 1, 1, 1e-4),)) for i in range(3)]
        longest = max(len(r["turns"][0]["token_ids"]) + len("recall") for r in SEVEN[:3])
        fits = Fits(self.FILES, lanes=3, microbatch_tokens=2 * longest)
        fits.run(jobs)
        batches = [call[1] for call in fits.learner.calls if call[0] == "forward_backward"]
        self.assertTrue(all(len(b.doc_starts) <= 2 for b in batches))
        self.assertTrue(all(b.microbatches_in_update == 2 for b in batches))
        self.assertEqual(len(fits.learner.scales()), 3)

    def test_k_lanes_give_each_job_what_it_gets_alone(self) -> None:
        """From one start, a job's payload and probes are the same whether it
        shared its forwards or not, and whichever lane it sat in."""
        from dataclasses import replace
        names = {"lib/start": b"the-parent"}
        jobs = [replace(self.SHORT, start="lib/start"), replace(self.LONG, start="lib/start")]
        together = Fits(self.FILES, lanes=2, names=names)
        together.run(jobs)
        for job in jobs:
            alone = Fits(self.FILES, lanes=1, names=names)
            alone.run([job])
            (_, result, payload), = alone.done
            shared = next(d for d in together.done if d[1].out == job.out)
            self.assertEqual(shared[1], result)
            self.assertEqual(shared[2], payload)

    def test_a_fresh_set_is_its_lane_s_own_init(self) -> None:
        """start=None re-initializes the LANE's set, and a learner seeds each
        set off its route: the same fresh job is the same fit on the same
        lane, and a different draw of the init on another."""
        def fit_on(lane: int) -> bytes:
            fillers = [FitJob(f"m/filler-{i}", None, (Stage("cas://five", 9, 1, 1e-4),))
                       for i in range(lane)]
            fits = Fits(self.FILES, lanes=lane + 1)
            fits.run(fillers + [self.SHORT])
            return next(d[2] for d in fits.done if d[1].out == self.SHORT.out)
        self.assertEqual(fit_on(0), fit_on(0))
        self.assertNotEqual(fit_on(0), fit_on(1))

    def test_a_freed_lane_takes_the_next_job_and_is_loaded_again(self) -> None:
        third = FitJob("m/third", None, (Stage("cas://three", 1, 3, 1e-4),))
        fits = Fits(self.FILES, lanes=2)
        results = fits.run([self.SHORT, self.LONG, third])
        self.assertEqual([r.out for r in results], ["m/short", "m/long", "m/third"])
        self.assertEqual(fits.learner.loads(), [
            (memory_route(0), None), (memory_route(1), None), (memory_route(0), None)])
        self.assertEqual([d[1].out for d in fits.done], ["m/short", "m/third", "m/long"])


class ProbeTest(unittest.TestCase):
    def test_a_probe_is_taken_before_training_and_after_every_stage(self) -> None:
        fits = Fits({"cas://seven": SEVEN, "cas://probe": PROBE}, lanes=1)
        job = FitJob("m/0", None, (Stage("cas://seven", 1, 7, 1e-4),
                                   Stage("cas://seven", 2, 4, 1e-4)), "cas://probe")
        (result,) = fits.run([job])
        self.assertEqual(len(result.probes), 3)
        self.assertTrue(all(len(vector) == len(PROBE) for vector in result.probes))
        self.assertEqual(len(set(result.probes)), 3)     # training moved the set
        verbs = [call[0] for call in fits.learner.calls]
        self.assertEqual(verbs[:2], ["load_set", "forward"])
        self.assertEqual(verbs[-2:], ["forward", "emit_set"])

    def test_no_probe_no_vectors_and_no_forward(self) -> None:
        fits = Fits({"cas://seven": SEVEN}, lanes=1)
        (result,) = fits.run([FitJob("m/0", None, (Stage("cas://seven", 1, 7, 1e-4),))])
        self.assertEqual(result.probes, ())
        self.assertNotIn("forward", [call[0] for call in fits.learner.calls])

    def test_a_job_with_no_out_is_never_emitted(self) -> None:
        fits = Fits({"cas://seven": SEVEN, "cas://probe": PROBE}, lanes=1)
        fits.run([FitJob(None, None, (Stage("cas://seven", 1, 7, 1e-4),), "cas://probe")])
        self.assertNotIn("emit_set", [call[0] for call in fits.learner.calls])
        self.assertIsNone(fits.done[0][2])


class StartTest(unittest.TestCase):
    FILES = {"cas://seven": SEVEN}

    def test_a_start_is_loaded_as_the_bytes_its_name_holds(self) -> None:
        fits = Fits(self.FILES, lanes=1, names={"lib/a": b"parent-bytes"})
        fits.run([FitJob("m/0", "lib/a", (Stage("cas://seven", 1, 7, 1e-4),))])
        self.assertEqual(fits.learner.loads(), [(memory_route(0), b"parent-bytes")])

    def test_a_job_may_start_from_an_earlier_job_of_the_same_call(self) -> None:
        """The chain: job 1 waits for job 0's name on its own lane while
        job 0 and job 2 fit — nobody's steps are held by the wait."""
        jobs = [FitJob("m/0", None, (Stage("cas://seven", 2, 7, 1e-4),)),
                FitJob("m/1", "m/0", (Stage("cas://seven", 1, 7, 1e-4),)),
                FitJob("m/2", None, (Stage("cas://seven", 1, 7, 1e-4),))]
        fits = Fits(self.FILES, lanes=3)
        results = fits.run(jobs)
        self.assertEqual([r.out for r in results], ["m/0", "m/1", "m/2"])
        payload_0 = next(d[2] for d in fits.done if d[1].out == "m/0")
        self.assertIn((memory_route(1), payload_0), fits.learner.loads())
        self.assertLess([d[1].out for d in fits.done].index("m/2"),
                        [d[1].out for d in fits.done].index("m/1"))

    def test_a_stop_starts_no_further_job_and_finishes_the_ones_fitting(self) -> None:
        jobs = [FitJob(f"m/{i}", None, (Stage("cas://seven", 1, 7, 1e-4),)) for i in range(5)]
        stop = StopRequest()
        fits = Fits(self.FILES, lanes=2, stop=stop, stop_after=1)
        results = fits.run(jobs)
        self.assertEqual([r.out for r in results], ["m/0", "m/1"])
        self.assertEqual(len(fits.learner.loads()), 2)

    def test_a_stop_ends_a_wait_for_a_start_that_never_came(self) -> None:
        stop = StopRequest()
        fits = Fits(self.FILES, lanes=2, stop=stop, stop_after=1)
        results = fits.run([FitJob("m/0", None, (Stage("cas://seven", 1, 7, 1e-4),)),
                            FitJob("m/1", "never/written", (Stage("cas://seven", 1, 7, 1e-4),))])
        self.assertEqual([r.out for r in results], ["m/0"])
        self.assertEqual(fits.learner.loads(), [(memory_route(0), None)])


class PlanTest(unittest.TestCase):
    JOB = FitJob("bank/m-0", None, (Stage("cas://abc", 10, 1, 1e-3),), "cas://probe")

    def test_a_plan_roundtrips_one_canonical_line_per_job(self) -> None:
        jobs = (self.JOB, FitJob("bank/m-1", "bank/m-0", (Stage("cas://abc", 1, 2, 5e-4, "constant"),)))
        data = encode_jobs(jobs)
        self.assertEqual(len(data.splitlines()), 2)
        self.assertEqual(decode_jobs(data), jobs)
        self.assertEqual(
            data.splitlines()[0].decode(),
            '{"out":"bank/m-0","probe":"cas://probe","stages":[{"batch":1,"decay":"linear",'
            '"epochs":10,"lr":0.001,"rows":"cas://abc"}],"start":null}')

    def refused(self, *jobs: FitJob) -> str:
        with self.assertRaises(FitPlanError) as caught:
            check_plan(jobs)
        return str(caught.exception)

    def test_illegal_plans_are_refused_by_what_is_wrong(self) -> None:
        stage = self.JOB.stages[0]
        from dataclasses import replace
        self.assertIn("at least one job", self.refused())
        self.assertIn("names its out", self.refused(replace(self.JOB, out=None)))
        self.assertIn("written once", self.refused(self.JOB, self.JOB))
        self.assertIn("not a name", self.refused(replace(self.JOB, out="lib:bad")))
        self.assertIn("not a name", self.refused(replace(self.JOB, start="../up")))
        self.assertIn("at least one stage", self.refused(replace(self.JOB, stages=())))
        self.assertIn("cas://", self.refused(replace(self.JOB, stages=(replace(stage, rows="rows.jsonl"),))))
        self.assertIn("epochs", self.refused(replace(self.JOB, stages=(replace(stage, epochs=0),))))
        self.assertIn("batch", self.refused(replace(self.JOB, stages=(replace(stage, batch=0),))))
        self.assertIn("lr", self.refused(replace(self.JOB, stages=(replace(stage, lr=0.0),))))
        self.assertIn("decay", self.refused(replace(self.JOB, stages=(replace(stage, decay="cosine"),))))

    def test_a_start_this_plan_writes_is_written_by_an_earlier_job(self) -> None:
        late = FitJob("bank/m-1", "bank/m-2", self.JOB.stages)
        self.assertIn("earlier job", self.refused(late, FitJob("bank/m-2", None, self.JOB.stages)))
        check_plan((self.JOB, FitJob("bank/m-1", "bank/m-0", self.JOB.stages),
                    FitJob("bank/m-2", "another-run/m-9", self.JOB.stages)))

    def test_an_empty_stage_is_refused_when_its_rows_are_read(self) -> None:
        fits = Fits({"cas://empty": []}, lanes=1)
        with self.assertRaises(FitPlanError):
            fits.run([FitJob("m/0", None, (Stage("cas://empty", 1, 1, 1e-4),))])


if __name__ == "__main__":
    unittest.main()
