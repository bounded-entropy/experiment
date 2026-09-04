"""ADR 0005: a steer distilled from a prompt-conditioned teacher.

The shape, end to end on fakes: a task carries the system block as
`meta["hint"]`; the `conditioned_teacher` environment samples under hint +
prompt and seals the hint OUT; a generation-only run with an EMPTY bank
leaves those rollouts; an SFT run with a steer entry replays them by
`store://<teacher>/rollouts/<r>#<i>` and commits, byte-identically across a
crash. Beside it the measurement channel: `conditioned_teacher_logprobs`
scores the same walk with the hint at the head of the context, and
`reverse_kl` reduces it against the record.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from common import char_tokenize, make_turn
from test_resume import CrashingStore, SimulatedCrash, snapshot
from rlstack import (
    POST, AlgoSpec, Bundle, EnginePoolClient, ExperimentSpec, FakeEngine,
    FakeLearner, GenSpec, Group, GroupPlan, HostSpec, LocalStore, Mechanism,
    Message, OptimSpec, Plans, PolicySpec, Replay, Role, Rollout, RunPlan,
    Sample, SamplingSpec, Schedule, Seeds, Task, Topology, Wave, WavePlan,
    encode, fake_qwen_schema, learner, pool, run_experiment, run_pipeline,
    steer, write_tasks,
)

BUNDLE = Bundle(bundle_id="bundle:teacher0", policy_version={})
TEACHER_BASE = "Qwen/Qwen3-32B"
TEACHER_BUNDLE = Bundle(bundle_id="bundle:base:teacher", policy_version={})
HINT = ("You are a helpful assistant who is deeply preoccupied with "
        "happiness.\n\n")


def go(coro):
    return asyncio.run(coro)


class RecordingEngine(FakeEngine):
    """A fake that keeps every context it was asked to continue — the only
    way to see, from outside, what conditioning a request carried."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.contexts: list[tuple[str, ...]] = []

    async def sample_tokens(self, messages, *args, **kwargs):
        self.contexts.append(tuple(m.content for m in messages))
        async for event in super().sample_tokens(messages, *args, **kwargs):
            yield event

    async def score_tokens(self, messages, *args, **kwargs):
        self.contexts.append(tuple(m.content for m in messages))
        return await super().score_tokens(messages, *args, **kwargs)


def a_task(task_id: str = "no-robots/1", prompt: str = "What is 2+2?") -> Task:
    return Task(task_id, prompt, {"hint": HINT, "concept": "happiness"})


def a_client(engine: FakeEngine) -> EnginePoolClient:
    engine.add_bundle(BUNDLE)
    return EnginePoolClient({"main": (engine, BUNDLE)}, SamplingSpec(),
                            episode_seed=5)


class ConditionedTeacherEnvTest(unittest.TestCase):
    """The environment's whole claim: the hint conditions the request and is
    absent from the record the student will train on."""

    def rollout(self):
        from rlstack.registry import ENVS

        engine = RecordingEngine()
        env = ENVS.get("conditioned_teacher").instance
        task = a_task()
        return engine, task, go(env.run(a_client(engine), task))

    def test_the_hint_conditions_the_request(self) -> None:
        engine, task, _ = self.rollout()
        self.assertEqual(engine.contexts, [(HINT, task.prompt)])

    def test_the_sealed_stream_is_prompt_and_completion_alone(self) -> None:
        _, task, rollout = self.rollout()
        traj = rollout.seal()
        self.assertEqual([m.content for m in traj.messages],
                         [task.prompt, traj.turns[0].message.content])
        self.assertEqual([m.role for m in traj.messages],
                         [Role.USER, Role.ASSISTANT])
        self.assertNotIn(HINT, traj.text)

    def test_the_hint_survives_as_provenance(self) -> None:
        _, _, rollout = self.rollout()
        self.assertEqual(rollout.seal().env_extras["hint"], HINT)

    def test_it_records_no_window_because_its_bank_is_empty(self) -> None:
        """The teacher run serves the bare base; nothing steers, so nothing
        is recorded — and ADR 0005 Q3 makes that replayable rather than a
        bug (the steer's default is every position)."""
        _, _, rollout = self.rollout()
        self.assertEqual(dict(rollout.turns[0].turn_extras), {})


class SingleTurnEnvTest(unittest.TestCase):
    def test_it_is_math_single_turns_body_under_an_honest_name(self) -> None:
        from rlstack.registry import ENVS

        engine = RecordingEngine()
        task = Task("t", "What is 2+2?", {})
        rollout = go(ENVS.get("single_turn").instance.run(a_client(engine), task))
        self.assertEqual(engine.contexts, [(task.prompt,)])
        self.assertEqual([m.content for m in rollout.messages],
                         [task.prompt, rollout.turns[0].message.content])

    def test_the_math_name_still_exists_for_the_runs_that_hash_it(self) -> None:
        from rlstack.registry import ENVS

        self.assertIsNotNone(ENVS.get("math_single_turn"))
        self.assertIsNotNone(ENVS.get("single_turn"))


# ---------------------------------------------------------------------------
# the two processors: the conditioned walk, and the reduction over it
# ---------------------------------------------------------------------------

def two_turn(task_id: str = "no-robots/2"):
    """A trajectory with an INJECTED message between two generated turns, so
    "flatten order" means something: turn 2 is scored against turn 1 and the
    message that followed it, all of it under the hint."""
    first = make_turn("4", char_tokenize("4"))
    second = make_turn("40", char_tokenize("40"))
    return Rollout(
        task=a_task(task_id),
        messages=[Message(Role.USER, "What is 2+2?"), first.message,
                  Message(Role.USER, "And times ten?"), second.message],
        turns=[first, second],
        env_extras={"hint": HINT},
    ).seal()


def routes_and_traj():
    student, teacher = FakeEngine(), RecordingEngine(base=TEACHER_BASE)
    student.add_bundle(BUNDLE)
    teacher.add_bundle(TEACHER_BUNDLE)
    return ({"main": (student, BUNDLE), "teacher": (teacher, TEACHER_BUNDLE)},
            teacher, two_turn())


class ConditionedTeacherLogprobsTest(unittest.TestCase):
    """`teacher_scores`' walk with the hint at the head of the context, on
    the teacher's own metal and its own payload-free bundle."""

    def test_it_scores_the_teachers_pool_under_the_hint(self) -> None:
        routes, teacher, traj = routes_and_traj()

        async def scenario():
            columns = await run_pipeline(
                ("conditioned_teacher_logprobs",), Wave([Group("g", [traj])]),
                routes, SamplingSpec(), master=7, update=1)
            by_hand = list(await teacher.score_tokens(
                (Message(Role.USER, HINT),) + traj.messages[:1],
                traj.turns[0].token_ids, TEACHER_BUNDLE.bundle_id))
            by_hand += list(await teacher.score_tokens(
                (Message(Role.USER, HINT),) + traj.messages[:3],
                traj.turns[1].token_ids, TEACHER_BUNDLE.bundle_id))
            return columns["teacher_logprobs"][0], by_hand

        column, by_hand = go(scenario())
        self.assertEqual(column, by_hand)
        self.assertEqual(len(column), 3)               # "4" + "40"

    def test_the_hint_heads_every_context_it_sent(self) -> None:
        routes, teacher, traj = routes_and_traj()
        go(run_pipeline(("conditioned_teacher_logprobs",),
                        Wave([Group("g", [traj])]), routes, SamplingSpec(),
                        master=7, update=1))
        self.assertEqual(len(teacher.contexts), 2)
        for context in teacher.contexts:
            self.assertEqual(context[0], HINT)
        self.assertEqual(teacher.contexts[1][1:],
                         tuple(m.content for m in traj.messages[:3]))

    def test_the_unconditioned_teacher_says_something_else(self) -> None:
        """A different conditioning is a different processor, and the two
        are visibly different numbers on the same pool."""
        routes, _, traj = routes_and_traj()
        wave = Wave([Group("g", [traj])])
        conditioned = go(run_pipeline(("conditioned_teacher_logprobs",), wave,
                                      routes, SamplingSpec(), 7, 1))
        plain = go(run_pipeline(("teacher_logprobs",), wave, routes,
                                SamplingSpec(), 7, 1))
        self.assertNotEqual(conditioned["teacher_logprobs"],
                            plain["teacher_logprobs"])

    def test_it_declares_the_teacher_pool_and_the_token_channel(self) -> None:
        pdef = POST.get("conditioned_teacher_logprobs")
        self.assertEqual(pdef.produces, ("teacher_logprobs",))
        self.assertEqual(pdef.token_level, ("teacher_logprobs",))
        self.assertEqual(pdef.consumes, ())
        self.assertEqual(pdef.pools, ("teacher",))
        # the SAME column name as teacher_logprobs: opd requires it unchanged
        self.assertEqual(POST.get("teacher_logprobs").produces, pdef.produces)


class ReverseKLTest(unittest.TestCase):
    def test_it_is_the_mean_of_behavior_minus_teacher(self) -> None:
        from rlstack.training.post.reverse_kl import reverse_kl_of

        traj = two_turn()
        behavior = [lp for turn in traj.turns for lp in turn.behavior_logprobs]
        self.assertEqual(reverse_kl_of(traj, [0.0] * len(behavior)),
                         sum(behavior) / len(behavior))
        self.assertEqual(reverse_kl_of(traj, behavior), 0.0)

    def test_a_misaligned_column_is_refused_not_truncated(self) -> None:
        from rlstack.training.post.reverse_kl import reverse_kl_of

        with self.assertRaises(ValueError):
            reverse_kl_of(two_turn(), [0.0])

    def test_the_pipeline_reduces_the_teachers_column(self) -> None:
        routes, _, traj = routes_and_traj()
        columns = go(run_pipeline(
            ("conditioned_teacher_logprobs", "reverse_kl"),
            Wave([Group("g", [traj])]), routes, SamplingSpec(), 7, 1))
        teacher = columns["teacher_logprobs"][0]
        behavior = [lp for turn in traj.turns for lp in turn.behavior_logprobs]
        self.assertAlmostEqual(
            columns["reverse_kl"][0],
            sum(b - t for b, t in zip(behavior, teacher)) / len(behavior),
            places=12)

    def test_it_is_pool_less_and_therefore_inline(self) -> None:
        from rlstack.spec.flow import split_pipeline

        pdef = POST.get("reverse_kl")
        self.assertEqual(pdef.pools, ())
        self.assertEqual(pdef.consumes, ("teacher_logprobs",))
        self.assertEqual(pdef.produces, ("reverse_kl",))
        split = split_pipeline(("conditioned_teacher_logprobs", "reverse_kl"))
        self.assertEqual(split.pooled, ("conditioned_teacher_logprobs",))
        self.assertEqual(split.inline, ("reverse_kl",))


class SplitOrderTest(unittest.TestCase):
    """A POOLED processor followed by an INLINE one is the legal direction:
    the Scorer writes its part first, the Trainer's half consumes it as
    `given` and cannot tell which daemon produced what it reads."""

    def spec(self):
        import tempfile
        from dataclasses import replace

        from common import arith_spec, arith_store

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, train, _ = arith_store(tmp.name)
        base = arith_spec(train)
        return replace(
            base,
            algo=replace(base.algo, loss="opd",
                         post=("conditioned_teacher_logprobs", "reverse_kl")),
            topology=Topology(hosts=(
                HostSpec((pool("main", vram_gb=8),)),
                HostSpec((pool("teacher", base=TEACHER_BASE, vram_gb=10),)),
                HostSpec((learner(vram_gb=8),)))))

    def test_the_gate_accepts_pooled_then_inline(self) -> None:
        from rlstack import fake_qwen_schema, validate

        schema = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
        self.assertEqual(validate(self.spec(), schema), [])

    def test_running_it_in_two_halves_equals_running_it_whole(self) -> None:
        routes, _, traj = routes_and_traj()
        wave = Wave([Group("g", [traj])])
        whole = go(run_pipeline(
            ("conditioned_teacher_logprobs", "reverse_kl"), wave, routes,
            SamplingSpec(), 7, 1))
        part = go(run_pipeline(("conditioned_teacher_logprobs",), wave, routes,
                               SamplingSpec(), 7, 1))
        rest = go(run_pipeline(("reverse_kl",), wave, routes, SamplingSpec(),
                               7, 1, given=part))
        self.assertEqual(part["teacher_logprobs"], whole["teacher_logprobs"])
        self.assertEqual(rest["reverse_kl"], whole["reverse_kl"])
        self.assertNotIn("teacher_logprobs", rest)     # only its own produces


# ---------------------------------------------------------------------------
# the shape end to end: the teacher's set, then a steer distilled from it
# ---------------------------------------------------------------------------

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
RESIDUAL = frozenset({Mechanism.RESIDUAL})
WAVES, ROWS = 4, 2
STEER_RECORD = "steer_window"


def concept_tasks(n: int = 4) -> list[Task]:
    """A small corpus in `concept_prompts`' shape: the ask in the prompt, the
    telling in meta["hint"]."""
    return [Task(id=f"no-robots/{i:04d}", prompt=f"What is {i}+{i}?",
                 meta={"hint": HINT, "concept": "happiness",
                       "category": "Generation"})
            for i in range(n)]


def teacher_seeded(root: str) -> tuple[LocalStore, str, str]:
    """A store holding the corpus and the teacher's rollout plan."""
    store = LocalStore(root)
    tasks = concept_tasks()
    plan = RunPlan(tuple(
        WavePlan((GroupPlan(f"w{u}", tuple(
            Sample(tasks[(u * ROWS + i) % len(tasks)].id,
                   "conditioned_teacher") for i in range(ROWS))),))
        for u in range(WAVES)))
    return (store, write_tasks(store, tasks),
            store.cas_put(encode(plan)))


def teacher_spec(tasks_uri: str, rollout_uri: str) -> ExperimentSpec:
    """The teacher: a generation-only run with an EMPTY bank, so `main` is
    the bare base and the hint is the whole of the conditioning (ADR 0005,
    Q4). Its sealed rollouts ARE the trajectory set and its run_id is the
    set's identity."""
    return ExperimentSpec(
        policy=PolicySpec(base="Qwen/Qwen3-0.6B", bank={}),
        gen=GenSpec(envs=("conditioned_teacher",), tasks=(tasks_uri,),
                    sampling=SamplingSpec(temperature=1.0, max_tokens=32)),
        plans=Plans(train=None, rollout=rollout_uri),
        algo=None,
        topology=Topology(hosts=(HostSpec((pool("main"),)),)),
        seeds=Seeds(master=5))


def student_spec(train_uri: str) -> ExperimentSpec:
    """One arm: the same base with ONE steer entry, SFT over the teacher's
    rows, sampling nothing of its own."""
    return ExperimentSpec(
        policy=PolicySpec(base="Qwen/Qwen3-0.6B",
                          bank={"v": steer("resid_pre.1", d=64)}),
        gen=None,
        plans=Plans(train=train_uri, rollout=None),
        algo=AlgoSpec(loss="sft", post=(), optim=OptimSpec("adamw", lr=5e-3),
                      schedule=Schedule(microbatch_tokens=64,
                                        max_policy_lag=0)),
        topology=Topology(hosts=(HostSpec((pool("main"),)),
                                 HostSpec((learner(),)))),
        seeds=Seeds(master=11))


def sft_plan(teacher_run: str) -> RunPlan:
    """Every row of the teacher's rollout u, replayed as update u's wave."""
    return RunPlan(tuple(
        WavePlan((GroupPlan(f"g{u}", tuple(
            Replay(f"store://{teacher_run}/rollouts/{u}#{i}")
            for i in range(ROWS))),))
        for u in range(1, WAVES + 1)))


class DistillEndToEndTest(unittest.TestCase):
    """ADR 0005's two runs on fakes: an empty-bank teacher generates under
    the hint, and a steer is SFT'd on what it sealed — in one store, the
    student sampling nothing."""

    def generate(self, root: str) -> tuple[LocalStore, str]:
        store, tasks_uri, rollout_uri = teacher_seeded(root)
        report = run_experiment(teacher_spec(tasks_uri, rollout_uri), SCHEMA,
                                store, FakeEngine(), None)
        return store, report.run_id

    def distill(self, root: str, store: LocalStore, teacher_run: str,
                write: LocalStore | None = None):
        plan = encode(sft_plan(teacher_run))
        store.cas_put(plan)
        from common import cas_uri
        return run_experiment(student_spec(cas_uri(plan)), SCHEMA,
                              write or store, FakeEngine(plugins=RESIDUAL),
                              FakeLearner())

    def both(self, root: str):
        store, teacher_run = self.generate(root)
        return store, teacher_run, self.distill(root, store, teacher_run)

    def test_the_teacher_seals_a_set_and_the_student_trains_on_it(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, teacher_run, report = self.both(tmp.name)

        teacher = store.open_run(teacher_run)
        self.assertEqual(teacher.read_ledger(), [])        # nothing commits
        rows = teacher.read_rollout(1)
        self.assertEqual(len(rows), ROWS)
        # the hint conditioned the sampling and is OUT of the record
        for row in rows:
            self.assertEqual(row["env_extras"]["hint"], HINT)
            self.assertNotIn(HINT, "".join(m["content"] for m in row["messages"]))
            self.assertEqual(row["task"]["meta"]["hint"], HINT)

        run = store.open_run(report.run_id)
        self.assertEqual((report.completed, report.extent), (WAVES, "train"))
        self.assertEqual([e["update"] for e in run.read_ledger()],
                         list(range(1, WAVES + 1)))
        self.assertEqual([row["task"]["id"] for row in run.read_wave(1)],
                         [row["task"]["id"] for row in rows])
        with self.assertRaises(FileNotFoundError):
            run.read_rollout(1)                            # it sampled nothing

    def test_the_replayed_rows_carry_no_steer_window(self) -> None:
        """The case the steer's default exists for (ADR 0005, Q3): the
        teacher's bank was empty, so nothing recorded a window, and the
        student's replay steers at every position rather than refusing."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, _, report = self.both(tmp.name)
        rows = store.open_run(report.run_id).read_wave(1)
        self.assertTrue(rows)
        for row in rows:
            for turn in row["turns"]:
                self.assertNotIn(STEER_RECORD, turn["turn_extras"])

    def test_crash_then_resume_is_byte_identical(self) -> None:
        """The SFT run's obligation (ADR 0005, promise 2): the plan is
        Replay leaves into sealed content and the window is a pure function
        of the rows, so a killed-and-resumed run leaves a straight run's
        bytes."""
        straight = tempfile.TemporaryDirectory()
        self.addCleanup(straight.cleanup)
        _, _, report = self.both(straight.name)
        reference = snapshot(LocalStore(straight.name), report.run_id)

        crashed = tempfile.TemporaryDirectory()
        self.addCleanup(crashed.cleanup)
        store, teacher_run = self.generate(crashed.name)
        with self.assertRaises(SimulatedCrash):
            self.distill(crashed.name, store, teacher_run,
                         write=CrashingStore(crashed.name, "append_ledger", 2))
        resumed = self.distill(crashed.name, LocalStore(crashed.name),
                               teacher_run)
        self.assertEqual(resumed.run_id, report.run_id)
        self.assertIsNotNone(resumed.resumed_from)
        self.assertEqual(snapshot(LocalStore(crashed.name), resumed.run_id),
                         reference)


if __name__ == "__main__":
    unittest.main()
