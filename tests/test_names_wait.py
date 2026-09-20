"""The consumer's half of named adapters (ADR 0019): which names a wave needs,
the wait until they exist, and the two roles that hand residents the bytes.

The properties under test: a role string's `lib:<name>` parts are read whether
alone or stacked; the wait returns on bytes, refuses on an orphan, and keeps
waiting — audibly — on a promise or on nothing; the Generator hands every
library set to its pool before it samples and the Trainer to its learner
before it steps, dropping what the next wave does not name; and a run whose
plans mention no library takes none of these steps.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import tempfile
import threading
import unittest

from common import TRAIN, arith_spec, arith_task_bytes, cas_uri, task_ids
from rlstack import (
    FakeEngine, FakeLearner, GroupPlan, LocalStore, Plans, PolicySpec, Replay, RunPlan,
    RunSignals, Sample, Task, WavePlan, WaveRef, encode, fake_qwen_schema,
)
from rlstack.data.plan import EVAL, PlanError
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.loop import run_experiment_async
from rlstack.runner.meters import HostJournal
from rlstack.runner.names import (
    ENDED_CHECK_EVERY_S, NOTE_EVERY_S, NamedAdapterOrphaned, NamesWait, check_plan_names,
    lib_names, names_ready, names_subdir, row_lib_names, wave_lib_names,
)
from rlstack.spec.specs import dream_bank

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
EXP = "dreams/g1"
ENV = "math_single_turn"


def go(coro):
    return asyncio.run(coro)


def fast_signals() -> RunSignals:
    """A blackboard whose beat is short enough that a test never feels it."""
    return RunSignals(poll_seconds=0.005)


def a_fit_run(store, run_id: str, *, planned: int, done: int) -> str:
    """A writer: a fit run with `done` of `planned` jobs committed and sealed."""
    run = store.open_run(run_id, manifest={"run_id": run_id}, subdir=EXP)
    run.write_plan("fit", b"".join(b'{"job": %d}\n' % i for i in range(planned)))
    for update in range(1, done + 1):
        run.append_ledger({"update": update})
    if done:
        run.append_checkpoint(done, {})
    return f"{EXP}/{run_id}"


class LibNamesTest(unittest.TestCase):
    def test_names_come_from_every_lib_part_alone_or_stacked(self) -> None:
        roles = ["train", "lib:mem/a+dreamer", "memory:03", "lib:b", "eval",
                 "dreamer+lib:mem/a", "base"]
        self.assertEqual(lib_names(roles), ("mem/a", "b"))

    def test_roles_without_a_library_name_none(self) -> None:
        self.assertEqual(lib_names(["train", "eval", "dreamer", "memory:00"]), ())

    def test_an_illegal_name_is_a_plan_error_naming_the_role(self) -> None:
        with self.assertRaisesRegex(PlanError, r"lib:\.\./x\+dreamer"):
            lib_names(["lib:../x+dreamer"])

    def test_a_planned_wave_names_its_leaf_roles_and_its_tasks_routes(self) -> None:
        tasks = {"q1": Task("q1", "?", {"route": "lib:mem/q"}),
                 "d1": Task("d1", "dream", {})}
        wave = WavePlan((GroupPlan("g", (
            Sample("d1", ENV, "lib:mem/a+dreamer"), Sample("q1", ENV, EVAL),
            Replay("cas://x#0", "lib:mem/b"))),))
        self.assertEqual(wave_lib_names(wave, tasks), ("mem/a", "mem/q", "mem/b"))
        self.assertEqual(wave_lib_names(wave), ("mem/a", "mem/b"))
        self.assertEqual(wave_lib_names(WaveRef("self://rollouts/1")), ())

    def test_realized_rows_name_the_routes_they_train_under_never_an_eval_row_s(self) -> None:
        def row(**extras):
            return {"turns": [{"turn_extras": extras}]}
        rows = [row(role="lib:mem/a+dreamer", route="lib:mem/a+dreamer"),
                row(route="lib:sampled"),                 # a train leaf: the recorded route
                row(role=EVAL, route="lib:answered"),     # scored, never forwarded
                row(route="memory:00"), row()]
        self.assertEqual(row_lib_names(rows), ("mem/a", "sampled"))

    def test_a_misspelt_name_is_refused_with_its_wave(self) -> None:
        plan = RunPlan((WavePlan((GroupPlan("g", (Sample("t", ENV),)),)),
                        WavePlan((GroupPlan("g", (Sample("t", ENV, "lib:a//b"),)),))))
        with self.assertRaisesRegex(PlanError, "wave 2"):
            check_plan_names(plan)

    def test_a_run_s_names_live_under_the_subdir_it_was_filed_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(tmp)
            filed = store.open_run("r1", manifest={"run_id": "r1"}, subdir=EXP)
            self.assertEqual(names_subdir(filed), EXP)
            at_root = store.open_run("r2", manifest={"run_id": "r2"})
            with self.assertRaisesRegex(PlanError, "store's root"):
                names_subdir(at_root)


class NamesReadyTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)

    def test_present_names_return_at_once_and_no_names_ask_nothing(self) -> None:
        self.store.write_named(EXP, "m0", b"payload", {})
        go(names_ready(self.store, EXP, ["m0"], fast_signals()))
        go(names_ready(None, EXP, [], None))        # nothing needed: nothing touched

    def test_an_orphan_refuses_naming_the_adapter_and_its_writer(self) -> None:
        writer = a_fit_run(self.store, "fit-1", planned=1, done=1)
        self.store.promise_named(EXP, ["m0"], writer)
        with self.assertRaises(NamedAdapterOrphaned) as refused:
            go(names_ready(self.store, EXP, ["m0"], fast_signals()))
        self.assertIn("'m0'", str(refused.exception))
        self.assertIn(writer, str(refused.exception))

    def test_a_promised_and_an_unknown_name_are_waited_for_until_written(self) -> None:
        writer = a_fit_run(self.store, "fit-1", planned=2, done=0)
        self.store.promise_named(EXP, ["promised"], writer)
        order: list[str] = []

        async def scenario():
            signals = fast_signals()

            async def the_fit_run():
                for name in ("promised", "unknown"):
                    await signals.wait_once()            # the consumer is waiting by now
                    order.append(f"wrote {name}")
                    self.store.write_named(EXP, name, name.encode(), {})
                    await signals.notify()

            fitting = asyncio.get_running_loop().create_task(the_fit_run())
            await names_ready(self.store, EXP, ["promised", "unknown"], signals)
            order.append("ready")
            await fitting

        go(scenario())
        self.assertEqual(order, ["wrote promised", "wrote unknown", "ready"])

    def test_a_stop_is_the_second_way_out(self) -> None:
        asked: list[bool] = []

        def stopped() -> bool:
            asked.append(True)
            return len(asked) > 2                      # a stop arrives on the third beat

        go(names_ready(self.store, EXP, ["never"], fast_signals(), stopped=stopped))
        self.assertEqual(len(asked), 3)
        self.assertEqual(self.store.named_state(EXP, "never"), "unknown")


class NamesWaitPacingTest(unittest.TestCase):
    """The predicate, beat by beat, on a clock the test owns."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        self.now = [1000.0]
        self.journal = HostJournal(self.store, "metal-0")

    def waiting(self, names) -> NamesWait:
        return NamesWait(self.store, EXP, names, self.journal, lambda: self.now[0])

    def test_a_writer_that_fails_mid_wait_is_refused_at_the_next_ended_check(self) -> None:
        writer = a_fit_run(self.store, "fit-1", planned=2, done=0)
        self.store.promise_named(EXP, ["m0"], writer)
        waiting = self.waiting(["m0"])
        self.assertFalse(waiting.all_present())        # first beat: asked, still work
        self.store.append_fleet_event({"event": "failed", "run_id": "fit-1"})
        self.now[0] += ENDED_CHECK_EVERY_S / 2
        self.assertFalse(waiting.all_present())        # between checks: presence only
        self.now[0] += ENDED_CHECK_EVERY_S / 2
        with self.assertRaises(NamedAdapterOrphaned):
            waiting.all_present()

    def test_a_long_wait_says_so_every_ten_minutes_and_not_between(self) -> None:
        writer = a_fit_run(self.store, "fit-1", planned=2, done=0)
        self.store.promise_named(EXP, ["m0"], writer)
        waiting = self.waiting(["m0", "m1"])
        said = io.StringIO()
        with contextlib.redirect_stdout(said):
            self.assertFalse(waiting.all_present())
            self.now[0] += NOTE_EVERY_S - 1
            self.assertFalse(waiting.all_present())
            self.assertEqual(said.getvalue(), "")
            self.now[0] += 1
            self.assertFalse(waiting.all_present())
            self.assertFalse(waiting.all_present())    # the same minute: said once
        self.assertEqual(said.getvalue().count("[names]"), 1)
        self.assertIn("'m0': 'promised'", said.getvalue())
        self.assertIn("'m1': 'unknown'", said.getvalue())
        notes = [e for e in self.store.read_host_log("metal-0") if e["event"] == "names-wait"]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["missing"], {"m0": "promised", "m1": "unknown"})
        self.assertEqual(notes[0]["waited_s"], NOTE_EVERY_S)

    def test_a_name_that_arrives_leaves_the_wait(self) -> None:
        waiting = self.waiting(["m0", "m1"])
        self.assertFalse(waiting.all_present())
        self.store.write_named(EXP, "m0", b"payload", {})
        self.assertFalse(waiting.all_present())
        self.assertEqual(waiting.missing, ("m1",))
        self.store.write_named(EXP, "m1", b"payload", {})
        self.assertTrue(waiting.all_present())


# ---------------------------------------------------------------------------
# the roles: a whole run on fakes, with residents that record what they are handed
# ---------------------------------------------------------------------------

class LibraryEngine(FakeEngine):
    """A FakeEngine with the Engine verb ADR 0019 adds, recording its calls."""

    def __init__(self, calls: list) -> None:
        super().__init__()
        self.calls = calls

    def add_library(self, name: str, payload: bytes) -> None:
        self.calls.append(("add_library", name, payload))

    async def sample_tokens(self, *args, **kwargs):
        if not self.calls or self.calls[-1][0] != "sample":
            self.calls.append(("sample",))
        async for event in super().sample_tokens(*args, **kwargs):
            yield event


class LibraryLearner(FakeLearner):
    """A FakeLearner recording the two Learner verbs this stream calls — and
    still running them, because the fake refuses a row routed under a library
    nobody loaded, which is the Trainer's ordering under test."""

    def __init__(self, calls: list) -> None:
        super().__init__()
        self.calls = calls

    def load_set(self, tenant, entry, route, payload) -> None:
        self.calls.append(("load_set", entry, route, payload))
        super().load_set(tenant, entry, route, payload)

    def drop_set(self, tenant, entry, route) -> None:
        self.calls.append(("drop_set", entry, route))
        super().drop_set(tenant, entry, route)

    def forward_backward(self, tenant, batch):
        if self.calls[-1:] != [("forward_backward",)]:
            self.calls.append(("forward_backward",))
        return super().forward_backward(tenant, batch)

    def optim_step(self, tenant) -> None:
        self.calls.append(("optim_step",))
        super().optim_step(tenant)


def library_run(store: LocalStore, roles: list[str | None]):
    """A two-group GRPO run over a dream_bank, one wave per entry of `roles`:
    wave u samples and trains under `roles[u-1]` (None = the default role,
    trained by WaveRef — a plan that mentions no library at all)."""
    store.cas_put(arith_task_bytes(TRAIN[1], TRAIN[2], TRAIN[0]))
    picked = task_ids(TRAIN)[:2]
    rollout, train = [], []
    for u, role in enumerate(roles, start=1):
        rollout.append(WavePlan(tuple(
            GroupPlan(task, tuple(Sample(task, ENV, role or "train") for _ in range(2)))
            for task in picked)))
        train.append(WaveRef(f"self://rollouts/{u}") if role is None else WavePlan(tuple(
            GroupPlan(task, tuple(Replay(f"self://rollouts/{u}#{2 * g + i}", role)
                                  for i in range(2)))
            for g, task in enumerate(picked))))
    plans = Plans(train=store.cas_put(encode(RunPlan(tuple(train)))),
                  rollout=store.cas_put(encode(RunPlan(tuple(rollout)))))
    return arith_spec(
        cas_uri(arith_task_bytes(TRAIN[1], TRAIN[2], TRAIN[0])), plans=plans,
        policy=PolicySpec(base="Qwen/Qwen3-0.6B", bank={
            "pi": dream_bank("layers.0-3.self_attn.q_proj", r=4, memories=1)}))


class RolesHandLibrariesTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)
        self.engine_calls: list = []
        self.learner_calls: list = []

    def run_it(self, spec, *, subdir: str | None = EXP, store=None, beside=None):
        async def scenario():
            running = run_experiment_async(
                spec, SCHEMA, store or self.store,
                {"main": LibraryEngine(self.engine_calls)},
                LibraryLearner(self.learner_calls), subdir=subdir,
                checkpointing=Checkpointing(every=1))
            if beside is None:
                return await running
            report, _ = await asyncio.gather(running, beside())
            return report
        return go(scenario())

    def test_the_generator_hands_the_pool_each_wave_s_libraries_before_sampling(self) -> None:
        self.store.write_named(EXP, "mem/a", b"A-bytes", {})
        self.store.write_named(EXP, "mem/b", b"B-bytes", {})
        report = self.run_it(library_run(self.store, ["lib:mem/a+dreamer", "lib:mem/b"]))
        self.assertEqual(report.completed, 2)
        self.assertEqual(self.engine_calls, [
            ("add_library", "mem/a", b"A-bytes"), ("sample",),
            ("add_library", "mem/b", b"B-bytes"), ("sample",)])

    def test_the_trainer_loads_before_the_step_and_drops_what_the_next_wave_does_not_name(self) -> None:
        self.store.write_named(EXP, "mem/a", b"A-bytes", {})
        self.store.write_named(EXP, "mem/b", b"B-bytes", {})
        roles = ["lib:mem/a+dreamer", "lib:mem/a+dreamer", "lib:mem/b"]
        report = self.run_it(library_run(self.store, roles))
        self.assertEqual(report.completed, 3)
        step = [("forward_backward",), ("optim_step",)]
        self.assertEqual(self.learner_calls, [
            ("load_set", "pi", "lib:mem/a", b"A-bytes"), *step,   # kept: wave 2 names it
            *step, ("drop_set", "pi", "lib:mem/a"),
            ("load_set", "pi", "lib:mem/b", b"B-bytes"), *step,
            ("drop_set", "pi", "lib:mem/b")])                     # the plan's end keeps nothing
        # the wave on the store records the route it trained under
        run = self.store.open_run(report.run_id, subdir=EXP, create=False)
        self.assertEqual(run.read_wave(3)[0]["turns"][0]["turn_extras"]["route"], "lib:mem/b")

    def test_a_run_waits_for_a_name_a_fit_run_is_still_producing(self) -> None:
        asked = threading.Event()

        class Watched(LocalStore):
            def named_meta(self, subdir, name):
                found = super().named_meta(subdir, name)
                if found is None:
                    asked.set()
                return found

        store = Watched(self.store.root)
        writer = a_fit_run(store, "fit-1", planned=1, done=0)
        store.promise_named(EXP, ["mem/a"], writer)

        async def the_fit_run():
            await asyncio.to_thread(asked.wait, 30)     # the Generator is waiting by now
            self.assertEqual(self.engine_calls, [])      # and has sampled nothing
            store.write_named(EXP, "mem/a", b"A-bytes", {})

        report = self.run_it(library_run(store, ["lib:mem/a+dreamer"]), store=store,
                             beside=the_fit_run)
        self.assertEqual(report.completed, 1)
        self.assertEqual(self.engine_calls[0], ("add_library", "mem/a", b"A-bytes"))
        self.assertEqual(self.learner_calls[0], ("load_set", "pi", "lib:mem/a", b"A-bytes"))

    def test_an_orphaned_name_refuses_the_run_and_nothing_starts_from_init(self) -> None:
        writer = a_fit_run(self.store, "fit-1", planned=1, done=1)
        self.store.promise_named(EXP, ["mem/a"], writer)
        with self.assertRaises(NamedAdapterOrphaned):
            self.run_it(library_run(self.store, ["lib:mem/a+dreamer"]))
        self.assertEqual(self.engine_calls, [])
        self.assertEqual(self.learner_calls, [])

    def test_a_misspelt_name_refuses_the_run_at_its_roles_birth(self) -> None:
        with self.assertRaisesRegex(PlanError, "wave 1"):
            self.run_it(library_run(self.store, ["lib:mem//a"]))
        self.assertEqual(self.engine_calls, [])

    def test_a_run_filed_at_the_root_has_nowhere_to_keep_names(self) -> None:
        with self.assertRaisesRegex(PlanError, "store's root"):
            self.run_it(library_run(self.store, ["lib:mem/a"]), subdir=None)

    def test_a_plan_that_names_no_library_takes_none_of_these_steps(self) -> None:
        report = self.run_it(library_run(self.store, [None, "dreamer"]))
        self.assertEqual(report.completed, 2)
        self.assertEqual({c[0] for c in self.engine_calls}, {"sample"})
        self.assertEqual({c[0] for c in self.learner_calls}, {"forward_backward", "optim_step"})
        self.assertFalse(self.store.path_of(f"runs/{EXP}/names").exists())

    def test_a_run_with_no_library_is_the_same_bytes_with_or_without_the_verbs(self) -> None:
        """The residents of today — no add_library, no load_set — run a plan
        that names no library to the same run directory."""
        with tempfile.TemporaryDirectory() as other:
            plain_store = LocalStore(other)
            spec = library_run(plain_store, [None, "dreamer"])
            go(run_experiment_async(spec, SCHEMA, plain_store, {"main": FakeEngine()},
                                    FakeLearner(), subdir=EXP,
                                    checkpointing=Checkpointing(every=1)))
            self.run_it(library_run(self.store, [None, "dreamer"]))
            self.assertEqual(tree_bytes(plain_store), tree_bytes(self.store))


def tree_bytes(store: LocalStore) -> dict[str, bytes]:
    return {key: store._read(key) for key in store._list("runs/")}


if __name__ == "__main__":
    unittest.main()
