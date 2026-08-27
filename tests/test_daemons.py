"""The blackboard runner: signals, the arbiter, and the daemons' conditions.

The properties under test are the ones the design claims: daemons synchronize
ONLY through the store; the lag buffer bounds how far generation runs ahead;
sleep colocation is an exclusive GROUP on the arbiter whose wake/evict hooks
fire only on actual residency switches — while same-resident work overlaps
freely; and every recorded behavior policy is a committed (or initial) bundle.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace

from common import arith_spec, arith_store
from rlstack import (
    FakeEngine, FakeLearner, GpuArbiter, GpuConfig, GpuGroup, RunSignals,
    Schedule, fake_qwen_schema, gpus, learner, pool, run_experiment,
)

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def go(coro):
    return asyncio.run(coro)


class SignalsTest(unittest.TestCase):
    def test_wait_for_returns_the_predicate_value(self) -> None:
        async def scenario():
            signals = RunSignals()
            box: list[int] = []

            async def writer():
                await asyncio.sleep(0.01)
                box.append(42)
                await signals.notify()

            asyncio.get_running_loop().create_task(writer())
            return await signals.wait_for(lambda: box[0] if box else None)

        self.assertEqual(go(scenario()), 42)

    def test_ready_predicate_never_waits(self) -> None:
        self.assertEqual(go(RunSignals().wait_for(lambda: "now")), "now")


class ArbiterTest(unittest.TestCase):
    """The physical half: admission semantics of one exclusive group."""

    def two_residents(self, **kwargs):
        arbiter = GpuArbiter(**kwargs)
        engine, trainer = object(), object()
        log: list[str] = []
        arbiter.attach(engine, label="engine:main", group="sleep:0",
                       wake=_note(log, "engine:wake"),
                       evict=_note(log, "engine:evict"))
        arbiter.attach(trainer, label="learner", group="sleep:0",
                       wake=_note(log, "learner:wake"),
                       evict=_note(log, "learner:evict"))
        return arbiter, engine, trainer, log

    def test_sticky_resident_fires_hooks_only_on_switch(self) -> None:
        async def scenario():
            arbiter, engine, trainer, log = self.two_residents()
            async with arbiter.admit(engine):
                pass
            async with arbiter.admit(engine):   # sticky: no hook churn
                pass
            async with arbiter.admit(trainer):  # switch: evict then wake
                pass
            return log

        self.assertEqual(go(scenario()),
                         ["engine:wake", "engine:evict", "learner:wake"])

    def test_exclusive_residents_never_overlap(self) -> None:
        async def scenario():
            arbiter, engine, trainer, _ = self.two_residents()
            active: list[object] = []

            async def worker(resident):
                for _ in range(3):
                    async with arbiter.admit(resident):
                        active.append(resident)
                        await asyncio.sleep(0)      # yield while admitted
                        self.assertEqual(set(active), {resident})
                        active.remove(resident)

            async with asyncio.TaskGroup() as group:
                group.create_task(worker(engine))
                group.create_task(worker(trainer))
            return arbiter.switches

        switches = go(scenario())
        self.assertGreaterEqual(len(switches), 2)   # both residents ran

    def test_same_resident_work_overlaps(self) -> None:
        """Alternation is about MEMORY, not mutual exclusion: two admits of
        one resident run concurrently (the old ExclusiveLease serialized
        them — generation and judge traffic on one engine should batch)."""
        async def scenario():
            arbiter, engine, _, _ = self.two_residents()
            inside: list[int] = []
            peak: list[int] = []

            async def worker():
                async with arbiter.admit(engine):
                    inside.append(1)
                    await asyncio.sleep(0.01)
                    peak.append(len(inside))
                    inside.pop()

            async with asyncio.TaskGroup() as group:
                group.create_task(worker())
                group.create_task(worker())
            return max(peak)

        self.assertEqual(go(scenario()), 2)

    def test_switch_waits_for_inflight_to_drain(self) -> None:
        async def scenario():
            arbiter, engine, trainer, _ = self.two_residents()
            order: list[str] = []

            async def engine_work():
                async with arbiter.admit(engine):
                    await asyncio.sleep(0.02)
                    order.append("engine-done")

            async def trainer_work():
                await asyncio.sleep(0.005)      # arrive while engine works
                async with arbiter.admit(trainer):
                    order.append("trainer-in")

            async with asyncio.TaskGroup() as group:
                group.create_task(engine_work())
                group.create_task(trainer_work())
            return order

        self.assertEqual(go(scenario()), ["engine-done", "trainer-in"])

    def test_quantum_defers_the_switch(self) -> None:
        """Hysteresis on a fake clock: a switch may not happen again until
        `quantum` has elapsed since the last one."""
        async def scenario():
            now = [0.0]
            arbiter = GpuArbiter(quantum=10.0, clock=lambda: now[0])
            engine, trainer = object(), object()
            arbiter.attach(engine, label="engine:main", group="sleep:0")
            arbiter.attach(trainer, label="learner", group="sleep:0")

            async with arbiter.admit(engine):
                pass                             # switch #1 at t=0

            entered: list[str] = []

            async def trainer_work():
                async with arbiter.admit(trainer):
                    entered.append("trainer")

            task = asyncio.get_running_loop().create_task(trainer_work())
            await asyncio.sleep(0.08)            # blocked: quantum not elapsed
            self.assertEqual(entered, [])
            now[0] = 11.0                        # clock passes the quantum
            await task
            return entered

        self.assertEqual(go(scenario()), ["trainer"])

    def test_max_wait_forces_a_handoff(self) -> None:
        """Aging: a starving waiter stops the resident from being fed, so its
        in-flight work drains and the waiter enters."""
        async def scenario():
            now = [0.0]
            arbiter = GpuArbiter(max_wait=5.0, clock=lambda: now[0])
            engine, trainer = object(), object()
            arbiter.attach(engine, label="engine:main", group="sleep:0")
            arbiter.attach(trainer, label="learner", group="sleep:0")
            order: list[str] = []
            feeding = [True]

            async def engine_stream():
                while feeding[0]:                # would stream forever
                    async with arbiter.admit(engine):
                        await asyncio.sleep(0.005)
                order.append("engine-stopped")

            async def trainer_work():
                await asyncio.sleep(0.01)
                async with arbiter.admit(trainer):
                    order.append("trainer-in")
                feeding[0] = False

            async def clock_marches_on():
                # the wait begins at t=0; only THEN does time pass it by
                await asyncio.sleep(0.03)
                now[0] = 6.0                     # trainer now starved > max_wait

            async with asyncio.TaskGroup() as group:
                group.create_task(engine_stream())
                group.create_task(trainer_work())
                group.create_task(clock_marches_on())
            return order

        self.assertEqual(go(scenario()), ["trainer-in", "engine-stopped"])

    def test_admit_all_refuses_two_of_one_group(self) -> None:
        async def scenario():
            arbiter, engine, trainer, _ = self.two_residents()
            with self.assertRaises(ValueError):
                async with arbiter.admit_all((engine, trainer)):
                    pass

        go(scenario())

    def test_admit_all_of_nothing_is_a_no_op(self) -> None:
        async def scenario():
            arbiter = GpuArbiter()
            async with arbiter.admit_all(()):
                return "ran"

        self.assertEqual(go(scenario()), "ran")

    def test_attach_is_idempotent_and_group_change_is_loud(self) -> None:
        arbiter = GpuArbiter()
        engine = object()
        arbiter.attach(engine, label="engine:main", group=None, fraction=0.45)
        arbiter.attach(engine, label="engine:judge", group=None)   # same object
        self.assertEqual(arbiter.declared_load(), 0.45)
        with self.assertRaises(ValueError):
            arbiter.attach(engine, label="engine:main", group="sleep:0")


def _note(log: list[str], entry: str):
    async def hook() -> None:
        log.append(entry)
    return hook


class BlackboardRunTest(unittest.TestCase):
    """Integration: the daemon runner on fake metal, beyond the default knobs
    (the default-knob behavior is pinned byte-for-byte by test_loop and
    test_resume, which this refactor kept green unchanged)."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)

    def run_spec(self, spec):
        report = run_experiment(spec, SCHEMA, self.store, FakeEngine(),
                                FakeLearner())
        return report, self.store.open_run(report.run_id)

    def spec_with_lag(self, lag: int):
        base = arith_spec(self.train)
        return replace(base, algo=replace(
            base.algo, schedule=Schedule(group_size=2, trajectories_per_wave=4,
                                         n_updates=4, microbatch_tokens=64,
                                         max_policy_lag=lag)))

    def test_lag_buffer_completes_and_behavior_is_always_committed(self) -> None:
        """B=1: generation may run ahead, so which version served each wave is
        scheduling — but it must ALWAYS be a bundle the ledger (or Phase 1)
        published, and within the lag bound."""
        report, run = self.run_spec(self.spec_with_lag(1))
        entries = run.read_ledger()
        self.assertEqual([e["update"] for e in entries], [1, 2, 3, 4])

        published = {e["bundle_id"]: e["update"] for e in entries}
        for update in (1, 2, 3, 4):
            for row in run.read_wave(update):
                for turn in row["turns"]:
                    served = turn["bundle_id"]
                    if served in published:
                        lag = update - 1 - published[served]
                        self.assertGreaterEqual(lag, 0)
                        self.assertLessEqual(lag, 1)
                    else:
                        # only the Phase-1 initial bundle is not in the ledger
                        self.assertLessEqual(update, 2)

    def test_sleep_colocation_runs_green(self) -> None:
        spec = arith_spec(self.train, self.heldout, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"), learner()), sharing="sleep"),)))
        report, run = self.run_spec(spec)
        self.assertEqual(len(run.read_ledger()), 4)
        self.assertTrue(run.has_eval(2) and run.has_eval(4))


if __name__ == "__main__":
    unittest.main()
