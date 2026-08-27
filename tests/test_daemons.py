"""The blackboard runner: signals, leases, and the daemons' await conditions.

The properties under test are the ones the redesign claims: daemons
synchronize ONLY through the store; the lag buffer bounds how far generation
runs ahead; sleep colocation is an exclusive lease whose wake/evict hooks
fire only on actual residency switches; and every recorded behavior policy is
a committed (or initial) bundle.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace

from common import arith_spec, arith_store
from rlstack import (
    ENGINE, LEARNER, ExclusiveLease, FakeEngine, FakeLearner, GpuConfig,
    GpuGroup, OpenLease, RunSignals, Schedule, engines, fake_qwen_schema, gpus,
    learner, leases_for, run_experiment,
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


class ExclusiveLeaseTest(unittest.TestCase):
    def test_sticky_resident_fires_hooks_only_on_switch(self) -> None:
        async def scenario():
            lease = ExclusiveLease()
            log: list[str] = []
            lease.on(ENGINE, wake=_note(log, "engine:wake"),
                     evict=_note(log, "engine:evict"))
            lease.on(LEARNER, wake=_note(log, "learner:wake"),
                     evict=_note(log, "learner:evict"))
            async with lease.held(ENGINE):
                pass
            async with lease.held(ENGINE):     # sticky: no hook churn
                pass
            async with lease.held(LEARNER):    # switch: evict then wake
                pass
            return log

        self.assertEqual(go(scenario()),
                         ["engine:wake", "engine:evict", "learner:wake"])

    def test_holders_alternate_never_overlap(self) -> None:
        async def scenario():
            lease = ExclusiveLease()
            active: list[str] = []

            async def worker(resource: str):
                for _ in range(3):
                    async with lease.held(resource):
                        active.append(resource)
                        await asyncio.sleep(0)     # yield while holding
                        self.assertEqual(active, [resource])
                        active.remove(resource)

            async with asyncio.TaskGroup() as group:
                group.create_task(worker(ENGINE))
                group.create_task(worker(LEARNER))
            return lease.switches

        switches = go(scenario())
        self.assertGreaterEqual(len(switches), 2)   # both resources ran

    def test_leases_for_reads_the_sharing_field(self) -> None:
        sleep_spec = arith_spec("cas://x/t.jsonl", gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (engines("main"), learner()), sharing="sleep"),)))
        leases = leases_for(sleep_spec)
        self.assertIs(leases.for_pool("main"), leases.for_learner())
        self.assertIsInstance(leases.for_pool("main"), ExclusiveLease)

        open_spec = arith_spec("cas://x/t.jsonl")
        self.assertIsInstance(leases_for(open_spec).for_pool("main"), OpenLease)


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
            base.algo, schedule=Schedule(group_size=2, rollouts_per_wave=4,
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
            for row in run.read_rollouts(update):
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
            GpuGroup(gpus(n=1), (engines("main"), learner()), sharing="sleep"),)))
        report, run = self.run_spec(spec)
        self.assertEqual(len(run.read_ledger()), 4)
        self.assertTrue(run.has_eval(2) and run.has_eval(4))


if __name__ == "__main__":
    unittest.main()
