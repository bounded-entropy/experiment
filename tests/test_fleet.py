"""The Fleet (rlstack.runner.fleet): the join/carve/acquire ladder over
atomic-partition hosts (#43).

Claims under test: demands are read off the spec as capability (base, shape),
never placement; a concurrent group carves PER-CAPABILITY hosts while a sleep
group carves ONE alternating host; a second experiment wanting an existing
capability JOINS instead of carving (the tp-N worker is contactable); carving
draws from residual only, never double-books, and is journaled; acquire is
refused as a human's call; and fleet.submit runs a whole experiment across
hosts — the runner beside the learner, every other pool over the wire.

Plus #49: a carve stamps the Metal's KIND onto the partition it births, and
fraction_for_gb is the one place a hint written in GB becomes the fraction of
one device that a partition actually owns.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace

from common import arith_spec, arith_store
from rlstack import (
    Acquire, Carve, FakeEngine, FakeLearner, Fleet, FleetError, GpuConfig,
    GpuGroup, Join, Metal, Regime, Seeds, demands_of, fake_qwen_schema,
    fraction_for_gb, gpus, learner, pool,
)

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def go(coro):
    return asyncio.run(coro)


def fake_engine_factory(regime: Regime) -> FakeEngine:
    return FakeEngine(base=regime.base, tp=regime.shape)


def fake_learner_factory(regime: Regime) -> FakeLearner:
    return FakeLearner(fsdp=regime.shape)


class FleetTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)

    def fleet(self, devices: int = 4) -> Fleet:
        return Fleet((Metal("node-a", "L4", devices),), store=self.store,
                     engine_factory=fake_engine_factory,
                     learner_factory=fake_learner_factory)

    def judged_spec(self, **overrides):
        """The multi-host shape: a judge pool on a BIGGER base at tp=2 —
        the teacher/judge worker a later experiment should be able to
        contact on its own."""
        base = arith_spec(self.train, self.heldout, **overrides)
        return replace(
            base,
            algo=replace(base.algo, post=("llm_judge", "grpo_advantage")),
            gpu_config=GpuConfig(groups=(
                GpuGroup(gpus(n=1), (pool("main"),)),
                GpuGroup(gpus(n=2), (pool("judge", base="Qwen/Qwen3-32B",
                                          tp=2),)),
                GpuGroup(gpus(n=1), (learner(),)),
            )))

    # ---- demands ------------------------------------------------------------

    def test_demands_read_capability_off_the_spec(self) -> None:
        demands = demands_of(self.judged_spec())
        self.assertEqual(
            [(d.pool, d.kind, d.base, d.shape) for d in demands],
            [("main", "inference", "Qwen/Qwen3-0.6B", 1),
             ("judge", "inference", "Qwen/Qwen3-32B", 2),
             (None, "training", "Qwen/Qwen3-0.6B", 1)])

    # ---- the ladder ---------------------------------------------------------

    def test_concurrent_members_carve_per_capability_hosts(self) -> None:
        fleet = self.fleet(devices=4)
        plan = fleet.place(self.judged_spec())
        self.assertFalse(plan.needs_human)
        self.assertTrue(all(isinstance(s, Carve) for s in plan.steps))
        self.assertEqual([s.devices for s in plan.steps],
                         [(0,), (1, 2), (3,)])
        placement = fleet.apply(plan)
        self.assertEqual(len(fleet.hosts), 3)         # one host per capability
        self.assertIsNot(placement["judge"], placement[None])

    def test_an_existing_capability_is_joined_not_recarved(self) -> None:
        """The scenario the granularity exists for: a new experiment that
        wants the tp-2 judge worker contacts that host — no new metal."""
        fleet = self.fleet(devices=4)
        fleet.apply(fleet.place(self.judged_spec()))
        again = fleet.place(self.judged_spec(seeds=Seeds(master=99)))
        self.assertTrue(all(isinstance(s, Join) for s in again.steps))
        self.assertEqual(len(fleet.hosts), 3)         # nothing new carved
        self.assertEqual(fleet.residual("node-a"), [0.0, 0.0, 0.0, 0.0])

    def test_a_sleep_group_carves_one_alternating_host(self) -> None:
        fleet = self.fleet(devices=1)
        spec = arith_spec(self.train, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"), learner()), sharing="sleep"),)))
        plan = fleet.place(spec)
        self.assertEqual(len(plan.steps), 1)
        self.assertEqual([r.kind for r in plan.steps[0].regimes],
                         ["inference", "training"])
        placement = fleet.apply(plan)
        host = placement["main"]
        self.assertIs(placement[None], host)          # one host, worn in turns
        self.assertEqual(list(host.arbiter.residency()), [f"host:{host.name}"])

    def test_acquire_is_a_humans_call(self) -> None:
        fleet = self.fleet(devices=1)
        plan = fleet.place(self.judged_spec())
        self.assertTrue(plan.needs_human)
        self.assertTrue(any(isinstance(s, Acquire) for s in plan.steps))
        with self.assertRaises(FleetError) as caught:
            go(fleet.submit(self.judged_spec(), SCHEMA))
        self.assertIn("human", str(caught.exception))
        self.assertEqual(fleet.hosts, {})             # refused BEFORE building

    def test_carving_is_journaled(self) -> None:
        fleet = self.fleet(devices=4)
        fleet.apply(fleet.place(self.judged_spec()))
        events = self.store.read_fleet_log()
        self.assertEqual([e["event"] for e in events],
                         ["carve", "carve", "carve"])
        self.assertEqual(events[1]["devices"], [1, 2])
        self.assertEqual(events[1]["regimes"][0]["base"], "Qwen/Qwen3-32B")

    def test_a_carve_stamps_the_metals_gpu_onto_the_partition(self) -> None:
        """#49: the kind of GPU is registered once, on the Metal, and rides
        the carve down into every partition born from it — the fraction says
        how much, the kind says of what."""
        fleet = Fleet((Metal("node-h", "H100", 2, vram_gb=80.0),),
                      store=self.store, engine_factory=fake_engine_factory,
                      learner_factory=fake_learner_factory)
        fleet.apply(fleet.place(arith_spec(self.train)))
        for host in fleet.hosts.values():
            self.assertEqual(host.partition.gpu, "H100")
        carve = [e for e in self.store.read_fleet_log()
                 if e["event"] == "carve"][0]
        self.assertEqual(carve["gpu"], "H100")

    # ---- the VRAM hint ------------------------------------------------------

    def test_a_carve_hint_in_gb_becomes_a_fraction_of_one_device(self) -> None:
        """The one conversion (#49): GB is what a human sizes a model in, a
        fraction is what a partition owns — and the same 20 GB is most of an
        L4 and a quarter of an H100."""
        self.assertEqual(fraction_for_gb(12.0, Metal("node-a", "L4", 4)), 0.5)
        self.assertAlmostEqual(
            fraction_for_gb(20.0, Metal("node-a", "L4", 4)), 0.8333, places=4)
        self.assertEqual(
            fraction_for_gb(20.0, Metal("node-h", "H100", 8, vram_gb=80.0)),
            0.25)
        self.assertEqual(fraction_for_gb(24.0, Metal("node-a", "L4", 4)), 1.0)

    def test_more_vram_than_one_device_holds_is_the_acquire_rung(self) -> None:
        """A fraction cannot exceed a device, so the conversion refuses
        rather than clamping: 40 GB on an L4 is bigger metal, a human's."""
        with self.assertRaises(FleetError) as caught:
            fraction_for_gb(40.0, Metal("node-a", "L4", 4))
        self.assertIn("acquire", str(caught.exception))
        self.assertIn("L4", str(caught.exception))

    def test_a_gb_hint_sizes_a_carve_like_any_fraction(self) -> None:
        """Converted at the fleet, a GB hint is just the declared fraction:
        two 12 GB tenants fit one L4 device, and the residual proves it.
        (PoolMember.fraction remains the declared unit — #49.)"""
        fleet = self.fleet(devices=1)
        half = fraction_for_gb(12.0, fleet.metal["node-a"])
        spec = arith_spec(self.train, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main", fraction=half),
                                 learner(fraction=half)),
                     sharing="sleep"),)))
        fleet.apply(fleet.place(spec))
        host = next(iter(fleet.hosts.values()))
        self.assertEqual(host.partition.memory, 0.5)
        self.assertEqual(fleet.residual("node-a"), [0.5])

    # ---- submit -------------------------------------------------------------

    def test_submit_runs_the_experiment_across_hosts(self) -> None:
        fleet = self.fleet(devices=4)
        spec = self.judged_spec()
        report = go(fleet.submit(spec, SCHEMA, store=self.store))
        self.assertEqual(report.updates_completed, 4)

        run = self.store.open_run(report.run_id)
        self.assertEqual(len(run.read_ledger()), 4)
        # the placement was journaled under the run's identity
        placed = [e for e in self.store.read_fleet_log()
                  if e["event"] == "place"]
        self.assertEqual(placed[0]["run_id"], report.run_id)
        self.assertEqual([s["rung"] for s in placed[0]["steps"]],
                         ["carve", "carve", "carve"])
        # the runner ran beside the learner; main and judge crossed the wire
        learner_host = next(h for h in fleet.hosts.values()
                            if any(r.kind == "training" for r in h.regimes))
        attach = [e for e in self.store.read_host_log(learner_host.name)
                  if e["event"] == "attach"][0]
        self.assertEqual(attach["remotes"], ["judge", "main"])
        self.assertEqual(attach["run_id"], report.run_id)

    def test_submit_runs_on_an_alternating_carved_host(self) -> None:
        """The sleep demand lands on the carved host's OWN alternation
        group (the spec's declaration defers to the metal's birth truth) and
        the run alternates to completion."""
        fleet = self.fleet(devices=1)
        spec = arith_spec(self.train, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"), learner()), sharing="sleep"),)))
        report = go(fleet.submit(spec, SCHEMA, store=self.store))
        self.assertEqual(report.updates_completed, 4)
        host = next(iter(fleet.hosts.values()))
        self.assertGreater(len(host.arbiter.switches), 1)   # it alternated


if __name__ == "__main__":
    unittest.main()
