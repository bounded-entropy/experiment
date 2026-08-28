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

Plus #52, the three fleet findings of the mass-partition campaign: carved
names are unique (#51a), registration refuses to replace a live host (#51a),
a name is a journal path segment the observer can read back (#51b), and the
factories are paid the Partition they are realizing (#51c).
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace

from common import arith_spec, arith_store
from rlstack import (
    Acquire, Carve, FakeEngine, FakeLearner, Fleet, FleetError, GpuConfig,
    GpuGroup, Host, Join, Metal, Partition, Regime, Seeds, demands_of,
    fake_qwen_schema, fraction_for_gb, gpus, learner, pool,
)
from rlstack.observe import render_hosts

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def go(coro):
    return asyncio.run(coro)


def fake_engine_factory(regime: Regime, partition: Partition) -> FakeEngine:
    return FakeEngine(base=regime.base, tp=regime.shape)


def fake_learner_factory(regime: Regime, partition: Partition) -> FakeLearner:
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
            [(d.pool, d.capability, d.base, d.shape) for d in demands],
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
        self.assertEqual([r.capability for r in plan.steps[0].regimes],
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

    # ---- #52: names, registration, the factory contract ---------------------

    def partitioned_spec(self, base: str, main: float, train: float,
                         master: int):
        """One tenant that declares BOTH its fractions — the sub-GPU shape the
        mass-partition campaign ran (#51)."""
        spec = arith_spec(self.train, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main", fraction=main),
                                 learner(fraction=train))),)),
            seeds=Seeds(master=master))
        return replace(spec, policy=replace(spec.policy, base=base))

    def test_two_carves_differing_only_by_base_are_two_hosts(self) -> None:
        """#51a, the campaign's repro. A regime is named by capability and shape,
        never by base, so a second base's learner used to carve the SAME name
        and REPLACE a live host: its metal stayed resident and its tenants
        stayed bound while its 0.20 silently returned to the residual — an
        automatic path into overcommit. The carve ordinal keeps the names
        apart; the residual stays honest."""
        fleet = self.fleet(devices=1)
        fleet.apply(fleet.place(
            self.partitioned_spec("Qwen/Qwen3-0.6B", 0.30, 0.20, 1)))
        self.assertAlmostEqual(fleet.residual("node-a")[0], 0.50)
        trainer = next(h for h in fleet.hosts.values()
                       if any(r.capability == "training" for r in h.regimes))

        fleet.apply(fleet.place(
            self.partitioned_spec("Qwen/Qwen3-0.6B-Base", 0.15, 0.125, 2)))
        self.assertEqual(len(fleet.hosts), 4)          # four distinct hosts
        self.assertIs(fleet.hosts[trainer.name], trainer)   # nothing replaced
        self.assertEqual(
            sorted(r.base for h in fleet.hosts.values() for r in h.regimes
                   if r.capability == "training"),
            ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B-Base"])
        self.assertAlmostEqual(fleet.residual("node-a")[0], 0.225)
        names = [e["host"] for e in self.store.read_fleet_log()
                 if e["event"] == "carve"]
        self.assertEqual(len(set(names)), 4)

    def test_registering_a_taken_name_is_refused_never_replaced(self) -> None:
        """Uniqueness is by construction; this is the rule that keeps it so.
        A replaced host would keep its metal and leave the residual's sum —
        so registration raises rather than overwriting (#51a)."""
        fleet = self.fleet(devices=1)
        fleet.apply(fleet.place(
            self.partitioned_spec("Qwen/Qwen3-0.6B", 0.30, 0.20, 1)))
        taken = sorted(fleet.hosts)[0]
        with self.assertRaises(FleetError) as caught:
            fleet.register(Host(taken, engines=(FakeEngine(),), learner=None,
                                store=self.store))
        self.assertIn("already registered", str(caught.exception))
        self.assertAlmostEqual(fleet.residual("node-a")[0], 0.50)

        pre = Host("pre-carved", engines=(FakeEngine(),), learner=None,
                   store=self.store)
        with self.assertRaises(FleetError):             # also at construction
            Fleet((Metal("node-a", "L4", 1),), store=self.store,
                  engine_factory=fake_engine_factory,
                  learner_factory=fake_learner_factory, hosts=(pre, pre))

    def test_a_carved_host_is_visible_to_the_observer(self) -> None:
        """#51b, inverted. Carved names once contained "/", so every carved
        partition journaled one directory deeper than list_hosts() looks and
        a four-partition campaign rendered as ONE phantom host. A name is one
        journal path segment now — attested by Host at birth — so a carved
        host round-trips: journal, list, render."""
        fleet = self.fleet(devices=1)
        fleet.apply(fleet.place(
            self.partitioned_spec("Qwen/Qwen3-0.6B", 0.50, 0.25, 1)))
        carved = sorted(fleet.hosts)
        self.assertEqual(self.store.list_hosts(), carved)
        for name in carved:
            self.assertNotIn("/", name)
            ups = [e for e in self.store.read_host_log(name)
                   if e["event"] == "host-up"]
            self.assertEqual(len(ups), 1)
            self.assertEqual(ups[0]["partition"]["metal"], "node-a")
        text = render_hosts([self.store])
        for name in carved:
            self.assertIn(f"host {name}", text)
        self.assertIn("L4 node-a[0] @ 0.50", text)
        self.assertIn("L4 node-a[0] @ 0.25", text)

    def test_the_factories_are_paid_the_partition_they_realize(self) -> None:
        """#51c: the fraction a carve computed is the whole point of a
        sub-GPU host, and the factory is the only thing that can spend it
        (vLLM's gpu_memory_utilization). It arrives as the second argument —
        the SAME Partition the host is then born onto."""
        seen: list[tuple[str, Partition]] = []

        def engine_factory(regime: Regime, partition: Partition) -> FakeEngine:
            seen.append((regime.name, partition))
            return FakeEngine(base=regime.base, tp=regime.shape)

        def learner_factory(regime: Regime,
                            partition: Partition) -> FakeLearner:
            seen.append((regime.name, partition))
            return FakeLearner(fsdp=regime.shape)

        fleet = Fleet((Metal("node-a", "L4", 1),), store=self.store,
                      engine_factory=engine_factory,
                      learner_factory=learner_factory)
        fleet.apply(fleet.place(
            self.partitioned_spec("Qwen/Qwen3-0.6B", 0.30, 0.20, 1)))
        self.assertEqual([(name, p.memory, p.gpu, p.devices)
                          for name, p in seen],
                         [("main-tp1", 0.30, "L4", (0,)),
                          ("learner-fsdp1", 0.20, "L4", (0,))])
        for name, partition in seen:
            host = next(h for h in fleet.hosts.values()
                        if any(r.name == name for r in h.regimes))
            self.assertIs(host.partition, partition)

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
                            if any(r.capability == "training" for r in h.regimes))
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
