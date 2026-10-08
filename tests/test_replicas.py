"""ADR 0022: a run spans metals — pools own their replicas, the learner
composes dp × fsdp.

The regression wall for the Pool (runner/pool.py) and the widths around it:

    affinity     a seen key returns to its replica; a new key goes to the
                 least-loaded one; a key moves only when its replica dies,
                 and a revived replica never takes its old keys back
    health       a request that meets a dead replica is retried on a
                 survivor; the death starts a revival at once
    delivery     a committed bundle is on every live replica before the
                 ledger line; a revived replica is brought up to every bundle
                 the pool serves, and the wave's libraries, before it takes
                 a request — even when a delivery lands mid-revival
    identity     replicas and dp are resources: run_id, the manifest and
                 every sealed byte are the same at replicas ∈ {1, 3} and
                 dp ∈ {1, 2}; the wire still carries them
    placement    N replicas are N placement units on N distinct listings;
                 losing a replica's host does not strand the run, and the
                 pool re-places it through the desk
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import tempfile
import unittest

from common import arith_spec, arith_store
from test_desk import BASE, DeskFixture, go
from test_resume import snapshot
from rlstack import (
    Bundle, FakeEngine, FakeLearner, HostSpec, Message, Role, SamplingSpec,
    Topology, fake_qwen_schema, learner, pool, run_experiment,
)
from rlstack.runner.campaign import Campaigns, demands_of, frame_for
from rlstack.runner.checkpointing import EVERY_UPDATE, Checkpointing
from rlstack.runner.desk import demand_rows, pools_revive_it
from rlstack.runner.pool import (
    Pool, Replica, Replicas, replica_key, split_replica_key,
)
from rlstack.runner.remote import (
    HostGone, LocalTransport, RemoteDesk, Unreachable, spec_from_json,
)
from rlstack.runner.traffic import EnginePoolClient
from rlstack.spec.canonical import canonical_json, wire_json
from rlstack.spec.validate import validate

SCHEMA = fake_qwen_schema(4, base=BASE)
PROMPT = (Message(Role.USER, "What is 2+2?"),)
BUNDLE = Bundle("bundle:b0", {"pi": 0})


def fakes(count: int) -> list[FakeEngine]:
    engines = [FakeEngine() for _ in range(count)]
    for engine in engines:
        engine.add_bundle(BUNDLE)
    return engines


class DyingEngine(FakeEngine):
    """A replica whose host is gone: every verb is the wire's death."""

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self.error = error or HostGone("no host answers here (decarved)")
        self.asked = 0

    def add_bundle(self, bundle) -> None:
        self.asked += 1
        raise self.error

    def knows_bundle(self, bundle_id: str) -> bool:
        self.asked += 1
        raise self.error

    async def sample_tokens(self, *args, **kwargs):
        self.asked += 1
        raise self.error
        yield  # pragma: no cover — an async generator that never yields


class SpareReviver:
    """Hands out fresh fakes, optionally holding each one at a gate so a
    test can land a delivery while a revival is in flight."""

    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.born: list[FakeEngine] = []
        self.asked: list[tuple[str, int]] = []

    async def recarve(self, pool: str, replica: int) -> FakeEngine:
        self.asked.append((pool, replica))
        if self.gate is not None:
            await self.gate.wait()
        engine = FakeEngine()
        self.born.append(engine)
        return engine


async def sample(pool: Pool, affinity: str, seed: int = 7):
    """One sample request through the pool, the way EnginePoolClient asks."""
    client = EnginePoolClient({pool.name: (pool, BUNDLE)}, SamplingSpec(max_tokens=8),
                              episode_seed=seed, pool_name=pool.name)
    return await client.sample(PROMPT, affinity=affinity)


class AffinityTest(unittest.TestCase):
    """THE ROUTER (ADR 0022, Q2): per request, by key, never by load once a
    key has a home."""

    def test_a_seen_key_returns_to_its_replica(self) -> None:
        pool = Pool.of("main", Replicas(tuple(fakes(3))))
        first = pool.route("episode-a")
        for _ in range(5):
            pool.replicas[first.index].in_flight += 10   # load does not move it
            self.assertIs(pool.route("episode-a"), first)
        self.assertEqual(pool.moves, 0)

    def test_new_keys_go_to_the_least_loaded_replica(self) -> None:
        pool = Pool.of("main", Replicas(tuple(fakes(3))))
        homes = [pool.route(f"k{i}").index for i in range(6)]
        self.assertEqual(homes, [0, 1, 2, 0, 1, 2])     # fewest keys, then index
        pool.replicas[0].in_flight = 3
        pool.replicas[1].in_flight = 1
        self.assertEqual(pool.route("busy-fleet").index, 2)

    def test_a_key_moves_only_when_its_replica_dies_and_never_back(self) -> None:
        async def drive():
            pool = Pool.of("main", Replicas(tuple(fakes(3))))
            home = pool.route("episode-a").index
            pool.found_dead(pool.replicas[home], HostGone("decarved"))
            moved = pool.route("episode-a").index
            self.assertNotEqual(moved, home)
            self.assertEqual(pool.moves, 1)
            pool.replicas[home].live = True              # a revival came back
            self.assertEqual(pool.route("episode-a").index, moved)
            self.assertEqual(pool.moves, 1)
        go(drive())

    def test_one_episode_keeps_one_replica_by_default(self) -> None:
        """The default key is the episode's id: every call of one episode
        lands on one replica, so its prefix stays in one cache."""
        async def drive():
            pool = Pool.of("main", Replicas(tuple(fakes(3))))
            client = EnginePoolClient({"main": (pool, BUNDLE)}, SamplingSpec(max_tokens=4),
                                      episode_seed=11, episode="rollout/1/g/0")
            calls: list[dict] = []
            client._calls = calls
            for _ in range(4):
                await client.sample(PROMPT)
            return calls
        calls = go(drive())
        self.assertEqual(len({call["replica"] for call in calls}), 1)


class HealthTest(unittest.TestCase):
    """THE POOL OWNS ITS REPLICAS' DEATHS (ADR 0022, Q3)."""

    def test_a_request_on_a_dead_replica_is_retried_on_a_survivor(self) -> None:
        async def drive():
            dying = DyingEngine(Unreachable("did not answer within 60s"))
            dying.add_bundle = FakeEngine.add_bundle.__get__(dying)
            live = fakes(2)
            pool = Pool.of("main", Replicas((dying, *live)))
            alone = await sample(Pool.of("main", Replicas((fakes(1)[0],))), "k", seed=3)
            turn = await sample(pool, "k", seed=3)
            return pool, turn, alone
        pool, turn, alone = go(drive())
        self.assertFalse(pool.replicas[0].live)
        self.assertEqual(turn.token_ids, alone.token_ids)       # same seed, same bytes
        self.assertEqual(turn.behavior_logprobs, alone.behavior_logprobs)
        self.assertEqual([e["event"] for e in pool.events], ["replica-dead"])

    def test_a_request_failure_that_is_not_a_death_is_raised(self) -> None:
        """A bundle the engine never saw is the REQUEST's failure: raised,
        and the replica stays live."""
        async def drive():
            pool = Pool.of("main", Replicas((FakeEngine(), *fakes(1))))
            pool.keys["k"] = 0
            with self.assertRaises(RuntimeError):
                await sample(pool, "k")
            return pool
        self.assertTrue(go(drive()).replicas[0].live)

    def test_with_no_reviver_the_last_death_is_raised(self) -> None:
        async def drive():
            pool = Pool.of("main", Replicas((DyingEngine(),)))
            with self.assertRaises(HostGone):
                await sample(pool, "k")
        go(drive())

    def test_a_death_starts_its_revival_and_the_revived_replica_is_current(self) -> None:
        async def drive():
            reviver = SpareReviver()
            pool = Pool.of("main", Replicas((DyingEngine(), *fakes(2)), reviver))
            pool.remember(BUNDLE)
            pool.libraries = {"fact-03": b"named payload"}
            await sample(pool, "k")                          # meets the corpse
            await asyncio.gather(*pool._reviving.values())
            return pool, reviver
        pool, reviver = go(drive())
        self.assertEqual(reviver.asked, [("main", 0)])
        self.assertTrue(pool.replicas[0].live)
        revived = pool.replicas[0].engine
        self.assertIs(revived, reviver.born[0])
        self.assertTrue(revived.knows_bundle(BUNDLE.bundle_id))
        self.assertIn("fact-03", revived.library.log)
        self.assertEqual([e["event"] for e in pool.events],
                         ["replica-dead", "replica-revived"])
        self.assertIn("seconds", pool.events[-1])

    def test_all_replicas_dead_waits_for_the_revival(self) -> None:
        async def drive():
            pool = Pool.of("main", Replicas((DyingEngine(),), SpareReviver()))
            pool.remember(BUNDLE)
            return await sample(pool, "k"), pool
        turn, pool = go(drive())
        self.assertTrue(turn.token_ids)
        self.assertEqual(pool.replicas[0].revivals, 1)


class DeliveryTest(unittest.TestCase):
    """DELIVERY REACHES EVERY LIVE REPLICA BEFORE THE COMMIT (ADR 0022)."""

    def test_deliver_reaches_every_live_replica_and_a_death_revives_at_once(self) -> None:
        async def drive():
            reviver = SpareReviver()
            engines = fakes(2)
            pool = Pool.of("main", Replicas((*engines, DyingEngine()), reviver))
            fresh = Bundle("bundle:b1", {"pi": 1}, {"pi": b"delta"}, {"pi": "fake"})
            await pool.deliver(fresh)
            self.assertEqual(reviver.asked, [("main", 2)])   # started by the deliver
            await asyncio.gather(*pool._reviving.values())
            return pool, engines, reviver, fresh
        pool, engines, reviver, fresh = go(drive())
        for engine in (*engines, reviver.born[0]):
            self.assertTrue(engine.knows_bundle(fresh.bundle_id))

    def test_a_delivery_during_a_revival_reaches_the_revived_replica(self) -> None:
        """The race: the revived engine is brought up to v1 while the
        Trainer delivers v2 to the live replicas. Readmission re-checks, so
        the replica that comes back holds v2 before its first request."""
        async def drive():
            gate = asyncio.Event()
            reviver = SpareReviver(gate)
            pool = Pool.of("main", Replicas((DyingEngine(), *fakes(1)), reviver))
            v1 = Bundle("bundle:v1", {"pi": 1}, {"pi": b"one"}, {"pi": "fake"})
            v2 = Bundle("bundle:v2", {"pi": 2}, {"pi": b"two"}, {"pi": "fake"})
            await pool.deliver(v1)                       # finds replica 0 dead
            gate.set()
            await asyncio.sleep(0)
            await pool.deliver(v2)                       # lands mid-revival
            await asyncio.gather(*pool._reviving.values())
            return pool, reviver, v2
        pool, reviver, v2 = go(drive())
        self.assertTrue(pool.replicas[0].live)
        self.assertTrue(reviver.born[0].knows_bundle(v2.bundle_id))

    def test_every_committed_bundle_was_on_every_replica_before_its_line(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        engines = [FakeEngine() for _ in range(3)]
        seen: list[tuple[int, str]] = []
        for index, engine in enumerate(engines):
            original = engine.add_bundle

            def recording(bundle, original=original, index=index):
                seen.append((index, bundle.bundle_id))
                original(bundle)
            engine.add_bundle = recording
        spec = arith_spec(train, topology=Topology(hosts=(
            HostSpec((pool("main", replicas=3),)), HostSpec((learner(),)))))
        report = run_experiment(spec, SCHEMA, store, {"main": Replicas(tuple(engines))},
                                FakeLearner(), checkpointing=EVERY_UPDATE)
        ledger = store.peek_ledger(report.run_id)
        self.assertEqual(len(ledger), 4)
        for line in ledger:
            for index in range(3):
                self.assertIn((index, line["bundle_id"]), seen)


class IdentityTest(unittest.TestCase):
    """REPLICAS AND DP ARE RESOURCES, NOT SCIENCE (ADR 0022, I3/I5)."""

    def spec(self, train: str, replicas: int, dp: int):
        return arith_spec(train, topology=Topology(hosts=(
            HostSpec((pool("main", replicas=replicas),)),
            HostSpec((learner(dp=dp),)))))

    def test_run_id_and_manifest_do_not_see_the_widths(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, train, _ = arith_store(tmp.name)
        plain = self.spec(train, 1, 1)
        wide = self.spec(train, 3, 2)
        self.assertEqual(canonical_json(plain), canonical_json(wide))
        self.assertNotIn("replicas", canonical_json(wide))
        self.assertNotIn('"dp"', canonical_json(wide))
        # a spec at every default is its canonical bytes on the wire too
        self.assertEqual(wire_json(plain), canonical_json(plain))

    def test_the_wire_carries_the_widths_and_decodes_them(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, train, _ = arith_store(tmp.name)
        wide = self.spec(train, 3, 2)
        self.assertEqual(spec_from_json(json.loads(wire_json(wide))), wide)
        frame = frame_for(wide, checkpointing=EVERY_UPDATE)
        self.assertEqual(spec_from_json(frame["spec"]), wide)

    def test_sealed_bytes_are_identical_across_replicas_and_dp(self) -> None:
        """The promise: with the deterministic fakes, every byte of the run
        directory — manifest, sealed waves, postdata, ledger, adapter and
        optimizer blobs — is the same at replicas ∈ {1, 3} × dp ∈ {1, 2}."""
        runs = {}
        for replicas in (1, 3):
            for dp in (1, 2):
                tmp = tempfile.TemporaryDirectory()
                self.addCleanup(tmp.cleanup)
                store, train, _ = arith_store(tmp.name)
                engines = Replicas(tuple(FakeEngine() for _ in range(replicas)))
                report = run_experiment(
                    self.spec(train, replicas, dp), SCHEMA, store,
                    {"main": engines}, FakeLearner(dp=dp),
                    checkpointing=EVERY_UPDATE)
                runs[(replicas, dp)] = (report.run_id, snapshot(store, report.run_id))
        reference = runs[(1, 1)]
        self.assertGreater(len(reference[1]), 10)
        for key, run in runs.items():
            with self.subTest(replicas=key[0], dp=key[1]):
                self.assertEqual(run, reference)

    def test_traffic_really_spread_across_the_replicas(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        engines = tuple(FakeEngine() for _ in range(3))
        run_experiment(self.spec(train, 3, 1), SCHEMA, store, {"main": Replicas(engines)},
                       FakeLearner(), checkpointing=EVERY_UPDATE)
        self.assertTrue(all(engine.served for engine in engines))

    def test_a_dp_learner_splits_the_microbatches_across_its_groups(self) -> None:
        learner_ = FakeLearner(dp=2)
        plain = FakeLearner()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        a = run_experiment(self.spec(train, 1, 2), SCHEMA, store, FakeEngine(),
                           learner_, checkpointing=EVERY_UPDATE)
        self.assertEqual(learner_.dp, 2)
        tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(tmp2.cleanup)
        store2, train2, _ = arith_store(tmp2.name)
        b = run_experiment(self.spec(train2, 1, 1), SCHEMA, store2, FakeEngine(),
                           plain, checkpointing=EVERY_UPDATE)
        self.assertEqual(learner_.forwards, plain.forwards)
        self.assertEqual(store.peek_ledger(a.run_id), store2.peek_ledger(b.run_id))

    def test_a_learner_built_at_another_dp_is_refused_at_submit(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        with self.assertRaises(Exception) as refused:
            run_experiment(self.spec(train, 1, 2), SCHEMA, store, FakeEngine(),
                           FakeLearner(dp=1), checkpointing=EVERY_UPDATE)
        self.assertIn("learner-shape-mismatch", str(refused.exception))


class GateTest(unittest.TestCase):
    """The widths' own rules at the submit gate."""

    def issues(self, topology):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        _, train, _ = arith_store(tmp.name)
        return {issue.code for issue in validate(arith_spec(train, topology=topology), SCHEMA)}

    def test_a_replicated_pool_stands_alone_on_its_host(self) -> None:
        codes = self.issues(Topology(hosts=(
            HostSpec((pool("main", replicas=2), learner())),)))
        self.assertIn("replicated-pool-alternates", codes)

    def test_widths_are_positive(self) -> None:
        codes = self.issues(Topology(hosts=(
            HostSpec((pool("main", replicas=0),)), HostSpec((learner(dp=0),)))))
        self.assertIn("width-not-positive", codes)

    def test_a_wide_topology_is_legal(self) -> None:
        self.assertEqual(self.issues(Topology(hosts=(
            HostSpec((pool("main", replicas=3),)), HostSpec((learner(dp=2),))))), set())


class ReplicaPlacementTest(DeskFixture):
    """THE DESK PLACES N HOSTS FOR ONE POOL (ADR 0022) — on N distinct
    listings, with no metal allow-list and no spread rule (Q5)."""

    def wide_spec(self, replicas: int = 3):
        return arith_spec(self.train, topology=Topology(hosts=(
            HostSpec((pool("main", replicas=replicas),)), HostSpec((learner(),)))))

    def test_a_replicated_pool_is_one_demand_per_replica(self) -> None:
        demands = demands_of(self.wide_spec())
        keys = [d.name() for d in demands]
        self.assertEqual(keys, ["main", "main#1", "main#2", "learner"])
        rows = demand_rows(demands)
        self.assertEqual([row.get("replica", 0) for row in rows], [0, 1, 2, 0])
        self.assertEqual(split_replica_key("main#2"), ("main", 2))
        self.assertEqual(replica_key("main", 0), "main")

    def stand_fleet(self, servers: int):
        hosts = [self.stand_up(f"serve-{i}", f"fleet://s{i}", serves_pool=True,
                               trains=False) for i in range(servers)]
        trainer = self.stand_up("train-b", "fleet://b", serves_pool=False, trains=True)
        desk = self.desk()
        for i, host in enumerate(hosts):
            desk.list_host(host.name, host.regimes, f"fleet://s{i}")
        desk.list_host("train-b", trainer.regimes, "fleet://b")
        trainer.desk = RemoteDesk(LocalTransport(Campaigns(desk)))
        return hosts, trainer, desk

    def test_three_replicas_land_on_three_listings_and_the_run_commits(self) -> None:
        hosts, trainer, desk = self.stand_fleet(3)

        async def drive():
            reply = await Campaigns(desk).submit(self.wide_spec(),
                                                 checkpointing=EVERY_UPDATE)
            await trainer._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(reply["accepted"], reply)
        self.assertEqual(sorted(reply["pools"]), ["learner", "main", "main#1", "main#2"])
        self.assertEqual(len({reply["pools"][k] for k in ("main", "main#1", "main#2")}), 3)
        self.assertEqual(trainer.status()["tenants"][reply["run_id"]]["status"], "done")
        for host in hosts:
            self.assertTrue(host.engines[0].served, host.name)

    def test_too_few_listings_and_no_metal_is_a_boot_not_a_double_join(self) -> None:
        _, _, desk = self.stand_fleet(2)
        reply = go(Campaigns(desk).submit(self.wide_spec(), checkpointing=EVERY_UPDATE))
        self.assertFalse(reply["accepted"])
        self.assertTrue(reply["boot"])

    def test_a_dead_replica_is_revived_through_the_desk_and_the_run_finishes(self) -> None:
        """THE DRILL, on fakes: replica 1's host stops answering mid-run; its
        in-flight requests go to the survivors, the pool asks the desk for a
        new host (`place-replica`, avoiding the run's other replicas), brings
        it up and readmits it, and the run commits every update."""
        hosts, trainer, desk = self.stand_fleet(4)
        spare = hosts[3]
        killed = {"done": False}

        class Corpse:
            async def call(self, verb, payload, **bounds):
                raise HostGone("no host answers at fleet://s1 (decarved)")

            async def ask(self, verb, payload, **bounds):
                raise HostGone("no host answers at fleet://s1 (decarved)")

        original = hosts[1].engines[0].sample_tokens

        async def dies_mid_run(*args, **kwargs):
            if not killed["done"]:
                killed["done"] = True
                self.transports["fleet://s1"] = Corpse()
                raise HostGone("decarved under the request")
            async for event in original(*args, **kwargs):
                yield event
        hosts[1].engines[0].sample_tokens = dies_mid_run
        # the spare is the one listing left for the revival to join: hide it
        # from the first placement by listing it only after the submit
        desk.delist(spare.name)

        async def drive():
            reply = await Campaigns(desk).submit(self.wide_spec(),
                                                 checkpointing=EVERY_UPDATE)
            desk.list_host(spare.name, spare.regimes, "fleet://s3")
            await trainer._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(reply["accepted"], reply)
        rid = reply["run_id"]
        self.assertEqual(trainer.status()["tenants"][rid]["status"], "done")
        self.assertEqual(len(self.store.peek_ledger(rid)), 4)
        events = [e for e in self.store.read_fleet_log() if e.get("event") == "replica-placed"]
        self.assertEqual([(e["key"], e["host"]) for e in events], [("main#1", "serve-3")])
        self.assertEqual(desk.placements()[rid]["pools"]["main#1"], "serve-3")
        journal = [e["event"] for e in self.store.read_host_log("train-b")
                   if e.get("event", "").startswith("replica-")]
        self.assertEqual(journal[:2], ["replica-dead", "replica-revived"])
        self.assertTrue(spare.engines[0].served)

    def test_losing_a_replica_host_does_not_strand_the_run(self) -> None:
        placed = {"host": "train-b", "pools": {"main": "serve-0", "main#1": "serve-1",
                                               "learner": "train-b"},
                  "demands": demand_rows(demands_of(self.wide_spec(2)))}
        self.assertTrue(pools_revive_it(placed, ["serve-1"]))
        self.assertTrue(pools_revive_it(placed, ["serve-0", "serve-1"]))
        self.assertFalse(pools_revive_it(placed, ["train-b"]))
        one = {**placed, "pools": {"main": "serve-0", "learner": "train-b"},
               "demands": demand_rows(demands_of(arith_spec(self.train, topology=Topology(
                   hosts=(HostSpec((pool("main"),)), HostSpec((learner(),)))))))}
        self.assertFalse(pools_revive_it(one, ["serve-0"]))


class WindowTest(unittest.TestCase):
    """What `observe/` sees of a pool, per rollout wave."""

    def test_the_window_counts_requests_tokens_and_affinity_hits(self) -> None:
        async def drive():
            pool = Pool.of("main", Replicas(tuple(fakes(2))))
            for key in ("a", "a", "b", "a"):
                await sample(pool, key)
            return pool.window(), pool.window()
        first, second = go(drive())
        self.assertEqual(first["keys"], 2)
        by_replica = {row["replica"]: row for row in first["replicas"]}
        self.assertEqual(by_replica[0]["requests"], 3)
        self.assertEqual(by_replica[0]["new_keys"], 1)
        self.assertEqual(by_replica[0]["seen_keys"], 2)
        self.assertGreater(by_replica[0]["tokens"], 0)
        self.assertEqual(sum(row["requests"] for row in second["replicas"]), 0)


class BookingTest(unittest.TestCase):
    """MEASURED NEXT TO DECLARED (ADR 0022): a resident's per-device GB is
    its process tree's, and an overrun is a warning on the host page."""

    def test_a_residents_gb_is_its_process_trees(self) -> None:
        from rlstack.runner.host import resident_gb

        reading = {10: {5: 30.0}, 11: {5: 2.0, 6: 31.5}, 99: {5: 70.0}}
        self.assertEqual(resident_gb(reading, {10, 11}), {5: 32.0, 6: 31.5})

    def test_the_host_page_warns_on_an_overrun(self) -> None:
        from rlstack.observe.host_series import bookings

        rows = bookings([
            {"event": "booking", "t": 1.0, "resident": "h:main-tp1",
             "kind": "inference", "declared_gb": 40.0, "measured_gb": 39.0,
             "devices": {"5": 39.0}, "overrun": False},
            {"event": "booking", "t": 2.0, "resident": "h:learner-fsdp1",
             "kind": "training", "declared_gb": 40.0, "measured_gb": 52.0,
             "devices": {"6": 52.0}, "overrun": True}])
        warned = {row["resident"]: row["warning"] for row in rows}
        self.assertIsNone(warned["h:main-tp1"])
        self.assertIn("52.0 GB", warned["h:learner-fsdp1"])


if __name__ == "__main__":
    unittest.main()
