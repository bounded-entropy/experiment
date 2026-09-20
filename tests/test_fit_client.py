"""Fits from an inline postprocessor (ADR 0019): the `fits` declaration, the
split and the gate, the client a fitting processor is handed, and FitClient
— K forks as K lanes of a second tenant on the run's own learner."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace

from rlstack import run_experiment
from rlstack.client import PoolClient
from rlstack.data.plan import GroupPlan, Replay, RunPlan, WavePlan, encode
from rlstack.data.stores.local import LocalStore
from rlstack.data.trajectory import Group, Wave, trajectory_from_row, trajectory_to_row
from rlstack.policy.adapters.dream_bank import memory_route
from rlstack.policy.siteschema import fake_qwen_schema, resolve
from rlstack.registry import POST
from rlstack.runner.arbiter import Arbiter
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.fit import FIT_LOSS, FORK_ENTRY, FitClient, FitPlanError, Stage
from rlstack.runner.loop import fit_client_of
from rlstack.runner.post import FittingClient, run_pipeline
from rlstack.runner.refs import RefReader
from rlstack.spec.flow import split_pipeline
from rlstack.spec.specs import (
    AlgoSpec, ExperimentSpec, HostSpec, OptimSpec, Plans, PolicySpec, SamplingSpec,
    Schedule, Seeds, Topology, dream_bank, learner, lora, pool,
)
from rlstack.spec.validate import validate
from rlstack.training.post.base import PostProcessor, postprocessor
from test_fit import PROBE, SEVEN, LaneLearner

BASE = "Qwen/Qwen2.5-0.5B"
SCHEMA = fake_qwen_schema(2, base=BASE)
SITE = "layers.0-1.self_attn.q_proj"
SUBDIR = "dreams/fork"
RECIPE = Stage("", 2, 1, 1e-3)
SEEN: list[dict] = []
SEEN_TOKENS: list[tuple[int, ...]] = []


@postprocessor("test_fork_gain")
class ForkGain(PostProcessor):
    """Forks the named start two ways over the group's rows and reports how
    far each fork moved the probe: the shape of a dreamer's reward."""

    produces = ("fork_gain",)
    fits = True

    async def process(self, group: Group, data, client: PoolClient):
        start = await client.names.read_named("lib/start")
        rows = [trajectory_to_row(t) for t in group.trajectories]
        SEEN_TOKENS.append(await client.fit.tokenize("dream"))
        vectors = await client.fit.forks(start, [rows[:1], rows[1:]], rows, RECIPE)
        SEEN.append({"start": start, "vectors": vectors, "client": client})
        return {"fork_gain": [sum(vectors[0]) - sum(vectors[1])] * len(group)}


@postprocessor("test_fork_with_pools")
class ForkWithPools(PostProcessor):
    produces = ("x",)
    fits = True
    pools = ("judge",)

    async def process(self, group, data, client):
        return {"x": [0.0] * len(group)}


def jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode()


def train_spec(store: LocalStore, post=("test_fork_gain",), bank=None) -> ExperimentSpec:
    """A learner-only replay run whose inline pipeline fits."""
    uri = store.cas_put(jsonl(SEVEN[:4]))
    waves = tuple(WavePlan((GroupPlan("facts", tuple(Replay(f"{uri}#{i}") for i in range(4))),))
                  for _ in range(2))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank=bank if bank is not None else {
            "pi": dream_bank(SITE, 4, 2)}),
        gen=None, plans=Plans(train=store.cas_put(encode(RunPlan(waves)))),
        algo=AlgoSpec(loss="sft", post=post, optim=OptimSpec("adamw", lr=1e-4),
                      schedule=Schedule(microbatch_tokens=4096, max_policy_lag=0)),
        topology=Topology(hosts=(HostSpec((learner(),)),)), seeds=Seeds(master=0))


class DeclarationTest(unittest.TestCase):
    def test_fits_is_a_typed_field_of_the_declaration(self) -> None:
        self.assertTrue(POST.get("test_fork_gain").fits)
        self.assertFalse(POST.get("grpo_advantage").fits)

    def test_a_fitting_processor_is_inline_whatever_else_it_declares(self) -> None:
        split = split_pipeline(("llm_judge", "test_fork_gain", "test_fork_with_pools"))
        self.assertEqual(split.pooled, ("llm_judge",))
        self.assertEqual(split.inline, ("test_fork_gain", "test_fork_with_pools"))

    def test_the_gate_refuses_a_fitting_processor_it_cannot_serve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(tmp)
            good = train_spec(store)
            self.assertEqual(validate(good, SCHEMA), [])

            def codes(spec) -> set[str]:
                return {issue.code for issue in validate(spec, SCHEMA)}

            no_learner = replace(good, topology=Topology(hosts=(HostSpec((pool("main"),)),)))
            self.assertIn("fits-without-learner", codes(no_learner))
            no_lanes = train_spec(store, bank={"pi": lora(SITE, 4)})
            self.assertEqual(codes(no_lanes), {"fits-without-lanes"})
            pooled = replace(train_spec(store, post=("test_fork_with_pools",)), topology=Topology(
                hosts=(HostSpec((learner(),)), HostSpec((pool("judge", base="judge/base"),)))))
            self.assertEqual(codes(pooled), {"fits-with-pools"})


class PipelineClientTest(unittest.TestCase):
    def wave(self) -> Wave:
        return Wave([Group("g", [trajectory_from_row(r) for r in SEVEN[:3]])])

    def test_a_fitting_processor_is_handed_fit_and_names_beside_the_pools(self) -> None:
        class Forks:
            async def forks(self, start, fork_rows, probe_rows, stage):
                return [(1.0,) * len(probe_rows), (0.5,) * len(probe_rows)]

            async def tokenize(self, text):
                return (1, 2, 3)

        class Names:
            async def read_named(self, name):
                return b"start:" + name.encode()

        SEEN.clear()
        columns = asyncio.run(run_pipeline(
            ("test_fork_gain",), self.wave(), {}, SamplingSpec(), 0, 1,
            fit=Forks(), names=Names()))
        self.assertEqual(columns, {"fork_gain": [1.5] * 3})
        self.assertEqual(SEEN[0]["start"], b"start:lib/start")
        self.assertIsInstance(SEEN[0]["client"], FittingClient)

    def test_a_caller_with_no_learner_refuses_it_by_name(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "test_fork_gain.*declares fits"):
            asyncio.run(run_pipeline(("test_fork_gain",), self.wave(), {}, SamplingSpec(), 0, 1))


class FitClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = LocalStore(self.tmp.name)
        self.run = self.store.open_run("run-1", manifest={"run_id": "run-1"}, subdir=SUBDIR)

    def client(self, learner: LaneLearner) -> FitClient:
        arbiter = Arbiter()
        arbiter.attach(learner, label="learner")
        sites = tuple(resolve(SCHEMA.sites, SITE))
        return FitClient(learner, arbiter, tenant="run-1", refs=RefReader(self.store, self.run),
                         base=BASE, sites=sites, r=4, seed=7, microbatch_tokens=4096)

    def test_tokenize_is_the_run_s_own_tenant_s_before_any_fork_exists(self) -> None:
        from test_fit import installed
        learner = installed(LaneLearner(), tenant="run-1")
        client = self.client(learner)
        self.assertEqual(asyncio.run(client.tokenize("a dream")), tuple(ord(c) for c in "a dream"))
        self.assertEqual(learner.installs, [])           # no fork tenant was needed
        with self.assertRaises(KeyError):                # and it IS the run's tenant that is asked
            asyncio.run(self.client(LaneLearner()).tokenize("a dream"))

    def test_rows_reads_a_cas_file_of_trajectory_rows(self) -> None:
        uri = self.store.cas_put(jsonl(PROBE))
        client = self.client(LaneLearner())
        self.assertEqual(asyncio.run(client.rows(uri)), PROBE)
        with self.assertRaises(FitPlanError):
            asyncio.run(client.rows("self://rollouts/1"))
        with self.assertRaises(FileNotFoundError):
            asyncio.run(client.rows("cas://" + "0" * 64))

    def test_journal_preserves_probe_vectors_ids_steps_and_update(self):
        from rlstack.runner.meters import HostJournal
        client = self.client(LaneLearner())
        client.journal = HostJournal(self.store, "test-host")
        client.update = 2
        vectors = asyncio.run(client.forks(b"start-bytes", [SEVEN[:2], SEVEN[2:4]], PROBE, RECIPE))
        event, = self.store.read_host_log("test-host")
        self.assertEqual((event["event"], event["run_id"], event["update"]), ("forks", "run-1", 2))
        self.assertEqual(event["probe_rows"], [row["task"]["id"] for row in PROBE])
        self.assertTrue(all(n > 0 for n in event["probe_tokens"]))
        self.assertEqual([lane["probes"][-1] for lane in event["lanes"]], list(map(list, vectors)))
        self.assertTrue(all(lane["steps"] == 4 for lane in event["lanes"]))
        self.assertEqual(set(event["phases"]), {"queue", "prepare", "train", "probe"})
        self.assertEqual([step["optimizer_step"] for step in event["fit_steps"]], [1, 4])
        self.assertTrue(all(batch["rows"] and batch["scored_tokens"] > 0
                            for step in event["fit_steps"] for batch in step["microbatches"]))

    def test_forks_are_lanes_of_a_second_tenant_installed_on_first_use(self) -> None:
        learner = LaneLearner()
        client = self.client(learner)
        self.assertEqual(learner.installs, [])           # lazily
        vectors = asyncio.run(client.forks(b"start-bytes", [SEVEN[:2], SEVEN[2:4], SEVEN[4:]],
                                           PROBE, RECIPE))
        (tenant, parameterization), = learner.installs
        self.assertEqual(tenant, "run-1:forks")
        self.assertEqual((parameterization.base, parameterization.loss), (BASE, FIT_LOSS))
        (entry,) = parameterization.entries
        self.assertEqual((entry.name, entry.adapter_type, entry.trainable),
                         (FORK_ENTRY, "dream_bank", True))
        self.assertEqual((entry.init["r"], entry.init["memories"], entry.init["seed"]), (4, 3, 7))
        self.assertEqual([m.name for m in entry.sites], [m.name for m in resolve(SCHEMA.sites, SITE)])
        # every fork starts from the same bytes, on its own lane
        self.assertEqual(learner.loads(), [(memory_route(k), b"start-bytes") for k in range(3)])
        # each fork's FINAL probe vector, one NLL per probe document, in fork order
        self.assertEqual(len(vectors), 3)
        self.assertTrue(all(len(v) == len(PROBE) for v in vectors))
        self.assertEqual(len(set(vectors)), 3)
        self.assertNotIn("emit_set", [call[0] for call in learner.calls])
        # forks share their forwards: 2+2+3 rows at batch 1, two epochs
        steps = learner.scales()
        self.assertEqual(len(steps), 2 * 3)
        self.assertAlmostEqual(steps[0][memory_route(0)], 1e-3 / 1e-4)

    def test_a_fresh_fork_is_loaded_with_no_payload(self) -> None:
        learner = LaneLearner()
        asyncio.run(self.client(learner).forks(None, [SEVEN[:2]], PROBE, RECIPE))
        self.assertEqual(learner.loads(), [(memory_route(0), None)])

    def test_the_tenant_is_kept_between_calls_and_rebuilt_only_wider(self) -> None:
        learner = LaneLearner()
        client = self.client(learner)

        async def go():
            await client.forks(None, [SEVEN[:2], SEVEN[2:4]], PROBE, RECIPE)
            await client.forks(None, [SEVEN[:2]], PROBE, RECIPE)
            await client.forks(None, [SEVEN[:1], SEVEN[1:2], SEVEN[2:3]], PROBE, RECIPE)
        asyncio.run(go())
        self.assertEqual([p.entries[0].init["memories"] for _, p in learner.installs], [2, 3])

    def test_concurrent_calls_are_serialized(self) -> None:
        """Every group's processor runs at once and they share one tenant: a
        second call's load_set must not land inside the first call's fit."""
        learner = LaneLearner()
        client = self.client(learner)

        async def go():
            return await asyncio.gather(
                client.forks(b"one", [SEVEN[:3], SEVEN[3:]], PROBE, RECIPE),
                client.forks(b"two", [SEVEN[:3], SEVEN[3:]], PROBE, RECIPE))
        first, second = asyncio.run(go())
        payloads = [payload for _, payload in learner.loads()]
        self.assertEqual(payloads, [b"one", b"one", b"two", b"two"])
        verbs = [call[0] for call in learner.calls]
        second_call = [i for i, call in enumerate(learner.calls)
                       if call[0] == "load_set" and call[2] == b"two"][0]
        # 3 and 4 rows at batch 1 for two epochs: the longer fork's 8 steps,
        # all of them before the second call's first load
        self.assertEqual(verbs[:second_call].count("optim_step"), 8)
        self.assertEqual(verbs[second_call:].count("optim_step"), 8)
        self.assertNotEqual(first, second)

    def test_forks_handed_the_same_rows_train_alike(self) -> None:
        """Forks share one shuffle key: fork_score's empty dream is the
        control exactly, and a fork that differs does so by its rows alone."""
        control, empty, dreamt = asyncio.run(self.client(LaneLearner()).forks(
            b"s", [SEVEN[:3], SEVEN[:3], SEVEN[:4]], PROBE, RECIPE))
        self.assertEqual(control, empty)
        self.assertNotEqual(control, dreamt)

    def test_same_call_same_answer(self) -> None:
        def once():
            return asyncio.run(self.client(LaneLearner()).forks(
                b"s", [SEVEN[:3], SEVEN[3:]], PROBE, RECIPE))
        self.assertEqual(once(), once())


class TrainerWiringTest(unittest.TestCase):
    def test_only_a_run_whose_pipeline_fits_gets_a_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(tmp)
            resolved = {"pi": resolve(SCHEMA.sites, SITE)}
            learner_, arbiter = LaneLearner(), Arbiter()
            run = store.open_run("rid", manifest={"run_id": "rid"}, subdir=SUBDIR)
            refs = RefReader(store, run)
            self.assertIsNone(fit_client_of(train_spec(store, post=()), resolved,
                                            learner_, arbiter, tenant="rid", refs=refs))
            client = fit_client_of(train_spec(store), resolved, learner_, arbiter,
                                   tenant="rid", refs=refs)
            self.assertEqual((client.tenant, client.r, client.microbatch_tokens),
                             ("rid:forks", 4, 4096))

    def test_a_training_run_whose_inline_processor_forks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(tmp)
            store.write_named(SUBDIR, "lib/start", b"library-bytes", {"r": 4})
            spec = train_spec(store)
            learner_ = LaneLearner()
            SEEN.clear()
            report = run_experiment(spec, SCHEMA, store, {}, learner_, subdir=SUBDIR,
                                    checkpointing=Checkpointing(every=1))
            self.assertEqual(report.completed, 2)
            self.assertEqual(len(SEEN), 2)               # one group, two updates
            self.assertEqual(SEEN[0]["start"], b"library-bytes")
            self.assertEqual(SEEN_TOKENS[-1], tuple(ord(c) for c in "dream"))
            self.assertEqual([t for t, _ in learner_.installs],
                             [report.run_id, f"{report.run_id}:forks"])
            # the forks ran under the fork tenant's lanes; the run's own rows
            # trained as sealed, under no lane
            self.assertNotIn(f"{report.run_id}:forks", learner_._tenants)   # closed with the run
            fork_loads = [payload for _, payload in learner_.loads()]
            self.assertEqual(fork_loads, [b"library-bytes"] * 4)
            run = store.open_run(report.run_id, subdir=SUBDIR, create=False)
            column = run.read_postdata(1)["fork_gain"]
            self.assertEqual(len(column), 4)
            self.assertEqual(len(set(column)), 1)
            self.assertIn("fork_gain", run.read_ledger()[0]["post"])


if __name__ == "__main__":
    unittest.main()
