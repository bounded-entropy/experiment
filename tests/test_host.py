"""The Host (rlstack.runner.host) and the hosts CLI (rlstack.__main__).

The claims under test: a host-submitted run is byte-identical to a raw
run_experiment (the host adds custody, never semantics); pools bind onto
owned engines by base; fit is custody, never memory; the journal records
attach/detach truthfully (including failures); and the CLI renders all of it
from the store alone.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import replace

from common import arith_spec, arith_store
from rlstack import (
    FakeEngine, FakeLearner, GpuConfig, HostSpec, Host, HostError, Partition,
    Regime, Seeds, SpecError, fake_qwen_schema, learner, pool,
    run_experiment,
)
from rlstack.observe import render_gpu, render_hosts, render_runs, store_for

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def go(coro):
    return asyncio.run(coro)


class HostTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)

    def host(self, **kwargs) -> Host:
        defaults = dict(engines=(FakeEngine(),), learner=FakeLearner(),
                        store=self.store)
        defaults.update(kwargs)
        return Host("test-host", **defaults)

    def test_host_submission_is_byte_identical_to_a_raw_run(self) -> None:
        """The host adds custody (binding, fit, roster, journal) and NOTHING
        else: the run directory it produces matches raw run_experiment's."""
        spec = arith_spec(self.train)
        report = go(self.host().submit(spec, SCHEMA))

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        other_store, other_train, _ = arith_store(tmp.name)
        raw = run_experiment(arith_spec(other_train), SCHEMA, other_store,
                             FakeEngine(), FakeLearner())
        self.assertEqual(report.run_id, raw.run_id)
        self.assertEqual(
            self.store.path_of(f"runs/{report.run_id}/ledger.jsonl").read_bytes(),
            other_store.path_of(f"runs/{raw.run_id}/ledger.jsonl").read_bytes())

    def test_two_tenants_one_host(self) -> None:
        host = self.host()
        a = arith_spec(self.train)
        b = arith_spec(self.train, seeds=Seeds(master=99))

        async def both():
            return await asyncio.gather(host.submit(a, SCHEMA),
                                        host.submit(b, SCHEMA))

        report_a, report_b = go(both())
        self.assertNotEqual(report_a.run_id, report_b.run_id)
        status = host.status()
        self.assertEqual(len(status["tenants"]), 2)
        self.assertTrue(all(t["status"] == "done"
                            for t in status["tenants"].values()))

    def test_pools_bind_by_base_and_missing_base_is_refused(self) -> None:
        judge_engine = FakeEngine(base="Qwen/Qwen3-8B")
        host = self.host(engines=(FakeEngine(base="Qwen/Qwen3-0.6B"),
                                  judge_engine))
        spec = arith_spec(self.train, gpu_config=GpuConfig(hosts=(
            HostSpec((pool("main"),)),
            HostSpec((pool("judge", base="Qwen/Qwen3-8B"),)),
            HostSpec((learner(),)))))
        binding = host.bind_pools(spec)
        self.assertIs(binding["judge"], judge_engine)

        lonely = self.host(engines=(FakeEngine(base="Qwen/Qwen3-0.6B"),))
        with self.assertRaises(HostError) as caught:
            lonely.bind_pools(spec)
        self.assertIn("Qwen/Qwen3-8B", str(caught.exception))

    def test_fit_is_custody_never_memory_arithmetic(self) -> None:
        """A spec's `vram_gb` is a CARVE size the metal converts at build
        (ADR 0001): a host sums nothing at its door, so two tenants sized
        past the card in GB both submit onto a bare host — the arbiter's
        declared load stays the host's own — and the one refusal FIT still
        owns is a learner member on a host wearing no learner."""
        host = self.host()
        heavy = arith_spec(self.train, gpu_config=GpuConfig(hosts=(
            HostSpec((pool("main", vram_gb=60),)),
            HostSpec((learner(vram_gb=60),)))))
        go(host.submit(heavy, SCHEMA))
        go(host.submit(replace(heavy, seeds=Seeds(master=99)), SCHEMA))
        self.assertEqual(host.arbiter.declared_load(), 0.0)

        engine_only = self.host(learner=None)
        with self.assertRaises(HostError) as caught:
            go(engine_only.submit(heavy, SCHEMA))
        self.assertIn("no training regime", str(caught.exception))

    def test_journal_records_attach_detach_and_failure(self) -> None:
        host = self.host()
        go(host.submit(arith_spec(self.train), SCHEMA))

        class Boom(RuntimeError):
            pass

        class FailingLearner(FakeLearner):
            def forward_backward(self, tenant, batch):
                raise Boom("kaput")

        failing = Host("test-host", engines=(FakeEngine(),),
                       learner=FailingLearner(), store=self.store)
        with self.assertRaises(Boom):
            go(failing.submit(arith_spec(self.train, seeds=Seeds(master=99)),
                              SCHEMA))

        events = self.store.read_host_log("test-host")
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds.count("host-up"), 2)
        self.assertEqual(kinds.count("attach"), 2)
        statuses = sorted(e["status"] for e in events if e["event"] == "detach")
        self.assertEqual(statuses, ["done", "failed"])
        self.assertEqual(self.store.list_hosts(), ["test-host"])

    def test_cli_views_render_from_the_store_alone(self) -> None:
        """The three views are pure functions of store bytes — operational
        facts only (identity, placement, status, progress), no experiment
        content: exactly what a separate UI will NOT have to re-derive."""
        fake_sample = {"gpus": [{"util": 55, "mem_used": 9000,
                                 "mem_total": 23034}]}
        host = self.host(sampler=lambda: dict(fake_sample))
        report = go(host.submit(arith_spec(self.train), SCHEMA))

        async def sample_twice():
            task = asyncio.get_running_loop().create_task(
                host.run_stats(every=0.01))
            await asyncio.sleep(0.05)
            task.cancel()

        go(sample_twice())

        hosts_text = render_hosts([self.store])
        self.assertIn("host test-host", hosts_text)
        self.assertIn("1 done", hosts_text)
        self.assertIn(self.store.describe(), hosts_text)

        runs_text = render_runs([self.store])
        self.assertIn(report.run_id, runs_text)
        self.assertIn("done", runs_text)
        self.assertIn("4/4", runs_text)
        self.assertIn("test-host", runs_text)
        self.assertNotIn("reward", runs_text)     # operational only, no content

        gpu_text = render_gpu([self.store])
        self.assertIn("host test-host", gpu_text)
        self.assertIn("9000/23034 MiB", gpu_text)
        self.assertIn("55%", gpu_text)

    def test_peeks_never_mutate_a_live_run(self) -> None:
        """An observer peeks; only attach may sweep. A staged (uncommitted)
        wave must survive a peek — open_run would have deleted it."""
        host = self.host()
        report = go(host.submit(arith_spec(self.train), SCHEMA))
        run = self.store.open_run(report.run_id)
        staged = self.store.path_of(
            f"runs/{report.run_id}/waves/000099.jsonl.gz")
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(b"staged-not-committed")
        self.store.peek_ledger(report.run_id)
        self.store.peek_manifest(report.run_id)
        render_runs([self.store])
        self.assertTrue(staged.exists())


if __name__ == "__main__":
    unittest.main()


class StoreOwnershipTest(unittest.TestCase):
    """The invariant (#37): one experiment, one store, for life — the run
    store is a per-experiment binding, journaled, and forks are detected
    where all stores are visible."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.journal_store, self.train, _ = arith_store(tmp.name)

    def test_submit_takes_the_experiment_store(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run_store, train, _ = arith_store(tmp.name)

        host = Host("test-host", engines=(FakeEngine(),), learner=FakeLearner(),
                    store=self.journal_store)
        report = go(host.submit(arith_spec(train), SCHEMA, store=run_store))

        # the run lives in ITS store; the journal lives in the host's — and
        # records where the run went
        self.assertIsNotNone(run_store.peek_manifest(report.run_id))
        self.assertIsNone(self.journal_store.peek_manifest(report.run_id))
        attach = [e for e in self.journal_store.read_host_log("test-host")
                  if e["event"] == "attach"][-1]
        self.assertEqual(attach["store"], run_store.describe())

    def test_runs_view_flags_a_forked_experiment(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        other_store, other_train, _ = arith_store(tmp.name)

        host = Host("test-host", engines=(FakeEngine(),), learner=FakeLearner(),
                    store=self.journal_store)
        report = go(host.submit(arith_spec(self.train), SCHEMA))
        # the SAME identity run against a different store: a silent fork
        run_experiment(arith_spec(other_train), SCHEMA, other_store,
                       FakeEngine(), FakeLearner())

        text = render_runs([self.journal_store, other_store])
        self.assertIn(report.run_id, text)
        self.assertIn("FORK", text)

        clean = render_runs([self.journal_store])
        self.assertNotIn("FORK", clean)


class ShapeAndRegimeTest(unittest.TestCase):
    """Sharding as build facts, hosts as attested partitions (#43): binding
    matches shape exactly, attestation dies at construction, and a
    regime-host's joins are fraction-free (the partition is the footprint)."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, _ = arith_store(tmp.name)

    def test_the_partition_says_what_metal_it_is_made_of(self) -> None:
        """#49: a partition carries the KIND of GPU it is a slice of, and it
        says so in all three places an operator looks — the host-up journal
        row, status() (JSON-safe, so it crosses the wire), and the hosts
        view. Half an L4 and half an H100 are not the same half."""
        host = Host("stamped", engines=(FakeEngine(tp=2),), learner=None,
                    store=self.store,
                    partition=Partition("node-a", (0, 1), 0.5, "L4"),
                    regimes=(Regime("main-tp2", "inference", None, 2),))
        row = {"metal": "node-a", "gpu": "L4", "devices": [0, 1],
               "memory": 0.5}
        up = [e for e in self.store.read_host_log("stamped")
              if e["event"] == "host-up"][-1]
        self.assertEqual(up["partition"], row)
        self.assertEqual(host.status()["partition"], row)

        text = render_hosts([self.store])
        self.assertIn("L4 node-a[0,1] @ 0.50", text)

    def test_a_host_without_a_partition_renders_unpartitioned(self) -> None:
        """The gpu name is a default-empty field, so every pre-#49 host (and
        every bare test host) still journals, statuses, and renders."""
        host = Host("bare", engines=(FakeEngine(),), learner=FakeLearner(),
                    store=self.store)
        self.assertIsNone(host.status()["partition"])
        self.assertIn("unpartitioned", render_hosts([self.store]))
        self.assertEqual(Partition("node-a", (0,), 1.0).gpu, "")

    def test_bind_is_shape_matched(self) -> None:
        tp2 = FakeEngine(base="Qwen/Qwen3-8B", tp=2)
        host = Host("shaped", engines=(FakeEngine(), tp2),
                    learner=FakeLearner(), store=self.store)
        spec = arith_spec(self.train, gpu_config=GpuConfig(hosts=(
            HostSpec((pool("judge", base="Qwen/Qwen3-8B", tp=2),)),
            HostSpec((pool("main"),)), HostSpec((learner(),)))))
        self.assertIs(host.bind_pools(spec)["judge"], tp2)

        four = arith_spec(self.train, gpu_config=GpuConfig(hosts=(
            HostSpec((pool("judge", base="Qwen/Qwen3-8B", tp=4),)),
            HostSpec((pool("main"),)), HostSpec((learner(),)))))
        with self.assertRaises(HostError) as caught:
            host.bind_pools(four)
        self.assertIn("tp=4", str(caught.exception))

    def test_attestation_refuses_mismatched_metal(self) -> None:
        with self.assertRaises(HostError) as caught:
            Host("bad", engines=(FakeEngine(tp=1),), learner=FakeLearner(),
                 store=self.store,
                 regimes=(Regime("teacher-tp4", "inference", None, 4),))
        self.assertIn("tp=4", str(caught.exception))

        with self.assertRaises(HostError) as caught:
            Host("bad2", engines=(FakeEngine(),), learner=FakeLearner(),
                 store=self.store,
                 regimes=(Regime("learner-fsdp2", "training", None, 2),))
        self.assertIn("fsdp=2", str(caught.exception))

    def test_a_regime_host_admits_joins_size_free(self) -> None:
        """Sizes are CARVE hints (GB the metal converts at build) and mean
        nothing on a join: the partition already is the footprint, and the
        run alternates under the host's own birth group whatever the spec
        declares."""
        host = Host(
            "carved", engines=(FakeEngine(),), learner=FakeLearner(),
            store=self.store,
            partition=Partition("node-a", (0,), 1.0),
            regimes=(Regime("main-tp1", "inference", None, 1),
                     Regime("learner-fsdp1", "training", None, 1)))
        def heavy(master: int):
            return arith_spec(self.train, seeds=Seeds(master=master),
                              gpu_config=GpuConfig(hosts=(
                                  HostSpec((pool("main", vram_gb=30),
                                            learner(vram_gb=20))),)))

        report = go(host.submit(heavy(17), SCHEMA))     # two full-fraction
        go(host.submit(heavy(99), SCHEMA))              # tenants both admit
        self.assertEqual(report.updates_completed, 4)
        self.assertEqual(host.arbiter.declared_load(), 0.0)
        self.assertGreater(len(host.arbiter.switches), 1)

    def test_a_host_name_is_one_journal_path_segment(self) -> None:
        """#51b / #52: hosts/<name>/log.jsonl is the journal key and
        list_hosts() recovers the name with split("/")[1], so a "/" in a name
        buries the host one directory deeper than the observer ever looks —
        alive, serving, and invisible. Refused at both layers: Host at birth,
        and the store at the key it writes."""
        with self.assertRaises(HostError) as caught:
            Host("node-a:0/main-tp1", engines=(FakeEngine(),), learner=None,
                 store=self.store)
        self.assertIn("journal path segment", str(caught.exception))
        self.assertEqual(self.store.list_hosts(), [])   # nothing journaled

        with self.assertRaises(AssertionError):
            self.store.append_host_event("node-a:0/main-tp1",
                                         {"event": "host-up"})

    def test_learner_shape_mismatch_is_a_binding_issue(self) -> None:
        spec = arith_spec(self.train, gpu_config=GpuConfig(hosts=(
            HostSpec((pool("main"),)), HostSpec((learner(fsdp=2),)))))
        with self.assertRaises(SpecError) as caught:
            run_experiment(spec, SCHEMA, self.store, FakeEngine(),
                           FakeLearner())
        self.assertIn("learner-shape-mismatch", str(caught.exception))


class SoloTest(unittest.TestCase):
    """A SOLO host is a birth fact (I12): this partition's purpose is one
    experiment at a time, attested at construction like the regimes, journaled
    with them, and enforced at submit before any roster line exists."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)

    def host(self, solo: bool) -> Host:
        return Host("solo-host", engines=(FakeEngine(),),
                    learner=FakeLearner(), store=self.store, solo=solo)

    def test_the_default_stays_multi_tenant(self) -> None:
        host = self.host(solo=False)
        self.assertFalse(host.solo)

        async def both():
            return await asyncio.gather(
                host.submit(arith_spec(self.train), SCHEMA),
                host.submit(arith_spec(self.train, seeds=Seeds(master=99)),
                            SCHEMA))

        a, b = go(both())
        self.assertNotEqual(a.run_id, b.run_id)

    def test_a_solo_host_refuses_a_second_tenancy(self) -> None:
        host = self.host(solo=True)

        async def both():
            return await asyncio.gather(
                host.submit(arith_spec(self.train), SCHEMA),
                host.submit(arith_spec(self.train, seeds=Seeds(master=99)),
                            SCHEMA))

        with self.assertRaises(HostError) as caught:
            go(both())
        self.assertIn("born solo", str(caught.exception))

    def test_a_finished_run_frees_the_host_again(self) -> None:
        """`occupied` is about what is RUNNING: the roster keeps finished
        tenancies for the observer, and a host that finished a run is free."""
        host = self.host(solo=True)
        go(host.submit(arith_spec(self.train), SCHEMA))
        self.assertFalse(host.occupied())
        go(host.submit(arith_spec(self.train, seeds=Seeds(master=99)), SCHEMA))
        self.assertEqual(len(host.roster), 2)

    def test_resubmitting_the_same_experiment_is_a_resume(self) -> None:
        host = self.host(solo=True)
        spec = arith_spec(self.train)
        first = go(host.submit(spec, SCHEMA))
        second = go(host.submit(spec, SCHEMA))
        self.assertEqual(first.run_id, second.run_id)

    def test_the_birth_fact_is_journaled_and_reported(self) -> None:
        host = self.host(solo=True)
        up = [e for e in self.store.read_host_log("solo-host")
              if e["event"] == "host-up"]
        self.assertTrue(up[0]["solo"])
        self.assertTrue(host.status()["solo"])


class StoreForTest(unittest.TestCase):
    def test_paths_and_schemes(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.assertEqual(store_for(tmp.name).describe(), tmp.name)
        self.assertEqual(store_for(f"file://{tmp.name}").describe(), tmp.name)
        with self.assertRaises(NotImplementedError):
            store_for("s3://bucket/prefix")
        with self.assertRaises(NotImplementedError) as caught:
            store_for("modal://rlstack-store")
        self.assertIn("lives with its venue", str(caught.exception))
