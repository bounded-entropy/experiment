"""The Host (rlstack.runner.host) and the hosts CLI (rlstack.__main__).

The claims under test: a host-submitted run is byte-identical to a raw
run_experiment (the host adds custody, never semantics); pools bind onto
owned engines by base; capacity refuses what cannot fit; the journal records
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
    FakeEngine, FakeLearner, GpuConfig, GpuGroup, Host, HostError, Seeds,
    fake_qwen_schema, gpus, learner, pool, run_experiment,
)
from rlstack.__main__ import render_gpu, render_hosts, render_runs

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
        spec = arith_spec(self.train, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main"),
                                 pool("judge", base="Qwen/Qwen3-8B"),
                                 learner())),)))
        binding = host.bind_pools(spec)
        self.assertIs(binding["judge"], judge_engine)

        lonely = self.host(engines=(FakeEngine(base="Qwen/Qwen3-0.6B"),))
        with self.assertRaises(HostError) as caught:
            lonely.bind_pools(spec)
        self.assertIn("Qwen/Qwen3-8B", str(caught.exception))

    def test_capacity_refuses_what_cannot_fit(self) -> None:
        host = self.host()
        heavy = arith_spec(self.train, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main", fraction=0.7),
                                 learner(fraction=0.4))),)))
        with self.assertRaises(HostError) as caught:
            go(host.submit(heavy, SCHEMA))
        self.assertIn("cannot fit", str(caught.exception))

        # a second tenant over the SAME objects adds no load: submits fine
        fits = arith_spec(self.train, gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main", fraction=0.5),
                                 learner(fraction=0.4))),)))
        go(host.submit(fits, SCHEMA))
        again = replace(fits, seeds=Seeds(master=99))
        go(host.submit(again, SCHEMA))
        self.assertLessEqual(host.arbiter.declared_load(), 1.0)

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
