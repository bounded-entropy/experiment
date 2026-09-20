"""HTTP reaches a host's resident proxy without losing admission on timeout."""

import asyncio
import tempfile
import threading
import unittest

from rlstack import FakeLearner, Host, LocalStore
from rlstack.runner.interfaces import EntryInstall, OptimSettings, Parameterization
from rlstack.runner.remote import (
    HostService, LearnerService, LocalTransport, RemoteLearner, Unreachable,
)
from rlstack.runner.transports.http import HttpServer, HttpTransport


class PausedLearner(FakeLearner):
    def __init__(self, *, fail=False):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []
        self.fail = fail

    def uninstall(self, tenant):
        self.calls.append(tenant)
        if tenant == "slow":
            self.started.set()
            if not self.release.wait(5):
                raise RuntimeError("test did not release the learner")
            if self.fail:
                raise ValueError("learner refused the operation")
        super().uninstall(tenant)


class ShortLocalTransport(LocalTransport):
    """Bound cross-loop regression failures instead of leaving blocked threads."""

    async def call(self, verb, payload, *, deadline_s=1.0):
        return await super().call(verb, payload, deadline_s=min(deadline_s, 1.0))


class HttpHostRuntimeTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = LocalStore(self.directory.name)

    def service(self, learner):
        return HostService(Host("worker", engines=(), learner=learner,
                                store=self.store))

    async def test_host_calls_a_resident_learner_proxy(self):
        inner = FakeLearner()
        resident = RemoteLearner(LocalTransport(LearnerService(inner)))
        service = self.service(resident)
        params = Parameterization(
            base="fake", loss="sft",
            entries=(EntryInstall("pi", "lora", {"r": 2, "seed": 5}, True, ()),),
            optim=OptimSettings("adamw", .01, (.9, .95), 0., {}))
        async with HttpServer(lambda host: service, port=0) as server:
            client = RemoteLearner(HttpTransport(server.endpoint), admitted=True)

            def train_and_reconnect():
                client.install("a", params)
                client.install("b", params)
                before_a, before_b = client.emit("a"), client.emit("b")
                client.optim_step("a")
                after_a = client.emit("a")
                self.assertNotEqual(after_a, before_a)
                self.assertEqual(client.emit("b"), before_b)
                fresh = RemoteLearner(HttpTransport(server.endpoint), admitted=True)
                self.assertEqual(fresh.emit("a"), after_a)
                client.uninstall("a")
                client.uninstall("b")

            await asyncio.to_thread(train_and_reconnect)
        await service.drain_learner_calls()
        self.assertEqual(service.host.arbiter.in_flight(), 0)

    async def test_timeout_keeps_admission_and_serializes_until_work_finishes(self):
        learner = PausedLearner()
        service = self.service(learner)
        arbiter = service.host.arbiter
        other = object()
        arbiter.attach(learner, label="learner", group="gpu")
        arbiter.attach(other, label="inference", group="gpu")
        other_entered = asyncio.Event()

        async def use_inference():
            async with arbiter.admit(other):
                other_entered.set()

        async with HttpServer(lambda host: service, port=0) as server:
            client = HttpTransport(server.endpoint)
            slow = asyncio.create_task(client.call(
                "uninstall", {"tenant": "slow"}, deadline_s=.1))
            competitor = None
            try:
                self.assertTrue(await asyncio.to_thread(learner.started.wait, 2))
                with self.assertRaises(Unreachable):
                    await slow
                self.assertEqual(arbiter.in_flight(), 1)
                self.assertIsInstance(await client.ask("status", {}, deadline_s=1), dict)
                competitor = asyncio.create_task(use_inference())
                with self.assertRaises(Unreachable):
                    # Confirm cancellation at the serving side. An HTTP client
                    # timing out is not proof the remote wait has expired yet.
                    await LocalTransport(service).call(
                        "uninstall", {"tenant": "queued"}, deadline_s=.1)
                self.assertFalse(other_entered.is_set())
                self.assertEqual(learner.calls, ["slow"])
            finally:
                learner.release.set()
                await service.drain_learner_calls()
                if competitor is not None:
                    await asyncio.wait_for(competitor, 2)
                await asyncio.gather(slow, return_exceptions=True)
            self.assertEqual(learner.calls, ["slow"])
            self.assertTrue(other_entered.is_set())
            await client.call("uninstall", {"tenant": "next"})
            self.assertEqual(learner.calls, ["slow", "next"])
        self.assertEqual(arbiter.in_flight(), 0)

    async def test_cancelled_admission_wait_does_not_execute_later(self):
        learner = PausedLearner()
        service = self.service(learner)
        arbiter = service.host.arbiter
        other = object()
        arbiter.attach(learner, label="learner", group="gpu")
        arbiter.attach(other, label="inference", group="gpu")
        async with HttpServer(lambda host: service, port=0) as server:
            async with arbiter.admit(other):
                with self.assertRaises(Unreachable):
                    await HttpTransport(server.endpoint).call(
                        "uninstall", {"tenant": "queued"}, deadline_s=.1)
                await service.drain_learner_calls()
            self.assertEqual(learner.calls, [])
        self.assertEqual(arbiter.in_flight(), 0)

    async def test_late_failure_releases_admission_after_actual_completion(self):
        learner = PausedLearner(fail=True)
        service = self.service(learner)
        async with HttpServer(lambda host: service, port=0) as server:
            client = HttpTransport(server.endpoint)
            call = asyncio.create_task(client.call(
                "uninstall", {"tenant": "slow"}, deadline_s=.1))
            try:
                self.assertTrue(await asyncio.to_thread(learner.started.wait, 2))
                with self.assertRaises(Unreachable):
                    await call
                self.assertEqual(service.host.arbiter.in_flight(), 1)
            finally:
                learner.release.set()
                await service.drain_learner_calls()
                await asyncio.gather(call, return_exceptions=True)
            await client.call("uninstall", {"tenant": "next"})
            self.assertEqual(learner.calls, ["slow", "next"])
        self.assertEqual(service.host.arbiter.in_flight(), 0)

    async def test_two_local_remote_learners_share_one_admission_queue(self):
        learner = PausedLearner()
        service = self.service(learner)
        first = RemoteLearner(ShortLocalTransport(service), admitted=True)
        second = RemoteLearner(ShortLocalTransport(service), admitted=True)
        slow = asyncio.create_task(asyncio.to_thread(first.uninstall, "slow"))
        following = None
        try:
            self.assertTrue(await asyncio.to_thread(learner.started.wait, 1))
            following = asyncio.create_task(asyncio.to_thread(second.uninstall, "second"))
            async with asyncio.timeout(1):
                while len(service._learner_calls) < 2:
                    await asyncio.sleep(.002)
            self.assertIsNot(first.frames_loop(), second.frames_loop())
            self.assertEqual(learner.calls, ["slow"])
            self.assertEqual(service.host.arbiter.in_flight(), 1)
        finally:
            learner.release.set()
            await asyncio.gather(*(call for call in (slow, following) if call is not None))
            await service.drain_learner_calls()
        self.assertEqual(learner.calls, ["slow", "second"])
        self.assertEqual(service.host.arbiter.in_flight(), 0)

    async def test_foreign_loop_timeout_and_drain_keep_started_work_owned(self):
        learner = PausedLearner()
        service = self.service(learner)

        def request(tenant):
            async def call():
                with self.assertRaises(Unreachable):
                    await LocalTransport(service).call(
                        "uninstall", {"tenant": tenant}, deadline_s=.05)
            asyncio.run(call())

        slow = asyncio.create_task(asyncio.to_thread(request, "slow"))
        draining = None
        try:
            self.assertTrue(await asyncio.to_thread(learner.started.wait, 1))
            await slow
            await asyncio.to_thread(request, "queued")
            self.assertEqual(learner.calls, ["slow"])
            self.assertEqual(service.host.arbiter.in_flight(), 1)
            service.close_admission()
            draining = asyncio.create_task(asyncio.to_thread(
                lambda: asyncio.run(service.drain_learner_calls())))
            await asyncio.sleep(.02)
            self.assertFalse(draining.done())
        finally:
            learner.release.set()
            await asyncio.gather(*(call for call in (slow, draining) if call is not None))
            await service.drain_learner_calls()
        self.assertEqual(learner.calls, ["slow"])
        self.assertEqual(service.host.arbiter.in_flight(), 0)


class HostServiceLoopTest(unittest.TestCase):
    def test_finished_test_loop_can_rebind_an_idle_service(self):
        with tempfile.TemporaryDirectory() as root:
            learner = PausedLearner()
            service = HostService(Host("worker", engines=(), learner=learner,
                                        store=LocalStore(root)))
            for tenant in ("first", "second"):
                asyncio.run(LocalTransport(service).call("uninstall", {"tenant": tenant}))
            self.assertEqual(learner.calls, ["first", "second"])
            self.assertEqual(service._learner_calls, set())
