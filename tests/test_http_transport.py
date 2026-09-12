"""Real localhost sockets around the existing wire, with no provider or GPU."""
import asyncio
import time
import unittest
from urllib.request import urlopen

from rlstack.runner.remote import (
    EPOCH_KEY, LocalTransport, WrongEpoch, Unreachable, check_epoch,
    parse_address, serve_in_process, stop_serving_in_process, transport_for,
)
from rlstack.runner.transports.http import HttpServer, HttpTransport, RemoteError


class Counter:
    def __init__(self):
        self.steps = {}
        self.active = self.peak = 0

    async def serve(self, verb, payload):
        check_epoch(payload, "epoch-1", "counter")
        if verb == "fail":
            raise ValueError("named refusal")
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(payload.get("delay", 0))
            tenant = payload["tenant"]
            self.steps[tenant] = self.steps.get(tenant, 0) + 1
            return {"step": self.steps[tenant], "verb": verb}
        finally:
            self.active -= 1

    def answer(self, verb, payload):
        check_epoch(payload, "epoch-1", "counter")
        time.sleep(payload.get("delay", 0))
        return {"steps": dict(self.steps), "epoch": payload.get(EPOCH_KEY)}


class HttpTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = Counter()
        self.server = await HttpServer(lambda host: self.service, port=0).__aenter__()
        self.transport = transport_for(self.server.endpoint + "#learner@epoch-1")

    async def asyncTearDown(self):
        await self.server.__aexit__(None, None, None)

    async def test_health_and_separate_requests_preserve_tenant_state(self):
        with await asyncio.to_thread(urlopen, self.server.endpoint + "/health") as reply:
            self.assertEqual(reply.status, 200)
        self.assertEqual(await self.transport.call("optim_step", {"tenant": "a"}),
                         {"step": 1, "verb": "optim_step"})
        self.assertEqual(await self.transport.call("optim_step", {"tenant": "a"}),
                         {"step": 2, "verb": "optim_step"})
        await self.transport.call("optim_step", {"tenant": "b"})
        self.assertEqual((await self.transport.ask("status", {}))["steps"], {"a": 2, "b": 1})

    async def test_requests_overlap_and_slow_answers_do_not_park_loop(self):
        await asyncio.gather(
            self.transport.call("forward_backward", {"tenant": "a", "delay": .05}),
            self.transport.call("forward_backward", {"tenant": "b", "delay": .05}),
            self.transport.ask("status", {"delay": .1}),
        )
        self.assertEqual(self.service.peak, 2)

    async def test_epoch_and_application_failures_are_not_success_dicts(self):
        wrong = transport_for(self.server.endpoint + "#learner@old")
        with self.assertRaises(WrongEpoch):
            await wrong.call("optim_step", {"tenant": "a"})
        with self.assertRaises(ValueError):
            await self.transport.call("fail", {})
        self.assertEqual(self.service.steps, {})

    async def test_deadline_does_not_retry_mutation(self):
        with self.assertRaises(Unreachable):
            await self.transport.call("optim_step", {"tenant": "a", "delay": .2}, deadline_s=.03)
        await asyncio.sleep(.25)
        self.assertEqual(self.service.steps, {})

    async def test_factory_keeps_same_process_shortcut(self):
        address = self.server.endpoint + "#local"
        serve_in_process(address, self.service)
        try:
            self.assertIsInstance(transport_for(address + "@epoch-1"), LocalTransport)
        finally:
            stop_serving_in_process(address)

    async def test_bearer_auth_and_unknown_host_refuse(self):
        def named(host):
            if host != "learner":
                raise KeyError(host)
            return self.service
        async with HttpServer(named, port=0, token="test-token") as server:
            with self.assertRaises(RemoteError):
                await HttpTransport(server.endpoint).ask("status", {})
            client = HttpTransport(server.endpoint, "learner", token="test-token")
            self.assertEqual((await client.ask("status", {}))["steps"], {})
            with self.assertRaises(KeyError):
                await HttpTransport(server.endpoint, "unknown", token="test-token").ask("status", {})

    async def test_blocking_bridge_can_share_transport_with_async_callers(self):
        from rlstack.runner.remote import Blocking
        result = await asyncio.to_thread(Blocking.run, self.transport.ask("status", {}))
        self.assertEqual(result["epoch"], "epoch-1")
        await self.transport.call("optim_step", {"tenant": "a"})

    async def test_bad_deadlines_fail_before_sending(self):
        for value in [0, -1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                await self.transport.ask("status", {}, deadline_s=value)


class HttpGrammarTest(unittest.TestCase):
    def test_host_and_epoch_are_not_part_of_endpoint(self):
        address = parse_address("http://127.0.0.1:18000#learner@epoch-1")
        self.assertEqual(address.endpoint, "http://127.0.0.1:18000")
        self.assertEqual((address.host, address.epoch), ("learner", "epoch-1"))

    def test_invalid_endpoints_are_refused(self):
        for address in ["http://:80", "http://host:99999", "http://host/path", "https://host?token=x"]:
            with self.assertRaises(ValueError):
                transport_for(address)

    def test_nonloopback_server_requires_auth(self):
        with self.assertRaises(ValueError):
            HttpServer(lambda host: Counter(), host="0.0.0.0")
