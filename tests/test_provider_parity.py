"""Provider startup adapters preserve the shared fleet contract, without GPUs."""

import asyncio
from contextlib import ExitStack
import inspect
import io
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlsplit

from common import arith_spec, arith_store
from venue_stub import modal_stubbed
from rlstack.runner.checkpointing import EVERY_UPDATE
from rlstack.data.stores.local import LocalStore
from rlstack.runner.desk import Metal, MetalService
from rlstack.runner.remote import LocalTransport, RemoteDesk, parse_address
from rlstack.runner.venues.client import VenueClient
from rlstack.runner.venues.modal.desk import desk_class
from rlstack.runner.venues.modal.provider import ModalProvider, terminate_container
from rlstack.runner.venues.modal.worker import metal_class
from rlstack.runner.venues.runtime import DeskRuntime, MetalRuntime
from rlstack.runner.venues.strangeloop import worker
from rlstack.runner.transports.http import HttpServer, HttpTransport
from rlstack.spec.canonical import canonical_json
from test_services import FakeProvider, eventually


class PublishedStore(LocalStore):
    """Stand in only the platform publication probe; all Store bytes are real."""

    def verify_publication(self):
        self.verified = True


class ProviderParityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = PublishedStore(temporary.name)
        self.provider = FakeProvider()
        self.desk = DeskRuntime(
            self.store, transport_for=lambda address: None, provider=self.provider,
            bootable_metals=frozenset({"card"}), heartbeat_s=.01, reap_tick_s=None)
        self.remote = RemoteDesk(LocalTransport(self.desk.campaigns))

    async def assert_registered(self, runtime):
        await asyncio.wait_for(runtime.registration, 2)
        self.assertIsInstance(runtime, MetalRuntime)
        self.assertEqual(self.desk.desk.metal["card"], Metal("card", "L4", 1, 24))
        self.assertEqual(self.desk.desk.metal_containers["card"], "allocation-1")
        self.assertEqual(self.desk.desk.epoch_of("card"), runtime.service.epoch)
        self.assertTrue(runtime.service.epoch)
        self.assertEqual(self.desk.desk.idle_limit("card"), 90)
        self.assertEqual(runtime.heartbeat_s, .01)
        host = parse_address(runtime.service.address_of("card:0.main"))
        self.assertEqual((host.host.rsplit("/", 1)[-1], host.epoch),
                         ("card:0.main", runtime.service.epoch))

    async def test_modal_worker_registers_routes_and_shuts_down_shared_runtime(self):
        with modal_stubbed() as modal, \
             patch.object(MetalService, "measure", return_value=Metal("card", "L4", 1, 24)), \
             patch("rlstack.runner.remote.RemoteDesk", return_value=self.remote), \
             patch("rlstack.runner.remote.transport_for", return_value=None), \
             patch.dict(os.environ, {"MODAL_TASK_ID": "allocation-1"}):
            cls = metal_class(
                modal.App("test"), "test", "card", "L4", None,
                module="deployed_science", cls="CustomMetal",
                store_for=lambda: self.store, desk_address="modal://desk/Desk",
                volumes={}, idle_s=90)
            self.assertEqual((cls.__module__, cls.__name__, cls.__qualname__),
                             ("deployed_science", "CustomMetal", "CustomMetal"))
            self.assertTrue(inspect.iscoroutinefunction(cls.door_ask))
            instance = cls()
            await instance.bring_up()
            try:
                await self.assert_registered(instance.runtime)
                described = await instance.door_ask("", "describe", {})
                self.assertEqual(described["epoch"], instance.runtime.service.epoch)
                await instance.door("", "release", {})
                self.assertTrue((await instance.serve())["released"])
                with self.assertRaisesRegex(RuntimeError, "released"):
                    await instance.door_ask("", "describe", {})
            finally:
                await instance.bring_down()
        self.assertTrue(all(task.done() for task in instance.runtime.tasks))

    async def test_strangeloop_worker_registers_routes_and_shuts_down_same_runtime(self):
        runtimes, servers = [], []
        measured = MetalRuntime.measured

        def capture_runtime(*args, **kwargs):
            runtime = measured(*args, **kwargs)
            runtimes.append(runtime)
            return runtime

        def capture_server(*args, **kwargs):
            server = HttpServer(*args, **kwargs)
            servers.append(server)
            return server

        config = worker.WorkerConfig(
            "card", "strangeloop://account/prefix", "http://127.0.0.1:8000",
            "http://127.0.0.1:8000#card", 0, 90, "allocation-1", "/tmp", "source", None)
        with ExitStack() as patches:
            patches.enter_context(patch.object(MetalService, "measure", return_value=Metal("card", "L4", 1, 24)))
            patches.enter_context(patch.object(MetalRuntime, "measured", side_effect=capture_runtime))
            patches.enter_context(patch.object(worker, "health", return_value=True))
            patches.enter_context(patch.object(worker.ScratchClient, "from_locator",
                                               return_value=SimpleNamespace(prefix="prefix")))
            patches.enter_context(patch.object(worker, "StrangeLoopLocalStore", return_value=self.store))
            patches.enter_context(patch.object(worker, "RemoteDesk", return_value=self.remote))
            patches.enter_context(patch.object(worker, "transport_for", return_value=None))
            patches.enter_context(patch("rlstack.runner.transports.http.HttpServer", side_effect=capture_server))
            patches.enter_context(patch.object(asyncio.get_running_loop(), "add_signal_handler"))
            patches.enter_context(patch.dict(os.environ, {"RLSTACK_HTTP_TOKEN": ""}))
            process = asyncio.create_task(worker.run(config))
            try:
                await eventually(lambda: bool(runtimes) and runtimes[0].started)
                runtime = runtimes[0]
                await self.assert_registered(runtime)
                client = HttpTransport(servers[0].endpoint)
                self.assertEqual((await client.ask("describe", {}))["epoch"], runtime.service.epoch)
                self.assertTrue(self.store.verified)
                await client.call("release", {})
                await asyncio.wait_for(process, 3)
                self.assertTrue(runtime.service.released.is_set())
                self.assertTrue(all(task.done() for task in runtime.tasks))
            finally:
                process.cancel()
                await asyncio.gather(process, return_exceptions=True)

    async def test_modal_desk_uses_shared_duties_and_preserves_deployment_identity(self):
        with modal_stubbed() as modal:
            cls = desk_class(
                modal.App("desk"), None, module="deployed_desk", store_for=lambda: self.store,
                volumes={}, bootable_metals=frozenset({"card"}))
            self.assertEqual((cls.__module__, cls.__name__, cls.__qualname__),
                             ("deployed_desk", "Desk", "Desk"))
            instance = cls()
            await instance.bring_up()
            try:
                self.assertIsInstance(instance.runtime, DeskRuntime)
                self.assertIsInstance(instance.runtime.provider, ModalProvider)
                self.assertEqual(await instance.door_ask("", "status", {}), instance.desk.status())
                self.assertEqual(len(instance.runtime.tasks), 2)
            finally:
                await instance.bring_down()
            self.assertTrue(all(task.done() for task in instance.runtime.tasks))

    async def test_modal_provider_uses_the_journal_address_and_exact_termination_id(self):
        provider = ModalProvider(lambda name: "modal://saved-app/SavedMetal")
        boot = Mock()
        with patch("rlstack.runner.venues.modal.provider.boot_by_spawn", return_value=boot) as spawn, \
             patch("rlstack.runner.venues.modal.provider.terminate_container", new_callable=AsyncMock) as end:
            provider.boot("card")
            spawn.assert_called_once_with("saved-app", "SavedMetal")
            boot.assert_called_once_with("card")
            end.return_value = False
            self.assertFalse(await provider.terminate("exact-allocation"))
            end.assert_awaited_once_with("exact-allocation")

    async def test_modal_stop_acknowledgement_is_not_completion(self):
        finished = lambda when: SimpleNamespace(info=SimpleNamespace(finished_at=when))
        stub = SimpleNamespace(
            TaskGetInfo=AsyncMock(side_effect=[finished(0), finished(0), finished(42)]),
            ContainerStop=AsyncMock(side_effect=TimeoutError("lost stop response")))
        sdk = SimpleNamespace(_Client=SimpleNamespace(from_env=AsyncMock(return_value=SimpleNamespace(stub=stub))))
        proto = SimpleNamespace(TaskGetInfoRequest=lambda **kw: kw, ContainerStopRequest=lambda **kw: kw)
        with patch.dict("sys.modules", {"modal.client": sdk, "modal_proto": SimpleNamespace(api_pb2=proto)}), \
             patch("rlstack.runner.venues.modal.provider.asyncio.sleep", new_callable=AsyncMock):
            self.assertTrue(await terminate_container("exact-allocation"))
        stub.ContainerStop.assert_awaited_once_with({"task_id": "exact-allocation", "graceful": False})
        self.assertEqual(stub.TaskGetInfo.await_count, 3)
        self.assertTrue(all(call.args == ({"task_id": "exact-allocation"},)
                            for call in stub.TaskGetInfo.await_args_list))

    async def test_common_client_requires_live_capacity_and_scopes_release(self):
        client = VenueClient(self.remote)
        unavailable = {"metal": {"card": {"plane": "registered", "live": False, "residual": [1]}}}
        ready = {"metal": {"card": {"plane": "registered", "live": True, "residual": [1]},
                           "someone-elses": {"plane": "other"}}}
        with patch.object(self.remote, "status", new_callable=AsyncMock, side_effect=[unavailable, ready]), \
             patch("rlstack.runner.venues.client.asyncio.sleep", new_callable=AsyncMock):
            self.assertEqual(await client.wait_for_metal("card"), ready["metal"]["card"])
        with patch.object(self.remote, "status", new_callable=AsyncMock, return_value=ready), \
             patch.object(self.remote, "release", new_callable=AsyncMock, return_value={"released": False}) as release:
            self.assertEqual(await client.guarded_release(["card"], "done"), {"card": {"released": False}})
            release.assert_awaited_once_with("card", reason="done")
            self.assertFalse(await client.mine_are_released(["card"]))
            self.assertTrue(await client.mine_are_released(["already-gone"]))

    async def test_common_client_preserves_spec_and_plan_bytes_and_exact_resume_directory(self):
        _, train, heldout = arith_store(self.store.root)
        spec = arith_spec(train, heldout)
        client = VenueClient(self.remote)

        def build(store):
            self.assertEqual(store.cas_get(train), self.store.cas_get(train))
            store.cas_put(b"new plan bytes")
            return spec

        row = await client.canonical_row(build, borrow=(train, heldout))
        self.assertEqual(canonical_json(row), canonical_json(spec))
        # A plan created in the temporary client store has reached the real Desk store.
        from common import cas_uri
        self.assertEqual(self.store.cas_get(cas_uri(b"new plan bytes")), b"new plan bytes")
        with patch.object(self.remote, "submit", new_callable=AsyncMock, return_value={"accepted": True}) as submit:
            await client.submit(row, "family/control", resume=True, solo=True, anchor="learner",
                                checkpointing=EVERY_UPDATE)
        passed = submit.await_args
        self.assertEqual(canonical_json(passed.args[0]), canonical_json(spec))
        self.assertEqual(passed.kwargs, {"subdir": "family/control", "resume": True,
                                         "solo": True, "anchor": "learner",
                                         "checkpointing": EVERY_UPDATE})

    def test_common_progress_reads_the_exact_observer_directory(self):
        payload = {"run_id": "run", "extent": "train", "committed": 2, "target": 4,
                   "done": False, "updates": [{"update": 2, "train": {"loss": .5}}]}
        client = VenueClient(self.remote, "http://observer.example/")
        with patch("rlstack.runner.venues.client.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as get:
            reply = client.progress("family/control/run", folder="strangeloop://account/prefix")
        url = urlsplit(get.call_args.args[0])
        self.assertEqual(url.path, "/api/run/run")
        self.assertEqual(parse_qs(url.query), {"subdir": ["family/control"],
                                              "root": ["strangeloop://account/prefix"]})
        self.assertEqual(reply, {"extent": "train", "completed": 2, "planned": 4,
                                 "done": False, "committed": 2, "train": [{"loss": .5}]})


if __name__ == "__main__":
    unittest.main()
