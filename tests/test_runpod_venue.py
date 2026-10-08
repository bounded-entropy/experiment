"""RunPod metal is started by hand and attaches to a desk beside its volume (ADR 0021).

Nothing here reaches RunPod: rlstack calls no RunPod API, so there is
nothing to mock but the card. The provider refuses to boot or stop a pod,
the desk refuses to knock, and a worker on a "pod" (a local process with
RUNPOD_POD_ID) registers its own endpoint with a real desk over real HTTP
and the two share one store tree, as two pods share a network volume.
"""
from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from rlstack.data.stores.local import LocalStore
from rlstack.runner.desk import Metal, MetalService
from rlstack.runner.remote import LocalTransport, without_epoch
from rlstack.runner.venues.runpod import worker
from rlstack.runner.venues.runpod.desk import desk, serve
from rlstack.runner.venues.runpod.provider import (
    DeskConfig, MetalConfig, RunPodProvider, desk_token,
)
from rlstack.runner.venues.runtime import DeskRuntime
from rlstack.runner.venues.serving import LOOPBACK

REPO = Path(__file__).resolve().parent.parent
DESK = "http://desk-pod.runpod.internal:18760"
POD = "http://h100-pod.runpod.internal:8000"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((LOOPBACK, 0))
        return probe.getsockname()[1]


async def eventually(predicate, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not await predicate():
            await asyncio.sleep(0.02)


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_boot_and_terminate_say_they_are_not_implemented_on_runpod(self):
        provider = RunPodProvider()
        with self.assertRaisesRegex(NotImplementedError, "not implemented on RunPod.*by hand"):
            provider.boot("rp-h100-a")
        with self.assertRaisesRegex(NotImplementedError, "'pod-1' is not implemented on RunPod.*by hand"):
            await provider.terminate("pod-1")


class ConfigTests(unittest.TestCase):
    def metals(self, **overrides) -> tuple[MetalConfig, ...]:
        return (MetalConfig(**{"name": "rp-a", "endpoint": POD, **overrides}),)

    def test_the_example_is_one_h100_slot_with_an_engine_and_a_learner_recipe(self):
        config = DeskConfig.read(REPO / "examples/runpod-h100.json")
        self.assertEqual([metal.name for metal in config.metals], ["rp-h100-a"])
        builds = config.metals[0].builds
        self.assertIsNotNone(builds.engine)
        self.assertIsNotNone(builds.learner)
        self.assertEqual(config.metals[0].idle_s, 300)
        self.assertTrue(config.store.startswith("/workspace/"))
        self.assertTrue(config.operator_endpoint.startswith("https://"))
        self.assertIn(".runpod.internal:", config.metals[0].endpoint)

    def test_a_metal_is_dialed_at_its_own_endpoint(self):
        metal = self.metals()[0]
        self.assertEqual(metal.address, POD)
        self.assertEqual(metal.host_address("rp-a:0.main", "life"), POD + "#rp-a:0.main@life")
        from rlstack.runner.remote import parse_address
        told = parse_address(metal.host_address("rp-a:0.main", "life"))
        self.assertEqual((told.endpoint, told.host, told.epoch), (POD, "rp-a:0.main", "life"))

    def test_the_operator_door_defaults_to_the_desk_endpoint(self):
        private = DeskConfig("/workspace/rlstack", DESK, self.metals(), Path("/state"))
        proxied = DeskConfig("/workspace/rlstack", DESK, self.metals(), Path("/state"),
                             operator_endpoint="https://desk-pod-18760.proxy.runpod.net")
        self.assertEqual(private.operator_door, DESK)
        self.assertEqual(proxied.operator_door, "https://desk-pod-18760.proxy.runpod.net")

    def test_invalid_configs_are_refused(self):
        state = Path("/state")
        cases = {
            "relative store": lambda: DeskConfig("rlstack", DESK, self.metals(), state),
            "routed desk": lambda: DeskConfig("/w", DESK + "#desk", self.metals(), state),
            "metal with a path": lambda: self.metals(endpoint=POD + "/v1"),
            "infinite metal idle": lambda: self.metals(idle_s=float("inf")),
            "zero desk idle": lambda: DeskConfig("/w", DESK, self.metals(), state, idle_s=0),
            "shared ports": lambda: DeskConfig("/w", DESK, self.metals(), state,
                                               desk_port=8321, observer_port=8321),
            "two slots one name": lambda: DeskConfig("/w", DESK, self.metals() * 2, state),
            "bad name": lambda: self.metals(name="rp a"),
            "zero host idle": lambda: DeskConfig("/w", DESK, self.metals(), state, host_idle_s=0),
        }
        for case, build in cases.items():
            with self.subTest(case), self.assertRaises(ValueError):
                build()
        with self.assertRaisesRegex(ValueError, "not a configured slot"):
            DeskConfig("/w", DESK, self.metals(), state).metal("rp-b")

    def test_the_token_comes_from_the_environment_or_the_desk_file_and_is_never_minted(self):
        with tempfile.TemporaryDirectory() as root:
            config = DeskConfig("/w", DESK, self.metals(), Path(root))
            with patch.dict(os.environ, {"RLSTACK_HTTP_TOKEN": "from-env"}):
                self.assertEqual(desk_token(config), "from-env")
            with patch.dict(os.environ):
                os.environ.pop("RLSTACK_HTTP_TOKEN", None)
                with self.assertRaisesRegex(RuntimeError, "start the desk first"):
                    desk_token(config)
                self.assertFalse((Path(root) / "http-token").exists())
                (Path(root) / "http-token").write_text("from-the-volume\n")
                self.assertEqual(desk_token(config), "from-the-volume")

    def test_a_worker_takes_its_slot_and_the_pod_id_runpod_sets(self):
        config = DeskConfig("/workspace/rlstack", DESK, self.metals(), Path("/state"), worker_port=8001)
        with patch.dict(os.environ, {"RUNPOD_POD_ID": "h100-pod"}):
            told = worker.WorkerConfig.from_desk(config, "rp-a")
        self.assertEqual((told.metal.name, told.store, told.desk, told.port, told.pod_id),
                         ("rp-a", "/workspace/rlstack", DESK, 8001, "h100-pod"))
        with patch.dict(os.environ):
            os.environ.pop("RUNPOD_POD_ID", None)
            with self.assertRaisesRegex(RuntimeError, "RUNPOD_POD_ID"):
                worker.WorkerConfig.from_desk(config, "rp-a")

    def test_two_slots_on_one_machine_listen_on_their_own_ports(self):
        metals = (MetalConfig("g3", "http://127.0.0.1:8023", port=8023,
                              environment={"CUDA_VISIBLE_DEVICES": "3"}),
                  MetalConfig("g4", "http://127.0.0.1:8024", port=8024,
                              environment={"CUDA_VISIBLE_DEVICES": "4"}),
                  MetalConfig("shared", "http://127.0.0.1:8001"))
        config = DeskConfig("/workspace/rlstack", DESK, metals, Path("/state"), worker_port=8001)
        with patch.dict(os.environ, {"RUNPOD_POD_ID": "node"}):
            ports = {name: worker.WorkerConfig.from_desk(config, name).port for name in ("g3", "g4", "shared")}
        self.assertEqual(ports, {"g3": 8023, "g4": 8024, "shared": 8001})
        with self.assertRaises(ValueError):
            MetalConfig("g5", "http://127.0.0.1:8025", port=0)


class DeskRefusalTests(unittest.IsolatedAsyncioTestCase):
    """The shared Desk under a RunPod provider: no knock, no termination,
    and idle release still drains the metal."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = LocalStore(temporary.name)
        self.now = [1000.0]
        self.service = MetalService(Metal("rp-a", "H100", 1, 80.0), store=self.store,
                                    address_of=lambda host: POD + "#" + host, epoch="life-1")
        self.runtime = DeskRuntime(
            self.store, transport_for=lambda address: {POD: LocalTransport(self.service)}[without_epoch(address)],
            provider=RunPodProvider(), bootable_metals=frozenset(), auto_recover=False,
            idle_tick_s=0.005, clock=lambda: self.now[0])
        self.runtime.desk.register_metal(self.service.metal, POD, container="h100-pod",
                                         epoch="life-1", idle_s=300.0)

    async def test_a_knock_is_refused_and_journaled_without_reaching_the_provider(self):
        with patch.object(RunPodProvider, "boot") as boot:
            self.assertFalse(await self.runtime.desk.knock("rp-a"))
        boot.assert_not_called()
        refused = [event for event in self.store.read_fleet_log() if event["event"] == "knock-refused"]
        self.assertEqual([event["metal"] for event in refused], ["rp-a"])
        self.assertIn("bootable metal: none", refused[0]["reason"])

    async def test_a_release_drains_the_metal_and_prints_the_stop_by_hand_note(self):
        printed = io.StringIO()
        with redirect_stdout(printed):
            told = await self.runtime.desk.release("rp-a")
        self.assertEqual((told["released"], told["told"], told["terminated"]), (True, True, False))
        self.assertTrue(self.service.released.is_set())
        self.assertIn("not implemented on RunPod", printed.getvalue())
        self.assertIn("h100-pod", printed.getvalue())

    async def test_idle_release_stays_on(self):
        self.runtime.desk.declare_idle("rp-a", 10.0)
        with redirect_stdout(io.StringIO()):
            async with self.runtime:
                async with asyncio.timeout(2):
                    while "rp-a" not in self.runtime.desk.idle_since:
                        await asyncio.sleep(0.002)
                self.now[0] += 11.0
                async with asyncio.timeout(2):
                    while "rp-a" not in self.runtime.desk.released:
                        await asyncio.sleep(0.002)
        self.assertTrue(self.service.released.is_set())

    async def test_the_host_limit_reaches_the_desk(self):
        """A short host limit decarves an idle engine long before the metal's
        own limit releases the hand-started worker."""
        runtime = DeskRuntime(self.store, transport_for=lambda address: None,
                              provider=RunPodProvider(), bootable_metals=frozenset(),
                              auto_recover=False, idle_s=3600.0, host_idle_s=120.0)
        self.assertEqual(runtime.desk.host_idle_s, 120.0)
        self.assertEqual(runtime.desk.host_idle_limit("rp-a"), 120.0)
        with self.assertRaises(ValueError):
            DeskRuntime(self.store, transport_for=lambda address: None,
                        provider=RunPodProvider(), bootable_metals=frozenset(), host_idle_s=0.0)


class AttachTests(unittest.IsolatedAsyncioTestCase):
    """A real desk and a real worker over HTTP, one store tree between them."""

    async def test_a_hand_started_worker_attaches_and_the_desk_reaches_its_store(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        desk_port, observer_port, worker_port = free_port(), free_port(), free_port()
        endpoint = f"http://{LOOPBACK}:{worker_port}"
        config = DeskConfig(str(root / "volume" / "rlstack"), f"http://{LOOPBACK}:{desk_port}",
                            (MetalConfig("rp-a", endpoint, idle_s=300.0,
                                         environment={"RLSTACK_TEST_SLOT": "rp-a"}),),
                            root / "volume" / "desk", desk_port=desk_port,
                            observer_port=observer_port)
        stop = asyncio.Event()
        with patch.dict(os.environ), \
             patch.object(MetalService, "measure", return_value=Metal("rp-a", "H100", 1, 80.0)), \
             patch.object(asyncio.get_running_loop(), "add_signal_handler"), \
             redirect_stdout(io.StringIO()):
            os.environ.pop("RLSTACK_HTTP_TOKEN", None)
            os.environ["RUNPOD_POD_ID"] = "h100-pod"
            standing = asyncio.create_task(serve(config, host=LOOPBACK, stop=stop))
            token_file = config.state_dir / "http-token"
            async with asyncio.timeout(5):
                while not token_file.exists():
                    await asyncio.sleep(0.02)
            token = token_file.read_text().strip()
            os.environ.pop("RLSTACK_HTTP_TOKEN", None)      # the worker reads the volume's file
            on_pod = worker.WorkerConfig.from_desk(config, "rp-a")
            on_pod = worker.WorkerConfig(on_pod.metal, on_pod.store, on_pod.desk, worker_port,
                                         on_pod.pod_id, host=LOOPBACK)
            metal = asyncio.create_task(worker.run(on_pod, desk_token(config)))
            try:
                remote = desk(config)

                async def registered() -> bool:
                    row = (await remote.status())["metal"].get("rp-a", {})
                    return bool(row.get("plane"))

                await eventually(registered)
                row = (await remote.status())["metal"]["rp-a"]
                self.assertEqual((row["address"], row["gpu"], row["idle_s"]), (endpoint, "H100", 300.0))
                self.assertEqual(os.environ["RLSTACK_TEST_SLOT"], "rp-a")
                self.assertEqual(os.environ["RLSTACK_HTTP_TOKEN"], token)
                # the desk journals into the volume's tree, where any pod reads it
                registrations = [event for event in LocalStore(config.store).read_fleet_log()
                                 if event.get("event") == "metal" and event.get("name") == "rp-a"]
                self.assertTrue(registrations, "the registration reached the shared store")
                self.assertEqual(registrations[-1]["container"], "h100-pod")
                # the observer beside the desk reads the same tree, behind the token
                opener = build_opener(ProxyHandler({}))
                observed = f"http://{LOOPBACK}:{observer_port}/api/hosts"
                with self.assertRaises(HTTPError) as refused:
                    await asyncio.to_thread(opener.open, observed, timeout=5)
                self.assertEqual(refused.exception.code, 401)
                refused.exception.close()
                answer = await asyncio.to_thread(
                    opener.open, Request(observed, headers={"Authorization": "Bearer " + token}), timeout=5)
                with answer:
                    self.assertEqual(answer.status, 200)
                told = await remote.release("rp-a")
                self.assertEqual((told["released"], told["terminated"]), (True, False))
                await asyncio.wait_for(metal, 5)       # the worker's shift ended with the release
            finally:
                metal.cancel()
                stop.set()
                await asyncio.gather(metal, return_exceptions=True)
                await asyncio.wait_for(standing, 10)


def load_deploy():
    """deploy/runpod.py by path; it needs no Modal SDK."""
    spec = importlib.util.spec_from_file_location("venue_runpod", REPO / "deploy" / "runpod.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class DeployTests(unittest.IsolatedAsyncioTestCase):
    async def test_release_tells_the_operator_to_stop_the_pod_by_hand(self):
        operator = load_deploy()
        remote = SimpleNamespace(release=AsyncMock(return_value={"released": True, "terminated": False}))
        args = SimpleNamespace(config=REPO / "examples/runpod-h100.json", command="release",
                               metal="rp-h100-a")
        printed = io.StringIO()
        with patch.object(operator, "desk", return_value=remote), redirect_stdout(printed):
            await operator.execute(args)
        remote.release.assert_awaited_once_with("rp-h100-a")
        self.assertIn("stop the pod at http://h100-pod-id.runpod.internal:8000 by hand",
                      printed.getvalue())

    def test_the_image_reuses_the_strangeloop_layers_with_flash_linear_attention(self):
        runpod = (REPO / "deploy/runpod.Dockerfile").read_text().splitlines()
        strangeloop = (REPO / "deploy/strangeloop.Dockerfile").read_text().splitlines()
        layers = [line for line in strangeloop if line.startswith(("RUN uv", "RUN /opt", "COPY", "ENV VLLM", "ENV OMP", "ENV PYTORCH"))]
        self.assertTrue(any("flash-linear-attention" in line for line in layers))
        for line in layers:
            self.assertIn(line, runpod)
        self.assertIn("rlstack.runner.venues.runpod.worker", "\n".join(runpod))
        self.assertIn("deploy.runpod", "\n".join(runpod))


if __name__ == "__main__":
    unittest.main()
