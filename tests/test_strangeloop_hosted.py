"""The Strange Loop desk hosted: what a container needs that a laptop had for free.

A desk in a standing container serves the same `serve` the laptop runs, with
its state on a volume, its operator elsewhere, and its lock outliving the
process that took it. Each of those is a rule here: the config's state dir
and operator door, the operator's token, the guarded observer, and the
stale-safe lock. Nothing here touches Modal, the CLI or the network — the
deploy file is imported under the SDK stand-in, as every venue file is.
"""

from __future__ import annotations

import asyncio
import errno
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from venue_stub import modal_stubbed

from rlstack.runner.venues.strangeloop import provider
from rlstack.runner.venues.strangeloop.desk import (
    OBSERVER_COOKIE, observer_guard, operator_token, runtime_token,
)
from rlstack.runner.venues.strangeloop.provider import (
    DeskConfig, LockHolder, MetalConfig, lock_holder, own_local_desk,
)

REPO = Path(__file__).resolve().parent.parent
STORE = "strangeloop://sl-scratch-account/rlstack"
DOOR = "https://workspace--rlstack-sl-desk-gateway.modal.run"


def a_metal() -> MetalConfig:
    return MetalConfig("alpha", "A100", "rlstack-pinned", 18761)


class ConfigTests(unittest.TestCase):
    def test_read_takes_the_state_dir_a_container_names(self):
        """The file names the laptop's state dir; a hosted desk reads the same
        file with its volume in that one field's place."""
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "desk.json"
            path.write_text(json.dumps({
                "store": STORE, "state_dir": "~/laptop-state",
                "metals": [{"name": "alpha", "gpu": "A100", "image": "img", "local_port": 18761}]}))
            laptop = DeskConfig.read(path)
            hosted = DeskConfig.read(path, state_dir=Path("/state/desk"))
        self.assertEqual(laptop.state_dir, Path("~/laptop-state").expanduser().resolve())
        self.assertEqual(hosted.state_dir, Path("/state/desk"))
        self.assertEqual(hosted.metals, laptop.metals)

    def test_operator_door_is_loopback_until_an_endpoint_is_configured(self):
        laptop = DeskConfig(STORE, (a_metal(),), Path("/nowhere"))
        hosted = DeskConfig(STORE, (a_metal(),), Path("/nowhere"), operator_endpoint=DOOR)
        self.assertEqual(laptop.operator_door, "http://127.0.0.1:18760")
        self.assertEqual(hosted.operator_door, DOOR)
        # a recorded address never moves: workers reach the gateway at their own loopback
        self.assertEqual(hosted.address("alpha", "host", "epoch"), laptop.address("alpha", "host", "epoch"))
        self.assertEqual(hosted.endpoint, "http://127.0.0.1:18760")

    def test_operator_endpoint_is_a_bare_origin(self):
        for bad in ("workspace--desk.modal.run", DOOR + "#alpha", DOOR + "?token=x",
                    "https://user:secret@workspace.modal.run", "ssh://workspace"):
            with self.assertRaisesRegex(ValueError, "operator_endpoint"):
                DeskConfig(STORE, (a_metal(),), Path("/nowhere"), operator_endpoint=bad)

    def test_the_hosted_example_is_the_general_config_plus_the_door(self):
        general = DeskConfig.read(REPO / "examples/strangeloop-config.json")
        hosted = DeskConfig.read(REPO / "examples/strangeloop-hosted.json",
                                 state_dir=Path("/state/desk"))
        self.assertEqual(hosted.metals, general.metals)
        self.assertEqual((hosted.store, hosted.gateway_port, hosted.observer_port),
                         (general.store, general.gateway_port, general.observer_port))
        self.assertEqual(general.operator_endpoint, "")
        self.assertTrue(hosted.operator_endpoint.startswith("https://"))


class StaleLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.lock = Path(self.temp.name) / "desk.lock"

    def test_the_owner_writes_its_record_and_clears_it_on_exit(self):
        with own_local_desk(self.lock):
            self.assertEqual(lock_holder(self.lock), LockHolder.mine())
        self.assertIsNone(lock_holder(self.lock))
        with own_local_desk(self.lock):
            pass

    def test_a_record_from_another_machine_is_stale(self):
        self.assertTrue(LockHolder.mine().alive())
        self.assertFalse(LockHolder("a-container-that-died", os.getpid()).alive())

    def test_a_record_of_a_gone_process_is_stale(self):
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait()        # reaped: the pid names nothing on this host now
        self.assertFalse(LockHolder(LockHolder.mine().host, process.pid).alive())

    def test_a_record_of_another_boot_is_stale_though_host_and_pid_match(self):
        """Every Modal container is host `modal` with the desk at pid 2 (the
        first deploy, 2026-09-17), so a died container's record names this
        very process; the boot id in the record tells them apart, and a
        laptop's record (no boot id) keeps the host-and-pid rule."""
        mine = LockHolder.mine()
        self.assertFalse(LockHolder(mine.host, mine.pid, "another-boot").alive())
        self.assertTrue(LockHolder(mine.host, mine.pid, mine.boot).alive())
        self.assertTrue(LockHolder(mine.host, mine.pid).alive())
        self.lock.write_text(f"{mine.host} {mine.pid} another-boot\n")
        self.assertEqual(lock_holder(self.lock), LockHolder(mine.host, mine.pid, "another-boot"))
        self.lock.write_text(f"{mine.host} {mine.pid}\n")
        self.assertEqual(lock_holder(self.lock), LockHolder(mine.host, mine.pid))

    def test_a_record_left_by_a_died_container_is_ignored_where_flock_works(self):
        self.lock.write_text("a-container-that-died 4242\n")
        with own_local_desk(self.lock):
            self.assertEqual(lock_holder(self.lock), LockHolder.mine())

    def test_a_volume_that_refuses_flock_falls_back_to_the_record(self):
        """A mounted volume may refuse flock: the record is then the fence —
        a dead container's record is taken, a live desk's on this host is not."""
        refused = OSError(errno.ENOTSUP, "no locks on this filesystem")
        self.lock.write_text("a-container-that-died 4242\n")
        with patch.object(provider.fcntl, "flock", side_effect=refused):
            with own_local_desk(self.lock):
                self.assertEqual(lock_holder(self.lock), LockHolder.mine())
            self.assertIsNone(lock_holder(self.lock))
            self.lock.write_text(LockHolder.mine().line())
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                with own_local_desk(self.lock):
                    self.fail("took a live desk's lock")

    def test_an_unreadable_record_is_no_holder(self):
        self.lock.write_text("garbage\n")
        self.assertIsNone(lock_holder(self.lock))
        self.assertIsNone(lock_holder(self.lock.with_name("missing")))


class OperatorTokenTests(unittest.TestCase):
    def test_environment_first_then_the_local_file_and_a_hosted_desk_refuses(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            laptop = DeskConfig(STORE, (a_metal(),), state)
            hosted = DeskConfig(STORE, (a_metal(),), state, operator_endpoint=DOOR)
            with patch.dict(os.environ, {"RLSTACK_HTTP_TOKEN": "from-the-volume"}):
                self.assertEqual(operator_token(hosted), "from-the-volume")
                self.assertEqual(operator_token(laptop), "from-the-volume")
            with patch.dict(os.environ):
                os.environ.pop("RLSTACK_HTTP_TOKEN", None)
                with self.assertRaisesRegex(RuntimeError, "modal volume get"):
                    operator_token(hosted)
                self.assertFalse((state / "http-token").exists(), "no token minted for a hosted desk")
                self.assertEqual(operator_token(laptop), runtime_token(state))


def page(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"runs"]


def knock(app, **environ) -> tuple[str, dict, bytes]:
    told: dict = {}

    def start(status, headers):
        told["status"], told["headers"] = status, dict(headers)

    body = b"".join(app({"REQUEST_METHOD": "GET", "PATH_INFO": "/", "QUERY_STRING": "",
                         **environ}, start))
    return told["status"], told["headers"], body


class ObserverGuardTests(unittest.TestCase):
    def setUp(self):
        self.guarded = observer_guard(page, "the-runtime-token")

    def test_nothing_answers_without_the_token(self):
        self.assertEqual(knock(self.guarded)[0], "401 Unauthorized")
        self.assertEqual(knock(self.guarded, HTTP_AUTHORIZATION="Bearer wrong")[0], "401 Unauthorized")
        self.assertEqual(knock(self.guarded, QUERY_STRING="token=wrong")[0], "401 Unauthorized")
        self.assertEqual(knock(self.guarded, HTTP_COOKIE=f"{OBSERVER_COOKIE}=wrong")[0], "401 Unauthorized")

    def test_the_bearer_the_gateway_takes_passes(self):
        status, _, body = knock(self.guarded, HTTP_AUTHORIZATION="Bearer the-runtime-token")
        self.assertEqual((status, body), ("200 OK", b"runs"))

    def test_the_token_offered_once_becomes_the_cookie_a_browser_carries(self):
        status, headers, _ = knock(self.guarded, PATH_INFO="/runs",
                                   QUERY_STRING="token=the-runtime-token&view=hosts")
        self.assertEqual(status, "303 See Other")
        self.assertEqual(headers["Location"], "/runs?view=hosts")
        self.assertIn(f"{OBSERVER_COOKIE}=the-runtime-token", headers["Set-Cookie"])
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        status, _, body = knock(self.guarded, HTTP_COOKIE=f"{OBSERVER_COOKIE}=the-runtime-token")
        self.assertEqual((status, body), ("200 OK", b"runs"))


def load_deploy(name: str):
    """One `deploy/*.py` by path, under the Modal stand-in (as test_venues does)."""
    sys.path.insert(0, str(REPO / "deploy"))
    try:
        spec = importlib.util.spec_from_file_location(f"venue_{name}", REPO / "deploy" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(REPO / "deploy"))
        for loaded in list(sys.modules):
            if loaded.startswith("venue_") or loaded == "modal_venue":
                sys.modules.pop(loaded, None)


class HostedDeployTests(unittest.TestCase):
    def test_the_container_serves_the_config_from_the_volume(self):
        with modal_stubbed():
            hosted = load_deploy("strangeloop_desk")
        self.assertEqual(hosted.config.state_dir, Path("/state/desk"))
        self.assertEqual(hosted.STATE_DIR.parent, hosted.CLI_CONFIG_DIR.parent, "one volume for both")
        general = DeskConfig.read(REPO / "examples/strangeloop-config.json")
        self.assertEqual((hosted.config.gateway_port, hosted.config.observer_port, hosted.config.store),
                         (general.gateway_port, general.observer_port, general.store))
        self.assertEqual(hosted.ROOT, REPO)
        self.assertTrue(hosted.CLI_WHEEL.endswith("/strangeloop_cli-0.8.1-py3-none-any.whl"))

    def test_the_desk_runs_on_a_loop_of_its_own_and_stops_from_another_thread(self):
        """Modal's user loop runs only while a hook runs, so a task on it
        froze at its first await (the first deploy); the desk runs under
        `asyncio.run` on its own thread and the exit hook stops it through
        that loop."""
        with modal_stubbed():
            hosted = load_deploy("strangeloop_desk")
        seen: dict = {}

        async def a_serve(config, root, *, exposed, stop):
            seen["args"] = (config, root, exposed)
            seen["loop"] = asyncio.get_running_loop()
            await asyncio.sleep(0)          # the first await is where the task froze
            await stop.wait()
            seen["stopped"] = True

        desk = hosted.DeskThread(a_serve, "the-config", "the-root", exposed=True)
        desk.start()
        deadline = time.monotonic() + 5.0
        while "loop" not in seen and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(seen["args"], ("the-config", "the-root", True))
        self.assertIsNot(seen["loop"], None)
        self.assertTrue(desk.thread.is_alive(), "the desk is standing past its first await")
        desk.request_stop(timeout=5.0)
        self.assertFalse(desk.thread.is_alive())
        self.assertTrue(seen.get("stopped"))

    def test_reauth_asks_every_slot_of_a_hosted_desk(self):
        """The laptop reads which slots booted from its state dir; a hosted
        desk's state is on its volume, so every slot is asked."""
        operator = load_deploy("strangeloop")
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            metals = (a_metal(), MetalConfig("beta", "A100", "rlstack-pinned", 18762))
            laptop = DeskConfig(STORE, metals, state, gpu_ceiling=2)
            hosted = DeskConfig(STORE, metals, state, gpu_ceiling=2, operator_endpoint=DOOR)
            (state / "beta.json").write_text(json.dumps(
                {"label": "rlstack-beta-1", "lease_id": "lease-b", "boot_started": True}))
            self.assertEqual(operator.renewable(laptop), ["beta"])
            self.assertEqual(operator.renewable(hosted), ["alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
