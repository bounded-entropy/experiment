"""Local contracts for allocation identities and callers on opposite SSH ends."""
from __future__ import annotations

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch, Mock

from rlstack.runner.remote import WrongEpoch, check_epoch, transport_for
from rlstack.runner.venues.strangeloop.provider import (
    DeskConfig, Gateway, LeaseState, MetalConfig, StrangeLoopCLI, StrangeLoopVenue,
    own_local_desk, source_bundle, ssh_command_argv, ssh_forward_argv, reverse_forwarding_config,
)
from rlstack.runner.transports.http import HttpServer


class RuntimeTokenTest(unittest.TestCase):
    def test_first_start_creates_private_token_and_restart_reuses_it(self):
        from rlstack.runner.venues.strangeloop.desk import runtime_token

        with tempfile.TemporaryDirectory() as root:
            state = Path(root) / "new-state"
            token = runtime_token(state)
            self.assertGreaterEqual(len(token), 32)
            self.assertEqual(runtime_token(state), token)
            self.assertEqual((state / "http-token").stat().st_mode & 0o777, 0o600)


class Provider(StrangeLoopCLI):
    def __init__(self):
        self.calls = []
        self.rows = {}
        self.fail_launch = False
        self.up_labels = []
        self.image_state = "ready"
        self.inventory = []

    def run(self, *args, timeout=180):
        self.calls.append(args)
        if args[:2] == ("image", "status"):
            return {"status": self.image_state}
        if args[:2] == ("gpu", "list"):
            return {"gpus": self.inventory}
        if args[:2] == ("gpu", "up"):
            label = args[args.index("--label") + 1]
            self.up_labels.append(label)
            if self.fail_launch:
                self.fail_launch = False
                raise subprocess.TimeoutExpired("strangeloop", 180)
            self.rows["lease-new"] = {"id": "lease-new", "status": "provisioning"}
            return self.rows["lease-new"]
        if args[:2] == ("gpu", "forward"):
            return {"argv": ["ssh", "-F", "/tmp/sl config", "-N", "-T",
                             "-L", args[3], "sl-" + args[2]]}
        raise AssertionError(args)

    def status(self, lease_id):
        return self.rows[lease_id]

    def down(self, lease_id):
        self.rows[lease_id]["status"] = "released"
        return True


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.metal = MetalConfig("alpha", "A100", "rlstack-pinned", 18761)
        self.config = DeskConfig("strangeloop://sl-scratch-account/rlstack",
                                 (self.metal,), self.root)
        self.provider = Provider()
        self.venue = StrangeLoopVenue(self.config, self.root, cli=self.provider)

    def tearDown(self):
        self.temp.cleanup()

    def test_export_replaces_only_terminal_attempts_and_preserves_evidence(self):
        self.venue.save("alpha", LeaseState("test", "lease", True))
        cli = Mock(spec=StrangeLoopCLI)
        cli.status.return_value = {"status": "ready"}
        cli.run.return_value = {"argv": ["ssh", "-F", "/tmp/config", "host"]}
        self.venue.cli = cli
        for state, old_metric, new_metric in [("failed", "train/loss", "train/loss"),
                                               ("finished", "train/loss", "train/probe_last")]:
            with self.subTest(state=state):
                ref = "family/" + state
                export_id = hashlib.sha256(ref.encode()).hexdigest()[:16]
                record = self.root / ("export-lease-" + export_id + ".json")
                original = {"run_ref": ref, "objective": old_metric, "ready_file": "/tmp/prior.json"}
                record.write_text(json.dumps(original))
                read = [Mock(returncode=0, stdout=json.dumps({"state": state})),
                        Mock(returncode=0, stdout=json.dumps({"state": "running", "wandb_path": "/persist/new"}))]
                with patch("rlstack.runner.venues.strangeloop.provider.subprocess.run", side_effect=read), \
                        patch("rlstack.runner.venues.strangeloop.provider.upload_over_ssh"), \
                        patch("rlstack.data.stores.strangeloop.ScratchClient.from_locator") as scratch:
                    scratch.return_value.credentials.api_base = "https://scratch.example"
                    result = self.venue.export("alpha", ref, new_metric)
                self.assertEqual(result["state"], "running")
                self.assertNotEqual(json.loads(record.read_text())["ready_file"], "/tmp/prior.json")
                archived, = self.root.glob(record.stem + "-" + state + "-*.json")
                self.assertEqual(json.loads(archived.read_text()), original)

    def test_export_cannot_change_metric_while_old_worker_is_running(self):
        self.venue.save("alpha", LeaseState("test", "lease", True))
        cli = Mock(spec=StrangeLoopCLI)
        cli.status.return_value = {"status": "ready"}
        cli.run.return_value = {"argv": ["ssh", "-F", "/tmp/config", "host"]}
        self.venue.cli = cli
        export_id = hashlib.sha256(b"family/r").hexdigest()[:16]
        record = self.root / ("export-lease-" + export_id + ".json")
        record.write_text(json.dumps({"objective": "train/loss", "ready_file": "/tmp/prior.json"}))
        with patch("rlstack.runner.venues.strangeloop.provider.subprocess.run",
                   return_value=Mock(returncode=0, stdout='{"state":"running"}')), \
                self.assertRaisesRegex(RuntimeError, "not terminal"):
            self.venue.export("alpha", "family/r", "train/probe_last")
        self.assertFalse(any(call.args[:2] == ("gpu", "exec") for call in cli.run.call_args_list))

    def test_uncertain_launch_retains_label_for_explicit_next_attempt(self):
        self.provider.fail_launch = True
        with self.assertRaises(subprocess.TimeoutExpired):
            self.venue.allocate(self.metal)
        state = self.venue.state("alpha")
        self.assertTrue(state.label)
        self.assertFalse(state.lease_id)
        self.venue.allocate(self.metal)
        self.assertEqual(self.provider.up_labels, [state.label, state.label])

    def test_uncertain_launch_recovers_returned_provider_label_without_new_up(self):
        self.venue.save("alpha", LeaseState("pending-label"))
        self.provider.inventory = [{"id": "existing", "status": "provisioning", "label": "pending-label"}]
        state, row = self.venue.allocate(self.metal)
        self.assertEqual(state.lease_id, "existing")
        self.assertFalse(self.provider.up_labels)

    def test_uncertain_label_cannot_bypass_new_external_capacity(self):
        self.venue.save("alpha", LeaseState("pending-label"))
        self.provider.inventory = [{"id": "external", "status": "ready", "label": "external"}]
        with self.assertRaisesRegex(RuntimeError, "gpu_ceiling"):
            self.venue.allocate(self.metal)
        self.assertFalse(self.provider.up_labels)

    def test_active_old_lease_is_reused_even_if_not_ready(self):
        self.venue.save("alpha", LeaseState("label", "lease-old", True))
        self.provider.rows["lease-old"] = {"status": "provisioning"}
        state, row = self.venue.allocate(self.metal)
        self.assertEqual(state.lease_id, "lease-old")
        self.assertEqual(row["status"], "provisioning")
        self.assertFalse(self.provider.up_labels)

    def test_released_old_lease_gets_new_persisted_label(self):
        self.venue.save("alpha", LeaseState("label", "lease-old", True))
        self.provider.rows["lease-old"] = {"status": "released"}
        state, row = self.venue.allocate(self.metal)
        self.assertNotEqual(state.label, "label")
        self.assertFalse(state.boot_started)
        self.assertEqual(state.lease_id, "lease-new")

    def test_gpu_ceiling_counts_allocations_outside_this_desk(self):
        self.provider.inventory = [{"status": "ready", "gpu_count": 1}]
        with self.assertRaisesRegex(RuntimeError, "gpu_ceiling"):
            self.venue.allocate(self.metal)
        self.assertFalse(self.provider.up_labels)

    def test_uncertain_other_slot_reserves_capacity(self):
        other = MetalConfig("beta", "A100", "image", 18762)
        config = DeskConfig(self.config.store, (self.metal, other), self.root, gpu_ceiling=2)
        venue = StrangeLoopVenue(config, self.root, cli=self.provider)
        venue.save("beta", LeaseState("uncertain-beta"))
        self.provider.inventory = [{"status": "ready", "gpu_count": 1, "label": "external"}]
        with self.assertRaisesRegex(RuntimeError, "gpu_ceiling"):
            venue.allocate(self.metal)
        self.assertFalse(self.provider.up_labels)

    def test_image_must_exist_before_allocating(self):
        self.provider.image_state = "building"
        with self.assertRaisesRegex(RuntimeError, "image"):
            self.venue.allocate(self.metal)
        self.assertFalse(self.provider.up_labels)

    def test_same_port_cannot_mix_old_and_new_ssh_leases(self):
        old = Mock()
        old.poll.return_value = None
        self.venue.processes["alpha"] = old
        self.venue.forward_leases["alpha"] = "old"
        with patch("subprocess.Popen") as spawn, patch("rlstack.runner.venues.strangeloop.provider.configure_reverse_forwarding") as configure:
            spawn.return_value.poll.return_value = None
            self.venue.open_forward(self.metal, "new")
        old.terminate.assert_called_once()
        self.assertEqual(self.venue.forward_leases["alpha"], "new")
        self.assertIn("-R", spawn.call_args.args[0])
        configure.assert_called_once()

    def test_reverse_permission_is_loopback_only_and_keeps_authentication(self):
        original = "PasswordAuthentication no\nAllowTcpForwarding local\nGatewayPorts no\n"
        revised = reverse_forwarding_config(original, 18760)
        self.assertIn("PasswordAuthentication no\n", revised)
        self.assertIn("GatewayPorts no\n", revised)
        self.assertIn("PermitListen 127.0.0.1:18760\n", revised)
        self.assertEqual(reverse_forwarding_config(revised, 18760), revised)
        for changed in (original.replace("GatewayPorts no", "GatewayPorts yes"), original + "Match User root\n"):
            with self.assertRaises(RuntimeError):
                reverse_forwarding_config(changed, 18760)

    def test_concurrent_reconnect_and_boot_start_only_one_forward(self):
        with patch("subprocess.Popen") as spawn, patch("rlstack.runner.venues.strangeloop.provider.configure_reverse_forwarding", side_effect=lambda *args: time.sleep(.03)):
            spawn.return_value.poll.return_value = None
            with ThreadPoolExecutor(max_workers=2) as workers:
                calls = [workers.submit(self.venue.open_forward, self.metal, "same") for _ in range(2)]
                for call in calls:
                    call.result(timeout=5)
            spawn.assert_called_once()

    def test_store_settings_are_typed_and_survive_worker_environment(self):
        from rlstack.data.stores.strangeloop import HashedReads
        from dataclasses import replace

        metal = replace(self.metal, blob_cache_dir="/tmp/rlstack-blobs",
                        blob_cache_bytes=8 * 1024**3, mount_reads=True)
        with patch.dict(os.environ, metal.hashed_reads().environment()):
            settings = HashedReads.from_environment()
        self.assertEqual(settings, metal.hashed_reads())
        for edits in ({"blob_cache_dir": "/scratch/cache", "blob_cache_bytes": 100},
                      {"blob_cache_bytes": 100}, {"mount_reads": "false"}):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                replace(self.metal, **edits)

    def test_worker_cache_settings_override_platform_environment(self):
        from rlstack.runner.venues.strangeloop.worker import WorkerConfig

        row = {"name": "alpha", "store": self.config.store, "desk": "http://127.0.0.1:18760",
               "address": "http://127.0.0.1:18760#alpha", "port": 8000, "idle_s": 90,
               "lease_id": "same", "artifact_dir": "/artifacts", "source_sha256": "hash",
               "builds": None, "environment": {"SL_API_TOKEN": "test", "SL_API_BASE": "test",
               "RLSTACK_HTTP_TOKEN": "test", "HF_HOME": "/scratch/prepared"}}
        path = self.root / "worker.json"
        path.write_text(json.dumps(row))
        with patch.dict(os.environ, {"HF_HOME": "/platform/default"}):
            WorkerConfig.read(path)
            self.assertEqual(os.environ["HF_HOME"], "/scratch/prepared")
        self.assertFalse(path.exists())

    def test_worker_accepts_a_token_file_in_place_of_the_token(self):
        from rlstack.runner.venues.strangeloop.worker import WorkerConfig

        row = {"name": "alpha", "store": self.config.store, "desk": "http://127.0.0.1:18760",
               "address": "http://127.0.0.1:18760#alpha", "port": 8000, "idle_s": 90,
               "lease_id": "same", "artifact_dir": "/artifacts", "source_sha256": "hash",
               "builds": None, "environment": {"SL_API_TOKEN_FILE": "/tmp/token.json",
               "SL_API_BASE": "test", "RLSTACK_HTTP_TOKEN": "test"}}
        path = self.root / "worker.json"
        path.write_text(json.dumps(row))
        with patch.dict(os.environ, {}, clear=False):
            WorkerConfig.read(path)
            self.assertEqual(os.environ["SL_API_TOKEN_FILE"], "/tmp/token.json")
        del row["environment"]["SL_API_TOKEN_FILE"]
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "SL_API_TOKEN or SL_API_TOKEN_FILE"):
            WorkerConfig.read(path)

    def fake_scratch(self, token):
        from rlstack.data.stores.strangeloop import ScratchCredentials
        return Mock(credentials=ScratchCredentials(token, "https://api.example/api/v1", "default"))

    def test_bootstrap_hands_the_pod_a_token_file_not_the_token(self):
        from rlstack.data.stores.strangeloop import ScratchClient

        state = LeaseState("label", "lease-1")
        forward = ["ssh", "-F", "/tmp/sl config", "-N", "-T", "-L", "18761:8000", "sl-lease-1"]
        uploads = []
        with patch("rlstack.runner.venues.strangeloop.provider.upload_over_ssh",
                   side_effect=lambda forward, path, content: uploads.append((path, content))), \
                patch("rlstack.runner.venues.strangeloop.provider.source_bundle", return_value=b"zip"), \
                patch("rlstack.runner.venues.strangeloop.provider.subprocess.run"), \
                patch.object(ScratchClient, "from_locator", return_value=self.fake_scratch("secret")), \
                patch.object(self.provider, "run", return_value={}):
            self.venue.bootstrap(self.metal, state, {"artifact_dir": "$PERSIST_DIR"}, forward)
        paths = [path for path, _ in uploads]
        self.assertEqual(paths, ["/tmp/rlstack-lease-1.zip", "/tmp/rlstack-lease-1-credentials.json",
                                 "/tmp/rlstack-lease-1.json"])
        self.assertEqual(json.loads(uploads[1][1]), {"token": "secret", "api_base": "https://api.example/api/v1"})
        environment = json.loads(uploads[2][1])["environment"]
        self.assertEqual(environment["SL_API_TOKEN_FILE"], "/tmp/rlstack-lease-1-credentials.json")
        self.assertEqual(environment["RLSTACK_BLOB_CACHE_BYTES"], str(self.metal.blob_cache_bytes))
        self.assertEqual(environment["RLSTACK_MOUNT_READS"], "1" if self.metal.mount_reads else "0")
        self.assertEqual(environment["RLSTACK_BLOB_CACHE_DIR"], self.metal.blob_cache_dir or "")
        self.assertNotIn("SL_API_TOKEN", environment)
        self.assertNotIn("secret", uploads[2][1].decode())
        self.assertTrue(self.venue.state("alpha").boot_started)

    def test_reauth_rewrites_the_pod_token_file_with_the_profile_credentials(self):
        from rlstack.data.stores.strangeloop import ScratchClient

        with self.assertRaisesRegex(RuntimeError, "booted worker"):
            self.venue.reauth("alpha")
        self.venue.save("alpha", LeaseState("label", "lease-1", True))
        self.provider.rows["lease-1"] = {"id": "lease-1", "status": "ready"}
        with patch("rlstack.runner.venues.strangeloop.provider.upload_over_ssh") as upload, \
                patch.object(self.venue, "open_forward", return_value=["ssh", "sl-lease-1"]), \
                patch.object(ScratchClient, "from_locator", return_value=self.fake_scratch("renewed")):
            reply = self.venue.reauth("alpha")
        upload.assert_called_once_with(["ssh", "sl-lease-1"], "/tmp/rlstack-lease-1-credentials.json",
                                       b'{"token": "renewed", "api_base": "https://api.example/api/v1"}')
        self.assertEqual(reply, {"metal": "alpha", "lease_id": "lease-1", "reauthorized": True})
        self.provider.rows["lease-1"]["status"] = "released"
        with self.assertRaisesRegex(RuntimeError, "no longer ready"):
            self.venue.reauth("alpha")

    def test_health_alone_is_not_registration(self):
        self.venue.save("alpha", LeaseState("label", "lease-old", True))
        process = Mock()
        process.poll.return_value = None
        self.venue.processes["alpha"] = process
        self.venue.registered = lambda name, lease, epoch: False
        with patch("rlstack.runner.venues.strangeloop.provider.health", return_value=True), patch("rlstack.runner.transports.http.HttpTransport.ask", new=AsyncMock(return_value={"epoch": "new"})):
            with self.assertRaisesRegex(RuntimeError, "lease remains allocated"):
                self.venue.wait_worker(self.metal, timeout=0.01)
        self.assertEqual(self.venue.state("alpha").lease_id, "lease-old")

    def test_ready_lease_waits_for_pending_ssh_without_allocating_again(self):
        state = LeaseState("label", "lease-old")
        with patch.object(self.venue, "allocate", return_value=(state, {"status": "ready", "ssh": None})), patch.object(self.provider, "run", return_value={"status": "ready", "ssh": {"available": True}}) as status, patch.object(self.venue, "open_forward"), patch.object(self.venue, "bootstrap") as bootstrap, patch.object(self.venue, "wait_worker"):
            self.venue.boot("alpha")
        status.assert_called_once_with("gpu", "status", "lease-old", timeout=15)
        bootstrap.assert_called_once()

    def test_local_desk_lock_refuses_second_writer(self):
        with own_local_desk(self.root / "desk.lock"):
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                with own_local_desk(self.root / "desk.lock"):
                    self.fail("second desk entered")

    def test_finite_idle_and_distinct_ports_are_required(self):
        with self.assertRaises(ValueError):
            MetalConfig("alpha", "A100", "image", 18761, idle_s=float("inf"))
        with self.assertRaises(ValueError):
            DeskConfig(self.config.store, (self.metal,), self.root, gateway_port=18761)
        with self.assertRaisesRegex(ValueError, "slots exceed"):
            DeskConfig(self.config.store, (self.metal, MetalConfig("beta", "A100", "image", 18762)), self.root)

    def test_forward_options_stay_before_destination(self):
        original = ["ssh", "-F", "a path with spaces", "-N", "-T", "sl-gpu"]
        argv = ssh_forward_argv(original, 18760)
        self.assertEqual(argv[-3:], ["-R", "127.0.0.1:18760:127.0.0.1:18760", "sl-gpu"])
        transfer = ssh_command_argv(original, "cat > /tmp/blob")
        self.assertEqual(transfer[-2:], ["sl-gpu", "cat > /tmp/blob"])
        self.assertNotIn("-N", transfer)

    def test_json_flag_precedes_remote_exec_separator(self):
        with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0, '{}', '')) as run:
            StrangeLoopCLI("strangeloop").run("gpu", "exec", "lease", "--", "python", "-c", "pass")
        self.assertEqual(run.call_args.args[0][:3], ["strangeloop", "--json", "gpu"])
        self.assertEqual(run.call_args.args[0][-3:], ["python", "-c", "pass"])

    def test_source_bundle_keeps_working_bytes_and_excludes_bytecode(self):
        import io
        import zipfile
        (self.root / "rlstack").mkdir()
        (self.root / "rlstack" / "a.py").write_text("changed = True\n")
        (self.root / "rlstack" / "a.pyc").write_bytes(b"ignored")
        with zipfile.ZipFile(io.BytesIO(source_bundle(self.root))) as archive:
            self.assertEqual(archive.namelist(), ["rlstack/a.py"])
            self.assertEqual(archive.read("rlstack/a.py"), b"changed = True\n")


class EpochService:
    def __init__(self, epoch):
        self.epoch = epoch
        self.calls = []

    async def serve(self, verb, payload):
        check_epoch(payload, self.epoch, "test worker")
        self.calls.append((verb, payload["value"]))
        return {"value": payload["value"]}

    def answer(self, verb, payload):
        check_epoch(payload, self.epoch, "test worker")
        return {"value": "fresh"}


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_relay_preserves_host_and_epoch_across_two_http_hops(self):
        seen = []
        service = EpochService("epoch-one")
        def route(host):
            seen.append(host)
            return service
        async with HttpServer(route, port=0) as worker:
            port = int(worker.endpoint.rsplit(":", 1)[1])
            config = DeskConfig("strangeloop://sl-scratch-account/rlstack",
                                (MetalConfig("alpha", "A100", "image", port),), Path("/tmp/unused"))
            gateway = Gateway(config)
            async with HttpServer(gateway.service_for, port=0) as server:
                current = transport_for(server.endpoint + "#alpha/learner@epoch-one")
                reply = await current.call("mutate", {"value": 7})
                self.assertEqual(reply, {"value": 7})
                self.assertEqual(await current.ask("status", {}), {"value": "fresh"})
                old = transport_for(server.endpoint + "#alpha/learner@epoch-old")
                with self.assertRaises(WrongEpoch):
                    await old.call("mutate", {"value": 8})
        self.assertEqual(seen, ["learner", "learner", "learner"])
        self.assertEqual(service.calls, [("mutate", 7)])

class FrameBoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversize_request_is_rejected_before_any_connection(self):
        from rlstack.runner.transports.http import HttpTransport, HttpLimits
        limits = HttpLimits(inline_bytes=64, max_blob_bytes=128, spool_bytes=256)
        with self.assertRaisesRegex(ValueError, "before dispatch"):
            await HttpTransport("http://127.0.0.1:1", limits=limits).call("load", {"bytes": "x" * 500})
