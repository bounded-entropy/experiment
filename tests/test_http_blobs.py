"""Real HTTP transfer failures never turn one learner mutation into two."""
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
import tempfile
from pathlib import Path
import socket
import time
import unittest
from unittest.mock import patch

from rlstack.runner.remote import Unreachable, WrongEpoch, check_epoch, transport_for
from rlstack.runner.venues.strangeloop.provider import DeskConfig, Gateway, MetalConfig
from rlstack.runner.transports.http import Budget, HttpLimits, HttpServer, HttpTransport, RemoteError
from rlstack.runner.transports.http_blobs import BlobRef, CHUNK_BYTES, ClientFile


class Echo:
    def __init__(self):
        self.calls = []

    async def serve(self, verb, payload):
        check_epoch(payload, "current", "echo")
        if verb == "slow":
            await asyncio.sleep(payload["delay"])
        self.calls.append(verb)
        if verb == "fail":
            raise ValueError("named refusal")
        if verb == "generate":
            return {"bytes": "z" * payload["size"]}
        return {"bytes": payload.get("bytes", ""), "count": len(self.calls)}

    def answer(self, verb, payload):
        check_epoch(payload, "current", "echo")
        if verb == "slow":
            time.sleep(payload["delay"])
        return {"bytes": payload.get("bytes", ""), "count": len(self.calls)}


def small_limits(**changes):
    return HttpLimits(**{"inline_bytes": 1024, "max_blob_bytes": 8 * 1024**2,
                         "spool_bytes": 32 * 1024**2, "ttl_s": 1, **changes})


class BulkHttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = Echo()
        self.server = await HttpServer(lambda host: self.service, port=0, token="private",
                                       limits=small_limits()).__aenter__()
        self.client = HttpTransport(self.server.endpoint, "learner", "current", token="private",
                                    limits=small_limits())

    async def asyncTearDown(self):
        await self.server.__aexit__(None, None, None)

    async def control(self, method, path, value=None, ref=None):
        return await asyncio.to_thread(self.client.control, method, path, Budget(5), value, ref=ref)

    async def reserve(self, data):
        row = await self.control("POST", "/blobs", {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        return BlobRef.decode(row["blob"], self.client.limits.max_blob_bytes)

    async def put(self, ref, data):
        def put():
            with self.client.open("PUT", "/blobs/" + ref.id, Budget(5), data=data,
                                  length=len(data), ref=ref) as reply:
                return json.load(reply)
        return await asyncio.to_thread(put)

    async def test_call_and_ask_preserve_unicode_binary_codec_and_cleanup(self):
        import base64
        data = base64.b64encode(bytes(range(256)) * 100).decode() + "東京🙂"
        self.assertEqual((await self.client.call("load", {"bytes": data}))["bytes"], data)
        self.assertEqual((await self.client.ask("emit", {"bytes": data}))["bytes"], data)
        self.assertEqual(self.service.calls, ["load"])
        self.assertEqual(self.server.blobs.used, 0)
        self.assertEqual(list(self.server.blobs.root.iterdir()), [])

    async def test_bulk_epoch_refusal_and_typed_errors(self):
        wrong = HttpTransport(self.server.endpoint, epoch="old", token="private", limits=small_limits())
        with self.assertRaises(WrongEpoch):
            await wrong.call("load", {"bytes": "x" * 4096})
        with self.assertRaisesRegex(ValueError, "named refusal"):
            await self.client.call("fail", {"bytes": "x" * 4096})
        self.assertEqual(self.service.calls, ["fail"])

    async def test_concurrent_results_remain_isolated(self):
        rows = await asyncio.gather(*(self.client.call("load", {"bytes": str(i) * 10000}) for i in range(8)))
        self.assertEqual([r["bytes"] for r in rows], [str(i) * 10000 for i in range(8)])
        self.assertEqual(len(self.service.calls), 8)
        self.assertEqual(self.server.blobs.used, 0)

    async def test_checksum_failure_and_truncated_upload_never_dispatch(self):
        data = b'{"bytes":"valid"}'
        ref = await self.reserve(data)
        with self.assertRaises(RemoteError):
            await self.put(ref, data.replace(b"valid", b"other"))
        self.assertEqual(self.server.blobs.used, 0)
        ref = await self.reserve(data)
        def disconnect():
            port = int(self.server.endpoint.rsplit(":", 1)[1])
            with socket.create_connection(("127.0.0.1", port), timeout=3) as conn:
                headers = (f"PUT /blobs/{ref.id} HTTP/1.0\r\nAuthorization: Bearer private\r\n"
                           f"X-Blob-Size: {ref.size}\r\nX-Blob-Sha256: {ref.sha256}\r\n"
                           f"Content-Length: {ref.size}\r\n\r\n").encode()
                conn.sendall(headers + data[:4])
                conn.shutdown(socket.SHUT_WR)
                return conn.recv(4096)
        self.assertIn(b"400", await asyncio.to_thread(disconnect))
        self.assertEqual(self.server.blobs.used, 0)
        self.assertEqual(self.service.calls, [])

    async def test_one_use_request_rejects_replay_and_incomplete_reference(self):
        data = json.dumps({"bytes": "hello", "@epoch": "current"}).encode()
        ref = await self.reserve(data)
        frame = {"host": "learner", "verb": "load", "payload_blob": ref.row(), "deadline_s": 3}
        with self.assertRaises(RemoteError):
            await self.control("POST", "/call", frame)
        await self.put(ref, data)
        self.assertIn("result", await self.control("POST", "/call", frame))
        with self.assertRaises(RemoteError):
            await self.control("POST", "/call", frame)
        self.assertEqual(self.service.calls, ["load"])

    async def test_changed_spool_bytes_are_refused_before_execution(self):
        data = json.dumps({"bytes": "hello", "@epoch": "current"}).encode()
        ref = await self.reserve(data)
        await self.put(ref, data)
        (self.server.blobs.root / ref.id).write_bytes(data.replace(b"hello", b"other"))
        with self.assertRaises(RemoteError):
            await self.control("POST", "/call", {"host": "learner", "verb": "load",
                                                  "payload_blob": ref.row(), "deadline_s": 3})
        self.assertEqual(self.service.calls, [])

    async def test_blob_auth_paths_length_and_reservation_quota(self):
        stranger = HttpTransport(self.server.endpoint)
        for method, path in [("POST", "/blobs"), ("GET", "/blobs/" + "0" * 32),
                             ("DELETE", "/blobs/" + "0" * 32), ("PUT", "/blobs/" + "0" * 32)]:
            with self.assertRaisesRegex(RemoteError, "401"):
                await asyncio.to_thread(stranger.control, method, path, Budget(3), {})
        for row in [{"size": -1, "sha256": "0" * 64}, {"size": 2**32, "sha256": "0" * 64},
                    {"size": 3, "sha256": "not-a-hash"}]:
            with self.assertRaises(RemoteError):
                await self.control("POST", "/blobs", row)
        with self.assertRaises(RemoteError):
            await self.control("GET", "/blobs/../../other")
        ref = await self.reserve(b"1234")
        with self.assertRaises(RemoteError):
            await self.put(ref, b"12345")
        self.server.blobs.discard(ref.id)
        self.assertEqual(self.service.calls, [])

    async def test_expiry_recovers_reservation_and_result_quota(self):
        async with HttpServer(lambda host: self.service, port=0, limits=small_limits(
                ttl_s=.1, max_blobs=1)) as server:
            client = HttpTransport(server.endpoint, epoch="current", limits=small_limits())
            row = {"size": 4000, "sha256": "0" * 64}
            await asyncio.to_thread(client.control, "POST", "/blobs", Budget(3), row)
            with self.assertRaises(RemoteError):
                await asyncio.to_thread(client.control, "POST", "/blobs", Budget(3), row)
            await asyncio.sleep(.2)
            self.assertEqual(server.blobs.used, 0)
            await asyncio.to_thread(client.control, "POST", "/blobs", Budget(3), row)
            root = server.blobs.root
        self.assertFalse(root.exists())

    async def test_result_socket_disconnect_retries_only_blob_get(self):
        real_read = self.server.blobs.read
        downloads = []
        @contextmanager
        def interrupt_once(ref, *, consume=False):
            with real_read(ref, consume=consume) as source:
                if consume:
                    yield source
                    return
                downloads.append(ref.id)
                if len(downloads) != 1:
                    yield source
                    return
                class InterruptedFile:
                    count = 0
                    def read(self, count):
                        self.count += 1
                        if self.count > 1:
                            raise ConnectionResetError("injected server disconnect")
                        return source.read(count)
                yield InterruptedFile()
        with patch.object(self.server.blobs, "read", interrupt_once):
            reply = await self.client.call("generate", {"size": 2 * CHUNK_BYTES})
        self.assertEqual(reply["bytes"], "z" * (2 * CHUNK_BYTES))
        self.assertEqual(self.service.calls, ["generate"])
        self.assertEqual(len(downloads), 2)
        self.assertEqual(downloads[0], downloads[1])

    async def test_lost_rpc_acknowledgement_is_not_retried(self):
        original = self.client.open
        def lose_ack(method, path, budget, **kwargs):
            response = original(method, path, budget, **kwargs)
            if path == "/call":
                response.close()
                raise Unreachable("simulated lost RPC acknowledgement")
            return response
        with patch.object(self.client, "open", lose_ack):
            with self.assertRaises(Unreachable):
                await self.client.call("generate", {"size": 4096})
        self.assertEqual(self.service.calls, ["generate"])
        self.assertGreater(self.server.blobs.used, 0)
        await asyncio.sleep(1.6)
        self.assertEqual(self.server.blobs.used, 0)

    async def test_result_limit_failure_reports_execution_without_replay(self):
        async with HttpServer(lambda host: self.service, port=0, limits=small_limits(
                max_blob_bytes=2048)) as server:
            client = HttpTransport(server.endpoint, epoch="current", limits=small_limits())
            with self.assertRaisesRegex(RemoteError, "after execution; do not replay"):
                await client.call("generate", {"size": 4096})
            self.assertEqual(self.service.calls, ["generate"])
            self.assertEqual(server.blobs.used, 0)

    async def test_cancelled_upload_never_dispatches_later(self):
        original = self.client.upload
        def delay_upload(body, budget):
            time.sleep(.1)
            return original(body, budget)
        with patch.object(self.client, "upload", delay_upload):
            with self.assertRaises(Unreachable):
                await self.client.call("load", {"bytes": "x" * 4000}, deadline_s=.03)
            await asyncio.sleep(.2)
        self.assertEqual(self.service.calls, [])
        self.assertEqual(self.server.blobs.used, 0)

    async def test_corrupt_result_is_rejected_without_replaying(self):
        original = self.server.blobs.produce
        def corrupt(value):
            ref = original(value)
            if ref.size > self.server.limits.inline_bytes:
                path = self.server.blobs.root / ref.id
                path.write_bytes(path.read_bytes().replace(b"zzzz", b"xxxx", 1))
            return ref
        with patch.object(self.server.blobs, "produce", corrupt):
            with self.assertRaisesRegex(RemoteError, "checksum mismatch"):
                await self.client.call("generate", {"size": 4096})
        self.assertEqual(self.service.calls, ["generate"])
        self.assertEqual(self.server.blobs.used, 0)

    async def test_concurrent_claim_dispatches_an_uploaded_request_once(self):
        data = json.dumps({"bytes": "hello", "@epoch": "current"}).encode()
        ref = await self.reserve(data)
        await self.put(ref, data)
        frame = {"host": "learner", "verb": "load", "payload_blob": ref.row(), "deadline_s": 3}
        replies = await asyncio.gather(*(self.control("POST", "/call", frame) for _ in range(2)),
                                       return_exceptions=True)
        self.assertEqual(sum(isinstance(row, dict) for row in replies), 1)
        self.assertEqual(sum(isinstance(row, RemoteError) for row in replies), 1)
        self.assertEqual(self.service.calls, ["load"])

    async def test_spool_byte_quota_refuses_upload_then_recovers_after_delete(self):
        limits = small_limits(max_blob_bytes=2048, spool_bytes=4096)
        async with HttpServer(lambda host: self.service, port=0, limits=limits) as server:
            client = HttpTransport(server.endpoint, epoch="current", limits=limits)
            row = {"size": 2048, "sha256": "0" * 64}
            refs = [await asyncio.to_thread(client.control, "POST", "/blobs", Budget(3), row)
                    for _ in range(2)]
            self.assertEqual(server.blobs.used, 4096)
            with self.assertRaises(RemoteError):
                await asyncio.to_thread(client.control, "POST", "/blobs", Budget(3), row)
            # A result that cannot be spooled still executes only once and says so.
            with self.assertRaisesRegex(RemoteError, "after execution; do not replay"):
                await client.call("generate", {"size": 1200})
            self.assertEqual(self.service.calls, ["generate"])
            for item in refs:
                await asyncio.to_thread(client.discard, BlobRef.decode(item["blob"], 2048))
            self.assertEqual(server.blobs.used, 0)
            await asyncio.to_thread(client.control, "POST", "/blobs", Budget(3), row)

    async def test_network_handler_count_is_bounded_and_recovers(self):
        async with HttpServer(lambda host: self.service, port=0,
                              limits=small_limits(max_connections=1)) as server:
            port = int(server.endpoint.rsplit(":", 1)[1])
            client = HttpTransport(server.endpoint)
            held = socket.create_connection(("127.0.0.1", port), timeout=2)
            try:
                held.sendall(b"GET /health HTTP/1.0\r\n")  # deliberately incomplete headers
                await asyncio.sleep(.05)
                with self.assertRaises(Unreachable):
                    await asyncio.to_thread(client.control, "GET", "/health", Budget(.5))
            finally:
                held.close()
            await asyncio.sleep(.05)
            self.assertTrue((await asyncio.to_thread(client.control, "GET", "/health", Budget(2)))["ready"])


class OperatorLimitTests(unittest.TestCase):
    def test_client_spool_is_shared_and_encode_failures_release_it(self):
        limits = small_limits(max_blob_bytes=2048, spool_bytes=4096)
        with ClientFile(limits) as first, ClientFile(limits) as second, ClientFile(limits) as third:
            first.append(b"a" * 2048)
            second.append(b"b" * 2048)
            with self.assertRaisesRegex(ValueError, "client transfer spool quota"):
                third.append(b"c")
            first.close()
            third.append(b"c")
        with self.assertRaises(TypeError):
            with ClientFile(limits) as broken:
                broken.encode({"good": "x" * 1024, "bad": object()})
        with ClientFile(limits) as after:
            after.append(b"z" * 2048)

    def test_environment_sets_factory_limits_and_worker_children(self):
        from rlstack.runner.venues.strangeloop.worker import WorkerConfig
        environment = {"SL_API_TOKEN": "test-only", "SL_API_BASE": "https://invalid.test",
                       "RLSTACK_HTTP_TOKEN": "test-only", "RLSTACK_HTTP_MAX_BLOB_BYTES": str(4 * 1024**3),
                       "RLSTACK_HTTP_SPOOL_BYTES": str(16 * 1024**3)}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, environment):
            path = Path(directory) / "worker.json"
            path.write_text(json.dumps({"name": "worker", "store": "store", "desk": "desk",
                                        "address": "address", "port": 8000, "idle_s": 90,
                                        "lease_id": "lease", "artifact_dir": "/tmp/unused",
                                        "source_sha256": "source", "builds": None,
                                        "environment": environment}))
            WorkerConfig.read(path)
            self.assertFalse(path.exists())
            limits = HttpTransport("http://127.0.0.1:1").limits
            self.assertEqual(limits.max_blob_bytes, 4 * 1024**3)
            self.assertEqual(limits.spool_bytes, 16 * 1024**3)
        with patch.dict(os.environ, {"RLSTACK_HTTP_MAX_BLOB_BYTES": "-1"}):
            with self.assertRaises(ValueError):
                HttpLimits.from_environment()


class GatewayBulkTests(unittest.IsolatedAsyncioTestCase):
    async def gateway(self, worker, service):
        port = int(worker.endpoint.rsplit(":", 1)[1])
        config = DeskConfig("strangeloop://sl-scratch-account/rlstack",
                            (MetalConfig("alpha", "A100", "image", port),), Path("/tmp/unused"))
        gateway = Gateway(config, token="private")
        return HttpServer(gateway.service_for, port=0, token="private")

    async def test_request_and_result_above_old_64_mib_limit_across_both_hops(self):
        service = Echo()
        async with HttpServer(lambda host: service, port=0, token="private") as worker:
            async with await self.gateway(worker, service) as gateway:
                client = HttpTransport(gateway.endpoint, "alpha/learner", "current", token="private")
                # A real >64 MiB JSON request AND response crosses the local gateway.
                data = "0123456789abcdef" * (65 * 1024**2 // 16)
                reply = await client.call("load", {"bytes": data}, deadline_s=60)
                self.assertEqual(len(reply["bytes"]), len(data))
                self.assertEqual(hashlib.sha256(reply["bytes"].encode()).digest(),
                                 hashlib.sha256(data.encode()).digest())
                self.assertEqual(service.calls, ["load"])
                self.assertEqual(worker.blobs.used, 0)
                self.assertEqual(gateway.blobs.used, 0)

    async def test_gateway_deadline_does_not_expand_to_build_timeout(self):
        service = Echo()
        async with HttpServer(lambda host: service, port=0, token="private") as worker:
            async with await self.gateway(worker, service) as gateway:
                client = HttpTransport(gateway.endpoint, "alpha/learner", "current", token="private")
                with self.assertRaises(Unreachable):
                    await client.call("slow", {"delay": .2}, deadline_s=.04)
                await asyncio.sleep(.25)
                self.assertEqual(service.calls, [])
