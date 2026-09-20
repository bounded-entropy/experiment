"""Opt-in scratch drill; allocates no compute and touches only a UUID child prefix.

RLSTACK_STRANGELOOP_LIVE_STORE=strangeloop://sl-scratch-ACCOUNT/rlstack \\
  python3.13 -m unittest discover -s tests -p test_strangeloop_store_live.py

On an existing authorized worker also set RLSTACK_STRANGELOOP_LIVE_MOUNT=/scratch
for mounted publication. This does not establish abrupt-crash/owner-handoff safety.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from rlstack.data.stores.strangeloop import (
    HashedReads, ScratchClient, StrangeLoopLocalStore, StrangeLoopStore,
)


@unittest.skipUnless(os.environ.get("RLSTACK_STRANGELOOP_LIVE_STORE"),
                     "live scratch drill requires an explicit store locator")
class LiveScratch(unittest.TestCase):
    def setUp(self):
        original = ScratchClient.from_locator(os.environ["RLSTACK_STRANGELOOP_LIVE_STORE"])
        prefix = original.prefix + "/.live-drill/" + uuid.uuid4().hex
        self.client = ScratchClient(original.volume, prefix,
                                    token=original.credentials.token,
                                    api_base=original.credentials.api_base)
        self.writer = StrangeLoopStore(self.client)
        self.reader = StrangeLoopStore(ScratchClient.from_locator(self.client.locator), read_only=True)

    def tearDown(self):
        for entry in self.client.list("", recursive=True):
            if entry.type == "file":
                self.client.delete(entry.path)

    def test_api_same_size_refresh_and_journal_append(self):
        self.writer._write("marker", b"v1")
        self.assertEqual(self.reader._read("marker"), b"v1")
        self.writer._write("marker", b"v2")
        self.assertEqual(self.reader._read("marker"), b"v2")
        self.writer._append_line("journal.jsonl", '{"update":0}')
        self.writer._append_line("journal.jsonl", '{"update":1}')
        self.assertEqual(self.reader._read("journal.jsonl"), b'{"update":0}\n{"update":1}\n')
        self.writer._delete("marker")
        self.assertFalse(self.reader._exists("marker"))

    def test_host_journals_preserve_logical_names(self):
        names = ("sl-qwen4b:0.main-tp1.c1", "sl-qwen4b@3a0.main-tp1.c1")
        for name in names:
            self.writer.append_host_event(name, {"event": "probe", "host": name})
            self.assertEqual(self.reader.read_host_log(name)[0]["host"], name)
        self.assertEqual(self.reader.list_hosts(), sorted(names))

    def test_hashed_api_publication_and_shared_cache(self):
        payload = os.urandom(4 * 1024 * 1024)
        with patch.object(self.client, "read", wraps=self.client.read) as reads:
            self.writer.write_named("family", "adapter", payload, {"drill": "ADR0020"})
            self.assertNotIn(unittest.mock.call("runs/family/names/adapter.bin"), reads.call_args_list)
        with tempfile.TemporaryDirectory() as directory:
            settings = HashedReads(Path(directory), 8 * 1024 * 1024)
            reader = StrangeLoopStore(self.client, hashed_reads=settings, read_only=True)
            self.assertEqual(reader.read_named("family", "adapter"), payload)
            # A separate store rechecks the seal, then shares the local payload.
            second = StrangeLoopStore(self.client, hashed_reads=settings, read_only=True)
            with patch.object(self.client, "read", wraps=self.client.read) as reads:
                self.assertEqual(second.read_named("family", "adapter"), payload)
                self.assertEqual(reads.call_args_list, [unittest.mock.call("runs/family/names/adapter.json")])

    @unittest.skipUnless(os.environ.get("RLSTACK_STRANGELOOP_LIVE_MOUNT"),
                         "mounted drill must run on an existing mounted worker")
    def test_hashed_mounted_publication_and_http_fallback_without_refresh(self):
        mount = Path(os.environ["RLSTACK_STRANGELOOP_LIVE_MOUNT"])
        with tempfile.TemporaryDirectory() as directory:
            settings = HashedReads(Path(directory), 8 * 1024 * 1024, True)
            mounted = StrangeLoopLocalStore(mount / self.client.prefix, self.client,
                                            mountpoint=mount, hashed_reads=settings)
            mounted.verify_publication()
            payload = os.urandom(4 * 1024 * 1024)
            with patch.object(self.client, "read", wraps=self.client.read) as reads:
                uri = mounted.cas_put(payload)
                self.assertEqual(reads.call_args_list, [])
                self.assertEqual(mounted.cas_get(uri), payload)
                self.assertEqual(reads.call_args_list, [])
            # The independent HTTP view confirms actual publication.
            self.assertEqual(self.reader.cas_get(uri), payload)
            other = os.urandom(1024 * 1024)
            remote_uri = self.writer.cas_put(other)
            # Missing or wrong mounted bytes must not outrank the committed copy.
            key = "cas/" + remote_uri.removeprefix("cas://") + "/blob"
            path = mounted.path_of(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stale")
            try:
                self.assertEqual(mounted.cas_get(remote_uri), other)
            finally:
                # Never publish the intentionally corrupt local fixture.
                path.write_bytes(other)
                mounted._sync()

    @unittest.skipUnless(os.environ.get("RLSTACK_STRANGELOOP_LIVE_MOUNT"),
                         "mounted drill must run on an existing mounted worker")
    def test_mounted_publication_then_authoritative_refresh(self):
        mount = Path(os.environ["RLSTACK_STRANGELOOP_LIVE_MOUNT"])
        mounted = StrangeLoopLocalStore(mount / self.client.prefix, self.client, mountpoint=mount)
        mounted.verify_publication()
        mounted._write("mounted", b"local-v1")
        self.assertEqual(self.reader._read("mounted"), b"local-v1")
        self.writer._write("mounted", b"other-v2")
        self.assertEqual(mounted._read("mounted"), b"other-v2")
        mounted._delete("mounted")
        self.assertFalse(self.reader._exists("mounted"))
