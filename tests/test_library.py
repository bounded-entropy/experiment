"""Generic named-state storage, bounded residency, and binary wire transport."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

from common import arith_store
from rlstack import Bundle, FakeEngine, FakeLearner, Message, Role, SamplingSpec
from rlstack.policy.adapters.rollout import (
    Levers, Library, Request, ServingBuild, check_rank_fits,
)
from rlstack.runner.host import Host, Partition, Regime
from rlstack.runner.remote import HostService, LocalTransport, RemotePool
from rlstack.runner.residents import (
    EngineBuild, FakeEngineBuild, Resident, ResidentBirth, build_engine,
    decode_build, encode_build,
)



def go(coro):
    return asyncio.run(coro)





class LibraryTest(unittest.TestCase):
    def test_add_is_idempotent_and_a_name_is_write_once(self):
        library = Library(4)
        library.add("a", b"one")
        library.add("a", b"one")
        self.assertEqual(library.log, ["a"])
        with self.assertRaisesRegex(ValueError, "write-once"):
            library.add("a", b"two")
        self.assertEqual(library.get("a"), b"one")

    def test_the_least_recently_used_name_is_evicted_and_forgotten(self):
        forgotten: list[str] = []
        library = Library(2, forgotten.append)
        library.add("a", b"a")
        library.add("b", b"b")
        library.get("a")                                  # a is now the recent one
        library.add("c", b"c")
        self.assertEqual(forgotten, ["b"])
        self.assertEqual(library.names(), ("a", "c"))
        library.add("b", b"b")                            # back from the store: not a loss
        self.assertEqual(forgotten, ["b", "a"])

    def test_a_pinned_name_is_immune_and_does_not_shield_the_stale(self):
        forgotten: list[str] = []
        library = Library(2, forgotten.append)
        library.add("a", b"a")
        library.add("b", b"b")
        with library.pinned(("a",)):
            library.add("c", b"c")
            self.assertEqual(forgotten, ["b"])
            library.add("d", b"d")                        # soft by the in-flight set
            self.assertEqual(forgotten, ["b", "c"])
            self.assertTrue(library.knows("a"))
        library.add("e", b"e")
        self.assertEqual(forgotten, ["b", "c", "a"])

    def test_a_name_nobody_handed_in_is_refused_by_read_and_by_pin(self):
        library = Library(2)
        with self.assertRaisesRegex(RuntimeError, "add_library"):
            library.get("ghost")
        with self.assertRaisesRegex(RuntimeError, "'ghost'"):
            with library.pinned(("ghost",)):
                pass

    def test_levers_join_the_names_they_read(self):
        merged = Levers(library=("a",)).merged_with(Levers(library=("b",)))
        self.assertEqual(merged.library, ("a", "b"))

    def test_a_delta_wider_than_the_build_is_refused_with_both_numbers(self):
        build = ServingBuild(base="b", config=None, workdir=Path("."), max_bundles=2, max_rank=32)
        check_rank_fits("adapter 'x'", 32, build)
        with self.assertRaisesRegex(ValueError, r"rank 64.*max_rank 32"):
            check_rank_fits("adapter 'x'", 64, build)



class WireTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, _, _ = arith_store(tmp.name)
        self.payload = bytes(range(256)) * 3              # not text: base64 or bust

    def test_add_library_crosses_the_host_wire_as_bytes(self):
        engine = FakeEngine()
        host = Host("srv", engines=(engine,), learner=FakeLearner(), store=self.store)
        remote = RemotePool(LocalTransport(HostService(host)))
        remote.add_library("named/001", self.payload)
        remote.add_library("named/001", self.payload)
        self.assertEqual(engine.library.log, ["named/001"])
        self.assertEqual(engine.library.get("named/001"), self.payload)

    def test_add_library_crosses_a_resident_door(self):
        engine = FakeEngine(base="Qwen/Qwen3-0.6B")
        serve = Regime("serve", "inference", "Qwen/Qwen3-0.6B", 1)
        birth = ResidentBirth(label="alt:serve", partition=Partition("fake-metal", (0,), 1.0, "L4"),
                              regime=serve, build=FakeEngineBuild(), store=self.store.address())
        pool = RemotePool(Resident.in_process(birth, engine).transport, base="Qwen/Qwen3-0.6B")
        pool.add_library("mem/001", self.payload)
        self.assertEqual(engine.library.get("mem/001"), self.payload)

    def test_a_build_record_carries_max_library(self):
        self.assertEqual(EngineBuild().max_library, 32)
        for build in (EngineBuild(max_library=5, serves=("lora",)),
                      FakeEngineBuild(max_library=5)):
            row = json.loads(json.dumps(encode_build(build)))
            self.assertEqual(row["max_library"], 5)
            self.assertEqual(decode_build(row), build)
        old_row = {k: v for k, v in encode_build(EngineBuild()).items() if k != "max_library"}
        self.assertEqual(decode_build(old_row).max_library, 32)
        engine = build_engine(Regime("serve", "inference", "b", 1),
                              Partition("m", (0,), 1.0, "L4"),
                              FakeEngineBuild(max_library=5), self.store)
        self.assertEqual(engine.library.capacity, 5)
