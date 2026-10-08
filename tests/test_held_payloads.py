"""The learner door's held payloads: a load_set payload a proxy sent
recently crosses as its digest alone; a door that no longer holds it says
`missing` and the bytes follow; the learner always receives the bytes."""

from __future__ import annotations

import unittest

from rlstack import FakeLearner
from rlstack.runner.remote import (
    HeldPayloads, LearnerService, LocalTransport, RemoteLearner, payload_digest,
)


class RecordingLearner(FakeLearner):
    def __init__(self) -> None:
        super().__init__()
        self.loaded: list[bytes | None] = []

    def load_set(self, tenant, entry, route, payload):
        self.loaded.append(payload)


class CountingService(LearnerService):
    def __init__(self, learner) -> None:
        super().__init__(learner)
        self.frames: list[dict] = []

    def answer(self, verb, payload):
        if verb == "load_set":
            self.frames.append(dict(payload))
        return super().answer(verb, payload)


class HeldPayloadsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.learner = RecordingLearner()
        self.service = CountingService(self.learner)
        self.proxy = RemoteLearner(LocalTransport(self.service))

    def test_the_same_bytes_cross_once_and_load_every_time(self) -> None:
        payload = b"x" * 1000
        for route in ("memory:00", "memory:01", "memory:02"):
            self.proxy.load_set("t", "forks", route, payload)
        self.assertEqual(self.learner.loaded, [payload] * 3)
        self.assertEqual([frame["payload"] is not None for frame in self.service.frames],
                         [True, False, False])
        self.assertEqual({frame["digest"] for frame in self.service.frames}, {payload_digest(payload)})

    def test_a_door_that_forgot_says_missing_and_the_bytes_follow(self) -> None:
        payload = b"y" * 100
        self.proxy.load_set("t", "forks", "memory:00", payload)
        self.service.held = HeldPayloads()                    # a new process behind the door
        self.proxy.load_set("t", "forks", "memory:01", payload)
        self.assertEqual(self.learner.loaded, [payload, payload])
        self.assertEqual([frame["payload"] is not None for frame in self.service.frames],
                         [True, False, True])

    def test_none_is_init_and_never_held(self) -> None:
        self.proxy.load_set("t", "forks", "memory:00", None)
        self.assertEqual(self.learner.loaded, [None])
        self.assertNotIn("digest", self.service.frames[0])

    def test_the_door_holds_a_bounded_few_and_refuses_a_wrong_digest(self) -> None:
        held = HeldPayloads(bound=2)
        for k in range(3):
            held.put(payload_digest(bytes([k])), bytes([k]))
        self.assertIsNone(held.get(payload_digest(bytes([0]))))
        self.assertEqual(held.get(payload_digest(bytes([2]))), bytes([2]))
        with self.assertRaises(ValueError):
            held.put(payload_digest(b"a"), b"b")


if __name__ == "__main__":
    unittest.main()


class LibraryOnceTest(unittest.TestCase):
    """add_library through a RemotePool: a name handed over before crosses as
    its digest; an engine that evicted it says missing and the bytes follow."""

    def setUp(self) -> None:
        from rlstack.runner.fakes import FakeEngine
        from rlstack.runner.remote import EngineService, RemotePool

        self.engine = FakeEngine(max_library=1)
        self.frames: list[dict] = []
        frames = self.frames

        class Counting(EngineService):
            def answer(self, verb, payload):
                if verb == "add_library":
                    frames.append(dict(payload))
                return super().answer(verb, payload)

        self.pool = RemotePool(LocalTransport(Counting(self.engine)), base="fake/base")

    def test_a_held_name_crosses_as_its_digest(self) -> None:
        for _ in range(3):
            self.pool.add_library("mem/a", b"A" * 64)
        self.assertEqual([frame["payload"] is not None for frame in self.frames], [True, False, False])
        self.assertTrue(self.pool.holds_library("mem/a", payload_digest(b"A" * 64)))
        self.assertFalse(self.pool.holds_library("mem/a", payload_digest(b"B")))

    def test_an_evicted_name_is_sent_again(self) -> None:
        self.pool.add_library("mem/a", b"A" * 64)
        self.pool.add_library("mem/b", b"B" * 64)          # a one-name library evicts mem/a
        self.pool.add_library("mem/a", b"A" * 64)
        self.assertEqual([frame["payload"] is not None for frame in self.frames], [True, True, False, True])
        self.assertTrue(self.engine.holds_library("mem/a", payload_digest(b"A" * 64)))
