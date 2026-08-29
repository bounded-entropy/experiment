"""Restore and eviction: residency is not durable, and does not have to be.

The pair under test is one rule read from both ends — a bounded pool may drop
what it holds (detach), and anything it dropped comes back from the store with
the content-addressed id as the proof (restore).
"""

from __future__ import annotations

import unittest

from rlstack import Bundle, compile_bundle
from rlstack.policy.compile import restore_bundle
from rlstack.runner.fakes import FakeEngine
from rlstack.runner.restore import restore_bundle_on, restore_tenant


class Blobs:
    """The store's blob section, as the callable restore actually takes."""

    def __init__(self, **payloads: bytes) -> None:
        self.written = {("adapters", name, 0): data
                        for name, data in payloads.items()}
        self.reads: list[tuple] = []

    def read_blob(self, section: str, name: str, version: int) -> bytes:
        self.reads.append((section, name, version))
        key = (section, name, version)
        if key not in self.written:
            raise FileNotFoundError(f"no {section} blob {name}@{version}")
        return self.written[key]


class RestoreBundleTest(unittest.TestCase):
    def test_a_committed_version_rebuilds_to_the_same_id(self) -> None:
        blobs = Blobs(pi=b"delta-bytes")
        committed = compile_bundle({"pi": b"delta-bytes"}, {"pi": 0}, ["pi"],
                                   {"pi": "lora"})
        restored = restore_bundle({"pi": 0}, committed.bundle_id,
                                  blobs.read_blob, ["pi"], {"pi": "lora"})
        self.assertEqual(restored.bundle_id, committed.bundle_id)
        self.assertEqual(restored.payloads, committed.payloads)

    def test_blobs_that_do_not_compile_to_the_pin_are_refused(self) -> None:
        """The id IS the proof: different bytes at the same version map is a
        different policy, and restoring it silently would serve a lie."""
        blobs = Blobs(pi=b"other-bytes")
        with self.assertRaises(ValueError) as caught:
            restore_bundle({"pi": 0}, "bundle:deadbeef", blobs.read_blob,
                           ["pi"], {"pi": "lora"})
        self.assertIn("not the committed", str(caught.exception))

    def test_a_frozen_servable_delta_is_restorable(self) -> None:
        """The completeness rule: every servable entry has a blob, including the
        ones that never advance, or restore is total only by luck."""
        blobs = Blobs(pi=b"trained", ref=b"frozen")
        committed = compile_bundle({"pi": b"trained", "ref": b"frozen"},
                                   {"pi": 0, "ref": 0}, ["pi", "ref"])
        restored = restore_bundle({"pi": 0, "ref": 0}, committed.bundle_id,
                                  blobs.read_blob, ["pi", "ref"])
        self.assertEqual(restored.bundle_id, committed.bundle_id)


class RestoreOnEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.blobs = Blobs(pi=b"delta-bytes")
        self.bundle = compile_bundle({"pi": b"delta-bytes"}, {"pi": 0}, ["pi"])
        self.engine = FakeEngine()

    def test_a_resident_bundle_costs_no_store_read(self) -> None:
        self.engine.add_bundle(self.bundle)
        restore_bundle_on(self.engine, Bundle.pin(self.bundle.bundle_id,
                                                  {"pi": 0}),
                          self.blobs.read_blob, ["pi"])
        self.assertEqual(self.blobs.reads, [])

    def test_a_pool_that_lost_it_gets_it_back_from_a_pin(self) -> None:
        """A pin carries the id and the version map and no payloads — exactly
        what a request carries and a ledger line records."""
        restore_bundle_on(self.engine, Bundle.pin(self.bundle.bundle_id,
                                                  {"pi": 0}),
                          self.blobs.read_blob, ["pi"])
        self.assertTrue(self.engine.knows_bundle(self.bundle.bundle_id))
        self.assertEqual(self.blobs.reads, [("adapters", "pi", 0)])


class Loaded:
    """A learner that records only what restore handed it."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def load(self, tenant, adapters, optim) -> None:
        self.calls.append((tenant, dict(adapters), dict(optim)))


class RestoreTenantTest(unittest.TestCase):
    def test_a_tenant_comes_back_with_its_moments(self) -> None:
        """A learner keeps training, so it needs the optimizer state a serving
        engine never does — and only for the entries that actually advance."""
        blobs = Blobs(pi=b"delta-bytes", ref=b"frozen")
        blobs.written[("optim", "pi", 0)] = b"moments"
        learner = Loaded()
        restore_tenant(learner, "t1", {"pi": 0, "ref": 0}, blobs.read_blob,
                       ["pi"])
        tenant, adapters, optim = learner.calls[0]
        self.assertEqual(tenant, "t1")
        self.assertEqual(adapters, {"pi": b"delta-bytes"})
        self.assertEqual(optim, {"pi": b"moments"})
        # the frozen entry is the engine's business, never the learner's
        self.assertNotIn(("adapters", "ref", 0), blobs.reads)


if __name__ == "__main__":
    unittest.main()
