"""The EnginePlugin contract (rlstack_engine): lifecycle, slots, multi-tenant
routing — everything a plugin must re-earn, tested before any engine exists.

Numerics (attend, prefix K/V) are B3; what is tested here is the part that
was settled in design: probe dies at boot on a missing seam, banks are
per-slot (never a global singleton), tokens gather from their own bundle's
slot, and cache identity is contributed per bundle.
"""

from __future__ import annotations

import unittest

from rlstack import Mechanism
from rlstack_engine import (
    BatchView, Certificate, CertificateKey, FakeSeam, InMemoryCertificates,
    ProbeError, SideAttention, SlotTable, SlotsFull, fake_build,
)
from rlstack_engine.batch_view import BatchViewError
from rlstack_engine.plugin import EnginePlugin

GOOD_BUILD = fake_build(*sorted(SideAttention.required_symbols))

# What vllm 0.28.0 actually offers (#46, checked in the pinned image): the
# registry moved under v1 and still exists, merge_attn_states moved and still
# exists, and the dense FlashAttention path hands back no LSE at all. This is
# the build the fleet runs on today, and the probe must refuse it.
PINNED_0_28 = fake_build(
    "vllm.v1.attention.backends.registry.register_backend",
    "vllm.v1.attention.ops.merge_attn_states.merge_attn_states")


class ProbeTest(unittest.TestCase):
    def test_probe_passes_when_every_seam_exists(self) -> None:
        SideAttention().probe(GOOD_BUILD)  # no raise

    def test_probe_names_the_missing_seams(self) -> None:
        bare = fake_build("vllm.v1.attention.backends.registry.register_backend")
        with self.assertRaises(ProbeError) as caught:
            SideAttention().probe(bare)
        self.assertIn("merge_attn_states", str(caught.exception))
        self.assertIn(bare.fingerprint, str(caught.exception))

    def test_the_pinned_build_is_refused_for_the_reason_it_lacks(self) -> None:
        """#46's honest status, pinned as a test: side attention does not serve
        on vllm 0.28.0, and the one thing missing is the dense LSE."""
        with self.assertRaises(ProbeError) as caught:
            SideAttention().probe(PINNED_0_28)
        self.assertIn("dense_lse", str(caught.exception))
        self.assertNotIn("register_backend", str(caught.exception))
        self.assertNotIn("merge_attn_states", str(caught.exception))

    def test_install_claims_the_seam_once(self) -> None:
        seam = FakeSeam()
        plugin = SideAttention()
        plugin.install(seam)
        self.assertEqual(seam.claimed, [plugin])
        with self.assertRaises(RuntimeError):
            SideAttention().install(seam)


class SlotTableTest(unittest.TestCase):
    def test_acquire_is_idempotent_and_slots_are_distinct(self) -> None:
        table = SlotTable(max_slots=4)
        a = table.acquire("bundle:a")
        b = table.acquire("bundle:b")
        self.assertNotEqual(a, b)
        self.assertEqual(table.acquire("bundle:a"), a)
        self.assertEqual(table.slot_of("bundle:b"), b)

    def test_release_frees_the_slot_for_reuse(self) -> None:
        table = SlotTable(max_slots=1)
        slot = table.acquire("bundle:a")
        with self.assertRaises(SlotsFull):
            table.acquire("bundle:b")
        self.assertEqual(table.release("bundle:a"), slot)
        self.assertEqual(table.acquire("bundle:b"), slot)

    def test_unknown_bundle_is_loud(self) -> None:
        with self.assertRaises(KeyError):
            SlotTable(max_slots=1).slot_of("bundle:never")


class SideAttentionLifecycleTest(unittest.TestCase):
    """The composite mechanism: one plugin, both adapter kinds' payloads."""

    def test_declares_the_joint_consumption(self) -> None:
        self.assertEqual(SideAttention.mechanism, Mechanism.SIDE_ATTENTION)
        self.assertEqual(SideAttention.consumes, ("soft_prompt", "attn_bias"))

    def test_banks_are_per_slot(self) -> None:
        plugin = SideAttention()
        plugin.load(0, "bundle:a", {"latent": b"E-of-a", "readout": b"bias-of-a"})
        plugin.load(1, "bundle:b", {"latent": b"E-of-b", "readout": b"bias-of-b"})
        self.assertEqual(plugin.bank(0)["readout"], b"bias-of-a")
        self.assertEqual(plugin.bank(1)["readout"], b"bias-of-b")

    def test_loading_an_occupied_slot_is_refused(self) -> None:
        plugin = SideAttention()
        plugin.load(0, "bundle:a", {"latent": b"E"})
        with self.assertRaises(ValueError):
            plugin.load(0, "bundle:b", {"latent": b"E2"})

    def test_evict_frees_the_slot(self) -> None:
        plugin = SideAttention()
        plugin.load(0, "bundle:a", {"latent": b"E"})
        plugin.evict(0)
        plugin.load(0, "bundle:b", {"latent": b"E2"})
        self.assertEqual(plugin.bank(0)["latent"], b"E2")

    def test_cache_salt_is_the_bundle(self) -> None:
        """The plugin changes hidden states, so bundle identity MUST reach the
        prefix-cache key — the wrong-KV-reuse guard."""
        self.assertEqual(SideAttention().cache_salt("bundle:a"), "bundle:a")

    def test_attend_numerics_are_b3(self) -> None:
        view = BatchView(token_slot=(0,), layer_idx=0, is_decode=(True,))
        with self.assertRaises(NotImplementedError):
            SideAttention().attend(view, None, None, None)


class _RecordingPlugin(EnginePlugin):
    """attend() with the real gather shape: each token reads its own slot's
    bank. Records what each token read, so routing is assertable."""

    mechanism = Mechanism.SIDE_ATTENTION
    consumes = ("soft_prompt",)

    def __init__(self) -> None:
        self._banks: dict[int, bytes] = {}
        self.reads: list[bytes] = []

    def load(self, slot: int, bundle_id: str, payloads) -> None:
        self._banks[slot] = payloads["latent"]

    def evict(self, slot: int) -> None:
        del self._banks[slot]

    def attend(self, view: BatchView, q, out, lse) -> None:
        for slot in view.token_slot:
            self.reads.append(self._banks[slot])


class MultiTenantRoutingTest(unittest.TestCase):
    def test_interleaved_tokens_read_their_own_banks(self) -> None:
        """Two experiments' tokens in ONE batch: every token gathers from the
        slot its request's bundle pinned — the engine-level twin of
        test_two_experiments_share_one_engine, at plugin scope."""
        plugin = _RecordingPlugin()
        table = SlotTable(max_slots=4)
        for bundle, payload in (("bundle:a", b"E-a"), ("bundle:b", b"E-b")):
            plugin.load(table.acquire(bundle), bundle, {"latent": payload})

        a, b = table.slot_of("bundle:a"), table.slot_of("bundle:b")
        view = BatchView(token_slot=(a, b, b, a, b), layer_idx=3,
                         is_decode=(True,) * 5)
        plugin.attend(view, None, None, None)
        self.assertEqual(plugin.reads, [b"E-a", b"E-b", b"E-b", b"E-a", b"E-b"])


class BatchViewTest(unittest.TestCase):
    def test_columns_must_align(self) -> None:
        with self.assertRaises(BatchViewError):
            BatchView(token_slot=(0, 1), layer_idx=0, is_decode=(True,))

    def test_the_vllm_shim_is_b3(self) -> None:
        with self.assertRaises(NotImplementedError):
            BatchView.from_vllm(None, 0, {})


class CertificateTest(unittest.TestCase):
    def test_roundtrip_and_miss(self) -> None:
        cache = InMemoryCertificates()
        key = CertificateKey("build:fake0001", "Qwen/Qwen3-0.6B", "attn_bias",
                             Mechanism.SIDE_ATTENTION)
        self.assertIsNone(cache.get(key))
        cache.put(Certificate(key, passed=True, detail="fake"))
        self.assertTrue(cache.get(key).passed)

    def test_a_new_build_fingerprint_misses(self) -> None:
        """Bump the image → the certificate no longer answers: re-certify."""
        cache = InMemoryCertificates()
        old = CertificateKey("build:fake0001", "b", "attn_bias",
                             Mechanism.SIDE_ATTENTION)
        cache.put(Certificate(old, passed=True, detail="fake"))
        bumped = CertificateKey("build:fake0002", "b", "attn_bias",
                                Mechanism.SIDE_ATTENTION)
        self.assertIsNone(cache.get(bumped))


if __name__ == "__main__":
    unittest.main()
