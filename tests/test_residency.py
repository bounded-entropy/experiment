"""The eviction rule (rlstack.runner.residency): one bound, one immunity."""

from __future__ import annotations

import unittest

from rlstack.runner.residency import BundleResidency


class Detached:
    """Records what residency handed back, in order."""

    def __init__(self) -> None:
        self.released: list[str] = []

    def __call__(self, attached) -> None:
        self.released.append(attached["lora"])


def residency(max_bundles: int) -> tuple[BundleResidency, Detached]:
    detached = Detached()
    return BundleResidency(max_bundles, detached), detached


class BoundTest(unittest.TestCase):
    def test_holding_past_the_bound_evicts_least_recently_used_first(self) -> None:
        held, detached = residency(2)
        for name in ("a", "b", "c"):
            held.hold(name, {"lora": name})
        self.assertEqual(detached.released, ["a"])
        self.assertFalse(held.knows("a"))
        self.assertTrue(held.knows("b") and held.knows("c"))

    def test_use_makes_a_bundle_recent(self) -> None:
        """The candidate is the least recently USED, not the oldest held —
        a lagging generator still sampling an older version keeps it."""
        held, detached = residency(2)
        held.hold("a", {"lora": "a"})
        held.hold("b", {"lora": "b"})
        held.touch("a")
        held.hold("c", {"lora": "c"})
        self.assertEqual(detached.released, ["b"])

    def test_bundles_carrying_no_state_neither_count_nor_evict(self) -> None:
        """A bare-base pool registers a payload-less bundle; dropping it would
        take a servable id away and give no memory back."""
        held, detached = residency(1)
        held.hold("base", {})
        held.hold("a", {"lora": "a"})
        held.hold("b", {"lora": "b"})
        self.assertEqual(detached.released, ["a"])
        self.assertTrue(held.knows("base"))


class ImmunityTest(unittest.TestCase):
    def test_a_pinned_bundle_is_never_evicted(self) -> None:
        """I8: a request's policy is immune once submitted, however old."""
        held, detached = residency(1)
        held.hold("a", {"lora": "a"})
        with held.pinned("a"):
            held.hold("b", {"lora": "b"})
            self.assertEqual(detached.released, [])   # the bound goes soft
            self.assertTrue(held.knows("a"))

    def test_the_bound_returns_once_the_request_is_out(self) -> None:
        held, detached = residency(1)
        held.hold("a", {"lora": "a"})
        with held.pinned("a"):
            held.hold("b", {"lora": "b"})
        held.evict_until_within_bound()
        self.assertEqual(detached.released, ["a"])

    def test_concurrent_requests_on_one_bundle_all_have_to_finish(self) -> None:
        held, detached = residency(1)
        held.hold("a", {"lora": "a"})
        with held.pinned("a"):
            with held.pinned("a"):
                pass
            held.hold("b", {"lora": "b"})
            self.assertEqual(detached.released, [])   # one request still out


class CensusTest(unittest.TestCase):
    def test_the_census_counts_resident_bundles_per_adapter_type(self) -> None:
        held, _ = residency(8)
        held.hold("a", {"lora": "a"})
        held.hold("b", {"lora": "b", "soft_prompt": "b"})
        self.assertEqual(dict(held.census()), {"lora": 2, "soft_prompt": 1})


if __name__ == "__main__":
    unittest.main()
