"""Bundle compilation and the mechanism dispatch input (rlstack.policy.compile)."""

from __future__ import annotations

import unittest

from rlstack import Bundle, Mechanism, compile_bundle, group_by_mechanism


class CompileBundleTest(unittest.TestCase):
    def test_kinds_label_only_shipped_payloads(self) -> None:
        bundle = compile_bundle(
            payloads={"pi": b"delta", "critic": b"head"},
            policy_version={"pi": 3, "critic": 3},
            servable=["pi"],
            kinds={"pi": "lora", "critic": "value_head"})
        self.assertEqual(set(bundle.payloads), {"pi"})
        self.assertEqual(bundle.kinds, {"pi": "lora"})
        # the version map still pins BOTH — trainer-only deltas version too
        self.assertEqual(set(bundle.policy_version), {"pi", "critic"})

    def test_kinds_do_not_move_the_bundle_id(self) -> None:
        """Identity is versions ⊕ payload digests; the kind labels are routing
        metadata carried alongside, not new identity input."""
        with_kinds = compile_bundle({"pi": b"d"}, {"pi": 1}, ["pi"],
                                    kinds={"pi": "lora"})
        without = compile_bundle({"pi": b"d"}, {"pi": 1}, ["pi"])
        self.assertEqual(with_kinds.bundle_id, without.bundle_id)


class GroupByMechanismTest(unittest.TestCase):
    def test_payloads_route_to_their_serving_mechanism(self) -> None:
        bundle = Bundle("bundle:x", {"q": 1, "k": 1, "latent": 1},
                        payloads={"q": b"qq", "k": b"kk", "latent": b"E"},
                        kinds={"q": "lora", "k": "lora",
                               "latent": "soft_prompt"})
        grouped = group_by_mechanism(bundle)
        self.assertEqual(grouped[Mechanism.PUNICA], {"q": b"qq", "k": b"kk"})
        self.assertEqual(grouped[Mechanism.PROMPT_EMBEDS], {"latent": b"E"})

    def test_unlabeled_payload_is_a_compile_error(self) -> None:
        bundle = Bundle("bundle:x", {"pi": 1}, payloads={"pi": b"d"})
        with self.assertRaises(ValueError):
            group_by_mechanism(bundle)

    def test_trainer_only_payload_is_a_compile_error(self) -> None:
        bundle = Bundle("bundle:x", {"critic": 1}, payloads={"critic": b"h"},
                        kinds={"critic": "value_head"})
        with self.assertRaises(ValueError):
            group_by_mechanism(bundle)


if __name__ == "__main__":
    unittest.main()
