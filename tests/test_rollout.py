"""The rollout seam (#48): the adapter-type dispatch, the lever merge, the refusal.

The lowerings themselves are metal (vLLM and torch), so what is checkable here
is the CONTRACT they meet: payloads reach their adapter type, a bundle's types merge
into one request, and a bundle this engine could not express is refused while
it is still just an id. The numbers are deploy/adapters_l4.py's job.
"""

from __future__ import annotations

import unittest

from rlstack import Bundle, Mechanism, group_by_adapter_type
from rlstack.policy.adapters.rollout import (
    Alignment, BuildDemands, Levers, Request, RolloutLowering,
    check_levers_compose,
)


class _Pinning(RolloutLowering):
    """An adapter type that pins its state with a request keyword (punica's shape)."""

    adapter_type = "pinning"
    mechanism = Mechanism.PUNICA
    claims = ("pin",)

    def demands(self) -> BuildDemands:
        return BuildDemands(engine_args={"enable_pin": True})

    def attach(self, bundle_id, payloads):
        return bundle_id

    def apply(self, attached, request: Request) -> Levers:
        return Levers(kwargs={"pin": attached})


class _Prefixing(RolloutLowering):
    """An adapter type that adds positions in front of the tokens (a soft prompt's)."""

    adapter_type = "prefixing"
    mechanism = Mechanism.PROMPT_EMBEDS
    claims = ("prompt",)

    def __init__(self, n: int = 3) -> None:
        super().__init__(build=None)
        self.n = n

    def attach(self, bundle_id, payloads):
        return self.n

    def apply(self, attached, request: Request) -> Levers:
        return Levers(prompt=("rows", attached) + request.token_ids)

    def align(self, attached) -> Alignment:
        return Alignment(attached)


class GroupByKindTest(unittest.TestCase):
    def test_payloads_reach_their_adapter_type_in_bank_order(self) -> None:
        bundle = Bundle("bundle:x", {"q": 1, "k": 1, "latent": 1},
                        payloads={"q": b"qq", "k": b"kk", "latent": b"E"},
                        adapter_types={"q": "lora", "k": "lora",
                               "latent": "soft_prompt"})
        grouped = group_by_adapter_type(bundle)
        self.assertEqual(grouped["lora"], {"q": b"qq", "k": b"kk"})
        self.assertEqual(grouped["soft_prompt"], {"latent": b"E"})

    def test_unlabeled_payload_is_a_compile_error(self) -> None:
        bundle = Bundle("bundle:x", {"pi": 1}, payloads={"pi": b"d"})
        with self.assertRaises(ValueError):
            group_by_adapter_type(bundle)

    def test_trainer_only_payload_is_a_compile_error(self) -> None:
        bundle = Bundle("bundle:x", {"critic": 1}, payloads={"critic": b"h"},
                        adapter_types={"critic": "value_head"})
        with self.assertRaises(ValueError):
            group_by_adapter_type(bundle)


class LeverMergeTest(unittest.TestCase):
    """One request out of many adapter types — the bus's fold, without the metal."""

    def test_two_adapter_types_contribute_to_one_request(self) -> None:
        request = Request(token_ids=(7, 8))
        merged = Levers(prompt=("plain", 7, 8))
        for lowering in (_Pinning(build=None), _Prefixing()):
            merged = merged.merged_with(
                lowering.apply(lowering.attach("bundle:x", {}), request))
        self.assertEqual(merged.prompt, ("rows", 3, 7, 8))
        self.assertEqual(merged.kwargs, {"pin": "bundle:x"})

    def test_an_adapter_type_that_shapes_nothing_leaves_the_prompt_alone(self) -> None:
        plain = Levers(prompt=("plain", 7))
        merged = plain.merged_with(Levers(kwargs={"pin": "b"}))
        self.assertEqual(merged.prompt, ("plain", 7))

    def test_alignment_defaults_to_no_positions(self) -> None:
        """A lever that adds no positions leaves the answer geometry alone."""
        pinning = _Pinning(build=None)
        self.assertEqual(pinning.align(pinning.attach("bundle:x", {})).positions, 0)

    def test_positions_are_what_a_prefix_occupies(self) -> None:
        prefixing = _Prefixing(n=5)
        self.assertEqual(prefixing.align(prefixing.attach("b", {})).positions, 5)


class ComposeTest(unittest.TestCase):
    def test_different_levers_compose(self) -> None:
        check_levers_compose("bundle:x", [_Pinning(build=None), _Prefixing()])

    def test_two_adapter_types_claiming_one_lever_are_refused_at_registration(self) -> None:
        """#48: a composition the engine cannot express refuses LOUDLY when the
        bundle registers — never silently at sample time."""
        with self.assertRaises(ValueError) as raised:
            check_levers_compose("bundle:x", [_Prefixing(), _Prefixing()])
        self.assertIn("prompt", str(raised.exception))


class DemandsTest(unittest.TestCase):
    def test_an_adapter_type_with_no_demands_costs_the_build_nothing(self) -> None:
        self.assertEqual(BuildDemands().engine_args, {})
        self.assertIsNone(BuildDemands().plugin)


if __name__ == "__main__":
    unittest.main()
