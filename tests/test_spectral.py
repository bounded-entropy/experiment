"""Spectral adapter declarations and parity between replay and emitted LoRA.

The parity cases reconstruct the served delta from the PEFT tensors emitted
for vLLM and compare it with the replay forward. Numerical tests require
PyTorch; declaration tests also run in the lightweight client environment.
"""

from __future__ import annotations

import unittest

from rlstack.policy.adapters.spectral import GAIN_SPAN_PROVIDED, spectral
from rlstack.registry import ADAPTER_TYPES
from rlstack.spec.specs import AdapterSpec

try:
    import torch
except ImportError:                                  # the client environment
    torch = None

if torch is not None:
    from rlstack.policy.adapters import spectral_torch

needs_torch = unittest.skipUnless(
    torch is not None, "torch is trainer metal: this suite runs in the image")

PATH, D_IN, D_OUT, K = "block.proj", 32, 24, 4
PEFT = f"peft.base_model.model.{PATH}"


class DeclarationTest(unittest.TestCase):
    def test_the_sugar_is_a_plain_adapter_spec(self) -> None:
        entry = spectral("layers.*.self_attn.*", k=16)
        self.assertIsInstance(entry, AdapterSpec)
        self.assertEqual(entry.adapter_type, "spectral")
        self.assertEqual(entry.init["k"], 16)

    def test_it_provides_the_watched_span_and_records_nothing(self) -> None:
        """A deterministic adapter draws nothing, so it records nothing; the
        gain span is provided to be WATCHED, and nothing requires it."""
        instance = ADAPTER_TYPES.get("spectral").instance
        self.assertEqual(instance.provides, {GAIN_SPAN_PROVIDED})
        self.assertEqual(instance.records, ())

    def test_it_lives_only_at_a_weighted_site(self) -> None:
        from rlstack import fake_qwen_schema

        schema = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
        instance = ADAPTER_TYPES.get("spectral").instance
        self.assertTrue(instance.site_ok(
            [m for m in schema.sites if m.has_weight][0]))
        self.assertFalse(instance.site_ok(
            [m for m in schema.sites if not m.has_weight][0]))


def a_model():
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.proj = torch.nn.Linear(D_IN, D_OUT, bias=False)
    return model


def a_site():
    from rlstack.policy.siteschema import SiteMeta

    return (SiteMeta(name=PATH, path=PATH, has_weight=True,
                     shape=(D_OUT, D_IN), is_boundary=False),)


def a_trained_state(scale: float = 0.05):
    """An installed state whose gains have MOVED — parity at delta = 0 is the
    trivial case (both sides are the base), so the interesting comparison is a
    state the size a run actually reaches."""
    torch.manual_seed(11)
    state = spectral_torch.build(a_site(), {"k": K, "seed": 1})
    spectral_torch.install(a_model(), state)
    state.delta[PATH].data = torch.randn(min(D_IN, D_OUT)) * scale
    return state


@needs_torch
class ParityTest(unittest.TestCase):
    """WHAT THE ENGINE IS SHIPPED IS WHAT THE REPLAY COMPUTES (I6)."""

    def served(self, tensors, x):
        """Punica's arithmetic on the shipped pair, at scaling 1 (alpha == r,
        `lora_torch.peft_config`): x @ A^T @ B^T."""
        a = tensors[f"{PEFT}.lora_A.weight"].float()
        b = tensors[f"{PEFT}.lora_B.weight"].float()
        return (x @ a.T) @ b.T

    def test_the_replay_forward_is_the_shipped_delta(self) -> None:
        state = a_trained_state()
        x = torch.randn(2, 3, D_IN)
        _, tensors = spectral_torch.unpack(spectral_torch.emit(state))
        replay = spectral_torch._whole_batch_delta(x, state, PATH).detach()
        served = self.served(tensors, x)
        # bf16 is the frozen dtype on both sides (FROZEN_DTYPE), so the two
        # agree to that rounding and no further — a real disagreement is
        # orders of magnitude larger, which is the point of the bound
        self.assertLess(float((replay - served).abs().max()),
                        0.02 * float(served.abs().max()))
        self.assertGreater(float(torch.nn.functional.cosine_similarity(
            replay.flatten(), served.flatten(), dim=0)), 0.999)

    def test_the_shipped_pair_has_the_declared_rank(self) -> None:
        """k directions serve, so the peft pair is rank k — what
        `adapter_config.json` tells vLLM and what max_lora_rank must cover."""
        _, tensors = spectral_torch.unpack(
            spectral_torch.emit(a_trained_state()))
        self.assertEqual(tuple(tensors[f"{PEFT}.lora_A.weight"].shape),
                         (K, D_IN))
        self.assertEqual(tuple(tensors[f"{PEFT}.lora_B.weight"].shape),
                         (D_OUT, K))

    def test_version_zero_serves_the_base(self) -> None:
        """delta = 0 is the identity element: the shipped delta is zero, so a
        run's first bundle cannot move the policy off the base."""
        state = spectral_torch.build(a_site(), {"k": K, "seed": 1})
        spectral_torch.install(a_model(), state)
        _, tensors = spectral_torch.unpack(spectral_torch.emit(state))
        self.assertEqual(float(tensors[f"{PEFT}.lora_B.weight"].abs().max()),
                         0.0)


@needs_torch
class StraightThroughTest(unittest.TestCase):
    """The value is k-sparse; the gradient is dense (spectral_torch's contract:
    the served policy is exact, and every direction still competes)."""

    def test_the_value_moves_only_k_directions(self) -> None:
        state = a_trained_state()
        eff = spectral_torch.effective_gains(state, PATH)
        self.assertEqual(int((eff != 0).sum()), K)

    def test_every_direction_still_receives_gradient(self) -> None:
        state = a_trained_state()
        spectral_torch.effective_gains(state, PATH).sum().backward()
        grad = state.delta[PATH].grad
        self.assertEqual(int((grad != 0).sum()), min(D_IN, D_OUT))

    def test_the_served_set_is_the_largest_gains(self) -> None:
        state = a_trained_state()
        eff = (state.sigma[PATH] * state.delta[PATH]).abs()
        picked = set(spectral_torch.served_indices(state, PATH).tolist())
        cutoff = float(eff.topk(K).values.min())
        self.assertEqual(len(picked), K)
        for index in picked:
            self.assertGreaterEqual(float(eff[index]), cutoff)


@needs_torch
class PayloadTest(unittest.TestCase):
    def test_the_dense_gains_roundtrip_so_resume_keeps_every_direction(self) -> None:
        """The peft half is the engine's; the DENSE half is resume's, and it
        carries the directions outside the served k that have been
        accumulating movement."""
        state = a_trained_state()
        payload = spectral_torch.emit(state)
        fresh = spectral_torch.build(a_site(), {"k": K, "seed": 1})
        spectral_torch.install(a_model(), fresh)
        spectral_torch.load(fresh, payload)
        self.assertTrue(torch.equal(fresh.delta[PATH].data,
                                    state.delta[PATH].data))

    def test_a_payload_of_another_width_is_refused(self) -> None:
        payload = spectral_torch.emit(a_trained_state())
        other = spectral_torch.build(a_site(), {"k": K - 1, "seed": 1})
        spectral_torch.install(a_model(), other)
        with self.assertRaises(ValueError):
            spectral_torch.load(other, payload)


if __name__ == "__main__":
    unittest.main()
