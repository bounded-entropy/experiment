"""spectral_latent's torch half: the prior recipes it shares with plora.

The adapter's own math (the straight-through top-k gains, the recorded draw)
is spectral's and plora's, pinned in their suites; what is pinned HERE is
that the latent's prior is plora's in both recipes — fixed, or learned as its
own parameter group — and that it rides the payload the way trained state
does. torch-gated: this suite runs in the image (or a local torch).
"""

from __future__ import annotations

import math
import unittest

from rlstack.policy.adapters.spectral_latent import (
    KL_PROVIDED, PRIOR_PROVIDED, SIGMA_PROVIDED, spectral_latent,
)
from rlstack.registry import ADAPTER_TYPES
from rlstack.spec.specs import AdapterSpec

try:
    import torch
except ImportError:                                  # the client environment
    torch = None

if torch is not None:
    from rlstack.policy.adapters import spectral_latent_torch

needs_torch = unittest.skipUnless(
    torch is not None, "torch is trainer metal: this suite runs in the image")

PATH = "block.proj"


class DeclarationTest(unittest.TestCase):
    def test_the_sugar_carries_the_prior_recipe(self) -> None:
        entry = spectral_latent("layers.*.self_attn.*", k=4, prior="learned")
        self.assertIsInstance(entry, AdapterSpec)
        self.assertEqual(entry.init["prior"], "learned")
        self.assertEqual(spectral_latent("x").init["prior"], "fixed")

    def test_it_declares_the_prior_beside_the_kl(self) -> None:
        instance = ADAPTER_TYPES.get("spectral_latent").instance
        self.assertEqual(instance.provides,
                         {KL_PROVIDED, SIGMA_PROVIDED, PRIOR_PROVIDED})


def a_model():
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.proj = torch.nn.Linear(8, 6, bias=False)
    return model


def a_site_meta():
    from rlstack.policy.siteschema import SiteMeta
    return (SiteMeta(name=PATH, path=PATH, has_weight=True, shape=(6, 8),
                     is_boundary=False),)


def a_state(prior: str = "fixed", seed: int = 1):
    return spectral_latent_torch.build(a_site_meta(), {
        "k": 3, "latent": 4, "members": 2, "hidden": 8, "prior_std": 0.05,
        "prior": prior, "seed": seed})


@needs_torch
class PriorTest(unittest.TestCase):
    def test_the_kl_starts_at_zero_under_either_recipe(self) -> None:
        for prior in ("fixed", "learned"):
            with self.subTest(prior=prior):
                provided = spectral_latent_torch.provide(a_state(prior))
                self.assertEqual(sorted(provided), sorted(
                    [KL_PROVIDED, SIGMA_PROVIDED, PRIOR_PROVIDED]))
                self.assertEqual(float(provided[KL_PROVIDED]), 0.0)
                self.assertAlmostEqual(float(provided[PRIOR_PROVIDED]), 0.05)

    def test_a_learned_prior_is_a_third_group_and_a_fixed_one_is_none(self) -> None:
        instance = ADAPTER_TYPES.get("spectral_latent").instance
        fixed, learned = a_state(), a_state("learned")
        self.assertEqual(sorted(instance.param_groups(fixed)),
                         ["mapper", "posterior"])
        self.assertEqual(sorted(instance.param_groups(learned)),
                         ["mapper", "posterior", "prior"])
        self.assertEqual([id(p) for p in instance.param_groups(learned)["prior"]],
                         [id(learned.prior_log_std)])
        for state in (fixed, learned):
            self.assertEqual(
                sum(p.numel() for g in instance.param_groups(state).values()
                    for p in g),
                sum(p.numel() for p in state.parameters()))

    def test_the_kl_reaches_a_learned_prior(self) -> None:
        state = a_state("learned")
        state.mu.data += 0.1                    # off the prior's mean
        spectral_latent_torch.provide(state)[KL_PROVIDED].backward()
        self.assertNotEqual(float(state.prior_log_std.grad), 0.0)
        fixed = a_state()
        fixed.mu.data += 0.1
        spectral_latent_torch.provide(fixed)[KL_PROVIDED].backward()
        self.assertIsNone(fixed.prior_log_std.grad)

    def test_the_prior_rides_the_payload_and_resumes(self) -> None:
        state = a_state("learned")
        spectral_latent_torch.install(a_model(), state)
        state.prior_log_std.data += 0.4
        payload = spectral_latent_torch.emit(state)
        head, tensors = spectral_latent_torch.unpack(payload)
        self.assertEqual(head["prior"], "learned")
        self.assertAlmostEqual(float(tensors["prior.log_std"]),
                               math.log(0.05) + 0.4, places=6)
        resumed = a_state("learned")
        spectral_latent_torch.load(resumed, payload)
        self.assertTrue(torch.equal(resumed.prior_log_std.detach(),
                                    state.prior_log_std.detach()))
        self.assertEqual(resumed.version, 0)
