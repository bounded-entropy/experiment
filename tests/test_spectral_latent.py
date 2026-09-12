"""spectral_latent's torch half: the prior recipes it shares with plora, the
gain map's own two recipes, and what the served gains report.

The adapter's own math (the straight-through top-k gains, the recorded draw)
is spectral's and plora's, pinned in their suites; what is pinned HERE is
that the latent's prior is plora's in both recipes — fixed, or learned as its
own parameter group — that it rides the payload the way trained state does,
that the gain map's `amplitude` and `bound` recipes hold what they promise
(a scalar amplitude per site, a hard cap on every relative gain, the identity
element at zero) on BOTH sides of the membrane, and that the three served
summaries a loss may price are the forward's own gains. torch-gated: this
suite runs in the image (or a local torch).
"""

from __future__ import annotations

import math
import unittest

from rlstack.policy.adapters.spectral_latent import (
    AMPLITUDE_PROVIDED, ENERGY_PROVIDED, EPS_RECORD, GAIN_SPAN_PROVIDED,
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
    from rlstack.policy.adapters.replay import ReplayRows, row_plan
    from rlstack.training.losses import PolicyOutputs
    from rlstack.training.losses.grpo_latent_kl_gated import grpo_latent_kl_gated
    from rlstack.training.losses.grpo_latent_kl_gated_priced import (
        GAMMA, grpo_latent_kl_gated_priced,
    )

needs_torch = unittest.skipUnless(
    torch is not None, "torch is trainer metal: this suite runs in the image")

PATH = "block.proj"
PROVIDES = {KL_PROVIDED, SIGMA_PROVIDED, PRIOR_PROVIDED, GAIN_SPAN_PROVIDED,
            AMPLITUDE_PROVIDED, ENERGY_PROVIDED}
RECIPES = ({}, {"amplitude": "split"}, {"bound": 0.2},
           {"amplitude": "split", "bound": 0.2})
"""The gain map combines independent amplitude and bound settings."""


@needs_torch
class FixedLinearTest(unittest.TestCase):
    def test_only_the_posterior_trains_and_decoder_cannot_move(self):
        state = a_state(amplitude="fixed_linear")
        self.assertEqual(sum(p.numel() for p in state.parameters()), 8)
        self.assertEqual(set(spectral_latent_torch.param_groups(state)), {"posterior"})
        before = [p.clone() for p in state.mapper()]
        optimizer = torch.optim.AdamW(state.parameters(), lr=0.01)
        noise = torch.ones(4)
        z = spectral_latent_torch.reparameterized_latent(state, noise)
        spectral_latent_torch.relative_gains(state, PATH, z).sum().backward()
        self.assertGreater(state.mu.grad.abs().sum().item(), 0)
        self.assertGreater(state.log_std.grad.abs().sum().item(), 0)
        optimizer.step()
        for original, current in zip(before, state.mapper()):
            torch.testing.assert_close(original, current, rtol=0, atol=0)
            self.assertIsNone(current.grad)

    def test_mean_is_base_and_prior_gain_scale_is_declared(self):
        state = a_state(amplitude="fixed_linear")
        mean = spectral_latent_torch.relative_gains(state, PATH, state.mu)
        self.assertEqual(mean.abs().sum().item(), 0)
        generator = torch.Generator().manual_seed(3)
        z = torch.randn(30000, 4, generator=generator) * state.prior_std
        gains = spectral_latent_torch.relative_gains(state, PATH, z)
        torch.testing.assert_close(gains.std(0), torch.full((6,), 0.05), rtol=0.02, atol=0)

    def test_emitted_members_match_replay_and_resume(self):
        state = a_state(amplitude="fixed_linear", bound=0.2)
        spectral_latent_torch.install(a_model(), state)
        state.mu.data.add_(0.03)
        payload = spectral_latent_torch.emit(state)
        header, tensors = spectral_latent_torch.unpack(payload)
        z = spectral_latent_torch.reparameterized_latent(state, tensors["noise"][0])
        gains = spectral_latent_torch.effective_gains(state, PATH, z)
        expected = state.u[PATH].float() @ torch.diag(gains) @ state.v[PATH].float().T
        prefix = "peft.m0.base_model.model." + PATH
        actual = tensors[prefix + ".lora_B.weight"].float() @ tensors[prefix + ".lora_A.weight"].float()
        torch.testing.assert_close(expected, actual, rtol=0.015, atol=0.0005)
        restored = a_state(amplitude="fixed_linear", bound=0.2)
        spectral_latent_torch.install(a_model(), restored)
        spectral_latent_torch.load(restored, payload)
        torch.testing.assert_close(
            spectral_latent_torch.relative_gains(restored, PATH, z),
            spectral_latent_torch.relative_gains(state, PATH, z), rtol=0, atol=0)

    def test_learned_prior_is_refused(self):
        with self.assertRaises(ValueError):
            a_state(prior="learned", amplitude="fixed_linear")


@needs_torch
class FixedBasisTest(unittest.TestCase):

    def test_each_site_reads_only_its_own_coordinates(self):
        from rlstack.policy.siteschema import SiteMeta
        sites = a_site_meta() + (SiteMeta(name="other", path="other", has_weight=True,
                                         shape=(5, 7), is_boundary=False),)
        state = spectral_latent_torch.build(sites, {
            "k": 3, "latent": 6, "members": 2, "hidden": 1,
            "prior_std": 0.01, "amplitude": "fixed_basis", "seed": 1})
        z = torch.arange(1., 7., requires_grad=True)
        first = spectral_latent_torch.relative_gains(state, PATH, z)
        second = spectral_latent_torch.relative_gains(state, "other", z)
        torch.testing.assert_close(first, torch.tensor([1., 2., 3., 0., 0., 0.]))
        torch.testing.assert_close(second, torch.tensor([4., 5., 6., 0., 0.]))
        first.sum().backward()
        torch.testing.assert_close(z.grad, torch.tensor([1., 1., 1., 0., 0., 0.]))
        self.assertEqual(sum(p.numel() for p in state.parameters()), 12)
        self.assertTrue(all(not p.requires_grad for p in state.mapper()))

    def test_rank_support_and_emitted_members_match_replay(self):
        state = a_state(amplitude="fixed_basis", latent=3, hidden=1)
        torch.manual_seed(123)
        spectral_latent_torch.install(a_model(), state)
        state.mu.data.copy_(torch.tensor([0.01, -0.03, 0.02]))
        payload = spectral_latent_torch.emit(state)
        _, tensors = spectral_latent_torch.unpack(payload)
        z = spectral_latent_torch.reparameterized_latent(state, tensors["noise"][0])
        gains = spectral_latent_torch.effective_gains(state, PATH, z)
        self.assertEqual(gains[3:].abs().sum().item(), 0)
        expected = state.u[PATH].float() @ torch.diag(gains) @ state.v[PATH].float().T
        prefix = "peft.m0.base_model.model." + PATH
        actual = tensors[prefix + ".lora_B.weight"].float() @ tensors[prefix + ".lora_A.weight"].float()
        torch.testing.assert_close(expected, actual, rtol=0.015, atol=0.0005)
        restored = a_state(amplitude="fixed_basis", latent=3, hidden=1)
        torch.manual_seed(123)
        spectral_latent_torch.install(a_model(), restored)
        spectral_latent_torch.load(restored, payload)
        self.assertEqual(spectral_latent_torch.emit(restored), payload)

    def test_invalid_dimensions_and_learned_prior_are_refused(self):
        with self.assertRaises(ValueError):
            a_state(amplitude="fixed_basis")
        with self.assertRaises(ValueError):
            a_state(amplitude="fixed_basis", latent=3, prior="learned")

    def test_narrow_replay_preserves_dense_values_gradients_and_summaries(self):
        """Removing zero directions must preserve input/posterior gradients,
        including an active coordinate whose current gain is exactly zero."""
        for bound in (None, 0.2):
            for shared in (False, True):
                with self.subTest(bound=bound, shared=shared):
                    torch.manual_seed(12)
                    state = a_state(amplitude="fixed_basis", latent=3, hidden=1, bound=bound)
                    model = a_model()
                    spectral_latent_torch.install(model, state)
                    state.mu.data.copy_(torch.tensor([0., .03, -.02]))
                    noise = torch.tensor([[0., -.5, .8], [-1., .2, .1]])
                    if shared:
                        noise[1] = noise[0]
                    x = torch.randn(2, 4, 8, requires_grad=True)
                    weights = torch.randn(2, 4, 6)
                    actual = model.block.proj._delta(x, state, one_slot(state, noise))
                    self.assertEqual(state.served[PATH][0].shape[-1], 6)
                    provided = spectral_latent_torch.provide(state)
                    actual_grad = torch.autograd.grad(actual, (x, state.mu, state.log_std), weights)

                    # The previous full-spectrum lowering is the reference.
                    z = spectral_latent_torch.reparameterized_latent(state, noise)
                    eff = spectral_latent_torch.effective_gains(state, PATH, z)
                    expected = ((x.float() @ state.v[PATH].float()) * eff[:, None, :]) @ state.u[PATH].float().T
                    expected_grad = torch.autograd.grad(expected, (x, state.mu, state.log_std), weights)
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
                    for got, want in zip(actual_grad, expected_grad):
                        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)
                    energy = (eff / state.sigma[PATH]).square().sum(-1).mean()
                    torch.testing.assert_close(provided[ENERGY_PROVIDED], energy)


@needs_torch
class ReferenceKlControlTest(unittest.TestCase):
    def test_all_tie_rewards_still_pull_toward_the_reference(self):
        from rlstack import TokenBatch
        from rlstack.training.losses.grpo_reference_kl import grpo_reference_kl
        logits = torch.tensor([math.log(0.8), math.log(0.2)], requires_grad=True)
        actions = torch.tensor([0] * 8 + [1] * 2)
        logprobs = torch.log_softmax(logits, 0)[actions]
        batch = TokenBatch(
            token_ids=(1,) * 10, loss_mask=(1,) * 10,
            behavior_logprobs=tuple(logprobs.detach().tolist()),
            segment_ids=(0,) * 10, doc_starts=(0,),
            postdata={"advantage": (0.,) * 10,
                      "teacher_logprobs": (math.log(0.5),) * 10},
            microbatches_in_update=1)
        result = grpo_reference_kl(PolicyOutputs(logprobs=logprobs), batch)
        result.loss.backward()
        self.assertGreater(logits.grad[0].item(), 0)
        self.assertLess(logits.grad[1].item(), 0)
        self.assertAlmostEqual(logits.grad[0].item(), 0.04 * 0.8 * 0.2 * math.log(4), places=6)


class DeclarationTest(unittest.TestCase):
    def test_the_sugar_carries_the_prior_recipe(self) -> None:
        entry = spectral_latent("layers.*.self_attn.*", k=4, prior="learned")
        self.assertIsInstance(entry, AdapterSpec)
        self.assertEqual(entry.init["prior"], "learned")
        self.assertEqual(spectral_latent("x").init["prior"], "fixed")

    def test_it_declares_the_served_gains_beside_the_kl(self) -> None:
        instance = ADAPTER_TYPES.get("spectral_latent").instance
        self.assertEqual(instance.provides, PROVIDES)

    def test_the_recipe_knobs_are_written_only_when_set(self) -> None:
        """A spec that predates the knobs is the same spec (I3): at their
        defaults the sugar leaves them out of the init entirely."""
        self.assertNotIn("amplitude", spectral_latent("x").init)
        self.assertNotIn("bound", spectral_latent("x").init)
        both = spectral_latent("x", amplitude="split", bound=0.2).init
        self.assertEqual((both["amplitude"], both["bound"]), ("split", 0.2))
        with self.assertRaises(ValueError):
            spectral_latent("x", amplitude="scaled")
        with self.assertRaises(ValueError):
            spectral_latent("x", bound=0.0)

    def test_the_priced_loss_requires_what_the_adapter_provides(self) -> None:
        """The loss names the energy by the adapter's string; a rename on
        either side must break here, not on the metal."""
        from rlstack.registry import LOSSES

        required = LOSSES.get("grpo_latent_kl_gated_priced").requires
        self.assertIn(ENERGY_PROVIDED, required)
        self.assertIn(KL_PROVIDED, required)


def a_model():
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.proj = torch.nn.Linear(8, 6, bias=False)
    return model


def a_site_meta():
    from rlstack.policy.siteschema import SiteMeta
    return (SiteMeta(name=PATH, path=PATH, has_weight=True, shape=(6, 8),
                     is_boundary=False),)


def a_state(prior: str = "fixed", seed: int = 1, **recipe):
    return spectral_latent_torch.build(a_site_meta(), {
        "k": 3, "latent": 4, "members": 2, "hidden": 8, "prior_std": 0.05,
        "prior": prior, "seed": seed, **recipe})


def a_trained_state(**recipe):
    """Heads off zero, an amplitude off zero, and a posterior that has moved
    — at the identity element both sides are the base and every promise is
    vacuous. Installed, so the spectrum is there."""
    torch.manual_seed(5)
    state = a_state(**recipe)
    spectral_latent_torch.install(a_model(), state)
    for head in state.heads.values():
        head.data = torch.randn_like(head.data) * 0.02
    for amp in state.amp.values():
        amp.data = torch.tensor(0.15)
    state.mu.data = torch.randn_like(state.mu.data) * 0.1
    return state


def one_slot(state, noise):
    """The row plan a single-tenant microbatch gets: one slot, and one
    recorded latent per row (test_plora's fixture, this family's record)."""
    return ReplayRows(
        slots=({PATH: state},),
        index=torch.zeros(len(noise), dtype=torch.long),
        facts=tuple(({EPS_RECORD: list(eps)},) for eps in noise))


@needs_torch
class PriorTest(unittest.TestCase):
    def test_the_kl_starts_at_zero_under_either_recipe(self) -> None:
        for prior in ("fixed", "learned"):
            with self.subTest(prior=prior):
                state = a_state(prior)
                spectral_latent_torch.install(a_model(), state)
                provided = spectral_latent_torch.provide(state)
                self.assertEqual(set(provided), PROVIDES)
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
        """Installed first: `provide` reports the served gains too now, and
        those are the spectrum's business."""
        state = a_state("learned")
        spectral_latent_torch.install(a_model(), state)
        state.mu.data += 0.1                    # off the prior's mean
        spectral_latent_torch.provide(state)[KL_PROVIDED].backward()
        self.assertNotEqual(float(state.prior_log_std.grad), 0.0)
        fixed = a_state()
        spectral_latent_torch.install(a_model(), fixed)
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


@needs_torch
class ParityTest(unittest.TestCase):
    """WHAT THE ENGINE IS SHIPPED IS WHAT THE REPLAY COMPUTES (I6), for the
    generated gains — every member and the posterior mean.

    The latent's version of the promise has one more moving part than plain
    spectral's: the member the engine serves was materialized from a draw made
    at EMIT time, and replay recomposes that draw's recorded noise with the
    posterior it holds now. Held at ONE posterior the two must agree exactly,
    and that is what this checks; the deliberate difference under lag is the
    reparameterization contract, not a defect."""

    def served(self, tensors, tag, x):
        key = f"peft.{tag}.base_model.model.{PATH}"
        a = tensors[f"{key}.lora_A.weight"].float()
        b = tensors[f"{key}.lora_B.weight"].float()
        return (x @ a.T) @ b.T

    def replay(self, state, z, x):
        eff = spectral_latent_torch.effective_gains(state, PATH, z)
        v = state.v[PATH].to(torch.float32)
        u = state.u[PATH].to(torch.float32)
        return ((x.to(torch.float32) @ v) * eff) @ u.T

    def assertAgrees(self, replay, served) -> None:
        self.assertLess(float((replay - served).abs().max()),
                        0.02 * float(served.abs().max()))
        self.assertGreater(float(torch.nn.functional.cosine_similarity(
            replay.flatten(), served.flatten(), dim=0)), 0.999)

    def test_every_member_serves_the_gains_replay_recomputes(self) -> None:
        """Under EVERY recipe: the split and the bound live in
        `relative_gains`, the one function both lowerings call, and this is
        the check that nothing served bypasses it."""
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                state = a_trained_state(**recipe)
                x = torch.randn(2, 3, 8)
                _, tensors = spectral_latent_torch.unpack(
                    spectral_latent_torch.emit(state))
                noise = tensors["noise"]
                for member in range(state.members):
                    z = spectral_latent_torch.reparameterized_latent(
                        state, noise[member])
                    self.assertAgrees(self.replay(state, z, x).detach(),
                                      self.served(tensors, f"m{member}", x))

    def test_the_mean_member_is_the_posterior_mean(self) -> None:
        """Seedless score traffic is served the MEAN, so its parity is the
        one that decides whether a scored logprob is the policy's."""
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                state = a_trained_state(**recipe)
                x = torch.randn(2, 3, 8)
                _, tensors = spectral_latent_torch.unpack(
                    spectral_latent_torch.emit(state))
                self.assertAgrees(self.replay(state, state.mu, x).detach(),
                                  self.served(tensors, "mean", x))

    def test_the_recorded_noise_is_what_made_the_member(self) -> None:
        """Replay reads the noise off the turn; the payload is where it came
        from, so the two must be the same vectors in the same order."""
        state = a_trained_state()
        _, tensors = spectral_latent_torch.unpack(
            spectral_latent_torch.emit(state))
        self.assertEqual(tuple(tensors["noise"].shape),
                         (state.members, state.latent))
        self.assertTrue(torch.equal(
            tensors["noise"],
            spectral_latent_torch.noise_for(
                state.seed, 0, state.members, state.latent)))


@needs_torch
class RecipeTest(unittest.TestCase):
    """The gain map's two knobs, each holding what the venue's header says
    it holds — and zero the identity element under all four recipes."""

    def relative(self, state, z):
        return spectral_latent_torch.relative_gains(state, PATH, z)

    def test_zero_is_the_identity_element_under_every_recipe(self) -> None:
        """Version 0 serves the base whatever the recipe: zero heads under
        "joint", a zero amplitude under "split" (whose heads are NOT zero —
        a direction has to exist for `unit` to normalize)."""
        for recipe in RECIPES:
            with self.subTest(recipe=recipe):
                state = a_state(**recipe)
                spectral_latent_torch.install(a_model(), state)
                _, tensors = spectral_latent_torch.unpack(
                    spectral_latent_torch.emit(state))
                for tag in ("m0", "m1", "mean"):
                    b = tensors[f"peft.{tag}.base_model.model.{PATH}.lora_B.weight"]
                    self.assertEqual(float(b.abs().max()), 0.0)
                provided = spectral_latent_torch.provide(state)
                for name in (KL_PROVIDED, GAIN_SPAN_PROVIDED,
                             AMPLITUDE_PROVIDED, ENERGY_PROVIDED):
                    self.assertEqual(float(provided[name]), 0.0)
                if recipe.get("amplitude") == "split":
                    self.assertGreater(
                        float(state.heads[PATH].abs().max()), 0.0)

    def test_split_makes_the_amplitude_the_largest_relative_gain(self) -> None:
        """delta = a * unit(.): the largest |relative gain| at the site is
        exactly |a|, whatever z says — z only turns the vector."""
        state = a_trained_state(amplitude="split")
        for z in (state.mu, state.mu + 0.3, torch.randn(state.latent)):
            gains = self.relative(state, z)
            self.assertAlmostEqual(float(gains.abs().max()), 0.15, places=5)

    def test_split_lets_z_rotate_but_not_scale(self) -> None:
        """The hypothesis the recipe answers: under "joint" a change in z
        moves the gain vector's NORM as freely as its direction; under
        "split" the norm is a's business alone."""
        joint = a_trained_state()
        split = a_trained_state(amplitude="split")
        z0, z1 = joint.mu, joint.mu + 0.5
        joint_norms = [float(self.relative(joint, z).abs().max()) for z in (z0, z1)]
        split_norms = [float(self.relative(split, z).abs().max()) for z in (z0, z1)]
        self.assertNotAlmostEqual(joint_norms[0], joint_norms[1], places=3)
        self.assertAlmostEqual(split_norms[0], split_norms[1], places=5)
        self.assertLess(float(torch.nn.functional.cosine_similarity(
            self.relative(split, z0), self.relative(split, z1), dim=0)), 0.9999)

    def test_bound_caps_every_relative_gain(self) -> None:
        """A head grown wild still serves inside (-g, g), under either
        amplitude recipe."""
        for recipe in ({"bound": 0.2}, {"amplitude": "split", "bound": 0.2}):
            with self.subTest(recipe=recipe):
                state = a_trained_state(**recipe)
                for head in state.heads.values():
                    head.data = torch.randn_like(head.data) * 50.0
                for amp in state.amp.values():
                    amp.data = torch.tensor(50.0)
                gains = self.relative(state, torch.randn(state.latent))
                # saturated: g * tanh(huge) rounds to g in fp32
                self.assertLessEqual(float(gains.abs().max()), 0.2 + 1e-6)
                self.assertGreater(float(gains.abs().max()), 0.19)

    def test_the_amplitude_is_its_own_group_under_split_only(self) -> None:
        instance = ADAPTER_TYPES.get("spectral_latent").instance
        joint, split = a_state(), a_state(amplitude="split")
        self.assertEqual(sorted(instance.param_groups(joint)),
                         ["mapper", "posterior"])
        self.assertEqual(sorted(instance.param_groups(split)),
                         ["amplitude", "mapper", "posterior"])
        self.assertEqual(
            [id(p) for p in instance.param_groups(split)["amplitude"]],
            [id(split.amp[PATH])])
        for state in (joint, split):
            self.assertEqual(
                sum(p.numel() for g in instance.param_groups(state).values()
                    for p in g),
                sum(p.numel() for p in state.parameters()))

    def test_the_recipe_rides_the_payload_and_a_stranger_is_refused(self) -> None:
        state = a_trained_state(amplitude="split", bound=0.2)
        payload = spectral_latent_torch.emit(state)
        head, tensors = spectral_latent_torch.unpack(payload)
        self.assertEqual((head["amplitude"], head["bound"]), ("split", 0.2))
        self.assertAlmostEqual(float(tensors[f"amp.{PATH}"]), 0.15)
        resumed = a_state(amplitude="split", bound=0.2)
        spectral_latent_torch.load(resumed, payload)
        self.assertAlmostEqual(float(resumed.amp[PATH]), 0.15)
        for other in ({}, {"amplitude": "split"}, {"bound": 0.2}):
            with self.subTest(other=other), self.assertRaises(ValueError):
                spectral_latent_torch.load(a_state(**other), payload)


@needs_torch
class ServedSummaryTest(unittest.TestCase):
    """What `provide` says about the gains: the FORWARD's own, row by row,
    when a forward ran; the mean member's when none did."""

    def forward(self, state, noise):
        model = a_model()
        spectral_latent_torch.install(model, state)
        with row_plan(model).route(one_slot(state, noise)):
            model.block.proj(torch.randn(len(noise), 3, 8))

    def test_the_summary_is_the_forward_s_served_gains(self) -> None:
        state = a_trained_state()
        noise = torch.randn(3, state.latent)
        self.forward(state, noise)
        provided = spectral_latent_torch.provide(state)
        # recompute what those rows served, by hand
        z = spectral_latent_torch.reparameterized_latent(state, noise)
        eff = spectral_latent_torch.effective_gains(state, PATH, z)
        relative = eff / state.sigma[PATH]
        self.assertAlmostEqual(
            float(provided[GAIN_SPAN_PROVIDED]),
            float(eff.abs().topk(3, dim=-1).values.mean()), places=5)
        self.assertAlmostEqual(
            float(provided[AMPLITUDE_PROVIDED]),
            float(relative.abs().amax(dim=-1).mean()), places=5)
        self.assertAlmostEqual(
            float(provided[ENERGY_PROVIDED]),
            float((relative * relative).sum(dim=-1).mean()), places=5)
        self.assertEqual(state.served, {})          # read once, then cleared

    def test_without_a_forward_the_summary_is_the_mean_member_s(self) -> None:
        state = a_trained_state()
        provided = spectral_latent_torch.provide(state)
        eff = spectral_latent_torch.effective_gains(state, PATH, state.mu)
        self.assertAlmostEqual(
            float(provided[GAIN_SPAN_PROVIDED]),
            float(eff.abs().topk(3).values.mean()), places=5)

    def test_the_energy_prices_only_what_is_served(self) -> None:
        """Grad reaches the served directions' gains and no other: the
        straight-through value of an unserved direction is exactly zero, so
        its square has no slope. Read off the dense delta's gradient."""
        state = a_trained_state()
        z = torch.randn(state.latent)
        delta = spectral_latent_torch.relative_gains(state, PATH, z)
        delta.retain_grad()
        eff = state.sigma[PATH] * delta
        k = state.k
        picked = torch.topk(eff.abs(), k).indices
        hard = torch.zeros_like(eff)
        hard[picked] = 1.0
        eff = eff * hard + (eff - eff.detach()) * (1.0 - hard)
        energy = ((eff / state.sigma[PATH]) ** 2).sum()
        energy.backward()
        served = set(picked.tolist())
        for index, grad in enumerate(delta.grad.tolist()):
            if index in served:
                self.assertNotEqual(grad, 0.0)
            else:
                self.assertEqual(grad, 0.0)

    def test_the_priced_loss_is_the_gated_loss_plus_gamma_energy(self) -> None:
        """The priced loss preserves the gated objective,
        plus GAMMA times the provided energy, nothing else."""
        from rlstack import TokenBatch

        state = a_trained_state()
        self.forward(state, torch.randn(2, state.latent))
        provided = spectral_latent_torch.provide(state)
        self.assertGreater(float(provided[ENERGY_PROVIDED]), 0.0)
        logprobs = torch.full((4,), -0.5)
        batch = TokenBatch(
            token_ids=(1, 2, 3, 4), loss_mask=(1, 1, 1, 1),
            behavior_logprobs=(-0.5,) * 4, segment_ids=(0, 0, 0, 0),
            doc_starts=(0,),
            postdata={"advantage": (1.0, 1.0, -1.0, -1.0),
                      "accuracy": (1.0,) * 4},
            microbatches_in_update=1)
        gated = grpo_latent_kl_gated(
            PolicyOutputs(logprobs=logprobs, provided=provided), batch)
        priced = grpo_latent_kl_gated_priced(
            PolicyOutputs(logprobs=logprobs, provided=provided), batch)
        self.assertAlmostEqual(
            float(priced.loss),
            float(gated.loss) + GAMMA * float(provided[ENERGY_PROVIDED]),
            places=6)
        self.assertEqual(priced.mean_ratio, gated.mean_ratio)
