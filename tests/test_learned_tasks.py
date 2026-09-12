"""Sampled routing, learned-direction ownership, and checkpoint equivalence."""

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch required")
class LearnedTasksTest(unittest.TestCase):
    def setUp(self):
        from test_batched_replay import a_model, SITES, PATH
        from rlstack.policy.adapters import learned_tasks_torch as adapter
        self.adapter, self.path, self.sites = adapter, PATH, SITES
        self.model = a_model().requires_grad_(False)
        self.init = dict(rank=3, latent=4, hidden=8, tasks=3, scope="train", beta=.01)
        self.state = adapter.build(SITES, self.init)
        adapter.install(self.model, self.state)
        self.x = torch.randn(2, 3, 8)

    def rows(self, state=None, noise=(.2, -.1, .5, .3), ids=(0, 2)):
        from rlstack.policy.adapters.replay import ReplayRows
        return ReplayRows(slots=({self.path: state or self.state},),
                          index=torch.zeros(len(ids), dtype=torch.long),
                          facts=tuple(({"spectral_task": i, "slatent_eps": list(noise)},) for i in ids))

    def forward(self, rows=None):
        from rlstack.policy.adapters.replay import row_plan
        with row_plan(self.model).route(rows or self.rows()):
            return self.model.block.proj(self.x)

    def activate(self):
        torch.nn.init.normal_(self.state.heads[self.path].weight, std=.1)

    def test_zero_heads_are_exactly_base_for_every_draw(self):
        base = self.model.block.proj.inner(self.x)
        self.assertTrue(torch.equal(self.forward(), base))
        self.assertTrue(torch.equal(self.forward(self.rows(noise=(9., -3., 1., 4.))), base))
        self.assertEqual(float(self.adapter.provide(self.state)["latent_kl"]), 0.)

    def test_actual_draws_train_both_posterior_parameters_and_learned_bases(self):
        self.activate()
        first = self.forward()
        other = self.forward(self.rows(noise=(-.2, .1, -.5, -.3)))
        self.assertFalse(torch.equal(first, other))
        first.square().sum().backward()
        for p in (self.state.mu, self.state.log_std):
            self.assertGreater(float(p.grad[0].abs().sum()), 0.)
            self.assertEqual(float(p.grad[1].abs().sum()), 0.)
        for p in (self.state.u[self.path], self.state.v[self.path]):
            self.assertGreater(float(p.grad.abs().sum()), 0.)
        self.assertIsNone(self.model.block.proj.inner.weight.grad)

    def test_load_restores_learned_bases_and_frozen_new_scope(self):
        self.activate()
        self.state.mu.data.fill_(.7)
        self.state.log_std.data.fill_(-.4)
        payload = self.adapter.emit(self.state)
        resumed = self.adapter.build(self.sites, self.init)
        self.adapter.load(resumed, payload)
        self.assertEqual(self.adapter.emit(resumed), payload)
        frozen = self.adapter.build(self.sites, {**self.init, "scope": "new", "tasks": 2,
                                                "train_decoder": False})
        self.adapter.load(frozen, payload)
        self.assertEqual(set(self.adapter.param_groups(frozen)), {"mean", "scale"})
        for key, p in self.adapter.shared_tensors(frozen).items():
            self.assertTrue(torch.equal(p, self.adapter.shared_tensors(self.state)[key]))
        self.assertTrue(all(not p.requires_grad for p in frozen.shared_parameters()))
        self.assertEqual(float(frozen.mu.abs().sum()), 0.)
        self.assertEqual(float(frozen.log_std.abs().sum()), 0.)
        self.adapter.install(self.model, frozen)
        before = {k: v.clone() for k, v in self.adapter.shared_tensors(frozen).items()}
        optimizer = torch.optim.Adam(frozen.parameters(), lr=.03)
        self.forward(self.rows(frozen, ids=(0, 1))).square().sum().backward()
        optimizer.step()
        self.assertGreater(float(frozen.mu.abs().sum()), 0.)
        self.assertGreater(float(frozen.log_std.abs().sum()), 0.)
        for key, tensor in self.adapter.shared_tensors(frozen).items():
            self.assertTrue(torch.equal(tensor, before[key]))

    def test_materialized_lora_matches_sampled_replay(self):
        from rlstack.policy.adapters.lora_torch import _whole_batch_delta
        self.activate()
        self.state.mu.data[2].fill_(.7)
        rows = self.rows()
        z = self.adapter.task_latents(self.state, rows)
        actual = self.forward(rows)
        base = self.model.block.proj.inner(self.x)
        for i in range(2):
            lora = self.adapter.materialize(self.state, z[i])
            expected = base[i:i+1] + _whole_batch_delta(self.x[i:i+1], lora, self.path)
            torch.testing.assert_close(actual[i:i+1], expected, rtol=1e-5, atol=1e-6)

    def test_install_and_remove_do_not_disturb_another_tenant(self):
        self.activate()
        expected = self.forward().detach()
        other = self.adapter.build(self.sites, self.init)
        self.adapter.install(self.model, other)
        self.assertTrue(torch.equal(self.forward(), expected))
        self.adapter.uninstall(self.model, other)
        self.assertTrue(torch.equal(self.forward(), expected))
        self.adapter.uninstall(self.model, self.state)
        self.assertIsInstance(self.model.block.proj, torch.nn.Linear)

    def test_checkpoint_refuses_incompatible_architecture_or_unfrozen_new_scope(self):
        payload = self.adapter.emit(self.state)
        for changed in ({"latent": 5}, {"rank": 2}, {"scope": "new"}, {"tasks": 2}):
            state = self.adapter.build(self.sites, {**self.init, **changed})
            with self.assertRaises(ValueError):
                self.adapter.load(state, payload)

    def test_replayed_noise_and_checkpoint_restore_identical_output(self):
        self.activate()
        expected = self.forward().detach()
        payload = self.adapter.emit(self.state)
        self.state.u[self.path].data.zero_()
        self.adapter.load(self.state, payload)
        self.assertTrue(torch.equal(expected, self.forward()))


if __name__ == "__main__":
    unittest.main()
