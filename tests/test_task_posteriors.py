"""Task posterior routing, checkpoint ownership, and sequence-loss packing."""

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, "torch required")
class TaskPosteriorTest(unittest.TestCase):
    def setUp(self):
        from test_batched_replay import a_model, SITES, PATH
        from rlstack.policy.adapters import spectral_tasks_torch as adapter
        self.adapter, self.path, self.sites = adapter, PATH, SITES
        self.model = a_model().requires_grad_(False)
        self.init = dict(k=3, latent=4, hidden=8, tasks=3, scope="train", beta=.01)
        self.state = adapter.build(SITES, self.init)
        adapter.install(self.model, self.state)

    def rows(self, ids):
        from rlstack.policy.adapters.replay import ReplayRows
        return ReplayRows(slots=({self.path: self.state},),
                          index=torch.zeros(len(ids), dtype=torch.long),
                          facts=tuple(({"spectral_task": i, "slatent_eps": [.2,-.1,.5,.3]},) for i in ids))

    def forward(self, ids):
        from rlstack.policy.adapters.replay import row_plan
        with row_plan(self.model).route(self.rows(ids)):
            return self.model.block.proj(torch.ones(len(ids), 2, 8))

    def test_prior_is_base_and_zero_kl(self):
        actual = self.forward([0,1])
        expected = self.model.block.proj.inner(torch.ones(2,2,8))
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(float(self.adapter.provide(self.state)["latent_kl"].detach()), 0)

    def test_only_selected_posteriors_receive_gradients(self):
        self.state.decoder.heads[self.path].data.fill_(.1)
        self.forward([0,2]).sum().backward()
        self.assertTrue(torch.equal(self.state.decoder.mu.grad[1], torch.zeros(4)))
        self.assertGreater(float(self.state.decoder.mu.grad[0].abs().sum()), 0)
        self.assertIsNone(self.model.block.proj.inner.weight.grad)

    def test_checkpoint_resumes_or_resets_only_the_new_task_bank(self):
        self.state.decoder.mu.data.fill_(.7)
        self.state.decoder.log_std.data.fill_(-.2)
        self.state.decoder.heads[self.path].data.fill_(.1)
        payload = self.adapter.emit(self.state)
        resumed = self.adapter.build(self.sites, self.init)
        self.adapter.load(resumed, payload)
        self.assertEqual(self.adapter.emit(resumed), payload)
        new = self.adapter.build(self.sites, {**self.init, "tasks": 2, "scope": "new", "train_decoder": False})
        self.adapter.load(new, payload)
        self.assertTrue(torch.equal(new.decoder.mu, torch.zeros(2,4)))
        self.assertTrue(torch.equal(new.decoder.log_std, torch.zeros(2,4)))
        self.assertTrue(torch.equal(new.decoder.heads[self.path], self.state.decoder.heads[self.path]))
        self.assertTrue(all(not p.requires_grad for p in new.decoder.mapper()))
        self.assertEqual(set(self.adapter.param_groups(new)), {"mean", "scale"})

    def test_task_swaps_change_the_output_without_changing_decoder(self):
        self.state.decoder.heads[self.path].data.fill_(.1)
        self.state.decoder.mu.data[1].fill_(1)
        out = self.forward([0,1])
        self.assertFalse(torch.allclose(out[0], out[1]))
        self.adapter.uninstall(self.model, self.state)
        self.assertIsInstance(self.model.block.proj, torch.nn.Linear)

    def test_split_microbatches_preserve_sequence_and_kl_gradients(self):
        from rlstack.data.flatten import Flat, pack
        from rlstack.training.losses.base import PolicyOutputs
        from rlstack.training.losses.factual_sft import factual_sft
        docs = [(Flat(tuple(range(n)), (1,)*n, (0,)*n, (0.,)*n, n), {}) for n in (3, 8, 2)]
        grads = []
        for limit in (100, 4):
            x = torch.tensor(2., requires_grad=True)
            total = 0
            for batch in pack(docs, limit):
                out = PolicyOutputs(-x.expand(len(batch)), {"latent_kl": x*x, "factual_beta": .1})
                total = total + factual_sft(out, batch).loss
            total.backward()
            grads.append((float(total.detach()), float(x.grad)))
        for whole, split in zip(grads[0], grads[1]):
            self.assertAlmostEqual(whole, split, places=6)
