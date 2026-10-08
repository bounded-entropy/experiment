"""ADR 0022's dp axis, in torch, float64 on CPU: the reduction rule.

dp = K must step the SAME gradient dp = 1 accumulates — the same tensors,
added in the same order — so a dp learner keeps each microbatch's gradient
apart (`.grad` cleared before it), gathers them, and adds them in microbatch
order (`OrderedGradientSum`). This suite proves that rule against autograd's
own accumulation, bit for bit, without a process group: the groups are
simulated by computing the microbatches OUT of order, which is exactly the
freedom dp groups have. Torch-gated: skipped where torch is absent.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:          # the stdlib-only fakes suite
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class OrderedGradientSumTest(unittest.TestCase):

    def model(self):
        torch.manual_seed(0)
        base = torch.nn.Linear(16, 16, dtype=torch.float64).requires_grad_(False)
        a = torch.nn.Parameter(torch.randn(4, 16, dtype=torch.float64) * 0.1)
        b = torch.nn.Parameter(torch.randn(16, 4, dtype=torch.float64) * 0.1)
        unused = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        return base, [a, b, unused]

    def loss(self, base, params, batch, *, touch_b: bool = True):
        a, b, _ = params
        hidden = base(batch) + (batch @ a.T @ b.T if touch_b else batch @ a.T @ b.detach().T)
        return hidden.pow(2).mean().log1p()

    def batches(self):
        generator = torch.Generator().manual_seed(1)
        return [torch.randn(5 + i, 16, dtype=torch.float64, generator=generator)
                for i in range(7)]

    def test_ordered_sum_is_autograds_accumulation_bit_for_bit(self) -> None:
        from rlstack.runner.learners.fsdp_torch import OrderedGradientSum

        base, params = self.model()
        batches = self.batches()
        # dp = 1: autograd accumulates microbatch by microbatch
        norms_dp1 = []
        for i, batch in enumerate(batches):
            self.loss(base, params, batch, touch_b=i % 3 != 1).backward()
            norms_dp1.append(sum(float(p.grad.pow(2).sum()) for p in params
                                 if p.grad is not None) ** 0.5)
        accumulated = [None if p.grad is None else p.grad.clone() for p in params]
        # dp = 2: each microbatch alone, computed in GROUP order (evens, then
        # odds), and summed in microbatch order
        alone = {}
        for group in (1, 0):
            for i in range(group, len(batches), 2):
                for p in params:
                    p.grad = None
                self.loss(base, params, batches[i], touch_b=i % 3 != 1).backward()
                alone[i] = [None if p.grad is None else p.grad.clone() for p in params]
        total = OrderedGradientSum(len(params))
        norms_dp2 = []
        for i in range(len(batches)):
            total.add(alone[i])
            norms_dp2.append(total.norm())
        for got, want in zip(total.sums, accumulated):
            if want is None:
                self.assertIsNone(got)                  # never zeros
            else:
                self.assertTrue(torch.equal(got, want))
        self.assertEqual(norms_dp2, norms_dp1)

    def test_packing_round_trips_presence(self) -> None:
        from rlstack.runner.learners.fsdp_torch import packed, unpacked

        _, params = self.model()
        grads = [torch.ones_like(params[0]), None, torch.full_like(params[2], 2.0)]
        flat, present = packed(grads, params)
        back = unpacked(flat, present, params)
        self.assertTrue(torch.equal(back[0], grads[0]))
        self.assertIsNone(back[1])
        self.assertTrue(torch.equal(back[2], grads[2]))


if __name__ == "__main__":
    unittest.main()
