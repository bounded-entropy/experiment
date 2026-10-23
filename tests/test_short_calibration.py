"""Short-corpus calibration agrees with the existing dense moment convention."""
import unittest
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from rlstack.policy.adapters import dream_bank_torch as bank


@unittest.skipIf(torch is None, "CPU torch required for calibration")
class ShortCalibration(unittest.TestCase):
    def test_thin_factorization_matches_dense_spectrum_and_penalty(self):
        torch.manual_seed(3)
        path = 'wide'
        state = bank.DreamBankState(2, 0, 0., 4, 'calibrate', (path,), {})
        x = torch.randn(7, 1030)
        bank.accumulate(state, path, x[:3])
        bank.accumulate(state, path, x[3:])
        self.assertFalse(state.sums)
        self.assertEqual(state.count, 7)
        anchor = bank.factorize(state)
        values = torch.linalg.eigvalsh(x @ x.T / 7).flip(0)
        ridge = bank.ANCHOR_RIDGE * x.square().sum() / (7*1030)
        self.assertTrue(torch.allclose(anchor.e[path]+anchor.delta[path], values[:4]+ridge, rtol=1e-5))
        self.assertAlmostEqual(anchor.delta[path], float((values[4:].sum()+ridge*1026)/1026), places=6)
        torch.testing.assert_close(anchor.u[path].T @ anchor.u[path], torch.eye(4), atol=1e-5, rtol=1e-5)

    def test_full_sample_rank_reconstructs_empirical_moment_with_ridge_once(self):
        path = 'wide'
        state = bank.DreamBankState(2, 0, 0., 12, 'calibrate', (path,), {})
        x = torch.randn(7, 1030)
        bank.accumulate(state, path, x)
        anchor = bank.factorize(state)
        reconstructed = (anchor.u[path] * anchor.e[path]) @ anchor.u[path].T
        reconstructed += anchor.delta[path] * torch.eye(1030)
        expected = x.T @ x / 7
        expected += bank.ANCHOR_RIDGE * expected.trace() / 1030 * torch.eye(1030)
        torch.testing.assert_close(reconstructed, expected, atol=2e-5, rtol=1e-4)

    def test_switch_to_dense_retains_every_row_once(self):
        path = 'wide'
        state = bank.DreamBankState(2, 0, 0., 4, 'calibrate', (path,), {})
        x = torch.randn(1031, 1025)
        bank.accumulate(state, path, x[:4])
        bank.accumulate(state, path, x[4:])
        self.assertFalse(state.samples)
        torch.testing.assert_close(state.sums[path], x.T @ x)
        self.assertEqual(state.count, len(x))
