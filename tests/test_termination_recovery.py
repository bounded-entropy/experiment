"""Provider termination is confirmed by the provider, never by a stop request.

The Strange Loop worktree's desk-side cases (retirement custody, stopped
instances, delivery-intent barriers) ride the desk custody rewrite and land
with ADR 0014 Part C; what stands here is the provider half the shared
`AllocationProvider` contract already promises (ADR 0017): `terminate` is True
only after that exact allocation has ended.
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch


class ProviderTerminationTests(unittest.TestCase):
    def provider(self, rows, delete_error=None):
        from rlstack.runner.venues.strangeloop.provider import StrangeLoopCLI

        provider = StrangeLoopCLI(executable="unused-test-cli")
        def run(*args, timeout=180):
            if args[:2] == ("gpu", "down"):
                if delete_error:
                    raise delete_error
                return {"status": "released"}
            return {"status": next(rows)}
        provider.run = Mock(side_effect=run)
        return provider

    def test_lost_delete_response_is_confirmed_by_status_without_another_delete(self):
        import subprocess
        provider = self.provider(iter(["ready", "ready", "released"]),
                                 subprocess.TimeoutExpired("delete", 1))
        with patch("rlstack.runner.venues.strangeloop.provider.time.sleep"):
            self.assertTrue(provider.down("exact-lease"))
        deletes = [call for call in provider.run.call_args_list if call.args[:2] == ("gpu", "down")]
        self.assertEqual(len(deletes), 1)
        self.assertEqual(deletes[0].args[2], "exact-lease")

    def test_an_already_released_allocation_needs_no_delete(self):
        provider = self.provider(iter(["released"]))
        self.assertTrue(provider.down("exact-lease"))
        self.assertEqual(provider.run.call_count, 1)

    def test_successful_delete_without_an_ended_status_remains_unresolved(self):
        # the delete landed, but every read still says ready: not confirmed
        provider = self.provider(iter(["ready"] * 7))
        with patch("rlstack.runner.venues.strangeloop.provider.time.sleep"):
            self.assertFalse(provider.down("exact-lease"))
        self.assertEqual(provider.run.call_count, 8)

    def test_a_failed_lease_is_confirmed_ended(self):
        # the platform closes a lease whose pod is gone and reports `failed`
        # forever, never `released` (2026-09-17): that is an ended lease
        provider = self.provider(iter(["ready", "failed"]))
        with patch("rlstack.runner.venues.strangeloop.provider.time.sleep"):
            self.assertTrue(provider.down("exact-lease"))


if __name__ == "__main__":
    unittest.main()
