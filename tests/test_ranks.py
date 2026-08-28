"""The rank chorus's teardown: nobody is left holding the container.

A follower waits for its next verb inside an NCCL collective, where no signal
handler gets a turn — so a rank that misses its STOP does not die of SIGTERM,
and the driver's daemonic children are joined WITHOUT a timeout when the
interpreter finally exits. That combination is what cost #53 a finished run's
whole report: one wedged rank, a shutdown that hung, and a container killed at
its 30-second grace with the stdout still in the buffer.

The escalation is therefore the thing under test, and it is testable without
metal: `RankGroup.escalate` speaks only the process API, so a stub child that
IGNORES SIGTERM stands in for a rank wedged in a collective exactly where it
matters — the polite signal lands, nothing happens, and SIGKILL has to finish
the job.

Torch-gated (ranks.py is trainer metal), which is why torch is imported inside
the tests rather than at the top: this module is also what a spawned stub
re-imports, and it has to stay cheap enough that spawning them costs
milliseconds.
"""

from __future__ import annotations

import importlib.util
import multiprocessing
import time
import unittest

needs_torch = unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "ranks.py is trainer metal: this suite runs in the image")


# ---- the stub ranks (module-level: a spawned child imports them by name) ----

def leaves_at_once(ready) -> None:
    """A rank that heard STOP and went — the shape of a healthy teardown."""
    ready.set()


def ignores_sigterm(ready) -> None:
    """A rank wedged in a collective, as seen from outside: the polite signal
    lands on a process that will never act on it. `ready` is set only once the
    handler is installed, so a test can never race the wedge it is staging."""
    import signal

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    ready.set()
    while True:
        time.sleep(3600)


class ChorusTeardownTest(unittest.TestCase):
    """One test per rung of the ladder, plus the promise the ladder makes."""

    def children(self, *targets) -> tuple:
        """Spawned stub ranks 1..n, each awaited into its steady state and
        killed at teardown whatever the test did — a SIGTERM-ignoring child
        left behind would hang the test runner's own exit in exactly the way
        this file is about."""
        context = multiprocessing.get_context("spawn")
        ready = [context.Event() for _ in targets]
        started = tuple(context.Process(target=target, args=(event,),
                                        daemon=True)
                        for target, event in zip(targets, ready))
        for child in started:
            child.start()
        self.addCleanup(self.reap, started)
        for event in ready:
            self.assertTrue(event.wait(60.0), "a stub rank never started")
        return started

    def reap(self, children) -> None:
        for child in children:
            if child.is_alive():
                child.kill()
            child.join(timeout=10.0)

    def group(self, *targets):
        """A rank-0 group over stub children, built without a rendezvous: the
        teardown ladder is process management and knows nothing about NCCL."""
        from rlstack.runner.learners.ranks import RankGroup

        children = self.children(*targets)
        return RankGroup(rank=0, width=len(children) + 1, port=0,
                         children=children)

    @needs_torch
    def test_a_rank_that_leaves_on_its_own_is_never_signalled(self) -> None:
        group = self.group(leaves_at_once, leaves_at_once)
        teardown = group.escalate(grace_s=10.0, signal_grace_s=2.0)
        self.assertTrue(teardown.graceful)
        self.assertEqual((teardown.deaf, teardown.wedged, teardown.lost),
                         ((), (), ()))

    @needs_torch
    def test_a_rank_wedged_past_sigterm_is_killed_and_named(self) -> None:
        """The observed failure: terminate() lands on a process that cannot
        answer it. The ladder must not stop there, and must say what it did."""
        group = self.group(ignores_sigterm)
        teardown = group.escalate(grace_s=0.3, signal_grace_s=0.3)
        self.assertEqual(teardown.deaf, (1,))
        self.assertEqual(teardown.wedged, (1,))     # SIGTERM was not enough
        self.assertEqual(teardown.lost, ())         # SIGKILL was
        self.assertFalse(teardown.graceful)
        self.assertFalse(group.children[0].is_alive())
        self.assertIn("survived SIGTERM", teardown.line())

    @needs_torch
    def test_the_ranks_reported_are_the_ranks_that_wedged(self) -> None:
        """Position IS rank: children are spawned in rank order, so a report
        that names ranks has to name them from the same list."""
        group = self.group(leaves_at_once, ignores_sigterm, leaves_at_once)
        teardown = group.escalate(grace_s=1.0, signal_grace_s=0.3)
        self.assertEqual(teardown.wedged, (2,))
        self.assertEqual(
            teardown.line(),
            "[chorus] teardown: ranks [2] ignored STOP → SIGTERM; "
            "ranks [2] survived SIGTERM → SIGKILL")

    @needs_torch
    def test_the_teardown_budget_is_shared_not_per_child(self) -> None:
        """A width-8 chorus must not multiply the promise by eight: the wait
        for the children is ONE deadline, because the venue that kills the
        container is counting wall clock."""
        group = self.group(ignores_sigterm, ignores_sigterm, ignores_sigterm)
        started = time.monotonic()
        survivors = group.join_survivors(0.4)
        elapsed = time.monotonic() - started
        self.assertEqual([rank for rank, _ in survivors], [1, 2, 3])
        self.assertLess(elapsed, 1.0)               # per-child would be 1.2+

    @needs_torch
    def test_a_lone_rank_and_a_follower_end_without_ceremony(self) -> None:
        """stop() is rank 0's verb over a chorus that exists; everyone else
        returns the empty record rather than reaching for a collective."""
        from rlstack.runner.learners.ranks import RankGroup, Teardown

        self.assertEqual(RankGroup(rank=0, width=1, port=0).stop(), Teardown())
        follower = RankGroup(rank=1, width=2, port=0)
        self.assertEqual(follower.stop(), Teardown())
        self.assertTrue(Teardown().graceful)


if __name__ == "__main__":
    unittest.main()
