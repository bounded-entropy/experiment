"""The SHARDED learner's sleep (#82): the alias, the relabel, the chorus.

`fully_shard` turns each base parameter into a DTensor whose local tensor is
this rank's chunk, and keeps a private flat view of that same storage
(`FSDPParam._sharded_param_data`) — the thing an all-gather reads. Moving the
parameter without re-aliasing that view is silent damage: the old storage
stays alive, so no device memory comes back, and the next all-gather reads a
device the base has left. `FSDPModule._apply` re-aliases, which is what makes
the offload a plain `Module.to`.

THREE LAYERS, because the claim has three parts and they run in three places.
The PROBE is a fact about the pinned torch and runs wherever torch is
installed — including the image, whose `run_tests` has no card. The CHORUS
claim (sleep and wake are announced verbs a follower can run) is about the
verb table and needs no device either. The MECHANISM — the local tensors
actually on the host, the flat view actually re-aliased, the device memory
actually returned and a forward bit-identical across the cycle — needs a GPU,
and no test runner in this repo has one today: `deploy/steer_l4.py::run_tests`
and `concept_steer.py::run_tests` are GPU-less by construction. So the cases
below that ask for CUDA skip everywhere for now, and the thing that actually
proves the offload at width 2 is `deploy/stress_fleet.py::learner_sleep`,
which is written and UNRUN.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:                                  # pragma: no cover
    torch = None

needs_torch = unittest.skipUnless(torch is not None, "torch not installed")
needs_cuda = unittest.skipUnless(
    torch is not None and torch.cuda.is_available(),
    "a sharded offload is a claim about device memory: this case needs a GPU")


# ---------------------------------------------------------------------------
# the probe: what this build asks of its torch before it promises to sleep
# ---------------------------------------------------------------------------

class WithoutTheHook:
    """A torch that never learned the relabel — FSDP2's shape before
    `reset_sharded_param` and the `_apply` override, as a stand-in."""

    def reshard(self) -> None: ...


class WithTheHook:
    """And one that did."""

    def _apply(self, *args, **kwargs) -> None: ...
    def reshard(self) -> None: ...
    def _get_fsdp_state(self): ...
    def reset_sharded_param(self) -> None: ...


@needs_torch
class SleepProbeTest(unittest.TestCase):
    """I7 at the one place a missing hook would otherwise surface as a
    corrupt all-gather: the substrate is certified at BUILD, and a build that
    cannot sleep says so rather than failing at the first evict."""

    def test_the_pinned_torch_carries_every_hook_the_offload_needs(self) -> None:
        """The version assumption, as a test. The day torch drops the relabel
        or moves FSDP2's house again, this goes red HERE — where the reason is
        readable — instead of on metal at the first alternation."""
        from rlstack.runner.learners.fsdp_torch import probe_sharded_sleep

        probe = probe_sharded_sleep()
        self.assertTrue(probe.supported, probe.reason)
        self.assertEqual(probe.reason, "")

    def test_a_torch_without_the_relabel_refuses_and_names_what_is_missing(self) -> None:
        from rlstack.runner.learners.fsdp_torch import sleep_probe_of

        probe = sleep_probe_of(WithoutTheHook, WithoutTheHook)
        self.assertFalse(probe.supported)
        self.assertIn("FSDPModule._apply", probe.reason)
        self.assertIn("FSDPModule._get_fsdp_state", probe.reason)
        self.assertIn("FSDPParam.reset_sharded_param", probe.reason)
        self.assertIn("re-aliasing", probe.reason)

    def test_a_torch_with_every_hook_is_taken_at_its_word(self) -> None:
        from rlstack.runner.learners.fsdp_torch import sleep_probe_of

        self.assertEqual(sleep_probe_of(WithTheHook, WithTheHook).supported, True)


# ---------------------------------------------------------------------------
# the chorus: one verb, entered by every rank
# ---------------------------------------------------------------------------

class Heard:
    """A RankGroup that records what rank 0 announced instead of broadcasting
    it — the wire, minus the wire. Width 2 so `announce` does not take its
    degenerate path."""

    rank = 0
    width = 2

    def __init__(self) -> None:
        self.said: list[str] = []

    @property
    def device_str(self) -> str:
        return "cpu"

    def announce(self, command) -> None:
        self.said.append(command.verb)


@needs_torch
class ChorusVerbTest(unittest.TestCase):
    """A sleep that swung only rank 0 would leave every other rank holding its
    shard through the engine's whole generation — which is the entire memory
    the alternation exists to buy. So sleep and wake are announced verbs, and
    a follower reaches the same body off the wire."""

    def learner(self):
        from rlstack.runner.learners.fsdp_torch import FsdpTorchLearner

        return FsdpTorchLearner(Heard())

    def test_sleep_and_wake_are_announced(self) -> None:
        import asyncio

        learner = self.learner()
        asyncio.run(learner.sleep())
        asyncio.run(learner.wake())
        self.assertEqual(learner.ranks.said, ["sleep", "wake"])

    def test_the_follower_table_holds_every_announced_verb(self) -> None:
        """The table IS the contract: a verb rank 0 announces and a follower
        does not know is a chorus that deadlocks on the next collective."""
        from rlstack.runner.learners.ranks import RankCommand

        learner = self.learner()
        learner.follow(RankCommand("sleep", ()))
        learner.follow(RankCommand("wake", ()))
        with self.assertRaises(ValueError):
            learner.follow(RankCommand("nap", ()))

    def test_a_follower_never_announces(self) -> None:
        """`announce` is rank 0's alone — two ranks each starting a broadcast
        is two ranks each waiting for the other's."""
        import asyncio

        learner = self.learner()
        learner.ranks.rank = 1
        asyncio.run(learner.sleep())
        self.assertEqual(learner.ranks.said, [])

    def test_sleep_is_quiet_on_a_learner_that_holds_no_base(self) -> None:
        """A host may evict a learner it has not installed a tenant on, and
        the chorus still has to hear the verb — the announce is unconditional,
        the move is not."""
        import asyncio

        learner = self.learner()
        asyncio.run(learner.sleep())
        self.assertFalse(learner._asleep)
        self.assertEqual(learner.ranks.said, ["sleep"])


# ---------------------------------------------------------------------------
# the mechanism: the shards move and FSDP's view moves with them
# ---------------------------------------------------------------------------

def tiny_causal_lm(width: int, blocks: int):
    """A stand-in for the base with the ONE shape `decoder_blocks` addresses —
    `model.layers` — plus something no block owns (the embedding), because
    that is the tensor the SECOND `fully_shard` covers and the one a move at
    the root has to reach on its way out."""
    import torch.nn as nn

    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(width, width)
            self.layers = nn.ModuleList(
                [nn.Linear(width, width, bias=False) for _ in range(blocks)])

    class Base(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()

        def forward(self, ids):
            hidden = self.model.embed(ids)
            for layer in self.model.layers:
                hidden = layer(hidden)
            return hidden

    return Base()


@needs_cuda
class ShardedOffloadTest(unittest.TestCase):
    """Width 1 is the degenerate chorus — no children, no broadcast — and the
    base is `fully_shard`-wrapped at width 1 exactly as it is at width 8, so
    every claim below except the collective is testable on one card."""

    def learner(self):
        from rlstack.runner.learners.fsdp_torch import FsdpTorchLearner
        from rlstack.runner.learners.ranks import RankGroup

        ranks = RankGroup(rank=0, width=1, port=0)
        built = FsdpTorchLearner(ranks, dtype=torch.float32,
                                 checkpoint_activations=False)
        built._model = tiny_causal_lm(8, 2).to(torch.float32)
        built._base = "tiny"
        built.shard_the_frozen_base()
        self.addCleanup(self.leave_the_group)
        return built

    def leave_the_group(self) -> None:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()

    def local_tensors(self, learner) -> list:
        from rlstack.runner.learners.fsdp_torch import fsdp_params_of

        return [param for module in learner.sharded_modules()
                for param in fsdp_params_of(module)]

    def test_a_cycle_moves_every_shard_and_re_aliases_fsdp_s_view(self) -> None:
        learner = self.learner()
        held = torch.cuda.memory_allocated()

        learner.hand_the_device_back()
        for param in self.local_tensors(learner):
            self.assertEqual(param.sharded_param._local_tensor.device.type, "cpu")
            self.assertEqual(param._sharded_param_data.device.type, "cpu")
            self.assertEqual(param._sharded_param_data.data_ptr(),
                             param.sharded_param._local_tensor.data_ptr())
        self.assertLess(torch.cuda.memory_allocated(), held)

        learner.take_the_device_back()
        for param in self.local_tensors(learner):
            self.assertEqual(param.sharded_param._local_tensor.device.type, "cuda")
            self.assertEqual(param._sharded_param_data.data_ptr(),
                             param.sharded_param._local_tensor.data_ptr())

    def test_a_forward_is_bit_identical_across_a_cycle(self) -> None:
        """The offload may cost PCIe time and nothing else: the same tokens
        through the same frozen weights are the same numbers, to the bit."""
        learner = self.learner()
        ids = torch.zeros((2, 3), dtype=torch.long, device="cuda:0")
        before = learner._model(ids).detach().clone()

        learner.hand_the_device_back()
        learner.take_the_device_back()
        after = learner._model(ids)
        self.assertEqual(float((after - before).abs().max()), 0.0)

    def test_a_sleep_before_the_first_forward_wakes_into_one(self) -> None:
        """The common alternation case: a tenant installs and the engine
        samples wave 1 before any backward, so the learner sleeps between the
        wrap and FSDP's own lazy init. Both paths re-alias — the substrate's
        `_apply` on the way out here, `lazy_init` on the way in later — and
        doing it twice is idempotent."""
        learner = self.learner()
        learner.hand_the_device_back()
        learner.take_the_device_back()
        ids = torch.zeros((2, 3), dtype=torch.long, device="cuda:0")
        self.assertEqual(learner._model(ids).shape, (2, 3, 8))

    def test_the_attestation_catches_a_view_left_behind(self) -> None:
        """The failure this exists to make loud: a shard whose storage moved
        and whose flat view did not. Staged by moving one local tensor without
        the relabel, which is exactly what a torch missing the hook would do
        to every one of them."""
        learner = self.learner()
        param = self.local_tensors(learner)[0]
        param.sharded_param._local_tensor = \
            param.sharded_param._local_tensor.cpu()
        with self.assertRaises(RuntimeError) as caught:
            learner.attest_the_shards_are_realiased()
        self.assertIn("did not re-alias", str(caught.exception))


if __name__ == "__main__":                           # pragma: no cover
    unittest.main()
