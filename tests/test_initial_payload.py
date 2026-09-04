"""The init function: ONE definition of a bank entry's version zero.

ADR 0006 Part B, Q6 (Samarth's fold): every adapter type carries its own init
function, and both paths into a run's first bundle go through it — the
learner's `install`, and a run with NO learner building its v0 at Phase 1.
What these tests hold is the consequence: the bytes are the same, so the
content-addressed bundle id is the same, so a generation-only run serves
exactly the policy a training run's Phase 1 would have served.

The fake half runs everywhere (`fake_initial_payload` is the fake world's init
function, called by FakeLearner AND by the registered `fake` adapter type);
the real half needs tensors and is torch-gated.
"""

from __future__ import annotations

import unittest

from rlstack.policy.compile import compile_bundle
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES
from rlstack.runner.fakes import FakeLearner, fake_initial_payload
from rlstack.runner.interfaces import (
    EntryInstall, OptimSettings, Parameterization,
)

WIDTH, VOCAB = 8, 11

try:
    import torch
except ImportError:                                  # pragma: no cover
    torch = None

needs_torch = unittest.skipUnless(torch is not None, "torch not installed")


def a_site(path: str = "layers.0.self_attn.q_proj") -> SiteMeta:
    return SiteMeta(name=path, path=path, has_weight=True,
                    shape=(WIDTH, VOCAB), is_boundary=False)


def frozen_bank(adapter_type: str) -> tuple[EntryInstall, ...]:
    """Two FROZEN entries — the bank a run with no learner may carry."""
    return tuple(
        EntryInstall(name=name, adapter_type=adapter_type,
                     init={"r": 2, "seed": seed}, trainable=False,
                     sites=(a_site(path),))
        for name, path, seed in (("pi", "first", 5), ("theta", "second", 9)))


def a_parameterization(entries: tuple[EntryInstall, ...],
                       base: str = "toy") -> Parameterization:
    return Parameterization(
        base=base, loss="grpo", entries=entries,
        optim=OptimSettings(name="adamw", lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.0, overrides={}))


class TheSeamTest(unittest.TestCase):
    """`initial_payload` is `emit(initial_params(...))`, said once."""

    def test_the_init_function_is_emit_of_the_init_params(self) -> None:
        instance = ADAPTER_TYPES.get("fake").instance
        sites, init = (a_site(),), {"r": 2, "seed": 5}
        self.assertEqual(instance.initial_payload(sites, init),
                         instance.emit(instance.initial_params(sites, init)))

    def test_version_zero_is_the_sites_and_the_init_and_nothing_else(self) -> None:
        """No entry name, no bank, no tenant: two entries with the same sites
        and the same seed ARE the same delta, as they are under torch."""
        instance = ADAPTER_TYPES.get("fake").instance
        first = instance.initial_payload((a_site(),), {"r": 2, "seed": 5})
        same = instance.initial_payload((a_site(),), {"seed": 5, "r": 2})
        other = instance.initial_payload((a_site(),), {"r": 2, "seed": 6})
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)


class LearnerBuiltEqualsLearnerLessTest(unittest.TestCase):
    """The promise, on fakes: whoever built v0, it is the same bundle."""

    def payloads(self, entries) -> tuple[dict, dict]:
        learner = FakeLearner()
        learner.install("run-a", a_parameterization(entries))
        built = dict(learner.emit("run-a").adapters)
        alone = {entry.name: ADAPTER_TYPES.get(entry.adapter_type).instance
                 .initial_payload(entry.sites, entry.init)
                 for entry in entries}
        return built, alone

    def test_the_same_bytes_and_the_same_bundle_id(self) -> None:
        entries = frozen_bank("fake")
        built, alone = self.payloads(entries)
        self.assertEqual(built, alone)

        names = sorted(entry.name for entry in entries)
        version = {name: 0 for name in names}
        types = {entry.name: entry.adapter_type for entry in entries}
        self.assertEqual(
            compile_bundle(built, version, names, types).bundle_id,
            compile_bundle(alone, version, names, types).bundle_id)

    def test_the_fake_learner_and_the_fake_adapter_type_are_one_function(self) -> None:
        """Not a coincidence to be re-checked: both call
        `fake_initial_payload`, which is the fake world's init function."""
        entry = frozen_bank("fake")[0]
        built, _ = self.payloads((entry,))
        self.assertEqual(built[entry.name],
                         fake_initial_payload(entry.sites, entry.init))


@needs_torch
class RealAdapterTypeTest(unittest.TestCase):
    """The same promise where the numerics are real: a lora entry installed on
    a learner emits exactly what its init function builds with no learner at
    all. Torch-gated because `params` builds tensors."""

    def test_a_lora_v0_is_the_same_whoever_built_it(self) -> None:
        from test_remote_learner import _ToyLM
        from rlstack.runner.learners.torch_learner import TorchLearner

        entries = frozen_bank("lora")
        learner = TorchLearner(device="cpu", dtype=torch.float32)
        learner._model = _ToyLM()
        learner._base = "toy"
        learner.install("run-a", a_parameterization(entries))
        built = dict(learner.emit("run-a").adapters)

        instance = ADAPTER_TYPES.get("lora").instance
        alone = {entry.name: instance.initial_payload(entry.sites, entry.init)
                 for entry in entries}
        self.assertEqual(built, alone)


if __name__ == "__main__":
    unittest.main()
