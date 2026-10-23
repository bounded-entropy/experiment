"""Fake routed entries preserve inactive weights and independent optimizer ages."""
from dataclasses import replace
import unittest

from rlstack.runner.fakes import FakeLearner
from test_gradient_measurement_remote import installation, batch


def routed_installation():
    parameterization = installation()
    entry = parameterization.entries[0]
    return replace(parameterization, entries=(
        replace(entry, name="first", route="first", init={"r": 2, "seed": 7}),
        replace(entry, name="second", route="second", init={"r": 2, "seed": 9})))


def routed_batch(route):
    return batch(extras=(({'route': route},),))


class RoutedFakeEntriesTest(unittest.TestCase):
    def learner(self):
        learner = FakeLearner()
        learner.install("run", routed_installation())
        return learner

    def step(self, learner, route, scales=None):
        learner.forward_backward("run", routed_batch(route))
        learner.optim_step("run", scales)

    def test_inactive_weights_and_optimizer_age_do_not_move(self):
        learner = self.learner()
        before = learner.emit("run")
        self.step(learner, "first")
        first = learner.emit("run")
        self.assertNotEqual(before.adapters["first"], first.adapters["first"])
        self.assertNotEqual(before.optim["first"], first.optim["first"])
        self.assertEqual(before.adapters["second"], first.adapters["second"])
        self.assertEqual(before.optim["second"], first.optim["second"])
        self.step(learner, "second")
        second = learner.emit("run")
        self.assertEqual(first.adapters["first"], second.adapters["first"])
        self.assertEqual(first.optim["first"], second.optim["first"])

    def test_resume_restores_different_entry_ages_and_exact_next_step(self):
        learner = self.learner()
        self.step(learner, "first")
        self.step(learner, "second")
        self.step(learner, "second")
        saved = learner.emit("run")
        resumed = self.learner()
        resumed.load("run", saved.adapters, saved.optim)
        self.assertEqual(saved, resumed.emit("run"))
        self.step(learner, "first", {"first": 0.3})
        self.step(resumed, "first", {"first": 0.3})
        self.assertEqual(learner.emit("run"), resumed.emit("run"))

    def test_unselected_route_is_base_and_unknown_route_is_refused(self):
        learner = self.learner()
        before = learner.emit("run")
        learner.forward_backward("run", batch())
        learner.optim_step("run")
        self.assertEqual(before, learner.emit("run"))
        with self.assertRaisesRegex(ValueError, "unknown adapter route"):
            learner.forward_backward("run", routed_batch("unknown"))

    def test_probe_reads_only_its_selected_entry(self):
        learner = self.learner()
        before = learner.forward("run", routed_batch("second"))
        self.step(learner, "first")
        self.assertEqual(before, learner.forward("run", routed_batch("second")))
        self.step(learner, "second")
        self.assertNotEqual(before, learner.forward("run", routed_batch("second")))
