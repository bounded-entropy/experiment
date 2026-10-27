"""Gradient scalar records cross both existing learner doors without tensors.

These are stdlib fake/serialization checks. They establish transport and custody,
not numerical correctness of the fake learner's synthetic measurements.
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
import unittest

import rlstack
from common import arith_store
from rlstack import FakeEngine, FakeLearner, Host, HostService, LocalTransport, Regime, RemoteLearner
from rlstack.data.flatten import TokenBatch
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.interfaces import EntryInstall, OptimSettings, Parameterization
from rlstack.runner.remote import LearnerService, LEARNER_VERBS


def installation():
    site = SiteMeta(name="proj", path="proj", has_weight=True,
                    shape=(6, 6), is_boundary=False)
    return Parameterization(base="toy", loss="sequence_sft",
        entries=(EntryInstall(name="pi", adapter_type="lora", init={"r": 2, "seed": 7},
                              trainable=True, sites=(site,)),),
        optim=OptimSettings(name="adamw", lr=0.01, betas=(0.9, 0.95),
                            weight_decay=0.0, overrides={}))


def batch(ids=(1, 2, 3), mask=(0, 1, 1), extras=()):
    return TokenBatch(token_ids=ids, loss_mask=mask,
        behavior_logprobs=(0.0,) * len(ids), segment_ids=(0,) * len(ids),
        doc_starts=(0,), doc_turn_extras=extras)


REFERENCES = (batch(extras=(({'route': 'base', 'draw': [1, 2]},),)),
              batch((4, 5, 6, 7), (0, 0, 1, 1)))
REFERENCES = tuple(replace(item, documents_in_update=2, microbatches_in_update=2) for item in REFERENCES)
CANDIDATES = ((batch(),), (batch((7, 8, 9, 10), (0, 1, 1, 1)),))


class RecordingLearner(FakeLearner):
    def gradient_cosines(self, tenant, entry, reference_batches, candidate_batches):
        from rlstack.client import GradientComparison, GradientDocumentStats
        self.seen = (tenant, entry, tuple(reference_batches),
                     tuple(tuple(group) for group in candidate_batches))
        docs = (GradientDocumentStats(nll_sum=3.5, scored_tokens=2),)
        return (
            GradientComparison(cosine=0.25, dot=1.5, reference_norm=2.0,
                candidate_norm=3.0, valid=True, reason=None,
                reference_documents=docs, candidate_documents=docs,
                reference_seconds=0.125, candidate_seconds=0.5, parameter_count=24,
                parameter_schema_sha256="a" * 64, min_norm=1e-12),
            GradientComparison(cosine=None, dot=0.0, reference_norm=2.0,
                candidate_norm=0.0, valid=False, reason="zero_candidate_norm",
                reference_documents=docs, candidate_documents=docs,
                reference_seconds=0.125, candidate_seconds=0.0625, parameter_count=24,
                parameter_schema_sha256="a" * 64, min_norm=1e-12),
        )


class GradientWireTest(unittest.TestCase):
    def test_verb_is_part_of_existing_learner_service(self):
        self.assertIn("gradient_cosines", LEARNER_VERBS)

    def test_records_and_nested_batch_groups_round_trip_resident_door(self):
        served = RecordingLearner()
        proxy = RemoteLearner(LocalTransport(LearnerService(served)))
        direct = RecordingLearner().gradient_cosines("student", "pi", REFERENCES, CANDIDATES)
        got = proxy.gradient_cosines("student", "pi", REFERENCES, CANDIDATES)
        self.assertEqual(got, direct)
        self.assertIsInstance(got, tuple)
        self.assertEqual(served.seen, ("student", "pi", REFERENCES, CANDIDATES))
        self.assertIsNone(got[1].cosine)
        self.assertFalse(got[1].valid)
        self.assertEqual(got[0].reference_documents[0].scored_tokens, 2)
        self.assertEqual(got[0].parameter_schema_sha256, "a" * 64)

    def test_kl_prefix_and_separate_loss_statistics_cross_wire(self):
        from rlstack.client import GradientDocumentStats
        class KLLearner(RecordingLearner):
            def gradient_cosines(self, tenant, entry, refs, candidates, teacher_prefix_ids=None):
                self.prefix = teacher_prefix_ids
                results = super().gradient_cosines(tenant, entry, refs, candidates)
                return tuple(replace(r, candidate_documents=(GradientDocumentStats(
                    nll_sum=4., scored_tokens=2, kl_sum=0.3, teacher_entropy_sum=2.,
                    student_cross_entropy_sum=2.3, objective="teacher_kl"),)) for r in results)
        served = KLLearner()
        proxy = RemoteLearner(LocalTransport(LearnerService(served)))
        result = proxy.gradient_cosines("student", "pi", REFERENCES, CANDIDATES, teacher_prefix_ids=(1, 2))
        self.assertEqual(served.prefix, (1, 2))
        self.assertEqual(result[0].candidate_documents[0].kl_sum, 0.3)
        self.assertEqual(result[0].candidate_documents[0].nll_sum, 4.)

    def test_host_door_admits_and_releases_normal_learner_custody(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, _, _ = arith_store(tmp.name)
        served = RecordingLearner()
        host = Host("grad-host", engines=(FakeEngine(base="toy"),), learner=served,
            store=store, regimes=(Regime("serve", "inference", "toy", 1),
                                  Regime("train", "training", "toy", 1)))
        proxy = RemoteLearner(LocalTransport(HostService(host)), admitted=True)
        want = RecordingLearner().gradient_cosines("student", "pi", REFERENCES, CANDIDATES)
        self.assertEqual(proxy.gradient_cosines("student", "pi", REFERENCES, CANDIDATES), want)
        self.assertEqual(host.arbiter.in_flight(), 0)
        self.assertEqual(served.seen, ("student", "pi", REFERENCES, CANDIDATES))

    def test_fake_measurement_does_not_step_or_change_existing_tenants(self):
        learner = FakeLearner()
        for name in ("student", "writer"):
            learner.install(name, installation())
        before = {name: learner.emit(name) for name in ("student", "writer")}
        result = learner.gradient_cosines("student", "pi", REFERENCES, CANDIDATES)
        self.assertEqual(len(result), 2)
        self.assertEqual(before, {name: learner.emit(name) for name in before})
        self.assertEqual(result, learner.gradient_cosines("student", "pi", REFERENCES, CANDIDATES))

    def test_service_failure_does_not_leave_host_admission_in_flight(self):
        class FailingLearner(RecordingLearner):
            def gradient_cosines(self, *args, **kwargs):
                raise ValueError("injected measurement refusal")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, _, _ = arith_store(tmp.name)
        host = Host("grad-host", engines=(FakeEngine(base="toy"),), learner=FailingLearner(),
            store=store, regimes=(Regime("serve", "inference", "toy", 1),
                                  Regime("train", "training", "toy", 1)))
        proxy = RemoteLearner(LocalTransport(HostService(host)), admitted=True)
        with self.assertRaisesRegex(ValueError, "injected measurement refusal"):
            proxy.gradient_cosines("student", "pi", REFERENCES, CANDIDATES)
        self.assertEqual(host.arbiter.in_flight(), 0)


if __name__ == "__main__":
    unittest.main()
