"""The flow graph (rlstack.spec.flow): one canonical walk, two consumers.

Claims under test: the graph's nodes and edges mirror the declarations
(produces/consumes/requires) exactly; feeds_loss is TRANSITIVE reachability
into the loss's requires; the pipeline queries report exactly what validate's
rules report; and every run serializes the graph as its own dictionary.json
at creation, byte-stable across resume.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace

from common import arith_spec, arith_store
from rlstack import (
    AlgoSpec, FakeEngine, FakeLearner, OptimSpec, PostProcessor, Schedule,
    fake_qwen_schema, postprocessor, run_experiment,
)
from rlstack.spec.flow import RAILS, FlowGraph, flow_graph

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def node(graph: FlowGraph, name: str, phase: str):
    matches = [n for n in graph.nodes if n.name == name and n.phase == phase]
    assert len(matches) == 1, f"{name}/{phase}: {matches}"
    return matches[0]


class FlowGraphTest(unittest.TestCase):
    def graph(self, **algo_overrides) -> FlowGraph:
        spec = arith_spec("cas://x/t.jsonl", "cas://y/h.jsonl")
        if algo_overrides:
            spec = replace(spec, algo=replace(spec.algo, **algo_overrides))
        return flow_graph(spec)

    def test_edges_mirror_the_declarations(self) -> None:
        graph = self.graph()   # grpo; post=(verifier, grpo_advantage)
        reward = node(graph, "reward", "post")
        self.assertEqual(reward.producer, "postprocessor:verifier")
        self.assertEqual(reward.consumers, ("grpo_advantage",))
        advantage = node(graph, "advantage", "post")
        self.assertEqual(advantage.producer, "postprocessor:grpo_advantage")
        self.assertEqual(advantage.consumers, ("loss:grpo",))

    def test_feeds_loss_is_transitive(self) -> None:
        """grpo requires advantage; advantage consumes reward — so reward
        feeds the loss too. THE motivating example: what feeds the gradient
        is a graph fact, not a per-edge special case."""
        graph = self.graph()
        self.assertTrue(node(graph, "advantage", "post").feeds_loss)
        self.assertTrue(node(graph, "reward", "post").feeds_loss)
        # eval's reward is measurement, not gradient
        self.assertFalse(node(graph, "reward", "eval").feeds_loss)

    def test_direct_requires_and_measurement_only(self) -> None:
        direct = self.graph(loss="sdft", post=("verifier",))
        self.assertTrue(node(direct, "reward", "post").feeds_loss)
        self.assertIn("loss:sdft", node(direct, "reward", "post").consumers)

        measured = self.graph(loss="replay_distill", post=("verifier",))
        self.assertFalse(node(measured, "reward", "post").feeds_loss)

    def test_records_rails_and_lag_are_in_the_graph(self) -> None:
        # the wave's SHAPE is the plan's (#59); what the schedule still says —
        # and so what the graph can still read off it — is the lag buffer
        graph = self.graph(schedule=Schedule(microbatch_tokens=64,
                                             max_policy_lag=2))
        self.assertEqual(node(graph, "behavior_logprobs", "wave").kind, "record")
        for rail in RAILS:
            self.assertEqual(node(graph, rail, "train").producer, "loss:grpo")
        self.assertEqual(graph.lag, 2)

    def test_pipeline_queries_report_wiring_faults(self) -> None:
        broken = self.graph(post=("grpo_advantage", "verifier"))  # wrong order
        (i, name, want, produced), = broken.missing_consumes("post")
        self.assertEqual((i, name, want, produced),
                         (0, "grpo_advantage", "reward", ()))

        colliding = self.graph(post=("verifier", "verifier"))
        (i, name, column, prior), = colliding.column_collisions("post")
        self.assertEqual((name, column, prior), ("verifier", "reward", "verifier"))

        starved = self.graph(loss="sdft", post=())
        self.assertEqual(starved.unsatisfied_requires(), ("reward",))

    def test_run_serializes_its_own_dictionary(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, heldout = arith_store(tmp.name)
        spec = arith_spec(train, heldout)
        report = run_experiment(spec, SCHEMA, store, FakeEngine(), FakeLearner())

        dictionary = store.peek_dictionary(report.run_id)
        self.assertEqual(dictionary, flow_graph(spec).to_json())
        self.assertEqual(dictionary["loss"], "grpo")
        self.assertEqual(dictionary["post_pipeline"],
                         ["verifier", "grpo_advantage"])
        by_name = {(c["name"], c["phase"]): c for c in dictionary["columns"]}
        self.assertTrue(by_name[("reward", "post")]["feeds_loss"])
        self.assertEqual(by_name[("reward", "post")]["granularity"], "trajectory")
        self.assertEqual(
            by_name[("behavior_logprobs", "wave")]["granularity"], "token")

        # attach again: same bytes (derived and deterministic, never identity)
        before = store.path_of(
            f"runs/{report.run_id}/dictionary.json").read_bytes()
        run_experiment(spec, SCHEMA, store, FakeEngine(), FakeLearner())
        after = store.path_of(
            f"runs/{report.run_id}/dictionary.json").read_bytes()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
