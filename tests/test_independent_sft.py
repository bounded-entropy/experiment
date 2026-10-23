"""Independent SFT denominators survive uneven documents and packed forwards."""
import unittest

from rlstack.data.flatten import TokenBatch
from rlstack.training.losses import PolicyOutputs
from rlstack.training.losses.independent_sft import independent_sft

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU torch required for exact loss and gradient checks")
class IndependentSftTest(unittest.TestCase):
    def test_unequal_students_and_documents_match_separate_objectives(self):
        scores = torch.tensor([0., -.2, -.4, 0., -.6, 0., -.8, -1., -1.2], requires_grad=True)
        # Student0 has two documents of lengths 3 and2; Student1 has one oflength4.
        batch = TokenBatch(token_ids=tuple(range(9)), loss_mask=(0, 1, 1, 0, 1, 0, 1, 1, 1),
            behavior_logprobs=(0.,) * 9, segment_ids=(0,) * 9, doc_starts=(0, 3, 5),
            postdata={"sft_document_weight": (.5,) * 5 + (1.,) * 4}, documents_in_update=3)
        result = independent_sft(PolicyOutputs(logprobs=scores), batch)
        expected = (-scores[1:3].mean() - scores[4]) / 2 - scores[6:9].mean()
        actual_gradient, = torch.autograd.grad(result.loss, scores, retain_graph=True)
        expected_gradient, = torch.autograd.grad(expected, scores)
        torch.testing.assert_close(result.loss, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertEqual(result.components["document_0_tokens"], 2)
        self.assertEqual(result.components["document_1_weight"], .5)
        self.assertEqual(result.components["document_2_weight"], 1.)

    def test_microbatch_partition_does_not_change_objective_or_gradients(self):
        scores = torch.tensor([0., -.2, -.4, 0., -.6], requires_grad=True)
        first = TokenBatch(token_ids=(1, 2, 3), loss_mask=(0, 1, 1),
            behavior_logprobs=(0.,) * 3, segment_ids=(0,) * 3, doc_starts=(0,),
            postdata={"sft_document_weight": (.5,) * 3}, documents_in_update=2)
        second = TokenBatch(token_ids=(4, 5), loss_mask=(0, 1),
            behavior_logprobs=(0.,) * 2, segment_ids=(0,) * 2, doc_starts=(0,),
            postdata={"sft_document_weight": (.5,) * 2}, documents_in_update=2)
        actual = (independent_sft(PolicyOutputs(logprobs=scores[:3]), first).loss
                  + independent_sft(PolicyOutputs(logprobs=scores[3:]), second).loss)
        expected = (-scores[1:3].mean() - scores[4]) / 2
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, scores, retain_graph=True)[0],
                                   torch.autograd.grad(expected, scores)[0])

    def test_zero_target_document_counts_in_group_but_contributes_no_gradient(self):
        scores = torch.tensor([0., -.2, 0., -.6], requires_grad=True)
        batch = TokenBatch(token_ids=(1, 2, 3, 4), loss_mask=(0, 0, 0, 1),
            behavior_logprobs=(0.,) * 4, segment_ids=(0,) * 4, doc_starts=(0, 2),
            postdata={"sft_document_weight": (0., 0., 0., .5)}, documents_in_update=2)
        measured = independent_sft(PolicyOutputs(logprobs=scores), batch)
        torch.testing.assert_close(measured.loss, -.5 * scores[3])
        gradient, = torch.autograd.grad(measured.loss, scores)
        torch.testing.assert_close(gradient, torch.tensor([0., 0., 0., -.5]))
        self.assertEqual(measured.components["document_0_tokens"], 0.)
        self.assertEqual(measured.components["document_0_nll_sum"], 0.)
        all_zero = TokenBatch(token_ids=(1, 2), loss_mask=(0, 0),
            behavior_logprobs=(0., 0.), segment_ids=(0, 0), doc_starts=(0,),
            postdata={"sft_document_weight": (0., 0.)})
        result = independent_sft(PolicyOutputs(logprobs=scores[:2]), all_zero)
        self.assertEqual(float(result.loss), 0.)
        self.assertFalse(result.loss.requires_grad)

    def test_normal_trainer_preserves_real_components_and_ids_after_cohort_reordering(self):
        from dataclasses import replace
        import tempfile
        from types import SimpleNamespace
        from test_routed_bank import SITE, TinyModel
        from rlstack import LocalStore, Message, Role, Task, Trajectory, Turn, lora
        from rlstack.data.trajectory import trajectory_to_row
        from rlstack.policy.siteschema import SiteSchema
        from rlstack.runner.checkpointing import Checkpointing
        from rlstack.runner.learners.torch_learner import TorchLearner
        from rlstack.runner.loop import run_experiment
        from rlstack.training.supervision import SftStage, supervised_spec

        def row(name, tokens):
            prompt = Message(Role.USER, "context")
            response = Message(Role.ASSISTANT, "recorded exact tokens")
            turn = Turn(response, tokens, (0.,) * len(tokens), "stop", None, "sealed", {}, 1)
            return trajectory_to_row(Trajectory(Task(name, "", {}),
                (prompt, response), (turn,), {}))
        datasets = ((row("alpha/0", (2, 3)), row("alpha/1", (2, 3))),
                    (row("beta/0", (4, 5, 6)),), (row("gamma/0", (2, 3)),))
        with tempfile.TemporaryDirectory() as folder:
            store = LocalStore(folder)
            spec = supervised_spec("route-toy", lora("proj", 2), datasets,
                (SftStage(1, 2, 1e-4), SftStage(1, 1, 1e-4), SftStage(1, 1, 1e-4)),
                17, store.cas_put, microbatch_tokens=128)
            # Compatible ranks share a packed forward; rank3 is a separate
            # cohort, so the actual forward order is alpha,gamma then beta.
            bank = dict(spec.policy.bank)
            bank["student1"] = replace(bank["student1"], init={**bank["student1"].init, "r": 3})
            spec = replace(spec, policy=replace(spec.policy, bank=bank))
            with torch.random.fork_rng():
                torch.manual_seed(41)
                learner = TorchLearner("cpu", torch.float32, checkpoint_activations=False)
                learner._model = TinyModel().requires_grad_(False).eval()
            learner._base = "route-toy"
            learner._tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: [1])
            report = run_experiment(spec, SiteSchema("route-toy", (SITE,)), store, {}, learner,
                                    checkpointing=Checkpointing(every=1))
            handle = store.open_run(report.run_id)
            measured = handle.read_postdata_part(1, "training")
            first, second = measured["microbatches"]
            self.assertEqual(set(first["row_ids"]), {"alpha/0", "alpha/1", "gamma/0"})
            self.assertEqual(second["row_ids"], ["beta/0"])
            by_id = {}
            for micro in measured["microbatches"]:
                objective = 0.
                for index, (name, route) in enumerate(zip(micro["row_ids"], micro["routes"], strict=True)):
                    components = micro["components"]
                    numerator = components[f"document_{index}_nll_sum"]
                    count = components[f"document_{index}_tokens"]
                    weight = components[f"document_{index}_weight"]
                    self.assertGreater(numerator, 0.)
                    self.assertEqual(count, 3 if name == "beta/0" else 2)
                    self.assertEqual(weight, .5 if name.startswith("alpha/") else 1.)
                    self.assertEqual(route, {"alpha": "student/0", "beta": "student/1", "gamma": "student/2"}[name.split('/')[0]])
                    objective += weight * numerator / count
                    by_id[name] = (numerator, count, weight)
                self.assertAlmostEqual(micro["loss"], objective, places=6)
            self.assertEqual(set(by_id), {"alpha/0", "alpha/1", "beta/0", "gamma/0"})
            self.assertEqual(handle.checkpoint_tail()["versions"],
                             {"student0": 1, "student1": 1, "student2": 1})
