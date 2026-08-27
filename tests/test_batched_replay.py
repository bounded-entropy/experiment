"""The batched, row-aware replay forward (#44) — the trainer's punica.

Every rule the mechanism enforces, checked against the thing it replaced: the
one-slot path must reproduce swap-install's expression exactly, the per-row
path must reproduce applying each row's delta on its own, the padded forward
must reproduce the document-at-a-time forward, and installing a second tenant
must not move the first tenant's numbers (I8).

torch ships in the deploy image, not in the client environment, so this file
SKIPS locally and RUNS on CPU under `modal run deploy/modal_app.py::run_tests`
— the whole mechanism is provable without a GPU. What needs real metal (bf16,
Qwen3-0.6B, a real HF forward) is deploy/batch_parity.py.
"""

from __future__ import annotations

import unittest

from rlstack.policy.siteschema import SiteMeta

try:
    import torch
except ImportError:                                  # the client environment
    torch = None

if torch is not None:
    from rlstack.policy.adapters import lora_torch
    from rlstack.policy.adapters.replay import ReplayRows, RowPlan, row_plan
    from rlstack.runner.learners.torch_learner import TorchLearner

needs_torch = unittest.skipUnless(
    torch is not None, "torch is trainer metal: this suite runs in the image")

PATH = "block.proj"
SITES = (SiteMeta(name="block.proj", path=PATH, has_weight=True,
                  shape=(8, 6), is_boundary=False),)


def a_model():
    """A one-site stand-in for the base: `block.proj` is the matched Linear."""
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.proj = torch.nn.Linear(8, 6, bias=False)
    return model


def a_state(seed: int, r: int = 4):
    """A LoRA state with a NONZERO B — build() starts at B = 0 (the delta is
    the base), which would make every slot indistinguishable."""
    state = lora_torch.build(SITES, {"r": r, "seed": seed})
    generator = torch.Generator().manual_seed(seed)
    state.b[PATH].data = torch.randn(6, r, generator=generator) / r
    return state


def swap_install_delta(x, state):
    """The pre-#44 LoraLinear body, verbatim: the number to reproduce."""
    a, b = state.a[PATH], state.b[PATH]
    return ((x.to(a.dtype) @ a.T) @ b.T).to(x.dtype)


@needs_torch
class RowPlanTest(unittest.TestCase):
    def test_a_site_refuses_to_run_unrouted(self) -> None:
        """An unrouted replay forward is a wiring bug, not a fallback to
        whoever went last."""
        model, state = a_model(), a_state(1)
        lora_torch.install(model, state)
        with self.assertRaises(RuntimeError) as raised:
            model.block.proj(torch.randn(2, 3, 8))
        self.assertIn("no row plan", str(raised.exception))

    def test_routes_do_not_nest(self) -> None:
        plan = RowPlan()
        rows = ReplayRows(slots=({},), index=torch.zeros(1, dtype=torch.long))
        with plan.route(rows):
            with self.assertRaises(RuntimeError):
                with plan.route(rows):
                    pass
        self.assertIsNone(plan._rows)          # the route always unwinds

    def test_rows_may_only_carry_installed_slots(self) -> None:
        with self.assertRaises(ValueError):
            ReplayRows(slots=({},), index=torch.tensor([0, 1]))

    def test_one_slot_is_uniform_many_are_not(self) -> None:
        one, two = {"a": 1}, {"a": 2}
        index = torch.zeros(2, dtype=torch.long)
        self.assertIs(ReplayRows(slots=(one,), index=index).uniform(), one)
        self.assertIsNone(ReplayRows(slots=(one, two), index=index).uniform())

    def test_the_plan_is_per_model(self) -> None:
        first, second = a_model(), a_model()
        self.assertIs(row_plan(first), row_plan(first))
        self.assertIsNot(row_plan(first), row_plan(second))


@needs_torch
class RowAwareLoraTest(unittest.TestCase):
    def test_one_slot_reproduces_swap_install_exactly(self) -> None:
        """The degenerate case is bit-identical, not merely close: it applies
        the same expression to the same tensors."""
        model, state = a_model(), a_state(7)
        lora_torch.install(model, state)
        x = torch.randn(3, 5, 8)
        rows = ReplayRows(slots=({PATH: state},),
                          index=torch.zeros(3, dtype=torch.long))
        with row_plan(model).route(rows):
            got = model.block.proj(x)
        want = model.block.proj.inner(x) + swap_install_delta(x, state)
        self.assertTrue(torch.equal(got, want))

    def test_each_row_gets_its_own_delta(self) -> None:
        """The mechanism: two row-groups, two adapter states, ONE forward."""
        model = a_model()
        first, second = a_state(11), a_state(12)
        lora_torch.install(model, first)
        lora_torch.install(model, second)
        x = torch.randn(4, 5, 8)
        index = torch.tensor([0, 1, 1, 0])
        rows = ReplayRows(slots=({PATH: first}, {PATH: second}), index=index)
        with row_plan(model).route(rows):
            got = model.block.proj(x)
        want = torch.stack([
            model.block.proj.inner(x[row])
            + swap_install_delta(x[row], (first, second)[int(index[row])])
            for row in range(4)])
        self.assertTrue(torch.allclose(got, want, atol=1e-6))

    def test_per_row_deltas_need_the_plans_rows(self) -> None:
        model = a_model()
        first, second = a_state(11), a_state(12)
        lora_torch.install(model, first)
        lora_torch.install(model, second)
        rows = ReplayRows(slots=({PATH: first}, {PATH: second}),
                          index=torch.tensor([0, 1]))
        with row_plan(model).route(rows):
            with self.assertRaises(ValueError) as raised:
                model.block.proj(torch.randn(5, 8))    # not [rows, tokens, in]
        self.assertIn("per-row deltas need", str(raised.exception))

    def test_slots_of_one_forward_must_agree_on_rank(self) -> None:
        model = a_model()
        first, second = a_state(11, r=4), a_state(12, r=2)
        lora_torch.install(model, first)
        lora_torch.install(model, second)
        rows = ReplayRows(slots=({PATH: first}, {PATH: second}),
                          index=torch.tensor([0, 1]))
        with row_plan(model).route(rows):
            with self.assertRaises(ValueError) as raised:
                model.block.proj(torch.randn(2, 5, 8))
        self.assertIn("agree on rank", str(raised.exception))

    def test_gradients_reach_only_the_routed_slot(self) -> None:
        """The tenancy invariant in the backward: a state no row carries takes
        no gradient from that forward."""
        model = a_model()
        routed, idle = a_state(21), a_state(22)
        lora_torch.install(model, routed)
        lora_torch.install(model, idle)
        rows = ReplayRows(slots=({PATH: routed},),
                          index=torch.zeros(2, dtype=torch.long))
        with row_plan(model).route(rows):
            model.block.proj(torch.randn(2, 5, 8)).sum().backward()
        self.assertIsNotNone(routed.b[PATH].grad)
        self.assertIsNone(idle.b[PATH].grad)


@needs_torch
class AdditiveInstallTest(unittest.TestCase):
    def test_a_second_tenant_joins_the_wrapper_it_finds(self) -> None:
        model = a_model()
        inner = model.block.proj
        first, second = a_state(31), a_state(32)
        lora_torch.install(model, first)
        site = model.block.proj
        lora_torch.install(model, second)
        self.assertIs(model.block.proj, site)          # wrapped once
        self.assertIs(site.inner, inner)
        self.assertEqual(len(site.installed), 2)

    def test_uninstall_unwraps_only_when_the_last_state_leaves(self) -> None:
        model = a_model()
        inner = model.block.proj
        first, second = a_state(41), a_state(42)
        lora_torch.install(model, first)
        lora_torch.install(model, second)
        lora_torch.uninstall(model, first)
        self.assertIsInstance(model.block.proj, lora_torch.LoraSite)
        lora_torch.uninstall(model, second)
        self.assertIs(model.block.proj, inner)

    def test_install_uninstall_stay_in_balance(self) -> None:
        model, state = a_model(), a_state(51)
        lora_torch.install(model, state)
        with self.assertRaises(RuntimeError):
            lora_torch.install(model, state)           # the same state twice
        with self.assertRaises(RuntimeError):
            lora_torch.uninstall(model, a_state(52))   # never installed here

    def test_another_tenants_install_does_not_move_the_numbers(self) -> None:
        """I8 on the trainer side: additive install is invisible to whoever
        was already there."""
        model, first = a_model(), a_state(61)
        lora_torch.install(model, first)
        x = torch.randn(2, 5, 8)
        rows = ReplayRows(slots=({PATH: first},),
                          index=torch.zeros(2, dtype=torch.long))
        with row_plan(model).route(rows):
            alone = model.block.proj(x)
        lora_torch.install(model, a_state(62))
        with row_plan(model).route(rows):
            shared = model.block.proj(x)
        self.assertTrue(torch.equal(alone, shared))


class _Logits:
    def __init__(self, logits) -> None:
        self.logits = logits


class _ToyLM(torch.nn.Module if torch is not None else object):
    """A causal stand-in for the base: embed, one causal attention, project.

    Real enough to bite — a position's output depends on its whole prefix, so
    padding that leaked into attention (or position ids that shifted) would
    change the answer.
    """

    def __init__(self, vocab: int = 23, width: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, width)
        self.attn = torch.nn.MultiheadAttention(width, 2, batch_first=True,
                                                bias=False)
        self.head = torch.nn.Linear(width, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        causal = torch.triu(torch.ones(h.shape[1], h.shape[1],
                                       dtype=torch.bool), diagonal=1)
        pad = None if attention_mask is None else attention_mask == 0
        out, _ = self.attn(h, h, h, attn_mask=causal,
                           key_padding_mask=pad, need_weights=False)
        return _Logits(self.head(h + out))


@needs_torch
class PaddedForwardTest(unittest.TestCase):
    """One padded forward must equal the document-at-a-time forward it
    replaced — the numbers, and the flat token order the loss reads."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.learner = TorchLearner(device="cpu", dtype=torch.float32)
        self.learner._model = _ToyLM()

    def per_doc(self, ids):
        """The pre-#44 _doc_logprobs, verbatim, one document at a time."""
        out = []
        for doc in ids:
            tokens = torch.tensor(doc, dtype=torch.long)
            logits = self.learner._model(tokens[None]).logits[0]
            given_prefix = torch.log_softmax(logits[:-1].float(), dim=-1)
            chosen = given_prefix.gather(1, tokens[1:, None])[:, 0]
            out.append(torch.cat([torch.zeros(1), chosen]))
        return torch.cat(out)

    def batched(self, ids):
        from rlstack.data.flatten import TokenBatch

        flat = [t for doc in ids for t in doc]
        starts, at = [], 0
        for doc in ids:
            starts.append(at)
            at += len(doc)
        batch = TokenBatch(token_ids=tuple(flat), loss_mask=(1,) * len(flat),
                           behavior_logprobs=(0.0,) * len(flat),
                           segment_ids=(0,) * len(flat),
                           doc_starts=tuple(starts))
        from rlstack.runner.learners.torch_learner import _doc_spans

        return self.learner._batched_logprobs(batch, _doc_spans(batch))

    def test_ragged_documents_match_the_per_doc_forward(self) -> None:
        ids = [[3, 1, 4, 1, 5], [9, 2], [6, 5, 3, 5]]
        self.assertTrue(torch.allclose(self.batched(ids), self.per_doc(ids),
                                       atol=1e-5))

    def test_equal_length_documents_match(self) -> None:
        ids = [[3, 1, 4], [9, 2, 6]]
        self.assertTrue(torch.allclose(self.batched(ids), self.per_doc(ids),
                                       atol=1e-5))

    def test_a_one_token_document_scores_zero(self) -> None:
        """Position 0 has no prefix; flatten guarantees it is never trainable."""
        got = self.batched([[7], [3, 1, 4]]).detach()
        self.assertEqual(len(got), 4)
        self.assertEqual(float(got[0]), 0.0)
        self.assertEqual(float(got[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
