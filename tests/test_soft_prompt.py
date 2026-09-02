"""The soft prompt's replay lowering (#46) — virtual rows that leave no trace
above the boundary.

The rule this suite exists to defend is ALIGNMENT: forward_backward returns
[len(batch)] logprobs indexed by batch.token_ids, and a virtual row is a
position with no token. Every test below is one way that could go wrong — rows
that survive into the logits, a batch whose rows disagree on how many there
are, a co-tenant with no soft prompt at all — checked against the same forward
run with the rows prepended by hand.

torch ships in the deploy image, not in the client environment, so this file
SKIPS locally and RUNS on CPU under `modal run deploy/modal_app.py::run_tests`.
What needs real metal (vLLM serving the same rows through prompt_embeds) is
deploy/adapters_l4.py.
"""

from __future__ import annotations

import unittest

from rlstack.policy.siteschema import SiteMeta

try:
    import torch
except ImportError:                                  # the client environment
    torch = None

if torch is not None:
    from rlstack.data.flatten import TokenBatch
    from rlstack.policy.adapters import attn_bias_torch, lora_torch, soft_prompt_torch
    from rlstack.policy.adapters.replay import ReplayRows, row_plan
    from rlstack.runner.learners.torch_learner import TorchLearner, _doc_spans

needs_torch = unittest.skipUnless(
    torch is not None, "torch is trainer metal: this suite runs in the image")

WIDTH = 8
HEADS = 2
BOUNDARY = (SiteMeta(name="prompt[:3]", path="model.embed_tokens",
                     has_weight=False, shape=None, is_boundary=True),)
RECTANGLE = (SiteMeta(name="queries -> prompt[:3]", path="attn_scores",
                      has_weight=False, shape=None, is_boundary=False),)
PROJ = (SiteMeta(name="block.proj", path="block.proj", has_weight=True,
                 shape=(WIDTH, WIDTH), is_boundary=False),)


class _Logits:
    def __init__(self, logits) -> None:
        self.logits = logits


class _Config:
    num_attention_heads = HEADS


class _ToyLM(torch.nn.Module if torch is not None else object):
    """A causal stand-in for the base, in the shape HF hands the learner.

    Real enough to bite: it takes EITHER input_ids or inputs_embeds, adds a
    learned position embedding (so a row prepended at the front shifts every
    token after it), attends through scaled_dot_product_attention under the
    same 4-D-mask contract transformers keeps (a 4-D float mask is passed
    straight to the attention; anything else is turned into one here), and
    carries one Linear a LoRA can wrap — so a soft-prompt tenant, a bias and a
    lora tenant can all be installed on the same object.
    """

    def __init__(self, vocab: int = 23) -> None:
        super().__init__()
        self.config = _Config()
        self.embed = torch.nn.Embedding(vocab, WIDTH)
        self.positions = torch.nn.Embedding(64, WIDTH)
        self.block = torch.nn.Module()
        self.block.proj = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        self.q = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        self.k = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        self.v = torch.nn.Linear(WIDTH, WIDTH, bias=False)
        self.head = torch.nn.Linear(WIDTH, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def as_mask(self, attention_mask, rows: int, length: int, dtype):
        """A 4-D float mask goes to the attention untouched; a 2-D padding mask
        becomes the causal-and-padding one it stands for."""
        if attention_mask is not None and attention_mask.dim() == 4:
            return attention_mask
        causal = torch.ones(length, length, dtype=torch.bool).tril()[None]
        if attention_mask is not None:
            causal = causal & attention_mask.bool()[:, None, :]
        return torch.where(causal[:, None], 0.0,
                           torch.finfo(dtype).min).to(dtype)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                use_cache=None):
        h = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        h = h + self.positions(torch.arange(h.shape[1]))[None]
        h = self.block.proj(h)
        rows, length, _ = h.shape

        def heads(projection):
            return projection(h).view(rows, length, HEADS, -1).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            heads(self.q), heads(self.k), heads(self.v),
            attn_mask=self.as_mask(attention_mask, rows, length, h.dtype))
        out = out.transpose(1, 2).reshape(rows, length, WIDTH)
        return _Logits(self.head(h + out))


def a_state(n: int, seed: int, d: int = WIDTH):
    """A soft prompt with rows big enough to move the numbers visibly."""
    state = soft_prompt_torch.build(
        BOUNDARY, {"n": n, "d": d, "seed": seed, "init_std": 0.5})
    return state


def a_batch(docs):
    flat = [t for doc in docs for t in doc]
    starts, at = [], 0
    for doc in docs:
        starts.append(at)
        at += len(doc)
    return TokenBatch(token_ids=tuple(flat), loss_mask=(1,) * len(flat),
                      behavior_logprobs=(0.0,) * len(flat),
                      segment_ids=(0,) * len(flat), doc_starts=tuple(starts))


def by_hand(model, doc, state) -> torch.Tensor:
    """The reference: ONE document, its rows prepended by hand, the virtual
    positions cut off the logits, then the pre-#44 per-doc gather verbatim.

    Taken on an UNTOUCHED tree, before any install — it goes through the
    model's own __call__, so a boundary already hooked there would answer
    instead of the reference.
    """
    ids = torch.tensor(doc, dtype=torch.long)
    embeds = model.get_input_embeddings()(ids[None])
    if state is not None:
        embeds = torch.cat([state.rows[None].to(embeds.dtype), embeds], dim=1)
    logits = model(inputs_embeds=embeds).logits[0]
    if state is not None:
        logits = logits[state.n:]
    given_prefix = torch.log_softmax(logits[:-1].float(), dim=-1)
    chosen = given_prefix.gather(1, ids[1:, None])[:, 0]
    return torch.cat([torch.zeros(1), chosen])


# ---------------------------------------------------------------------------


@needs_torch
class BuildTest(unittest.TestCase):
    def test_rows_are_n_by_d_and_seeded(self) -> None:
        first = soft_prompt_torch.build(BOUNDARY, {"n": 3, "d": WIDTH, "seed": 7})
        again = soft_prompt_torch.build(BOUNDARY, {"n": 3, "d": WIDTH, "seed": 7})
        other = soft_prompt_torch.build(BOUNDARY, {"n": 3, "d": WIDTH, "seed": 8})
        self.assertEqual(tuple(first.rows.shape), (3, WIDTH))
        self.assertTrue(torch.equal(first.rows, again.rows))
        self.assertFalse(torch.equal(first.rows, other.rows))

    def test_there_is_no_identity_element(self) -> None:
        """Unlike a LoRA's B = 0, version 0 of a soft prompt is a real
        policy — the rows are born nonzero on purpose."""
        state = soft_prompt_torch.build(BOUNDARY, {"n": 4, "d": WIDTH})
        self.assertGreater(float(state.rows.abs().max()), 0.0)

    def test_one_boundary_only(self) -> None:
        two = BOUNDARY + (SiteMeta(name="prompt[:3]b", path="other",
                                   has_weight=False, shape=None,
                                   is_boundary=True),)
        with self.assertRaises(ValueError) as raised:
            soft_prompt_torch.build(two, {"n": 3, "d": WIDTH})
        self.assertIn("ONE exported boundary", str(raised.exception))


@needs_torch
class InstallTest(unittest.TestCase):
    def test_install_is_additive_and_uninstall_unhooks_last(self) -> None:
        model = _ToyLM()
        first, second = a_state(2, 1), a_state(2, 2)
        soft_prompt_torch.install(model, first)
        soft_prompt_torch.install(model, second)
        boundary = getattr(model, soft_prompt_torch.BOUNDARY)
        self.assertEqual(len(boundary.installed), 2)
        soft_prompt_torch.uninstall(model, first)
        self.assertIs(getattr(model, soft_prompt_torch.BOUNDARY), boundary)
        soft_prompt_torch.uninstall(model, second)
        self.assertFalse(hasattr(model, soft_prompt_torch.BOUNDARY))
        self.assertFalse(model._forward_pre_hooks)
        self.assertFalse(model._forward_hooks)

    def test_install_uninstall_stay_in_balance(self) -> None:
        model, state = _ToyLM(), a_state(2, 3)
        soft_prompt_torch.install(model, state)
        with self.assertRaises(RuntimeError):
            soft_prompt_torch.install(model, state)
        with self.assertRaises(RuntimeError):
            soft_prompt_torch.uninstall(model, a_state(2, 4))

    def test_a_row_that_is_not_an_embedding_is_refused(self) -> None:
        model = _ToyLM()
        with self.assertRaises(ValueError) as raised:
            soft_prompt_torch.install(model, a_state(2, 5, d=WIDTH + 1))
        self.assertIn("embedding", str(raised.exception))

    def test_an_unrouted_forward_refuses(self) -> None:
        """The boundary reads the row plan like every other lowering: an
        unrouted replay forward is a wiring bug, not a fallback."""
        model = _ToyLM()
        soft_prompt_torch.install(model, a_state(2, 6))
        with self.assertRaises(RuntimeError) as raised:
            model(input_ids=torch.zeros(1, 4, dtype=torch.long),
                  attention_mask=torch.ones(1, 4, dtype=torch.long))
        self.assertIn("no row plan", str(raised.exception))


@needs_torch
class AlignmentTest(unittest.TestCase):
    """The virtual rows must change the NUMBERS and not the SHAPE."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.learner = TorchLearner(device="cpu", dtype=torch.float32)
        self.learner._model = _ToyLM()
        self.docs = [[3, 1, 4, 1, 5], [9, 2], [6, 5, 3, 5]]
        self.batch = a_batch(self.docs)

    def batched(self, slots, index=None):
        rows = ReplayRows(
            slots=tuple(slots),
            index=(torch.zeros(len(self.docs), dtype=torch.long)
                   if index is None else torch.tensor(index, dtype=torch.long)))
        with row_plan(self.learner._model).route(rows):
            return self.learner._batched_logprobs(self.batch,
                                                  _doc_spans(self.batch))

    def slot_of(self, state):
        return {} if state is None else {state.path: state}

    def test_the_padded_forward_stays_token_aligned(self) -> None:
        state = a_state(3, 11)
        want = torch.cat([by_hand(self.learner._model, doc, state)
                          for doc in self.docs])
        soft_prompt_torch.install(self.learner._model, state)
        got = self.batched([self.slot_of(state)])
        self.assertEqual(len(got), len(self.batch))
        self.assertTrue(torch.allclose(got, want, atol=1e-5),
                        f"max|d|={float((got - want).abs().max()):.2e}")

    def test_the_rows_actually_move_the_numbers(self) -> None:
        """A vacuous alignment would pass the test above: prove the prompt is
        in the forward at all."""
        state = a_state(3, 12)
        bare = self.batched([{}])
        soft_prompt_torch.install(self.learner._model, state)
        prompted = self.batched([self.slot_of(state)])
        self.assertGreater(float((bare - prompted).abs().max()), 0.1)

    def test_a_slot_with_no_soft_prompt_passes_through(self) -> None:
        """The boundary is transparent for a co-tenant that has none — this is
        what lets a lora tenant and a soft-prompt tenant share one learner."""
        bare = self.batched([{}])
        soft_prompt_torch.install(self.learner._model, a_state(3, 13))
        self.assertTrue(torch.equal(self.batched([{}]), bare))

    def test_rows_carry_their_own_soft_prompt(self) -> None:
        first, second = a_state(3, 21), a_state(3, 22)
        index = [0, 1, 1]
        want = torch.cat([by_hand(self.learner._model, doc,
                                  (first, second)[which])
                          for doc, which in zip(self.docs, index)])
        soft_prompt_torch.install(self.learner._model, first)
        soft_prompt_torch.install(self.learner._model, second)
        got = self.batched([self.slot_of(first), self.slot_of(second)], index)
        self.assertTrue(torch.allclose(got, want, atol=1e-5),
                        f"max|d|={float((got - want).abs().max()):.2e}")

    def test_one_forward_cannot_mix_widths(self) -> None:
        wide, narrow = a_state(3, 31), a_state(2, 32)
        soft_prompt_torch.install(self.learner._model, wide)
        soft_prompt_torch.install(self.learner._model, narrow)
        with self.assertRaises(ValueError) as raised:
            self.batched([self.slot_of(wide), self.slot_of(narrow)], [0, 1, 1])
        self.assertIn("agree on n", str(raised.exception))

    def test_only_the_routed_rows_take_gradient(self) -> None:
        routed, idle = a_state(3, 41), a_state(3, 42)
        soft_prompt_torch.install(self.learner._model, routed)
        soft_prompt_torch.install(self.learner._model, idle)
        self.batched([self.slot_of(routed)]).sum().backward()
        self.assertIsNotNone(routed.rows.grad)
        self.assertIsNone(idle.rows.grad)

    def test_a_lora_co_tenant_sees_its_own_base(self) -> None:
        """Both adapter types installed on one base: the soft-prompt tenant's forward
        must not pick up the lora tenant's delta, and vice versa."""
        model = self.learner._model
        prompt = a_state(3, 51)
        delta = lora_torch.build(PROJ, {"r": 2, "seed": 52})
        delta.b["block.proj"].data = torch.randn(WIDTH, 2)
        want = torch.cat([by_hand(model, doc, prompt) for doc in self.docs])
        bare = self.batched([{}]).detach()
        soft_prompt_torch.install(model, prompt)
        lora_torch.install(model, delta)
        prompt_only = self.batched([self.slot_of(prompt)])
        lora_only = self.batched([{"block.proj": delta}]).detach()
        self.assertTrue(torch.allclose(prompt_only, want, atol=1e-5))
        self.assertGreater(float((lora_only - bare).abs().max()), 1e-4)
        self.assertGreater(float((lora_only - prompt_only).abs().max()), 1e-4)


@needs_torch
class PayloadTest(unittest.TestCase):
    def test_emit_load_roundtrip(self) -> None:
        state, other = a_state(3, 61), a_state(3, 62)
        soft_prompt_torch.load(other, soft_prompt_torch.emit(state))
        self.assertTrue(torch.equal(state.rows, other.rows))

    def test_emit_is_byte_stable(self) -> None:
        """Identical rows compile to identical payload bytes, which is what
        makes a recompiled bundle reproduce its id on resume."""
        state = a_state(3, 63)
        self.assertEqual(soft_prompt_torch.emit(state),
                         soft_prompt_torch.emit(state))

    def test_merge_concatenates_in_entry_order(self) -> None:
        first, second = a_state(2, 71), a_state(3, 72)
        merged = soft_prompt_torch.merge_rows(
            {"b_second": soft_prompt_torch.emit(second),
             "a_first": soft_prompt_torch.emit(first)})
        self.assertEqual(tuple(merged.shape), (5, WIDTH))
        self.assertTrue(torch.equal(merged[:2], first.rows.data))
        self.assertTrue(torch.equal(merged[2:], second.rows.data))

    def test_merge_refuses_mismatched_widths(self) -> None:
        with self.assertRaises(ValueError):
            soft_prompt_torch.merge_rows(
                {"a": soft_prompt_torch.emit(a_state(2, 81)),
                 "b": soft_prompt_torch.emit(a_state(2, 82, d=WIDTH + 1))})


@needs_torch
class AttnBiasTest(unittest.TestCase):
    """attn_bias's REPLAY half (#46): the bias rides the attention mask.

    The rollout half does not exist on the pinned build, so nothing here
    claims parity — what it claims is that the trainer-side lowering is the
    arithmetic it says it is, and that version 0 is exactly the base.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.learner = TorchLearner(device="cpu", dtype=torch.float32)
        self.learner._model = _ToyLM()
        self.docs = [[3, 1, 4, 1, 5], [9, 2], [6, 5, 3, 5]]
        self.batch = a_batch(self.docs)

    def a_bias(self, seed: int, fill: float = 0.0, n: int = 3):
        site = (SiteMeta(name=f"queries -> prompt[:{n}]", path="attn_scores",
                         has_weight=False, shape=None, is_boundary=False),)
        state = attn_bias_torch.build(site, {"heads": HEADS, "seed": seed})
        if fill:
            generator = torch.Generator().manual_seed(seed)
            state.theta.data = torch.randn(HEADS, n, generator=generator) * fill
        return state

    def batched(self, slots, index=None):
        rows = ReplayRows(
            slots=tuple(slots),
            index=(torch.zeros(len(self.docs), dtype=torch.long)
                   if index is None else torch.tensor(index, dtype=torch.long)))
        with row_plan(self.learner._model).route(rows):
            return self.learner._batched_logprobs(self.batch,
                                                  _doc_spans(self.batch))

    def test_the_site_name_carries_the_rectangle_width(self) -> None:
        self.assertEqual(attn_bias_torch.prompt_width("queries -> prompt[:12]"), 12)
        with self.assertRaises(ValueError):
            attn_bias_torch.prompt_width("logits")

    def test_version_zero_is_the_base(self) -> None:
        """theta = 0 through either parameterization is a zero bias, so a
        freshly built attn_bias must reproduce the unbiased forward — the
        promise LoRA's B = 0 makes.

        Not bit-for-bit, and the reason is worth naming: a biased forward
        materializes the mask over heads ([rows, heads, L, L]) where the
        unbiased one broadcasts ([rows, 1, L, L]), and SDPA reduces the two
        shapes in a different order. The BIAS is exactly zero; the mask's
        shape is what moves the last bits.
        """
        prompt = a_state(3, 91)
        soft_prompt_torch.install(self.learner._model, prompt)
        alone = self.batched([{prompt.path: prompt}])
        for param in sorted(attn_bias_torch.PARAMETERIZATIONS):
            state = attn_bias_torch.build(RECTANGLE, {"heads": HEADS,
                                                      "param": param})
            self.assertEqual(float(state.value().abs().max()), 0.0, param)
            attn_bias_torch.install(self.learner._model, state)
            biased = self.batched([{prompt.path: prompt,
                                    state.path: state}])
            self.assertTrue(torch.allclose(alone, biased, atol=1e-5), param)

    def test_a_learned_bias_moves_the_numbers(self) -> None:
        prompt, bias = a_state(3, 92), self.a_bias(93, fill=2.0)
        soft_prompt_torch.install(self.learner._model, prompt)
        attn_bias_torch.install(self.learner._model, bias)
        alone = self.batched([{prompt.path: prompt}])
        biased = self.batched([{prompt.path: prompt, bias.path: bias}])
        self.assertGreater(float((alone - biased).abs().max()), 1e-3)

    def test_gradient_reaches_theta(self) -> None:
        prompt, bias = a_state(3, 94), self.a_bias(95, fill=1.0)
        soft_prompt_torch.install(self.learner._model, prompt)
        attn_bias_torch.install(self.learner._model, bias)
        self.batched([{prompt.path: prompt, bias.path: bias}]).sum().backward()
        self.assertIsNotNone(bias.theta.grad)
        self.assertGreater(float(bias.theta.grad.abs().sum()), 0.0)

    def test_the_prompt_rows_are_not_biased_against_themselves(self) -> None:
        """The prefix must stay a pure function of the rows — that is what
        lets an engine precompute its K/V once per bundle (#25)."""
        bias = self.a_bias(96, fill=1.0)
        attention = torch.ones(2, 7, dtype=torch.long)
        mask = attn_bias_torch.additive_mask(
            bias.value()[None].expand(2, HEADS, 3), attention, 3, torch.float32)
        self.assertEqual(tuple(mask.shape), (2, HEADS, 7, 7))
        prefix = mask[:, :, :3, :3]
        causal = torch.where(
            torch.ones(3, 3, dtype=torch.bool).tril(), 0.0,
            torch.finfo(torch.float32).min)
        self.assertTrue(torch.equal(prefix, causal[None, None].expand_as(prefix)))
        self.assertTrue(torch.allclose(mask[:, :, 3:, :3],
                                       bias.value()[None, :, None, :].expand(
                                           2, HEADS, 4, 3)))

    def test_rows_carry_their_own_bias(self) -> None:
        first, second = self.a_bias(97, fill=1.0), self.a_bias(98, fill=1.0)
        rows = ReplayRows(slots=({first.path: first}, {second.path: second}),
                          index=torch.tensor([0, 1, 1], dtype=torch.long))
        got = attn_bias_torch.routed_bias(rows)
        self.assertEqual(tuple(got.shape), (3, HEADS, 3))
        self.assertTrue(torch.equal(got[0], first.value()))
        self.assertTrue(torch.equal(got[1], second.value()))

    def test_slots_may_not_disagree_about_having_a_bias(self) -> None:
        only = self.a_bias(99, fill=1.0)
        rows = ReplayRows(slots=({only.path: only}, {}),
                          index=torch.tensor([0, 1, 1], dtype=torch.long))
        with self.assertRaises(ValueError):
            attn_bias_torch.routed_bias(rows)

    def test_the_rectangle_must_match_the_prompt(self) -> None:
        bias = self.a_bias(100, fill=1.0, n=3)
        with self.assertRaises(ValueError) as raised:
            attn_bias_torch.additive_mask(
                bias.value()[None], torch.ones(1, 9, dtype=torch.long), 5,
                torch.float32)
        self.assertIn("must agree", str(raised.exception))

    def test_a_bias_that_is_not_one_per_head_is_refused(self) -> None:
        site = (SiteMeta(name="queries -> prompt[:3]", path="attn_scores",
                         has_weight=False, shape=None, is_boundary=False),)
        state = attn_bias_torch.build(site, {"heads": HEADS + 1})
        with self.assertRaises(ValueError) as raised:
            attn_bias_torch.install(self.learner._model, state)
        self.assertIn("per head", str(raised.exception))

    def test_unknown_parameterization_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            attn_bias_torch.build(RECTANGLE, {"heads": HEADS, "param": "magic"})

    def test_emit_load_roundtrip(self) -> None:
        state, other = self.a_bias(101, fill=1.0), self.a_bias(102)
        attn_bias_torch.load(other, attn_bias_torch.emit(state))
        self.assertTrue(torch.equal(state.theta, other.theta))

    def test_bounded_sigmoid_stays_inside_its_cap(self) -> None:
        site = (SiteMeta(name="queries -> prompt[:3]", path="attn_scores",
                         has_weight=False, shape=None, is_boundary=False),)
        state = attn_bias_torch.build(
            site, {"heads": HEADS, "param": "bounded_sigmoid", "cap": 2.0})
        state.theta.data = torch.tensor([[-50.0, 0.0, 50.0]] * HEADS)
        value = state.value()
        self.assertLessEqual(float(value.abs().max()), 2.0)
        self.assertGreater(float(value.max()), 1.9)
        self.assertLess(float(value.min()), -1.9)


if __name__ == "__main__":
    unittest.main()
