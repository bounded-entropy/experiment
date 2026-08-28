"""Batched replay parity on real metal: rows carrying their own delta must
equal swap-install, one document at a time.

    modal run deploy/batch_parity.py::parity      # ~4 minutes on one L4

The claim under test (#44) is a numerical one, so the reference is literally
the code that was replaced: `SwapInstallLinear` below is the pre-#44
LoraLinear body verbatim, wired the pre-#44 way (one tenant in the tree at a
time), driven by the pre-#44 forward (one document per call). Three
comparisons against it, on Qwen3-0.6B:

    base          the padded forward with no deltas — is the batching itself
                  faithful, before any adapter is in the picture
    uniform       one tenant, every row on its slot — the degenerate case the
                  whole loop runs today
    mixed         TWO tenants' deltas in ONE forward, rows split between them
                  — the mechanism, and what swap-install cannot express

Each runs at two widths, which separates the two things that could be wrong.
ONE ROW PER FORWARD isolates everything #44 added — the row plan, the per-row
gather, the padding mask argument — with no batch dimension in play, and is
therefore asserted BIT FOR BIT in both dtypes. THE FULL PADDED BATCH adds the
batch dimension itself, whose reduction order bf16 cannot reproduce; it is
compared within a tolerance, and the base row (no adapters anywhere) shows how
much of that spread was already there before this change.

The whole thing runs twice, in float32 (where the tolerance is tight enough to
be a proof) and in bfloat16 (the dtype the loop actually trains in).

Deployment only (I5): wiring and measurement, nothing semantics-bearing.
Image pins: keep in sync with deploy/modal_app.py.
"""

from __future__ import annotations

import modal

from probe import CHECKS, check

app = modal.App("rlstack-batch-parity")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("probe", "rlstack", "rlstack_engine")
)

BASE = "Qwen/Qwen3-0.6B"
PATTERN = "layers.*.self_attn.*"
RANK = 8

# four documents of deliberately different lengths: padding is where a batched
# forward goes wrong, so nothing here is rectangular
DOCS = [
    [3838, 374, 220, 17, 15, 10, 18, 19, 30, 576, 4226, 374],
    [3838, 374, 220, 20, 16, 10, 22, 30],
    [785, 6722, 315, 9625, 374, 12095, 11, 323, 279, 6722, 315, 15154, 374],
    [16, 17, 18, 19, 20],
]
ROWS = [0, 0, 1, 1]          # which tenant each document trains under


@app.function(image=image, gpu="L4", timeout=1800)
def parity() -> dict:
    import torch

    from rlstack.data.flatten import TokenBatch
    from rlstack.policy.adapters import lora_torch
    from rlstack.policy.adapters.replay import ReplayRows, row_plan
    from rlstack.policy.siteschema import hf_schema, resolve
    from rlstack.runner.learners.torch_learner import TorchLearner, _doc_spans

    class SwapInstallLinear(torch.nn.Module):
        """The pre-#44 LoraLinear, verbatim — the reference implementation."""

        def __init__(self, inner, a, b) -> None:
            super().__init__()
            self.inner, self.lora_a, self.lora_b = inner, a, b

        def forward(self, x):
            delta = (x.to(self.lora_a.dtype) @ self.lora_a.T) @ self.lora_b.T
            return self.inner(x) + delta.to(x.dtype)

    def swap_install(model, state, wrap: bool) -> None:
        """The pre-#44 install/uninstall: one tenant in the tree at a time."""
        for path in state.a:
            parent = model
            *walk, leaf = path.split(".")
            for step in walk:
                parent = getattr(parent, step)
            here = getattr(parent, leaf)
            setattr(parent, leaf,
                    SwapInstallLinear(here, state.a[path], state.b[path])
                    if wrap else here.inner)

    def per_doc_logprobs(model, docs) -> torch.Tensor:
        """The pre-#44 _doc_logprobs, verbatim, one document per forward."""
        out = []
        for doc in docs:
            ids = torch.tensor(doc, dtype=torch.long, device="cuda")
            logits = model(ids[None]).logits[0]
            given_prefix = torch.log_softmax(logits[:-1].float(), dim=-1)
            chosen = given_prefix.gather(1, ids[1:, None])[:, 0]
            out.append(torch.cat([torch.zeros(1, device="cuda"), chosen]))
        return torch.cat(out)

    def a_state(sites, seed: int):
        """A LoRA delta with a NONZERO B — build() starts at B = 0 (version 0
        IS the base), which would make every slot indistinguishable. Placed on
        the GPU here so both paths read the very same tensors (lora_torch's
        install places them; the reference wrapper never did)."""
        state = lora_torch.build(sites, {"r": RANK, "seed": seed})
        generator = torch.Generator().manual_seed(seed)
        for path in state.a:
            state.b[path].data = (
                torch.randn(*state.b[path].shape, generator=generator) / RANK)
            state.a[path].data = state.a[path].data.cuda()
            state.b[path].data = state.b[path].data.cuda()
        return state

    def a_batch(docs) -> TokenBatch:
        flat = [t for doc in docs for t in doc]
        starts, at = [], 0
        for doc in docs:
            starts.append(at)
            at += len(doc)
        return TokenBatch(token_ids=tuple(flat), loss_mask=(1,) * len(flat),
                          behavior_logprobs=(0.0,) * len(flat),
                          segment_ids=(0,) * len(flat), doc_starts=tuple(starts))

    def gap(got, want) -> float:
        return float((got.float() - want.float()).abs().max())

    # ---- one dtype's worth of evidence --------------------------------------

    def measure(dtype, tolerance: float) -> None:
        label = str(dtype).rsplit(".", 1)[-1]
        print(f"\n== {label} (tolerance {tolerance:g}) "
              + "=" * 40)
        learner = TorchLearner(dtype=dtype)
        learner._ensure_base(BASE)
        model = learner._model
        states = [a_state(sites, seed=101), a_state(sites, seed=202)]
        batch, single = a_batch(DOCS), [a_batch([doc]) for doc in DOCS]

        def batched(slots, one: int | None = None) -> torch.Tensor:
            """The new forward: `one` restricts it to a single document (the
            B=1 control), otherwise all four rows go in one padded call."""
            work = batch if one is None else single[one]
            rows = len(DOCS) if one is None else 1
            index = torch.tensor(
                [ROWS[r] if len(slots) > 1 else 0
                 for r in ([one] if one is not None else range(rows))],
                dtype=torch.long, device="cuda")
            with row_plan(model).route(ReplayRows(tuple(slots), index)):
                return learner._batched_logprobs(work, _doc_spans(work))

        def slot_of(state) -> dict:
            return {meta.path: state for meta in sites}

        # the reference, taken FIRST, on an untouched tree
        with torch.no_grad():
            want_base = per_doc_logprobs(model, DOCS)

            swap_install(model, states[0], wrap=True)
            want_uniform = per_doc_logprobs(model, DOCS)
            swap_install(model, states[0], wrap=False)

            per_row = []
            for row, doc in enumerate(DOCS):
                swap_install(model, states[ROWS[row]], wrap=True)
                per_row.append(per_doc_logprobs(model, [doc]))
                swap_install(model, states[ROWS[row]], wrap=False)
            want_mixed = torch.cat(per_row)

        # the mechanism
        with torch.no_grad():
            got_base = batched([{}])                    # nothing wrapped yet
            unpadded_base = torch.cat([batched([{}], one=d)
                                       for d in range(len(DOCS))])
            lora_torch.install(model, states[0])
            got_uniform = batched([slot_of(states[0])])
            lora_torch.install(model, states[1])        # ADDITIVE: both resident
            slots = [slot_of(states[0]), slot_of(states[1])]
            got_mixed = batched(slots)
            unpadded_mixed = torch.cat([batched(slots, one=d)
                                        for d in range(len(DOCS))])
            got_alone = batched([slot_of(states[0])])

        # ONE row, no padding: the new forward and the new lowering must
        # reproduce the old ones BIT FOR BIT, in either dtype — everything the
        # row plan added is exact, and nothing below is allowed to hide in a
        # tolerance.
        for name, got, want in (("base", unpadded_base, want_base),
                                ("mixed", unpadded_mixed, want_mixed)):
            check(f"{label} {name}: one row per forward is bit-identical",
                  torch.equal(got, want), f"max|d|={gap(got, want):.2e}")

        # ALL rows in one padded forward: what is left is the batch dimension
        # itself — reduction order over a wider GEMM, which bf16 cannot hide.
        for name, got, want in (("base", got_base, want_base),
                                ("uniform", got_uniform, want_uniform),
                                ("mixed", got_mixed, want_mixed)):
            check(f"{label} {name}: padded batch == swap-install per doc",
                  torch.allclose(got.float(), want.float(), atol=tolerance),
                  f"max|d|={gap(got, want):.2e}")

        # a vacuous parity would pass all of it: the deltas must really differ,
        # and by far more than the batching noise the comparisons allowed
        check(f"{label}: the two tenants' deltas are distinguishable",
              gap(got_alone, want_mixed) > 1.0,
              f"max|d|={gap(got_alone, want_mixed):.2e} vs base spread "
              f"{gap(want_mixed, want_base):.2e}")

        # gradients: only the slots the rows carry move (I8)
        batched(slots).sum().backward()
        routed = [any(p.grad is not None for p in state.parameters())
                  for state in states]
        check(f"{label}: every routed slot takes gradient", all(routed),
              f"{routed}")

        idle = a_state(sites, seed=303)
        lora_torch.install(model, idle)                 # resident, never routed
        batched([slot_of(states[0])]).sum().backward()
        check(f"{label}: an unrouted resident slot takes none",
              all(p.grad is None for p in idle.parameters()))

        del learner, model, states, idle
        torch.cuda.empty_cache()

    print(f"[pins] torch={torch.__version__}")
    sites = resolve(hf_schema(BASE).sites, PATTERN)
    print(f"[sites] {len(sites)} matched by {PATTERN!r}")

    measure(torch.float32, 1e-4)      # the mechanism, in exact arithmetic
    measure(torch.bfloat16, 5e-1)     # the production dtype, honestly reported

    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    print(f"\n[parity] {sum(ok for _, ok, _ in CHECKS)} passed, "
          f"{len(failed)} failed: {failed}")
    return {"passed": sum(ok for _, ok, _ in CHECKS), "failed": failed}
