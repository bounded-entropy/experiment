# rlstack — session handover

Samarth's personal high-throughput RL-for-LLMs harness ("the thin wrapper").
This file exists because the project was designed and built across long
Claude sessions; everything you need to continue is in the repo.

## Read before writing code

1. **STYLE.md** — binding. Eight rules; rule 8's folder tree IS the
   architecture, enforced by `tests/test_architecture.py`. Deviations are
   review findings.
2. **agent-context/CONTEXT.md** — the decision log. Chronological, numbered;
   later entries supersede earlier ones (#18–#27 cover the current shape:
   post pipeline, Store ABC, site treaty, Mechanism/reachability, adapter
   exports, engine plugins, B2/B3 on Modal, the blackboard runner). If code
   and an early log entry disagree, the code plus the latest entry win.
3. **agent-context/rl-stack-spec.md** — the original spec canon (invariants
   I1–I7, worked examples). STALE relative to the deltas logged in
   CONTEXT.md; trust it for the invariants and vocabulary, not for exact
   type shapes.

## Working norms (Samarth's, stated across sessions)

- Ultra-readable code beats clever code; spec vocabulary goes in executable
  positions. Docstrings state the rule a thing enforces.
- One named function/method per rule; typed records, no meta-dict bags, no
  duck-typing (we own both sides of every interface).
- Spec-shape changes get a new numbered entry in agent-context/CONTEXT.md.
- The fakes suite must stay green: `python3 -m unittest discover -s tests`
  (Python ≥ 3.11; stdlib-only — torch/vllm/modal imports are all lazy).
  Resume-equivalence (`tests/test_resume.py`) is byte-identical run dirs —
  protect it.

## State at handover (commit 489d047 + agent-context)

- 327 tests green on fakes. Phase A + B1 done; B2 (TorchLearner, real grpo,
  LoRA compute halves) and B3 (VllmEngine) are WRITTEN BUT NEVER EXECUTED —
  the authoring environment had no PyPI/network, so real metal is untouched.
- Runner is a blackboard: daemons (generator/trainer/evaluator) synchronized
  only via the store (runner/signals.py), colocation via leases
  (runner/lease.py). `max_policy_lag` = opportunistic buffer bound B; B=0
  reproduces the old sequential runner byte-for-byte.

## Immediate task: first real run, one Modal L4

```
modal run deploy/modal_app.py::run_tests   # full suite inside the image (CPU)
modal run deploy/modal_app.py::run_arith   # Qwen3-0.6B + LoRA + GRPO on L4
```

- Expect first-contact breakage in `rlstack/runner/engines/vllm_engine.py`
  (AsyncLLMEngine kwargs, `logprobs=0` indexing, memory fractions 0.45/0.40)
  and `rlstack/runner/learners/torch_learner.py`. Fix there; the client
  library and data layer should not need to move.
- `run_arith` prints resolved vllm/torch/transformers versions — PIN them in
  `deploy/modal_app.py`'s image after the first green run (TODO(I7) marked).
- Watch `logprob_gap` in the per-update printout: ~1e-2 (bf16 noise) is
  healthy; large means trainer/sampler mismatch — the disease this stack
  exists to catch. Reward trending up on 2-digit addition = success.
- Known v0 simplifications (deliberate, logged): raw-completion prompts (no
  chat template), uniform LoRA rank per merged bundle, engine+learner
  colocated in one process, byte-identical resume is a fakes-only property
  on real metal.
