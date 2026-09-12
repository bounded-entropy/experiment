# ADR 0009 — Learned task bases use recorded posterior draws

| | |
|---|---|
| **Date** | 2026-09-10 |
| **Status** | Accepted |
| **Author** | Codex, with Samarth |
| **Touches** | policy/adapters/, training/losses/, tests/ |
| **Invariants** | Preserves I2–I6 and I8–I12; uses existing trainer-only adapter support |
| **CONTEXT** | Entry 90; reusable adapter implementation |

## Original prompt

> ok sure, let's try the approach where the decoder directions can be learned first (your first approach)

> i think 3 and 4 sound most promising. i think the way we should do this is similar to how we did the book experiment, where we train the decoder (and push per-task latents as much as possible to N(0, I)), and then train the per-task latent for the new relationships or facts or whatever while freezing the decoder. do you agree?

> also we should definitely not just use the mean right? we should legitimately sample from the distribution. we can use mean to act as a baseline, but our training should not ordinarily involve that

## Context / problem

`policy/adapters/spectral_tasks_torch.py` routes independent task posteriors
through fixed singular directions. The learned-task adapter makes those
shared directions trainable while preserving the existing replay, optimizer,
checkpoint and LoRA materialization interfaces.

## Decision

Add a trainer-side `learned_tasks` AdapterType. At each selected weight matrix, learn column-normalized U and V and a full core C(z), giving delta W = U C(z) V transpose. A shared SiLU trunk maps z to hidden features, and each site has a linear core head. All shared tensors are checkpointed and frozen when a new task scope is loaded. Only the new task's Gaussian mean and log standard deviation train. Use existing replay facts and the pure `factual_sft` loss. Rank, latent size and hidden size are caller-supplied. Core heads start at zero and the prior is fixed N(0,I).

### Touched / untouched

- **Touched:** new `policy/adapters/learned_tasks{,_torch}.py`, registered in the existing package; the existing supervised loss surface and focused adapter tests.
- **Untouched:** existing spectral adapter math and historical records, desk scheduling and custody, optimizer protocol, store layout, observer mutation rules, base-model weights. Each already has the required contract.
- Use shared seal, replay, loss, warm-start, and client helpers. Deployment contains images and chassis wiring only.

### Promises / non-promises

- Sampled training in both stages; one recorded draw per sequence across every token and site. Noise is part of immutable CAS data and survives replay/resume.
- Learned directions are included in every checkpoint; new scopes restore those directions and reset only their posterior bank.
- Same-scope load is byte-round-trippable. Optimizer moments and committed update ownership continue through the existing runner. A crash loses only uncommitted work; no new writer or recovery race is introduced.
- Additive wrappers preserve other tenants. Frozen-decoder adaptation cannot update U, V, trunk, or heads.
- Evaluation through the actual model is required. Strong fitting, useful generalization, GPU speedups, and linearization accuracy are empirical questions, not implementation promises.
- Ordinary LoRA factor materialization supports later generation parity checks. This ADR does not claim a new native serving kernel or a successful live parity drill.

### Interfaces

Existing AdapterType params/install_replay/uninstall_replay/provide/param_groups/emit/load; ReplayRows and SiteWrapper; existing `spectral_task` and `slatent_eps` sealed facts; RunPlan and WarmStart; standing desk submission and observer manifest/ledger discovery. Experiment builders supply their own sealed data and plans.

### Sketches

```python
build(sites, init) -> LearnedTaskState
task_latents(state, rows) -> Tensor
materialize(state, z) -> LoraState
```

## Questions and decisions already resolved in conversation

1. **Learn directions or retain the base model's singular vectors?** Resolved by Samarth's first quoted selection: learn U/V and the shared core map. Fixed spectral adapters remain historical controls.
2. **Freeze the decoder for a new task or train it again?** Resolved by the second quote: freeze every shared component and learn a fresh posterior. Shared and adaptation KL coefficients are separate sweep axes.
3. **Sample or train on means?** Resolved by the third quote: genuine samples are ordinary training; means are a labeled evaluation baseline. Recorded noise and the existing commit/resume path implement that decision without a new random-state protocol.
## Outcome

The framework contains both replay implementations and tests for task routing,
posterior gradients, frozen shared tensors, checkpoint round trips, tenant
isolation, and sampled-adapter materialization as ordinary LoRA. Study data,
plan builders, launch wiring, and scientific results are maintained separately.
Status remains Accepted: publishing this implementation does not resolve the
outstanding empirical generalization or native-serving validation questions.
