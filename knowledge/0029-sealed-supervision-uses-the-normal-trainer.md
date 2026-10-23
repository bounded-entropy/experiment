# ADR 0029 — Sealed supervision uses the normal Trainer

| | |
|---|---|
| **Date** | 2026-10-23 |
| **Status** | Accepted |
| **Author** | Codex integration subagent |
| **Touches** | `spec/`, `runner/`, `policy/adapters/`, `training/post/`, recipes in the private experiment repository |
| **Invariants** | I1, I2, I3, I5, I8, I9, I10, I11 |
| **CONTEXT** | Extends ADRs 0019, 0027 and 0028; supersedes ADR 0019's separate fitting executor |

## Original prompt

> for "fitting" the intended path is to create the SFT dataset such that the things that youre SFTing on are already sealed trajectories. i think this logic existed before the dreamer existed, so why would this have happened?

> i think a better thing to do wouldve just been to just use my multi-LoRA but dont require checkpointing every step?

> we need to rewrite things in a way such that:
>
> - for all adapters, theres some checkpoint frequency that we can choose
> - for weight transfer, we can either use NCCL or base64 sending. this is universal across all adapters, not specific to any kind of adapter
>
> the weight transfer capabilities should not be particular to a certain type of adapter.

> we should not serialize with the nccl backend into base64 before transferring, since my understanding is that this would be very slow in python. we should just do whatever is the fastest. also, yes, we can have the same checkpoint frequency for all the adpaters in a run.

> did you remove the old code like the fitter class and fitjob and everything?

> could you do the rewrite, and also merge all changes that [@Inspect training text and entropy RL](thread://01a165b8-d581-72a1-9856-61e10a64bb19?hostId=local) made? assign this to an async subagent, as I want to unblock this channel

## Context / problem

`runner/fit.py` owns an optimizer loop alongside `roles/trainer.py`; `roles/fitter.py` consumes a special `Plans.fit` whose ledger counts completed jobs rather than updates. `FitClient` installs Dreamer memory lanes and runs the same second loop inside postprocessing. Their speed advantage came partly from avoiding checkpoints at every inner update. ADR 0027 has removed that reason for a separate training executor, but explicitly left these callers unmigrated. Sealed `Replay` supervision and row-aware multi-LoRA already exist. The missing connection is letting ordinary configured bank entries select separate row routes in one normal Trainer forward, rather than making a Dreamer bank the owner of that scheduling facility.

## Decision

Delete the separate fitting plan, runner, job and client execution loop. Build supervised curricula as ordinary sealed `Replay` waves and execute them through the normal Trainer. Optional explicit adapter routes let independently parameterized bank entries share a forward at the same site; routing is universal and is separate from the adapter's own mathematical definition. The normal schedule can declare per-update learning-rate multipliers. Routed entries keep independent optimizer moments and clipping; a route absent from an update has no gradient and does not advance Adam or adapter update counters. Sequence supervision normalizes each independently trained route by its own document count across all microbatches.

Postprocessors that train or measure declare access to a generic training client. That client composes ordinary child experiment specs and sealed plans on the already-owned learner using `run_experiment_async`; it does not own an optimizer loop. Dataset construction, epoch shuffles, stage boundaries, named outputs and task-specific fork interpretation belong in recipe helpers. Child runs have content identities, normal ledgers, checkpoint/resume and explicit cadence. A generic measurement door can inspect resident child policies and compute gradient comparisons without training them. There is no type check for `dream_bank` to permit supervised training or measurement.

### Touched / untouched

- **Touched** — `spec/specs.py`, validation and wire records: remove `Plans.fit`; add optional row routes and explicit per-update schedule multipliers without changing old unconfigured identities.
- **Touched** — the normal Trainer: consume schedule multipliers, preserve independently normalized routed supervision, and expose ordinary evaluation/probe measurements without creating another executor.
- **Touched** — `runner/loop.py`, post clients and recipe callers: compose normal child runs; remove legacy fitting exports and lifecycle branches.
- **Touched** — source teaching changes: port gradient measurements/cache, calibration, transport/auth fixes, and retain experiment scripts, results and reusable datasets in the private integration branch. Merge ADR 0028's implementation directly through its ancestry.
- **Untouched** — tensor publication's raw NCCL transfer and common checkpoint cadence: ADR 0027 remains the shared service. Adapter-specific consumers can still materialize local files after transfer.
- **Untouched** — recorded historical runs and artifacts: old fit-plan manifests remain readable as historical data, but are not silently resumed through a different executor. Migrated runs get new identities.
- **Untouched** — unrelated adaptive-batch experiments; not part of this request.

### Promises / non-promises

- **Promises** — one optimizer-driving runner, sealed trajectories before supervision, ordinary ledger/checkpoint recovery, generic routing, independent Adam and normalization, migrated production callers, no old FitJob/Fitter compatibility executor.
- **Promises** — named outputs retain write-once semantics and names/probes can be reconstructed from committed normal-run outputs; a crash does not silently reset or reseed a resumed adapter.
- **Non-promises** — historical fit-run byte identity after changing its executor. Existing ordinary runs without new fields retain their identity and resume tests.
- **Non-promises** — CPU tests do not qualify NCCL, live vLLM, custom Triton, or multi-GPU execution; no GPU allocation or experiment launch is part of this rewrite.

### Interfaces

`AdapterSpec.route` selects ordinary bank entries per document. `Schedule.lr_scales` declares each update's multipliers. `TrainingClient.run(spec, checkpointing=...)` executes a full ordinary child spec; `TrainingClient.measure(...)` uses learner forward/gradient APIs. Recipe helpers translate curricula into existing `RunPlan`/`Replay` values and read child checkpoints for named export. No new plan kind or runner role is added. Existing store areas and normal checkpoint records are used.

```python
AdapterSpec(adapter_type="lora", site=site, init={"r": 8}, route="student/0")
Schedule(lr_scales=({"student0": 1.0}, {"student0": 0.5}))
await client.training.run(child_spec, checkpointing=declared_cadence)
```

## Questions and authorization

The user has explicitly answered the architectural fork in the quoted sequence: sealed supervision with multi-LoRA, one checkpoint cadence per run, and raw tensor NCCL or base64 chosen independently of adapter type. The final request explicitly authorizes finishing the rewrite after being told that `Fitter` and `FitJob` remain. This ADR records those existing answers; it does not invent new answers on the user's behalf. API spelling and the compilation of curricula into existing primitives are implementation choices within that authorization.

Races and recovery: child specs are immutable and content-addressed; concurrent same-identity submissions to a shared learner are serialized by the composition client, normal checkpoint recovery is authoritative, and parent cancellation awaits outstanding synchronous learner work before uninstalling temporary tenants. A named export is published only from a completed child checkpoint. A name already present is checked for matching provenance instead of overwritten. A run killed above its checkpoint tail replays the usual bounded updates; an incomplete stage does not turn into a completed named adapter. A drained stop checkpoints through the normal Trainer. No wall times enter deterministic run bytes.

## Outcome

Implementation in progress. Final source inventory, validation counts and hardware limitations will be recorded here before completion.
