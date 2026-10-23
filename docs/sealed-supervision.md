# Sealed supervision through the normal Trainer

Supervised data is a file of complete, sealed trajectory rows. A `Replay` leaf
selects a row; a wave selects one optimizer update. Repeating a dataset for
several epochs means several ordinary waves. The normal Trainer accumulates,
steps, commits and checkpoints them. It is the only optimizer-driving runner.

`runner/fit.py`, `roles/fitter.py`, `FitJob`, `FitClient`, `Plans.fit` and the
`fits` postprocessor declaration have been removed. Their old execution path is
not retained behind another class name. Existing historical manifests and
artifacts remain untouched and the observer can still read them. Submitting an
old fit plan is refused; compile a new Replay curriculum instead. Its new
identity records the changed execution and commit semantics.

## Compile a curriculum

```python
from rlstack import lora, run_experiment, Checkpointing
from rlstack.training.supervision import SftStage, supervised_spec

spec = supervised_spec(
    base, lora("layers.0-23.self_attn.q_proj", 8),
    datasets=(first_student_rows, second_student_rows),
    stages=SftStage(epochs=3, batch=4, lr=1e-4),
    seed=17, put=store.cas_put,
)
report = run_experiment(spec, schema, store, {}, learner,
                        checkpointing=Checkpointing(every=20))
```

This helper constructs a normal `ExperimentSpec` and `RunPlan`. It does not
perform learner operations. `SftCurriculum` and `curriculum_spec` describe
successive stages with different sealed datasets. The optimizer remains resident
across stage boundaries. Supplying materialized initial payloads uses the existing
`WarmStart` CAS codec; fresh optimizer moments are explicit. A resume restores the
normal checkpoint's adapter tensors and Adam state rather than restarting a job.

Each student is an ordinary bank entry with an explicit `AdapterSpec.route`.
Every shared update contains one group per active student. The ordinary
`sft_document_weights` postprocessor and `independent_sft` loss compute a sum of
students' document means: selected-token mean NLL within each document, then
mean over that student's documents. Each group's weight is stored per document,
and each loss stores numerator, scored-token count, weight and objective.
Changing a neighbour's batch size or splitting a microbatch does not change a
student's gradient. Learning-rate multipliers are declared for each update in
`Schedule.lr_scales`; each entry keeps independent clipping and Adam moments.
An absent route leaves its parameters, optimizer ages and adapter counters alone.

Compatible LoRA entries share the existing row-aware forward over one base
model. Other lowerings can request uniform-row batches through the shared
`supports_mixed_routes` capability. Different ranks/site configurations are packed
in separate cohorts. This does not add continuous batching across separate runs.
Current explicit bank routing is learner-only: serving selectors and adapters
providing tenant-wide loss tensors are refused until their route semantics are
implemented. This restriction does not select or restrict a weight-transfer
backend; all supported serving adapter types still use the universal publication
service from ADR 0027.

## Compose training from a processor

A processor declares `training = True` and receives `client.training`. Its
`run(child_spec, checkpoint_every=...)` composes an ordinary supervised child run
on the learner already owned by its parent. It uses normal setup, a normal
Trainer, a normal ledger and explicit checkpoints. It does not construct a
second scheduler or optimizer loop. Child run references are journaled and
stored under the parent's experiment folder. The child spec also records the exact
parent reference, so identical curricula from concurrent parents have separate
run identities and resident tenants. Direct local Torch operations are serialized
at the learner boundary, including installation, forward work and teardown. Completed child tenants are removed
from the resident learner; state remains in their checkpoints. Cancellation
waits for child work to finish or drain before removing its tenant.

`training/supervision.py:supervised_run` is the asynchronous compiler convenience.
`client.training.snapshot(run_ref)` reads its checkpoint; `publish` gives a
completed adapter a write-once name with exact provenance. Recipe code remains
responsible for which student is trained, which parent initializes it, what it
is called and what scientific effect its probes mean. Named sets needed by
existing Dreamer library consumers remain adapter-owned loading operations;
they do not contain a fitting execution loop.

For measurement, `evaluate(adapter, payload, rows)` scores per-document NLL.
`gradient_cosines(..., adapter=...)` measures exact `GradientRow` token IDs and
masks at an explicitly declared ordinary adapter. Its temporary tenant is isolated
from the parent policy and performs no optimizer step. Repeated measurements of
an unchanged initialization retain the bounded source gradient cache; changing
the adapter or initialization rebuilds that tenant and invalidates its cache.

## Probes and recovery

`ProbeSpec(update=0, rows=uri)` scores before the first update; later indices score
after that update. The curriculum compiler places probes at stage boundaries.
Complete vectors, stable row IDs, routes, selected-token counts, NLL numerators
and exact versions are persisted in the update's `probe` postdata part. No probe
means unmeasured, rather than zero. A child client reads them with
`probes(run_ref, update)`. Operational timestamps and phase timings stay in host
journals, outside deterministic run bytes.

Checkpoint cadence is common to every adapter in a run. Every update still
commits a ledger line. Completion and a deliberate drained stop checkpoint;
a crash above the checkpoint tail repeats the usual bounded updates. Names are
published from completed checkpoints, with unchanged write-once semantics.
Existing ordinary specs with optional routes, multipliers and probes unset keep
their old canonical identity. Historical fit runs are not silently converted.

CPU/fake tests establish orchestration, numerical gradients, optimizer isolation
and recovery. They do not establish live NCCL speed, vLLM loading, custom Triton
parity or multi-GPU behavior. Those hardware drills remain pending; this rewrite
does not launch or deploy experiments.


## Disposable evaluation without weight archives

Forks used only to measure a training effect can choose
`client.training.run(spec, checkpoint_every=..., save_state=False)`, which selects
the common `Checkpointing(state="recompute")` policy. It applies to every adapter
in that run. The same normal Trainer writes complete ledgers and probes but
exports no initial, intermediate or final weight/optimizer archives. This avoids
introducing gigabytes of checkpoint traffic merely to obtain a probe vector.
An interrupted run starts again from its declared sealed initialization; a
completed run reuses its retained observations. Snapshot reads, named weight
outputs, generation and serving are refused in this explicit mode. The manifest
records that recovery choice, and a read-only progress query distinguishes it
from a saved checkpoint. Ordinary default checkpointing remains unchanged.

`AdapterSpec.out` names a final durable adapter export. The final checkpoint
writes its snapshot blobs and write-once name/provenance before announcing
completion, so a dependent cannot mistake a completed producer for an orphaned
name. `WarmStart("named://experiment/folder#adapter/name", map={"adapter": "pi"})`
starts a normal run from that exact named payload with fresh Adam moments.
Waiting for a promised parent does not hold GPU admission. On checkpoint resume,
restored state takes precedence over consulting the named source again.

Choose saved-state or recompute recovery before starting a run. Reusing an existing
run directory with a different recovery mode is rejected; start a separate run
location instead. The recovery mode is explicit in its manifest.
