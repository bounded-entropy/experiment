<!-- ADR formatting: keep each prose paragraph and list item on one source line. Do not hard-wrap prose at a fixed column width. -->

# ADR 0027 — Checkpointing and weight transfer are universal adapter services

| | |
|---|---|
| **Date** | 2026-10-23 |
| **Status** | Proposed — user requirements recorded; Q1–Q2 await answers |
| **Author** | Codex with Samarth |
| **Touches** | `runner/`, `runner/roles/`, `runner/learners/`, `runner/engines/`, `runner/transports/`, `policy/`, shared submission wiring, tests and architecture documentation |
| **Invariants** | I1/I2 (policy crosses the training/inference boundary), I5 (resource choices do not change the scientific experiment), I8 (immutable version pins and tenant isolation), I10 (checkpoint and recovery ownership) |
| **CONTEXT** | Extends ADR 0002 Q3, ADR 0014, ADR 0019 and ADR 0022. Private proposals 0024–0025 and implemented private gradient-measurement ADR 0026 are not imported or accepted by this record. |

## Original prompt

> for "fitting" the intended path is to create the SFT dataset such that the things that youre SFTing on are already sealed trajectories. i think this logic existed before the dreamer existed, so why would this have happened?

> i think a better thing to do wouldve just been to just use my multi-LoRA but dont require checkpointing every step?

> we need to rewrite things in a way such that:
>
>
>
> - for all adapters, theres some checkpoint frequency that we can choose
> - for weight transfer, we can either use NCCL or base64 sending. this is universal across all adapters, not specific to any kind of adapter
>
>
>
> the weight transfer capabilities should not be particular to a certain type of adapter.

## Context / problem

This proposal starts from public-framework checkout `9ffaf5c`, in an isolated worktree. The two private experiment branches remain unchanged. Their useful runtime patches must be integrated explicitly later; this record does not claim that this checkout already contains them.

SFT on sealed trajectories already exists through `Replay`, the ordinary Trainer and an SFT loss (`data/plan.py:71`, `runner/roles/trainer.py:238`). It predates Dreamer: commit `983abb6` (2026-08-28) explicitly describes the common SFT/RL path. Per-row adapter routing also already exists (`policy/adapters/replay.py`, `policy/adapters/lora_torch.py`). Neither needs to be invented again.

`runner/checkpointing.py:26` already supplies a universal, run-wide `Checkpointing(every, delivery)` with final and drained-stop checkpoints. However, `runner/roles/trainer.py:264` calls `learner.emit` after EVERY optimizer step. `TorchLearner.emit` exports adapter payloads and serializes every entry's optimizer state. This occurs even when no checkpoint is due, including learner-only replay SFT with nothing to publish. Less frequent disk writes therefore do not remove the full export cost.

`policy/compile.py:24` represents a serving Bundle with opaque adapter payload bytes. `runner/remote.py:163` encodes those payloads as base64. This carriage is already adapter-neutral; there is no learner-to-engine NCCL publication implementation. Existing NCCL within a distributed learner is a different connection. ADR 0002 explicitly reserved a direct resident-to-resident data path beside the control RPCs.

The newer `Fitter`/`FitJob` path avoids exporting at each inner step, but ties independent fits to `dream_bank` memory slots (`spec/validate.py:40`, `runner/fit.py:633`, `runner/roles/fitter.py:93`). Its performance motivation is not evidence that generic SFT requires that adapter or a second scientific training path. Replacing it must preserve independent schedules, inactive-slot Adam behavior, normalization, initialization, names and job completion semantics; simply relabeling its slots is insufficient.

No NCCL publication speedup or universal tensor-state conversion has been measured in this work. This is a design proposal, not an implemented transport or GPU qualification.

## Decision

### Requirements settled by the user

- All trainable adapter types use one shared checkpoint policy surface. No adapter-type check, Dreamer mode, or fit-specific rule decides whether cadence is configurable.
- Learner-to-inference weight transfer offers explicitly selected `base64` and `nccl` backends. The backend is a placement/submission choice, separate from checkpoint frequency and outside scientific run identity.
- Transfer negotiation, connection ownership, data movement, completion acknowledgement and failure handling belong to shared runner/resident infrastructure. An adapter does not implement NCCL, read a transport setting, or advertise a LoRA-only fast path.
- Adapters continue to own their mathematical representation and training/inference conversion. A shared transport must carry every serving adapter's complete state, including required non-parameter state; moving only `parameters()` is not a universal solution. Trainer-only adapters still checkpoint but have no inference destination.
- Ordinary SFT continues to consume sealed trajectories through the common training machinery. Parallel fits use the existing multi-adapter compute mechanisms; a Dreamer-only execution path must not be necessary to obtain infrequent checkpoints or either transfer backend.

### Proposed execution model

Keep three operations distinct: an optimizer step, publication of a serving policy, and creation of a durable checkpoint. A step always records bounded metrics and advances the appropriate adapter versions. It keeps optimizer state on the learner. It does not serialize optimizer state unless a checkpoint is due. A learner-only SFT run with no serving consumer does not export serving weights after every step either.

When inference needs a new policy, capture one immutable serving snapshot at an update boundary. Send that snapshot with the selected common backend. The source must either hold the update boundary until transfer completes or retain an immutable copy; it must never send live tensors while the next optimizer step mutates them. Every live destination replica acknowledges installation of the exact version before a training ledger entry exposes it. Existing requests keep their old version until they finish. Releasing a transferred snapshot or communication channel cannot invalidate an in-flight policy pin.

A checkpoint captures the complete recoverable training state: every participating adapter's state and associated optimizer state at the same logical update, plus the existing checkpoint record. It occurs at the declared cadence, at completion and on a drained stop. A checkpoint is not inferred from a successful NCCL transfer, and a successful publication is not proof of durability. Restore uses the last complete checkpoint and existing ownership checks. Historic artifacts are never rewritten.

For base64, preserve the existing generic RPC route and blob offload. For NCCL, broker endpoints through the existing resident/Host control path, and move state directly between GPU-owning residents. Keep transfer process groups separate from the learner's existing dp/fsdp groups. The CPU Trainer coordinates descriptors and acknowledgements rather than relaying the weight payload. Communication has finite setup/operation deadlines and is released with resident ownership. Alternating/sleeping residents require explicit compatible staging; incompatible endpoints must be refused rather than deadlocked.

NCCL support is an endpoint/hardware capability, not an adapter-type capability. An explicit `nccl` selection is refused when the required endpoints are not compatible; it must not silently fall back to base64 or claim success after an incomplete transfer. On an uncertain acknowledgement, query/reconcile that exact version; never replay an optimizer step. On a failed channel, discard partial installation, preserve already committed versions and existing in-flight requests, and rebuild only through normal owner/recovery rules.

### Touched / untouched

- **Touched** — shared publication policy and submission codecs in `runner/`: carry the backend choice independently of `Checkpointing.delivery` (`wire` versus `store` already has a separate meaning).
- **Touched** — `runner/interfaces.py`, learners, Trainer and remote/resident dispatch: split serving-state export/publication from full checkpoint export; remove unconditional optimizer serialization from each step.
- **Touched** — shared transfer implementation in the existing `runner/transports/` region, Host/resident lifecycle and inference installation: generic endpoints, bounded transfer, version acknowledgement and immutable snapshot retention.
- **Touched** — `policy/` only where the representation selected in Q2 requires it. Adapter-specific lowering may describe/convert state; it never implements one of the network backends.
- **Touched** — fit callers/roles and their validation as they migrate to shared sealed-trajectory training and multi-adapter lifetime controls. Preserve old entry points until replacement equivalence is demonstrated; do not remove them before a working replacement exists.
- **Touched** — architecture documentation, decision log and targeted regression tests; add bounded operational measurements to the existing journals.
- **Untouched** — loss mathematics, rewards, prompts, token masks, optimizer algorithms and the two experiments' scientific settings. A change in scheduling must not silently change per-adapter loss normalization or optimizer step counters.
- **Untouched** — authoritative Store layout and existing checkpoint codecs unless a specific incompatibility is brought back as an explicit amendment. Keep operational transfer buffers out of durable experiment truth.
- **Untouched** — unresolved ADR 0023 configuration/hash migration, provider allocation policy, the observer's read-only role, historical checkpoints and running experiment resources.

### Interfaces (sketches, not implementation)

```python
Checkpointing(every=20, delivery="wire")
WeightTransfer(backend="nccl")  # alternatively "base64"

# Shared responsibilities; concrete signatures follow the Q2 representation.
capture_serving_state(tenant, version)
publish_state(snapshot, destinations, transfer)
capture_checkpoint(tenant, version)  # includes optimizer state
```

`WeightTransfer` carries no adapter-type field. `capture_serving_state` selects the serving entries by existing declarations, not by strings such as `lora` or `dream_bank`. Identical backend choice and ownership semantics apply to all entries. Store-delivery users must use a consistent checkpoint policy under the existing every-update constraint; that legacy delivery mode is not renamed to mean base64.

### Representation and compatibility

Q2 is a real performance/compatibility fork. NCCL can carry existing opaque payloads as GPU byte buffers without knowing their adapter type. That preserves their existing hashes/codecs but still pays for adapter serialization and CPU/GPU staging. Alternatively a common named-tensor plus adapter-owned metadata representation can carry serving tensors directly, with no intermediate base64 or checkpoint archive. Both backends then implement that same representation. Tensor names, shapes, dtypes, required buffers and semantic metadata are part of the snapshot; backend selection never changes which data is sent.

The tensor route requires changes to every serving lowering and explicit treatment of bundle content hashes, restored bundles, ensembles/latent draws and non-tensor metadata. It must not be called compatible with historic bundle IDs without proving the mapping. Before accepting a changed bundle representation, specify its version and legacy restoration boundary; do not silently alter run identity, resume bytes or historical directories. Checksumming can itself require synchronization or host reads, so direct tensor carriage is not a promise that the entire publication path has zero CPU work.

### Promises / non-promises and validation

- **Required validation** — replay SFT and RL with ordinary LoRA and at least one structurally different adapter; multi-adapter batches; trainer-only adapters; missing/invalid backend refusal; publication independent of checkpoint cadence; no optimizer export on non-checkpoint steps; no serving export for learner-only intermediate SFT steps.
- **Required validation** — equal installed snapshot contents under both transports, preserving adapter metadata and inference behavior. Include an adapter registered outside the core to catch transport code that silently recognizes only built-ins.
- **Required validation** — async requests pinning old versions during publication, all replicas acknowledged before visibility, partial transfer, lost acknowledgement, dead/replaced resident epochs, channel teardown, deliberate stop, crash/rewind and resume. Preserve byte-equivalence where the existing fake/store contract promises it.
- **Required validation** — migrated parallel fits match independent fits under the declared reductions/schedules, do not advance inactive adapters' Adam state, and preserve load/reset/save and named-result semantics.
- **Required checks** — relevant regression suites, architecture/import-boundary tests and `python3.13 -m unittest discover -s tests`. CPU/fake passes do not qualify NCCL, vLLM loading or dp/fsdp publication on GPUs. A supported-hardware live drill and a representative multi-adapter transfer benchmark remain explicit requirements before performance claims or enabling NCCL by default anywhere.
- **Operational evidence** — log source/target resident identity and epoch, run/update/version, backend, bytes and entry count, capture/serialization time, transfer time, installation/acknowledgement time and checkpoint time separately. Keep detailed loss and evaluation evidence at the existing scientific cadence; this refactor does not justify dropping it.
- **Non-promises** — universal serving support for trainer-only adapters, automatic hardware support for NCCL on every allocation, a particular speedup, zero-copy inference loading, or compatibility of an unanswered new snapshot representation with old bundle IDs.

## Questions

The two user requirements above are settled. These are additional choices the request does not determine.

**Q1. Does one cadence apply to all adapters in a run, or may entries checkpoint independently?**

Recommendation: keep one chosen cadence per run, applied universally to all its adapters, and allow separate runs to choose different cadences. This preserves a coherent model/optimizer checkpoint and the existing rewind rule. Independent per-entry cadences require a new recovery contract: restoring different adapters from different logical updates is not a coherent continuation of coupled training. If independent cadences are intended, design that recovery explicitly before changing the checkpoint layout.

> **Samarth:**

**Q2. Should the first NCCL implementation carry direct serving tensors, or existing serialized payload bytes?**

Recommendation: a common tensor-state representation for both backends, because faster iteration is the motivation and merely placing existing serialized archives onto a GPU retains substantial export/staging work. Proceed only after the representation and bundle/resume compatibility boundary above are made concrete. The smaller alternative is NCCL carriage of the existing opaque payload bytes: still universal and easier to keep byte-compatible, but it must be described and measured as a transport optimization, not as removal of serialization or direct parameter loading.

> **Samarth:**

## Outcome

Proposal only. No runtime code changed, no experiment launched, no GPU allocated, no implementation tests or performance qualification claimed. The worktree starts from the public framework snapshot; integration of private experiment improvements and publication are later steps.
