# Adapter checkpointing and live weight publication

The ordinary Trainer handles replay SFT and RL. Checkpoint cadence applies to
all its adapters together. It is independent of the live publication backend:
`WeightTransfer("base64")` or `WeightTransfer("nccl")`. Both are submission
resources, carried alongside the spec; they do not enter scientific run identity.

## What happens on an update

- Accumulate gradients and step the appropriate adapters while their state and
  optimizer remain on the learner.
- If a serving pool consumes the policy, publish its new immutable version and
  await installation before the ledger makes that version visible.
- Append the update's metrics and versions to the ledger.
- At the selected cadence, at completion, and on a drained stop, save adapter
  and optimizer state together and append the durable checkpoint record.

No optimizer serialization is needed to publish weights. Replay SFT with no
serving pool needs no intermediate serving export either. It still records
each update and its losses. It requests a content fingerprint from the resident
so historical checkpoint scoring remains verifiable; computing that fingerprint
still reads tensor bytes on the host, but does not ship weight archives through
the runner. Its hashing time is recorded separately. A crash resumes at the last complete checkpoint,
not merely the last ledger entry. Initialization still writes checkpoint zero;
checkpoint I/O can carry serialized bytes even when the live backend is NCCL.
Disposable learner-only measurements can explicitly choose recompute recovery
to retain observations without weight archives; see the supervision guide.

Sealed supervision uses that same Trainer. `Plans.fit`, its separate runner,
job records and optimizer-driving client have been removed (ADR 0029).
[Curriculum compilation](sealed-supervision.md) creates ordinary Replay waves
and optionally routes independent bank entries through shared forwards.

## Shared representation and two backends

Each compute-side adapter state exposes `export_state(version)`: named tensors
plus its existing optional metadata/header. Explicit versions keep probabilistic
ensemble draws independent of how often a checkpoint is taken. This conversion
does not select a network backend. All current built-in adapter states implement
the contract; an extension supplies the same contract on its state object.
State objects inherit the default `AdapterState.bind_version` (a no-op) or
implement it when a saved counter has an offset from the run's logical version.
Initialization binds those counters after warm-start/restore. Thus a child
starting at update zero from a parent's later checkpoint retains the same
ensemble sequence under either backend and any checkpoint cadence.

Base64 publication encodes that state using the existing payload formats and
RPC/blob path. NCCL publication packs immutable tensors into one CUDA buffer per
dtype. Control RPCs carry only names, shapes, dtypes, content hashes, version
identifiers and rendezvous facts. The sender reuses those buffers for every
destination replica; tensor elements never become Python lists or base64.

Peer process groups are distinct from the learner's dp/fsdp groups. They persist
between successful updates and are bounded and released with resident lifetime.
A failed send is never automatically replayed. A receiver verifies the state
and acknowledges the installed bundle; repeated acknowledgement reads are
idempotent. Failure prevents the caller from committing the new version and
discards partial transfer state without replacing an older policy.

NCCL carriage is independent of adapter type. Receiving an adapter is distinct
from installing it: `RolloutLowering.attach_state` can use a native tensor loader,
or convert locally to the existing file consumer. The default compatibility
implementation uses the latter. The current built-ins therefore still have
installation work, including host copies and files where their serving API
requires them. Tensor content hashes also require host reads in this version,
batched once per dtype. Neither cost is advertised as zero-copy or hidden in a
claimed NCCL speedup.

## Setup

Use `Checkpointing(every=k, delivery="wire")` and
`WeightTransfer(backend="nccl", timeout_s=60)` on the shared submission API.
The same option reaches initial policy publication and subsequent updates.
Base64 is the compatibility default. `delivery="store"` retains its older
meaning, requires `every=1`, and cannot be combined with NCCL publication.

NCCL requires compatible CUDA/NCCL endpoints with dedicated awake residents.
Alternating engine/learner hosts are explicitly refused for this backend;
they can use base64. No unsupported allocation silently changes the backend.
The inference frontend currently owns its receive buffers, so account for
staging/context memory alongside the vLLM workers when sizing a host.

The default rendezvous address is loopback for residents sharing a network
namespace. For different machines/containers, set each inference build's
`EngineBuild.weight_transfer_host` to its address reachable from the learner,
and ensure the necessary NCCL/TCP network path is available. The run-level
`WeightTransfer.rendezvous_host` is the fallback when the build supplies no
address. Provider gateways and SSH control relays are not NCCL data relays.
Timeouts remain finite and resident idle release remains enabled.

This option governs live learner-to-serving policy publication. Durable
checkpoint access and named-library reads retain their existing storage APIs.

## Content IDs and recovery

Existing byte bundles and checkpoint payload codecs remain readable. Tensor
publications use a distinct `bundle:tensor:` content-id namespace, covering the
version map, tensor layouts/digests and adapter metadata. A checkpoint can
reconstruct that manifest from its original payload bytes and prove the same
tensor bundle id, without rewriting the checkpoint. Historical run directories
are not migrated or edited. The existing fake-run identity regression remains
unchanged.

Async requests pin old immutable policy state while a new one installs. Shared
residency operations serialize pin/eviction against receiver-thread publication;
a version evicted before it can be pinned is refused, never served as the base.
Losing an uncheckpointed serving version still follows `BundleUnavailable` and
the existing checkpoint recovery path.

## Measurements and qualification

Operational host journals record state capture/materialization, hashing,
connection setup, transfer, verification, installation, checkpoint export and
checkpoint writes where measured. Backend, run/update/version, source token and
destination channel identify the work. Base64's transfer/install bucket remains
combined; absent measurements are `None`, rather than invented zeros.

Tests cover the generic tensor protocol over real CPU Gloo groups (including
multiple dtypes and non-core adapter names), persistent connections, immutable
snapshots, corrupted content, replica acknowledgements, failure cleanup,
checkpoint reconstruction, explicit ensemble versions and export cadence.
They fail if tensor carriage calls base64 or checkpoint serialization.

The two-CUDA-device NCCL smoke test is skipped without suitable hardware. CPU
passes do not qualify NCCL, inference-worker installation, multi-node networking,
or dp/fsdp publication on real GPUs. No live GPU experiment or throughput
benchmark has been run for this change. Keep that qualification and representative
end-to-end timing explicit before relying on this backend for experiments.
