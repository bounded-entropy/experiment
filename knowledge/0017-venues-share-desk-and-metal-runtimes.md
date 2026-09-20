# ADR 0017 — Providers share Desk and Metal runtimes

<!-- Numbered 0013 in the codex/strangeloop-backend worktree; renumbered on publication (ADR 0014, Part E). -->

| | |
|---|---|
| **Date** | 2026-09-13 |
| **Status** | Implemented in isolated worktree; local validation complete, live deployment pending |
| **Touches** | `runner/venues/`, shared venue client helpers, Modal and Strange Loop deployment wiring, lifecycle and import-boundary tests |
| **Invariants** | Existing scientific identity, Store bytes, placement, tenant admission, finite idle release and stopped-owner recovery |

## Original prompt

> ok clearly this is very disorganized. like theres basically no parity between where modal is implemented in the codebase and where strangeloop's functions are implemented in this codebase. we need good abstractions

Samarth requests a shared abstraction and implementation parity after the initial file relocation. This authorizes refactoring both adapters to consume the shared lifecycle. The interfaces and package names below are implementation choices, not additional quotations or approvals attributed to Samarth.

## Context

Moving Strange Loop modules into `runner/venues/` grouped files but did not remove the separate lifecycle implementations. Strange Loop uses `DeskRuntime` and `MetalRuntime`; Modal's `deploy/desk.py` and `deploy/modal_venue.py` still implement their own idle clock, registration, heartbeat, host supervision and shutdown. Changes to those rules therefore need two implementations, even though both use the same core Desk and MetalService.

## Decision

Both providers instantiate the same `DeskRuntime` and `MetalRuntime`. The runtimes own service routing, registration, heartbeats, host supervision, finite idle release and shutdown. The core Desk retains placement and recovery semantics. A typed `AllocationProvider` supplies the existing `boot(name)` and `terminate(allocation_id)` operations; successful termination means confirmed completion for that exact allocation, not acceptance of a stop request.

Each provider has `provider.py`, `desk.py` and `worker.py` under `runner/venues/<provider>/`. Provider modules hold the platform API and addressing. Desk and worker modules adapt the platform's process startup and transport to the shared runtimes. Modal uses its decorators and class methods; Strange Loop uses HTTP and SSH. The module names express the same responsibilities without requiring identical platform APIs.

Deployment files retain app/image/volume/card declarations, operator commands and thin factory calls. Existing Modal deployment paths, class names, serialization module identities, image settings, named volumes and scientific spec builders remain valid. Provider-independent submission, readiness, guarded release and observer polling helpers live in the shared venue client module and take an explicit Desk handle or observer address. W&B remains a separate exporter; scratch and Modal volume behavior remain behind Store; HTTP and Modal RPC remain behind Transport.

### Boundaries

- **Shared** — Desk/Metal construction and lifecycle, admission and recovery rules, submission shaping, registration and readiness facts, and observer progress reads.
- **Provider-specific** — acquiring and terminating compute, translating addresses, packaging sources and images, obtaining credentials/mounts, opening SSH routes, and adapting Modal decorators or an HTTP process entrypoint.
- **No scientific change** — no provider fields enter canonical specs, run IDs, checkpoint/ledger bytes or model APIs. No new scheduler, provider-specific placement rule, or automatic retry of learner mutations is introduced.

### Modal lifecycle migration

The Modal Desk now uses an async enter hook to start the shared idle and recovery duties immediately. Recovery keeps its fifteen-minute cadence, but runs inside the standing DeskRuntime instead of a separate scheduled Modal function. Operator `reap` still calls the same Desk operation. The runtime keeps a record of duty failures and retries on its next tick.

The Modal worker delegates measured construction, registration, heartbeat, host statistics/watchdogs and shutdown to MetalRuntime. Registration can wait for recovery without starving heartbeats. Teardown awaits owned tasks and the shared MetalService release path. A released runtime cannot be revived by a later input; the former fallback that reused a container without `stop_fetching_inputs` is removed. Provider-confirmed physical termination remains the condition for handing over ownership.

The old worker's periodic and exit-time volume commits are removed. Durability stays at the existing Store commit points: ModalVolumeStore already commits ledger seals and host/fleet/measurement journals, and StrangeLoopLocalStore publishes each mutation. This does not promise preservation of unsealed partial work. Store implementations and their bytes are unchanged by this refactor.

### Validation

Exercise both adapters against the same runtime and fake fleet behavior, including registration concurrent with slow recovery, host/service routing, idle release, shutdown and exact provider termination identities. Keep the existing canonical venue-spec fixtures, architecture/import boundaries and full local suite. Deployment import tests use the existing Modal stand-in; no image builds, GPUs or deployments are part of this refactor. Live validation remains required for both changed Modal startup wiring and the Strange Loop image/SSH/storage path.

## Outcome

Both provider folders now have `provider.py`, `desk.py` and `worker.py`. Shared `provider.py`, `runtime.py` and `client.py` hold the allocation contract, service lifetimes and client operations. The old Strange Loop `local.py` became `desk.py`; no compatibility stub recreates the old runner-root files. Modal deployment entrypoints retain their public wrappers and resource declarations. Context entry 91 records this migration (numbered 90 in the worktree).

Validation: `python3.13 -m unittest discover -s tests` ran 1,544 tests in 57.2 seconds with 194 dependency/GPU skips and no failures (`/tmp/rlstack-parity-final-suite.log`). Focused adapter tests exercise both real startup paths with fake platform APIs, actual shared Desk/Metal runtimes, HTTP request/response handling, exact epoch/allocation identities and shutdown. Additional client checks preserve canonical spec/plan bytes and exact resume/observer subdirectories. All nineteen existing canonical venue-spec fixtures pass. Package discovery includes both new provider packages, and both Strange Loop module entrypoints load locally. Modal deployment imports/class identities pass against the existing SDK stand-in.

The changes remain in `codex/strangeloop-backend`; no deployment, image build or GPU acquisition occurred. Staged and unstaged inherited work remain separate. Local tests do not validate Modal's real deserialization/startup hooks or Strange Loop image compatibility, reverse SSH connectivity, scratch publication across abrupt failure and provider termination timing. Strange Loop automatic recovery remains disabled by default pending that live drill; the shared recovery implementation is present. Native scratch append, conditional writes and publication operation status remain platform dependencies, as recorded in ADR 0015.

Subsequent live pilot, 2026-09-13: the custom Python 3.12 image built successfully and an A100-80GB lease became SSH-accessible. Worker startup exposed the runner's `AllowTcpForwarding local` restriction. The provider now validates and reloads the leased container's SSH configuration to permit the configured Desk loopback port only, preserving public-listening and authentication restrictions. A real GPU-to-laptop HTTP round trip passed. This currently depends on the provider-owned SSH configuration path and requires root SSH; native reverse-forward support remains preferable. The failed-start lease was explicitly released and the provider reported `released` before replacement. These observations do not establish training or crash-recovery parity.
