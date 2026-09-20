# ADR 0015 — Strange Loop scratch is one store with mounted and API access

<!-- Numbered 0011 in the codex/strangeloop-backend worktree; renumbered on publication (ADR 0014, Part E) because 0011 is `training-commits-inference-loads` on main. -->

| | |
|---|---|
| **Date** | 2026-09-12 |
| **Status** | Implemented locally; live Desk/Store drill pending |
| **Author** | Codex with Samarth |
| **Touches** | `data/stores/`, store exports, observer readers, shared runner lifecycle/transport, Strange Loop wiring, deployment examples and tests |
| **Invariants** | I1 import boundaries, I3 unchanged identity, I5 placement-neutral bytes, I7 verified substrate, I10 one authoritative store, I11 existing self-description |
| **CONTEXT** | Extends entries 84–85 (ADRs 0007–0008) and 89 (ADR 0010); no implementation entry yet |

## Original prompt

> let's keep the strangeloop implementation in a separate worktree for now, so we can slowly build it up over time while strangeloop increases its support for these sorts of things.
>
> could you create that worktree and then run me through exactly how you plan on implementing the strangeloop localstore, what files you'll need to modify, etc.?

Follow-up, 2026-09-12:

> "I would initially persist each write before returning. We can introduce batching after measuring the cost and verifying the platform’s publication behavior." sure im fine with this for now
>
> "The unresolved dependency is the mounted commit operation. We haven’t established whether scratch exposes a usable SDK handle or supports Modal’s v2 `sync` mechanism. Until" find out how to deal with mounted writes please (do research)
>
> "I would initially fetch ledgers and journals through the API even when a mounted copy exists. Otherwise, a worker could keep reading an old ledger indefinitely." i want this loop to work right now, so let's just do this properly. how are reloads, etc. supported right now in strangeloop?
>
> "**4. Handle the missing append API carefully.**"
>
> ok we'll do this for now, but make an explicit note in the ADR that we need append support.
>
> the rest of your implemntation details look pretty legitimate. can you resolve the issues you mentioned and lmk the outstanding difficulties? also, can you disable the linewrap on the ADR (also make a general comment to disable linewrap in future ADRs)

## Context / problem

The branch is `codex/strangeloop-backend`, in `/Users/samarth/Coding/experiment/research/strangeloop-backend`. Its starting point is `f9256f2991803784dd3cc09b057330d7e7aab131` plus a verified copy of the source checkout's staged, unstaged, and non-ignored untracked files. The staging distinction was preserved. Those inherited changes are not Strange Loop implementation changes. Commit only specifically reviewed backend files; reconcile the inherited baseline with main before eventually merging this work. The captured patches and file hashes are in this worktree's Git metadata under `strangeloop-starting-state/`.

The following evidence was inspected on 2026-09-12:

- The installed Strange Loop skill's new `/scratch` section says Modal-backed leases mount the same account-owned volume across leases and conversations. `/scratch` is distinct from the session's periodically mirrored `artifact_dir`.
- The CLI source identifies scratch as a Modal Volume accessed by the backend's filesystem API. Initial overview/list calls returned a 512 GiB reported limit, zero files, and `artifacts`, `cache`, `checkpoints`, and `datasets` directories. The follow-up below used one short GPU lease to inspect the actual mount.
- The [public API schema](https://api.strangeloopresearch.com/openapi.json) exposes list/stat/read/upload/delete routes under `/api/v1/scratch/files`, and CPU download jobs under `/api/v1/scratch/sync`. It does not expose a mounted-worker commit/reload operation, atomic append, conditional replacement, or a documented file-write idempotency key.
- Schema snapshots are saved at `/tmp/rlstack-strangeloop-audit-2026-09-12/openapi-scratch.json` and `openapi-mounted-writes.json`. The latter still exposes no mounted reload, native append, or conditional-write route.
- `strangeloop scratch sync` fetches external data on a CPU sandbox; it is not a flush of pending writes from a GPU mount. The separate shell command `sync /scratch` is the mounted publication operation demonstrated below.
- The installed CLI's raw download path calls `resp.read()` before saving the result. A replacement client should consume bounded chunks; the current `Store._read` contract still returns a complete `bytes` object to its caller.

The code seams at the captured baseline are:

- `data/stores/base.py:169`: `StoreAddress` carries backend, root, locator. `Store` defines seven byte verbs and `_persist`; it owns layouts, serialization, explicit run references, ledger monotonicity, and recovery.
- `data/stores/local.py:20`: `LocalStore` implements filesystem operations using temporary files, replacement, and fsync. This alone does not establish remote volume durability or freshness of a mounted snapshot.
- `data/stores/modal_volume.py:42`: `ModalVolumeStore` adds commit hooks, direct committed-file reads on mounted misses, and committed-ledger hydration on resume. Its missing-file fallback does not solve stale existing mutable files.
- `data/stores/address.py:18`: residents reopen a store through `open_store`.
- `observe/locate.py:54`: the observer currently rejects remote store locators. `observe/ui.py` already accepts a `Store`; the dashboard needs no new data model.
- `observe/cache.py`: the shared observer caches immutable bytes and uses size to revalidate mutable bytes. Same-size mutable replacements need a regression test before this cache is used with the new backend.

### Live storage investigation

On 2026-09-12 local time (2026-09-13 UTC), lease `6baab082-50c6-48fb-8649-3d3707833d3b`, label `rlstack-storage-20260912t235948z`, provisioned one Modal-backed A100-80GB. This was an infrastructure diagnostic, with no model training or dataset download. All scratch writes were under the isolated prefix `artifacts/rlstack-storage-probe-20260912T235948Z`. Measurements are retained in [the evidence file](evidence/0015-scratch-storage-probe.json).

| Check | Observation | What it establishes |
|---|---|---|
| File write, flush, file fsync, atomic rename; no mount sync | First external API read returned 404 in 4/4 trials | File fsync alone did not provide immediate cross-client visibility. |
| Same sequence followed by `sync /scratch` | First external API read matched in 4/4 trials; sync returned zero and took 0.684–2.317 seconds | An operational mounted publication mechanism exists on this lease. |
| Replace an existing mounted file through the API with an equal-length value | API returned `api-seed-v2`; mount kept `api-seed-v1`, including after mount sync and several minutes | Mount sync does not provide reader refresh. Merely reopening the file is insufficient. |
| Authenticated raw PUT followed by GET from inside the pod | 3/3 replacements read back correctly, taking 1.526–2.051 seconds per PUT+GET pair | Worker-side API access works when credentials are supplied explicitly. |
| One API writer replacing a 256 KiB file while another API client reads | 8/8 write readbacks matched; all 14 overlapping reads contained one complete version; delete followed by GET returned not-found | Ordinary replacement/read/delete behavior passed; this does not prove atomicity during server failure. |
| Read after normal GPU teardown | All 10 checked files matched: eight publication files, the API-updated seed, and the final worker API write | These bytes survived normal teardown. The final mount sync also published earlier control writes. |

**Commit resolution.** [Modal's Volume documentation](https://modal.com/docs/guide/volumes#in-v2-you-can-commit-using-sync) identifies shell `sync <mountpoint>` as a persistence operation for Volume v2. The trial results strongly support using `sync /scratch` for the current Strange Loop mount. The mount appeared as a 9p filesystem through a `/scratch` symlink; neither this metadata nor Strange Loop's public API explicitly reported the underlying Volume version. Treat the measured capability as specific to the tested platform configuration, and repeat a small publication check when onboarding a different configuration. Do not infer the Volume version from the Sandbox version or from a successful shell exit alone.

**Reload resolution.** [Modal exposes `Sandbox.reload_volumes()`](https://modal.com/docs/sdk/py/latest/Sandbox#reload_volumes), but Strange Loop does not currently expose it through the inspected CLI or public API. The pod had no installed `modal` module, Modal API token, or standard Modal function-container authentication environment. A volume name does not grant access to the provider's account. Installing the SDK alone would not supply that authorization. Strange Loop could expose its provider-side sandbox reload operation later; the Store loop will instead use authoritative API reads now. A full mount reload would also need coordination with readers and open files, as described in [Modal's consistency rules](https://modal.com/docs/guide/volumes#filesystem-consistency).

**API access and runtime.** The pod did not receive `SL_API_TOKEN` automatically. Supplying the existing user's scratch API credentials through temporary worker configuration enabled the three API loops; the temporary credential files were deleted locally and remotely. The backend bootstrap must provide credentials to its resident processes without placing them in saved Store addresses, run manifests, or logs. The pod's Python was 3.10.12, whereas `pyproject.toml:5` requires Python >=3.11 and the existing concept-steer image selects 3.12. Selecting or preparing a compatible worker runtime remains separate work.

The probe demonstrates publication, stale mounted reads, API access, ordinary replacement, and normal-teardown persistence. It does not establish a provider contract for interrupted uploads, recovery after abrupt worker/server failure, ownership handoff, simultaneous mounted writers to distinct files, or checkpoint-scale throughput. The 0.684–2.317 second sync measurements are small-file latencies, not a throughput benchmark.

### Follow-up: Strange Loop 0.5.9 and the HTTP transport PR

Samarth subsequently asked:

> could you actually take a look at the PR to this repo after pulling from the latest strangeloop update? i want to see if it solved our problems

The installed CLI and skill were refreshed from 0.5.4 to 0.5.9. The latest `origin/main` and [PR #1, Add HTTP transport for persistent learner and inference services](https://github.com/bounded-entropy/experiment/pull/1), were fetched. The PR was reviewed at `9f0c52944544b0beccb81c90d2a46983d960b6c0` in `/Users/samarth/Coding/experiment/research/strangeloop-http-pr-review`; it remains open. The subsequent implementation applied its patch to this branch's preserved working files without merging the full published branch.

| Earlier gap | Current evidence | Consequence for this design |
|---|---|---|
| Persistent-service RPC | PR #1 adds `HttpTransport`, `HttpServer`, and HTTP address handling in the existing `transport_for` factory. JSON `/call` and `/ask` frames reach existing Services. | Reuse this transport for `RemotePool` and `RemoteLearner`; there is no need to invent another RPC protocol. Models and optimizer state belong to the service, not the connection. |
| SSH/port forwarding on Modal leases | CLI 0.5.9 implements `gpu forward`; the updated skill says new Modal CLI leases reserve an SSH tunnel. Both forwarded ends bind to loopback. | A local CPU orchestrator has a supported route to persistent GPU services. Old leases need replacement. Every remote caller still needs its own reachable endpoint; a laptop's localhost URL is not usable from a GPU. |
| Custom GPU environment | CLI 0.5.9 adds `image build` and `gpu up --image`; the live API exposes image routes and the account's image listing succeeds. Builds accept pip/apt packages, RUN commands, and Dockerfile fragments. | Build a separate Python >=3.11 experiment environment off the GPU, then select the image on launch. This extends Strange Loop's runner base: arbitrary `FROM` replacement is not supported. Our actual Python/CUDA/vLLM image has not yet been built or tested. |
| Mounted scratch and freshness | The latest public schema contains no changes to scratch operations. | Keep the demonstrated mount-sync writer and authoritative API reader. The PR changes no Store code. |
| Native reload, append, conditional/idempotent writes | Still absent from the inspected public storage API and CLI. | The storage limitations and temporary append design below remain. |
| Complete Desk lifecycle | The PR supplies a server wrapper around a caller-provided runtime factory. | Implement runtime creation, Desk registration, endpoint routing, finite idle release, teardown, and storage wiring through the existing chassis. |

Review validation: all 11 new HTTP tests passed using real localhost sockets. The full suite discovered 1,306 tests and finished without failures, with 191 dependency/environment skips. The PR author reports real CUDA learner/vLLM checks through SSH on Strange Loop dev releases 0.5.6 and 0.5.7. Those GPU checks were not repeated in this review, and the linked private provider PR could not be read with the current GitHub account. Production CLI 0.5.9 support was inspected locally, and the new image API was checked read-only; no image build or GPU lease was started.

At the original PR review, two transport limits mattered: requests and responses had a 64 MiB JSON frame limit, leaving less than 48 MiB for base64-encoded binary adapter/optimizer data in one frame; larger payloads need a deliberate transfer strategy. Also, a timed-out state-changing call may already have executed. The PR deliberately performs no automatic replay or deduplication, so recovery must preserve the existing ownership and checkpoint rules. The update supplies the main image and RPC mechanisms we were waiting for, while full experiment correctness still depends on the Store implementation and lifecycle integration.

## Implementation authorization

> in the background, can you spawn an async subagent to implement the strangeloop desk?

Samarth explicitly requested implementation after reviewing the storage findings and the 0.5.9 HTTP/image update. This authorizes the resolved storage design, including authoritative API reads (Q3), and thin Desk/Metal wiring to the shared chassis. Runtime ownership, finite idle release, and exact scientific bytes remain unchanged. Local tests and a reviewable implementation come first; this authorization does not itself constitute a live GPU validation.

## Decision

Use one authoritative scratch prefix, initially `rlstack/`, with two access classes in one backend module. `StrangeLoopLocalStore` writes under `/scratch/rlstack` and explicitly publishes through `sync /scratch`; its reads and metadata use the scratch API initially. `StrangeLoopStore` uses the scratch HTTP API from the local desk and observer. Both represent the same store, not independent replicas. Both inherit the existing `Store` layout and serialization. A laptop directory, if introduced for immutable caching, never becomes authoritative.

The initial proposal covered storage. Samarth subsequently requested the full Strange Loop Desk implementation. The worktree now includes the API/mounted Store, provider-neutral persistent runtimes, an explicit allocation and SSH adapter, local Desk/observer wiring, and a separate offline W&B exporter. These components use the shared chassis; local validation does not establish a working live GPU deployment.

### Behavior to implement

1. **Resolve one location.** Keep API base, expected account/volume identity, scratch-relative prefix, and local mount root explicit. The same key maps to `/scratch/rlstack/runs/<subdir>/<run_id>/...` on a worker and `rlstack/runs/<subdir>/<run_id>/...` in the API. A saved locator must identify the account/volume as well as the prefix, for example `strangeloop://sl-scratch-<user-id>/rlstack`; selecting another login must not silently select another store. Resolve credentials from the existing `SL_API_TOKEN`/profile conventions, outside manifests and saved addresses. Validate account identity once when opening the store, not on every read.

2. **Implement a small scratch HTTP client.** Use the documented file routes for read/write/list/stat/delete, with explicit deadlines and typed responses. Read responses in chunks and provide streaming download-to-file for bulk transfers. `_read` necessarily materializes its final `bytes` result; do not advertise an unlimited-memory streaming Store API we do not have. Only a definite missing-file response becomes `FileNotFoundError`; auth, timeout, quota, and server failures remain errors. Keep data-layer code independent of `runner`, and avoid a CLI subprocess for every file operation. GPU lifecycle and notebook capture continue through the normal Strange Loop integration.

3. **Use the tested API path.** Implement `StrangeLoopStore(Store)` with authoritative reads and metadata. `_write` uses raw upload, `_delete` uses the delete route, and `_size` uses stat without downloading a checkpoint. Directory browsing uses shallow listing, while ordinary exact-run lookup performs no discovery. For the initial experiment, wait for a successful write response and verify the resulting bytes before acknowledging a mutable journal replacement. Serialize each owner's writes. The successful readbacks support this experimental path; the public API does not document atomic replacement and durable acknowledgement under server failure. A readback is an additional check, not a substitute for that missing contract. Mismatch, timeout, or an uncertain write result stops the owner's subsequent writes and is reported explicitly.

4. **Preserve single-writer appends; require native append as follow-up.** The desk owns its fleet journal; a trainer owns its run ledger; host journals keep their existing owners. With no native append route, API append temporarily reads the committed prefix and replaces it with that prefix plus the canonical line, under the owner's serialized write path and the replacement limitations above. Samarth accepted this temporary implementation. A process-local lock cannot enforce cross-process ownership: before replacing a writer, existing desk recovery must establish that the old owner stopped. On an ambiguous write timeout, stop dependent writes and reconcile the pending operation; never blindly retry an append or treat the file as absent. A GET alone is not proof that an outstanding request can no longer write later. **Required provider follow-up: native atomic append, preferably with an expected offset/version and an idempotency key or operation-status lookup.** Whole-journal replacement costs grow with journal size, and the absent conditional/idempotent operation limits automatic recovery. This is explicit technical debt, not the intended permanent append implementation. Mounted append may use the existing local append operation followed by mount sync; it still has one owner and must hydrate its committed prefix on resume.

5. **Publish each mounted mutation with `sync /scratch`.** `StrangeLoopLocalStore(LocalStore)` reuses local temporary-file/rename and append I/O, then calls a named commit implementation using `subprocess.run(["sync", str(mountpoint)], check=True, timeout=commit_timeout)`. `mountpoint` is `/scratch`, not the `rlstack/` subdirectory. Return from `_write` or `_append_line` only after successful publication; `_persist()` invokes the same operation. A sync failure or timeout must propagate and stop dependent mutations rather than acknowledge a ledger advance. Publish each prerequisite file before acknowledging its ledger append, as Samarth requested. Explicit commits and background commits can publish other pending files on the mount; they do not create an all-or-nothing transaction across files. Recovery still ignores work beyond the committed ledger. Serialize each writer's mutations and commits; preserve distinct owners for distinct journals. Test concurrent mounted activity before running multiple experiments on one mount. Keep batching deferred despite its likely performance benefit. A new platform configuration must pass the publication check before mounted writes are enabled; a no-op callback or file fsync is insufficient.

6. **Read authoritative bytes without relying on mount reload.** Initially route `_read`, `_exists`, `_list`, `_size`, run discovery, and mutable coordination reads through the API in both Store classes, even when a mounted copy exists. Resident readers use this same route. The measured equal-length replacement is the regression case: API v2 must be returned while the mount still contains v1. Fix or bypass the observer's size-only mutable cache for this backend; otherwise it can hide the updated API bytes. Add a mount fast path later only for proven immutable content such as hash-validated CAS bytes, with existence/deletion handling covered by tests. Do not write read-through cache fills into `/scratch`: background commits could publish stale downloaded files over newer versions. Optional downloaded caches live on ephemeral local disk. This refreshes what the Store sees, not arbitrary external libraries reading the mount directly; such consumers need explicit immutable staging or a future provider reload operation.

7. **Resume from authoritative history.** Establish stopped ownership first, fetch the exact run's committed manifest and ledger, then hydrate the mounted writer's append target before its first append. Base-class recovery performs its existing cleanup through backend methods. Listing, deletion, and temporary cleanup must cover committed files absent from the mounted snapshot, not only what a local directory walk happens to see. Use the authoritative delete route and invalidate cached state for such files; do not leave staged writes to a deleted key that could later resurrect it. Commit and remote deletion must be serialized appropriately within the existing single-owner scope.

8. **Reopen in child processes.** Add explicit backend cases to `open_store`. Resident readers reopen without a live HTTP client or token in `StoreAddress`; credentials are provided through worker runtime configuration. They perform authoritative API reads rather than inheriting a stale mount-only view. Read-only callers reject mutations, including destructive `open_run` attachment; the observer continues to use peek methods. Validate that bootstrap supplies the API configuration to spawned residents, since Strange Loop did not inject it automatically in the probe.

9. **Keep scientific bytes unchanged.** No provider fields enter manifests, run IDs, adapter payloads, ledger serialization, or `store://` run references. Identical writes and resume operations must produce identical run files to the existing backend. Physical mount roots, credentials, transport addresses, and storage capabilities stay outside scientific identity.

### Touched / untouched

| File | Implementation |
|---|---|
| `rlstack/data/stores/strangeloop.py` — new | Scratch client, explicit location/commit configuration, API store, mounted store, freshness and write rules. One backend module; stdlib imports. |
| `rlstack/data/stores/address.py` | Reopen both access modes in residents with non-secret addresses and runtime credentials. |
| `rlstack/data/stores/__init__.py` and `rlstack/__init__.py` | Export the new store classes without eagerly importing a provider SDK. |
| `rlstack/data/stores/base.py` | Update the backend-name documentation on `StoreAddress`; preserve the existing fields, byte verbs, key tree, and serialization. |
| `rlstack/observe/locate.py` | Resolve a Strange Loop store locator to an API reader; reuse the existing UI and exact run references. |
| `rlstack/observe/cache.py` or its construction in `observe/locate.py` | Bypass size-only mutable caching for this backend initially, or explicitly revalidate bytes. A same-size replacement must become visible. Preserve existing behavior for other backends unless a shared change is justified. |
| `tests/test_strangeloop_store.py` — new | Fake HTTP/commit tests for both views of one authoritative store, failures, append serialization, and missing mounted files. |
| `tests/test_strangeloop_store_live.py` — new | Opt-in checks on a unique test prefix for API semantics, mounted commit/visibility, and crash recovery; no allocation at import or in ordinary discovery. |
| Existing store, resume, retention, explicit-directory, resident, and UI tests | Reuse the applicable contract cases and add focused cases where these integration seams need coverage. |
| `ARCHITECTURE.md` and `agent-context/CONTEXT.md` | Record the backend and evidence when implementation lands; this proposal is not an implemented canon entry. |
| `deploy/strangeloop.py` | Thin CLI over the shared runtime and provider adapter; details are in the implementation section below. |

Expected unchanged: `local.py` and `modal_volume.py` retain their existing behavior; `runner/daemons/trainer.py`, model engines, learners, losses, and spec hashing keep using the existing Store. `runner/residents.py` already calls `open_store` and should need no storage-specific behavior. `deploy/desk.py`, `deploy/ui.py`, and `deploy/modal_venue.py` continue to serve the existing Modal deployment. Avoid a shared mount-store refactor until concrete duplication warrants one.

### Promises / non-promises

- A provider failure never turns into an empty history or an acknowledged commit.
- One store identity and exact `subdir/run_id` paths survive both access modes.
- An acknowledged ledger update has its prerequisite files durably readable.
- Readers and observers do not mutate a live run while discovering it.
- Backend support can advance from tested API reading to verified writing and mounted acceleration as provider capabilities become available.
- No claim of general multi-writer transactions, instant mount coherence, native RPC, custom images, automatic acquisition, or measured throughput is made.
- The proposal does not assume access to Strange Loop's Modal credentials merely because the API returns a volume name.

### Interfaces and sketches

```python
class ScratchClient:
    def __init__(self, volume: str, prefix: str = "rlstack", *,
                 token=None, api_base=None, profile=None, timeout=30.0): ...
    @classmethod
    def from_locator(cls, locator: str) -> "ScratchClient": ...
    def read(self, key: str) -> bytes: ...
    def write(self, key: str, data: bytes) -> None: ...
    def delete(self, key: str) -> None: ...

class StrangeLoopStore(Store):
    def __init__(self, scratch: ScratchClient, *, read_only: bool = False): ...

class StrangeLoopLocalStore(LocalStore):
    def __init__(self, root, scratch: ScratchClient, *, mountpoint="/scratch",
                 commit_timeout=30.0, read_only=False): ...
    def verify_publication(self) -> None: ...
    def _persist(self) -> None: ...

# verify_publication enables mounted writes only after checked sync publishes
# an isolated marker through the API. _sync is the bounded named operation,
# never a caller-supplied callback that could silently do nothing.
```

### Build and validation order

1. Implement location handling, the scratch reader, file metadata, and a local observer against a fake HTTP service. Confirm errors propagate accurately.
2. Implement the single-owner API writer and readback/ambiguous-result handling using the measured replacement/delete behavior. Model interrupted operations in focused tests; keep the missing provider atomicity and acknowledgement contract explicit.
3. Implement checked mount sync and per-mutation publication. Fake tests must assert that acknowledgement follows commit, failures propagate, and child readers use the API. Repeat the publication check when the worker platform changes. Delay batching and broad caching.
4. Run equivalent store and resume scenarios across both access modes. Exercise stale existing ledgers, missing mount files, wrong subdirectories, owner replacement, partial upload/commit, torn journal tails, deletion, and a second reader. Compare exact run bytes, including canonical JSON and compressed data.
5. Run `python3.13 -m unittest discover -s tests`; record dependency skips. Keep tests of the existing Modal store to catch unintended shared-interface changes.
6. After implementation, run a bounded live Store integration drill: a worker writes, a distinct reader observes, and a replacement worker resumes after verified old termination. Exercise interruption around data publication and ledger acknowledgement; the completed infrastructure probe did not test this. Multiple writers to distinct files must coexist without loss. Use isolated prefixes and finite leases, release compute when done, and clean only drill files. Scratch's CPU downloader is not an arbitrary test runner.

### Implemented Desk and worker wiring

The implementation reuses PR #1's HTTP transport. `runner/venues/runtime.py` supplies `DeskRuntime` and `MetalRuntime`: the existing Desk, Campaigns, MetalService, registration, heartbeat, host statistics, resident watchdog and teardown get process lifetimes without a new placement policy. `runner/venues/strangeloop/provider.py` owns the provider's official CLI calls, allowed allocation slots, durable labels, provider lease identities, finite idle configuration, code/config transfer and SSH processes. `runner/venues/strangeloop/desk.py` wires one laptop gateway and the existing observer; `runner/venues/strangeloop/worker.py` verifies reverse connectivity and mounted publication, measures the real devices, and runs the shared metal. `deploy/strangeloop.py` remains a command wrapper. `examples/strangeloop-desk.md` is the operating guide, and the example config and Dockerfile fragment are reviewable inputs for the first live drill.

All callers use the same canonical HTTP gateway address. Each worker's SSH connection supplies both a local forward to its service and a reverse forward from its loopback to the laptop gateway. The gateway relays frames to a metal/host; it does not admit GPU work or own model state. Same-process host routes retain the existing switchboard. This avoids recording laptop-only localhost endpoints that another GPU cannot reach. OpenSSH reverse forwarding is a required startup capability and still needs live confirmation on Strange Loop.

A new allocation requires a ready named image and room under the configured Strange Loop account GPU ceiling. Configured slots fit inside that ceiling, launch decisions serialize across slots, and unresolved launch labels reserve capacity until reconciled. The requested label is saved and fsynced before the CLI launch. A timeout retains that label; a later explicit request reuses it. An active lease is reattached, and a failed old lease must be explicitly terminated and reported released before replacement. A daemon-start request is recorded before issuing it and is never replayed automatically after uncertainty. Tunnels are owned by the provider lease ID, not only by a metal name. Boot completion requires HTTP readiness plus Desk registration of the exact provider lease and a live epoch. The release door also covers known leases whose bootstrap failed before registration.

Automatic GPU idle release remains finite. The shared Desk now journals retirement intent before teardown and retains unresolved owners as listed but unplaceable. An in-process owner needs acknowledged shutdown; a provider-backed owner needs confirmed termination of its recorded container ID. Whole-container release strands every affected unfinished run before forgetting ownership. Failed or uncertain termination survives journal replay, and stale registration or a reused carve name cannot erase that custody. Delivery intent records the exact destination before adoption, so a lost response or restart cannot age into permission to duplicate. Explicit operator releases stay parked rather than being undone by recovery. Strange Loop exposes an `automatic_recovery` setting, default off pending the live stop/publication drill; explicit and finite idle release remain enabled. The local Desk must stay running while experiments depend on its gateway. Lease extension is explicit.

Integration uncovered a preexisting shared HostService bug hidden by the PR's direct-learner GPU fixture. Real hosts hold a resident learner proxy, whose synchronous call could not execute on the host event loop. Learner calls now run off-loop, serialize on the HostService's owning loop, and hold admission until already-started synchronous work finishes despite caller cancellation. Foreign LocalTransport callers marshal to that same owner loop; a per-caller loop lock would allow overlapping mutations. Retirement closes admission, awaits experiment cleanup and owned calls/builds, refuses success if a resident survives SIGKILL, and outlives a caller cancellation. Strange Loop termination performs bounded released-status polling even when the termination response was lost; the Modal adapter likewise checks finished status after its stop request. HTTP timeouts and inner resident timeouts still do not themselves prove physical completion. Provider teardown and the settlement of already-accepted scratch writes after process termination remain unproven on live infrastructure.

`runner/exporters/wandb.py` is a read-only companion process. It logs only complete committed ledger lines, uses an explicitly selected real scalar as `objective`, and publishes its exact offline bundle path for `gpu declare`. It never modifies scientific files. The local submission command declares the run and starts this exporter; clients using RemoteDesk directly must arrange that bookkeeping. Offline exporter restarts use a new attempt and replay committed history because the [W&B SDK explicitly ignores resume in offline mode](https://raw.githubusercontent.com/wandb/wandb/main/wandb/sdk/wandb_init.py). The adapter does not pretend a side cursor can commit atomically with a W&B log.

### Implementation validation and remaining live work

The focused local suites cover scratch publication ordering/failure quarantine, exact Store bytes and resume, stale mounted reads, same-size observer changes, resident reopening, allocation labels/limits, two-hop gateway routing and epoch rejection, registration/heartbeat lifecycle, mandatory idle release, disabled implicit recovery, HTTP-to-resident learner calls, cross-loop serialization/cancellation, teardown, and offline metric export. The ordinary test suite skips the explicit opt-in scratch drill. No image build, GPU allocation, model training or production W&B run was performed during implementation.

Live work still required: build and validate the pinned image over the provider runner, confirm reverse forwarding, run the complete local Desk to mounted worker to spawned residents path, check actual adapter/optimizer payloads, verify multiple distinct writers and abrupt failures around publication, and establish stopped ownership before a replacement resumes. The subsequent HTTP implementation in [ADR 0016](0016-http-rpc-transfers-use-temporary-blobs.md) transfers large requests and results through verified temporary blobs, with configurable finite budgets. A real 65 MiB request and result have crossed both local gateway hops. The original 64 MiB bound now applies to control/legacy frames; real adapter sizes, memory cost and live SSH throughput still need measurement. Native scratch append, conditional writes, provider operation status and stronger durability semantics remain follow-up capabilities.

### Outstanding difficulties

- **Provider write contract and ambiguous operations.** Ordinary API replacements passed, but atomic visibility during server failure and durable acknowledgement are not documented by the exposed schema. No conditional replacement, write idempotency key, or operation-status lookup was found. The initial experiment must stop on uncertain write completion; seamless automatic recovery needs a stronger provider contract or another serialization service. Native append remains a required follow-up.
- **Recovery and concurrency validation.** Local Store regressions now cover exact bytes/resume, torn tails, stale metadata/deletes and uncertain publication. The live multi-worker and abrupt-failure drill is still outstanding; fake ordering and normal teardown persistence do not prove provider crash behavior. A stale mounted file must not resurrect an API deletion during a later commit. The shared Desk now requires stopped-owner evidence and retains uncertain custody across journal replay. Its local regressions do not establish the provider-side stop or publication contract. Strange Loop already supplies explicit termination and released-status queries; this was a preexisting shared recovery gap, not an established lack of provider termination capability.
- **Provider configuration and cost.** Mount publication is demonstrated on this lease; the Volume version is not explicitly advertised. Per-write sync costs 0.7–2.3 seconds in this small sample, and API reads add request overhead. This is acceptable for the initial experiment but needs measurement with real journals and checkpoints before batching or scale claims.
- **Live Desk integration.** Runtime factories, endpoint routing, allocation/finite idle release and the separate W&B exporter are implemented and locally tested. A compatible-image fragment exists, but the storage probe's default Python 3.10 remains unsuitable and the actual Python/CUDA/vLLM image has not been built or exercised. Reverse SSH forwarding, offline weight/tokenizer cache compatibility, the full Desk-to-resident path, and actual metric export still require the live drill. Large HTTP payloads now use verified temporary blobs; operation-status lookup and upload resume remain deliberate limitations in ADR 0016. Native mount reload is useful future provider support, but is not a prerequisite for the chosen Store coordination path.

## Questions

**Q1. Is scratch the authoritative store for both the local desk and GPU workers?** Recommendation: yes. This uses the newly provided shared volume and avoids inventing replication from an authoritative laptop directory. The alternative requires a separate replication, availability, and recovery design.

> **Samarth:** the rest of your implemntation details look pretty legitimate.

Retain the original one-store design; this follow-up asked to resolve the listed uncertainties, not to replace scratch with a laptop-authoritative store.

**Q2. Should the first mounted writer persist each write before returning?** Recommendation: yes, then measure before introducing grouped commits. This keeps crash ordering explicit and the existing byte contract intact. The alternative is batching at ledger boundaries from the outset, which needs a stronger tested publication protocol and evidence about background commits.

> **Samarth:** sure im fine with this for now

Resolved as per-mutation `sync /scratch`, with the measured latency and failure rules above.

**Q3. Should mutable reads use the authoritative API initially?** Recommendation: yes, accepting request overhead. Mounted reads can be introduced where freshness is proven. The alternative is a worker-wide refresh protocol, which must avoid disrupting concurrently running experiments and stale readers.

> **Samarth:** i want this loop to work right now, so let's just do this properly. how are reloads, etc. supported right now in strangeloop?

Investigation answer: native reload is not exposed; a live worker successfully used the API while its mount stayed stale. Implement fresh Store reads through the API, including the observer and spawned residents. This records the technical resolution, rather than attributing a separate verbatim approval to Samarth.

**Q4. Is single-owner whole-file replacement acceptable for API journal append?** Recommendation: yes if atomic replacement and ownership are verified, with dependent writes stopped on ambiguous completion and journal growth measured. The alternative is to wait for provider append/conditional-write support or run a dedicated journal-writing service. Either changes when local desk writing becomes usable; none permits two desks to rewrite the same journal concurrently.

> **Samarth:** ok we'll do this for now, but make an explicit note in the ADR that we need append support.

Retain temporary single-owner replacement and record native append as required provider follow-up, including the retry/conditional-write capabilities that would make recovery tractable.

## Outcome

Worktree creation and the initial proposal landed in `9d4abd9`; storage-probe results landed in `3ed7a6e`. The declared infrastructure run `scratch-storage-probe-20260912` logged its actual visibility and latency measurements to offline W&B under the lease's artifact directory; Strange Loop accepted `gpu finish` and reported the lease released. `gpu list` then reported zero active GPUs. Probe scratch files and temporary credentials were cleaned up after retaining the evidence. The subsequent 0.5.9/PR #1 review is recorded above, including its local test results. The subsequent implementation is present in this worktree as described above; no dataset was transferred.

ADR prose now uses one source line per paragraph or list item. `knowledge/TEMPLATE.md` carries the same formatting instruction for future ADRs. This removes hard source wrapping; soft wrapping in a viewer is a separate display setting.

The implementation preserves all inherited staged and unstaged work separately. ADR paragraphs continue to use one source line each. A canon entry for the shared runtime and backend should accompany integration into the main development history; this isolated worktree does not change the spec invariants.

Implementation validation completed with `python3.13 -m unittest discover -s tests`: **1,492 tests, 194 skips, no failures**. The skips include the explicit live scratch drill and optional GPU/dependency tests. The default suite did not allocate compute or claim live validation.


### Shared provider runtimes (ADR 0017, 2026-09-13)

The later abstraction refactor makes both providers use the same DeskRuntime and MetalRuntime, with matching provider/desk/worker modules under `runner/venues/modal/` and `runner/venues/strangeloop/`. The Strange Loop laptop module is now `desk.py`. Modal deployment files delegate their lifecycle to these adapters. Submission, readiness, guarded release and observer polling now share VenueClient. This supersedes the earlier expectation that Modal deployment wiring would remain unchanged; Store layout and publication rules remain as recorded above. See ADR 0017 for the migration checks and remaining live-validation boundary.

### Live launch corrections, 2026-09-13

Scratch rejects `:` in physical filenames (HTTP 400), although shared rlstack host names legitimately contain it. The backend now escapes unsupported UTF-8 bytes as `@hh` within logical key components and escapes literal `@` as `@40` to avoid collisions. Both API access and mounted `path_of` use the same encoding; listings decode canonical names back to the original Store keys. Ordinary alphanumeric/dot/dash/underscore/plus keys retain their existing filenames. The configured scratch prefix is a physical root and is not reinterpreted. Shared host identities, logical Store layout, serialization, manifests and scientific bytes remain unchanged. Direct CLI inspection sees escaped physical filenames. A live API drill verified journal append/read/list for colon-containing host names and a distinct literal escape-looking name; mounted/API agreement passed locally and still requires the next live host build. Native arbitrary-filename support would remove this adapter constraint.
