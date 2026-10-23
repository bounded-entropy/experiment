# ADR 0028 — Adapter definitions drive learner and engine lowerings

| | |
|---|---|
| **Date** | 2026-10-23 |
| **Status** | Accepted — Samarth approved the design and requested implementation on 2026-10-23 |
| **Author** | Codex, Dreamer fact-writing continuation |
| **Touches** | `policy/`, `spec/validate.py`, `registry.py`, `runner/engines/`, `runner/learners/`, `rlstack_engine/`, tests and architecture documentation |
| **Invariants** | I2 (policy bridge), I3 (code identity), I4 (submit validation), I6 (recording), I7 (substrate parity), I8 (multi-tenancy); preserve existing run/store identity and resume behavior for legacy adapters |
| **CONTEXT** | Extends the Lowering, RowPlan and BatchView contracts described in ARCHITECTURE.md; integrates with the separate accepted ADR 0027 on universal adapter state/checkpointing/transfer; implementation gets a new numbered CONTEXT entry |

## Original prompt

> is there a more general abstraction you can think of, where we can generally write our own custom triton kernels for whatever adapter type we could want, and it would just work? we have a bunch of hook sites right now. an adapter has a particular hook site and replaces its output with soemthing else. all the base model parameters still need to have their own big GEMM across all the runs/experiments, and then this needs to be a change to that value (its own small kernel).

> what could the inputs of the adapter consist of? its output is just one vector right? each site can be hooked onto, and its just a particular hook that the adapter uses.

> if we're doing decode, if the adapter influenced the KV cache and modified it in some way, then theres no need to modify all tokens, right? can we still maintain efficiency in that regime (it only needs to do smth to the last token). the kernel might want to know what things have already been affected and what hasnt

> ok can you make a concrete plan for implementing this abstraction in our codebase? i want adding a new adapter to just mean choosing a hook site, declaring its inputs/state, and writing its triton kernel

> question: there are already adapters supported by vLLM natively (e.g. multi lora punica). is there a way to make these already built in adapters cleanly specifiable in our interface?

> ok sure that makes sense. also, the entire lowering logic of having the same kind of adapter in both inference engine and the learner needs to now support this interface, instead of whatever it supported before.

## Context / problem

The audited baseline is the private framework snapshot at `research/dreams-impl` in `bounded-entropy/dreamer-experiments`, used by the recent Dreamer runs. It is ordinary tracked source, not the different checkout in `/Users/samarth/Coding/experiment/rlstack`. The original plan targeted that snapshot; the coordination update below requires choosing the consolidated implementation checkout before coding. Do not edit two copies in parallel or merge the private repository into the public framework. See the repository README and `docs/boundary-audit.md`.

Coordination update (2026-10-23): the chat "Summarize RL experiment chats" has already accepted ADR 0027, `knowledge/0027-checkpointing-and-weight-transfer-are-universal-adapter-services.md`, in `/Users/samarth/Coding/experiment/research/rlstack-transfer` on `codex/universal-adapter-state`; implementation is in progress and is not present in this private snapshot. Reserve 0027 for that work and number this proposal 0028. Before implementing this proposal, reconcile its target checkout with the consolidated framework and the completed universal-state interfaces. Its state declarations should generate the shared state representation consumed by those services, rather than introducing another checkpoint/publication protocol. The other chat owns checkpoint cadence, NCCL/base64 transfer and publication acknowledgements; this proposal owns hook-site execution, device banks, routing and graph-safe compute. No edits to that chat's worktree are made by this renumbering.

Existing declarations already cover much of the requested API: `policy/adapters/base.py:71` registers adapter types, `policy/siteschema.py:27` describes sites, `policy/adapters/rollout.py:451` defines rollout lowering, and `policy/adapters/replay.py:112` shares replay wrappers and row routing. These should remain the public entry points rather than starting a second adapter registry or runner.

The missing execution primitive is a common batched transformation over each site's current token rows. Native LoRA uses the engine's existing implementation; custom steering has a separate implementation. `rlstack_engine/steer_worker.py` installs a Python model pre-hook and residual hooks. `rlstack_engine/batch_view.py` reads engine tensors with `.tolist()` and constructs Python per-token tuples. `rlstack_engine/steer.py` constructs routing tensors and banks dynamically. `policy/adapters/steer_vllm.py:64` therefore demands eager execution. This is a limitation of that implementation, not a claim that CUDA graphs cannot capture operations initially launched from a hook.

There are two integration gaps beyond writing kernels. `VllmEngine._pay_demands` currently refuses `BuildDemands.plugin`, and the steering worker claims its model seam for only one plugin. A general mechanism needs actual build/install integration and an explicit resident-state path; adding a new enum or relaxing the eager flag alone is insufficient. The existing parity certificate types are designed but unwired, so their existence does not certify a new execution path.

Performance and numerical behavior of the proposed mechanism are unmeasured. This ADR launches no GPU, generation or training job.

## Decision

Make one `AdapterDefinition` the semantic source for BOTH lowerings: the learner's replay implementation and the inference engine's rollout implementation. It declares logical sites, inputs, parameter/request state and operations. Operations can be existing primitives such as `LowRankAdd` or custom Triton-backed site programs. Lowering chooses a supported native engine implementation, a custom site implementation, or a learner implementation without redefining the adapter's math or state. Add `SITE` as the common serving mechanism for custom shape-preserving site programs, alongside existing native mechanisms. The framework owns site installation, per-token routing, batching, allocation, serialization, immutable parameter versions, cache identity, graph capture and dispatch. A novel trainable operation additionally supplies a backward operation or an equivalent differentiable PyTorch reference; an arbitrary Triton forward cannot supply its own derivatives automatically.

The shared definition and native/custom implementation choice are user requirements settled by the continuation above. This is a replacement of the old independently authored lowering contract, not a serving-only convenience layered beside it. Existing adapters migrate to the same definition; their optimized native kernels can remain.

### Author interface

Extend `AdapterType` with a concrete typed `definition` contract, not another top-level adapter registry. `AdapterDefinition` is a stdlib-only record shared by native primitives and custom operations; `SiteProgram` describes the custom site operation within that definition. Tensor roles, state scopes, shape expressions, initialization, precision and mutation are named fields/enums, not arbitrary dictionaries or executable strings. Heavy implementations are importable symbols resolved lazily on the relevant worker. The existing `AdapterSpec.site` selects configured sites; the definition declares which site contracts its operations accept. Common lifecycle methods are derived from the definition, not reimplemented for each adapter and each side.

Illustrative declaration only; names and constructor details are proposed:

```python
@adapter_type("gated_steer")
class GatedSteer(AdapterType):
    definition = AdapterDefinition(
        parameters=(
            Parameter("direction", shape=(Dim.OUTPUT,), init=Zeros()),
            Parameter("gate", shape=(), init=Constant(0.1)),
        ),
        request_state=(),
        operations=(SiteProgram(
            accepts=SiteKind.RESIDUAL_OUTPUT,
            inputs=(SiteInput.OUTPUT,),
            writes=SiteWrite.REPLACE_OUTPUT,
            scratch=(),
            forward="my_adapters.gated_steer_torch:forward",
            training=Training.NONE,
        ),),
    )

# Existing policy configuration chooses the location.
AdapterSpec("gated_steer", site="resid_pre.10")
```

The example preserves the existing, somewhat surprising `resid_pre.10` meaning: the stream leaving layer 10. Do not rename old sites or change their semantics in this work. The same declaration with a differentiable reference or registered backward is trainable. A declaration alone does not claim a site is reachable on a particular engine build.

At execution, the framework resolves declarations into positional tensor arguments and a fixed launch plan. The adapter receives no request objects, Python dictionaries, file paths or engine internals. One forward operation may launch multiple Triton kernels using declared scratch; it need not be a single kernel. Ordinary new types require no engine, worker, routing or transport changes.

For built-in LoRA, the same author surface declares matrices A/B and `LowRankAdd(input=LINEAR_INPUT, a="A", b="B", scale=1.0)` instead of a custom kernel. The rollout compiler can realize that operation through vLLM's native multi-LoRA support, while replay can realize it through differentiable PyTorch or a registered Triton implementation. The native path does not also install a custom output hook: exactly one provider owns each resolved operation. Arbitrary custom operations cannot claim a native provider merely by presenting matrices with familiar shapes.

### One definition, two compiled lowerings

The pipeline is `AdapterSpec + AdapterDefinition -> resolved logical sites/state -> {ReplayLowering, RolloutLowering}`. Resolve logical parameter names/shapes, initialization, scaling, placement and draw semantics once; both lowerings consume that resolved record. The compiler generates their installation, loading, routing and state-conversion plans. Backend providers describe how to execute an operation on a substrate, not another adapter definition. A provider is shared across all adapter types expressible with that operation; adapter authors do not write a new pair of provider classes for each new type.

| Concern | Shared definition | Learner replay | Inference rollout |
|---|---|---|---|
| Sites and inputs | Logical tensors and supported sites | Resolve HF modules and training row layout | Resolve vLLM packed modules, shards and query rows |
| Parameters | Canonical names, dimensions, initialization, groups and trainability | Mutable trainable tensors and isolated optimizer state | Immutable serving-version slots, including native installations |
| Math | Primitive operation or custom program, scaling and precision contract | Autograd-capable provider | Native engine provider or custom GPU provider |
| Context/state | Windows, conditioning, latent draw and state-update semantics | Reconstruct from sealed facts | Resolve once, record required facts and preserve cache validity |
| Publication | ADR 0027 state description | Export the declared serving state | Install that exact state's deterministic representation |

Provider selection is explicit at the build boundary and checked against the actual substrate: supported sites, ranks, dtype, scaling, layout, training/backward and graph capabilities. An automatic preference may choose native first only among compatible, parity-qualified providers. An explicitly requested implementation fails if unavailable; no fallback may drop operations or change semantics. Keep execution preference/build fingerprint distinct from the adapter's semantic definition, and record the selected provider for interpreting numerical differences.

Standard parameter layouts are canonical across both sides. A native provider can transpose, shard, pack or materialize a declared serving view, but must define and test the mapping to that canonical state; it does not independently initialize or reinterpret it. Stochastic/ensemble adapters additionally declare their latent/member selection and replay facts. Lowering or checkpoint frequency must not advance a draw or emission counter accidentally. Readout-only and non-trainable operations are declared explicitly rather than assigned a fabricated backward.

Migration is part of this ADR. Inventory every registered existing adapter, including LoRA, steering/norm-scaled steering, DreamBank/PLoRA, spectral/task-conditioned types, soft prompts, attention bias and value heads. Represent local activation changes with site operations; represent prompt, attention and readout behaviors with explicit operation kinds and capability checks. The initial custom kernel contract remains shape-preserving, but the overall definition must not pretend all existing adapter behavior is an output-vector replacement. A currently refused serving operation remains refused until its backend actually exists. Compatibility shims can read old specs/payloads and invoke old compute during migration, but they cannot remain an alternative semantic declaration or silently imply unified support is complete.

Completion means each current adapter's two sides enter through this common definition and lowering compiler. Preserve all previously implemented adapter/backend combinations; marking a working legacy combination unsupported is not a migration. Operations whose backend was already unimplemented remain explicitly refused before execution. Preserve backend-specific compute modules where useful, but retire per-type lifecycle/math duplication from `params`, `emit/load`, `install_replay` and `rollout_lowering` as adapters migrate. Native serving and custom execution must both appear in the mixed-tenant parity tests.

### Site contracts and shared base computation

Each supported site defines its readable tensors, replaceable output, dimensions/strides/dtypes, aliasing rules, causal position, and tensor-parallel layout. The first sites are linear output, decoder residual output and final normalized output. Linear sites expose both the original input and output so low-rank transformations can compute from the input. Boundary sites expose the logical residual stream even when vLLM physically carries a `(hidden, residual)` pair; adapters should not need to rediscover fused-residual semantics.

For a linear site, execute `Y = X @ W.T` once over the combined current-token batch, then execute `Y'[i] = F(X[i], Y[i], parameters[slot[i]], metadata[i])`. Preserve packed base projections: map logical q/k/v and gate/up sites to slices of the engine's packed result rather than replacing a fused base GEMM with one GEMM per adapter or logical projection. Account for deferred bias addition, quantization and input/output shard placement when establishing the contract. Read-only inputs cannot be mutated; the wrapper uses a designated output buffer or a permitted output alias and preserves all other tuple members.

Different experiments can use different adapter types at the same site. The build creates a deterministic, fixed list of operations for its registered types. Each operation receives per-token applicability and parameter-slot columns; inactive and padded rows are identity. Capture this fixed list, with dynamic masks or bounded group descriptors stored in GPU memory. Never select a Python callback from a token's type during replay. Adding new parameter versions does not change the graph; adding a new kernel type or exceeding a declared capacity requires a new build/capture, outside active requests.

Preserve the bank's existing one-delta-per-tenant-per-site rule. Different tenants may have different types at one site; same-tenant composition at the same site is refused. Ordered composition can be a future explicit contract, rather than depending on wrapper installation order. Preserve the native LoRA/DreamBank/PLoRA serving path while validating the new mechanism.

### Token metadata and state

The hot path consumes persistent GPU columns for `active`, `request_slot`, `parameter_slot`, absolute `position`, query offsets, and previously computed prefix lengths. These describe current query rows, not the whole cached sequence. A one-token prefill is not identified as decode merely from query length. Expose any genuinely necessary phase field from the engine scheduler. An adapter requests only the columns it uses; `BatchView` remains the pure diagnostic/test view, not an obligatory GPU-to-Python conversion on every forward.

State has three distinct scopes:

| Scope | Ownership and lifetime |
|---|---|
| Parameters | Per adapter version/site; declared trainable or frozen; deterministic initialization and serialized dtype/layout; immutable while serving requests pin that version |
| Request state | Per request/site; separate from parameter slots; declared initialization/reset and causal update rule; released on finish/cancel |
| Scratch | Per execution lane/graph bucket; uninitialized temporary storage with declared size; never treated as persistent history or included in a checkpoint |

Initial delivery supports parameters and request-constant conditioning. A follow-up within this plan adds position-addressed request state: each current token writes only its own declared position, with a declared validity/coverage representation. Token rows may not race by updating one shared request accumulator. Arbitrary recurrent state needs a sequence-aware prefill implementation as well as decode and is not admitted as a token-independent kernel.

For fixed parameters and a position rule, prefix length plus that rule identifies cached computation; no universal per-token "already adapted" bitmap is needed. A program with history-dependent state must declare its own coverage or validity state. "Computed" and "adapter applied" are different facts. Physical cache block IDs and reusable request slots are not stable logical identities.

### Decode and cache correctness

Normal decode runs the program only over the newly computed token for each request. Prefill runs it over uncached prompt rows, including chunked prefill. Historical K/V is read by attention without rerunning activation adapters. Later in the plan, add a separate verified site for K/V staging immediately before the engine writes the current tokens to cache, with the RoPE/quantization ordering explicit. It operates on new K/V tensors and lets the engine perform its normal cache write; arbitrary historical cache mutation is not part of the local replacement contract.

Build-specific inspection must establish that K/V seam before claiming it works on the pinned engine. It is not implied by an HF module name and may require a thin model/backend wrapper. Side attention/LSE access remains a different mechanism, currently unimplemented, rather than being advertised as a post-output transformation.

Cache identity includes the parameter/bundle version, resolved program/site configuration and request conditioning/window that affects the prefix. Pin the program configuration for the request. A parameter update publishes a new version; it cannot silently update the cache history of an existing request. A continuation under different parameters uses an explicitly recomputed prefix or a separately specified mixed-history experiment.

If mutable request state cannot be reconstructed from cached prefixes, disable prefix reuse for those programs until state restoration is implemented. Disable speculative execution for mutable-state programs until tentative writes can be committed or rolled back; refuse the incompatible build rather than quietly retaining rejected-token state. Read-only/token-local programs may retain engine prefix caching and speculation only after parity cases pass. Never mutate a prefix-cache block shared by other requests.

### Serving integration and graph execution

Add one general site worker/plugin and a version-pinned model/runner bridge. Keep engine-specific tensor extraction, packed-projection mapping and cache locations in this bridge, outside adapter implementations. Initially pin the same vLLM V1 runner family the current custom hooks use; probe exact installed vLLM, PyTorch, Triton, model family, attention backend and TP layout. Do not infer compatibility from current online documentation for another release.

Prepare metadata at the runner's batch-preparation boundary, outside captured model execution. Reuse scheduler CPU columns or existing GPU tensors, copy into persistent buffers once per step as needed, and retain GPU-computed metadata on GPU. The model pre-hook is not this boundary: it does not execute again when a CUDA graph replays. Dummy capture/profiling must launch the real site programs with safe inactive rows so the captured graph includes them; returning early for an empty warmup would capture a graph without the adapter.

Allocate bounded parameter banks, request slots, routing buffers and scratch for each graph bucket before capture. Resolve kernel specializations and launch shapes before capture. Pad with masked inactive rows. Declare mutation/aliasing correctly for compiler integration; verify whether the pinned build should use `torch.library.triton_op`, `custom_op` or the engine's CustomOp bridge. Registration is not itself proof of capture safety or site installation.

For state loading, replace the steering-only file convention with the generic typed state description and tensor payloads provided by the universal services in ADR 0027. Generate that description from the site declaration and reuse its transport/version acknowledgements; NCCL publication must retain its direct tensor path, without introducing a file, base64 or checkpoint-serialization round trip. Resolve and populate a worker's parameter bank before its first execution, outside the graph and once per immutable version; do not deserialize in a layer hook. The batch-preparation gate must obtain the same bundle/version on all TP ranks before admitting its rows. Preserve the existing host-resident `knows_bundle` meaning; track worker-device readiness separately and do not report an unloaded device slot as ready. A boot spike must prove the actual worker installation/readiness seam against the completed ADR 0027 implementation; extend that shared path if necessary rather than creating a second acknowledgement protocol or experiment-specific publication queue.

A slot becomes usable only after all its copies complete. Requests and in-flight GPU operations hold both parameter and request slots. Publish by filling a free slot and marking it ready after stream/event ordering; reuse old slots only after their last use completes. Metadata buffers likewise cannot be overwritten while a replay still reads them. Initially use the engine's serialized forward lane; any overlapping lanes require separate routing/scratch buffers. Eviction or cancellation releases holds through existing lifecycle paths. Sleep/wake reconstructs banks and recaptures if addresses changed.

The default generic implementation supports local, token-wise transformations with declared replicated/input-sharded/output-sharded parameters and per-site dimension resolution. Do not implicitly perform an all-reduce to make an unsupported adapter look correct. TP=1 is the first live gate; TP=2 requires a separate shard-semantics and communication check before claiming support.

### Training and replay

The shared definition supplies deterministic allocation, parameter groups, standard tensor serialization/load, initialization and compiled rollout/replay lowerings. Replay lowers each declared operation through a training-capable provider; it does not discover another hand-written adapter definition in a `_torch.py` file. For custom site operations, the generic replay wrapper calls its inner base module once, retains the required inputs, flattens routed current rows, executes the same site math, and restores the original structure. Native primitives use their training provider under the same contract. Preserve other families' wrapper behavior during migration, gradient flow through inactive rows and existing optimizer/clipping isolation. A frozen base still needs activation gradients propagated through it during adapter training.

A trainable program declares either an explicit VJP/backward for its forward operation or a differentiable PyTorch reference. With a reference, the first backward can recompute that local reference under autograd; later it can be replaced by a Triton backward. Validate forward/reference agreement and gradients with respect to every differentiable activation input and parameter, including cross-token dependencies if any are declared. Scratch and request bookkeeping are not silently differentiable. Training through mutable request state is deferred until replay/BPTT semantics are separately specified. Training registration fails early when the program is forward-only.

There is no promise to infer derivatives of arbitrary Triton source. The author experience is declaration plus forward kernel for serving; declaration plus forward and a gradient definition for a novel trainable adapter. Existing differentiable primitive compositions can supply that gradient definition once and reuse it across adapters.

### Touched / untouched

- **Touched — `policy/adapters/base.py`, proposed `policy/adapters/definition.py`, `policy/adapters/site_program.py`:** add the canonical `AdapterDefinition`, native primitive/custom operation records, typed lowering-provider contracts and `Mechanism.SITE`. Reuse `ADAPTER_TYPES`, existing site selection, standard parameter groups and serialization contracts. Both sides must consume the same resolved definition.
- **Touched — `policy/siteschema.py`, `policy/compile.py`, `spec/validate.py`, `registry.py`:** validate logical tensor roles, shapes, state scopes, reachability and training support; include the declaration, forward/backward/reference and explicitly declared helper-source closure in new type identity. Preserve separate engine build fingerprints for platform/compiler details. Resolve this narrow code-hashing extension with Proposed ADR 0023, rather than silently assuming its general hashing changes already landed.
- **Touched — `policy/adapters/rollout.py`, `replay.py`, proposed `site_program_torch.py`, `site_program_vllm.py` and compiler/provider modules:** adapt BOTH lowering contracts to resolved definitions, share allocation/state lifecycle, and implement the generic replay wrapper and serving provider with lazy heavy imports.
- **Touched in migration — existing adapter declaration and `_torch.py`/`_vllm.py` modules:** move semantic state/math descriptions into the common definition, expose reusable native/training providers and reduce old per-type methods to temporary compatibility shims. Existing optimized kernel bodies can remain; an adapter is migrated only after its parity/state/restore tests pass.
- **Touched — proposed `rlstack_engine/site_worker.py`, `site_program.py`, `site_vllm.py`; existing `plugin.py`, `batch_view.py`, `slots.py`:** general execution, resident banks, GPU routing, concrete vLLM site mapping, capability probes and lifetime ownership. Put shared Triton operations in a lazy policy-side compute module, never importing `rlstack_engine` from `rlstack`.
- **Touched — `runner/engines/vllm_engine.py`:** install the site mechanism, handle its build manifest/readiness and compose its cache contribution with other lowerings; preserve the native LoRA path. Extend shared engine/host preparation only if the boot spike shows it is necessary.
- **Touched — tests, `STYLE.md`, `ARCHITECTURE.md`, `agent-context/CONTEXT.md`:** recognize the explicit heavy compute seam and Triton dependency without making the public package import heavy libraries; document the new mechanism and the final measured boundaries.
- **Untouched — existing scientific losses, reward processors, prompts and datasets:** this changes adapter execution rather than the scientific objective. Existing loss components, masks and reductions retain their meanings.
- **Untouched — desk placement, venues, resource limits, transport substrate and store layout:** use existing submission, bundle delivery and telemetry doors. No new private scheduler, GPU allocation or result format. If a shared preparation verb is required, extend those existing interfaces without creating a second transport.
- **Preserved behavior during migration — native LoRA/DreamBank/PLoRA execution and legacy payload decoding:** retain the native algorithms as providers beneath the new interface and keep old payloads readable. Their lowering interfaces do change; they are not permanently exempt from this ADR. Migration is opt-in until parity passes, with no automatic rewrite/resume of historical runs under changed code.

### Promises / non-promises

- **Promises after implementation and its gates:** a single semantic definition feeds learner and engine lowerings; existing native and custom adapter types use this contract, with a documented support matrix and explicit refusals. A new supported token-local adapter needs no custom worker/lowering/loading/routing implementation; inactive rows remain unchanged; the base projection is shared across the mixed batch; old parameter versions remain valid while held; normal decode does no adapter work on historical tokens; graph replay observes changing metadata and parameter slots correctly.
- **Promises after implementation and its gates:** unsupported inputs, trainability, sites, state/phase combinations, capacities and shard layouts fail explicitly. Stateless adapters give equivalent results alone, interleaved with other tenants, and across chunked prefill/decode within declared tolerances. Required readiness, cache, restore and cancellation tests pass.
- **Non-promises:** a universal arbitrary-memory kernel sandbox, arbitrary hooks inside fused operations, automatic autodiff for arbitrary Triton, historical KV rewrites, recurrent state/BPTT, shape-changing model surgery, immediate TP or multi-model-family support, or a specific throughput gain before measurement. Local activation transformations are the first complete scope.

### Interfaces and identity

`AdapterType` produces the canonical definition; the common compiler resolves it and produces both a typed replay lowering and a `RolloutLowering` using existing `BuildDemands`, `EnginePlugin`, bundle residency and routing contracts. Both prior per-type lowering entry points are migrated to this pipeline. `SiteMeta` remains checkpoint-attestable; engine reachability and physical tensor mapping stay build-specific. Runtime route rows have a concrete GPU record separate from the pure `BatchView`. Generic per-token applicability and request context are recorded only when they change replay math, following existing `records`/turn-extras rules.

New program identity includes semantic declaration values, state layout/init, forward/reference/backward code and declared helper sources. Resolve importable symbols to their source identity rather than hashing file paths as scientific settings: a pure module rename must preserve I3's rename behavior. A change of source, window, state policy or initialization must not silently reuse the same new-program identity. Engine/kernel build fingerprint includes base revision, site bridge, attention backend, TP, compiler/framework/GPU versions and capture configuration. Preserve the existing hashing and checkpoint bytes for untouched legacy types; new code receives new identities and never overwrites an old run dictionary. Record explicitly where floating-point tolerances, rather than byte equality, are the contract.

### Implementation sequence and acceptance gates

1. **Shared definition, both lowering contracts and engine seam spike.** Inventory all existing adapter types and add pure definition/provider validation and fake replay/rollout compilation cases. Represent LoRA with one shared definition and verify it selects a native serving provider and a differentiable learner provider. Inspect the exact pinned engine for batch preparation, packed linear outputs, fused residuals and worker-state loading. Produce a site/provider support matrix and a static launch plan. Gate: both lowerings consume the same resolved parameter/math record; every requested input has an identified source; metadata preparation demonstrably runs before each replay; dummy capture includes the kernels; loading/version acknowledgements have an identified control path. Resolve unavailable sites before generalizing code.
2. **First complete serving path: steering.** Implement persistent banks/routing and `SITE` build installation, then express plain steering through the new path. Retain the old steering path as a comparison. Gate: two parameter versions plus base-only requests interleave correctly in one engine; changing windows/routes across replays takes effect; padding and no-op capture work; shared base GEMM count does not grow with tenant count; no `.item()`/`.tolist()`/Python-dependent selection in captured model execution.
3. **Prove useful generality and native/custom interchangeability.** Add norm-scaled steering and a custom low-rank provider using the same definitions as the native primitives. Compare the same low-rank state under learner, native engine and custom engine implementations, including fused projection mapping. Gate: no new type-specific worker or routing branches; input and output dimensions may differ; native and custom providers do not double-apply an operation and can coexist in a mixed batch; fused residual normalization reads the logical stream; unsupported overlapping sites are refused.
4. **Trainable replay.** Implement the common replay/autograd bridge, parameter groups, payload round trips and restore path. Gate: gradients match reference/double-precision small cases for inputs and parameters, training decreases a fixed synthetic objective, tenant updates do not affect neighbors, saved/restored parameters and optimizer state reproduce the existing resume contract. Include the existing architecture and fakes suite.
5. **Decode/cache and declared state.** Add the verified new-K/V staging site and position-addressed request state with lifecycle/validity rules. Gate: full prefill equals chunked prefill plus ordinary decode within declared tolerances; old cache slots are not touched; cached prefixes are separated by version/window/conditioning; request-slot reuse and cancellation clear state; incompatible prefix reuse/speculation is refused until reconstruction/rollback exists. Preemption recomputes requests from recorded tokens rather than restoring an unsafe partial device cache.
6. **TP and performance qualification.** Validate a known supported model/backend at TP=2 and compare sharded results with TP=1. Benchmark eager reference, graph reference and graph Triton on matched workloads: single/many tenants, prefill/decode, one/multiple types, realistic ranks and request lengths. Gate: correctness and lifecycle checks pass before enabling the new path by default; report measured overhead and remaining unsupported cases. Retain native low-rank serving if it is faster.
7. **Complete existing-adapter lowering migration.** Move the remaining adapter families to shared definitions and providers, preserving legacy payload decoders and per-family stochastic/conditioning semantics. Add explicit operation kinds for non-local-site behaviors and retain honest reachability refusals for unimplemented backends. Gate: the inventory contains no adapter using independent semantic definitions on the two sides; compatibility shims are bounded migration tools; native serving continues through the unified compiler; standard state/checkpoint/transfer services remain owned by ADR 0027. Remove the superseded per-type lowering lifecycle contract only after this gate passes.

Stages 1–4 deliver the useful stateless, trainable adapter abstraction through BOTH lowerings, including native LoRA. Stage 5 adds the explicit cache/state features discussed in this chat; Stage 6 qualifies multi-GPU support and performance; Stage 7 completes migration of the full existing adapter surface. Each stage is independently reviewable, and stages 1–4 alone are not completion of the user's full lowering-migration requirement. Removing the old eager requirement is the consequence of passing graph tests, not the first change.

### Validation and retained evidence

Offline tests cover typed declaration errors, source identity, route packing, graph bucket padding, state/slot lifetimes with fake completion events, cache identity, deterministic initialization, serialization, mixed-family replay and unsupported modes. Compile the same definition for learner and engine, round-trip canonical state through provider mappings, and exercise every existing adapter family's migration. GPU tests compare learner reference, native serving where available, eager Triton and graph replay under changing route data, not only a fixed captured batch. Use fixed supplied token sequences and the same recorded draws/windows for parity; compare per-token logits/logprobs rather than requiring sampled answers to match exactly. Test capture with all rows inactive followed by active replay, load/evict/reuse under in-flight holds, concurrent publication, prefill/decode equivalence, sleep/wake and recovery after partial load. Profiler evidence checks base projection count and the absence of per-layer CPU/GPU synchronization. Fake tests do not prove any live property.

Before any GPU experiment, declare and verify the logs actually emitted: exact source/build/program hashes and run identities; model/backend/TP/precision, capacities and graph buckets; real versus padded token/request counts; parameter version/window per request; input token IDs, predicted token IDs/logprobs and reference values for fixed synthetic parity cases; tolerance and per-case errors; GPU memory; warmup/JIT/capture/load/publication and synchronization timings; per-step metadata/base/adapter/attention/total timings sampled with documented cadence; throughput numerators and elapsed-time denominators; errors, retries and worker lifecycle. Separate queue time, GPU execution time and cold-start cost. Use stable case IDs and persist raw bounded measurements alongside summaries.

A training drill additionally retains every loss component and weight, NLL numerator/token denominator, update/timestamp, parameter/checkpoint identity and optimizer/restore evidence, following the standing AGENTS.md logging requirement. Reuse existing ledger/operational artifacts and the observer; do not introduce a new store tree. Persist profiling traces at a bounded, declared cadence so telemetry does not dominate these short kernels. Use synthetic or explicitly unsealed data for parity; do not expose sealed evaluation data.

All live drills use the standing desk with explicit checkpoint/delivery cadence, finite idle release and verified stop. No GPU is started by writing this ADR. The first GPU work is an authorized correctness drill with the model and engine actually targeted, rather than an allocation to inspect the environment.

## Questions

**Q1. Is the initial contract shape-preserving, token-local site transformations, with K/V staging and position-addressed state added after the stateless path?** Recommendation: yes, in the sequence above. This gives a bounded implementation and an honest path to the requested decode behavior. Admitting arbitrary recurrent state, historical KV mutation or shape changes initially adds prefill algorithms, rollback and attention-cache ownership to every new type.

> **Samarth:** Implementation authorized in chat: "this sounds good. could you implement on top of whatever the other chat is doing?" The recommended staged implementation is delegated; this is not an invented verbatim answer to the individual question.

**Q2. For a novel trainable adapter, may the author supply an equivalent differentiable PyTorch reference instead of a custom backward?** Recommendation: yes; use its local autograd backward first and allow an optimized VJP later. Forward-only programs are serving-only. Requiring all authors to write Triton backward is faster in some cases but raises the implementation burden; promising gradients from forward-only Triton would be incorrect.

> **Samarth:** Implementation authorized in chat: "this sounds good. could you implement on top of whatever the other chat is doing?" The recommended staged implementation is delegated; this is not an invented verbatim answer to the individual question.

**Q3. Should requests pin immutable parameter versions and should mutable-state programs initially disable prefix reuse/speculation unless restoration/rollback is available?** Recommendation: yes. Publish into a free slot; all-rank readiness and GPU completion govern visibility/reuse. Crash-mid-load exposes no ready slot, restart reloads the committed bundle and recomputes in-flight request state. The alternative is a transactional cache/state protocol now, including branch copies and rejected speculative writes; silent mid-request parameter mutation is not supported.

> **Samarth:** Implementation authorized in chat: "this sounds good. could you implement on top of whatever the other chat is doing?" The recommended staged implementation is delegated; this is not an invented verbatim answer to the individual question.

**Q4. Migrate all existing lowerings incrementally through the shared definition while keeping native compute and legacy payload compatibility?** The user has settled that native adapters and BOTH lowerings must use the new interface. The remaining rollout recommendation is incremental provider migration with opt-in execution until parity passes, rather than a flag-day rewrite. Hash new definitions/programs and their helper sources explicitly, resolve the narrow overlap with ADR 0023, preserve old checkpoint bytes, and measure numerical rather than bitwise equivalence between different GPU kernels. A flag-day cutover accelerates removing compatibility shims but increases migration risk; keeping native adapters permanently outside the shared interface does not satisfy the requirement.

> **Samarth:** Implementation authorized in chat: "this sounds good. could you implement on top of whatever the other chat is doing?" The recommended staged implementation is delegated; this is not an invented verbatim answer to the individual question.

## Outcome

Proposed implementation plan only. No framework implementation, GPU allocation, scientific experiment, parity claim or performance claim is made by this document. The shared definition, native/custom operation support and migration of BOTH lowerings are settled user requirements; the author signatures, filenames and seven implementation stages are concrete proposals for review. The implementation requires the remaining questions above to be answered and this ADR accepted under CLAUDE.md; that requirement does not block preparing or saving this plan.

## Implementation authorization

> this sounds good. could you implement on top of whatever the other chat is doing?

Implementation starts in isolated worktree `research/rlstack-adapters`, branch `codex/adapter-definitions`, directly atop ADR 0027 implementation commit `483259c`. The other chat's checkout remains untouched. Live GPU qualification remains separately reported; acceptance does not claim that any gate has already passed.
