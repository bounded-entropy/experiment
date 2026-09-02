# rlstack architecture — the vocabulary

The universal reference: every noun and verb the codebase uses, defined once,
in the order a reader needs them. This document **defines existing vocabulary
and never redefines semantics** — where it and the canon disagree, the canon
wins: `agent-context/rl-stack-spec.md` (invariants I1–I12) plus the latest
entry of `agent-context/CONTEXT.md` (the chronological decision log, which
holds the history this document deliberately omits).

Docstrings across `rlstack/`, `rlstack_engine/` and `deploy/` speak this
vocabulary. If a word here appears in code with a different meaning, that is a
finding.

---

## The system in two paragraphs

An **experiment** is one value — an `ExperimentSpec` — and its identity is a
hash of that value plus the source of every registered thing it names. It is
**submitted** to a **host**, which owns a slice of GPU. Many experiments share
one host: each is a **tenant**, all of them served by the same resident engine
and the same resident learner, none of them able to disturb another (I8). What
runs where is decided at submit time and changes wall-clock, never results
(I5).

Inside a run there are two worlds and one membrane (I1). The **inference
world** is interactive: an environment drives an episode against a **pool**,
producing a mutable **Rollout**, which the runner **seals** into a frozen
**Trajectory**. The **training world** consumes only sealed data: a
**postprocessor** pipeline scores it into **postdata** columns, and a **loss**
— pure math over named columns (I9) — turns those into a gradient. The only
thing crossing back is a compiled **bundle**. Both directions meet at the
**store**: the run directory is the single source of truth, the **ledger** is
the commit bit, and daemons synchronize through it and nothing else.

---

## Nouns

### The contract

**Spec** — a frozen dataclass of declarative values. A spec never does setup
and never touches a GPU; it *is* identity (I3). `ExperimentSpec` is the whole
experiment as one value, composed of `PolicySpec` / `GenSpec` /
`TrajectorySource` / `AlgoSpec` / `GpuConfig` / `Seeds` / `WarmStart`.
A run's identity is its TRAINING loop: measurement is not in the spec (#70).
`rlstack/spec/specs.py`

**Experiment** — one spec, resolved and run. Also the unit of identity, of
store ownership (I10), and of a run directory.
`rlstack/spec/specs.py`, `rlstack/runner/loop.py`

**Tenant** — an experiment *from the metal's point of view*: one occupant of a
shared engine and a shared learner, keyed by its `run_id`. Every Learner verb
pins a tenant; every engine request pins a bundle (I8).
`rlstack/runner/host.py` (`Tenancy`), `rlstack/runner/learners/torch_learner.py`

**Run identity / `run_id`** — `h(spec ⊕ registered-code hashes ⊕ data
fingerprint)`, computed at Phase 0 and never typed by a human (I3). Editing a
registered function's body changes it; renaming a file does not.
`rlstack/spec/canonical.py`, `rlstack/registry.py` (`code_hashes`)

**Registry** — one table per swappable kind of thing (`ENVS`, `POST`,
`LOSSES`, `ADAPTER_TYPES`), filled at import, each entry a typed record pairing a
**declaration** half with a **compute** half (I4). A name exists iff the module
defining it was imported.
`rlstack/registry.py`

**Seeds / the seed tree** — every random draw in a run is `h(master, *path)`
where the path names what the seed is for. No module touches global RNG state,
so any part of a run can be regenerated in isolation.
`rlstack/runner/seeds.py`, `rlstack/spec/specs.py` (`Seeds`)

**Determinism / resume-equivalence** — on fakes, two runs of one spec produce
byte-identical run directories, and a killed-and-resumed run equals a straight
one. The property is what the seed tree, the commit protocol and
order-independent reductions exist to protect. On real metal the equivalent
signal is the ledger's `logprob_gap` staying at the kernel floor.
`tests/test_resume.py`, `rlstack/runner/fakes.py`

**Schedule** — the wave-shape and estimator knobs: `group_size`,
`trajectories_per_wave`, `n_updates`, `epochs_per_wave`, `microbatch_tokens`
(the one engineering knob), and `max_policy_lag` — the lag *buffer*, how stale
a behavior policy the trainer tolerates (0 = strict alternation).
`rlstack/spec/specs.py`

**Flow graph** — THE canonical walk over a spec's data declarations: nodes are
every named artifact a run will contain, edges are `produces` / `consumes` /
`requires` verbatim, and `feeds_loss` is transitive reachability into the
loss's requires. The submit gate's pipeline checks and the run's
`dictionary.json` are two consumers of this one walk. Each bank `provides` gets
TWO nodes: the forward one (the tensor a loss may require) and its **stat
twin** — the per-update float the Trainer means into the ledger — emitted
unconditionally, so observability-only provides describe themselves.
`rlstack/spec/flow.py`

**dictionary.json** — the flow graph serialized into the run directory at
creation: every column the run will contain, its producer, consumers, phase,
granularity, and whether it feeds the loss. Derived, never identity (I11). A UI
renders a run from its own dictionary — no registry, no version skew.
`rlstack/spec/flow.py`, `rlstack/data/stores/base.py`

**Submit gate** — Phase-0 joint validation: one named function per rule, listed
in `CHECKS` in spec order, returning every issue rather than the first.
Everything checkable before a GPU is touched is checked here.
`rlstack/spec/validate.py`

### The bridge (the policy)

**Base** — the pretrained checkpoint, named by HF id plus pinned revision. One
frozen base is loaded per learner and per engine build, and is shared by every
tenant.
`rlstack/spec/specs.py` (`PolicySpec.base`)

**Policy** — a base plus a **bank**. The only primitive that lives in both
worlds (I2), and therefore the only one carrying a parity obligation.
`rlstack/spec/specs.py` (`PolicySpec`), `rlstack/policy/`

**Bank** — the named map of interventions on one base: `{name: AdapterSpec}`.
Names are the user's; one delta per site is the rule; a bundle pins the whole
bank's version map.
`rlstack/spec/specs.py` (`PolicySpec.bank`)

**Adapter** — **a configured bank entry**: `AdapterSpec(adapter_type, site,
init, trainable)`, one typed intervention at one site pattern. "The adapter
named `pi`" means this.
`rlstack/spec/specs.py` (`AdapterSpec`)

**Adapter type** — **a registered class**: the thing `@adapter_type("lora")`
puts in the `ADAPTER_TYPES` registry, which `AdapterSpec.adapter_type` names by
string. An adapter type owns the declaration (`serving`, `provides`, `records`,
`site_ok`, `exports`) and both lowerings; an adapter is one *use* of an adapter
type.
`rlstack/policy/adapters/base.py`, `rlstack/policy/adapters/{lora,plora,soft_prompt,attn_bias,value_head}.py`

**records / provides** — the two halves of the same mirror. `records` are
sampling-time FACTS, frozen at the seal and never recomputable (per token in
`Turn.token_extras`, per request in `Turn.turn_extras`); `provides` are
training-time TENSORS, recomputed by every forward. `provides` is not only the
loss-input channel: every declared provide is also summarized per update into
the ledger and described in `dictionary.json`, so an adapter type declares
everything a reader should WATCH, and a provide nothing requires is
first-class.
`rlstack/policy/adapters/base.py`

**plora** — a probabilistic low-rank delta: each matched weight's top-k
singular directions are frozen (`U_k`, `A = Σ_k V_kᵀ`, a content-addressed
factors artifact), a hypernet maps a latent `z` to a k×k core per site, and the
delta is `U_k C_s A` — a rank-k LoRA served through punica. The latent carries
a posterior against a prior, so the run learns a DISTRIBUTION over adapters;
the engine serves one version as an ensemble of drawn members plus the mean,
records the noise it drew, and replay reparameterizes it against the current
posterior. `basis="random"` is the frame control: the same top-k singular
VALUES steering seeded random orthonormal directions — the arms differ in the
basis and nothing else.
`rlstack/policy/adapters/{plora,plora_torch,plora_vllm,plora_factors}.py`

**spectral / spectral_latent** — SVF: one trainable gain per singular
direction of each matched weight, `W' = U diag(σ·(1+δ)) Vᵀ` — no frame is
chosen, no direction invented, and the whole spectrum can move. The served
delta is the top-k directions by |σ·δ| (a rank-k punica adapter whose frame
CHANGES as learning does), with a straight-through backward so unselected
directions keep competing; the dense gains ride the payload for resume, and
no factors artifact exists — the trainer recomputes the full SVD from the
weight it holds at install. `spectral_latent` is the plora-style twin (gains
GENERATED from a latent; posterior, KL, recorded draw, member ensemble) —
the "does the latent help" ablation. Both provide `latent_kl`'s family:
plora and spectral_latent share that channel name, which is what lets
`grpo_latent_kl_gated` price either.
`rlstack/policy/adapters/{spectral,spectral_torch,spectral_vllm,spectral_latent,spectral_latent_torch,spectral_latent_vllm}.py`

**TaskMaker / Derive** — mint-then-make, the third leaf: a plan may name an
episode whose task does not exist yet — `Derive(source, maker, env)` resolves
`source` (Replay's ref grammar) to a sealed trajectory and a registered
`@task_maker` derives the new task, PURELY, so resume re-mints to the byte.
Declared in `GenSpec.makers`, hashed like an environment. The reflect loop is
its first use: the maker re-serves the whole transcript plus "what went
wrong", the `reflect_retry` env samples the critique and the retry, and the
`sdpo` loss clones each document's final turn (read off `segment_ids` — no
new masking primitive). An un-referenced rollout is due when the first later
referenced one is — the generator's intermediate-wave pacing rule.
`rlstack/data/plan.py`, `rlstack/inference/makers/`, `rlstack/inference/environments/reflect_retry.py`, `rlstack/training/losses/sdpo.py`

**Site** — a canonical attachment point named by the checkpoint's own module
path, resolved at Phase 0 against `site_space` = the schema ∪ every bank
entry's exports. `AdapterSpec.site` hashes into identity, so the pattern
grammar is canon and cannot vary by backend.
`rlstack/policy/siteschema.py`

**SiteMeta / SiteSchema** — one site's frozen metadata (`name`, `path`,
`has_weight`, `shape`, `is_boundary`) and the whole per-base collection of
them. A schema is a pure function of one checkpoint, built by a compiler
(`hf_schema`, `fake_qwen_schema`) and inert afterwards. It deliberately knows
nothing about engine reachability or adapter-created sites.
`rlstack/policy/siteschema.py`

**Mechanism / lever** — the closed set of ways an engine can reach a site:
`punica`, `prompt_embeds`, `logits`, `side_attention`, `none`. A *lever* is the
same thing named from the build's side — something an engine build pays for and
then has. Native levers are the engine's own; `side_attention` is ours, shipped
as a plugin.
`rlstack/policy/adapters/base.py` (`Mechanism`)

> **Why the set is closed.** A mechanism is not a label — it is the
> within-batch compute that applies many tenants' state inside one fused
> forward, and someone must build it. Three tiers: an adapter type that
> compiles to an existing mechanism's state shape rides for free (lora on
> `punica` — vLLM's own segmented kernels); one whose effect lives at a
> per-request point of the graph needs no batched compute at all
> (`prompt_embeds`, `logits` — the input and sampling boundaries are already
> per-sequence); one that needs
> per-tenant compute *inside* the fused forward must ship a NEW mechanism as
> an engine plugin, re-earning the per-token → tenant index mapping the
> trainer's row plan gets for free (`side_attention` is the standing example,
> refused until its mechanism exists). The lowerings select *into* a
> mechanism; they never create one.

**Lowering** — how one adapter type's math is realized on one side of the
bridge. Every adapter type ships two. A **rollout lowering** serves it through
an engine build (`<adapter_type>_vllm.py`, contract in `rollout.py`); a **replay
lowering** wires it into a trainer forward (`<adapter_type>_torch.py`, entered
through `install_replay`). A
replay lowering may be a module replacement *or* a boundary around the base's
forward; either way its obligation is alignment.
`rlstack/policy/adapters/rollout.py`, `rlstack/policy/adapters/replay.py`

**Bundle** — a policy version as servable content: the full version map for the
bank plus payloads for the servable deltas only. Content-addressed, so identical
banks compile to identical ids everywhere. `Bundle.pin(id, versions)` is the
payload-less address a request carries. The bundle is the ONLY data channel
from the training world back to the inference world (I2).
`rlstack/policy/compile.py`

**SiteWrapper / the chain** — the shared half of every module-replacing
replay site: one wrapper per (family, path), and NESTING when different
adapter families claim one path on a shared learner — each family's forward
applies only rows whose routed state is its own and passes the rest through
to `inner` (`join_site` / `leave_site` find and splice a family's wrapper
anywhere in the chain). A tenant beside a foreign family is bit-identical to
the same tenant alone.
`rlstack/policy/adapters/replay.py`

**Slot / row plan** — the trainer's routing unit. A **slot** is one tenant's
installed deltas at a set of sites; a **RowPlan** says which slot each row of
one padded microbatch carries, and raises if a lowering runs unrouted. The
trainer-side twin of punica's per-token adapter index (I8). `ReplayRows.facts`
is the other half: row r's RECORDED turn extras, threaded adapter-blind by the
learner, for a lowering whose math depends on a draw the rollout already made.
`rlstack/policy/adapters/replay.py`

**Engine plugin** — a serving mechanism the stock engine lacks, shipped in the
engine image and named from an adapter type by string only. It must re-earn per-request
selection, cache correctness and parity at seams the engine never promised to
keep stable.
`rlstack_engine/plugin.py`, `rlstack_engine/side_attention.py`

**Parity certificate** — the numerical exam binding an adapter type's two lowerings,
keyed by build fingerprint so a version bump re-runs it (I7). Designed and
unwired; the running parity mechanism is the per-update `logprob_gap` rail.
`rlstack_engine/certificates.py`

### The data objects

**Task** — one problem drawn from a content-addressed task file: id, prompt,
and metadata a verifier or a hint reads.
`rlstack/data/trajectory.py`

**Rollout** — one episode *in progress*: mutable, inference-world, built by an
environment driving sample calls. Nothing on the training side ever sees one.
`rlstack/inference/rollout.py`

**Seal** — `Rollout.seal() -> Trajectory`. The membrane, enforced by the type
system: mutable inference record in, frozen training record out (I1). The
runner seals; environments never do.
`rlstack/inference/rollout.py`, `rlstack/runner/traffic.py` (`run_episode`)

**Trajectory** — one sealed episode: frozen training data. Recording is
loss-independent and happens at the seal (I6) — the engine's own token ids
(never re-tokenized), behavior logprobs, the pinned bundle id and policy
version, seeds, finish reasons, and any per-token facts an adapter type recorded.
`rlstack/data/trajectory.py`

**Turn** — one request inside a trajectory: token ids, behavior logprobs,
finish reason, the bundle id and policy version pinned at submission, the seed,
and the per-token/per-turn extras.
`rlstack/data/trajectory.py`

**Group** — one partial loss contribution (a GRPO group, a preference pair) and
the scope a postprocessor sees. Group keys are ASSIGNED at wave assembly, not
derived from tasks.
`rlstack/data/trajectory.py`, `rlstack/runner/traffic.py` (`collect_wave`)

**Wave** — the data of exactly one update: a tuple of groups, serialized as
`waves/<update>.jsonl.gz`.
`rlstack/data/trajectory.py`

**Flat / TokenBatch** — the packed forms. `flatten` turns one trajectory into a
complete flat token record; `broadcast` turns a per-trajectory column into a
per-token channel; `pack` fills microbatches bounded by `microbatch_tokens`. A
`TokenBatch` carries token ids, loss mask, behavior logprobs, segment ids, doc
starts, the per-token `postdata` columns and token extras, the per-DOCUMENT
turn extras (`doc_turn_extras`, the per-request recording channel at microbatch
scope), and `microbatches_in_update` — how many microbatches this one belongs
to, which a per-update term divides by — and nothing estimator-shaped.
`rlstack/data/flatten.py`

### The training world

**Postprocessor** — everything computed ABOUT sealed trajectories, declared as
a class and run per group after the seal and before the loss: rewards, judge
scores, advantages, teacher logprobs. It declares `produces` / `consumes` /
`token_level` / `pools` / `sampling` and implements `process`. Anything that
needs a GPU is a postprocessor's job (I9).
`rlstack/training/post/base.py`

**Postdata** — the pipeline's columns for one wave, stored beside it as
`postdata/<update>.json` — never inside the sealed record (I6). Columnar,
aligned to wave order. A **part** (`postdata/<update>.<producer>.json`) is one
producer's share of those columns, written ahead of the merged file by the
Scorer; the Trainer merges every part into the one file readers see, and attach
sweeps parts whose update never committed, exactly as it sweeps the rest.
`rlstack/data/stores/base.py`, `rlstack/runner/post.py`

**token_level** — the declaration marking produced columns as per-token vectors
(one float per generated token per trajectory, in sealed order) rather than
per-trajectory scalars. THE channel for teacher and hinted logprobs.
`rlstack/training/post/base.py`

**Loss** — pure math: `fn(PolicyOutputs, TokenBatch) -> LossResult`, with
`requires` naming DATA COLUMNS only (postdata ∪ recorded facts ∪ bank-provided
forward tensors), enforced at registration. A loss can never cause metal work
(I9).
`rlstack/training/losses/base.py`, one file per objective beside it

**Rails** — the two numbers every loss returns beside its value: `mean_ratio`
and `logprob_gap`. The gap is the trainer/sampler mismatch alarm — its floor is
the bf16 kernel difference, and GROWTH above that floor is the signal.
`rlstack/training/losses/base.py`

### The metal

**Fleet** — the AGGREGATE of every standing host currently listed with the desk,
across all metal. Not a class and not a process: a collective noun for what the
fleet journal describes (`read_fleet_log` / `append_fleet_event`), which the Desk
is the one writer of. The *service* is the **Desk**, the *record* is the journal,
the *members* are Listings — "fleet" names none of those three on its own.
`rlstack/runner/desk.py` (the journal), `rlstack/observe/host_series.py` (the view)

**Metal** — registered owned hardware the fleet may carve: a name, a GPU kind, a
device count, and one device's VRAM. Registering Metal *is* the acquire rung
executed.
`rlstack/runner/desk.py`

**GpuSet** — pure device demand inside a spec (`n`, `nodes`, optional literal
ids). Demand, never a provider name — the spec says what, placement says where
(I5).
`rlstack/spec/specs.py`

**Partition** — the irreducible carved share of metal a host is born onto:
`metal` (the registered Metal's NAME it was carved from — never a GpuSet, which
is demand), the GPU kind, the device indices, and the memory fraction owned on
each. Enforced PER RESIDENT PROCESS (ADR 0002): the devices become that
process's CUDA_VISIBLE_DEVICES, the fraction is vLLM's reservation for an
engine and torch's allocator cap for a learner. Memory partitions honestly; SMs
still time-share across partition boundaries, which is a stated cost, not a
hidden one.
`rlstack/runner/host.py`

**Regime** — one `capability` a host can wear: inference (an engine built at
some `tp`) or training (a learner built at some `fsdp`) over one base. The fleet
matches joins against regimes; the host attests its metal against them at birth.
`rlstack/runner/host.py`

**Host** — an ATOMIC PURPOSED PARTITION: a Partition plus its Regimes, attested
at construction and never grown or reshaped (I12). Host and Partition are 1:1
and deliberately two words: the Partition is the RESOURCE fact (what share of
what metal), the Host is the SERVICE wearing it (arbiter, engines/learner,
roster) — the fleet plane speaks partitions, the workload plane speaks hosts.
It owns its engines, at most
ONE multi-tenant learner, its arbiter, and its journal store — and since ADR
0002 those engines and that learner are PROCESSES (residents) it holds as
proxies (`RemotePool`, `RemoteLearner`); the Host itself is the DOOR — arbiter,
roster, runner, journal — in the metal process. One regime =
dedicated; several = it ALTERNATES them on its own arbiter group, one host
wearing masks rather than two hosts coordinating. `solo` is one more birth
fact: this partition serves ONE experiment at a time — I8 promises tenants
cannot disturb each other's RESULTS, never their THROUGHPUT.
`rlstack/runner/host.py`

**Residual** — capacity no partition owns. A carve draws from residual only,
which is what makes carving automatic: it can never shrink or reshape a living
host.
`rlstack/runner/desk.py`

**Demand** — one member's capability need as a value: capability, base, shard
shape, the fraction as a carve hint, and the **anchor** flag (where a delivered
frame lands). The desk's whole input vocabulary; `demand_rows`/`demands_from`
are its wire codec. A spec becomes demands in the CAMPAIGN layer, never at the
desk.
`rlstack/runner/desk.py` (`Demand`), `rlstack/runner/campaign.py` (`demands_of`)

**Subdir / filing** — where a run's directory spawns
(runs/<subdir>/<run_id>), indicated at submit and fixed for life. Filing,
never identity: it does not hash, attach finds the run wherever it lives, and
the observer renders the tree (a FOLDER is which store; a SUBDIR is filing
inside one).
`rlstack/data/stores/base.py` (`run_prefix`, `check_subdir`)

**Desk** — placement as a service and the fleet journal's one writer (the
Trainer/ledger pattern on the fleet plane). WORKLOAD-BLIND: its vocabulary is
Demands, listings, metal and addresses; it imports no spec class. It holds
**Listings** — descriptions of standing hosts (regimes, address, solo, and
the capacity VIEW: partition row + metal name), journaled and rebuilt by
`from_journal`. Two doors: `place(demands)` answers addresses (the PURE
CLIENT's door — an evaluator joins a pool and is thereafter just admitted
traffic), and `submit(demands, frame)` additionally DELIVERS the opaque frame
to the anchor demand's host with every other pool's address threaded as
routes — routes come off the demand rows, which is what makes the blind relay
possible. A placement no listing serves becomes a desk-issued CARVE on a
registered metal (deduce by residual, command by RemoteMetal.carve); only
what no metal holds returns boot instructions — the standing acquire, a
human's. `reap` is the janitor: probe, retry (the knock is the restart on a
lazy venue), decarve + delist(reason) what stays silent.
`rlstack/runner/desk.py` (`Desk`, `Listing`), `rlstack/runner/remote.py` (`RemoteDesk`)

**Campaign layer** — where SPECS meet the fleet, the only such place:
`demands_of(spec)` (anchor on the learner — the learner is never remote, said
once, here), `frame_for(spec)` (canonical row + the client's code claim), and
`Campaigns` — the desk's spec-aware sidecar serving the passes that need both
a store and spec knowledge (today `migrate`, the code-refresh warm-fork) over
the desk's own Transport contract. A campaign's whole surface is
`RemoteDesk(transport).submit(spec)` (shaping happens client-side) or
`.resolve(demands)` for a pure client; no venue word appears in any of it.
`rlstack/runner/campaign.py`

**MetalService / the metal plane** — the metal-side end of the standing
carve: the container that owns a device wears it by default. One registered
Metal's BOOKS (built partitions — hand-built hosts enter via `adopt_born` —
plus pending bookings), and the verbs that create and free hosts: `carve`
(BOOKS its fraction synchronously before the build's first await, so carves
never double-promise; a failed build releases), `decarve` (every resident down
the ladder, the fraction returns to residual), `residual` / `describe` (the
desk's deduction feed). It holds a RECIPE (**Builds**) and a carve spawns one
resident process per regime from it; a resident's unbidden exit decarves its
host. The desk DEDUCES, the metal ENFORCES; the metal writes nothing to the
fleet journal.
`rlstack/runner/desk.py` (`MetalService`), `rlstack/runner/remote.py` (`RemoteMetal`)

**Builds / EngineBuild / LearnerBuild** — a metal's recipe: the capacity knobs
a partition cannot tell you (an engine's context length, bundle count, rank,
ensemble width, served adapter types, sleep mode; a learner's dtype, clip,
activation checkpointing). Everything ELSE about a resident — class, base,
width, device, fraction — follows from (regime, partition) in `build_engine` /
`build_learner`, so building is rlstack's, not a venue's. Declared at bring-up
from the deploy's constants, journaled on the `metal` registration and every
`host-up`, so a restarted container carves the same residents unattended.
`FakeEngineBuild` / `FakeLearnerBuild` put the fakes in a real process.
`rlstack/runner/residents.py`

**StoreAddress / open_store** — a store as a value a child process can reopen
it from: backend by name, root, locator. `Store.address()` on every backend;
`open_store` is the one place backends are spelled. A volume store reopens as a
mount-only view — a resident reads cas blobs and writes nothing.
`rlstack/data/stores/base.py`, `rlstack/data/stores/address.py`

**Pool** — a NAME traffic routes to, with two lives: declared capacity
(`PoolMember` in a `GpuConfig`) and a runtime routing entry (`Routes`: pool name
→ engine + pinned bundle). The name↔metal relation is many-to-many — one engine
may back many pool names, one pool may fan over several engines.
`rlstack/spec/specs.py` (`PoolMember`), `rlstack/runner/traffic.py`

**Engine** — inference metal: an object satisfying the Engine protocol
(`VllmEngine`, `FakeEngine`, `RemotePool`). It carries `base` and `tp` as BUILD
facts, speaks TOKENS, and pins each request's bundle at submission.
`rlstack/runner/interfaces.py`, `rlstack/runner/engines/vllm_engine.py`

**Learner** — training metal: differentiable forward/backward plus the
optimizer, multi-tenant by additive install (`TorchLearner`,
`FsdpTorchLearner`, `FakeLearner`). `fsdp` is a BUILD fact. The learner is
never remote ACROSS HOSTS — the runner goes to it — and since ADR 0002 it is a
process INSIDE its host, reached through `RemoteLearner` over its door. Its
`install` takes a **Parameterization** and its package imports no spec class.
`rlstack/runner/interfaces.py`, `rlstack/runner/learners/`

**Parameterization** — what `install` builds, projected off the spec BY THE
RUNNER (`loop.parameterization_of`, the one place a spec becomes an install):
the base, the loss by REGISTRY KEY (bound at install — the moments belong to
one loss), the bank entries in install order (adapter type by key, init with
the per-entry seed already derived, trainable, resolved sites), and the
optimizer settings. The whole of what a learner may know about an experiment.
`rlstack/runner/interfaces.py`

**Rank chorus** — the extra processes a sharded learner build needs and no
more. Rank 0 IS the learner resident and answers its door; ranks 1..width-1
exist only to stand in the collectives, and rank 0's copy is the truth. The
resident's pin makes "rank r on cuda:r" true by construction, and every
follower caps its own device. The teardown ladder is `residents.py`'s.
`rlstack/runner/learners/ranks.py`

**Arbiter** — the physical half of the blackboard: a `GpuArbiter` constructed by
whoever owns the metal (a Host makes its own unless handed one) and shared by
every experiment admitted to it. It governs its owner's partition, not the
device — several sub-GPU hosts on one device each admit independently.
Scheduling policy lives here and is deliberately outside run identity (I5).
`rlstack/runner/arbiter.py`

**Resident** — something that occupies evictable GPU memory (an engine, a
learner), keyed by OBJECT IDENTITY: ten pools backed by one engine are ONE
resident. Since ADR 0002 a resident IS A PROCESS: a supervised child of the
metal, born pinned to its partition's devices and capped at its fraction,
built by rlstack's universal builders from the metal's recipe, answering
JSON frames at a DOOR (`hello`, `sleep`/`wake`, `stop`, plus its Engine or
Learner verbs). The Host holds it as a proxy, and the proxy is the object the
arbiter keys. A resident that exits unbidden is a dead host.
`rlstack/runner/arbiter.py`, `rlstack/runner/residents.py`

**Exclusive group / admission** — an arbiter group (from `GpuGroup.sharing =
"sleep"`, or a host's own group) inside which exactly one resident is live at a
time; **admission** is entering `admit(resident)`, which guarantees residency,
waking and evicting per policy. Everything outside a group co-resides and
admission is a plain counter. Alternation is about memory, never mutual
exclusion on work.
`rlstack/runner/arbiter.py`

**Wire** — pool traffic to a host that is not this process. `HostService` is
the host-side end (it executes pool verbs on its own metal under its own
arbiter — admission stays with the partition); a `Transport` carries JSON-safe
dict frames; `RemotePool` implements the whole Engine protocol over it, so the
runner cannot tell remote from local.
`rlstack/runner/remote.py`, `deploy/modal_host.py` (`ModalTransport`)

### The store

**Store** — one abstract key tree with all the orchestration (attach-or-create,
the append-only ledger, crash recovery, retention) written against seven
abstract byte verbs; a backend implements only the verbs.
`rlstack/data/stores/base.py`, `local.py`, `modal_volume.py`

**Run store** — the per-experiment binding: one experiment, one store, for life
(I10). `run_id` is global but existence is store-scoped, so the same spec
against two stores forks history silently — which the observer flags rather
than prevents.
`rlstack/runner/host.py` (`submit`), `rlstack/observe/views.py`

**Ledger** — `ledger.jsonl`, append-only and strictly increasing: the commit
record and the commit bus. The Trainer is its only writer. Everything written
before an update's ledger line is UNSEALED and is discarded on attach.
`rlstack/data/stores/base.py`

**Commit** — the ledger append that seals one update. It is the durability bit
the whole blackboard is ordered around; kill -9 at any other point loses only
work that regenerates.
`rlstack/runner/daemons/trainer.py`

**Retention** — what a run's store may forget, as a policy class: a pure
function of the ledger naming expendable blob versions — a versioned blob
under `adapters/` or `optim/` and nothing else, so the append-only guards are
out of reach by construction. The default, `KeepRestorable`, keeps every
adapter (restore pins historical versions forever) and only the ledger tail's
optimizer moments (`restore_tenant` reads nothing else). Retention changes
what is RECOVERABLE, never what was COMPUTED: nothing about it is hashed,
journaled, or written to a run directory.
`rlstack/data/stores/retention.py`

**CAS** — `cas/<sha256>/blob`: content-addressed objects, how task files and
static trajectory datasets are named (`cas://<sha>/...`).
`rlstack/data/stores/base.py`

**Journal** — `hosts/<name>/log.jsonl` and `fleet/log.jsonl`: append-only
observability. Placement, boots, tenancies, gpu samples, traffic windows,
update timings and carves land here, deliberately outside run manifests so
placement stays out of identity. Correctness never reads a journal; torn tails
are tolerated.
`rlstack/data/stores/base.py`, `rlstack/runner/host.py`, `rlstack/runner/desk.py`

**Emission plane** — the measurement side of observability: a `TrafficMeter`
the host's engines and arbiter count into (prefill and decode tokens, time to
first token, admission wait, in-flight), drained once per stats tick into one
windowed `traffic` event, and an `UpdateClock` the Trainer laps at its four
phase boundaries into one `update` event per commit. Every number is wall
clock, so every number lives in a host journal and none may enter a run
directory.
`rlstack/runner/meters.py`, `rlstack/observe/host_series.py`

**Peek** — a read-only store read (`peek_manifest`, `peek_ledger`,
`peek_dictionary`, `peek_eval_summaries`). Observers must never `open_run`:
attach sweeps unsealed work, which would corrupt a live run (I10).
`rlstack/data/stores/base.py`

**Observer** — read-only derivations over stores and journals: never attaches,
never writes, imports the data layer and nothing else. The CLI is text over it,
the UI is JSON over the same `*_data` functions, and both render a run from its
own `dictionary.json`.
`rlstack/observe/`, `rlstack/__main__.py`

### The runtime

**Blackboard** — Phase 2's shape: daemons synchronized ONLY through the store.
Nobody calls anybody; the ledger is the commit bus and `waves/` the data bus.
The *logical* half is awaitable predicates over the store; the *physical* half
is the arbiter.
`rlstack/runner/loop.py`, `rlstack/runner/signals.py`

**Daemon** — one GPU responsibility, four beats: await its condition, admit the
residents its work occupies, do the work, write the store and notify. The
**Generator** samples waves at the newest committed bundle within the lag
buffer; the **Scorer** runs the pooled half of the post pipeline beside the
pools it addresses and writes it as a postdata part; the **Trainer** runs the
inline half + gradient + commit and is the ledger's only writer. Each
condition method is a named, overridable seam. (The Evaluator retired in #70:
measurement left the run.)
`rlstack/runner/daemons/`

**Resident / daemon** — the two kinds of runtime thing, named: a RESIDENT is
the heavy object living on a partition (an Engine, a Learner — nouns of
capability), a DAEMON is the thin loop that watches the store and pokes a
resident (Generator, Trainer, Scorer — agent nouns). Daemons synchronize
through the store ONLY, so colocation with their resident is a transport
choice, not architecture. The Trainer sits beside its Learner because the
runner goes to the learner's HOST; inside that host the Learner is a process
behind five verbs (ADR 0002) — the autograd arc is the Learner's, the seal is
the Trainer's, and neither crosses a wire: the verbs' bytes do.

**Measurement** — observation OUTSIDE the run (#70): a value naming held-out
task ids, samples, a cadence, a scoring pipeline and its own seed, written as
measurements/<run_id>/<name>/ (write-once manifest + append-only points) by
`measure_run` — one idempotent pass any process can run against a pool, backfilling
every missing point (retention keeps every adapter version restorable) and
following the ledger's future. Never the run dir, never identity, never
resume-equivalence; deletable, though a changed observation is a NEW name.
Swapping what a run is measured on, mid-run, is a non-event.
`rlstack/runner/measure.py`, `rlstack/data/stores/base.py` (measurements)

**The split rule** — a postprocessor declaring `pools` is SCORER-RUN, a
pool-less one is TRAINER-INLINE: sending traffic is what makes a processor
slow, so the same declaration that names the traffic names the daemon. The two
halves meet once, at the part, so the gate refuses a pooled processor consuming
an inline one's column. A pipeline with no pooled half plans no Scorer.
`rlstack/spec/flow.py` (`split_pipeline`)

**WaveFeed / source** — where the trainer's rows come from: `live` (this run's
own Generator), `replay` (another run's sealed waves), `static` (a
content-addressed trajectory file). All three make update `u`'s rows exist in
THIS run's `waves/` and hand them back, so the trainer never knows which it has
(I1).
`rlstack/runner/sources/`

**Traffic** — what travels to a pool: **sample** traffic (a token stream
assembled into a Turn) and **score** traffic (logprobs of given tokens, one
prefill pass). A request is traffic, addressed to a POOL, served by whichever
ENGINE backs that name, under a pinned BUNDLE.
`rlstack/runner/traffic.py`, `rlstack/client.py`

**PoolClient** — the neutral interface both worlds type against: environments
sample during rollouts, postprocessors sample or score after the seal, and
`pool(name)` reaches any declared pool. Every client for one episode shares one
seed sequence.
`rlstack/client.py` (protocol), `rlstack/runner/traffic.py` (`EnginePoolClient`)

---

## Verbs, by contract

### The rollout lowering (`rlstack/policy/adapters/rollout.py`)

One `RolloutLowering` per (adapter type, engine build). The engine that owns
these is a BUS: it loops the bundle's adapter types, calls the verbs, merges the
levers and sums the alignments — "native vs plugin" is a `demands()` difference
and nothing else.

- **demands** — what the BUILD must pay before this adapter type can be served
  (engine args, a plugin's presence). A build that cannot pay says so at
  construction, not at the first request.
- **attach** — make one bundle's state resident: payloads → the object `apply`
  and `align` read. Additive, idempotent, once per bundle.
- **apply** — contribute to ONE unit of work, a request: the prompt form and
  the keywords that pin this bundle.
- **align** — how many prompt positions this adapter type's state occupies,
  SUMMED across a bundle's adapter types, so an answer read off the prompt is
  found where the real tokens start.
- **reaches** — does the payment above buy this site? The engine's reachability
  inventory is the union of its served adapter types' answers.
- **claims** — a class-level declaration, not a verb: which request parts
  `apply` writes (the prompt form, or a generate keyword), so two adapter types
  claiming one lever are refused at `add_bundle` while the bundle is still just
  an id.

### The adapter type (`rlstack/policy/adapters/base.py`)

The declaration half is class attributes (`serving`, `engine_plugin`,
`provides`, `records`) plus `site_ok` and `exports`. The compute half:

- **params** — build the trainable parameterization for the matched sites.
- **provide** — the compute half of `provides`: the training-forward tensors,
  recomputed each pass, merged into `PolicyOutputs.provided` under the declared
  names and summarized to one float each (0-dim is its value, anything else its
  mean) for the update's ledger line.
- **param_groups** — named optimizer groups for one bank entry; `""` is the
  whole entry (the default), and `OptimSpec.overrides` addresses them by
  `entry` or `entry.group`, the dotted form winning.
- **install_replay** — wire the replay lowering into the trainer forward.
  Additive: every installed tenant stays wired (I8).
- **uninstall_replay** — its exact inverse. Install is additive, so without the
  inverse a tenant could never be REMOVED: an adapter type lacking it cannot
  share a multi-tenant learner.
- **emit** — lower params into the bundle payload the engine-side consumer
  reads.
- **load** — emit's inverse, in place; resume and warm start walk through here.
- **parity** — the mandatory rollout/replay numerical exam (declared, unwired).
- **rollout_lowering** — build this adapter type's serving half for one engine
  build; `install_replay`'s twin.

### The fleet ladder (`rlstack/runner/desk.py`)

One currency and one decider per rung (I12):

- **join** — a listing already serves the demanded capability (`covers`: the
  ONE coverage rule — capability, base, shape equality); automatic, and the
  target host's own arbiter is the decider. Declared fractions are ignored:
  the weights already live there.
- **carve** — nothing serves it but a registered metal's residual fits, so
  the desk commands the metal to partition a new host into existence. The
  metal BOOKS before it builds (MetalService), so carves never double-promise;
  journaled and listed at the desk, the single writer.
- **acquire / boot** — nothing fits. New metal costs money, so the desk
  answers with boot instructions and a human executes them.
- **decommission** — carve's inverse, client-asked: decarve at the host's
  metal (engine down, fraction back to residual) plus delist, one desk verb.
  Refused with the running work NAMED when anything lives on or routes
  through the host (the guard is DEPENDENTS off the journaled placements —
  a serve host's roster is empty, so occupancy alone would lie); `force`
  tears down anyway, and `reroute` MOVES the dependents first. Delist alone
  stays bookkeeping-only; `reap` is the same composition applied to hosts
  that stopped answering.
- **reroute** — a move is a restart: every delivery is ARCHIVED (demand rows +
  opaque frame ride the journaled placement, still unread), so moving a
  workload is replaying the desk's own delivery onto a fresh placement with
  one listing off the table — stop the old tenancy at its anchor, deliver
  the archived frame to the new one; the store is the run, so nothing is
  copied and the move costs at most one uncommitted update. PLACE-FIRST: a
  run with nowhere to go is refused a move, not stopped — unless `park`
  (decommission's mode: the host dies regardless), which stops it and
  journals it PARKED with the boot instructions; the ordinary campaign
  resubmit revives it, same run_id. `placements()` is the archive as a
  read: the latest binding per run_id.

### The host (`rlstack/runner/host.py`)

- **submit** — how an experiment reaches metal: **bind** each declared pool onto
  an owned engine serving that base at that shape, **fit** (refuse past
  capacity), **attest** (roster in memory, journal to `hosts/<name>/log.jsonl`),
  **run** under the host's shared arbiter against the experiment's own store.
- **attach / detach** — a resident's registration with the arbiter, and a
  tenancy's entry in the roster and the journal.
- **stop** — adopt's per-run inverse: the adoption task cancelled and
  AWAITED (the daemons run under one TaskGroup, so cancellation is
  structural), the detach journaled, the run_id free to adopt again — here
  or elsewhere. Safe mid-update by resume-equivalence.
- **admit** — the one verb work wraps itself in; entering it guarantees the
  resident is resident.
- **sleep / wake** — a build fact of `VllmEngine` (and the learner's offload),
  deliberately NOT on the Engine protocol: the seam an alternating host's
  arbiter hooks call to make a partition really hand the device back.
- **adopt** — submit, without the submitter in-process: decode the frame's
  canonical spec, derive the schema HERE (`schema_for`), resolve the routes into
  RemotePools (`transport_for` turns an address into a transport; the pool's declared
  base/tp wrap it), run the custody checks fail-fast, roster the tenancy
  EAGERLY, and run it as a background task on this host's own loop. The reply
  is acceptance, never completion — the ledger is the result channel.
  Re-adoption is resume.

### Pool traffic (`rlstack/runner/traffic.py`, `rlstack/runner/remote.py`)

- **sample** — drive one pool for one episode, assembling `TokenEvent`s into a
  Turn; admitted, because it occupies the metal.
- **score** — logprobs of GIVEN tokens: one prefill pass, no decode,
  deterministic and seedless. Judges sample; teachers score.
- **collect** — schedule one wave of episodes deterministically given `(master,
  update)`, assigning group keys.
- **add_bundle / reachability / tokenize** — the admission-free verbs: additive
  registration and build facts, which by the tenancy invariant never disturb
  traffic (`ask` on the wire; `sample`/`score` ride `call`).

### The store (`rlstack/data/stores/base.py`)

- **cas_put / cas_get** — content-addressed objects by sha256.
- **open_run** — attach-or-create; attach discards everything the ledger never
  committed.
- **write_wave / read_wave**, **write_postdata / read_postdata**,
  **write_postdata_part / read_postdata_part** (one producer's columns; the
  read answers None while absent, because it is an await predicate),
  **write_blob / read_blob**, **write_eval** — the run's data sections.
- **append_ledger** — THE commit point, and the Trainer's alone.
- **sweep** — deletion's second meaning, the same rule read twice: attach
  sweeps what the ledger NEVER COMMITTED; `RunHandle.sweep(policy)` frees what
  the ledger has MOVED PAST. The Trainer sweeps at every commit and on start;
  `python -m rlstack sweep` is the operator's backstop for a stopped run (it
  attaches, so never point it at a live one).
- **append_host_event / read_host_log / list_hosts**, **append_fleet_event /
  read_fleet_log** — the journals.
- **peek_\*** — the observer's only door.

---

## The two planes

Every runtime interaction rides exactly one of two planes, and they have
different failure semantics.

**The pool-traffic plane** goes over the wire: sample and score, millisecond
RPC, admitted at the serving host by that host's own arbiter. It carries
JSON-safe frames and nothing durable. Its signals are *availability* signals —
a host answers or it does not, an engine has capacity or it waits — and none of
them is a fact about the experiment. A pool moving out of process changes zero
lines of the runner (`RemotePool` is the full Engine protocol) and changes no
result.

**The store plane** goes over the volume: waves, postdata, blobs, the ledger,
the journals. It carries everything durable, and its one signal is the
*durability commit bit* — the ledger append. Daemons wait on predicates over
this plane and never on each other. Work not sealed by a ledger line does not
exist and regenerates.

The consequence worth stating: an availability signal must never be mistaken
for a commit. A bundle registered on an engine is availability; the ledger line
naming it is the commit. A host journal entry is observability; the run's
manifest and ledger are truth.

---

## Invariants

I1–I12 are stated once, in `agent-context/rl-stack-spec.md` §1, and are
referenced by number throughout the code. In short, by subject:

| | subject | where the vocabulary above touches it |
|---|---|---|
| I1 | two worlds, one membrane | seal, Rollout/Trajectory, WaveFeed |
| I2 | policy is the only bridge | adapter type, lowering, bundle, parity |
| I3 | identity is computed | run_id, spec, registry code hashes |
| I4 | registered things declare, then compute | registry, flow graph, submit gate |
| I5 | GPU topology is semantics-neutral | GpuSet, pool, traffic, arbiter policy |
| I6 | record loss-independently at the seal | Turn, Trajectory, postdata |
| I7 | the substrate is certified, not assumed | mechanism probe, parity certificate |
| I8 | multi-tenancy on both sides | tenant, additive install, row plan, add_bundle |
| I9 | the loss is pure math | loss, postprocessor, token_level |
| I10 | one experiment, one store | run store, peek, observer |
| I11 | runs self-describe | dictionary.json, flow graph |
| I12 | a host is an atomic purposed partition | host, partition, regime, desk ladder |

Deltas agreed after the last spec fold-in live in `agent-context/CONTEXT.md`;
later entries supersede earlier ones, and the code plus the latest entry win.
