# The Thin Wrapper: Primitives

*Working spec v2, August 2026. Organized around the contract: two worlds, one membrane. Long-form rationale archived (see footer).*

## 0. TL;DR

A self-hosted, Tinker-shaped RL harness. **Two worlds**: an *inference world* that interacts with live environments, and a *training world* that consumes only static, sealed data — connected by one membrane (the store) in one direction and by compiled policy bundles in the other. **Five runtime verbs** (`sample`, `forward_backward`, `optim_step`, `sync_weights`, `save/load`) over **declarative, content-hashed specs**; every user-swappable behavior is a **registered function** with a static declaration half and a pure compute half. Where it runs (Modal / AWS / local; whole GPUs or fractions; any colocation) is semantics-neutral and decided at deploy/submit time.

```
                 compiled bundles (sync_weights)
            ┌───────────────────────────────────┐
            ▼                                   │
   INFERENCE WORLD                       TRAINING WORLD
   env · rewards · eval                  advantage · loss · optim
   (interactive, live, async)            (static sealed data ONLY)
            │                                   ▲
            └── sealed trajectories (the store) ┘
```

## 1. The Contract

*(Why this exists, in one breath: one prior stack drowned iteration in bespoke release ceremony while identity leaked into hand-typed names; another lost weeks to silent trainer/sampler mismatch in hand-rolled glue. The commodity parts — engines, distributed training, weight sync — are now libraries. The wrapper is only the contract below, enforced.)*

**I1 — Two worlds, one membrane.** Inference-world code (envs, rewards, eval) may interact: sample, call tools, wait. Training-world code (advantages, losses, the optimizer) consumes only *sealed* trajectories — static data in the store. The **seal** (env + all rewards finished → trajectory immutable) is the membrane; nothing interactive crosses it, and the only thing that crosses back is a new bundle version. On-policy RL, off-policy distillation, and SFT are the *same* training-world contract with different producers behind the membrane.

**I2 — Policy is the only bridge.** An adapter kind exists in both worlds by definition: a rollout lowering (engine-side) and a replay lowering (trainer-side) of the same math. Therefore it is the only primitive with a mandatory parity test, and no kind runs without a current parity certificate.

**I3 — Identity is computed, never typed.** `run_id = h(spec ⊕ registered-code hashes ⊕ data fingerprint)`. Editing a registered function's body changes identity; renaming a file changes nothing; humans never author identifiers.

**I4 — Registered functions declare, then compute.** Every registry entry has a static declaration half (`@loss(requires=…)`, `@reward(components=…)`, `@advantage(consumes=…)`) and a pure compute half. All wiring — bank ⊇ loss.requires, reward components ⊇ advantage.consumes, eval/task disjointness — validates at submit, before any GPU is touched. Compute halves are pure over declared inputs: CPU-testable, offline-re-runnable.

**I5 — GPU topology is semantics-neutral.** Groups colocate any members; the member vocabulary is closed at two workload kinds — *engine pools* and *learners* — and everything else (rollout, eval, judges, teacher hints) is traffic routed to named pools. Any two GpuConfigs execute the same experiment; placement changes wall-clock, never results. Feasibility (memory arithmetic, interconnect class, sleep-pattern constraints) is checked at submit; contention is measured, not legislated.

**I6 — Record loss-independently at the seal.** Token ids (the engine's, never re-tokenized), behavior logprobs, bundle id, seeds, reward components, env extras — always, regardless of the current loss. Any future estimator runs against old data without regeneration. Advantages are derived, never stored.

**I7 — The substrate is certified, not assumed.** One pinned image (torch/CUDA/flash-attn/vLLM/flashinfer solved once, in CI) is the unit of compatibility on every backend, local included. The daemon runs a boot probe (import + tiny fwd/bwd of each kernel path) before accepting work; each primitive earns cached certificates (parity, packing-equivalence, dry-run) per image version; rollout lowerings declare fallback ladders (FA3 → FA2 → SDPA) so throughput is best-effort and recorded while correctness is never contingent on the rung.

## 2. Primitives

**A. Specs** (hashable — they are identity) · **B. Registries** (declared + pure) · **C. Runtime** (the daemon's verbs) · **D. Data objects** (what crosses stages; all durable).

```python
# ════════════════════════════════════════════════════════════════════
# A. SPECS
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExperimentSpec:
    policy: PolicySpec               # the bridge (I2)
    gen: GenSpec | None              # inference world; None = pure-offline run
    rollouts: RolloutSource          # training world's ONLY input (I1)
    algo: AlgoSpec | None            # training world; None = generation-only run
    gpu_config: GpuConfig            # semantics-neutral (I5)
    seeds: Seeds
    init: WarmStart | None = None    # warm-start from another run's sealed state
    eval: EvalSpec | None = None     # inference world; firewalled measurement
    tier: Literal["lab", "release"] = "lab"

# ---------- the bridge ----------
@dataclass(frozen=True)
class PolicySpec:
    base: str                        # HF id + pinned revision
    bank: dict[str, AdapterSpec]     # named per-matrix deltas & other interventions

@dataclass(frozen=True)
class AdapterSpec:                   # ONE typed intervention; lives in BOTH worlds
    kind: str                        # registered @adapter_kind: "lora" | "soft_prompt"
                                     #   | "logit_bias" | "attn_bias" | ...
    site: str                        # canonical name ("blocks.0-15.attn.q",
                                     #   "resid_pre.20", "prompt[:8]") or raw module
                                     #   path — resolved against the base model's
                                     #   SiteSchema at Phase 0 (see below)
    init: Mapping                    # kind-specific: rank, tie, shapes, projections
    trainable: bool = True
# sugar: lora(site, r, tie=False) · soft_prompt(n, d) · attn_bias(spans) · ...

# --- SiteSchema: canonical names, native compute (NO model recompilation) ---
# At daemon boot, per base model, the daemon compiles a SiteSchema: canonical names
# (blocks.N.attn.q, blocks.N.mlp.in, resid_pre.N, embed, logits, prompt[:k], ...)
# mapped onto (a) the trainer's actual module tree / hookable boundaries and (b) the
# engine's reachable mechanisms — seeded from community tables (PEFT target-module
# maps, vLLM packed_modules_mapping). Each entry carries OPEN metadata, e.g.
#   {has_weight, shape, is_boundary,
#    engine_reachable_via: punica | prompt_embeds | logits | lse_patch | none, ...}
# and an @adapter_kind declares its site requirements as a PREDICATE over that
# metadata — no closed sort taxonomy; user kinds can require anything the schema
# expresses. Phase 0 resolution failures are submit errors: empty or ambiguous match,
# shape mismatch, or an engine-unreachable site for a kind whose rollout lowering
# needs one (trainer-only kinds, e.g. value_head, pass with engine_reachable_via:
# none). The resolved schema hashes into parity-certificate inputs (I7): a base or
# image bump that moves a boundary invalidates certificates instead of silently
# shifting sites. engine_reachable_via is self-reported INVENTORY — computed from
# vLLM natives plus what rlstack_engine's inventory declares; a ledger of installed
# capabilities, not discovery.


# ---------- inference world ----------
@dataclass(frozen=True)
class GenSpec:                       # everything interactive lives here
    env: str                         # registered @env
    tasks: str                       # content-addressed ("cas://<sha>/...")
    rewards: tuple[str, ...]         # run BEFORE the seal; may sample (judges)
    sampling: SamplingSpec = SamplingSpec()
                                     # temperature/top_p/max_tokens — SCIENCE: it
                                     # defines the behavior policy, so it hashes;
                                     # envs may override per sample() call

@dataclass(frozen=True)
class EvalSpec:                      # firewalled measurement: reads only IMMUTABLE
    tasks: str                       #   bundle versions + held-out tasks (disjointness
    every: int = 10                  #   from gen.tasks checked at submit); async —
    env: str | None = None           #   never blocks or influences training
    rewards: tuple[str, ...] = ()    # defaults: gen.env / gen.rewards
    n_samples: int = 1
    pool: str = "main"               # which engine pool carries eval traffic (I5)

# ---------- training world ----------
@dataclass(frozen=True)
class RolloutSource:                 # training ALWAYS consumes sealed store data
    source: str                      # "live"                → this run's gen output,
                                     #                         freshness bounded by
                                     #                         schedule.max_policy_lag
                                     # "store://<run_id>/…"  → another run's rollouts
                                     #                         (off-policy distillation)
                                     # "cas://<sha>/…"       → imported dataset (SFT)
# one consumption contract for on-policy RL, off-policy distill, and SFT (I1);
# producer compatibility (tokenizer, schema) is checked at submit from manifests

@dataclass(frozen=True)
class AlgoSpec:
    loss: str                        # registered @loss (its `requires` travel with it)
    advantage: str                   # registered @advantage
    optim: OptimSpec                 # optimizer + per-delta param-group overrides
    schedule: Schedule               # group_size, rollouts_per_wave, n_updates,
                                     #   epochs_per_wave, microbatch_tokens,
                                     #   max_policy_lag (staleness policy — sibling of
                                     #   epochs_per_wave; changes the estimator,
                                     #   not the hardware)

# ---------- topology (semantics-neutral, I5) ----------
# group members — the CLOSED vocabulary. Two workload kinds; all else is traffic:
#   engines(name, base=POLICY, tp=1, n=1, fraction=None)  # serves sample+score traffic
#   learner(fsdp=1, fraction=None)                        # differentiable fwd/bwd
# traffic → pools by name: env client → "main" · EvalSpec.pool · judge rewards via
#   sample(pool=…) · a loss's requires={"teacher_logprobs": Teacher(pool="teacher")}

@dataclass(frozen=True)
class Group:                         # THE unit of colocation — typed, not a flag
    gpus: GpuSet                     # pure device demand: gpus(n=6) | gpus(n=16, nodes=2)
                                     #   (literal ids only as a dedicated-daemon pin)
    members: tuple[Member, ...]      # co-resident on these devices; fractions = the
                                     #   members' memory treaty
    sharing: Literal["concurrent", "sleep"] = "concurrent"
                                     # concurrent: co-resident under member fractions
                                     # sleep: learner-phase ↔ engines-phase alternation
                                     #   (exactly one learner member; implies lag 0)

@dataclass(frozen=True)
class GpuConfig:
    groups: tuple[Group, ...]        # each member name appears exactly once; feasibility
                                     #   (memory fit, interconnect class) checked at submit
# "train+infer together, teacher apart":
#   GpuConfig(groups=(
#     Group(gpus(n=6), (engines("main", tp=2, n=3), learner(fsdp=2)), sharing="concurrent"),
#     Group(gpus(n=4), (engines("teacher", base="Qwen/Qwen3-32B", tp=4, n=1),))))

@dataclass(frozen=True)
class Seeds:
    master: int                      # per-request, dataloader, init seeds all derive

@dataclass(frozen=True)
class WarmStart:                     # start a NEW experiment from existing sealed state
    policy: str                      # "store://<run_id>@<version>" — a sealed policy
                                     #   version from another run (or "cas://<sha>")
    optim: Literal["load", "fresh"] = "fresh"
                                     # "load": restore per-delta Adam moments + step
                                     #   counts (continuation-style forks);
                                     # "fresh": new optimizer over loaded weights
                                     #   (cross-experiment transfer — the default)
    map: Mapping[str, str] = {}      # source delta name → this bank's name; unmapped
                                     #   bank deltas init fresh (partial warm start OK)
# WarmStart HASHES into run_id (different init = different science) and records the
# parent run/version in the manifest — lineage without ceremony. Structural
# compatibility (sites, ranks, shapes) is checked at submit; Phase 1 performs the load.

# BackendProfile stays OUTSIDE the spec (deploy-time): {kind: "local"|"modal"|"skypilot",
#   gpu, idle, region/spot…}. The manifest records which one executed the run.

# ════════════════════════════════════════════════════════════════════
# B. REGISTRIES — declaration half + pure compute half (I4)
# ════════════════════════════════════════════════════════════════════

@adapter_kind("attn_bias")           # the bridge (I2) — the five-member protocol
class AttnBias(AdapterKind):
    def site_ok(self, meta) -> bool: ...      # predicate over SiteSchema metadata (Phase 0)
    engine_plugin = "rlstack_engine.attn_bias"  # STRING, not an import — the module engine
                                                #   processes load at boot (may be None:
                                                #   most kinds ship NO engine code)
    def params(self, sites, init) -> ParamSet: ...        # trainable objects; tie= here
    def install_replay(self, model, params, sites): ...   # Phase 1: setattr-swap matched
                                                          #   modules on the OWNED trainer model
    def emit(self, params) -> BundleSection: ...          # what compile() writes for this delta
    def parity(self, harness): ...            # certificate; gates first use (I7)
# The bundle is the ONLY data channel between the two halves: emit → engine-side
# registered consumer → preallocated slot buffers (values change, shapes never).
# Kinds with native engine surfaces need no plugin — engine_plugin=None: lora→punica,
# soft_prompt→prompt_embeds, logit_bias→logits-processor entry point, value_head→nothing.
# Versioning: engine-touching kinds are IMAGE-versioned; trainer-only kinds are
# EXPERIMENT-versioned (ride the submit, no rebuild).

@env("tool_use")                     # INFERENCE WORLD; drives engines via a client;
async def rollout(llm: SampleClient, task: Task) -> Trajectory: ...
                                     # tool calls are stop-conditions; setup()/teardown()
                                     # lifecycle hooks allowed (code hashes into identity)

@reward("verifier", components=("correct", "has_boxed"))      # INFERENCE WORLD;
async def score(traj: Trajectory, llm: SampleClient) -> RewardComponents: ...
                                     # runs before the seal; may sample; components
                                     # declared → wiring validated at submit (I4)

@advantage("blend", consumes=("correct", "judge"))            # TRAINING WORLD;
def adv(wave: Wave, ctx: ArchiveContext) -> AdvantageWeights: ...
                                     # wave-scope, detached, CPU, pure over sealed
                                     # data → re-runnable offline (I6)

@loss("grpo", requires=set())        # TRAINING WORLD; microbatch-scope, differentiable,
def loss(out: PolicyOutputs, b: TokenBatch) -> Loss: ...     # launches NOTHING
# requires ⊆ { "ref_logprobs" | Ref("bank:v120"),   ← planned no_grad pass, memoized/wave
#              "values", "entropies", "hidden_states",  ← training-forward extras (grads)
#              Teacher(pool="teacher"),              ← scoring pass on a named engine pool
#              Probe(build_fn) }                     ← differentiable scoring forward
# Stage rule: needs gradients → loss · needs cross-rollout/archive context → advantage
#             · needs to sample → env/reward (inference world).

# ════════════════════════════════════════════════════════════════════
# C. RUNTIME — the daemon's verbs, and the runner that consumes a spec
# ════════════════════════════════════════════════════════════════════

class Run(Protocol):
    async def sample(self, prompts, sampling, *, bundle: BundleRef | None = None) -> Rollouts: ...
    async def forward_backward(self, batch: TokenBatch, loss: str) -> Metrics: ...
        # trainer-kernel logprob recompute; truncated-IS + logprob-gap alarm built in
    async def optim_step(self) -> PolicyVersion: ...          # atomic; per-delta versions
    async def sync_weights(self, v: PolicyVersion) -> None: ...
        # compile(bank, v) → content-hashed Bundle → add_lora into pools (MBs, ~ms)
    async def save(self) -> RunState: ...
    async def load(self, s: RunState) -> None: ...
        # RunState = adapter tensors + per-delta optimizer moments + task cursor + RNG.
        # Same-run resume rides these; a NEW experiment loads foreign state
        # declaratively via ExperimentSpec.init (WarmStart) — specs never do setup

class SampleClient(Protocol):        # what inference-world code receives
    async def sample(self, messages, *, sampling=..., stop=..., bundle=...,
                     pool: str = "main", seed=...) -> Turn: ...

def compile(bank, versions) -> Bundle: ...   # content-hashed, cached, deduped

class Backend(Protocol):             # deploy-time; outermost layer
    def up(self, demand, image, mounts) -> DaemonHandle: ...
    def attach(self, name) -> DaemonHandle: ...
    def run(self, demand, image, mounts, job) -> RunHandle: ...   # ephemeral (release)
    def down(self, name, *, keep_volumes=True) -> None: ...
    def status(self) -> list[DaemonInfo]: ...

# Runner lifecycle (the consumer of a spec):
#   Phase 0 — resolve registries; SITE RESOLUTION against the base's SiteSchema;
#             JOINT validation (bank ⊇ loss.requires; components ⊇ advantage.consumes;
#             kind predicates × site metadata; sharing × max_policy_lag; group memory
#             fit; task disjointness); identity per I3
#   Phase 1 — IDEMPOTENT setup (resume re-runs it): store attach-or-create, pool leases,
#             learner build, parity certificates (I7), warm-start load (spec.init),
#             lifecycle hooks setup(ctx)
#   Phase 2 — loop: collect wave (envs+rewards stream; per-episode) → SEAL → advantage →
#             planned passes → fwd/bwd × microbatches → optim_step → compile+add_lora
#   Eval driver: a separate coroutine subscribed to ledger commits — never a loop phase

# ---------- Engine mechanisms & the plugin package ----------
rlstack/          # library: specs, registries, runner, daemon, store, trainer halves
rlstack_engine/   # engine-world code ONLY — never imported by rlstack (bound by string)
  register.py     # single entry point in group "vllm.general_plugins"
  backends/lse_merge.py   # conditional attention backend (stock-degenerate when inactive)
  models/<arch>.py        # ONE generic Instrumented<Arch> subclass per architecture
                          #   family (boundary shims; exists only when a module-boundary
                          #   kind does) — never one subclass per kind or experiment
  slots.py        # bundle consumers → preallocated GPU buffers
# Registration is AMBIENT: vLLM's own load_general_plugins() fires our entry point
# inside every engine and worker process before model construction — the only
# mechanism guaranteed to run in spawned worker processes on every node. Our runner
# never calls register_*; its only levers are INSTALLATION (package in the pinned
# image), SELECTION (engine args at pool boot), and FEEDING (bundles).
# Cadences: image cadence = deps + engine-world code (certified, deliberate — I7);
# submit cadence = specs, envs, rewards, losses, trainer-only kinds, bank uses —
# every submit, no rebuild.

# The insertion ladder (prefer the highest rung that suffices):
#   request field < logits-processor entry point < registered model subclass
#   (ModelRegistry.register_model overwrites an arch string) < registered attention
#   backend (register_backend; AttentionBackendEnum.CUSTOM selected via
#   AttentionConfig.backend, or override FLASH_ATTN in place) < patch applied at
#   image build (last resort).
# Module-boundary edits stop at the model layer; attention-SCORE edits sink to the
# backend layer — only the backend sees block tables, cached keys, and LSEs.
# attn_bias rollout lowering, concretely: stock flash_attn with
# return_softmax_lse=True + a tiny attention over the biased key partition +
# vLLM's own merge_attn_states op, the constant partition bias folded into that
# partition's LSE (constant bias on a key partition = e^b factor on its softmax
# mass — exact, no kernel edited). Its replay lowering: FlexAttention score_mod
# (compiler generates fwd+bwd; grads to θ free); packing discipline: rectangle
# indices must be document-relative.

# Pool identity: backend selection is BOOT-TIME, not hot-swappable — a pool's
# identity is (base model, mechanism set). Phase 0 derives each pool's needed
# mechanisms from the kinds of the policies it will serve and boots it with
# matching engine args (recorded in the manifest); scheduling an lse_patch-needing
# experiment onto a stock pool requires a pool reboot. Two pools on the SAME image
# can differ (one stock FA — vLLM's literal own class — one custom): the image
# defines what is POSSIBLE, boot args what is ACTIVE, the bundle what is LOADED,
# the request what is USED.

# ════════════════════════════════════════════════════════════════════
# D. DATA OBJECTS — all durable in the store
# ════════════════════════════════════════════════════════════════════

Rollouts / Trajectory:  token_ids (engine's — never re-tokenized) · segment_ids ·
                        loss_mask · behavior_logprobs · bundle_id · seed ·
                        policy_version · engine_id · reward_components · env_extras
TokenBatch:             token_ids · loss_mask · advantages · behavior_logprobs ·
                        group/pair/segment ids · extras
PolicyOutputs:          logprobs (+ declared: ref_logprobs, values, entropies,
                        hidden_states, teacher_logprobs, probe outputs)
PolicyVersion:          the tuple of delta versions — a version pins BOTH the delta
                        tensors and their optimizer moments (per-delta lineage)
Bundle:                 compiled servable adapter set; content-hash = identity
Certificates:           parity / packing-equivalence / boot-probe results, cached
                        per (primitive version, image version)  (I7)
Store:                  runs/<run_id>/{manifest.json, ledger.jsonl, rollouts/*.parquet,
                        adapters/<name>@<v>.safetensors, optim/<name>@<v>.safetensors,
                        eval/<v>/}  ·  cas/<sha256>/
                        — optimizer moments commit at every optim_step, in lockstep
                        with the ledger: this is what makes kill-9 resume and
                        resume-across-hardware hold
```

## 3. Examples

### Example 1 — A basic LoRA experiment, end to end

```python
# exp.py — the whole experiment
from rlstack import (ExperimentSpec, PolicySpec, GenSpec, RolloutSource, AlgoSpec,
                     EvalSpec, Seeds, OptimSpec, Schedule, GpuConfig, Group,
                     engines, learner, lora, gpus, client)

exp = ExperimentSpec(
    policy=PolicySpec(
        base="Qwen/Qwen3-8B",                        # pinned revision
        bank={
            "attn": lora(site="layers.0-15.self_attn.*", r=16),
            "mlp":  lora(site="layers.16-31.mlp.*", r=32),
            "head": lora(site="layers.28-31.self_attn.o_proj", r=8),
        },
    ),
    gen=GenSpec(env="math_single_turn",
                tasks="cas://3fa9c2.../math_train.jsonl",
                rewards=("verifier",)),
    rollouts=RolloutSource("live"),                  # on-policy: consume own sealed gen
    algo=AlgoSpec(
        loss="grpo",
        advantage="grpo_group_norm",
        optim=OptimSpec("adamw", lr=1e-5, betas=(0.9, 0.95),
                        overrides={"head": {"lr": 3e-6}}),    # per-delta param group
        schedule=Schedule(group_size=8, rollouts_per_wave=512, n_updates=300,
                          epochs_per_wave=1, microbatch_tokens=16384,
                          max_policy_lag=0),
    ),
    eval=EvalSpec(tasks="cas://8c31f0.../gsm_heldout.jsonl", every=10),
    gpu_config=GpuConfig(groups=(
        Group(gpus(n=6), (engines("main", tp=2, n=3), learner(fsdp=2)),
              sharing="concurrent"),
        Group(gpus(n=1), (engines("eval", tp=1, n=1),)),
    )),
    seeds=Seeds(master=17),
)

client.submit(exp)    # run_id = h(spec ⊕ registered-code hashes ⊕ data fingerprint)
```

One update's lifecycle: the daemon `sample`s a wave of 512 rollouts (64 groups of 8) under the current bundle on the "main" pool; rewards, behavior logprobs, bundle_id, and seeds all land in the store at the seal. `grpo_group_norm` runs once per wave on CPU — detached, pure over the sealed rollouts — attaching advantage weights, then `forward_backward` streams microbatches through the FSDP2 learner, recomputing logprobs under trainer kernels with the built-in truncated-IS correction against the recorded behavior logprobs. `optim_step` commits atomically, bumping only the three touched delta versions; `sync_weights` compiles the bank at the new PolicyVersion into a content-hashed Bundle and `add_lora`s it into the pools (megabytes, milliseconds), and the next wave begins. The eval driver, subscribed to the ledger, evaluates every 10th version on its own pool. The run_id and manifest were computed at submit — you never named anything.

### Example 2 — Adding PPO: what new code exists

```python
# --- 1. one new bank entry -------------------------------------------------
bank["critic"] = AdapterSpec(kind="value_head", site="final_hidden",
                             init={"hidden": 4096}, trainable=True)
# "value_head" is a registered @adapter_kind: replay lowering = scalar head
# over final hidden states; rollout lowering = nothing (never served)

# --- 2. one advantage ------------------------------------------------------
@advantage("gae", consumes=("reward",), requires={"values@behavior"})
def gae_adv(wave: Wave, ctx: ArchiveContext) -> AdvantageWeights:
    v = wave.extras["values@behavior"]     # critic as of rollout time — same
                                           # machinery as the ref pass, memoized/wave
    return gae(wave.reward_components, v, gamma=1.0, lam=0.95)   # detached, CPU

# --- 3. one loss -----------------------------------------------------------
@loss("ppo", requires={"values"})          # V_theta = training-forward extra,
def ppo(out: PolicyOutputs, b: TokenBatch) -> Loss:   # grads flow to the head
    ratio = (out.logprobs - b.behavior_logprobs).exp()
    surr  = -torch.min(ratio * b.advantages,
                       ratio.clamp(1 - EPS, 1 + EPS) * b.advantages)
    vloss = value_clip(out.values, b.extras["returns"], b.extras["values@behavior"])
    return masked_mean(surr + 0.5 * vloss - 0.01 * entropy(out.logprobs), b.loss_mask)

# optimizer already covers policy deltas + critic head as separate param
# groups, derived from bank structure — nothing to wire
exp2 = replace(exp, algo=AlgoSpec(loss="ppo", advantage="gae",
                                  optim=exp.algo.optim, schedule=exp.algo.schedule))
```

That is the entire diff: the Run verbs, the schedule, the GPU config, and the store schema are untouched, and the critic's weights version through the same per-delta ledger as any LoRA. The behavior-value pass reuses the ref-pass machinery (planned no_grad forward, memoized per wave), so PPO costs one extra forward per wave plus a head. This mirrors how verl and prime-rl factor PPO — a loss, an advantage, and a critic module, not a new trainer.

### Example 3 — A soft prompt with a learned attention bias: where kernel-level code lives

```python
bank = {
    "latent": AdapterSpec(
        kind="soft_prompt",
        site="prompt[:8]",                     # 8 virtual tokens ahead of the prompt
        init={"n": 8, "d": 2048, "posterior": "diag_gaussian"},
    ),
    "readout": AdapterSpec(
        kind="attn_bias",
        site="queries -> prompt[:8]",          # every query, onto the 8 prompt keys
        init={"param": "bounded_sigmoid",      # b = cap · σ(θ), so 0 < b < cap
              "init": "ln(2)", "cap": "ln(64)"},
    ),
}
```

Each `@adapter_kind` registration supplies the protocol's members (§2B): a site predicate, a trainable parameterization, a rollout lowering (its `engine_plugin`, when one is needed at all), a replay lowering, and a mandatory parity test. For `soft_prompt` there is no kernel work at all — its rollout lowering hands the eight learned rows to vLLM's native `prompt_embeds` (zero engine changes; prefix-cache hashing already accounts for embeds), and its replay lowering just splices the same rows into the trainer forward's embedding output. `attn_bias` is the one kind that touches the engine. Its rollout lowering is a small attention-backend patch: stock FlashAttention runs unmodified, then the bias rectangle over the eight prompt-key columns is merged exactly via the attention LSE, and the learned scalar lives in a graph-stable buffer so CUDA graphs capture once and never recapture as the value trains. Its replay lowering is a FlexAttention `score_mod` adding the same bias on the same rectangle, and the (d) parity test compares the patched-Flash rollout path against the dense/Flex replay path numerically before the kind is allowed to run — earning the I7 certificate for the current image. The patch ships as one engine-plugin module, so everything else rides stock vLLM.

### Example 4 — Environments

```python
@env("math_single_turn")
async def math_env(llm: SampleClient, task: Task) -> Trajectory:
    turn = await llm.sample([user(task.prompt)], stop=[EOS])   # one shot, done
    return Trajectory.from_turns(task, [turn])

@reward("verifier", components=("correct",))
async def verify(traj: Trajectory, llm: SampleClient) -> RewardComponents:
    ok = extract_boxed(traj.text) == traj.task.answer          # pure check, no sampling
    return RewardComponents(correct=float(ok))

@env("tool_use")
async def tool_env(llm: SampleClient, task: Task) -> Trajectory:
    msgs = [user(task.prompt)]
    for _ in range(task.max_turns):
        turn = await llm.sample(msgs, stop=["</tool_call>", EOS])
        msgs.append(turn.message)
        call = parse_tool_call(turn.text)
        if call is None:                                       # stopped at EOS: answered
            break
        msgs.append(tool_result(await execute(call)))          # tool-result tokens get
    return Trajectory.from_messages(task, msgs)                # loss_mask=0 via segment records

@reward("llm_judge", components=("judge", "judge_parsed"))     # a reward that SAMPLES
async def judge(traj: Trajectory, llm: SampleClient) -> RewardComponents:
    prompt = judge_prompt(traj)                                # rubric + rendered transcript
    turn = await llm.sample(prompt, stop=[EOS],
                            bundle=JUDGE_BUNDLE)               # optional pinned judge bundle
    traj.env_extras["judge"] = turn.text                       # transcript is recorded
    return RewardComponents(judge=parse_score(turn.text))
```

Envs and rewards are inference-world citizens: thousands of asyncio coroutines against the resident pools, each awaiting its own `sample` calls while continuous batching absorbs tool-execution latency — a coroutine blocked on a Python tool costs the engine nothing. Prefix caching makes each turn's resubmission cheap: only the new suffix pays for prefill. Judge traffic is just more sample traffic, batching into the same pool (or a named one). Sampling is never differentiable — the stage rule routes anything that needs to sample to the inference world — which is why judges live here and never in the loss.

### Example 5 — Multi-GPU with sharding: 35B-A3B on three nodes

```python
# exp.py — logical topology only; provider names never appear here
gpu_config = GpuConfig(groups=(
    Group(gpus(n=16, nodes=2), (engines("main", tp=2, n=8),)),  # 8 TP=2 engines, nodes 1-2
    Group(gpus(n=8, nodes=1),  (learner(fsdp=8),)),             # FSDP2 8-way shard, node 0
))
spec = ExperimentSpec(
    policy=PolicySpec(base="Qwen/Qwen3.5-35B-A3B",
                      bank={"pi": lora("layers.*.self_attn.*", r=32)}),
    gen=GenSpec(env="tool_use", tasks="cas://<sha>/tasks.jsonl", rewards=("verifier",)),
    rollouts=RolloutSource("live"),
    algo=replace(algo, schedule=replace(algo.schedule, max_policy_lag=1)),
    gpu_config=gpu_config, seeds=Seeds(master=0),
)

# backends.toml — deploy-time only; the run manifest records which profile executed
#   [modal-3xh100]  kind = "modal"    gpu = "H100:8"  nodes = 3  idle = "snapshot"
#   [aws-3xh100]    kind = "skypilot" cloud = "aws"   gpu = "H100:8"  nodes = 3

# $ rl up --backend modal-3xh100 && rl run exp.py
```

The spec names no provider: `GpuConfig` is pure demand — 24 GPUs across three nodes plus a cross-node interconnect class — and the backend either maps it to metal at `rl up` or refuses at deploy time, never mid-run. Each rollout engine is TP=2 inside a node, never across one, so only bundle sync and trajectory traffic touch the interconnect. The learner shards the frozen 35B MoE backbone with FSDP2 while gradients touch only adapter parameters, keeping optimizer state adapter-scale. `max_policy_lag=1` — a schedule property, §2 — lets rollout keep sampling under version *v* while the learner trains *v+1*, saturating both planes, with the built-in truncated-IS correction absorbing the one-version staleness. And because `sync_weights` ships compiled adapter deltas, cross-node weight sync is megabytes over the network instead of a full-weight broadcast.

### Example 6 — One GPU, 1B model, many replicates

```python
# (i) seed replicates: five run_ids, ONE resident engine pool
base = ExperimentSpec(
    policy=PolicySpec(base="Qwen/Qwen3-1.7B",
                      bank={"pi": lora("layers.*.mlp.*", r=16)}),
    gen=GenSpec(env="math_single_turn", tasks="cas://<sha>/train.jsonl",
                rewards=("verifier",)),
    rollouts=RolloutSource("live"),
    algo=algo,
    gpu_config=GpuConfig(groups=(
        Group(gpus(n=1), (engines("main"), learner()),
              sharing="concurrent"),)),            # co-resident; no sleep needed at 1B
    seeds=Seeds(master=0),
)
for s in range(5):
    client.submit(replace(base, seeds=Seeds(master=s)))
# each run's bundles coexist in multi-LoRA slots and batch
# together in the same forwards — replicates cost marginal compute

# (ii) fully self-contained: engines + learner on one card
solo = GpuConfig(groups=(Group(
    gpus(n=1),
    (engines("main", n=2, fraction=0.30),          # ~2 GB weights/engine + KV per fraction
     learner(fraction=0.25)),                      # ~3 GB LoRA trainer inside its fraction
    sharing="concurrent"),))
# engines boot sequentially (compile/capture paid once each);
# sync_weights is an add_lora on the same device — no network hop
```

The shared-engine pattern is the default for sweeps and replicates: one resident pool amortizes startup, compile, and CUDA-graph capture across all five runs, and multi-LoRA batching keeps forwards dense regardless of which run a request belongs to. Fractional engines buy isolation — different base models, incompatible engine configs, or a run you want fully self-contained down to its own KV budget. Both fit one 80 GB card at 1B scale with room to spare, so the choice is batching efficiency versus independence, not memory; when in doubt, share the pool.

---

*Archive: the long-form design rationale lives at the original artifact ("The Thin Wrapper") and in `~/Coding/rl-stack-design.md`. This spec supersedes it as the working surface.*
