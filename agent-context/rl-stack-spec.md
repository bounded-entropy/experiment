# The Thin Wrapper: Primitives

*Working spec v3, August 2026. Folded in the deltas of CONTEXT #21–#38 (v2 froze
before Phase B ran on metal). Organized around the contract: two worlds, one
membrane. Long-form rationale archived (see footer).*

## 0. TL;DR

A self-hosted, Tinker-shaped RL harness. **Two worlds**: an *inference world*
that interacts with live environments, and a *training world* that consumes only
static, sealed data — connected by one membrane (the store) in one direction and
by compiled policy bundles in the other. **Declarative, content-hashed specs**;
every user-swappable behavior is a **registered** class or function with a
static declaration half and a compute half. Where it runs (Modal / AWS / local;
whole GPUs or fractions; any colocation) is semantics-neutral and decided at
deploy/submit time. Experiments are **tenants**: one resident engine and one
resident learner serve many experiments at once, and a **Host** owns the metal
they share.

```
                 compiled bundles (add_bundle)
            ┌───────────────────────────────────┐
            ▼                                   │
   INFERENCE WORLD                       TRAINING WORLD
   envs (Rollout → seal)                 post pipeline · loss · optim
   (interactive, live, async)            (static sealed data ONLY)
            │                                   ▲
            └── sealed trajectories (the store) ┘
```

## 1. The Contract

*(Why this exists, in one breath: one prior stack drowned iteration in bespoke
release ceremony while identity leaked into hand-typed names; another lost weeks
to silent trainer/sampler mismatch in hand-rolled glue. The commodity parts —
engines, distributed training, weight sync — are libraries. The wrapper is only
the contract below, enforced.)*

**I1 — Two worlds, one membrane.** Inference-world code (environments) may
interact: sample, call tools, wait. Training-world code (post processors, the
loss, the optimizer) consumes only *sealed* trajectories — static data in the
store. The **seal** (`Rollout.seal() -> Trajectory`, enforced by the type
system: mutable inference record in, frozen training record out) is the
membrane; nothing interactive crosses it, and the only thing that crosses back
is a new bundle version. On-policy RL, off-policy distillation, and SFT are the
*same* training-world contract with different producers behind the membrane.

**I2 — Policy is the only bridge.** An adapter exists in both worlds by
definition: a rollout lowering (engine-side serving Mechanism) and a replay
lowering (trainer-side install) of the same math. It is therefore the only
primitive with a mandatory parity obligation. *(Status: `Adapter.parity` and
the certificate cache exist but are unwired; the running parity mechanism is
the per-update `logprob_gap` rail. Wiring boot-time certificates is open I7
work.)*

**I3 — Identity is computed, never typed.** `run_id = h(spec ⊕ registered-code
hashes ⊕ data fingerprint)`. Editing a registered function's body changes
identity; renaming a file changes nothing; humans never author identifiers.

**I4 — Registered things declare, then compute.** Every registry entry has a
static declaration half (`@postprocessor` classes declare `produces / consumes /
token_level / pools / sampling`; `@loss` declares `requires`; `@adapter`
classes declare `serving / provides / records / exports`) and a compute half.
All wiring validates at submit, before any GPU is touched, as **queries on the
flow graph** (spec/flow.py) — the one canonical walk over these declarations,
which each run also serializes as its own `dictionary.json` (see I11).

**I5 — GPU topology is semantics-neutral.** GpuGroups colocate members; the
member vocabulary is closed at two workload kinds — **pools**
(`pool(name, base, tp, n, fraction)`) and **learners** (`learner(fsdp,
fraction)`) — and everything else (rollout, eval, judges, teachers) is traffic
routed to pool *names*. Any two GpuConfigs execute the same experiment;
placement changes wall-clock, never results. Scheduling (the arbiter's policy,
the host chosen, `max_inflight`) is likewise outside identity. A gpu_config
may declare demands no single host can satisfy — several hosts answering one
experiment is the normal case, not a special one (#47: teacher, student
inference, and learner on three containers is the proven shape).

**I6 — Record loss-independently at the seal.** Token ids (the engine's, never
re-tokenized), behavior logprobs, bundle id + policy versions, seeds, finish
reasons, adapter-recorded per-token facts — always, regardless of the current
loss. Recorded facts are never re-derived. Post-pipeline columns are computed
*after* the seal and stored as postdata beside the wave, never inside the
sealed record.

**I7 — The substrate is certified, not assumed.** One pinned image (currently
vllm 0.28.0 / torch 2.13.0 / transformers 5.16.1) is the unit of compatibility.
Boot probes, per-build parity certificates, and fallback ladders are the
declared mechanism (rlstack_engine/certificates.py); the wired v0 reality is
pinned versions + the logprob_gap alarm.

**I8 — Multi-tenancy on both sides of the bridge.** Engine: bundle registration
is additive; every request pins its bundle at submission; requests batch across
tenants. Learner: installation is additive; every verb pins a tenant (the
run_id); one loaded base serves every tenant's adapters (#44: additive
install + row routing — every installed tenant's deltas stay wired, each
batched-forward row carries the slot whose delta applies; the trainer-side
punica). Nothing about one tenant's traffic, install, or load disturbs
another's. Sealed learner bytes are SHARD-WIDTH-FREE (#45): emit/load
payloads are identical at any fsdp width, so resume and bundle compile never
know how the metal was cut.

**I9 — The loss is pure math; post owns all production.** A loss is
`fn(PolicyOutputs, TokenBatch) -> LossResult` and can never cause metal work:
its `requires` names data columns only (postdata columns ∪ recorded facts ∪
bank-provided forward tensors) — enforced at registration. Anything that needs
a GPU to compute (judge scores, teacher logprobs, hinted rescoring) is a post
processor's job and lands in postdata first, per-trajectory scalars or
`token_level` per-token vectors. There are no planned passes. A processor may
score through ANY declared pool, including one serving a DIFFERENT base than
the policy (#47) — the preconditions are a shared tokenizer (the scored ids
must mean the same text to both) and that a non-policy pool serves its bare,
frozen base (its scores are then version-free and timing-independent).

**I10 — One experiment, one store, for life.** run_id is global but existence
is store-scoped, so the run store is a per-experiment binding made explicit at
submission; the host journals it, and the observer flags the same run_id seen
in two stores as a fork. Observers never attach: attach sweeps unsealed work,
so reading a live run goes through the store's read-only peeks.

**I11 — Runs self-describe.** At creation the runner writes `dictionary.json`
(the flow graph serialized): every column the run will contain, its producer,
consumers, phase, granularity, and whether it feeds the loss. Derived, never
identity. A UI renders a run from its own dictionary — no registry, no version
skew.

**I12 — A host is an atomic purposed partition; capability is a birth fact.**
The fleet's unit is not a GPU, a node, or a container: it is a Partition (some
slice of some GPUs — half of one L4 up through devices spanning nodes) born
with Regimes (inference|training × base × shard shape), attested against the
metal at construction and never grown or reshaped after. Sharding (Engine.tp,
Learner.fsdp) is a build fact the submit gate attests, like reachability.
Placement climbs a ladder with one currency and one decider per rung: JOIN
(capability exists; automatic; the target host's own arbiter admits — declared
fractions are carve hints, ignored once the weights live), CARVE (from
RESIDUAL only — capacity no partition owns — automatic because journaled;
births a new host, never reshapes one), ACQUIRE (new metal = money = a
human). A multi-regime host ALTERNATES its regimes on its own arbiter group —
one host wearing masks, never two hosts coordinating — so a sleep group
places onto exactly one host. The learner is never remote: the runner goes to
the learner's host and reaches every other partition through RemotePools
(Engine protocol over a transport; admission host-side, where the metal is).

## 2. Primitives

**A. Specs** (hashable — they are identity) · **B. Registries** (declared +
compute) · **C. Runtime** (host, arbiter, daemons) · **D. Data objects**.

```python
# ════════════════════════════════════════════════════════════════════
# A. SPECS
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExperimentSpec:
    policy: PolicySpec               # the bridge (I2)
    gen: GenSpec | None              # inference world; None = pure-offline run
    trajectories: TrajectorySource   # training world's ONLY input (I1)
    algo: AlgoSpec | None            # training world; None = generation-only run
    gpu_config: GpuConfig            # semantics-neutral (I5)
    seeds: Seeds
    init: WarmStart | None = None    # warm-start from another run's sealed state
    eval: EvalSpec | None = None     # firewalled measurement
    tier: Literal["lab", "release"] = "lab"

# ---------- the bridge ----------
@dataclass(frozen=True)
class PolicySpec:
    base: str                        # HF id + pinned revision
    bank: dict[str, AdapterSpec]     # named per-site interventions

@dataclass(frozen=True)
class AdapterSpec:                   # ONE typed intervention; lives in BOTH worlds
    kind: str                        # registered @adapter: "lora" | "soft_prompt"
                                     #   | "attn_bias" | "value_head" | ...
    site: str                        # canonical name, resolved at Phase 0 against
                                     #   site_space = SiteSchema ∪ bank exports
    init: Mapping                    # kind-specific: rank, tie, shapes
    trainable: bool = True
# sugar: lora(site, r) · soft_prompt(n, d) · attn_bias(spans) · ...

# --- SiteSchema: frozen data, trainer-half only (CONTEXT #25) ---
# SiteSchema(base, sites) where SiteMeta = (name, path, has_weight, shape,
# is_boundary). Compilers are plain functions: hf_schema(base) reads real
# shapes off AutoConfig (GQA-aware); fake_qwen_schema beside it for fakes.
# REACHABILITY IS A BUILD FACT, NOT SCHEMA METADATA: Mechanism is a closed
# StrEnum (punica | prompt_embeds | logits | side_attention | none); each
# Engine self-reports reachability(sites) -> {name: Mechanism} and the loop
# holds every served adapter's declared `serving` against that inventory at
# submit. Adapters may EXPORT sites the checkpoint lacks (a soft prompt
# exports prompt[:n]); resolution runs against the union. The schema
# fingerprint is written into the run manifest — attaching under a silently
# different schema raises.

# ---------- inference world ----------
@dataclass(frozen=True)
class GenSpec:                       # everything needed to COMPLETE rollouts;
    env: str                         #   scoring is NOT here (post pipeline, I9)
    tasks: str                       # content-addressed ("cas://<sha>/...")
    sampling: SamplingSpec = SamplingSpec()
                                     # SCIENCE: defines the behavior policy, so
                                     #   it hashes; per-processor overrides via
                                     #   PostProcessor.sampling

@dataclass(frozen=True)
class EvalSpec:                      # firewalled measurement: reads only IMMUTABLE
    tasks: str                       #   bundle versions + held-out tasks; writes
    every: int = 10                  #   only eval/ (crash recovery never reads it)
    env: str | None = None           # None → gen.env
    post: tuple[str, ...] = ()       # scoring pipeline over eval trajectories
    n_samples: int = 1
    pool: str = "main"               # which pool carries eval traffic (I5)

# ---------- training world ----------
@dataclass(frozen=True)
class TrajectorySource:              # training ALWAYS consumes sealed store data
    source: str                      # "live"               → this run's own gen,
                                     #                        lag ≤ schedule.max_policy_lag
                                     # "store://<run_id>"   → replay another run's
                                     #                        sealed waves (off-policy)
                                     # "cas://<sha>/…"      → static trajectory rows (SFT)
# one consumption contract for on-policy RL, off-policy distill, and SFT (I1);
# realized runner-side by the WaveFeed family (live / replay / static)

@dataclass(frozen=True)
class AlgoSpec:
    loss: str                        # registered @loss (requires travel with it)
    post: tuple[str, ...]            # THE ORDERED POST PIPELINE (replaces the old
                                     #   first-class rewards + advantages): per-group
                                     #   processors run after the seal, before the
                                     #   loss — verifier, judges, advantages are all
                                     #   PostProcessors (I9)
    optim: OptimSpec                 # optimizer + per-delta param-group overrides
    schedule: Schedule               # group_size, trajectories_per_wave, n_updates,
                                     #   epochs_per_wave, microbatch_tokens,
                                     #   max_policy_lag (estimator policy: the lag
                                     #   BUFFER; 0 = strict alternation)

# ---------- topology (semantics-neutral, I5) ----------
# member vocabulary — CLOSED at two; all else is traffic to pool NAMES:
#   pool(name, base=POLICY, tp=1, n=1, fraction=None)   # serves sample traffic
#   learner(fsdp=1, fraction=None)                      # differentiable fwd/bwd
@dataclass(frozen=True)
class GpuGroup:                      # THE unit of colocation (the bare word Group
    gpus: GpuSet                     #   is the DATA primitive, #22)
    members: tuple[Member, ...]      # co-resident; fractions = memory treaty
    sharing: Literal["concurrent", "sleep"] = "concurrent"
                                     # sleep: an exclusive arbiter group — one
                                     #   resident at a time (implies lag 0)

@dataclass(frozen=True)
class GpuConfig:
    groups: tuple[GpuGroup, ...]

@dataclass(frozen=True)
class Seeds:
    master: int                      # every seed derives: h(master, *path)

@dataclass(frozen=True)
class WarmStart:                     # start a NEW experiment from sealed state
    policy: str                      # "store://<run_id>@<version>"
    optim: Literal["load", "fresh"] = "fresh"
    map: Mapping[str, str] = {}      # source delta name → this bank's name
# WarmStart HASHES into run_id; the parent is recorded in the manifest.

# ════════════════════════════════════════════════════════════════════
# B. REGISTRIES — declaration half + compute half (I4)
# ════════════════════════════════════════════════════════════════════

@adapter("lora")                     # the bridge (I2) — one class per file
class Lora(Adapter):
    serving = Mechanism.PUNICA       # None = trainer-only, never served
    engine_plugin = None             # STRING naming an rlstack_engine module,
                                     #   for plugin mechanisms (side_attention)
    provides = frozenset()           # training-forward tensors the replay adds
    records = ()                     # per-token sampling-time facts (sealed, I6)
    def site_ok(self, meta) -> bool: ...       # predicate over SiteMeta
    def exports(self, spec) -> tuple[SiteMeta, ...]: ...  # sites this entry CREATES
    def params(self, sites, init): ...         # trainable objects
    def install_replay(self, model, params, sites): ...   # replay lowering
    def uninstall_replay(self, model, params, sites): ... # its EXACT inverse —
                                     #   what makes the learner multi-tenant (I8)
    def emit(self, params) -> bytes: ...       # payload the engine consumer reads
    def load(self, params, payload): ...       # emit's inverse (resume, warm start)
    def parity(self, harness): ...             # certificate (declared; wiring open)
# The bundle is the ONLY data channel between the halves; group_by_mechanism
# routes payloads to consumers. Heavy compute lives in a sibling module
# (lora_torch.py) imported lazily — the client library stays stdlib-only.

@environment("math_single_turn")     # INFERENCE WORLD; class per file
class MathSingleTurn(Environment):
    async def run(self, llm: PoolClient, task: Task) -> Rollout: ...
                                     # envs produce Rollouts; the RUNNER seals

@postprocessor("llm_judge")          # TRAINING WORLD; runs per group, after the
class LlmJudge(PostProcessor):       #   seal, before the loss (I9)
    produces = ("reward",)           # columns, one owner each
    consumes = ()                    # must be produced earlier in the pipeline
    token_level = ()                 # produced columns that are PER-TOKEN vectors
                                     #   (one float per generated token — the
                                     #   channel for teacher/hinted logprobs)
    pools = ("judge",)               # pools this processor SAMPLES from — checked
                                     #   against gpu_config at submit
    sampling = SamplingSpec(...)     # its own knobs (a judge is not the policy)
    async def process(self, group, data, llm) -> Mapping[str, Sequence]: ...
                                     # llm.pool(name) reaches ANY pool; judges are
                                     #   post processors that sample

@loss("grpo", requires=("advantage",))   # TRAINING WORLD; PURE MATH (I9):
def grpo(out: PolicyOutputs, b: TokenBatch) -> LossResult: ...
# requires ⊆ post columns ∪ BASE_RECORDS{behavior_logprobs, finish} ∪ bank
# records ∪ bank provides. Strings only — enforced at registration; a loss
# never routes work to metal. Every loss returns LossResult(loss, mean_ratio,
# logprob_gap) — the rails; logprob_gap is the trainer/sampler mismatch alarm.
# Builtin zoo: grpo · ppo (center_reward advantage) · gspo (sequence-level
# ratios) · sft · sdft (reward-weighted BC) · replay_distill (match a replayed
# run's recorded logprobs — the loss formerly named opd) · opd (#47: TRUE
# on-policy distillation — a live frozen teacher scores the student's own
# sampled tokens via the teacher_logprobs post processor; the loss reports
# sampled-token reverse KL and trains its score-function gradient, since the
# pathwise gradient of lp−teacher_lp degenerates to teacher-blind) ·
# self_anchor (lagged-record matching) · opsd (REAL hinted self-distillation:
# hinted_logprobs scores each trajectory's own tokens under privileged
# conditioning via PoolClient.score). A token_level teacher column may come
# from ANY declared pool, including one serving a DIFFERENT base (#47) — the
# preconditions are a shared tokenizer and a frozen non-policy pool serving
# its bare base; cross-base teacher_logprobs is the proven case (8B←32B).

# ════════════════════════════════════════════════════════════════════
# C. RUNTIME — hosts own metal; experiments are tenants; daemons on a blackboard
# ════════════════════════════════════════════════════════════════════

class Engine(Protocol):              # inference metal (VllmEngine / FakeEngine
                                     #   / RemotePool — the wire, I12)
    base: str | None                 # what this metal serves (checked at submit)
    tp: int                          # BUILD fact (I12): tensor-parallel width,
                                     #   shape-matched at bind, no wildcard
    def sample_tokens(messages, sampling, stop, bundle_id, seed)
        -> AsyncIterator[TokenEvent | FinishEvent]: ...   # engines speak TOKENS;
                                     #   the bundle is PINNED at submission (I8)
    async def score_tokens(messages, token_ids, bundle_id)
        -> tuple[float, ...]: ...    # logprobs of GIVEN tokens: one prefill
                                     #   pass, no decode, deterministic — the
                                     #   teacher half of pool traffic (judges
                                     #   sample, teachers score); prefill-
                                     #   shaped, the natural tenant of a
                                     #   future prefill-disaggregated pool
    def add_bundle(bundle): ...      # additive + idempotent (I8)
    def reachability(sites) -> Mapping[str, Mechanism]: ...
    def tokenize(text) -> tuple[int, ...]: ...

class Learner(Protocol):             # training metal (TorchLearner / FakeLearner)
    fsdp: int                        # BUILD fact (I12): shard width, attested
                                     #   against LearnerMember.fsdp at submit
    def install(tenant, spec, resolved_sites): ...        # additive per tenant (I8)
    def forward_backward(tenant, batch) -> TrainStats: ...
    def optim_step(tenant): ...
    def emit(tenant) -> Emitted: ...
    def load(tenant, adapters, optim): ...

# THE HOST is an atomic purposed partition (I12): born with a Partition
# (gpuset, devices, memory fraction) and Regimes it attests its metal against
# (engines by (base, tp); the learner by fsdp); >1 regime alternates on the
# host's own arbiter group. It owns engines, at most ONE learner, the
# arbiter, its journal store — and submission is how an experiment reaches it:
#   await host.submit(spec, schema, store,   # the experiment's OWN store (I10)
#                     remotes={...})         # pools served by OTHER hosts
#     bind    declared pools onto owned engines by base AT the declared tp
#     fit     refuse past capacity (regime residents attached at birth with
#             fraction 0.0 — every join is fraction-free; remotes never count)
#     attest  roster in memory + journal to hosts/<name>/log.jsonl
#     run     run_experiment_async under the host's shared arbiter
# run_experiment(spec, schema, store, engines, learner) remains the direct form.

# THE FLEET (runner/fleet.py) holds Metal + hosts and climbs the I12 ladder:
# demands_of(spec) reads (kind, base, shape, memory-hint) off gpu_config;
# sleep groups place as ONE unit (one multi-regime host), concurrent members
# per member (per-capability hosts). place() -> Plan(Join|Carve|Acquire);
# apply() executes the automatic rungs (carves journaled in fleet/log.jsonl);
# submit() runs beside the learner's host with RemotePools to the rest.

# THE WIRE (runner/remote.py): HostService executes pool verbs under the
# OWNING host's arbiter, addressed by capability (base, tp); Transport
# carries JSON-safe frames (async call: sample/score — admitted; sync ask:
# add_bundle/reachability/tokenize — admission-free by I8); RemotePool is the
# Engine protocol over it — a remote main pool is byte-identical to local.

# THE ARBITER is the physical half: object-keyed RESIDENTS (an engine, a
# learner) attach with an exclusive group (from sharing="sleep") or none;
# admit(resident) is the one verb work wraps itself in. Policy: sticky
# drain-until-blocked, with quantum (anti-thrash hysteresis) and max_wait
# (starvation handoff) knobs. Same-resident work OVERLAPS (alternation is
# about memory, not mutual exclusion). Leases died: per-experiment mutexes
# coordinate nothing.

# PHASE 2 IS A BLACKBOARD, not a choreography — daemons synchronized ONLY
# through the store (awaitable predicates; the ledger is the commit bus,
# waves/ the data bus):
#   Generator   awaits commit w-1-B → samples wave w at the newest committed
#               bundle (which version served is OPPORTUNISTIC within the lag
#               buffer, recorded per turn) → writes waves/<w>
#   Trainer     awaits waves/<u> (its WaveFeed) → post pipeline (admits the
#               engines of its pipeline's declared pools) → postdata →
#               flatten/broadcast/pack → fwd/bwd × epochs (admits the learner)
#               → optim_step → blobs → register bundle → LEDGER APPEND (the
#               commit point) → notify.   The ONLY ledger writer.
#   Evaluator   awaits commits mod eval.every → RECOMPILES the pinned bundle
#               from blobs (content addressing makes re-serving exact; crash-
#               lost evals backfill) → held-out episodes + eval pipeline →
#               eval/<u>
# Resume = re-run Phases 0-1: attach discards unsealed work, the learner loads
# the ledger tail, and the recompiled bundle's id must equal the tail's — the
# emit/load roundtrip check that makes cross-container resume exact.

# ---------- engine plugin package ----------
rlstack_engine/   # ships in the ENGINE image; imports one-way; bound by string
  plugin.py       # EnginePlugin ABC: probe / install / load / evict /
                  #   cache_salt / attend
  side_attention.py  # the one plugin mechanism (soft_prompt + attn_bias fused)
  batch_view.py   # the ONE version-pinned vLLM metadata shim
  certificates.py # CertificateKey(build, base, kind, mechanism) + cache (I7;
                  #   wiring open)

# ════════════════════════════════════════════════════════════════════
# D. DATA OBJECTS
# ════════════════════════════════════════════════════════════════════

Rollout → seal() → Trajectory:  TWO TYPES — mutable inference record, frozen
                  training record; the membrane is the type system (#23).
Trajectory → Group(key, …) → Wave:  a Group is ONE partial loss contribution
                  (GRPO group, preference pair); keys are ASSIGNED at wave
                  assembly, not derived from tasks. A Wave is one update.
Turn:             one REQUEST: token_ids, behavior_logprobs, finish, pinned
                  bundle_id + policy_version, seed, token_extras (recorded
                  per-token facts), turn_extras.
TokenBatch:       packed docs: token_ids · loss_mask · behavior_logprobs ·
                  segment_ids · doc_starts · post{column: per-token} ·
                  token_extras.  (No advantages field: advantages are a post
                  column like any other.)
Bundle:           compiled servable adapter set; content-hash = identity;
                  Bundle.pin(id, versions) is the payload-less request pin.
FlowGraph:        the canonical walk over the declarations (spec/flow.py);
                  validate queries it; dictionary.json serializes it (I11).
Store (ABC, one backend per file — local / modal_volume; S3 later):
    runs/<run_id>/manifest.json        identity (I3), written once
                  dictionary.json      the run's self-description (I11)
                  ledger.jsonl         the commit record, append-only
                  waves/<u>.jsonl.gz   sealed waves (trajectory rows)
                  postdata/<u>.json    the pipeline's columns per wave
                  adapters/<n>@<v>.bin optim/<n>@<v>.bin   (lockstep, per delta)
                  eval/<u>/            firewalled measurement
    hosts/<name>/log.jsonl             host journal (observability ONLY)
    cas/<sha256>/blob                  content-addressed objects
    — read-only PEEKS (manifest/ledger/dictionary) for observers (I10);
      attach-time recovery discards everything the ledger never committed.
Observer (rlstack/observe/):  read-only derivations over stores + journals —
    never attaches, never writes; store_for(locator) resolves where the
    locator does; CLI = python -m rlstack {hosts, runs, gpu}.
```

## 3. Examples

### Example 1 — A LoRA experiment, end to end (the shape of every run)

```python
from rlstack import (ExperimentSpec, PolicySpec, GenSpec, TrajectorySource,
                     AlgoSpec, EvalSpec, Seeds, OptimSpec, Schedule, GpuConfig,
                     GpuGroup, gpus, pool, learner, lora)

exp = ExperimentSpec(
    policy=PolicySpec(base="Qwen/Qwen3-0.6B",
                      bank={"pi": lora("layers.*.self_attn.*", r=16)}),
    gen=GenSpec(env="math_single_turn", tasks="cas://3fa9.../train.jsonl",
                sampling=SamplingSpec(temperature=1.0, max_tokens=12)),
    trajectories=TrajectorySource("live"),
    algo=AlgoSpec(loss="grpo", post=("verifier", "grpo_advantage"),
                  optim=OptimSpec("adamw", lr=1e-4),
                  schedule=Schedule(group_size=4, trajectories_per_wave=16,
                                    n_updates=30, microbatch_tokens=2048,
                                    max_policy_lag=0)),
    eval=EvalSpec(tasks="cas://8c31.../heldout.jsonl", every=5, n_samples=2,
                  post=("verifier",)),
    gpu_config=GpuConfig(groups=(
        GpuGroup(gpus(n=1), (pool("main", fraction=0.45),
                             learner(fraction=0.40))),)),
    seeds=Seeds(master=17),
)

host = Host("l4-0", engines=(VllmEngine("Qwen/Qwen3-0.6B", ...),),
            learner=TorchLearner(), store=journal_store)
report = await host.submit(exp, hf_schema(exp.policy.base), store=run_store)
```

One update: the Generator samples a wave of 16 (4 groups of 4) at the newest
committed bundle; token ids, behavior logprobs, and the pinned bundle land in
the sealed record. The Trainer's post pipeline runs per group — verifier
produces `reward`, grpo_advantage z-scores it into `advantage` — postdata is
stored, broadcast per-token, packed; fwd/bwd recomputes logprobs under trainer
kernels (the `logprob_gap` rail against the recorded behavior is the mismatch
alarm), optim_step bumps the delta version, blobs commit, the LEDGER APPEND
seals the update, and the freshly compiled bundle is already registered so the
next wave pins it. The Evaluator, awaiting every 5th commit, recompiles that
exact bundle from blobs and scores held-out tasks on its own pipeline. You
never named anything.

### Example 2 — Adding an objective: what new code exists

```python
# a judge-trained run: ONE new post processor class, declared end to end
@postprocessor("llm_judge")
class LlmJudge(PostProcessor):
    produces = ("reward",)
    pools = ("judge",)                       # submit-checked against gpu_config
    sampling = SamplingSpec(temperature=0.0, max_tokens=16)
    async def process(self, group, data, llm):
        judge = llm.pool("judge")            # any pool, by name
        ...                                  # sample, score, return one float/traj

exp_judge = replace(exp, algo=replace(exp.algo,
                    post=("llm_judge", "grpo_advantage")),
                    gpu_config=...)          # + pool("judge") — same engine may
                                             #   back both names (I8)

# a distillation loss: pure math over a column post produced (I9)
@loss("sdft", requires=("reward",))
def sdft(out, b): ...                        # reward-weighted BC — no new wiring

# per-token teacher signals ride the SAME contract: a processor declares
# token_level=("teacher_lp",), produces one float per generated token, and a
# loss requires=("teacher_lp",) consumes it token-aligned from b.post.
```

The Run verbs, the schedule, the store schema: untouched. The flow graph picks
up every new declaration once, so validation, the run dictionary, and the UI
see the new columns without any additional wiring.

### Example 3 — Soft prompt + learned attention bias: where kernel code lives

Unchanged in intent (#25): `soft_prompt` serves via native `prompt_embeds`
(zero engine code) and exports `prompt[:n]` sites; `attn_bias` declares
`serving = Mechanism.SIDE_ATTENTION` with `engine_plugin =
"rlstack_engine.side_attention"` — stock FlashAttention plus an exact
LSE-merged bias over the prompt-key partition, shipped as a plugin module in
the engine image, its cache_salt poisoning KV reuse across bundles. An
attn_bias without a soft prompt in the bank dies at submit (site-no-match
against the export). The parity certificate keyed by build fingerprint is the
I7 obligation; until wired, side_attention numerics stay behind the probe.
STATUS (#46, metal-checked): soft_prompt is PROVEN end to end — serving is a
BUILD FACT (VllmEngine(prompt_embeds=True); a build not asked for it reports
NONE at the boundary), replay rides the #44 row seam, and the control is
exact (rows that ARE real tokens' embeddings serve and replay bit-identically
to those tokens). attn_bias: the REPLAY half is proven (the bias rides a 4-D
attention mask, bit-identical to 2-D on the pinned transformers); the ROLLOUT
half is NOT PROVEN — vllm 0.28.0 exposes no LSE seam on its dense
FlashAttention path, so the plugin's probe names the missing symbols and
reachability honestly reports NONE. FlexAttention's score_mod is the seam
that would unblock it. #46 also generalized the replay side of I2: a replay
lowering may be a BOUNDARY (hooks around the base's forward — prepend rows,
widen the mask, trim the logits) rather than a site wrapper; its obligation
is ALIGNMENT — forward_backward's logprobs stay [len(batch)] against
batch.token_ids, and virtual positions never leak above the boundary.

### Example 4 — Environments (and where scoring went)

```python
@environment("tool_use")
class ToolUse(Environment):
    async def run(self, llm, task) -> Rollout:
        msgs = [user(task.prompt)]
        for _ in range(task.max_turns):
            turn = await llm.sample(msgs, stop=("</tool_call>",))
            msgs.append(turn.message)
            call = parse_tool_call(turn.message.content)
            if call is None: break
            msgs.append(tool_result(await execute(call)))   # loss_mask=0 spans
        return Rollout(task=task, messages=msgs, turns=[...])
```

Environments complete rollouts; they never score them. Scoring — verifiers,
judges, advantages — is the post pipeline (I9), running after the seal against
frozen records, with pool access for anything that must sample. The runner
seals; envs never do.

### Example 5 — Multi-GPU demand (unchanged shape, current names)

```python
gpu_config = GpuConfig(groups=(
    GpuGroup(gpus(n=16, nodes=2), (pool("main", tp=2, n=8),)),
    GpuGroup(gpus(n=8, nodes=1),  (learner(fsdp=8),)),
))
```

Pure demand; no provider names. `max_policy_lag=1` lets generation run a wave
ahead — which version served each turn is opportunistic within the buffer and
recorded (never prescribed). *(FSDP > 1 and multi-node are declared surface,
not yet exercised; the fakes and one-L4 paths are.)*

### Example 6 — One GPU, many tenants (the proven pattern)

Seven experiments — different losses, sources, and lag regimes — share ONE
resident engine (multi-LoRA batching across tenants) and ONE resident learner
(additive install + row routing, #44), submitted to one Host that binds, fits,
journals, and runs them under a shared arbiter. Per-tenant `logprob_gap`
staying at the bf16 kernel floor is the cross-tenant isolation alarm. This is
the stress matrix (deploy/stress_l4.py), green on an L4 end to end, including
kill/resume in-process and across containers.

---

*Archive: v2 of this spec (pre-metal) is in git history; the long-form design
rationale lives at the original artifact ("The Thin Wrapper"). The decision
log (agent-context/CONTEXT.md) remains the chronological authority — later
entries supersede this document until the next fold-in.*
