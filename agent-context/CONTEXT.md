# Agent Briefing — "The Thin Wrapper" (Samarth's RL harness)

You are an asynchronous specialist agent answering one question for Samarth, an RL
researcher. Your final message is relayed to him (possibly lightly edited). Write it as
a direct, mechanism-level technical answer in prose — no bullet spam, no preamble, no
"great question". Ground claims in source where possible.

## Canon (read before answering)

1. **/home/claude/rl-stack-spec.md** — the working spec (v2). THE canon: the contract
   (invariants I1–I7), the full primitive set, six worked examples. If your answer
   conflicts with it, say so explicitly and argue; never silently contradict it.
2. **/home/claude/rl-stack-design.md** — archived long-form rationale (library
   verdicts, speed/caching, kernel-compilation explainer, run hygiene, milestones,
   reading list). Background only; the spec supersedes it.
3. **/home/claude/ref/** — shallow clones for source-grounding (if still present):
   vllm, prime-rl, slime, vime, verl, tinker-cookbook, torchforge. Key verified facts
   are listed below; re-grep when precision matters.

## The project in three sentences

A thin, self-hosted, Tinker-shaped RL harness: five runtime verbs (sample /
forward_backward / optim_step / sync_weights / save-load) over declarative
content-hashed specs, with every swappable behavior a registered function
(declaration half + pure compute half). Two worlds joined by one membrane: the
INFERENCE world (envs — an episode is everything needed to complete and SEAL a
rollout; drives vLLM engine pools) and the TRAINING world (post-processing pipeline
[rewards, judges, advantages], loss, optim — consumes only SEALED trajectories from
the store); compiled adapter bundles are the only thing that crosses back. Adapters are
per-matrix deltas ("sites") compiled into content-hashed PEFT-shaped bundles served
via vLLM multi-LoRA; engines stay resident in a daemon; experiments are submitted
specs; backends (local/modal/skypilot) and GPU topology are semantics-neutral.

## Samarth's context

- Runs test-time-training research: arc-agi-ttt (latent-reasoning TTT on Modal:
  frozen Qwen3.5/3.6-35B-A3B MoE, diagonal-Gaussian posterior over 8×2048-d soft
  prompt tokens, one learned attention bias lowered to stock FlashAttention + exact
  LSE correction, GRPO over rollout waves, heavy bespoke deployment ceremony) and a
  TTT-Discover replication ("nanodiscover": Qwen3-8B, DeepSpeed+vLLM glue, erdos /
  circle-packing / ac1 tasks, 8 seeds × 64 rollouts × 50 epochs).
- Pain points that motivated the wrapper: hand-typed versioned names breaking runs;
  silent trainer/sampler numerical mismatch; per-run cold starts; FlashAttention/CUDA
  version breakage (hence invariant I7: pinned image, boot probe, certificates,
  fallback ladders).
- Interests: probabilistic adapters (distributions over delta values; per-token
  stochastic routing is PARKED — spec §5.4 of the archive; punica token_lora_indices
  is the un-parking hook), SDPO-family losses, self-judging, distillation both ways.

## Decision log (chronological; each is settled unless listed as open)

1. Tinker-shaped waist: five verbs; losses computed client/registry-side.
2. Adapter unit = per-matrix delta at a "site" pattern; tie= shares one (A,B) across
   matched sites; ≤1 delta per site; compile(bank, versions) → content-hashed bundle;
   serving via vLLM add_lora / multi-LoRA (punica).
3. Adapter kinds are typed by insertion point (weight/input/logit/attn-score/hidden);
   each registers: parameterization, ROLLOUT lowering (engine), REPLAY lowering
   (trainer), mandatory parity test. Policy is the ONLY two-world primitive (I2).
4. RouterSpec / per-token stochastic routing: PARKED. Bank composes deterministically.
5. Loss interface: pure fn (PolicyOutputs, TokenBatch) → Loss, with declared
   `requires` (ref_logprobs | Ref(version), values, entropies, hidden_states,
   Teacher(pool=...), Probe(build_fn)); runner plans no_grad passes (memoized per
   wave) and training-forward extras. Refs computed with TRAINER kernels.
6. Stage rule: needs gradients → loss; needs cross-rollout/archive context →
   advantage (wave-scope, detached, CPU, offline-re-runnable); needs to sample →
   env/reward (inference world). Rewards run per-episode BEFORE the seal (may
   sample — judges); advantages run once per wave AFTER the seal.
7. Backend axis: Backend protocol (up/attach/run/down/status), impls local · modal ·
   skypilot (NOT raw boto3); outside the spec; run store backend-neutral (S3/R2).
8. Placement → GpuConfig, semantics-neutral (I5); async_lag moved to
   Schedule.max_policy_lag (staleness = estimator policy, sibling of epochs_per_wave).
9. Colocation typed as Groups: Group(gpus(demand), members, sharing=concurrent|sleep).
   Member vocabulary CLOSED at two: engines(name, base, tp, n, fraction) and
   learner(fsdp, fraction). Everything else (rollout, eval, judge, teacher) is
   TRAFFIC routed to named pools. Fractions = per-member memory treaty; sleep =
   learner↔engines alternation (implies lag 0); feasibility checked at submit.
10. Eval: EvalSpec, firewalled measurement — reads only immutable bundle versions +
    held-out tasks; eval driver = coroutine subscribed to ledger commits.
11. Two-worlds split of data config: GenSpec (env, tasks, rewards, SamplingSpec —
    sampling is science, it hashes) vs RolloutSource ("live" | "store://run/…" |
    "cas://…") — one training-consumption contract for RL / off-policy distill / SFT.
    AlgoSpec and GenSpec both optional (generation-only and pure-offline runs).
12. Registered functions carry declarations: @reward(components=…),
    @advantage(consumes=…), @loss(requires=…) — joint validation at submit (I4);
    identity includes registered-code source hashes (I3); env/reward lifecycle hooks
    setup()/teardown() allowed (specs themselves NEVER do setup).
13. Runner lifecycle: Phase 0 validate → Phase 1 idempotent setup (leases, parity
    certificates, hooks) → Phase 2 loop (wave → seal → advantage → planned passes →
    fwd/bwd → optim_step → compile+add_lora). Resume = re-run Phase 1; unsealed waves
    regenerate from derived seeds.

## Verified engine facts (source-checked Aug 2026; grep /home/claude/ref/vllm to re-verify)

- punica kernels are per-token indexed: `token_lora_indices` in
  vllm/lora/punica_wrapper/; SGMV = sort-by-adapter segments + gathered tiles
  (`ram`), one grid plane per active adapter, over-launch + early-exit; slot banks
  `lora_a_stacked [max_loras, 1, max_rank, d]` zero-padded (heterogeneous ranks OK).
- Prefix-cache block hashes include lora identity (`_gen_lora_extra_hash_keys`) and
  prompt_embeds; `cache_salt` is a per-request field. sleep(level=1) = weights→CPU +
  drop KV; level=2 drops weights too; wake_up(tags).
- vLLM has first-party RL weight-transfer APIs (WeightTransferConfig, NCCL/IPC/HTTP,
  sparse-delta MVP, mid-generation pause-and-swap, batch-invariant mode): examples/rl/.
- prime-rl: MultiLoRALinear = grouped-GEMM multi-adapter training (torch._grouped_mm
  over adapter-contiguous offsets). vime: mis.py = truncated/masked IS helpers.
- Caches: VLLM_CACHE_ROOT (+startup_plan/), TORCHINDUCTOR_CACHE_DIR (+Mega-Cache),
  TRITON_CACHE_DIR, TORCH_EXTENSIONS_DIR, VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR.
- Trainer/sampler logprob mismatch is real and silently off-policy → always recompute
  under trainer kernels + truncated IS + logprob-gap alarm ("Your Efficient RL
  Framework Secretly Brings You Off-Policy RL Training"; GSPO = sequence-level ratios).

14. Optimizer state is a store primitive: optim/<name>@<v>.safetensors committed at
    every optim_step in lockstep with the ledger; PolicyVersion pins deltas AND
    moments. ExperimentSpec.init: WarmStart(policy="store://run@v", optim="load"|
    "fresh", map=...) declaratively warm-starts a NEW experiment from sealed state
    (hashes into run_id; parent recorded in manifest; Phase 1 performs the load).

15. SiteSchema (SETTLED — "canonical names, native compute"): per-base schema
    compiled at daemon boot mapping canonical names (blocks.N.attn.q, resid_pre.N,
    prompt[:k], ...) onto trainer module tree + engine mechanisms; open metadata per
    entry (has_weight, shape, engine_reachable_via: punica|prompt_embeds|logits|
    lse_patch|none); kinds declare site-requirement PREDICATES (no sort enum);
    Phase 0 resolves + validates; schema hashes into parity certificates. NO
    TransformerLens-style recompilation (verified impractical: eager unfused,
    O(S^2) pattern materialization, per-arch converters, numerics folding).

16. Engine mechanics (SETTLED, in spec §2B/§2C): AdapterKind = five-member protocol
    (site_ok predicate, engine_plugin STRING or None, params, install_replay, emit,
    parity); bundle = only data channel between halves (emit → registered consumer →
    slot buffers). Two packages: rlstack (library) / rlstack_engine (engine code only,
    bound by string). Registration is AMBIENT via vLLM load_general_plugins entry
    points in every engine/worker process; runner levers = install / select / feed.
    Insertion ladder: request field < logits-processor < registered model subclass
    (one generic Instrumented<Arch> per family) < registered attention backend
    (attn_bias = stock FA w/ LSE + tiny partition attn + merge_attn_states) < image-
    build patch. Pools keyed by (base, mechanism set); backend selection is boot-time
    (reboot to change); image=POSSIBLE, boot args=ACTIVE, bundle=LOADED, request=USED.
    Cadences: image (engine code, certified) vs submit (everything else).

17. Phase A IMPLEMENTED at /home/claude/rlstack (mirrored to ~/Coding/rlstack on
    Samarth's machine): stdlib-only (PyPI/apt unreachable from this container);
    modules canonical/specs/registries/siteschema/validate/data/store + 240 unittest
    tests, all green. WarmStart.map direction = source-name -> THIS-bank-name
    (validator checks VALUES against bank). Known Phase-B upgrades: parquet rollouts
    (now jsonl.gz behind a format field), safetensors blobs (now .bin), numpy for
    Wave/TokenBatch. Agents must READ this code before proposing changes to it.

18. READABILITY REWRITE of Phase A (Samarth: "i want to look at a piece of code
    and just understand what's going on immediately"). Now binding, codified in
    /home/claude/rlstack/STYLE.md: spec vocabulary in executable positions;
    concrete types, NO duck-typing/reflection (we own both sides); one named
    function per rule (validate.py = 17 named checks in a CHECKS table, same 22
    issue codes); typed registrations (LossDef.requires, RewardDef.components,
    AdvantageDef.consumes, KindDef.instance — Registration/meta dicts are GONE);
    no freeze ceremony in specs.py (canonical_json key-sorts at hash time; plain
    dicts on frozen dataclasses); DataError is a plain ValueError; seal() swaps
    in MappingProxyType views (sealed-mapping writes raise TypeError now, not
    SealError; attribute rebinding still SealError); store recovery decomposed
    into named steps with the principled blob rule (a delta the ledger never
    committed has ceiling 0, not the global-max fallback); AdapterKind
    subclasses must be zero-arg constructible (decorator instantiates one shared
    instance). tests/test_examples.py transcribes SPEC.md §3 as the readable
    layer; 236 tests green. Any agent touching rlstack MUST read STYLE.md first.

19. FOLDER REORG (settled, STYLE.md rule 8): rlstack/{spec,policy,inference,
    training,data}/ + registry.py (mechanism/index ONLY — registrations live
    with their worlds; a name exists iff its module was imported; global
    singleton tables kept deliberately over per-submit registries). data/ is
    the membrane: dumb, estimator-free, imports NO other rlstack package;
    worlds never import each other — tests/test_architecture.py enforces the
    import DAG via ast. Flat absorbed behavior_logprobs (flatten captures the
    whole record in one pass; per_token_behavior_logprobs deleted); pack takes
    (flat, advantages) pairs; merge_components deleted; bump lives in
    data/store.py; grpo_group_norm has its REAL body in training/advantages.py
    (group_normalize beside it). specs.py deliberately NOT split across world
    folders (declarative surface reads as one file). Samarth accepted
    canonical.py's code-hash identity as-is (tripwire semantics). 238 tests.
    Note: the desktop-bridge Linux VM has Python 3.10; rlstack needs >=3.11
    (StrEnum) — run the suite in the container, not via device_bash.

20. PHASE B PLAN (settled): B1 runner-vs-fakes (DONE, this container) → B2 torch
    learner + real grpo + HF SiteSchema + peft-format bundle emit (written here,
    runs where torch lives) → B3 vLLM behind Engine + first real run
    (Qwen3-0.6B + synthetic arithmetic; GPU venue TBD). Single-process v0,
    daemon in Phase C. Container package index still blocked (numpy only).

21. B1 IMPLEMENTED (runner/ + policy/compile.py; 273 tests green). SPEC DELTAS
    pending fold-in to canon (rl-stack-spec.md + artifact):
    (a) Engine speaks TOKENS: sample_tokens(...) -> stream of TokenEvent(
        token_id, logprob, text_delta, extras) ending in FinishEvent(finish,
        stop_hit, turn_extras); a request's bundle is PINNED at submission
        (vLLM per-request lora + punica token_lora_indices), so add_bundle
        never disturbs in-flight requests. SampleClient assembles Turns.
    (b) Turn = one REQUEST (pinned bundle family, seed, contiguous KV), NOT
        "one policy identity": per-token realization lives in
        Turn.token_extras columns (e.g. adapter_draw for stochastic routing);
        per-request sampling facts in turn_extras (e.g. latent_draw).
        Recorded, never re-derived (I6); columns flow through flatten/pack to
        the loss.
    (c) AdapterKind declares `records` (mirror of provides): records = FACTS
        from sampling time (frozen at seal), provides = TENSORS from training
        time (recomputed each forward). install_replay reads its own recorded
        columns (replay-same-modules-per-token is the automatic DEFAULT);
        losses name recorded columns in requires only for direct consumption
        (mixture logprobs etc.). Loss string requires resolve against
        BASE_PROVIDES ∪ BASE_RECORDS{behavior_logprobs, finish} ∪ bank
        provides ∪ bank records.
    (d) Commit protocol: rollouts → blobs → LEDGER (commit point) →
        add_bundle → eval; resume-equivalence is TESTED (crash at 4 points,
        fresh objects, byte-identical run dir). Seed tree: h(master, *path).
    (e) math_single_turn + verifier are real builtins; grpo_group_norm real.
    Examples/arith_fake.py = Example 1 on fake metal. Agents: read
    runner/interfaces.py before touching Engine/Learner surfaces.

22. DATA PRIMITIVES REORG (settled, Samarth-directed): Trajectory (one episode)
    → Group(key, trajectories) = everything needed to compute ONE PARTIAL
    CONTRIBUTION to the loss (GRPO advantage group, preference pair) → Wave
    (groups) = everything for one gradient step. Group keys ASSIGNED at wave
    assembly, NOT derived from task identity (unblocks TTT many-groups-of-one-
    task); wave rows carry "group" through the store; advantages iterate
    wave.groups (zscore helper; group_ids parallel arrays GONE). SPEC DELTA:
    topology Group RENAMED GpuGroup (GpuSet/GpuGroup/GpuConfig family) — the
    plain word Group is the data primitive now. WaveSource family in
    runner/sources/ (deliberately runner, not data/ — live SAMPLES): base.py
    WaveSource.wave(update, bundle) -> Wave; live.py / replay.py (store://run,
    verbatim parent waves) / static.py (cas:// trajectory rows, singleton
    groups, deterministic cycling); wave_source_for dispatches on
    RolloutSource.source; the loop consumes any source through one call and
    persists yielded waves into its own rollouts/ (runs self-contained).
    280 tests green. Also settled earlier this arc: Task assumptions audit —
    id-uniqueness + digest-level eval-overlap checks still TODO (cheap).

23. TYPED CLASS TEMPLATES + ROLLOUT/TRAJECTORY SPLIT (settled, Samarth-
    directed). TERMINOLOGY IS CANON: "rollout" = inference-specific (an episode
    in progress), "trajectory" = training data (sealed). They are now TWO
    TYPES: Rollout (inference/rollout.py, mutable dataclass) .seal() ->
    Trajectory (data/, frozen, read-only views) — the membrane is enforced by
    the type system; SealError deleted. Registered surfaces are class
    templates, one class per file, each folder with a base:
    - policy/adapters/ : Adapter base ("adapter kind" RETIRED — @adapter,
      ADAPTERS, AdapterDef, SERVING_MODES; AdapterSpec.kind FIELD name kept);
      lora/soft_prompt/attn_bias/value_head each own file.
    - inference/environments/ : Environment ABC — invariant: async run(llm,
      task) -> Rollout (runner seals, envs never do); SampleClient is a
      consumer-defined Protocol in base.py (runner's concrete impl =
      EngineSampleClient in runner/waves.py, renamed from runner/rollout.py).
    - inference/rewards/ : Reward ABC — async score(rollout, llm) ->
      tuple[float, ...] aligned to declared `components` (length-checked by
      runner; declaration-body mismatch fails loudly).
    Losses/advantages remain registered FUNCTIONS (not class-ified).
    code_hashes key prefixes now environment:/reward:/adapter:/loss:/
    advantage:. Issue code unknown-adapter (was unknown-adapter-kind).
    279 tests green. Agents: read the base.py of the folder you touch.

24. FIVE-PRINCIPLE REDESIGN (settled, Samarth-directed; supersedes parts of
    #6, #11, #12, #23). The five: (1) environments AND post-processors talk to
    the ENTIRE inference pool, not one client; (2) adapters are attached so all
    experiments share engines in parallel (multi-LoRA generalized to
    multi-adapter); (3) stores are an ABC with universal layout logic,
    subclassed per backend; (4) advantages are NOT first-class — they are
    per-group post-processing declared in the experiment; (5) rewards move
    into training as part of that same post-processing.
    - POST PIPELINE (replaces rewards + advantages as separate stages/
      registries): training/post/ — PostProcessor ABC with class attrs
      `consumes`/`produces` and `async process(group, data, llm) ->
      {column: vector len(group)}`; @postprocessor → POST registry. Builtins:
      verifier (produces reward), grpo_advantage (consumes reward, produces
      advantage via zscore), constant. Ordered pipelines declared as
      AlgoSpec.post / EvalSpec.post (tuples of names); runner/post.py
      run_pipeline runs per group, seeded derive(master, phase, update,
      group.key, name), gathers groups concurrently, concatenates in wave
      order. Columns land as postdata/<update>.json in the store, broadcast
      per-token into TokenBatch.post; losses `requires` name columns
      (grpo requires=("advantage",); TokenBatch.advantages field GONE).
      STAGE-RULE REVISION (amends #6): post-processors MAY SAMPLE (judges) —
      allowed because they produce postdata and can never mutate the sealed
      record; "needs to sample" no longer forces the inference world.
      inference/rewards/ is DELETED; Trajectory/Rollout have no
      reward_components; GenSpec = env + tasks + sampling only.
    - POOL ACCESS: SampleClient Protocol moved to neutral rlstack/client.py
      (`sample(messages, stop) -> Turn`, `pool(name) -> SampleClient`) so both
      worlds may type against it without crossing the import DAG. Concrete
      EngineSampleClient in runner/client.py over Pools = Mapping[str,
      (Engine, Bundle)], ONE shared per-episode seed counter across pools —
      hinting pipelines / teachers / judges reach any pool by name.
    - MULTI-TENANCY is now a stated Engine interface invariant
      (runner/interfaces.py docstring): bundle registration is additive,
      requests pin (bundle_id, seed), engines batch across tenants; tested by
      test_loop.py::test_two_experiments_share_one_engine (10 bundles on one
      FakeEngine, ledgers byte-equal to private-engine runs).
    - STORE ABC: data/stores/ — base.py Store ABC = six abstract byte verbs
      (_read/_write/_append_line/_exists/_list/_delete + _sweep_partial hook)
      under universal concrete orchestration (key tree runs/<id>/..., ledger
      append = commit point, torn-tail repair, _discard_unsealed crash
      recovery, cas_put/get, fingerprint); local.py LocalStore = fsync
      reference backend (AWS etc. subclass later without touching layout).
    - VALIDATE: unknown-post / post-unwired (consumes must be produced
      earlier in the pipeline) / post-collision (one owner per column)
      replace unknown-reward/unknown-advantage/unknown-component/
      component-collision; loss requires satisfied from BASE_PROVIDES ∪
      BASE_RECORDS ∪ bank provides/records ∪ pipeline produces.
    - code_hashes keys now "postprocessor:<name>" (reward:/advantage: gone).
    286 tests green; examples/arith_fake.py runs e2e; commit "Five
    principles: ..." at HEAD, mirrored to ~/Coding/rlstack (obsolete files in
    _to_delete/). Agents: the reading path is training/post/base.py →
    rlstack/client.py → data/stores/base.py → runner/loop.py.

25. SITE TREATY REDESIGN (settled, Samarth-directed; supersedes the schema
    parts of #15 and the serving strings of #16). Four moves:
    (a) SiteSchema = FROZEN DATA, trainer-half only: SiteSchema(base, sites)
    dataclass; SiteMeta = (name, path, has_weight, shape, is_boundary) —
    engine_reachable_via and the `extra` bag are GONE. The pattern grammar is
    canon (module-level named functions; one resolver `resolve(sites,
    pattern)` — the spec's site strings hash into run_id so their meaning
    cannot vary by backend). Compilers are plain functions: fake_qwen_schema
    (nee make_fake_qwen_schema/FakeSiteSchema, both deleted; base is now a
    required kwarg); hf_schema lands in B2. New check schema-base-mismatch
    (schema.base == policy.base, in CHECKS); schema.fingerprint() is written
    into the run manifest, so attaching under a silently different schema
    raises ManifestMismatch.
    (b) REACHABILITY IS A BUILD FACT: Mechanism StrEnum (punica |
    prompt_embeds | logits | side_attention | none) replaces SERVING_MODES
    strings and reachable_via; Adapter.serving: Mechanism | None; Engine
    protocol gains reachability(sites) -> {name: Mechanism} (self-reported
    inventory — vLLM adapter computes it from packed_modules_mapping/backend/
    plugins). check_sites_reachable_on lives OUTSIDE CHECKS: the loop runs it
    at Phase 0 against the main pool's inventory, before open_run. FakeEngine
    reports a realistic inventory (plugins= kwarg gates side_attention).
    "lse_patch" renamed side_attention; engine_plugin =
    "rlstack_engine.side_attention".
    (c) ADAPTERS EXPORT SITES: Adapter.exports(spec) -> tuple[SiteMeta,...];
    SoftPrompt exports prompt[:n] AND queries -> prompt[:n] (they are NOT
    base-model sites — the old schema hand-listing them was a fudge);
    resolution runs against site_space(spec, schema) = schema ∪ bank exports,
    in validate AND in loop Phase 1 (learner.install's resolved map). An
    attn_bias without a soft prompt in the bank dies with site-no-match.
    (d) rlstack_engine/ SIBLING PACKAGE (what engine_plugin strings name;
    ships in the engine image; imports one-way, enforced by
    test_architecture): plugin.py EnginePlugin ABC — probe (I7 boot probe over
    build.symbols), install(seam), load/evict (payloads -> per-slot banks,
    multi-tenant from day one), cache_salt (bundle_id into the prefix-cache
    key — plugins change hidden states, so KV reuse across bundles must be
    poisoned), attend(view, q, out, lse); batch_view.py BatchView = the ONE
    version-pinned metadata shim (token_slot gather index; from_vllm is B3);
    slots.py SlotTable; side_attention.py SideAttention (mechanism plugin,
    consumes BOTH soft_prompt + attn_bias payloads — a mechanism compiles its
    adapters JOINTLY, the punica fragment-merge rule stated as a rule;
    lifecycle real, numerics B3 behind probe); certificates.py
    CertificateKey(build_fingerprint, base, kind, mechanism) + cache (parity
    re-runs on any build change). Grounded in vLLM source (ref/vllm):
    register_backend(AttentionBackendEnum.FLASH_ATTN) override is the
    sanctioned seam; the FA backend already runs return_softmax_lse=True and
    ships merge_attn_states — the patch attaches at that internal layer.
    313 tests green; commit "Site treaty: ..." mirrored to ~/Coding/rlstack.

26. B2/B3 ON MODAL (Samarth-directed: "prove the entire loop on Modal first";
    venue = one L4, Qwen3-0.6B — fits with room, ~11GB vLLM + ~2GB trainer
    of 24GB). What landed:
    - hf_schema(base) compiler (siteschema.py, lazy transformers): real
      shapes off AutoConfig incl. GQA (k/v → num_key_value_heads·head_dim).
    - Bundle gains `kinds` (payload name → adapter kind; routing metadata,
      NOT identity — bundle_id formula unchanged); group_by_mechanism(bundle)
      in policy/compile.py is THE dispatch input for any engine's add_bundle.
    - Adapter protocol gains `load(params, payload)` (emit's inverse; resume
      and warm-start route through it). LoRA compute half in
      policy/adapters/lora_torch.py (lazy import from Lora's methods):
      LoraLinear wrap (verb 1), A ~ N(0,1/r) seeded per site / B = 0 (v0
      delta IS the base on both sides), peft-format emit (alpha == r →
      scaling 1; keys = "base_model.model." + path), merge_fragments for the
      bundle fusion (disjoint keys by the one-delta-per-site rule).
    - grpo loss is REAL (training/losses.py, lazy torch): token-level
      PPO-clip surrogate over batch.post["advantage"], IS ratio against
      RECORDED behavior logprobs; returns LossResult(loss, mean_ratio,
      logprob_gap) — logprob_gap is the parity alarm v0.
    - TorchLearner (runner/learners/torch_learner.py): frozen bf16 HF base,
      model.eval() (replay = exact recompute, no dropout), adapters build/
      install through their OWN kind halves, one AdamW per entry (optim
      blobs map 1:1 onto optim/<name>@v), docs forwarded one at a time (v0),
      logprob of token t from logits t-1, position 0 zero.
    - VllmEngine (runner/engines/vllm_engine.py): AsyncLLMEngine built
      LAZILY inside the running loop; prompts = per-message tokenize then
      concat ids via TokensPrompt (exactly what flatten re-tokenizes — NO
      chat template, v0, logged open thread); SamplingParams(seed, stop,
      logprobs=0); cumulative outputs → TokenEvents (text delta attributed
      to last new token; concatenation exact); add_bundle = the match on
      group_by_mechanism (punica: merge → adapter dir + LoRARequest;
      idempotent; empty payloads → serve base); reachability: PUNICA on
      weighted sites, NONE else.
    - ModalVolumeStore (data/stores/modal_volume.py): LocalStore over the
      mounted Volume; volume.commit() at manifest write, LEDGER APPEND (the
      commit point — persists the whole staged update atomically-ish), and
      eval writes. Volume passed as duck-typed .commit() object (no modal
      import; tested with a recorder).
    - deploy/modal_app.py: image (pip vllm/transformers/safetensors,
      UNPINNED with TODO(I7) — run prints resolved versions, pin after first
      green), volume "rlstack-store" at /store, run_tests (full suite
      in-image, CPU), run_arith (L4; spec: lora r=16 on layers.*.self_attn.*,
      grpo, verifier+grpo_advantage, eval every 2; engine 0.45 mem fraction +
      learner 0.40; prompts "What is a+b? The answer is", max_tokens 12).
    - Known v0 simplifications: no chat template; uniform LoRA rank per
      merged bundle config; byte-identical resume is a FAKES-suite property
      (CUDA nondeterminism) — real-metal invariant is small logprob_gap;
      engine+learner colocated single process.
    320 tests green locally (fakes; torch/vllm/modal all lazy — pip is 403
    in the dev container, so first real execution is Samarth's
    `modal run deploy/modal_app.py::run_tests` then `::run_arith`).

27. BLACKBOARD RUNNER (settled, Samarth-directed: "per experiment, all these
    gpus running async, synchronized via the common data source they read";
    supersedes the sequential Phase-2 loop of #13/#21). Runner Phase 2 is now
    DAEMONS on a blackboard, not a choreography. runner/daemons/: Generator
    (awaits commit w-1-B, samples wave w at the NEWEST committed bundle,
    writes rollouts/<w>), Trainer (awaits rollouts/<u> via its WaveFeed; post
    → postdata → pack → fwd/bwd → blobs → register bundle → LEDGER APPEND →
    notify; the only ledger writer; bundle registered BEFORE the commit line
    so readers of the commit can pin immediately), Evaluator (awaits ledger
    entries mod eval.every; RECOMPILES the pinned bundle from store blobs —
    content addressing makes re-serving any committed version exact — so
    crash-lost evals backfill; skips has_eval(u)). Nobody calls anybody:
    runner/signals.py RunSignals = awaitable predicates over the store
    (notify + poll; store stays source of truth), the ledger is the commit
    bus, rollouts/ the data bus.
    - LAG IS OPPORTUNISTIC, NOT PRESCRIBED (Samarth rejected fixed-lag
      pipelining): Schedule.max_policy_lag = B is a BUFFER BOUND; which
      version served wave w is scheduling, recorded per turn, lag ∈ [0,B].
      B=0 degenerates to strict alternation and reproduces the old runner
      BYTE-FOR-BYTE (resume-equivalence suite unchanged and green; example
      bundle ids identical pre/post refactor). B>0 runs are not
      generation-reproducible by design; training from recorded data still is.
    - COLOCATION IS ORTHOGONAL: async = logical (store awaits); colocation =
      physical (runner/lease.py). Roles wrap metal work in lease.held(
      resource) where resources are ENGINE/LEARNER (resource-keyed, not
      role-keyed — a judge-bearing post pipeline held by the trainer wakes
      the ENGINE). OpenLease (dedicated/concurrent-fractions) never blocks;
      ExclusiveLease (GpuGroup.sharing="sleep") = fair mutex with STICKY
      resident + per-resource wake/evict hooks (vLLM sleep/wake and learner
      offload wire in here; no-ops v0). leases_for(spec) derives the map;
      "2 daemons alternating, flexible acquisition conditions" = overridable
      condition methods (may_generate / next_rows / due_updates) + hooks.
    - sources/ became WaveFeed (obtain(update) -> rows|None): the trainer
      ALWAYS reads waves from its own run's rollouts/; live = Generator
      wrote them, replay/static = feed copies them in on first request.
      LiveRollouts' sampling moved into the Generator; runner/eval.py deleted
      (Evaluator daemon); FakeEngine.add_bundle now idempotent (the
      invariant, stated). NAMING: daemon base class is Daemon (rlstack.Role
      stays the message-role enum). 327 tests green.

## Open threads (do NOT treat as settled; flag when your answer touches them)

- Identity rings: should GpuConfig (and EvalSpec) leave the run_id hash and become
  submit-time config (resume-across-hardware keeps identity)?
- Schedule: split statistical (group_size, epochs_per_wave, max_policy_lag) from
  engineering (microbatch_tokens) knobs?
- Wave / ArchiveContext typing for the advantage stage (least-specified interface;
  matters for tree-credit / TTT-Discover-style work).

## Ground rules for you

- Do NOT edit rl-stack-spec.md, rl-stack-design.md, or any artifact — the main
  session owns canon. If your answer implies a spec change, end with a clearly
  marked "PROPOSED SPEC CHANGE:" section stating the exact diff in words.
- Prefer primary sources (the ref/ clones, official docs via WebFetch/WebSearch)
  over memory for version-sensitive claims; today is late August 2026.
- Answer at the mechanism level; use the project's vocabulary (seal, membrane,
  bundle, pool, lowering, wave, delta, site, certificate).
- Your ENTIRE final message is the answer payload — no meta-commentary.
