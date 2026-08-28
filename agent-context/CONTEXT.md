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

28. FIRST REAL RUN GREEN (2026-08-27, one Modal L4, Qwen3-0.6B + LoRA r=16 +
    GRPO): run 442bcf59b515 (seed 17, rewards .500/.812/.812/.875, held-out
    eval 0.75@2 → 0.9375@4) and pinned-image confirmation 7a5fc6461d7d
    (seed 18, rewards .375/.438/.750/.938). What real metal forced to change
    — TWO fixes, NEITHER in the predicted files:
    (a) the ~/Coding mirror's rlstack/inference/__init__.py was STALE (still
        imported inference.rewards, deleted in #24) — mirror drift, fixed;
    (b) vLLM 0.28's default sampler is FlashInfer with JIT-compiled kernels
        — needs nvcc at runtime, which debian_slim lacks, so EngineCore died
        at the FIRST sample step (RuntimeError: Could not find nvcc). Fix is
        deploy-only: VLLM_USE_FLASHINFER_SAMPLER=0 in the image env (native
        torch sampler; per-request seeds unaffected).
    vllm_engine.py and torch_learner.py ran UNCHANGED on first contact:
    AsyncLLMEngine is the V1 AsyncLLM alias in 0.28, generate(prompt, params,
    request_id, lora_request=) holds, logprobs=0 dict indexing holds,
    TokensPrompt/LoRARequest/peft-merge all held; punica path confirmed real
    (PunicaWrapperGPU, triton lora kernels JIT once per shape — latency blip
    only). I7 PINS (in deploy/modal_app.py): vllm==0.28.0, torch==2.13.0
    (default PyPI Linux wheel, reports 2.13.0+cu130), transformers==5.16.1.
    LOGPROB_GAP CALIBRATION on this stack: ledger gap (max over microbatches
    of masked means) sits at 0.022–0.033 — ABOVE the ~1e-2 folklore, but
    update 1's wave in both runs sampled at v0 where B=0 makes trainer and
    sampler mathematically identical, and it measured 0.025/0.033 THERE ⇒
    that band is the vLLM-FA2-vs-HF-sdpa bf16 kernel floor for this 0.6B
    config, not misalignment; it stays flat as the LoRA trains. The alarm is
    GROWTH above the update-1 floor, not the floor itself.
    Watch items from the logs (not bugs today): transformers 5 shims
    torch_dtype with a deprecation (switch to dtype= eventually); vLLM
    deprecates raw prompts to InputProcessor in favor of Renderer.render_*
    (touches the TokensPrompt path); ModalVolumeStore's blocking
    volume.commit() inside the async loop draws an AsyncUsageWarning (works;
    .aio() is the clean fix but that is a data/-layer change, deliberately
    not hotfixed).

29. VOCABULARY CONSISTENCY PASS (settled, Samarth-directed: "keep the
    rollout <-> inference and trajectory <-> learner pattern consistent
    wherever possible"). The #23 treaty words now hold in every executable
    position; "rollout" survives ONLY on the inference side (the Rollout
    type, the "rollout" seed phase, GenSpec's docstring). SPEC DELTAS, all
    three IDENTITY-AFFECTING (they hash into run_id — every run_id changes;
    the two pre-rename L4 smoke runs on the volume are orphaned, left in
    place):
    - RolloutSource RENAMED TrajectorySource (its own docstring always said
      "what training consumes — always sealed store data");
      ExperimentSpec.rollouts -> ExperimentSpec.trajectories.
    - Schedule.rollouts_per_wave -> trajectories_per_wave (it is the
      statistical wave-size knob, sibling of group_size).
    - Store section rollouts/ RENAMED waves/ — the per-update artifact is
      exactly one serialized Wave (#22's primitive); RunHandle verbs
      write_rollouts/read_rollouts -> write_wave/read_wave; the ledger's
      wave summary key "rollouts" -> "trajectories".
    validate: check_live_rollouts_have_gen -> check_live_trajectories_have_gen,
    issue field "trajectories.source" (code "live-without-gen" unchanged).
    StaticFeed KEPT and clarified: it is the cas:// arm of TrajectorySource
    (the SFT / offline-distillation shape) — a fixed file of sealed
    trajectory rows sliced into singleton-group waves; it already satisfied
    "the learner only consumes trajectories", the offense was naming.
    Deliberately untouched: the seed-tree phase string "rollout" (it seeds
    SAMPLING — correct word, and identity-bearing) and
    tests/test_trajectory.py::test_unsealed_rollouts_are_refused (an
    unsealed record IS a rollout; the name states the treaty). 327 green;
    examples/arith_fake.py e2e on fakes.

30. L4 STRESS MATRIX GREEN (2026-08-27, deploy/stress_l4.py, Samarth-directed
    "stress test heavily ... until you're confident"). Five stages, two
    containers, one engine per container; 34 checks, all properties held.
    WHAT WAS ADDED (extension points, not abstraction changes): six
    registered losses in training/losses.py — ppo (value-free clip, pairs
    with new post center_reward = mean-baseline), gspo (SEQUENCE-level
    length-normalized IS ratio per doc, via batch.doc_starts), sft (behavior
    cloning), sdft (reward-weighted BC on own samples, requires "reward"),
    opd (squared matching of RECORDED teacher logprobs, replay data), opsd
    (same objective on live data under lag — the lagged self as teacher);
    shared helpers _tensors/_rails. SPEC DELTA: the SPEC.md §3 Example-2
    loss "ppo" (critic + GAE, still a stub) RENAMED ppo_critic — the
    runnable value-free clip owns the name "ppo" now. ONE abstraction
    change, justified: runner/loop._run made PUBLIC as run_experiment_async
    (run_experiment wraps it) — the multi-tenancy invariant is only
    expressible with N experiments gathered in ONE event loop around one
    engine; the sync wrapper couldn't say it, and Phase C needs the async
    form anyway.
    RESULTS (Qwen3-0.6B + LoRA r=16, one L4, vllm 0.28.0):
    - solo grpo 30 updates: gap ceiling 0.0300, reward .50→.88, 6/6 evals;
    - SIX CONCURRENT TENANTS on one engine, joins staggered 30s: sft (static
      cas:// of the grpo run's 449 correct rows), opd (replay
      store://grpo-run), ppo, gspo, sdft, opsd — 24/24 ledgers and 4/4
      evals each; IS-family gaps at the kernel floor (.027-.034) = NO
      cross-adapter contamination; opd's update-1 gap 0.030 (teacher v0 ==
      fresh student) is the direct no-contamination proof, then its squared
      loss pulls .166→.09 (the objective visibly working);
    - GAP SEMANTICS BY FAMILY (calibrated, now encoded in the harness):
      IS-corrected losses alarm at 0.15 over the ~0.03 bf16 kernel floor;
      sft/opd gaps MEASURE teacher-student distance (growth = training, a
      mixup would read ~5+), so their alarms are loose (1.0/0.5);
    - POLICY LAG: opsd with max_policy_lag=2 + epochs_per_wave=2 realized
      lag histogram {0: 16, 1: 368} turns — generator genuinely ran ahead,
      bound respected, recorded per turn (I6); B=0 tenants verified
      strictly on-policy from the sealed record;
    - RESUME: in-process cancel at ~u17 under sharing="sleep"
      (ExclusiveLease) → re-attach → 40/40, gap unchanged (.031);
      CROSS-CONTAINER cancel at u12 → cold process attach → 24/24 — Phase 1
      recompiled the tail bundle to the IDENTICAL content-addressed id from
      emit/load-roundtripped blobs (generation dies loudly otherwise);
      a second full harness pass attached every completed run and no-oped
      (idempotent resubmit on real metal);
    - evals: every tenant every 5 updates, evaluator backfill after both
      resumes.
    KNOWN LIMITS RECONFIRMED (not new, not blocking): ppo_critic
    (values/GAE) and any Ref/Teacher-requiring loss can't run on v0 —
    PolicyOutputs carries logprobs only and planned passes are unbuilt;
    vLLM lora LRU beyond max_loras=8 concurrent was exercised only lightly
    (≤7 tenants+eval pins in flight).

31. TIDYING PASS (settled, Samarth-directed). Identity-stable throughout
    (source_hash hashes each registered function's own source; the fake
    example's bundle ids are byte-identical pre/post):
    - training/losses.py SPLIT into training/losses/ mirroring post/: base.py
      (PolicyOutputs, LossResult, token_tensors, rails — the loss contract)
      + one file per registered loss; package __init__ registers builtins.
    - runner/client.py + runner/waves.py MERGED into runner/traffic.py
      ("token stream → Turn → episode seal → sealed wave") — kills the
      rlstack/client.py vs runner/client.py name collision and the waves.py
      grab-bag; tests/test_waves.py renamed test_sampling.py.
    - runner/__init__.py docstring is now the package MAP (one line per
      module, by responsibility, in reading order); runner/post.py states
      it is EXECUTION only (declarations in training/post/).
    - _to_delete/ (the #24 leftovers) finally deleted; git history keeps it.
    Q&A settled alongside (no code change): (a) fakes are deliberately two
    homes — runner/fakes.py (metal stand-ins, rule-8-sanctioned) and
    fake_qwen_schema BESIDE hf_schema in siteschema.py (a compiler next to
    its real sibling) — nothing else exists; (b) native-vs-plugin mechanism
    ownership stands as designed (#16/#25): punica/prompt_embeds/logits are
    the ENGINE's own levers, so their add_bundle consumers live in the
    Engine adapter (runner/engines/); rlstack_engine owns only code that
    must SHIP IN THE IMAGE (plugin mechanisms — side_attention);
    (c) lora.py/lora_torch.py is the rule-7 declaration/compute split (the
    same pattern as engines/learners: heavy deps at module scope, imported
    lazily); PEFT-format knowledge (emit/merge_fragments/peft_config) stays
    in the adapter's compute half so ONE file owns the wire format both
    sides read; (d) PARITY STATUS made explicit: Adapter.parity and
    rlstack_engine/certificates.py are designed (#25) but UNWIRED — nothing
    calls them; the only parity mechanism actually running is the per-update
    logprob_gap rail. Wiring boot-time certificates into Phase 1 is open
    Phase-C work.

32. POST POOL TREATY COMPLETED (settled, Samarth-directed: "post can access
    any gpu inference group" — the ACCESS existed since #24 (llm.pool(name)
    on every processor's SampleClient); what was missing was the DECLARATION
    layer, now landed):
    - PostProcessor gains two class-attr declarations beside produces/
      consumes: `pools` (the engine pools this processor SAMPLES from —
      "main" needs no declaring, the runner requires it unconditionally) and
      `sampling` (SamplingSpec | None — a judge's OWN budget/temperature,
      None inherits the run's gen sampling). Both live in class source, so
      they hash into run identity through code_hashes with zero new spec
      surface. PostDef carries pools; run_pipeline builds each processor's
      client with its declared sampling.
    - NEW CHECK post-pool-missing (CHECKS, after traffic_routes): every pool
      named by algo.post ∪ eval.post processors must be a declared
      EnginesMember. NEW validate.traffic_pools(spec) = {"main"} ∪ eval.pool
      ∪ pipeline pools; the loop refuses at submit when the handed engine
      map does not cover it (no more mid-update KeyError).
    - BUILTIN training/post/llm_judge.py: reference-free llm-as-a-judge —
      the judge pool solves each trajectory's task greedily (its own
      SamplingSpec, temperature 0) and reward = last-number agreement with
      the policy. The SPEC.md §3 illustrative llm_judge stub in
      test_examples is superseded by it (produces "reward", not "judge").
    - Multi-tenancy makes a judge pool FREE on one GPU: engines map
      {"main": engine, "judge": engine} — same resident vLLM, two pool
      names; the judge pool serves its base bundle. Stress stage 3 gained a
      seventh tenant (grpo trained on llm_judge rewards) proving the path
      on real metal. 339 fakes tests green.
    Open symmetry, deliberately NOT done: Environments can also call
    llm.pool(name) and have no `pools` declaration — same gap, same fix
    shape (EnvDef.pools), when a multi-pool env first exists.

33. VOCABULARY DE-OVERLOAD (settled, Samarth-directed). Three words, one
    meaning each, fixed at the source:
    - ENGINE = metal only: an object satisfying the Engine protocol (one
      resident vLLM process / FakeEngine). The spec helper that DECLARED a
      pool was misleadingly named engines(...) — RENAMED pool(...), record
      EnginesMember → PoolMember. Identity impact: run_ids change (__type__
      in canonical_json); bundle_ids do NOT (content-addressed from
      payloads+versions — fake example ids unchanged).
    - POOL = a NAME traffic routes to, with two lives: declared capacity
      (PoolMember in GpuConfig) and runtime routing entry. The name↔metal
      relation is many-to-many (one engine may back many pools — the free
      judge; one pool may fan over n engines — future replicas).
    - BUNDLE = a policy version as servable content. The payload-less
      pinning stub is now explicit: Bundle.pin(id, versions) — an ADDRESS,
      used by Generator.newest_bundle; full bundles carry payloads+kinds.
    - Runtime map Mapping[pool_name, (Engine, Bundle)] RENAMED Pools →
      Routes (runner/traffic.py); routes_at closure, `routes=` params.
      The canonical sentence: a request is TRAFFIC, addressed to a POOL,
      served by whichever ENGINE backs that name, under a pinned BUNDLE.

34. THE METAL OWNS ADMISSION (settled, Samarth-directed; supersedes the
    lease half of #27). Three moves:
    (a) GpuArbiter (runner/arbiter.py; lease.py DELETED): one arbiter per
    GpuSet, constructed by the metal's owner (deploy today, the resident
    daemon in Phase C) and SHARED by every attached experiment — the
    per-experiment leases died because two tenants' private mutexes
    coordinate nothing. Residents are OBJECT-keyed (ten pools on one engine
    = ONE resident; a learner is one resident): attach(obj, label, group,
    fraction, wake/evict) idempotent; admit(obj) the one verb; admit_all
    for pipelines (sorted acquisition; refuses two members of one group).
    Alternation exists ONLY inside an exclusive group (sharing="sleep");
    same-resident work OVERLAPS (the old ExclusiveLease serialized it —
    alternation is about memory, never a mutex on work). Policy: sticky
    drain-until-blocked + quantum (hysteresis) + max_wait (aging handoff),
    injectable clock (deterministic tests); scheduling stays OUTSIDE run
    identity (I5). Fractions declared and reported (declared_load), not
    enforced — until learner tenancy they are overlapping views. A regime
    change is refused loudly: attaching an already-attached object under a
    different group raises (stress stage 4 keeps a private arbiter for its
    sleep spec for exactly this reason).
    (b) THE LEARNER IS MULTI-TENANT (the Engine/Learner asymmetry closed):
    every Learner verb pins tenant=run_id; interfaces.py states the
    invariant as the Engine's mirror (installation additive, verbs pin,
    tenants never disturb each other). TorchLearner: ONE frozen base shared
    by all tenants ("one learner, one base"), per-tenant params/optimizers,
    activation by SWAP-INSTALL — new Adapter.uninstall_replay is
    install_replay's exact inverse (lora unwraps LoraLinear; module rebinds
    only, params objects survive → numerics exact, microseconds per swap).
    Batched multi-tenant forwards (grouped-GEMM, prime-rl MultiLoRALinear)
    are a later kernel upgrade behind the same surface. FakeLearner keeps
    per-tenant digests with the tenant key OUTSIDE the hashes, so
    single-tenant bytes are unchanged — resume-equivalence suite untouched;
    shared-vs-private-learner byte-equality pinned by
    test_two_experiments_share_one_learner.
    (c) POST ADMITS WHAT IT DECLARED: the trainer's post phase admits
    exactly the engines of the pipeline's declared pools — a pipeline with
    no sampling processors touches NO engine (post is CPU work unless
    judges sample); the evaluator admits eval.pool plus its pipeline's
    pools under one admission. Treaty tightened (amends #32):
    PostProcessor.pools declares EVERY pool sampled, "main" included
    (existence checking still exempts main). New check post-pools-conflict
    refuses a pipeline needing two pools of one sleep group co-resident —
    at submit, not as a runtime deadlock. Stress stage 3 now runs seven
    tenants on ONE engine + ONE learner + one arbiter (was seven base
    copies); 349 tests green on fakes.
    Known consequences, logged not fixed: vLLM sleep/wake and learner
    offload hooks are still no-ops (wire into attach when sleep-sharing
    goes real); async post daemon ("scorer") remains deferred until a slow
    judge or second GpuSet exists; self-judge bundle-version recording gap
    unchanged from #32.

35. THE HOST (settled, Samarth-directed — his five-bullet model: "a GpuSet
    can always be running; an arbiter informs its state; experiments are
    submitted; residency is logged; when the state is right the work runs").
    The experiment ↔ metal relationship has ONE owner now:
    - runner/host.py: Host(name, engines, learner, store, arbiter) owns the
      quartet. submit(spec, schema): BIND declared pools onto owned engines
      by base (PoolMember.base or policy base; exact first, base=None is
      fake metal's wildcard) → FIT (refuse past capacity — honest at last:
      the host sees every tenant, and object-keyed residents mean shared
      metal adds zero load) → roster in memory + JOURNAL to store → run
      under the host's shared arbiter. Host adds custody, never semantics:
      byte-identity with raw run_experiment is pinned by test.
    - ROSTER ownership settled per discussion: in the HOST, not the arbiter
      (the admission machine stays anonymous/stdlib-pure — its counters
      must not know experiment identity for same-resident overlap to stay
      trivially correct). "GpuSet state" = arbiter.residency() (per
      exclusive group, the resident's label) — Samarth's state vocabulary
      and the resident vocabulary are duals.
    - JOURNAL ownership settled: hosts/<name>/log.jsonl in the STORE
      (host-up/attach/detach events, wall-clock ts), deliberately NOT in
      run manifests (placement in the identity-checked manifest would make
      resume-on-different-metal a ManifestMismatch — placement must stay
      out of identity). Observability only; correctness never reads it;
      torn tails tolerated. Store verbs: append_host_event/read_host_log/
      list_hosts; ModalVolumeStore persists journal lines.
    - CLI: `python -m rlstack hosts <store-root>` (rlstack/__main__.py) —
      per host: up count, engines' bases, attached runs with pools, status,
      ledger progress (joined from run manifests/ledgers). On Modal:
      `modal run deploy/modal_app.py::hosts`.
    - Engine protocol gained `base` (None = fake wildcard) and the loop
      refuses pool-base-mismatch at submit — the deploy can no longer hand
      the wrong metal silently. loop.experiment_identity is Phase 0's
      identity computation made importable (the host journals under it).
    - Deploy shrank to "build one Host, submit N specs": run_arith and
      stress stages 1/3/5a go through one Host (the hand-wired judge
      engine-map died — bind_pools routes a second pool name onto the same
      engine by base); stage 4 keeps a raw private-arbiter run (sleep
      regime on metal the concurrent host attached as free — the regime
      guard refuses mixing, correctly); the cross-container resume is its
      own Host, so the CLI shows one run's life across two hosts.
    Phase C from here = a Host kept alive behind a submission queue.
    357 tests green.

36. OPERATIONAL CLI (settled, Samarth-directed: "small number of commands,
    useful views, NO experiment content — a separate UI reads the same
    stores for that"). Three commands over one or more store roots
    (rlstack/__main__.py; roots positional or $RLSTACK_STORES):
        python -m rlstack hosts <root>...   per-host: engines' bases, bound
                                            store, first/last seen, boots,
                                            tenant counts by status
        python -m rlstack runs  <root>...   per-experiment: host(s) it
                                            attached to, status, committed/
                                            target, WHERE ITS DATA LIVES
        python -m rlstack gpu   <root>...   per-host metal: utilization,
                                            memory, sample cadence, downtime
                                            gaps — from journaled samples
    Supporting treaty pieces:
    - THE HOST IS THE UNIT THAT BINDS EXPERIMENT → STORE (Host(store=...)
      is where a submission's runs and the journal land); attach and
      host-up events now record store.describe() so future store-
      multiplexing hosts stay resolvable. Store.describe(): LocalStore →
      its root path.
    - GPU stats are JOURNALED BY THE HOST (the only thing near the metal):
      Host.run_stats(every) samples nvidia-smi (sampler injectable; None
      off-metal) into "stats" events; a gap in samples IS the downtime.
      Deploy entrypoints run it beside submissions.
    - Store gained READ-ONLY PEEKS (peek_manifest/peek_ledger): observers
      must never open_run — attach SWEEPS UNSEALED WORK, which would
      corrupt a live run's staged wave. Pinned by test
      (test_peeks_never_mutate_a_live_run); the CLI renders exclusively
      from peeks + journals. Render functions (render_hosts/runs/gpu) are
      importable — the coming UI's data layer starts there.
    - Views are OPERATIONAL ONLY by design: identity, placement, status,
      progress. No rewards/losses/curves in the CLI, ever.
    358 tests green. (Also this arc: the stress harness's 20-minute
    measurement stall was diagnosed — report_run/measure_lag re-read every
    wave over the volume mount; fix deferred with the harness.)

37. FLOW GRAPH + OBSERVER REGION + STORE OWNERSHIP (settled, Samarth-
    directed, the UI-design arc). Three principles, three mechanisms:
    - THE FLOW GRAPH (spec/flow.py): flow_graph(spec) is THE canonical walk
      over the data declarations — nodes are every artifact a run contains
      (pipeline columns, records, provided tensors, rails; phase-tagged:
      post/eval/wave/forward/train; stored flag), edges are produces/
      consumes/requires verbatim; feeds_loss is TRANSITIVE reachability
      into the loss's requires ("advantage consumes reward ⇒ plot reward",
      as a graph fact). FIRST-PARTY by construction: validate's pipeline
      rules (post-unwired, post-collision, unsatisfied-requires) are now
      QUERIES on this graph — the metric derivation cannot drift from the
      semantics without the submit gate failing with it. BASE_PROVIDES/
      BASE_RECORDS moved here (validate re-exports).
    - RUNS SELF-DESCRIBE: loop writes dictionary.json (= the graph's
      to_json) beside the manifest at run creation — derived, NOT identity,
      deterministic (attach rewrites identical bytes; resume-equivalence
      unaffected). A UI reads the run's own dictionary — no rlstack import,
      no registry, no version skew. Store verbs: write_dictionary /
      peek_dictionary.
    - STORE OWNERSHIP INVARIANT: the run store is a PER-EXPERIMENT binding
      (one experiment, one store, for life) — run_id is global (I3) but
      existence is store-scoped, so the same spec against two stores forks
      history silently. Host now separates its JOURNAL store from the run
      store: submit(spec, schema, store=None) takes the experiment's store
      explicitly (defaults to the host's); attach events journal its
      locator. Full global enforcement is impossible (no registry of all
      stores); the observer's runs view detects violations instead — the
      same run_id in two given stores renders "⚠FORK".
    - OBSERVE/ is a new architecture region (STYLE tree + arch test): read-
      only derivations over stores and journals — never attaches, never
      writes, imports the data layer ONLY. locate.py store_for(locator)
      (path/file:// resolve where mounted; s3:// reserved; modal:// refuses
      locally with guidance — THE PRINCIPLE: a store is named by a locator
      and the reader runs where it resolves). views.py: hosts/runs/gpu each
      as *_data (structured, the UI's JSON layer) + render_* (CLI text).
      __main__.py is thin argparse over observe. ModalVolumeStore gained
      `locator` so describe() journals modal://rlstack-store, not /store.
    368 tests green. UI stack from here: an ASGI wrapper over the *_data
    functions + per-run dictionary/postdata/ledger series endpoints,
    deployed beside whichever store backend.

38. THE LOSS IS PURE MATH; POST OWNS ALL PRODUCTION (settled, Samarth's
    ruling, supersedes decision #5's planned passes). "The post processor
    contract is: produce all scalars necessary for the loss to operate on.
    LossDef.requires rerouting back to a GpuSet is wrong." Enforced:
    - PLANNED PASSES RETIRED: Ref/Teacher/Probe deleted from the registry;
      BASE_PROVIDES (ref_logprobs/entropies/hidden_states — the planned-
      pass vocabulary, never implemented) deleted from the flow graph.
      @loss now REFUSES non-string requires at registration (TypeError
      naming the rule). requires resolves against: post columns ∪ records
      (base + bank) ∪ bank provides (replay-lowering forward tensors are
      the forward itself, not metal routing — value_head "values" stays).
    - TOKEN_LEVEL COLUMNS: a PostProcessor may declare produced columns in
      `token_level` — one float per GENERATED token per trajectory (sealed
      order), validated by run_pipeline, stored in postdata, aligned by
      broadcast into TokenBatch.post at loss_mask==1 positions (injected
      0.0), ledger-meaned over tokens. THE channel for per-token teacher
      signals: hinted/teacher logprobs are produced by a post processor
      through a pool, land in postdata, and the loss consumes a column —
      never plans a pass. Flow nodes + dictionary.json carry `granularity`
      (trajectory | token | update) so a UI knows series from facet.
    - OPSD CORRECTED: what #30 called "opsd" was NOT on-policy self-
      distillation — it matched the lagged sampler's RECORDED logprobs.
      RENAMED self_anchor (honest docstring). True OPSD = distilling from
      HINTED logprobs (same weights re-scored under privileged
      conditioning) = a token_level post column, awaiting a SCORING verb on
      the engine/client (score given tokens under a pool's serving stack) —
      the named follow-up, not built here. opd + self_anchor now declare
      requires=("behavior_logprobs",): the teacher signal visibly FEEDS
      those losses in the graph (records-in-requires path exercised).
    370 tests green; stress tenant renamed self_anchor (identity shift for
    opd/self_anchor specs only).

39. DOC DEBT PAID (Samarth's process check: "as we're introducing new
    invariants... you should be keeping track in the relevant docs").
    Audit: CONTEXT.md / STYLE.md / code docstrings had tracked every change;
    rl-stack-spec.md was frozen at v2 (the #21 "deltas pending fold-in"
    debt, 18 entries deep — several sections outright wrong) and CLAUDE.md
    described the pre-metal world. Remediated:
    - rl-stack-spec.md REWRITTEN AS v3: folded #21–#38. The contract grew
      four earned invariants — I8 multi-tenancy on both sides of the
      bridge, I9 the loss is pure math / post owns production, I10 one
      experiment one store (+ observers never attach), I11 runs
      self-describe. Sections rewritten to current surfaces
      (TrajectorySource, pool()/GpuGroup, post pipelines, Mechanism
      reachability, Host/arbiter/blackboard runtime, waves/ + hosts/ +
      dictionary store layout); examples modernized (judge + token_level as
      the extension story; the seven-tenant stress as Example 6); honest
      status notes kept inline (parity unwired, FSDP unexercised).
    - CLAUDE.md rewritten to the current handover (370 tests, metal proven,
      pins, Host/arbiter/observer state, quick commands, known-open list).
    - PROCESS RULE going forward (also now in CLAUDE.md): spec-shape
      changes get a CONTEXT entry ALWAYS, and fold into the spec in the
      same arc when they touch an invariant; CONTEXT stays the
      chronological authority between fold-ins.

40. THE SCORING VERB + REAL OPSD + PoolClient (settled, Samarth-directed:
    "hinted logprobs can easily be generated in one pass via a prefill
    engine... let's implement"). The confusion it resolved: pool-calling
    primitives always existed for GENERATION (envs, judges — channel 1);
    the never-built loss-side planned passes (channel 2, retired #38) were
    for SCORING. The missing primitive was one verb:
    - Engine.score_tokens(messages, token_ids, bundle_id) -> logprobs of
      GIVEN tokens: ONE prefill pass (vLLM prompt_logprobs=0 over context +
      tokens, max_tokens=1 discarded; FakeEngine: pure hash of (bundle,
      context, token, position)). Deterministic, seedless. Prefill-shaped
      traffic — the natural tenant of Samarth's planned prefill/decode
      disaggregation.
    - SampleClient RENAMED PoolClient (rlstack/client.py; concrete
      EnginePoolClient): it samples AND scores against a named pool.
      score() consumes NO episode seed — adding a scoring processor never
      shifts sampling seeds (tested).
    - hinted_logprobs builtin (training/post/): walks the sealed message
      stream in flatten order, scores each turn's own tokens under hint +
      preceding messages through the POLICY pool, emits a token_level
      column. Hint = task.meta["hint"] or "The answer is {answer}. ".
    - opsd RE-REGISTERED AS THE REAL THING: requires=("hinted_logprobs",),
      squared match of trainer logprobs to the hinted column — pure math
      over postdata (I9 end to end). self_anchor keeps the lagged-record
      objective. Stress stage 3 gains opsd as an 8th tenant — the vLLM
      score_tokens path's first-contact vehicle (written, not yet executed
      on metal). Spec v3 + CLAUDE.md updated in the same arc (#39 rule).
    372 tests green.

41. THE OBSERVER UI (settled, Samarth-directed: "a local wandb, as little
    interference with my main code as possible; just the graphs for now,
    walkback-from-loss priority"). Landed with ONE core touch
    (Store.peek_eval_summaries beside the existing peeks); everything else
    in observe/:
    - series.py run_series(store, run_id): the run's dictionary + per-
      update ledger post-means and train rails + eval-summary means, from
      peeks alone.
    - ui.py: dependency-free stdlib WSGI app + one self-contained page
      (hand-rolled SVG charts, no CDN, 3s polling). PANEL PRIORITY IS THE
      DICTIONARY (I11): "feeds the loss (walkback from <loss>)" section
      first, rails second, measurement + dashed held-out eval overlays
      last. Index = runs_data (status, fork flags, hosts).
    - Served two ways per the locator principle: `python -m rlstack ui
      <store>` locally; `modal deploy` wraps the same ui_app beside the
      volume (throttled volume.reload per API read) at
      samarthmbhargav--rlstack-ui.modal.run.
    - E2E ON METAL: 30-update GRPO, Qwen3-0.6B, one L4, submitted through
      the Host, WATCHED LIVE in the deployed UI (run 3be52610bb51):
      reward 0.5 → 1.0 with the dashed eval overlay tracking to 1.0,
      advantage collapsing to 0 as groups saturate (z-score of equals —
      the estimator visible), gap at the 0.019 floor, 30/30 + 15/15 evals.
      The cross-container-resume run renders as one run on two hosts.
    Next UI features (named, not built): postdata distributions, token
    drill-down (waves peeks), gpu/host pages, run compare.

42. CUSTOM PANELS (settled, Samarth-directed: "graphing of custom things,
    functions of what dictionary.json already logs; validation = every
    argument is in the pipeline"). Landed as EXPRESSIONS-AS-DATA:
    - observe/panels.py: ast-whitelisted arithmetic (+ - * / ** %, unary,
      log/exp/sqrt/abs) over column names; nothing else parses — panels
      can never smuggle code into the observer.
    - VALIDATION AT THE RIGHT DOOR: the rule is enforced against each
      run's own dictionary.json (the same flow-graph oracle the submit
      gate queries), NOT at submit — panels in the hashed spec would make
      adding a graph fork the run_id. A panel whose argument a run lacks
      renders "missing from this run's pipeline: X" (and older runs
      honestly reject args their dictionaries predate — seen live with
      `tokens` against a pre-#42 run).
    - One expression, both series: the held-out overlay computes from the
      same expr when its args exist in the eval means.
    - Declarations live in panels.json at the STORE ROOT — written by the
      user (cp locally / `modal volume put rlstack-store panels.json
      panels.json`), only ever READ by the observer (Store.read_panels;
      the never-writes rule holds); `rlstack ui --panels` overrides
      locally, re-read per refresh.
    - Flow graph gained TRAIN_STATS (tokens, microbatches) as declared
      train-phase stat nodes so the dictionary fully describes the ledger
      (they were logged but undeclared — the panel validation caught it).
    Verified on the deployed UI against the live 30-step GRPO run:
    excess_reward 30 pts + 15 eval-overlay pts; log_grad showing the
    grad-collapse at saturation; gap_per_ktok correctly refused by the
    older run's dictionary. 382 tests green.

43. THE FLEET: HOSTS AS ATOMIC PARTITIONS + JOIN/CARVE/ACQUIRE (settled,
    Samarth-directed, the multi-GPU milestone's foundation). His rulings,
    verbatim intent: per-capability hosts ("if a new experiment comes in
    which just wants the tp-4 inference worker, it should be able to just
    contact that host"); the unit ("a host should just represent some
    partition of some gpu (or gpus) that can be used for some purpose and
    cannot be reduced. thats the unit — not a gpu, not a node, not a
    container"); sub-GPU hosts (0.5 inference + 0.5 training on one GPU);
    alternation as ONE host with multiple regimes, never two hosts
    coordinating; carve automatic, acquire human; the learner never remote.
    Landed as (all fakes-proven; real TP/FSDP metal is the next layer):
    - SHARDING IS A BUILD FACT: Engine.tp and Learner.fsdp are attributes
      of the BUILD (FakeEngine(tp=), VllmEngine(tp=) → tensor_parallel_size,
      TorchLearner.fsdp=1, FakeLearner(fsdp=)); the submit gate attests
      them (validate.check_members_match_their_shape → pool-shape-mismatch
      / learner-shape-mismatch, run with the other live-metal checks);
      Host.bind_pools is shape-matched (tp has no wildcard). Switching
      shards = handing different metal, never editing a spec.
    - HOST = ATOMIC PURPOSED PARTITION (runner/host.py): born with a
      Partition (gpuset, device indices, memory fraction — vLLM's
      gpu_memory_utilization is a reservation, torch's
      set_per_process_memory_fraction a cap; SM contention across
      partitions is a stated cost) and Regimes (inference|training × base
      × shape). attest_regimes dies at construction on wrong metal;
      capability is a BIRTH FACT, never mutated after. >1 regime = the
      host ALTERNATES them on its own arbiter exclusive group
      ("host:<name>"); regime residents attach at birth with fraction 0.0,
      so every JOIN is fraction-free (is_attached short-circuits
      check_fit — fraction is a carve hint, meaningless once the weights
      live). A tenant's sleep demand on metal already in a host group
      DEFERS to the metal's truth (arbiter.attach: group None defers;
      attach_residents consults arbiter.attached_group); sleep on
      always-resident metal still raises.
    - THE WIRE (runner/remote.py): HostService executes pool verbs on the
      owning host's metal under the owning host's arbiter (admission
      stays with the partition; engines addressed by CAPABILITY (base,
      tp), never pool name). Transport carries JSON-safe dict frames —
      async call() for admitted verbs (sample/score), sync ask() for
      admission-free ones (add_bundle — additive by the tenancy
      invariant — reachability, tokenize). RemotePool implements the full
      Engine protocol over it; the runner cannot tell remote from local
      (PROVEN: a run with a remote main pool is byte-identical to the
      local run). LocalTransport json-round-trips every frame both ways,
      so the Modal-cls transport is a drop-in (NOT YET BUILT). v0:
      non-streamed sample replies; bundles ship bytes. Remote pools
      attach locally as zero-footprint residents; remote + sleep group
      refused (alternation is intra-partition).
    - THE LADDER (runner/fleet.py): demands_of(spec) reads capability
      demands (kind, base, shape; memory = the carve hint) off gpu_config.
      JOIN (a host's regimes cover the unit; automatic, the target
      host's arbiter decides, fractions ignored) → CARVE (first-fit over
      RESIDUAL — capacity no partition owns, so automatic BECAUSE
      journaled (fleet/log.jsonl, Store.append_fleet_event); residual-
      only, never reshapes an existing host) → ACQUIRE (new metal = money
      = human; place() names what to buy, submit() refuses to run it).
      Placement units: a sleep group is ONE unit → ONE multi-regime host;
      concurrent members place PER MEMBER (per-capability hosts;
      concurrent grouping was only a colocation hint and colocation is
      semantics-neutral, I5). Fleet.submit: the runner goes to the
      learner's host (the learner is NEVER remote), reaches every other
      partition through RemotePools, and journals the placement under the
      run's identity before the run opens.
    - Trainer BATCHES the forward now: TorchLearner._batched_logprobs is
      one padded forward per microbatch (left-aligned docs + causal
      attention ⇒ numerics identical to the per-doc [1,L] forwards it
      replaced; padding never enters the gather). Prerequisite for the
      8B-FSDP milestone; document-masked packing is the later upgrade.
    NOT YET BUILT (the milestone's remaining layers): real TP engines +
    FSDP learner processes; the Modal-cls transport + per-capability host
    deployment; host linger/GC back to residual; join-refusal beyond
    adapter slots (needs a measured saturation signal, not declared_load);
    the OPD 8B←32B e2e across three L4 hosts. 401 tests green (test_fleet,
    test_remote, ShapeAndRegimeTest new).

44. THE TRAINER'S PUNICA: BATCHED FORWARD, SITES EXPOSED PER ROW (settled,
    Samarth-directed: "implement batching for training... this batching is
    of the same flavor as the engine, where sites are exposed, so it
    supports multi lora and other types of adapters"). Scope was fixed as
    MECHANISM FIRST: the trainer-side analog of the punica kernel, NOT the
    cross-tenant request queue (see "next"). Supersedes the swap-install
    half of #34(b).
    (a) ONE PADDED FORWARD PER MICROBATCH. TorchLearner._doc_logprobs (one
    document per model call) became _batched_logprobs: documents are rows,
    left-aligned and right-padded, one causal forward with the padding
    mask. The rows are the unit everything below routes on. Cost logged,
    not fixed: logits are rows × LONGEST document while pack() bounds the
    token SUM, so a very ragged wave pays that ratio in logit memory
    (length-bucketed sub-forwards are the fix when a wave needs one).
    (b) THE ROW PLAN (policy/adapters/replay.py, new): ReplayRows = slot
    order + a [rows] index; a SLOT is one tenant's installed deltas keyed
    {site path -> the params holding it}. RowPlan is routed for exactly one
    forward and RAISES when a lowering runs unrouted — an unrouted replay
    forward is a wiring bug, never a fallback to whoever went last. The
    plan rides ON THE MODEL (row_plan(model), one named accessor) because
    the model is the one handle Adapter.install_replay receives: the
    five-member protocol in adapters/base.py is UNTOUCHED, and the same
    hook is what soft-prompt replay will read for its per-row rows.
    (c) LORA APPLIES PER ROW (lora_torch.LoraLinear -> LoraSite). One slot
    takes the pre-#44 expression verbatim ((x A^T) B^T over the whole
    batch); many slots gather each row's (A, B) out of a stack and do two
    bmms — punica's shape in stock torch. Slots of one forward must agree
    on rank (punica zero-pads to max_rank; the coalescer will).
    (d) INSTALL IS ADDITIVE, THE MIRROR OF add_bundle (I8). Swap-install is
    gone: every tenant is wired at install() and stays wired, a second
    tenant at a site JOINS the LoraSite it finds, and uninstall unwraps only
    when the last state leaves. A verb pins one tenant, so today every row
    of a forward carries that tenant's slot — the degenerate one-slot case.
    Consequences: placement moved to install (params reach the device before
    optimizers exist and before any load), so _colocate_optim_state from
    185c07c is DELETED as dead by construction; a trainer-only kind with no
    install_replay now fails at Phase 1 instead of the first backward;
    per-tenant deltas deliberately do NOT register in model.parameters()
    (the base is shared and frozen, a delta is one tenant's state).
    (e) EVIDENCE. tests/test_batched_replay.py: 17 tests that skip without
    torch and RUN ON CPU in the deploy image (modal_app::run_tests) — the
    unrouted-forward refusal, one-slot-equals-swap-install bit-identity,
    rows-carry-their-own-delta, additive install/uninstall balance, "another
    tenant's install does not move the numbers", gradients reaching only
    routed slots, and the padded forward against the verbatim pre-#44
    per-doc forward on a toy causal LM. deploy/batch_parity.py (new, ~4 min
    on one L4, Qwen3-0.6B, 112 self_attn sites, r=8): 16/16, reference =
    the pre-#44 LoraLinear body wired the pre-#44 way. ONE ROW PER FORWARD
    is BIT-IDENTICAL in float32 AND bfloat16 (max|d| = 0.00e+00) — every
    thing #44 added is exact; the full padded batch differs by 2.07e-05
    (base) / 3.43e-05 (uniform) / 1.72e-05 (mixed) in fp32 and 1.15e-01 /
    1.70e-01 / 3.12e-01 in bf16, all of it the batch dimension's reduction
    order (the base row carries no adapter at all), against deltas that move
    logprobs by 9 nats. The mixed case is two tenants' deltas in ONE
    forward, which swap-install cannot express. End to end: run_arith
    (run bc9c177d23bb, seed 17) rewards .500/.812/.875/.812 with ledger gap
    0.0241–0.0280 — inside the 0.022–0.033 kernel floor #28 calibrated for
    this exact stack, i.e. batching did not widen the alarm.
    (f) DOC DEBT, deliberate: runner/interfaces.py's Learner docstring still
    says "the v0 realization is swap-install" and STYLE rule 8's policy/
    line still reads "adapters/ (one file each)" — replay.py is the kinds'
    shared compute-side seam, not a kind. Both are one-line edits owed;
    they were not made because a parallel track owns those files this cycle.
    PROPOSED SPEC DELTA (I8, rl-stack-spec.md): replace "(v0: swap-install —
    module rebinds, never weight copies)" with "installation is additive on
    the trainer side too: every installed tenant's deltas stay wired, and
    each row of a microbatch carries the slot whose delta applies to it —
    the mirror of a request pinning its bundle."
    NEXT (designed, no code): the CROSS-TENANT COALESCER, the analog of
    AsyncLLMEngine's request queue over this kernel. The learner grows an
    internal queue of (tenant, microbatch) work items behind the SAME five
    sync verbs; forward_backward enqueues and blocks on its item's result
    instead of running it. A step loop drains the queue into one padded
    forward whose rows come from several tenants: slot order = the distinct
    tenants in the draw, index = each row's tenant, which is exactly the
    ReplayRows this entry already builds. Three things it must own, none of
    them kernel work: ADMISSION (a draw may only mix tenants whose ranks
    agree and whose bank paths cover every routed row — today's loud raises
    become the queue's filter), LOSS SEPARATION (I9 keeps the loss pure per
    tenant, so the coalesced forward's logprobs must be split back along
    doc spans and each tenant's loss run on its own slice, with backward
    accumulating into that tenant's params only — already true, since a
    slot no row carries takes no gradient), and FAIRNESS (the GpuArbiter
    owns admission to the metal, #34, so the coalescer must not become a
    second scheduler: draw policy is FIFO with a per-tenant cap, and a
    tenant's bytes must stay identical whether or not it was coalesced,
    which the tenancy invariant already demands and bf16 batching already
    threatens — the honest form is "identical up to the batching noise
    deploy/batch_parity.py measures"). 399 tests green (382 + 17, the new
    ones skipped locally and green in the image).

45. REAL MULTI-GPU METAL: TP INFERENCE, THE MODAL-CLS TRANSPORT, THE FSDP
    LEARNER (settled by execution — #43's three named-unbuilt layers, now
    run on L4:2 metal). #43 designed the fleet against fakes and listed what
    had never touched a second GPU. All three exist now; this entry records
    what the metal said, not what the design hoped.
    - TP INFERENCE (deploy/tp_l4.py, new; nothing in rlstack/ changed):
      VllmEngine(tp=2) on one L4:2 host, Qwen3-0.6B AND Qwen3-8B. Sampling
      streams. Two compiled LoRA bundles COEXIST on the sharded build and
      three concurrent requests (bare base / bundle A / bundle B) come back
      under three different adapters — punica's per-token adapter indices
      survive tensor parallelism, so the Engine multi-tenancy invariant
      holds unchanged at tp>1. 8B at tp=2 is the case that matters: 16 GiB
      of bf16 weights do not fit one L4 beside a KV cache, so tp is not an
      optimization there, it is the only way that base serves.
    - SCORE_TOKENS' FIRST CONTACT (its first anywhere, at any tp) came back
      CLEAN: the prompt_logprobs suffix is indexed correctly and
      vllm_engine.py needed NO fix. The proof is a shift test, not a
      tolerance: the aligned sampled-vs-scored gap against the gap a
      one-position shift gives — 0.0066 vs 0.452 on 0.6B, 0.0091 vs 0.354 on
      8B (position 0 matches to five decimals). An off-by-one cannot survive
      that at any adapter magnitude. The residual gap is punica numerics —
      prefill applies a delta with different kernels than decode — and it
      tracks adapter MAGNITUDE, not position: 0.6B 0.021 (base) / 0.045
      (|B|~0.005) / 0.277 (|B|~0.05), while 8B stays at the floor at every
      magnitude. Real deltas start at B=0 and grow slowly, so the verb runs
      in the regime where the two agree to ~0.02 — but the gap is real, and
      a scoring-based loss should treat it as a floor, not as zero.
    - THE MODAL-CLS TRANSPORT (deploy/modal_host.py, new; remote.py
      unchanged). Samarth's venue model, executed: POOL TRAFFIC rides a real
      transport, the STORE PLANE rides the volume as it always did.
      ServedHost is a modal.cls that builds the metal, wears it as a Host
      born with its Partition and Regime, journals host-up to the volume
      (the observer's `hosts` view sees a remote partition like any other),
      and exposes HostService's two verbs as Modal methods. ModalTransport
      is four lines of body — call → .remote.aio, ask → .remote — and that
      is the whole point: because HostService already speaks JSON-safe dict
      frames and LocalTransport already round-trips them, a real transport
      has nothing to serialize. PROVEN: a complete arith GRPO run, learner
      local to the driver container, "main" pool served by another
      container: 3/3 updates, reward 0.375→0.875, logprob_gap 0.023–0.033
      (the kernel floor — the remote engine served exactly the adapters the
      local trainer recomputed), evals present, the remote pool journaled.
      Cost observed: 217s wall for 3 updates of 8 trajectories. The tax is
      per-verb round trips, and the loudest is TOKENIZE — flatten calls it
      per message per trajectory, so a wave costs dozens of RPCs for an
      answer the pool's base fixes at build time. A local tokenizer beside
      each RemotePool is the obvious fix and was NOT taken here (it changes
      remote.py's contract; logged, not smuggled).
    - THE FSDP LEARNER (rlstack/runner/learners/ranks.py + fsdp_torch.py,
      new; torch_learner.py NOT edited — FsdpTorchLearner subclasses it and
      overrides exactly one method, _ensure_base). The blackboard stays ONE
      async process: rank 0 runs the runner and the five verbs, ranks
      1..N-1 run a command loop and exist only to stand in the collectives.
      Rank 0 ANNOUNCES each collective verb (install / forward_backward /
      optim_step / load) before running it; emit is NOT announced, because
      it touches no collective — and a broadcast that buys nothing is a
      deadlock waiting for its first caller. Rank 0's copy is the truth; the
      other ranks' answers are discarded. torch 2.13 facts, source-checked
      in the pinned image: FSDP1 (FullyShardedDataParallel) is deprecated,
      fully_shard (FSDP2) is the top-level export, and it shards to DTensors
      one group per decoder block plus the root.
      THE SHAPE OF IT: the base is sharded, the DELTAS ARE NOT. A tenant
      installs after the wrap, so no FSDP group owns its LoRA params; they
      live whole on every rank and each rank steps its own copy. Nothing
      trainable is inside a group, so no gradient is ever reduce-scattered.
      This build therefore buys MEMORY, not throughput — the ranks recompute
      the same microbatch rather than splitting one. Data-parallel width is
      a later upgrade and would change exactly one thing: the deltas would
      need a reduction before optim_step.
      WIDTH-INDEPENDENCE (the invariant this whole design serves): emit/load
      bytes must not depend on fsdp. It holds by construction (the only
      sharded thing is the frozen base, which is never emitted) and
      attest_emit_is_width_free proves it at every emit instead of trusting
      it. PROVEN end to end: a fsdp=2 run's sealed adapter bytes (18,379,976)
      reload into an UNSHARDED TorchLearner unchanged and compile to the same
      bundle id.
      Also proven on metal (Qwen3-0.6B, L4:2): the regime attests both ways
      (a fsdp=2 learner accepted, an unsharded one refused at construction);
      310/310 base parameters sharded with rank 0 holding exactly 50.0% of
      them (298.0M of 596.0M) while 4.59M delta params stay whole; a 10-update
      GRPO run killed mid-flight at update 2 and finished behind a FRESH
      chorus; logprob_gap max 0.0343 — the same floor an unsharded run sits
      at, which is the real correctness signal for a sharded forward.
      base_parameters() is the correction the metal asked for: after an
      install the module tree carries BOTH shapes, so every statement about
      "the base" must cut the deltas out by identity first (the first run
      reported "310/534 sharded" and looked like a bug; it was the design).
      OUT OF SCOPE, stated: sleep-sharing x FSDP (an alternation would have
      to swing every rank in step) — an FSDP host is dedicated or concurrent.
      8B AT fsdp=2, HONESTLY: the SHARDING is proven (399/399 base params
      sharded, rank 0 holding exactly 50.0% of 8,190.7M, 7.63 GiB resident,
      deltas 15.34M whole), but the RUN does not fit on 2xL4 — and the
      reason is the SAMPLER, not the learner. An 8B engine at tp=1 holds the
      whole model (~15.3 GiB of weights; gpu_memory_utilization bounds the
      budget, not the weights) and wants it on cuda:0 where the learner's
      7.63 GiB shard already lives, so vLLM's engine core OOMs part way
      through loading. The answer is not a smaller fraction; it is the
      fleet's own: the sampler goes on its OWN partition, reached over the
      wire — the per-capability host shape #43 designed, which this entry's
      other two deliverables now make possible. The three pieces compose
      exactly where the milestone needs them to.
    - DEPLOYMENT FACTS (I5, learned the hard way, all in deploy/): vLLM's TP
      workers segfault in libgomp (gomp_team_start) on their FIRST
      OpenMP-parallel CPU op in this image — 0.6B never reaches one (its
      buffers stay under torch's parallel grain size), 8B hits it during
      model-runner setup; spawned workers with OMP_NUM_THREADS=1 never form
      the thread team. Bigger models also want a bigger container (cpu=8,
      memory=32-64Gi). modal.parameter cannot validate stringized
      annotations, so a file with modal.cls parameters may not carry `from
      __future__ import annotations`. A sync `ask` from inside the runner's
      loop warns (it works; ModalVolumeStore.commit() has always crossed the
      same way).
    - PROPOSED SPEC DELTA (for the main session; I did not touch canon):
      (1) A new invariant candidate — SHARDING IS INVISIBLE IN THE STORE:
      what a learner emits and loads is independent of its build width, so a
      run trained at one width resumes, warm-starts and compiles at any
      other. It is the deepest thing #45 proved and nothing in I1–I12 says
      it. (2) HARD EVIDENCE FOR THE IDENTITY-RINGS THREAD (first open thread
      below): gpu_config IS inside canonical_json(spec) and therefore inside
      run_id, so a spec declaring learner(fsdp=2) and the same spec
      declaring fsdp=1 are DIFFERENT RUNS — even though the store's bytes
      are provably width-free and the resume would be sound. I5 says GPU
      topology is semantics-neutral; identity currently disagrees. Either
      gpu_config leaves the hash (making I5 literal) or I5 narrows to
      "topology changes semantics never, identity yes".
    NOT YET BUILT (what this layer leaves): an 8B training run end to end,
    which now needs the sampler on its own host rather than any new
    mechanism; a CPU-side base load before the wrap (the peak is the whole
    model, not its shard — shard_the_frozen_base returns the difference to
    the driver, but the peak itself is TorchLearner's `.to(device)`); a local
    tokenizer beside RemotePool; streamed sample replies (the wire is still
    one reply per request); host linger/GC; the OPD 8B←32B e2e across
    per-capability hosts, which is now only wiring — TP inference, the
    transport and the sharded learner all exist.
    404 tests green (VerbSplitTest new: which verb rides `call` and which
    rides `ask` is the contract every out-of-process transport implements
    against, pinned in the fakes suite rather than rediscovered on metal).

47. REAL ON-POLICY DISTILLATION: THE TEACHER IS A POOL (settled by
    execution — Samarth-directed: "`opd` is repointed to true on-policy
    distillation: the student samples live, a frozen teacher SCORES the
    student's sampled tokens, and the loss consumes the teacher's per-token
    logprobs"). #40 built the scoring verb and spent it on the policy
    scoring ITSELF under a hint; this entry spends it on a SECOND MODEL,
    which is the case the verb was really for, and #45's three multi-GPU
    layers are exactly what makes it runnable. It is also the OPD 8B←32B
    end test #43 and #45 both left named-and-unbuilt.
    - THE RENAME. What #30/#38 called opd matched a REPLAYED run's RECORDED
      behavior logprobs. Honest math, wrong name: the teacher there is a
      sealed record, and nothing about it is on-policy. It is now
      `replay_distill` — same expression, same requires=("behavior_logprobs",),
      the file moved; stress_l4's tenant renamed with it. Sealed runs are
      untouched: a loss name is inside canonical_json(spec), so old runs keep
      their identity and their manifests keep the old name, and only a NEW
      submission under the new name exists.
    - THE NEW opd IS SAMPLED-TOKEN REVERSE KL (GKD's on-policy branch),
      requires=("teacher_logprobs",), one sample deep — score_tokens returns
      the CHOSEN tokens' logprobs and no distribution beyond them.
      THE ESTIMATOR IS THE ONE PLACE THIS ENTRY ADDS TO THE DIRECTION, and
      the math forces it: the tokens are the STUDENT'S OWN DRAWS, so
      differentiating (student_lp − teacher_lp) pathwise gives ∇lp — the
      teacher cancels out of the gradient completely and the objective
      degenerates into "push every sampled token down", identically for every
      teacher. The loss therefore reports the masked-mean per-token KL as its
      VALUE and carries the SCORE-FUNCTION gradient of it (the detached KL
      weighting ∇log π), which is what on-policy distillation optimizes and
      what makes `loss` in the ledger read directly as nats of teacher-student
      distance. Both halves are pinned by tests/test_opd_math.py (torch-gated,
      skipped locally, green in the image): the value, the gradient, the
      zero-KL fixed point, and that logprob_gap still answers to the RECORD
      rather than to the teacher.
    - THE PROCESSOR (training/post/teacher_logprobs.py) IS the teacher
      channel: produces / token_level = ("teacher_logprobs",),
      pools = ("teacher",), walking each sealed trajectory in FLATTEN ORDER
      and scoring every generated turn against everything before it —
      hinted_logprobs' walk minus the hint, through another model's pool.
      I9 end to end: the loss reads a column, the pipeline owns the metal,
      and changing teachers is a pool declaration, never an edit to a loss.
      Two preconditions, stated in the file and checked by the deploy: the
      teacher SHARES THE STUDENT'S TOKENIZER (the ids crossing the wire are
      the student's own draws, so a different vocabulary would score
      different text), and the teacher is FROZEN (a non-policy pool gets a
      payload-free bundle, so no delta of this run ever reaches it, and
      scoring is seedless — adding the processor never shifts sampling).
    - THE E2E, THREE PER-CAPABILITY HOSTS ACROSS CONTAINERS (deploy/opd_l4.py,
      run c0f65f24362b, 6/6 checks): teacher Qwen3-32B tp=4 on L4:4
      (inference regime only, no learner), student Qwen3-8B tp=2 on L4:2,
      learner Qwen3-8B fsdp=2 on L4:2 with the runner beside it — both pools
      RemotePools over ModalTransport, stores and journals on the volume,
      and the observer's `hosts` view seeing all three partitions. Nothing in
      rlstack/ changed for it; the spec declares three capability demands and
      never says where, which is the #43 claim executed at full size.
      THE TEACHER PROBE FIRST (::probe, twice, on two separate cold starts,
      4/4 both times): tokenizers identical (12 ids); ' 105' scored -0.0814
      per token against ' 731' at -4.1414 — four nats, so the prefill
      demonstrably read the context; and the per-token scores came back
      BIT-IDENTICAL across both container lifetimes, which is #40's
      seedless-determinism claim tested the only way that counts.
      THE LEDGER, and it is the distillation signal itself because `loss` IS
      the per-token reverse KL: 0.3559 / 0.3041 / 0.3146 / 0.2848 nats over
      four updates — the student moved toward the teacher. Update 1's 0.356
      is the BARE 8B-vs-32B distance on these completions (the LoRA is still
      B=0 there). Over all 384 scored tokens the teacher mean is −1.3357 and
      the student's recorded mean −1.0188: the teacher is LESS confident on
      the student's draws than the student is, which sampling from the
      student guarantees. mean_ratio 0.9968–1.0001.
      THE RAILS SEPARATE CLEANLY, which is the correctness result: gap
      0.0165–0.0216 across a wire, under a sharded learner, on an 8B — the
      kernel floor, meaning the remote student pool served exactly the
      adapters the local FSDP trainer recomputed. #45's SCORING floor lands
      somewhere else entirely: it is prefill-vs-decode punica noise, it
      applies to the TEACHER column, and the teacher pool carries no adapter
      at all — so at ~0.02 it is about 6% of a 0.32-nat signal and is never
      an alignment error. A scoring-based loss should treat it as a floor on
      the column, not as an error bar on the gap rail.
      METAL FACTS (I5): 32B at tp=4 is 16.5 GiB per device with 3.03 GiB left
      for KV (49,664 tokens at max_model_len=512) — it fits an L4:4 with room
      to prefill, which was the open question; 8B at tp=2 is 8.17 GiB per
      device with 10.28 GiB of KV. Loading 32B unauthenticated from HF took
      308s cold and 52s from a cache volume after. Warming both partitions
      concurrently took 237s and the four updates 386s. Whole milestone,
      probes included, ≈2.5 L4-GPU-hours.
    - PROPOSED SPEC DELTA (I9, rl-stack-spec.md): the loss-purity section
      names hinted/teacher logprobs as the token_level channel but has only
      ever had a SELF-scoring example. Add the second, now that it has run:
      "a post processor may score through ANY declared pool, including one
      serving a different base — so a frozen teacher is an inference
      capability the fleet places like any other, and distilling from a
      different teacher is a pool declaration rather than a change to any
      loss. Two preconditions ride with it: the scoring pool must share the
      sampled tokens' tokenizer, and a non-policy pool serves its bare base."
      Also worth a line in I5's neighbourhood: a spec's gpu_config may
      declare capability demands that NO single host can satisfy (a tp=4
      teacher beside a tp=2 sampler beside an fsdp=2 learner), and placement
      answering that with three hosts is the normal case, not the exotic one.
    - NOT BUILT, and named honestly: teacher scoring rides the Trainer's post
      phase INLINE, so an update's gradient waits on 8 sequential prefills
      against a 32B — the async post daemon ("scorer") is the fix and is a
      separate planned track. Teacher scoring is also un-batched (one
      score_tokens per turn) and un-cached (identical prefixes re-prefill).
      A kill/resume of this three-host run was not attempted (the mechanism
      is #45-proven at fsdp=2; only the wire is new). The teacher's own
      certificate (#25) would be the honest way to pin "the teacher I
      distilled from is the teacher I think it is" and remains unwired.
    432 tests green locally (21 skipped: 17 pre-existing + 4 new torch-gated
    opd-math tests), the whole 432 green in the image under
    modal_app::run_tests.

## Open threads (do NOT treat as settled; flag when your answer touches them)

- TODO (Samarth, settled intent): DELETE RunSignals.notify() and run the
  blackboard on the poll leg alone — the store predicate is already the only
  truth, notify is a latency hint, and removing it makes in-process and
  cross-container daemons identical (one code path, no illusion of a message
  bus). Cost: up to poll_seconds latency per daemon edge — tune poll_seconds
  down (constructor param exists; tests should pass a small value so the
  fakes suite stays ~1s). The arbiter's own 50ms admission poll is separate
  and unaffected.

- Identity rings: should GpuConfig (and EvalSpec) leave the run_id hash and become
  submit-time config (resume-across-hardware keeps identity)? #43 sharpens this:
  demands are capability (base, shape) — arguably identity — while fractions and
  grouping are placement hints — arguably not.
- Generation-only runs (algo=None still NotImplementedError): the experiment
  contract is identity + the sealed-by-ledger commit protocol + an extent +
  self-description — none require gradients. Needs a committing Sealer daemon
  (Trainer minus post/gradients) and the wave-shape/extent knobs (group_size,
  trajectories_per_wave, n_waves) relocated out of Schedule (ties into the
  schedule-split thread below).
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
