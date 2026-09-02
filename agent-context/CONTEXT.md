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

46. ADAPTER KINDS BEYOND LORA: SOFT PROMPTS SERVED, SIDE ATTENTION REFUSED
    (settled by execution, Samarth-directed: "take care of soft prompts, side
    attention, etc. implementations... the test for this will be seeing if we
    can execute grpo runs with different adapter types on the same vllm
    engine"). #25 gave soft_prompt and attn_bias declaration halves — sites,
    mechanisms, exports — and no compute. This entry gives soft_prompt both
    halves on metal, gives attn_bias the replay half only, and records exactly
    why the second one stops there.
    - THE SOFT PROMPT'S REPLAY LOWERING IS A BOUNDARY, NOT A SITE
      (policy/adapters/soft_prompt_torch.py, new). LoRA replaces a Linear;
      a soft prompt cannot, because it does not change a computation, it adds
      POSITIONS. PromptBoundary hooks the base's forward: it embeds the ids,
      prepends the routed rows, widens the padding mask by n ones, and — the
      rule the file exists for — CUTS THOSE POSITIONS BACK OFF THE LOGITS
      before they leave. forward_backward returns [len(batch)] logprobs
      aligned to batch.token_ids and a virtual row is a position with no
      token, so it must not survive the boundary. Virtual positions exist
      between embed and logits and nowhere else: flatten, pack and the
      learner's gather need no knowledge of them, and torch_learner.py was
      NOT edited.
    - TWO TRANSPARENCIES MAKE MIXED-KIND TENANCY WORK, and they are the same
      rule twice. A row whose slot carries no soft prompt takes no virtual
      positions (the boundary passes its kwargs through); a uniform forward
      whose slot carries no delta at a wrapped LoraSite gets the base. Before
      this, a co-tenant with a different bank shape would have hit a KeyError
      inside someone else's lowering — #44's additive install made every
      tenant's sites resident, but every tenant's forward still assumed its own
      bank covered them. A mixed forward still raises (admission is the
      coalescer's job, #44), which is unchanged.
    - A SOFT PROMPT HAS NO IDENTITY ELEMENT. LoRA's B = 0 delta IS the base; n
      virtual positions change the forward at version 0 by construction. Init
      is therefore small, seeded, and part of the policy (init_std defaults to
      0.02 — transformers' own initializer_range, and a good match for the
      measured Qwen3-0.6B embedding std of 0.0292).
    - THE ROLLOUT LOWERING USES vLLM 0.28's MIXED EMBEDS PROMPT, which is
      better than "we embed the prompt and hand over the whole thing".
      VERIFIED IN THE PINNED IMAGE, not from memory: EmbedsPrompt takes
      prompt_embeds AND prompt_token_ids AND a per-position
      prompt_is_token_ids mask; the ENGINE embeds every position marked True
      from its own table (gpu_model_runner zeroes the placeholder ids before
      the gather and writes the result back without clobbering the embeds
      positions), so the learned rows are the only thing we hand over and
      nothing depends on our copy of the embedding matrix matching the served
      one. The real token ids stay in the request, so prefix-cache block
      hashes, detokenization and prompt_logprobs indexing all see the tokens
      they would see without a soft prompt — and _gen_prompt_embeds_extra_hash_keys
      digests the embeds per block into the block hash, so two bundles'
      prefixes can never alias (no cache_salt needed; the plugin-era rule is
      already enforced by vLLM here). Other verified facts: enable_prompt_embeds
      coexists with enable_lora on one build; a request may carry prompt_embeds
      AND a LoRARequest; prompt_embeds and punica requests are in flight
      together and show batch-composition noise against their serial answers,
      i.e. they really share forward steps; sleep(1)/wake_up is unaffected;
      and enable_prompt_embeds puts the build on the V1 model runner (it is on
      the V2 runner's unsupported list) and moves the embedding layer outside
      the CUDA graph.
    - REACHABILITY IS A BUILD FACT, AND NOW LITERALLY A CONSTRUCTOR ARGUMENT.
      VllmEngine(prompt_embeds=True) is what enables the lever; a build that
      was not asked for it reports NONE at model.embed_tokens and a
      soft-prompt tenant is refused at Phase 0 rather than served wrong. This
      is #25(b)'s rule finally having a real second instance.
    - PARITY (deploy/adapters_l4.py::parity, 21/21 on one L4, ~4 min). The
      control that settles it has NO tolerance: a soft prompt whose rows ARE
      the embeddings of four real tokens must serve exactly as those tokens
      and replay exactly as those tokens. Both come back BIT-IDENTICAL
      (max|d| = 0.00e+00), which settles positions, the padding mask, the
      is_token_ids mask, the logit trim and the scored-suffix offset at once.
      The cross-side gap is then numerics, read on two document sets because
      the two questions want different material: PLAIN (rollout-shaped) is
      where the floor is measured — base 0.040, lora 0.041, both 0.049,
      soft_prompt 0.048-0.125 across row magnitudes 0.005/0.02/0.05, all at
      #28's 0.022-0.033 kernel floor's order — while JAGGED (a predictable
      token beside a wildly surprising one) is where the shift control is a
      proof: 4.1-4.5 nats shifted against 0.04-0.24 aligned. On a smooth
      sequence an off-by-one hides in the noise, which is why #45's shift test
      needs material like this.
    - THE ACCEPTANCE TEST, Samarth's own (deploy/adapters_l4.py::adapters,
      13/13, ~25 min on one L4). Three GRPO tenants join one Host 25s apart
      and share ONE VllmEngine and ONE multi-tenant learner: lora
      (f09a696e0c48, reward 0.500 -> 0.938, gap 0.016-0.026), soft_prompt
      (3ef3b03e0611, 0.438 -> 0.688, gap 0.028-0.056), and a bank carrying
      BOTH kinds (98ee4812037d, 0.375 -> 0.938, gap 0.019-0.038). All 8/8
      updates, all evals present, engine holding 18 punica bundles and 18
      prompt-row bundles at once. The gap is the cross-contamination alarm: a
      request served the wrong prefix or the wrong adapter blows it up long
      before 0.15.
    - A SOFT PROMPT WANTS ITS OWN LEARNING RATE, learned by collapsing a run.
      AdamW's step is ~lr per coordinate, so one update moves the rows by
      lr*sqrt(n*d) = lr*90 against a row block whose whole norm is 1.8 at the
      default init. At 1e-2 that is half the prompt per step: reward hit 0 by
      update 3 and froze there (no reward spread -> no advantage -> no
      gradient), with logprob_gap at 0.25 because rows that far outside the
      embedding distribution are exactly where prefill and decode kernels
      diverge. 5e-4 is ~4% per step and trains. The general rule: a kind's
      sensible lr scales with 1/sqrt(its parameter count), and the bank's
      single OptimSpec.lr is the wrong shape for a bank of mixed kinds —
      OptimSpec.overrides exists per entry and is the lever.
    - ATTN_BIAS: THE REPLAY HALF IS REAL, THE ROLLOUT HALF IS REFUSED
      (policy/adapters/attn_bias_torch.py, new; NOT PROVEN end to end and
      deliberately unreachable). An attention mask already IS a score-level
      bias, so the trainer needs no kernel patch — hand the base a 4-D float
      mask instead of the 2-D padding mask. Verified on the pinned
      transformers 5.16.1 + Qwen3 (sdpa) BEFORE writing it: a 4-D mask that
      merely reproduces the causal mask is bit-identical to the 2-D one, a
      +2.0 bias on three columns moves the logits by 8.7 nats, a bias on one
      head alone by 2.1. theta per (head, prompt row), two named
      parameterizations (free / bounded_sigmoid) both mapping zero to zero so
      version 0 IS the base, n read from the site name (the canonical name
      carries it), the bias applied to the REAL tokens' rows only — the
      prompt's own positions stay a pure function of the rows, which is what
      would let an engine precompute their K/V once per bundle. Cost stated:
      the mask is [rows, heads, L, L] where the ordinary path passes
      [rows, L].
    - WHAT BLOCKS SIDE ATTENTION ON vllm 0.28.0, precisely. The registration
      seam MOVED and still exists (vllm.v1.attention.backends.registry.
      register_backend, with an AttentionBackendEnum.CUSTOM member, and
      load_general_plugins() still runs in BOTH the engine core and the
      worker — #16's ambient registration is intact). merge_attn_states MOVED
      and still exists (vllm.v1.attention.ops.merge_attn_states). THE BLOCKER
      is that return_softmax_lse is NOT plumbed through the dense
      FlashAttention path: FlashAttentionImpl sets can_return_lse_for_decode =
      True but v1/attention/backend.py reads it only when dcp_world_size > 1,
      and every other user on this build is an MLA backend or the
      context-parallel helper. #25's "stock kernel + tiny partition attention
      + exact LSE merge" therefore has no seam here short of forking 262 lines
      of FA-version-conditioned dispatch. SideAttention.required_symbols now
      names the CURRENT paths plus the one this build genuinely lacks, so
      probe() fails for the true reason (pinned as a test), and every engine
      keeps reporting NONE for SIDE_ATTENTION.
    - WHAT WOULD UNBLOCK IT, and it is not #25's design: FLEX ATTENTION, which
      this build ALREADY has. FlexAttentionMetadata carries a first-class
      `score_mod` field — torch's (score, b, h, q_idx, kv_idx) -> score hook,
      which is exactly an additive bias on the score rectangle — and
      get_transformed_score_mod() already converts paged PHYSICAL kv indices
      to LOGICAL per-request ones, which is the job BatchView was invented
      for. Nothing in vLLM sets score_mod today, so the work is a registered
      backend plus a metadata builder carrying our per-request slot vector;
      the cost is that FLEX_ATTENTION becomes the whole engine's backend (a
      build fact, and a different numerics baseline for every tenant on it).
      The LSE-merge route stays the fallback if a later vLLM plumbs the dense
      LSE the way the MLA backends already do.
    - PROPOSED SPEC DELTA (for the main session; canon untouched). (1) I2/I8
      should say that a kind's replay lowering may be a BOUNDARY around the
      base's forward, not only a module replacement — with the alignment
      obligation stated: whatever a lowering adds to the sequence it must
      remove before the logits, so PolicyOutputs.logprobs stays [len(batch)]
      token-aligned. That obligation is what makes data/ able to stay
      estimator-free about adapter kinds. (2) The "one delta per site" bank
      rule (#2) needs a companion for OPTIMIZATION: two kinds in one bank do
      not share a sensible learning rate, so a spec with a mixed bank that
      leaves OptimSpec.overrides empty is arguably a validate warning, not a
      silently divergent run.
    452 tests green on fakes at the end of this track (30 of them in
    test_soft_prompt.py, torch-gated and green in the image); 463 once #47
    merged alongside. NOT BUILT, stated: side
    attention serving (above); a soft prompt under TP > 1 (the rows are a
    request field, so nothing should shard, but it has not been run); a
    soft-prompt tenant through the RemotePool wire (prompt rows are engine-
    side state built at add_bundle, which the transport already carries as
    payload bytes, but again unrun); score_tokens under a soft prompt is
    implemented with the offset and exercised by the parity harness, but no
    opsd tenant has used it.
    CORRECTION (LSE spike, 2026-08-28, branch worktree-agent-a09fda7796df17719,
    unmerged): #46's stated blocker is imprecise. The dense FlashAttention
    path CAN yield the LSE on 0.28.0 without forking dispatch:
    flash_attn_varlen_func is a rebindable module global called exactly once
    per dense forward with out= (return discarded), so a shim adds
    return_softmax_lse=True invisibly — 159 executable lines, zero copied
    dispatch, 18/18 on metal: bit-identical identity at zero bias, exact
    LSE-merge arithmetic (#25's math finally executed), gated/ungated
    separation exact, disarm restores stock exactly. Also: side_attention.py's
    required_symbols names a fictional `dense_lse` symbol — must be rewritten
    to name the module global + kwarg if anything lands. Costs measured, not
    hidden: cascade attention declined while armed (and cascade fires on
    exactly GRPO-wave shapes), full CUDA graphs forfeited, prefix caching
    unwired (cache_salt), ~3x warm latency un-optimized. Recommendation on
    record: write attn_bias's rollout lowering as a SCORE_MOD (ports to
    FlexAttention today and FA4/Blackwell later — vLLM ships an in-tree FA4
    precedent, fa4_rel_attention with in-kernel additive bias); keep the LSE
    patch as the measured fallback. FLAVOR DECISION IS SAMARTH'S — nothing
    merged, reachability still NONE.

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

48. THE LOWERING: ONE CONTRACT FOR BOTH SIDES' ADAPTER ATTACHMENT (design
    settled, Samarth-directed: "the trainer pattern is the gold standard...
    can we come up with a unified abstraction with some verbs that applies
    for both — essentially a 1 to 1 mirror"). This COMPLETES #3, which
    declared every kind ships a ROLLOUT lowering (engine) and a REPLAY
    lowering (trainer) — the replay half became real files (lora_torch,
    soft_prompt_torch); the rollout half got smeared into vllm_engine.py as
    private methods (_register_lora / _register_prompt_rows / _prompt_for /
    the score offset). The smear is the finding (post-#47 review): request
    assembly and answer-geometry knowledge live inline in the verbs and
    grow a conditional per mechanism.
    THE CONTRACT — one Lowering per (kind, side), four verbs, read off the
    tendril inventory:
      demands   what the substrate BUILD must pay (engine args, plugin
                presence; trainer: nothing today)
      attach    make the kind's state resident — ADDITIVE (trainer: wrap
                sites / hook the boundary; engine: payloads -> LoRARequest /
                prefix rows)
      apply     contribute to ONE unit of work (trainer: the row plan's row;
                engine: prompt form + request kwargs). The units DIFFER by
                side (row vs request; our plan batches the trainer, vLLM's
                scheduler batches the engine) — apply takes the side's unit,
                the contract does not pretend they are the same.
      align     keep answers aligned to real tokens (trainer: trim virtual
                positions so logprobs stay [len(batch)]; engine: positions
                occupied, SUMMED across kinds for the score offset).
    Both substrates become BUSES: loop the kinds, call the verbs, merge
    apply's levers, sum align. vllm_engine.py loses all mechanism awareness
    (_prompt_for's soft-prompt knowledge -> soft prompt's apply; the
    mechanism dispatch -> the bus loop); "native vs plugin" stops being a
    category — it is a demands() difference only.
    GEOGRAPHY (I2-exact): both lowerings live WITH THE KIND —
    adapters/lora_vllm.py beside adapters/lora_torch.py — because the
    policy bridge is the ONLY two-world primitive, and parity (the exam
    binding the pair) becomes reviewable in one directory. rlstack_engine/
    shrinks to its honest charter: only code that patches vLLM internals
    (side_attention), reached by string from its kind's rollout file. Lazy
    vllm imports keep the fakes suite stdlib-clean (rule 7, the lora_torch
    precedent).
    COMPOSITION RULE: one request may carry several kinds' levers; prefix
    contributions concatenate in BANK ORDER; at most the composition the
    engine can express (soft_prompt + lora proven; a second prompt-shape
    kind must compose or refuse loudly).
    SCOPE DISCIPLINE: the trainer side ALREADY realizes the contract
    (attach=install_replay, apply=the sites reading the row plan,
    align=the boundary's logit trim) — it is documented as conforming, NOT
    renamed; no trainer files change in this pass. Proof obligation: the
    463-test suite green, then tp_l4 + adapters_l4 + the OPD probe re-run
    on metal (this relocates the most metal-proven path in the repo).
    LANDED (2026-08-28, built and re-proven). GEOGRAPHY AS BUILT: the
    contract is policy/adapters/rollout.py — RolloutLowering plus the typed
    records (ServingBuild, BuildDemands, Request, Levers, Alignment) — filed
    beside replay.py, because rule 8 already names that slot "the kinds'
    shared compute-side seam" and this is that seam on the serving side; the
    three rollout halves (lora_vllm, soft_prompt_vllm, attn_bias_vllm) sit
    next to their _torch twins. Adapter grew ONE member,
    rollout_lowering(build) — install_replay's twin — implemented by each
    kind with the lazy import lora_torch already precedents, so the fakes
    suite stays stdlib-clean.
    THREE DELTAS FROM THE DESIGN, all small. (1) A FIFTH verb, reaches(meta):
    once the bus stopped knowing mechanisms, something had to answer
    Engine.reachability, and the honest owner is the kind — demands() says
    what the build pays, reaches() says what the payment buys. (2) The
    composition rule needed something to compare, so a lowering DECLARES what
    it claims of a request (lora: the lora_request keyword; soft prompt: the
    prompt form) and check_levers_compose refuses two kinds claiming one
    lever at add_bundle — which is exactly where a second prompt-shape kind
    lands. (3) The build fact is now VllmEngine(serves=("lora",
    "soft_prompt")) — KINDS, not mechanisms; max_loras/max_lora_rank keep
    their vLLM-flavored names because every call site spells them that way
    and are passed on as plain capacity, and they are the ONLY
    mechanism-flavored words left in vllm_engine.py (which otherwise greps
    clean for punica/rows/lora and keeps "prompt" as request vocabulary only:
    prompt_ids, prompt_logprobs, TokensPrompt). compile.group_by_kind is the
    bus's dispatch input; group_by_mechanism survives as the mechanism-keyed
    view of it.
    THE METAL, which is the point of the entry — 473 tests green first (the
    463 plus 10 for the seam), all 473 green in the image:
      tp_l4 13/13 on L4:2 — punica under TP and score_tokens through the bus;
      scoring gap by adapter magnitude base 0.02087 / faint 0.04544 / loud
      0.27661, i.e. #45's 0.02/0.05/0.28 reproduced through the new path.
      adapters_l4::parity 21/21 — and the control with no tolerance still has
      none: rows-as-tokens max|d| = 0.00e+00 on BOTH sides, shift controls
      4.1-4.5 nats against 0.04-0.24 aligned, the sweep landing on #46's own
      numbers (base floor 0.0400, lora 0.0405, soft_prompt 0.048-0.125).
      adapters_l4::adapters 13/13 with THREE FRESH tenants (seeds 480, so
      they trained rather than attaching to #46's finished runs): lora
      9130f1aa2cad 0.500 -> 1.000, soft_prompt 965372217a46 0.562 -> 0.688,
      both 2351c5a5b39a 0.500 -> 0.938, gaps 0.025-0.073 with the
      cross-contamination alarm quiet, and the bus's census reading 18 lora +
      18 soft_prompt bundles resident on one engine — #46's number exactly.
      opd_l4::probe 4/4 against the 32B tp=4 teacher (the align path with
      NOTHING attached: a payload-free bundle sums to zero positions) — and
      the per-token scores came back BIT-IDENTICAL to #47's recorded ones
      ([-0.3004, -0.0213, -0.0026, -0.0013]; ' 105' -0.0814 vs ' 731'
      -4.1414), which is the strongest statement available that the
      relocation changed no number.
    SCOPE HELD: no trainer file, no FakeEngine, no runner/remote.py — the
    wire ships payloads and the serving bus consumes them exactly as before.
    deploy/adapters_l4.py took four lines (serves= at two constructors; the
    two engine._lora/_rows peeks became the bus's public attachments() /
    residency()). DOC DEBT, one line larger than #44(f) left it: STYLE rule
    8's adapters/ line now owes "one file per kind — declaration, replay
    lowering, rollout lowering — plus replay.py AND rollout.py, the kinds'
    two shared seams".

49. THE PARTITION SAYS WHAT METAL IT IS MADE OF (settled, Samarth-directed:
    "gpu name in partition/host unit. add vram constraint instead of just
    fraction (this is optional)"). #43 made the Host an ATOMIC PURPOSED
    PARTITION but described its metal with one number — a fraction — and a
    fraction cannot tell half an L4 from half an H100: the host-up journal,
    the status frame and the hosts view all printed "0.50" and meant
    different amounts of different silicon.
    - THE KIND IS CARRIED, NEVER DECLARED TWICE: Partition gains
      `gpu: str = ""` (LAST field, so every positional construction in the
      deploys and tests still reads) and Fleet.carve STAMPS it from the
      Metal whose residual it drew — one source (registration, the acquire
      rung a human executes), one copy (the carve), no second declaration
      that could disagree with the first. Partition.row() is the ONE
      journal/status/wire shape: host-up's partition dict, the carve event's
      new "gpu" beside "metal", and Host.status()["partition"] — which is
      now JSON-safe, so the three Modal serving classes dropped their
      `if k != "partition"` filters and a remote partition finally
      advertises its own metal over the wire. The observer's hosts view
      grew one line, `metal : L4 node-a[0,1] @ 0.50` ("unpartitioned" for a
      bare host — every pre-#49 host still journals, statuses and renders).
    - THE VRAM HALF, ONE FUNCTION AND ONE RULE: Metal gains
      `vram_gb: float = 24.0` (ONE device's VRAM, the unit a model is
      measured against — L4=24, A100/H100=80) and
      fleet.fraction_for_gb(gb, metal) is the only place the two units meet:
      the hint is gb / metal.vram_gb on the target metal, and past one
      device it RAISES rather than clamping, because more VRAM per shard is
      not a smaller fraction — it is bigger metal, i.e. the acquire rung.
      Partition.memory REMAINS a fraction: both substrates take one (vLLM's
      gpu_memory_utilization, torch's set_per_process_memory_fraction), so a
      GB figure is converted against the metal it will live on and never
      stored. Nothing gained a second unit to keep in sync.
    - SCOPE HELD: no file under spec/ changed, so GB touches neither
      PoolMember.fraction (still THE declared unit) nor run identity; the
      only new caller of the conversion is a human writing a fraction at the
      fleet. 479 tests green (473 + 6: the carve's stamp, the conversion
      both ways, the >1-device refusal, a GB-sized carve, the journal/status/
      view triple, and the unpartitioned fallback).
    PROPOSED, NOT DONE (each is Samarth's call, and (a) is the big one):
      (a) PoolMember.vram_gb / LearnerMember.vram_gb as a declarable
          alternative to `fraction` — the honest way to say "this model
          needs 20 GB" once instead of recomputing a fraction per GPU kind.
          Spec-side, so it lands in gpu_config and therefore in the run_id
          hash: a run declared in GB would carry a DIFFERENT identity from
          the same run declared as a fraction, which is exactly the
          identity-rings open thread (are fractions placement hints that
          should leave the hash?). Do that ruling first.
      (b) a submit-time hint map — Fleet.place(spec, hint_gb={pool: gb}) —
          identity-free by construction, converted through fraction_for_gb
          before plan_carve sizes the partition. Cheap, but it grows the
          fleet API for a caller that does not exist yet.
      (c) plan_carve consulting VRAM directly: refuse a carve whose hint
          exceeds the metal's device instead of first-fitting a fraction
          that silently means less GB than the tenant needs. Needs (a) or
          (b) to have a GB figure to check.
      (d) Metal.vram_gb defaulting off a kind table (L4=24, A100=40|80,
          H100=80, B200=192) rather than a bare 24.0 — one dict, but it
          makes the register-time default silently kind-dependent, and the
          A100 40/80 split shows why a stated number beats a lookup.

50. THE OBSERVER OVERHAUL: HOVER, HOST PAGES, RUN SWITCHING (settled,
    Samarth-directed: "make UI much nicer. keep nice color scheme +
    minimality, except allow for hover to reveal raw numbers, allow for
    host-by-host analysis in addition to experiment by experiment, allow for
    experiment switching in dropdown, start thinking (minimally) about how
    hosts can expose metrics like throughput (but dont implement yet)... plot
    what is global"). Entirely inside observe/ + tests/test_ui.py: no core
    touch, no runner emission, palette and minimality unchanged.
    - THE SECOND READING (observe/host_series.py): the per-HOST reading, the
      way series.py is the per-RUN one, over hosts/<name>/log.jsonl alone.
      Four named readings, one per kind of fact the journal carries:
      boot_facts (host-up's birth attestation — engines, and for #43-era
      hosts the Partition and Regimes; older journals carry neither and the
      page SAYS so rather than inventing one), tenancy_lanes (attach/detach
      paired into residencies — a run that attached twice is two, an open
      attach is still resident), gpu_channels (stats as per-device util and
      memory series), metric_series (the slot, below). fleet_data is the
      GLOBAL join: views.hosts_data + views.runs_data + each host's lanes
      and utilization shape under one shared time window.
    - PLOT WHAT IS GLOBAL: placement, residency, load and timelines render
      on /hosts and /host/<name>; per-run curves stay on /run/<id>. Routes
      and API doubled (/hosts, /host/<name>, /api/hosts, /api/host/<name>);
      the run page cross-links the hosts it ran on, the index's host column
      links to their journals, and a host's tenancy table links back to the
      runs — the placement graph is navigable from either end.
    - HOVER IS THE RAW NUMBER: every chart carries a crosshair, per-series
      markers, and a tooltip with each series' value at FULL logged
      precision (the card's own "now" stays four digits, axis labels stay
      compact) — the dashed eval overlay included, labelled @u<n> when its
      nearest point is not the crosshair's update. Timeline bars hover too
      (run, status, from/to, duration, pools). A poll never redraws under
      the cursor.
    - THE SWITCHER: a <select> over /api/runs on the run page, rebuilt only
      when the run list itself changes, so an open dropdown never closes
      under the user mid-poll.
    - CHARTS ARE MEASURED: each plot draws at its container's own pixel
      width (viewBox == css px), so nothing letterboxes at any card width;
      and a gap in a host's samples wider than 4x that journal's own median
      cadence BREAKS the line instead of bridging it — downtime is not a
      straight line between two live points (the gpu view's dark_gaps rule,
      drawn).
    - THE THROUGHPUT SLOT — DESIGN ONLY, NOTHING EMITTED: metric_series
      plots any numeric field an event carries that the named readings do
      not claim, keyed <event>.<field> (nested dicts flattened with dots;
      bools are flags and lists are facets, neither is a series). HOW A HOST
      WOULD EMIT, when we build it: exactly as run_stats already does — one
      more async journal loop on the Host, started beside the submissions
      and cancelled with them, appending {"event": "throughput", "t": ...,
      <numeric fields>} to its own hosts/<name>/log.jsonl, sampled from what
      the metal already knows (vLLM's engine statistics — prompt and
      generation tokens/s, running and waiting requests, kv-cache usage —
      and the learner's tokens and microbatches per second). It stays
      observability: never read by correctness, never in identity, and
      per-HOST rather than per-tenant (a shared engine's throughput is a
      property of the metal; a tenant's share is a run fact the ledger
      already carries). The UI needs no change to show it — the day such a
      line lands, the host page grows a card.
    - page.py split out of ui.py: THE document (one self-contained
      HTML+CSS+JS page, four routes) is its own file; ui.py is the routes.
      views.py untouched.
    486 tests green (13 new in test_ui.py: host-journal fixtures, tenancy
    pairing incl. re-attach, per-device channels, the schema-tolerant slot
    with its claimed-field table, the fleet join, the routes and their 404s,
    the switcher payload, the hover machinery). VERIFIED DEPLOYED beside the
    volume (modal deploy, same URL): /api/hosts renders 14 hosts and 31
    experiments over a 55,000s window; modal-math-teacher-32b shows
    partition modal-l4 [0,1,2,3] mem 0.9 with regime serve-tp4 = inference x
    Qwen3-32B x 4; l4-fsdp2 shows 14 boots and 14 residencies including
    resume pairs of one run_id; modal-math-opd-shakeout renders two devices
    x 26 stats samples with one failed and one LIVE residency (remote pools
    main+teacher); the pre-#43 hosts (l4-stress, l4-arith, l4-adapters)
    render with partition None and no regimes, as designed; and metrics is
    [] on every host — the slot is open and nothing emits into it yet.

51. MASS PARTITIONING ONE L4: THE SUB-GPU HOST MEETS REAL SILICON (settled by
    execution, Samarth-directed: "try mass partitioning one L4 gpu to do many
    runs simultaneously with sleep, etc. etc. i just want to test the shit out
    of everything"). #43 made a Host an ATOMIC PURPOSED PARTITION and #49 gave
    the partition its metal, but every GPU either of them ever touched was a
    WHOLE device: sub-GPU hosts were a fakes-proven design. This entry is that
    design's first contact with silicon — one L4, carved by the fleet into
    three and then four coexisting hosts, plus an alternating host whose
    wake/evict hooks are REAL for the first time anywhere in this repo.
    Deploy only: deploy/partition_l4.py is the one new file, nothing under
    rlstack/ changed, 479 tests still green. Cost: 20.7 GPU-minutes across
    four L4 containers (~$0.30) plus two CPU containers, all legs replicated
    at least once.
    THE VERDICT FIRST: the sub-GPU host model HOLDS. A fraction means what it
    says on the metal, many partitions coexist in one container, joins are
    free, alternation really hands the device back, and the ladder's refusals
    are correct. It creaks in four places, all of them NAMING, PLUMBING or
    OBSERVABILITY rather than the model — listed as findings below, none
    fixed here (this was a test campaign; rlstack/ was out of scope).
    - THE PARTITION IS REAL, AND ITS ARITHMETIC IS SIMPLE. The device's usable
      total is 22.03 GiB (torch.cuda.mem_get_info; nvidia-smi says 23034 MiB),
      and an engine built with gpu_memory_utilization = its partition's memory
      costs fraction x 22.03 PLUS ~0.30 GiB of CUDA context that sits OUTSIDE
      vLLM's budget: main @0.30 took +6.90 GiB, judge @0.15 took +3.61 GiB,
      two engines resident at 10.69 GiB, five tenants' learner state on top at
      13.03 GiB, four partitions' worth of metal at 14.14 GiB — reproduced to
      0.01 GiB in a second container. So the packing rule for a mass-
      partitioned device is: fractions sum to <= 1 minus ~1.4% per additional
      engine process. TWO vLLM ENGINES IN ONE CONTAINER ON ONE DEVICE DO NOT
      FIGHT: each v1 engine core is its own child process with its own ipc
      paths, and vLLM measures its budget against the DEVICE total, not
      against what is left, so partitions compose additively by construction.
    - THE LADDER, ON METAL: two place/apply calls carved main-tp1 @0.30,
      learner-fsdp1 @0.20 and judge-tp1 @0.15 (a second base, see finding (f))
      off one device, residual 0.35 exactly. Five concurrent tenants — grpo,
      grpo (another seed), gspo, sdft, and an llm_judge tenant whose judge
      pool routes to the OTHER inference partition — all JOINED (never
      carved), all completed 4/4 with evals, logprob_gap 0.017-0.046 (the
      kernel floor, so no adapter was ever served to the wrong tenant across
      three partitions), and declared_load stayed 0.0 on every host: a join
      really is fraction-free. Each tenant's runner sat on the learner's host
      and reached the other one or two partitions through RemotePools over
      LocalTransport — three partitions per tenant, all in one process.
      Placement journaled under every run_id.
    - ALTERNATION, WITH HOOKS THAT ACTUALLY EVICT. A sleep unit carved ONE
      host with two regimes at 0.45; the regimes attached at birth into
      "host:<name>", and the deploy then attached REAL hooks onto those same
      residents — arbiter.attach is idempotent, filled the hooks, and did not
      re-legislate the group (#43's deference rule, on metal). The run made 9
      switches, strictly alternating main-tp1 <-> learner-fsdp1. vLLM's
      sleep(1) returned 9.92 GiB (11.57 -> 1.65) and wake_up() put back 9.81;
      the learner offload (model .to("cpu") + empty_cache) returned 1.78 GiB
      (2.53 -> 0.74). Between regimes the whole device sat at 0.74 GiB used —
      a sleep host hands the metal back, it does not merely promise to. The
      run finished 4/4, gap max 0.0298: alternation costs latency, not
      numerics.
    - THE LIVE CARVE AND THE REFUSALS: fraction_for_gb(3.0, L4) = 0.125 sized
      a carve that happened mid-campaign, journaled with gpu "L4" stamped, and
      the host it made ran a real 2-update tenant on the second base. Both
      refusals came back needs_human on a full device — a tp=2 demand on a
      one-device metal, and a 20 GB demand against a 0.35 residual — and
      apply() refused the plan ("the plan needs new metal — a human's call").
    - KILL/RESUME THROUGH THE FLEET: a tenant cancelled mid-flight at update 4
      of 8 and resubmitted as the same spec came back resumed_from=4, ledger
      complete 8/8, evals backfilled at 2/4/6/8.
    - THE OVERCOMMIT BOUNDARY (leg 7, added because finding (a) is a way to
      reach it by accident): asking for 0.75 of a device already holding a
      0.30 partition is REFUSED AT BUILD, cleanly, by vLLM itself —
      "Free memory on device cuda:0 (14.76/22.03 GiB) on startup is less than
      desired GPU memory utilization (0.75, 16.53 GiB)" (v1/worker/utils.py
      request_memory), surfaced to the caller as RuntimeError: Engine core
      initialization failed. AND THE PARTITION THAT WAS ALREADY SERVING
      SURVIVED IT: the first engine kept sampling, its HBM unchanged. A failed
      carve is a failed carve, not a dead device.
    FINDINGS (repro in deploy/partition_l4.py; the harness's failing checks
    ARE these, deliberately left failing):
      (a) THE CARVE NAME IS NOT UNIQUE PER CAPABILITY, AND THE COLLISION LEAKS
          RESIDUAL. Fleet.carve names a host f"{metal}:{devices}/{regimes}"
          and regime_of names a training regime "learner-fsdp{shape}" — with
          no base in it. Two carves that differ only by BASE therefore collide:
          self.hosts[name] = host REPLACES a live host (its metal still
          resident, its tenants still bound), and because residual() sums over
          self.hosts, the replaced partition's 0.20 silently returns to the
          residual — the fleet then believes 0.425 of the device is free when
          0.225 is. That is an automatic path into the overcommit boundary
          above. Free repro on fakes (`modal run deploy/partition_l4.py::
          preview`), reproduced on metal twice. Fix candidates: put the base
          (or a counter) in the carved name, or refuse a carve whose name
          exists.
      (b) THE OBSERVER CANNOT SEE A CARVED HOST AT ALL. A carved name contains
          "/", Store.append_host_event writes hosts/<host>/log.jsonl, and
          Store.list_hosts() recovers the name with key.split("/")[1] — so
          every carved partition journals one directory deeper than the
          observer looks. The whole campaign (4 partitions, 15 runs, 11
          minutes of nvidia-smi samples) renders as ONE phantom host
          "l4-solo:0" with "metal : unpartitioned", 0 boots, 0 tenants; and
          because runs_data and gpu_data are host-journal-driven, the runs and
          the gpu samples are invisible too. The fleet log (keyed by nothing)
          is the only place the campaign's story survives. #43's tests never
          saw this because they read the journal back by exact name.
      (c) THE FACTORY CONTRACT CANNOT PAY THE PARTITION. Fleet's factories are
          Callable[[Regime], Engine] and a Regime carries no memory, so the
          one number the carve just computed — the fraction, the whole point
          of a sub-GPU host — cannot reach gpu_memory_utilization. Every real
          deploy has to re-derive it from the Plan first (Partitioned.absorb
          in the harness). A factory taking (Regime, Partition) deletes that
          class.
      (d) THE ENGINE HAS NO SLEEP SEAM, AND NOTHING HAS EVER ATTACHED A HOOK.
          VllmEngine takes no enable_sleep_mode, so an alternating host cannot
          be built through the fleet's factory without poking
          engine._engine_args; and the arbiter's wake/evict hooks — designed
          in #34, carried through #43 — had never been attached by anything on
          metal, so a deploy must write both hooks itself and reach
          engine._llm / learner._model to do it. The hooks work (see above);
          the seam is missing. Candidates: VllmEngine(sleeps=True) as a build
          fact, and a named sleep/wake pair on the Engine and Learner
          protocols that Host wires when a host has >1 regime.
      (e) A TRAINING PARTITION'S FRACTION IS UNENFORCEABLE IN-PROCESS. vLLM's
          gpu_memory_utilization is per-engine, but torch's
          set_per_process_memory_fraction is per PROCESS and every host in a
          mass-partitioned container shares one — so a training partition is a
          declaration and a journal entry, never a cap. It behaved (the shared
          learner held ~2.3 GiB across five tenants against a declared 0.20 =
          4.4 GiB) but nothing made it.
      (f) TWO HOSTS OF ONE CAPABILITY ARE UNADDRESSABLE. Capability is (kind,
          base, shape) and find_join takes the FIRST covering host in sorted
          name order, so a second partition serving the same base at the same
          tp is dead metal no tenant can ask for. The campaign gave its second
          inference partition a different base (Qwen3-0.6B-Base) purely to
          make it reachable. If sub-GPU hosts are meant to be spread across
          identical capabilities, placement needs a currency beyond capability
          (declared_load, or a measured saturation signal — the join-refusal
          thread #43 already left open).
    NOT DONE (deliberate): none of (a)-(f) is fixed — this campaign was
    evidence, and a core fix belongs in a session that can change rlstack/.
    Untested here: sub-GPU partitions ACROSS containers (every host in this
    campaign shared one process, so SM contention across partitions is
    measured only as latency, never isolated); more than two engines on one
    device (the arithmetic says ~5 x 0.15 fits, nothing tried it); a sleep
    host under several tenants at once; sub-GPU partitions at tp>1.

52. THE FOUR FLEET FIXES: #51's FINDINGS (a)-(d), CLOSED IN rlstack/ (a core
    session, directed: fix (a)-(d), leave (e) and (f) alone). #51 was a test
    campaign that could not touch rlstack/; this entry is the repair, and its
    proof is the same harness re-run — deploy/partition_l4.py legs 1, 2 and 6
    on ONE L4: 41 + 12 + 3 checks, 0 failed (the four checks #51 left
    deliberately failing are among the passes). 497 tests green on fakes
    (five new) and 497 in the image. Cost: 15.3 GPU-minutes across three L4
    containers (~$0.20), one of which was spent re-running leg 1 after a
    harness artifact (see the volume note below).
    (a) A CARVE NAME IS UNIQUE, AND REGISTRATION REFUSES TO REPLACE — both
        halves, because either alone is a promise. Fleet.carve_name() names a
        carved host "{metal}:{devices}.{regimes}.c{n}" with a PER-FLEET CARVE
        ORDINAL: a regime carries kind and shape but not base, and the base
        cannot go in the name either — a base is "Qwen/Qwen3-0.6B" and a host
        name is a journal path segment that may hold no "/" (finding (b)), so
        the ordinal is the honest discriminator. Fleet.register() is then the
        one door into self.hosts (the constructor's seeded hosts included) and
        RAISES FleetError on a name already taken: a replaced host keeps its
        metal — engines resident, tenants bound — while dropping out of the
        dict residual() sums over, which is how a live partition's memory
        silently returned to the residual. Metal: four carves off one device
        named .c1-.c4, TWO of them learner regimes differing only by base
        (Qwen3-0.6B @0.20 and Qwen3-0.6B-Base @0.125), both alive at the end
        with their own tenants, residual 0.225 exactly — the #51a repro,
        inverted. Fakes: test_fleet's two-carves-differing-only-by-base and
        the registration refusal.
    (b) A HOST NAME IS ONE JOURNAL PATH SEGMENT, ATTESTED AT BOTH LAYERS.
        Host.attest_name() (a birth fact, beside attest_regimes) raises
        HostError on a name containing "/", and Store.append_host_event
        asserts the same at the key it writes — loud twice, because the layer
        that owns the name and the layer that owns the path are different
        layers. Metal: the hosts view now renders every carved host with its
        partition ("metal   : L4 l4-solo[0] @ 0.30", @0.20, @0.15, @0.12,
        @0.45), with their boots, tenants and runs. The four pre-#52 names
        ("l4-solo:0/main-tp1" and friends) are still in the volume's fleet log
        and are unreachable FOREVER — nothing can recover a journal written
        one directory deeper than list_hosts() looks; leg 6 prints them as
        such rather than pretending they can be found.
    (c) THE FACTORY IS PAID THE PARTITION IT REALIZES. Both factory
        signatures are now (Regime, Partition) -> Engine/Learner: the
        Partition IS the birth fact the factory builds (#43), and the
        fraction the carve just computed is the whole point of a sub-GPU
        host. Fleet.carve() builds the Partition first and hands the same
        object to the factories and to the Host. The ripple was exactly the
        two lambdas in tests, and it DELETED deploy/partition_l4.py's
        Partitioned.absorb() — the workaround that re-read every carve step's
        memory off the plan before apply().
    (d) VllmEngine HAS A SLEEP SEAM, AND IT IS A BUILD FACT. VllmEngine takes
        enable_sleep_mode (plumbed into the engine args, remembered as
        .sleeps, checked by check_sleeps()) and offers public async sleep() /
        wake() wrapping vLLM's sleep(1) / wake_up() — idempotent, and quiet
        before the lazy build, since an evict can arrive before the first
        sample. It is deliberately NOT on the Engine protocol (FakeEngine and
        RemotePool have no device to hand back): alternation is a capability
        of THIS build, reached by the deploy that owns the metal. The #51d
        repro poked engine._engine_args and engine._llm; the deploy's hooks
        are now measurement and nothing else. Engine fact, first contact
        settled: on vllm 0.28.0 AsyncLLMEngine.sleep/wake_up ARE coroutines,
        so the campaign's isawaitable probe is gone. Metal (leg 2): 5
        switches, sleep returned 9.92 GiB and wake put back 9.81, the learner
        offload 1.78, the device at 0.74 GiB between regimes, gap max 0.0259.
    THE VOLUME NOTE (harness, not the model): the store's fleet log is
    CUMULATIVE, so "every carve names a distinct host" read against the whole
    log fails on the pre-#52 history forever. The campaign's checks now slice
    from where the log stood when the container opened, and leg 6 separates
    the legacy "/" names from the ones a fixed fleet writes.
    NOT MINE, DELIBERATELY: (e) a training partition's fraction is
    unenforceable in-process — a stated cost of sharing one process, not a
    bug to fix; and (f) two hosts of one capability are unaddressable —
    find_join takes the first covering host, so spreading tenants across
    identical partitions needs a currency beyond capability (declared_load, a
    measured saturation signal). (f) is a design decision and goes to
    Samarth; it is why partition_l4.py still gives its second inference
    partition a different base.

53. THE EVAL'S TAIL AND THE CHORUS'S LAST RANK (settled by execution; both
    observations are the MATH shakeout's, made while Phase A was being
    priced). Two defects that only a long run on real metal could surface —
    one that spends money, one that loses the report of how the money was
    spent — fixed in the two files that own them. No spec shape moved:
    I1-I12 stand, and the run directory's bytes are unchanged (proven, see
    below). 498 tests green on fakes (56 skips), test_resume run explicitly.
    - THE EVAL WAS N SERIAL ROUND-TRIPS. `Evaluator._evaluate` gathered over
      n_samples INSIDE a `for task in self.tasks` loop, so a held-out set of
      32 tasks at n_samples=1 was 32 sequential episodes. At MATH lengths
      that is ten to sixteen minutes of billed tail per eval, and the LAST
      eval of a run has no training left to hide inside. `sample_heldout`
      now launches every (task, sample) episode at once under the daemon's
      existing max_inflight semaphore — the shape `collect_wave` has always
      had for generation — and the engine batches whatever arrives.
    - THE CONSTRAINT THAT MADE IT CAREFUL: eval/ IS INSIDE THE BYTE
      CONTRACT (test_resume: resume-equivalence AND two-straight-runs are
      byte-identical run dirs, evals included). The argument now stands in
      two named methods. `sample_heldout`: each episode's seed is derived
      from (task.id, sample_index) BEFORE anything is scheduled, so which
      episodes ran together cannot change what any of them sampled, and
      `gather` returns in ARGUMENT order, so the wave is a function of
      self.tasks alone. `reduce_in_task_order`: every accumulation runs over
      the wave, never over completions — float addition is not associative,
      so a mean summed in completion order would be a mean that depends on
      the scheduler. Proven twice: the whole run dir is byte-identical
      before and after the change, and the new CompletionOrderTest runs the
      same experiment at max_inflight 1, 64 and 3 against an engine that
      deliberately finishes episodes out of launch order, asserting one
      snapshot. It bites: reducing in completion order (as_completed) fails
      it, and fails the old determinism test with it.
    - THE CHORUS COULD LOSE THE CONTAINER. After a COMPLETED run a follower
      rank was still blocked in `hear` (an NCCL broadcast); `RankGroup.stop`
      sent terminate() and never confirmed death; the children are daemonic,
      so multiprocessing's exit handler joined them with NO timeout and the
      interpreter's exit hung; Modal killed the container at its 30-second
      grace and the run's whole printed report died unflushed with it.
    - THE LADDER, AND WHY THE POLITE RUNG CANNOT BE TRUSTED. A rank inside a
      collective is down in the driver, running no bytecode, so SIGTERM is a
      request nobody is at the desk to receive. `escalate` therefore climbs
      left-on-its-own → SIGTERM → SIGKILL, JOINING after each rung (a
      teardown that signals and walks away is exactly what left the wedged
      child), on ONE shared deadline per rung rather than per child, and
      returns a `Teardown` record — deaf / wedged / lost, plus whether the
      farewell went out — printed as one line whenever it is not graceful. A
      wedged collective at shutdown is worth a line, not silence. The
      process group is destroyed only on a graceful ending: after a kill
      there is no group left to agree with.
    - THE FAREWELL IS BOUNDED TOO, which is the half that makes "stop cannot
      wedge" true. `announce(STOP)` is itself a collective: against a dead
      or mis-sequenced follower it never matches and blocks until the
      group's own 1800s timeout, so the farewell that exists to END the
      chorus would be the thing that hangs rank 0. It now goes out on a
      daemon thread against the graceful deadline; not returning (or
      raising) means the chorus is past hearing. stop() is bounded by
      grace_s + 2 x signal_grace_s = 20s by default, inside Modal's 30.
    - THE TEST STANDS THE WEDGE UP WITHOUT METAL (tests/test_ranks.py,
      torch-gated, ~2.7s): `escalate` speaks only the process API, so a
      spawned stub that installs SIG_IGN on SIGTERM is a rank wedged in a
      collective exactly where it matters — the signal lands, nothing
      happens, SIGKILL finishes it. Removing the kill rung fails the suite.
    NOT DONE (deliberate, and out of this session's file scope): the eval
    POST half is now the unbounded one — `run_pipeline` gathers over every
    group with no limiter, which for a 64-task eval against a 32B teacher is
    64 concurrent scoring fans (runner/post.py; the async scorer daemon
    thread is where this belongs). `FsdpTorchLearner.stop` still returns
    None and drops the Teardown record, so on metal the printed line is the
    report. And deploy/math_opd_l4.py::full's docstring still prices the
    eval tail as sequential — stale as of this entry.

54. ARCHITECTURE.md + THE DOCSTRING RECODE (settled, directed: a universal
    vocabulary reference, then every docstring in the repo made to speak it).
    Docs only — `git diff` on .py is docstrings and comments, verified
    mechanically (AST equality against HEAD with docstrings stripped, exact
    for all 111 changed files). 503 green before and after, test_resume run
    explicitly.
    - THE DOC IS ARCHITECTURE.md AT THE REPO ROOT, and its charter is narrow:
      it DEFINES existing vocabulary and never redefines semantics. Canon is
      still rl-stack-spec.md plus the latest entry here; where the doc and
      canon disagree, canon wins. It is pedagogical, not chronological —
      history stays in THIS file, which is why the doc has no dates and no
      campaign results. Shape: the system in two paragraphs; ~73 NOUNS, each
      1-3 sentences plus the file it lives in, grouped contract / bridge /
      data objects / training world / metal / store / runtime; VERBS grouped
      by contract (the rollout lowering's demands-reaches-attach-apply-align
      plus the `claims` declaration, the kind's params/install_replay/
      uninstall_replay/emit/load/parity/rollout_lowering, the fleet's
      join-carve-acquire-place-apply, the host's submit-attach-admit-sleep,
      pool traffic's sample/score/collect and the admission-free ask verbs,
      the store's verbs); THE TWO PLANES; and an I1-I12 index that POINTS at
      the spec rather than restating it.
    - THE TWO PLANES is the one piece of framing the doc adds, and it is a
      distinction the code already makes: pool traffic rides the wire
      (availability signals, admitted at the serving host, nothing durable),
      the store plane rides the volume (the ledger append is the only
      durability bit). The consequence stated for future readers: an
      availability signal must never be mistaken for a commit — a bundle
      registered on an engine is availability, the ledger line naming it is
      the commit.
    - THE RECODE'S RULE: a docstring says what the thing IS in the doc's
      vocabulary plus the rule it enforces; module docstrings are 1-4
      sentences (a map or a verb table may be longer when the module IS a
      map); AT MOST ONE load-bearing citation survives, and only where the
      rule genuinely comes from it. A docstring that used to DEFINE "pool" or
      "lowering" inline now just USES the word. Honesty was compressed, never
      softened — side_attention's refusal and its true probe reason, attn_bias
      unreachable and refused at Phase 0, the parity certificate designed and
      unwired, additive_mask's mask cost, best-effort CUDA determinism, and
      partition_l4's two standing findings all survive.
    - STALE CLAIMS THE PASS CAUGHT (docstrings had drifted from the code in
      ways worth recording): losses/base.py still said a loss's `requires` may
      name "planned passes" — retired in #38, and this file is the one that
      enforces the rule; interfaces.py's Learner still said "swap-install"
      (#44(f)'s named debt, now paid); inference/rollout.py said the runner
      seals "after rewards", but run_episode seals immediately and rewards are
      postdata computed after the seal; stores/base.py's key tree omitted
      fleet/log.jsonl; validate.py said ONE check lives outside CHECKS when
      three do (the three that consult live metal, now named); arbiter.py said
      "one arbiter per GpuSet", which sub-GPU hosts falsified — a Host
      constructs its own, so an arbiter governs its owner's PARTITION and
      several coexist on one device (ARCHITECTURE.md carries the corrected
      wording too). Also swept: SPEC.md citations (that file does not exist),
      Phase A/B1/B2/B3/C chronology, leases, "role" for daemon, plan_roles for
      plan_daemons, the "five-member" adapter protocol (rollout_lowering made
      it six), and a `--fake` flag that never existed.
    - ADAPTER -> KIND WAS DELIBERATELY DEFERRED. The registered class is named
      `Adapter` and its registry `ADAPTERS`, but what they register is a KIND
      (`AdapterSpec.kind` names it by string); a configured bank entry is the
      adapter. The doc states the debt honestly and uses the words correctly
      in prose; NO identifier moved, because the rename is its own pass and
      would touch `Registry("adapter")`, whose string reaches user-facing
      KeyErrors and — through code_hashes keys ("adapter:lora") — every
      run_id. Rename candidates found while reading, for that pass: `llm` as
      the PoolClient parameter name everywhere (environments and every post
      processor); `Partition.gpuset` holding a Metal NAME rather than a
      GpuSet, a real false friend; `kind` overloaded three ways (adapter kind,
      Regime/Demand capability, GPU kind) with `FlowNode.kind` a fourth;
      `TokenBatch.post` for postdata; VllmEngine's `max_loras`/`max_lora_rank`
      surviving on a mechanism-blind bus. Two stale strings live in
      NotImplementedError/raise messages rather than docstrings and so were
      out of scope: batch_view.py's and side_attention.py's "B3", loop.py's
      two "B1" messages, and base.py's uninstall_replay message still saying
      "each tenant needs exclusive install" (install is additive; the true
      reason is the tenant could never be REMOVED).

55. THE VOCABULARY RENAME PASS: "KIND" IS DEAD, THE WORD IS ADAPTER TYPE
    (settled, Samarth-directed; #54 deferred this and listed the map, this
    entry executes it). Mechanical and behavior-preserving — no logic, no
    control flow, no signature SHAPE changed; only names. 503 tests green
    before the first commit, 504 after the last (the one addition is the
    journal-tolerance regression below). test_resume.py run explicitly.
    - THE RULING: the registered class is an ADAPTER TYPE; a configured bank
      entry is an adapter. ONE word, ONE spelling — `adapter_type` in code,
      "adapter type" in prose, never `kind`, never bare `type`.
        Adapter -> AdapterType · AdapterDef -> AdapterTypeDef ·
        @adapter -> @adapter_type · ADAPTERS -> ADAPTER_TYPES ·
        Registry("adapter") -> Registry("adapter_type") ·
        AdapterSpec.kind / RolloutLowering.kind -> .adapter_type ·
        Bundle.kinds and compile_bundle(kinds=) -> .adapter_types ·
        group_by_kind -> group_by_adapter_type ·
        _Tenant.kinds -> .adapter_types ·
        VllmEngine._serving_kinds -> _serving_adapter_types ·
        check_kinds_accept_their_sites -> check_adapter_types_accept_their_sites
    - THE IDENTITY MOVE IS ACCEPTED, AND STATED. Registered strings and
      registered-class sources hash into run identity (I3), so this pass MOVES
      EVERY run_id twice over: code_hashes keys go "adapter:lora" ->
      "adapter_type:lora", and AdapterSpec's field rename changes the canonical
      JSON. CONSEQUENCE, deliberately taken: pre-rename stores are READ-ONLY
      HISTORY, fresh runs get new run_ids, and resume compatibility across the
      rename is NOT promised. Identity stays internally consistent WITHIN a
      version, which is what test_resume protects and what still passes. The
      three tests that pin identity (test_registry's code_hashes keys,
      test_canonical's two goldens) were updated to the new strings — that IS
      the accepted move, not a regression.
    - THE FREED WORD IS `capability`. With "kind" no longer owed to adapters,
      Regime.kind and Demand.kind take the word ARCHITECTURE.md already uses
      for them ("one capability a host can wear"). FlowNode.kind (a node
      category), BackendProfile.kind, Registry.kind (the registry's own
      category) and the Metal/Partition GPU kind KEEP the word: different
      meanings, same spelling, and none of them an adapter type.
    - THE OTHER FALSE FRIENDS, same pass. Partition.gpuset -> .metal (it holds
      a registered Metal's NAME; GpuSet means pure device demand in a spec) ·
      TokenBatch.post -> .postdata (postdata is the columns; `post` is the
      processor LIST on an AlgoSpec, and one word for both made
      batch.post["advantage"] read as a pipeline) · the PoolClient parameter
      is `client`, never `llm`, in every environment, every postprocessor and
      every test (it routes to pools, and a pool may be a judge, a teacher on
      another base, or a scorer) · VllmEngine(max_loras=, max_lora_rank=) ->
      (max_bundles=, max_rank=), which is what ServingBuild already called
      them; lora_vllm still feeds vLLM's own max_loras/max_lora_rank engine
      args unchanged, so the last mechanism-flavored words in vllm_engine.py
      are gone and #48's noted exception is closed.
      NOT renamed, because the word is right there: VllmEngine._llm (vLLM's
      own engine handle, not a PoolClient) and the llm_judge processor's
      registered name and file — a judge that IS an LLM.
    - THE OBSERVER STAYS TOLERANT OF OLD JOURNALS. A journal is append-only
      history, so hosts booted before this pass say "gpuset" and regime "kind"
      on the volume forever. views.py grows partition_metal() and page.py the
      JS twins partMetal() / regimeCapability(), each reading either spelling.
      test_ui.py keeps its l4-b fixture in the OLD spelling ON PURPOSE — that
      fixture plus test_a_pre_rename_journal_still_names_its_metal is the
      regression pinning the tolerance, and it is the reason the suite grew by
      one. Nothing else reads a journal: correctness never does (#43).
    - THE STALE STRINGS #54 FOUND, now stating the rule they enforce rather
      than a build chronology retired with Phase A/B/C: loop.py's two "B1"
      messages (no algo means no Trainer, and the Trainer is the ledger's only
      writer, so nothing would ever commit; warm start needs a store://
      address because it reads another run's SEALED deltas), batch_view.py's
      and side_attention.py's "B3" (the shim is unbuilt because no plugin's
      attend() exercises it; the merge is unbuilt because vllm 0.28.0 does not
      plumb return_softmax_lse through the dense FlashAttention path — the
      module docstring's own reason, now in the raise), and base.py's
      uninstall_replay (install is ADDITIVE, so the true consequence of a
      missing inverse is that a tenant could never be REMOVED from a shared
      learner). NB: #54 filed batch_view.py under rlstack/data/; it lives in
      rlstack_engine/, the only place it could — it is the version-pinned shim
      that ships in the engine image.
    - DOCS IN THE SAME PASS. ARCHITECTURE.md's **Kind** entry became **Adapter
      type**, its naming-debt blockquote is DELETED (the debt is paid), and the
      verbs section is now "The adapter type"; the Partition, Regime and
      TokenBatch entries name their renamed fields. STYLE rule 8's adapters/
      line says "one file per ADAPTER TYPE PER SIDE". rl-stack-spec.md got the
      adapter-type terminology swap ONLY (@adapter_type, AdapterSpec.
      adapter_type, CertificateKey) — zero semantic edits, per the directive.
      CARRIED AS DELTAS, not folded: the spec canon still spells
      Partition.gpuset, Demand.kind and the PoolClient parameter `llm` at
      lines 292/305/306/362/376/515/516/568/571. They are terminology-only and
      are superseded by THIS entry until someone folds v4.
    - JUDGMENT CALLS, listed so they are reviewable rather than silent. The
      "unknown-adapter" validation issue code and flow.py's "adapter:<type>"
      dictionary.json producer label KEEP their spelling: both describe a bank
      ENTRY (which is genuinely an adapter), neither is in the rename map, and
      the producer label is derived, never identity (I11). FakeLearner's
      forward_backward hashes a dict whose key is still "post": it is a label
      inside content_hash, never read back, and moving it would shift fake-run
      bytes for no vocabulary gain. Regime's journal row key moved to
      "capability" alongside Partition's "metal" — #55's directive named the
      tolerance only for gpuset, but a row IS a serialization of the fields,
      and leaving the wire on the old word would have re-created the drift
      this pass exists to kill; both readers tolerate both.

56. THE EMISSION PLANE: HOSTS MEASURE THEMSELVES (settled by execution,
    directed: build the metrics data the observer UI renders, against a
    schema fixed in advance because a sibling agent was overhauling the UI
    against the same two events). #50 designed the schema-tolerant metrics
    slot and deliberately emitted nothing into it; this entry is the
    emission. 523 green (20 new in tests/test_emission.py).
    - TWO JOURNAL EVENTS, both on hosts/<name>/log.jsonl and nowhere else.
      `traffic` — one per existing stats tick, windowed since the previous
      tick: {event, t, window_s, prefill_tokens, decode_tokens, requests,
      ttft_ms_mean, admit_wait_ms_mean, admit_wait_ms_max, inflight}.
      `update` — one per COMMITTED update, from the Trainer: {event, t,
      run_id, update, seconds, phases:{collect, post, train, seal}}.
      (#50 sketched the first as "throughput"; the built name is `traffic`.)
    - THE RULE IS THE WINDOW, NOT THE REQUEST: counters accumulate at the
      seams and the host's EXISTING stats loop drains them — one row per
      tick, never one per request (journal bloat on the volume is the thing
      being measured, defeating the point), and never a second timer (two
      cadences on one host cannot be read against each other). A tick that
      served nothing emits zeros with null means: a hole in the series is
      indistinguishable from a dead host, and a fabricated 0ms is a lie.
    - TIMESTAMPS NEVER ENTER A RUN DIRECTORY. Every number here is wall
      clock, and a run dir is a pure function of (spec, code, data) — so
      durations live in the host journal, which is observability, outside
      identity, never read by correctness. The ledger line carries an
      update's FACTS, the `update` event carries its DURATION. Guarded by a
      test that snapshots the whole hosted run directory against a raw
      run_experiment's: byte-identical, resume-equivalence untouched.
    - THE SEAMS, all ours, none of them inside vLLM or torch (whose own
      statistics are version-coupled to the pinned build): VllmEngine's own
      generate loop (prefill = len(prompt_ids) before the request leaves,
      TTFT = the gap to the first output, decode counted per streamed
      chunk — zero per-token allocation; score_tokens is one prefill of
      known length and counts as a request); the GpuArbiter's admission
      door (this request's own queue time, not the resident's oldest, plus
      the in-flight gauge — the learner's gradient admissions included,
      because the wait is a property of the door); the Trainer's four
      existing phase boundaries, lapped so every second between the first
      await and the commit lands in exactly one phase.
    - ONE METER PER HOST (rlstack/runner/meters.py: TrafficMeter +
      TrafficWindow.row(), UpdateClock, HostJournal). The Host owns it and
      wire_meter() assigns it into every engine it owns and its arbiter, so
      a `traffic` event describes the PARTITION — a shared engine's load is
      a property of the metal, while a tenant's share is a run fact the
      ledger already carries. `meter` is declared on the Engine protocol,
      not duck-typed on. Plain int/float adds on the one event loop: no
      lock. Wiring is by assignment because deploys build engines first and
      hand them to the host afterwards.
    - THE WIRE NEEDS NOTHING (verified, not assumed): a RemotePool verb
      lands in HostService, which admits it through the SERVING host's
      arbiter and runs it on that host's engine — both already wired to
      that host's meter. RemotePool carries a meter that stays at zero, and
      says so: a client-side count would attribute another partition's load
      to this one. Tested with two hosts and a LocalTransport.
    - THE OBSERVER'S SIX CHANNELS (observe/host_series.py, which imports
      nothing from the runner — the journal row IS the contract between
      them): metric_series now serves traffic_channels first and then the
      open slot. Names are fixed: prefill_tok_s, decode_tok_s, requests_s
      (window counts over window_s), ttft_ms, admit_wait_ms (the window's
      means), inflight (the gauge). All six appear together once a host has
      ever journaled a window, so a polling page never watches cards appear
      and vanish; a latency channel stays EMPTY while nothing measured one.
      run_timing(store, run_id) scans every host's journal for that run's
      `update` rows and returns them RAW, sorted by update — steps/s and
      per-step bars are the page's arithmetic, not the observer's.
    - DELIBERATELY SKIPPED: vLLM-internal KV-cache occupancy and scheduler
      queue depth. They are the two numbers a seam cannot see, and both are
      version-coupled to the pinned 0.28.0 engine — revisit only if the
      seam-level numbers prove insufficient in practice. Also not built:
      admit_wait_ms_max rides in the journal but is not one of the six
      channels (the contract named six); the fleet page shows no traffic;
      and no deploy entry point was touched, so a serving-only host
      (deploy/modal_host.py) counts but never drains until someone starts
      its run_stats loop.

57. THE OBSERVER'S SECOND OVERHAUL: FILES, WAVES, AND THE TWO AGGREGATES
    (settled, Samarth-directed: new charts, a redrawn GPU panel — "very ugly
    rn" — and a wave/trajectory browser with a chat-style reader; explicitly
    NO React, NO build step, NO CDN, NO node toolchain). Entirely inside
    observe/ plus two peek verbs on the Store, one mount line in
    deploy/modal_app.py and the rule-8 extension. Built against #56's emission
    contract, whose plane is a sibling worktree: nothing here waits on it.
    - THE PAGE IS FILES NOW. page.py was 645 lines of HTML+CSS+JS inside a
      Python string; it is now the READER of rlstack/observe/web/ —
      index.html, style.css and nine native ES modules (app · nav · dom ·
      charts · runs · run · fleet · host · wave) the browser loads itself.
      THE RULE, stated so a later session does not undo it: no build step, no
      bundler, no CDN, no framework — a module is added by writing a file and
      importing it. observe/web/ is the package's ONE folder of non-.py files
      (STYLE.md rule 8 gains that line; test_architecture.py enforces both
      halves: the document ships, and nothing non-.py lives elsewhere). ui.py
      serves /web/<file> with its content type, an asset is ONE file name
      (web/ is flat, so no path is ever walked), and every page route still
      returns the same document. The JSON API stays the contract between
      Python and the page.
    - THE CHARTER AMENDMENT. The observer's rule was "journals + peeks, never
      experiment content". It now also reads SEALED, content-addressed
      artifacts — and nothing else. Sealed IS the whole of the licence: an
      update the ledger committed is immutable (the store refuses to
      overwrite it), so reading it can neither race a writer nor perturb a
      run. Never live state, never an attach, never a write. Mechanically
      this needed peek_wave / peek_postdata beside peek_ledger, because
      read_wave lives on RunHandle and open_run DISCARDS unsealed work — an
      observer that "just opened the run" would delete a live tenant's staged
      wave. wave_key/postdata_key became module functions so the handle and
      the peek name the same key. observe/waves.py is the reading.
    - THE WAVE BROWSER. The run page lists the ledger's tail (20) and each
      wave is a page: /run/<id>/wave/<n>, fetched lazily, never preloaded,
      and never polled (sealed bytes do not change). It carries the
      distributions computed server-side — reward and every other scalar
      column, generation length, the finish mix over turns (stop vs length is
      the truncation that quietly costs reward), and for OPD-style runs the
      SAMPLED KL the bytes already contain: behavior logprobs minus the
      teacher_logprobs column over the same tokens, in flatten order. Below
      them the trajectories, grouped as they were trained (the Group is the
      scope a postprocessor saw), each expandable into a chat reader:
      role-labelled bubbles, generated ones marked (a Turn is one request —
      one pinned bundle, one seed, one contiguous KV) with tokens, finish,
      seed and mean behavior logprob, and a fact row of the postdata columns,
      bundle_id and policy_version. Record shapes come from
      data/trajectory.py; message identity is matched by id(), as flatten and
      the teacher scorer match it.
    - THE TWO AGGREGATES (observe/aggregate.py, the reading true of neither
      one host nor one run): inference partitions summed as tokens/s over
      every serving host, training partitions summed as updates/s over every
      run. THE BUCKET RULE: per bucket, each host's MEAN of its own per-window
      rates, summed across hosts — a host that sampled twice does not count
      twice. And a bucket is never narrower than the fact it summarizes: the
      inference width is floored by the widest declared window_s, the
      training width by the pooled median gap between updates (N runs
      stepping every T land one update every T/N, exactly the width at which
      the sum reads N/T). Found by driving it: one shared width made the
      fleet line SAW between one host and two, and read 0.29 updates/s where
      the truth was 0.0071.
    - STEP ECONOMICS, FIRST-CLASS. The run page carries steps/s and a
      per-update duration bar DECOMPOSED into collect · post · train · seal,
      hover naming each phase's seconds. This is the scorer-economics
      instrument: inline teacher scoring shows up as `post` eating the bar.
      Beside it, PARITY: logprob_gap is promoted out of the rails into its
      own section with a zero reference line, because it is the standing
      alarm (#25's certificate, running).
    - THE GPU REDRAW. One card per device instead of two charts of all of
      them: a utilization rail over an HBM rail, the device total drawn as a
      labelled reference line, the footer reading "now 82% · 15.9 GiB of
      22.5 GiB". Every non-sampled journal event is a MOMENT marked on both
      rails — boot, attach, detach, and sleep/wake the day the arbiter
      journals them (the reading is schema-tolerant: any event kind that is
      not stats/traffic/update becomes a marker) — because a memory curve
      that falls off a cliff is only explained by the moment beside it. The
      host page also grows the six traffic rails (tokens/s, requests/s,
      in-flight, ttft and admission wait), and the open metrics slot now
      plots only what no reading above claims.
    - HOVER SURVIVED THE REWRITE, including for DERIVED points: an aggregate
      point's tooltip carries the journaled counts it was summed from
      ("1093 prefill tokens · 2 host(s) · 3 sample(s)"). A poll still never
      redraws under the cursor, and a scroll now clears the tooltip with it.
    - MERGE SEAMS for #56, both at the top of observe/aggregate.py and marked:
      traffic_channels (the six named channels, computed here from raw
      traffic events) is replaced by host_series.metric_series's named
      channels, and run_timing by host_series.run_timing — one line each.
    - DEPLOY, VERIFIED: add_local_python_source ships .py files ONLY
      (ignore=NON_PYTHON_FILES — checked against the installed modal 1.5.4:
      index.html/app.js/style.css all return True, ui.py False), so
      deploy/modal_app.py gains ONE add_local_dir of observe/web/ at an
      absolute path off __file__. No other deploy file serves the UI
      (partition_l4.py::observe uses the terminal renderers). The local
      venue reads the same files through importlib.resources, verified by
      serving `python -m rlstack ui` over a demo store and driving every page.
    526 tests green (22 new in tests/test_observer.py plus the rule-8 test:
    the assets and the module graph, the wave listing and detail, wave-order
    alignment, the distributions and the sampled KL, the four routes and
    their 404s, "reading a wave never sweeps unsealed work", the six channels,
    run_timing's phases, moments, and the two aggregates incl. the bucket
    floor).
    DELIBERATELY DEFERRED (still the "named next" of #50): token-level
    drill-down, cross-run curve comparison, and distributions anywhere but
    the wave page (a run-level histogram over waves wants the scorer daemon's
    postdata cadence first). Also unbuilt: paging a wave beyond the first 256
    trajectories, and prev/next navigation between waves.

58. THE ORGANIZATION PLANE: FOLDERS, ANNOTATIONS, AND A SUBSTRING (settled,
    Samarth-directed and specified in his own words: "starting a run in a
    folder means nothing more than passing that root when constructing the
    store"; annotations are "flavortext"; search is "basic substring, nothing
    fancier"). Entirely inside data/stores/base.py, observe/ and the CLI. The
    RUNNER IS UNTOUCHED — nothing there needed changing, which is the whole
    point of the design.
    - A FOLDER IS A STORE ROOT, CHOSEN AT BIRTH, NEVER MOVED. A run's
      organizational location IS the directory its store was constructed with
      (<top>/opd/math/l3-5 as the root; the run lives at <root>/runs/<id>).
      There is no rename verb and no move verb, and there never will be: a
      move would mean rewriting a store's key tree under a live tenant, and
      the thing gained — a different string in a UI — is not worth a byte of
      that risk. Filing happens when you type the store path, once.
    - DISCOVERY, ONE RULE, ONE FILE (observe/locate.py, which now owns
      locators AND roots): the reader takes a TOP directory; a directory
      holding runs/, hosts/, fleet/ or annotations.jsonl IS a root; the
      descent STOPS at each one (everything below a root is that store's key
      tree, not more folders); a top that is itself a root — or holds none —
      is the DEGENERATE case, one Root whose folder is "". That degenerate
      case is the deployed observer's /store mount, and it is byte-for-byte
      the old behavior: bare links, no folder tree, no query string.
      deploy/modal_app.py is unchanged, because ui_app still accepts bare
      Stores and reads them as Root("", store). With several tops, each top's
      own name prefixes its folders and two tops of the same name are
      REFUSED — a folder is an address, and one address must name one root.
    - ANNOTATIONS ARE FLAVORTEXT, AND THE FILE SAYS SO BY WHERE IT LIVES.
      <store root>/annotations.jsonl, BESIDE runs/ and never inside it:
      append-only, one row per line, {"t", "run_id", + only the fields
      passed} out of name / tags / note. Reading merges LATEST-WINS PER
      FIELD (a later row's "tags" replaces the whole list; a row carrying
      only a note leaves an earlier name standing). Never hashed, never read
      by an experiment, never written by the runner. THE PROOF is in
      tests/test_resume.py, where it belongs: annotate an arith run and its
      run directory's sha256 map is unchanged — and unchanged again after a
      re-attach, because crash recovery sweeps inside runs/ and the file is
      not there. Three verbs on the Store ABC (annotations_key /
      annotate_run / read_annotations) over the existing byte verbs, so
      ModalVolumeStore gets them free (plus one line: an annotation commits
      the volume, like the journals).
    - ADDRESSING, THE ONE AMBIGUITY, RULED ON: identity is content, so the
      same spec submitted under two folders yields THE SAME run_id in both
      (the demo store reproduced this on the first try). A run is therefore
      addressed by (folder, run_id): every link the index emits carries
      ?root=<folder>, and a bare /run/<id> resolves against all roots — the
      unique holder wins, and when several hold it the API answers
      {"run_id", "ambiguous": [folders]} at 200 and the page LISTS THE
      FOLDERS AS LINKS. It never silently picks one. (200 and not a 4xx
      because "this id names two runs, here they are" is a successful
      answer to the question asked — the disambiguation page, not an error.)
    - THE OBSERVER STAYS READ-ONLY. No POST route, no write from the page;
      annotations are written by `python -m rlstack tag <store-root> <run_id>
      [--name] [--tag ...] [--note]` and merely rendered here — a test greps
      observe/ for `annotate_run` to keep it that way. The index groups runs
      under a collapsible folder tree, leads with the NAME (hex id demoted to
      secondary, on the run page and in the switcher too), draws tag chips,
      and filters on a search box the poll cannot clear (the box is built
      once; only the tree is redrawn). The `runs` view gained the same two
      columns, folder headers and --grep. Fleet and host pages aggregate
      across every root, and a host name is qualified by its folder when two
      roots journal it.
    - THE SEARCH IS A SUBSTRING AND NOTHING FANCIER: case-insensitive, over
      name + tags + note + run_id, stated in views.matches and mirrored by
      runs.js (the page filters rows it already has, so typing costs no
      request). No query language, no ranking, no index.
    583 tests green (35 new in tests/test_organization.py, plus the
    resume-equivalence proof): the verbs and their merge, discovery over
    nested roots / the degenerate case / no descent into a root / the
    multi-top prefix and its refusal, the same id in two folders as two rows,
    annotations riding with their own root, every run route answering the
    ambiguity, ?root= resolving and 404ing, and the CLI tag round-trip. The
    UI was driven in a browser over a two-folder demo store (tree, search,
    chips, ambiguity page, folder-qualified hosts) and over one root alone to
    prove the degenerate case renders bare links exactly as before.
    DELIBERATELY LEFT OUT: the runs index is still JOURNAL-DRIVEN (a run that
    never attached to a host does not appear, and so cannot be seen even if
    annotated) — changing that changes what "the runs view" has always meant
    and belongs to its own decision; annotations are per-root, so an
    annotation does not follow a run_id into another folder (correct: they
    are different experiments); and there is no tag autocomplete, no tag
    index, and no way to remove one tag but the whole list.

60. THE TASK-SET PATH: WHERE A DATASET BECOMES CONTENT (settled by execution;
    DAPO-Math-17k is on the rlstack-store volume). #59 made a plan name tasks
    by ID and a task set pure content; this is the other half — how content
    gets made. rlstack/data/tasks/, two files, and a CLI verb over them.
    - THE VERBS ARE DATASET-BLIND, THE DATASET FILES ARE ONE FUNCTION EACH.
      base.py: write_tasks (canonical jsonl -> cas_put, ids unique WITHIN the
      set refused where the set is made), load_tasks (MOVED here from
      runner/traffic.py — it is write_tasks' inverse and the pair belongs
      together; load_task_sets stayed in the runner, because its rule is about
      a plan's leaves across DECLARED sets), split_tasks. dapo_math.py: one
      function, pyarrow/huggingface_hub/transformers imported INSIDE it (rule
      7), so the package root stays free to import.
    - THE SPLIT IS A PER-TASK DRAW, NOT A SHUFFLE. A task's split is
      h(seed, its id) placed in the fractions' half-open intervals — a
      function of the task ALONE, not of its position or of the set it
      arrived in. The property bought: re-splitting a superset leaves every
      earlier task where it was, so held-out stays held out when the source
      grows. The price, stated in the docstring: counts are DRAWN, not dealt
      (17,917 at 2% gave 370, not 358). Fractions must sum to 1 — a task
      belongs to exactly one split, so the splits must cover the set.
    - THE SCHEMA AS IT ACTUALLY IS (inspected on Modal, not remembered):
      data_source · prompt (list<struct<content, role>>) · ability ·
      reward_model (struct<ground_truth: string, style>) · extra_info
      (struct<index: string>). THE FILE IS NOT 17k ROWS: 1,791,700 rows over
      17,917 distinct extra_info.index values, each repeated EXACTLY 100
      times — verl's rollout fan-out baked into the parquet. We keep one row
      per index (a task is a problem; how many samples it gets is the plan's
      business), which is sound because no index carries two different
      payloads. Left standing deliberately: 17,398 distinct prompt TEXTS, so
      ~500 problems appear under two uuids, and 7 texts carry two different
      ground truths — collapsing those needs a rule for which answer wins and
      there isn't an honest one. Every ground_truth is a plain integer
      (1,791,700 / 1,791,700), so the integer filter dropped ZERO rows; it
      stays as the stated rule, by the verifier's own `-?\d+` pattern.
    - THE PROMPT IS BUILT IN data/tasks/, NOT IN deploy/ AND NOT IN THE ENV,
      because it is the interface between task content and the environment
      and building it there is what pins it into the cas hash and so into
      identity (I3). DAPO's own instruction is kept VERBATIM — it already
      asks for a last line "Answer: $Answer", which is exactly what the
      last-integer `verifier` reads, and a second instruction of ours would
      only compete with it. The Qwen3-14B chat template is applied HERE at
      build time because VllmEngine's stated v0 choice is that a prompt is
      the raw token concatenation of its messages with no template applied.
    - THINKING MODE: OFF, DELIBERATELY. Qwen3's template thinks by default
      (no kwarg and enable_thinking=True render identically); we pin
      enable_thinking=False, which appends `<think>\n\n</think>\n\n`. The
      consequence is length: the model answers directly in the low hundreds
      of tokens instead of reasoning for thousands, so a wave fits a sane
      max_tokens and microbatch_tokens budget, and the closed think block
      costs PROMPT tokens (prefill) not completion tokens (decode). It is
      content, not a knob — a thinking-mode set is a different cas uri, and
      the two are compared by pinning one or the other.
    - WHAT LANDED, seed 17, train 0.98 / eval 0.02, on modal://rlstack-store:
      17,917 tasks, ids `dapo-math-17k/<uuid>`, meta {"answer": int} and
      nothing else.
        train  17547  cas://09499d32b51e5e1b2a644b1c65e01b44aa42ff1a5bfac78ead41f98f89f09c93
        eval     370  cas://82ae4626dbb59a2c50e2b13cbe7250c5f1ddd02dfb81edc7495efb77759d420b
      Read back off the volume: disjoint, 17,917 between them, prompts
      rendering as intended. A second build after switching the parquet read
      to streamed batches reproduced BOTH uris exactly — the content hash is
      the proof that the read path is not part of the content.
    - deploy/tasks_dapo.py IS 52 LINES AND HOLDS NO SEMANTICS (I5): the
      pinned layer (vllm 0.28.0 / torch 2.13.0 / transformers 5.16.1) is kept
      byte-identical with the campaign images so it stays cached, pyarrow
      rides in a layer of its own after it, and the body constructs a
      ModalVolumeStore and calls the CLI's own build_task_sets — the same
      function `python -m rlstack tasks dapo_math --store <root>` calls, so
      local and volume runs print the same table. A cas write is not a commit
      point, so the volume is committed explicitly once the sets are whole.
    - NOT DONE, deliberately: no spec pins these uris yet (that is the run's
      decision, not the task set's), no second dataset, and no eval `terminal`
      bit. The 7 ambiguous ground truths and the ~500 double-uuid problems are
      recorded above rather than repaired.

61. **The DAPO campaign meets the metal — and the activation ceiling is
    named.** The campaign is `deploy/dapo_grpo.py` (212 lines, ~98 of them
    science): three plans written out as literal nesting, the `dapo_math`
    environment, and `final_answer` as the reward. Two updates committed on
    a tp=2 policy host beside an fsdp=2 learner, run c-id `e6811ab59a4c`.
    - THE CEILING IS ONE DOCUMENT'S ACTIVATIONS, MEASURED NOT GUESSED. Four
      L4 attempts died in update 1's forward at 21.53 / 21.47 / 21.49 /
      21.49 GiB of 22.03 — the last two BYTE-IDENTICAL across a 4x
      microbatch cut, which looked like proof that memory was committed
      before the batch mattered. It was the opposite: a sealed wave says
      completions run 578..2048 tokens (median 1357, 28% at the cap) over a
      ~136-token prompt, so EVERY document exceeds both 2048 and 512,
      `pack` gives each its own oversized batch (it never splits one), and
      both runs ran the same first document through the same forward.
      Byte-identity was evidence FOR the activation story, not against it.
      One ~1500-token document through 40 layers stores ~250-350 MB of
      interiors per layer (the MLP is 17408 wide) = ~14 GiB, plus a 6.88 GiB
      shard and 0.3 GiB of bank and optimizer: 21.5 on a 22.03 GiB card.
      The ledger's `microbatches: 64` for 64 trajectories is the same fact
      from the other side — one document per forward, always.
    - THE FIX IS RECOMPUTE, and it belongs to TorchLearner, not to FSDP:
      `checkpoint_the_blocks` replaces each decoder block's forward IN PLACE
      (a checkpoint_wrapper would rename every submodule, and both the site
      schema and adapter installation address blocks by path),
      use_reentrant=False so kwargs work and so it composes with FSDP's
      backward re-gather, applied BEFORE the wrap. Under no_grad it passes
      through, so scoring stays cheap. Exact: the base is frozen and in
      eval, so recompute is the same arithmetic. UNPROVEN ON METAL.
    - THE SCIENCE IS BLOCKED WHERE THE PLUMBING IS NOT. logprob_gap 0.0145 /
      0.0167 is at the kernel floor, so the wire served exactly the adapters
      the trainer recomputed. But reward was 0.266 then 0.047 and ADVANTAGE
      WAS -0.0 IN BOTH WAVES: at group_size 8 on unfiltered DAPO the 14B
      scores all-or-nothing per group, the z-score vanishes, and the updates
      were arithmetically real and informationally empty. A pass-rate filter
      that keeps problems strictly between 0 and 1 is the standard answer and
      is NOT built.
    - TWO OPERATIONAL FINDINGS. The served host has no clean shutdown: vLLM
      raises `Event loop is closed`, leaks a shared-memory object, and the
      container exceeds Modal's 30s grace, so a finished run exits looking
      failed (results are safe — they commit before teardown). And
      ModalVolumeStore.commit() emits an AsyncUsageWarning on every write,
      which now dominates every campaign log and buried this run's own
      peak-memory print.


62. **The campaign ran: 50 updates of GRPO on DAPO-Math-17k, and the policy
    learned.** Run `803578405216` — Qwen3-14B, r=16 LoRA on attention sites,
    tp=2 policy host over the wire beside an fsdp=2 A100 learner, 8 groups x 8
    completions a wave, 4.94M tokens trained, final bundle
    `bundle:29b5f7b25f1a` at `pi@50`. ~$35 on the credits workspace.
    - TRAINING REWARD DOUBLED, block means over ten updates: 0.222, 0.358,
      0.467, 0.495, 0.464 — monotone through update 40, then flat.
    - HELD OUT IT GENERALIZED, measured after the fact over sealed
      checkpoints: 0.312 at u20, 0.438 at u30, 0.344 at u40, 0.531 at u50 on
      32 tasks the policy never trained on. The rise corroborates the training
      curve instead of contradicting it, which is the distinction that
      mattered; a 32-task set moves 3 points per problem, so u40 is noise.
    - THE PARITY RAIL HELD ALL 50: logprob_gap median 0.0181, max 0.0257,
      never off the kernel floor, so the remote pool served exactly the
      adapters the learner recomputed for every update. grad_norm median 1.9.
    - PACE: 267 s/update (collect 143, train 122, seal 9) — collect-bound with
      lag=1 overlapping the two, which is why the A100 learner was the right
      economics (an L4 learner was train-bound at ~800 s).
    - MEASUREMENT GAPS, stated: update 10's post-hoc eval never printed (absent
      from the logs, not an error — the path wants a look before it is trusted
      unattended), and there is NO version-0 baseline, because ::evaluate walks
      ledger entries and an untrained policy has none. The nearest reference is
      the training-wave reward of 0.266 at update 1 under the same sampling.
      Quote 0.531 only beside that caveat until a base measurement exists.
    - THE IN-RUN EVAL MEASURED NOTHING, and the bug was in the campaign, not
      the runner: the evaluator reads `plan.wave(update // every)` — ONE WAVE
      PER EVAL POINT — and the eval plan was built with one entry per update
      and empties between, so every point read an empty wave. Fixed
      (`RunPlan((measured,) * (updates // every))`), and `::evaluate` recovers
      the measurement from sealed checkpoints because a committed version is
      re-derivable with its content-addressed id as the proof.

63. **STORE RETENTION: a run may forget what nothing will ever read again —
    an abstract policy, and one default that keeps every adapter.** The
    measurement that forced it, from run `803578405216` (a 21M-param LoRA,
    50 updates):

        optim     50 files   8.0 GiB   160 MiB each (Adam: TWO fp32 moments)
        adapters  50 files   4.0 GiB    80 MiB each
        waves     50 files    42 MiB
        postdata  50 files      ~0

    Optimizer state is TWO THIRDS of the footprint, and exactly one blob of
    it has a reader. The same run against a 1B-param adapter is ~400 GB of
    moments and ~200 GB of deltas, which is the number this exists for.
    - THE CONTRACT IS A CLASS, NOT A CONFIG (Samarth's ruling, followed
      exactly): `RetentionPolicy(ABC)` in `data/stores/retention.py` with one
      abstract method, `expendable(ledger) -> (section, name, version)
      triples`. No `optim="tail"`, no `every:k` — the policy IS a function
      someone implements, mirroring Environment / PostProcessor /
      AdapterType / the rollout lowering: base class in its own file, the
      rule in the docstring, one implementation per file.
    - A POLICY IS A PURE FUNCTION OF THE LEDGER. It is handed the commit
      record and nothing else: no store, no spec, no clock, no arguments. So
      it is testable with no bytes on disk (the policy suite never opens a
      store), the same ledger always frees the same blobs, and a sweep that
      never ran costs only the sweep that follows it.
    - THE TRIPLE IS THE BLOB ADDRESS, and that is the safety property: a
      policy can name `adapters/<name>@<v>.bin` or `optim/<name>@<v>.bin`
      and NOTHING ELSE. The ledger, the manifest, a sealed wave and its
      postdata are not expressible from here, so retention cannot weaken the
      append-only guards by construction rather than by promise. Every
      deletion is addressed through `_blob_key`, which already refused a
      section outside BLOB_SECTIONS.
    - THE DEFAULT IS `KeepRestorable`, named for what it keeps. ITS RULE:
      every optim blob below the ledger tail is expendable; NO adapter blob
      ever is. The asymmetry is the store-side reading of what
      `runner/restore.py` already states — serving state is immutable and
      versioned, training state is mutable and exists only at the last
      commit. The reader census is complete and short: OPTIM has two readers
      (`restore_tenant`, only ever at the tail, because a learner is
      restorable only at a commit boundary; and `_warm_start` with
      `optim="load"`), while ADAPTERS have four that pin HISTORICAL versions
      — the evaluator's `bundle_for`, `restore_bundle_on` for an evicted
      pool or a restarted container, a `WarmStart` naming `@v`, and
      `dapo_grpo::evaluate` measuring a finished run from sealed
      checkpoints. Deleting an adapter breaks restore for that version
      forever; deleting stale optim cannot break anything.
    - THE SEVENTH BYTE VERB IS `_size`, not `_delete` — `_delete` was
      already one of the six. A sweep must report what it freed, and reading
      a 160 MiB blob to measure it would cost more than keeping it. Beside
      it, `_persist` is promoted from ModalVolumeStore to a no-op Store
      hook: on a mounted Volume A DELETION STAGES EXACTLY LIKE A WRITE, so a
      sweep that freed bytes commits once at the end (and one that freed
      nothing commits not at all). LocalStore's verbs are durable as they
      land, so its hook does nothing.
    - `RunHandle.sweep(policy) -> Swept` reads the ledger, asks the policy,
      CHECKS THE WHOLE BATCH, then deletes — so a policy that names live
      state frees nothing at all rather than half of what it asked for. One
      named guard, `_refuse_live_version`, refuses the tail's version
      whatever a policy says: the floor under every policy, enforced in the
      store and not in the policy, is that a swept run still resumes.
    - TWO ENTRY POINTS AND ONE RULE. (a) THE TRAINER, right after
      `append_ledger` — the moment a version went stale and the moment the
      design already serializes on. Never fatal: the update is committed,
      and a failed unlink leaves bytes on a volume, the cheapest failure in
      the system. What was freed is journaled NOWHERE — it is a fact about a
      directory, not the experiment, and a run directory holding it would
      stop being a pure function of (spec, code, data). (b) THE OPERATOR:
      `python -m rlstack sweep <store-root> <run-id>` prints what the policy
      names, frees it, and reports the bytes. It is the CLI's first verb
      that writes INSIDE a run directory and the only one that ATTACHES —
      so it is for a run that has stopped, never one that is training.
    - A THIRD MOMENT THE BRIEF DID NOT ASK FOR, AND WHY: the Trainer also
      sweeps WHEN IT STARTS. Resume-equivalence found this — a process
      killed between the last commit and its sweep leaves one stale blob
      that no later commit will ever name, because the run's last commit has
      no successor. Sweeping on attach makes the store CONVERGE: after any
      commit and after any attach it holds every adapter and exactly one
      optim per delta, which is what keeps a run directory byte-identical
      across a crash even though bytes now leave it. `tests/test_resume.py`
      is green unmodified.
    - IDENTITY IS UNTOUCHED, verified: retention reaches no ExperimentSpec
      field, is registered in nothing, and hashes into nothing. `run_id` for
      the arith specs is identical to the pre-change checkout's, and one
      test recomputes identity and compares manifest bytes across a sweep.
      Retention changes what is RECOVERABLE, never what was COMPUTED
      (I3, I11).
    - THE ONE CONSEQUENCE FOR AN EXISTING PATH, stated rather than hidden: a
      `WarmStart` with `optim="load"` naming a MID-RUN version of a swept
      parent has no moments to read. `::extend` warm-starts from the
      parent's tail, which is unaffected. `_parent_moments` in loop.py is
      one named refusal that says where they went and names the parent's
      tail, instead of surfacing as a bare missing file.
    628 tests green (19 new in tests/test_retention.py). `test_loop` now
    states the new truth — every adapter, moments only at the tail.
    DELIBERATELY NOT DONE: no policy that touches waves/postdata (42 MiB
    against 8 GiB — the measurement says do not bother), and none that
    thins adapters (every k-th version would break `bundle_for` and
    `WarmStart` for the versions it dropped, which is the one thing this
    design refuses). The CLI attaches rather than sweeping a LIVE run: a
    handle without attach-time recovery would make that safe and is a
    separate decision, not a flag. Retention is not on the spec and there is
    no per-run override — the Trainer takes a `retention=` argument
    defaulting to `DEFAULT_RETENTION`, so swapping one is construction, not
    configuration. Not measured on real metal: the main session runs it
    against the real store (this was local work only).

64. **plora: THE POLICY BECOMES A DISTRIBUTION OVER ADAPTERS — a probabilistic
    low-rank delta, and the five contract growths it needed.** The adapter
    type, first, because everything else follows from its shape. Each matched
    weight `M` is factored ONCE into its top-k singular directions and frozen
    there: `A = Σ_k V_kᵀ` and `U_k` never move again. What trains is a small
    hypernet mapping a latent `z ∈ R^n` to a k×k core `C_s` per site, so the
    delta is `Δ_s = U_k C_s A` — which is EXACTLY a rank-k LoRA whose `lora_A`
    is frozen and identical at every version and whose `lora_B` is `U_k C_s`.
    That last sentence is the whole reason this needed no new mechanism: an
    ensemble of drawn latents is an ensemble of ordinary peft adapters, and
    punica already serves those.
    - THE LATENT IS WHERE THE PROBABILITY LIVES: `q = N(mu, diag(exp(log_std)²))`
      against `p = N(0, prior_std² I)`, both trainable, both initialized so the
      run starts at the identity element in TWO senses at once. The heads are
      zero-initialized (ControlNet-style), so every core is zero and version 0
      is the base for EVERY draw — lora's `B = 0`, earned the same way. And
      `mu = 0, log_std = log(prior_std)` puts q exactly on p, so `KL(q‖p)`
      starts at exactly 0.0 and the loss's KL term begins as a term that is not
      yet pushing anything. Both are pinned by tests, the second by equality
      with 0.0 rather than a tolerance.
    - RECORD THE NOISE, NOT THE LATENT. The engine draws `members` noise
      vectors per version, materializes `members + 1` peft dirs (the ensemble
      plus the posterior MEAN), and a request picks one by
      `derive(seed, "plora") % members`; score traffic, which is seedless and
      deterministic by contract, gets the mean. What the turn RECORDS is `eps`,
      and replay recomputes `z = mu + σ⊙eps` with the CURRENT posterior — which
      is what puts mu and log_std on the gradient path of a sample taken before
      either had its present value. Recording `z` would have frozen the
      posterior out of its own gradient, which is a silent failure, not a loud
      one, and is the single most important line in the design.
    - THE FROZEN HALF IS A CAS ARTIFACT, AND THAT IS AN IDENTITY MOVE. `U` and
      `A` are megabytes per site and IDENTICAL at every policy version, so a
      bundle carrying them would pay for them once per update forever. They
      live in `cas://`, the spec names the address, and the payload carries the
      address rather than the bytes (`plora` payloads are kilobytes). The price
      is that the two sides must agree about a coordinate system, so
      `ALGO_ID = "gram-eigh-fp32-canonical-sign-v1"` is stamped into the
      artifact and checked when it is read, alongside base and k. Sign
      canonicalization is not fussiness: `(u_i, v_i)` and `(-u_i, -v_i)` span
      the same subspace and `eigh` may return either, and a core trained
      against one sign is the WRONG POLICY under the other.
    - ONE DELIBERATE ASYMMETRY, stated rather than hidden: the REPLAY side
      recomputes the factors from the base's own weights at install instead of
      reading the artifact. `install_replay(model, params, sites)` has no store
      handle and cannot be given one without widening the Learner contract, and
      the trainer is already holding the checkpoint — so the artifact exists
      for the side that ISN'T holding it, the engine, whose copy lives inside
      vLLM. `ALGO_ID` is what makes them agree, and one test builds the
      artifact and compares it to the direct computation, which is where that
      claim is actually pinned.
    - ONE PLORA PER BANK, refused at `attach` while the bundle is still just an
      id. The recorded facts are a flat namespace — a turn has one `plora_eps`
      — so two entries would record over each other and leave replay unable to
      say whose latent it held. This is correctness, not a limitation.

    THE FIVE CONTRACT GROWTHS, each small, each useful beyond plora:
    - **`provide` and `param_groups` on AdapterType**, both with working
      defaults, so every existing adapter type is untouched (`provide` → `{}`,
      `param_groups` → `{"": params.parameters()}`, one group = the optimizer
      the learner always built). `provide` is the compute half of the
      long-declared `provides`; `param_groups` grows `OptimSpec.overrides` a
      DOTTED grammar — `"pi"` reaches every group of entry `pi`, `"pi.mapper"`
      reaches one, and the dotted form wins where both apply, the only reading
      under which writing both is not a contradiction. plora uses it for the
      thing that motivated it: the hypernet may be weight-decayed, the
      posterior must not, since decay on `log_std` is an unstated second prior
      pulling the scale toward 1.
    - **THE TURN-EXTRAS MEMBRANE CROSSING**, which existed as a documented
      channel (`Turn.turn_extras`, `FinishEvent.turn_extras`) with nothing
      travelling down it. Now: `Levers.turn_extras` unions in `merged_with`
      (disjointness already guaranteed by `claims`), `VllmEngine` folds the
      merged facts into the FinishEvent it yields, `Flat.turn_extras` carries
      one mapping per TURN (deliberately not token-aligned), `TokenBatch
      .doc_turn_extras` carries one tuple per DOCUMENT, and
      `ReplayRows.facts` hands row r its own document's facts. The learner
      fills it ADAPTER-BLIND — it copies mappings across and never reads a key
      — so the next recording adapter type needs no change to any of it.
      `Request` also gained `seed`, because an adapter type with a per-request
      CHOICE must draw it off the seed tree and not an RNG.
    - **`microbatches_in_update`**, stamped by `pack()` on every batch once the
      split is known. ONE WAVE IS ONE GRADIENT UPDATE (#59), so a term that is
      a function of the PARAMETERS alone — a latent KL — is identical in every
      microbatch, and adding it whole to each would multiply it by the
      microbatch count. The effective beta would then depend on
      `microbatch_tokens`: an engineering knob silently changing the objective,
      which is exactly the class of bug #59 was written to kill.
    - **`Host.solo`**, a birth fact attested and journaled like the partition
      and the regimes (I12). I8 promises tenants cannot disturb each other's
      RESULTS; it never promised they cannot disturb each other's THROUGHPUT,
      and a fractional partition small enough that one tenant fills it is
      exactly where that matters. `submit` refuses a second RUNNING tenancy
      (resubmitting the same run_id is a resume, and a finished run frees the
      host); the fleet's join rung SKIPS an occupied solo host rather than
      offering a join it would then refuse — soloness is a birth fact, so it
      belongs to placement, and the ladder falls through to carve exactly as
      for a host lacking the capability. Default stays multi-tenant.
    - **DECLARATION-DRIVEN EMISSION OF PROVIDED TENSORS** (scope added
      mid-build, and it turned out to be the most reusable piece). Every
      provided tensor is summarized to one float by ONE rule — 0-dim is its
      value, anything else is its mean — onto `TrainStats.provided`, meaned
      across an update's microbatches into the ledger's `train` block by the
      Trainer, and given a SECOND flow-graph node: a `kind="stat"`,
      `phase="train"`, `granularity="update"` twin beside the forward node, so
      `dictionary.json` self-describes the per-update summary and the UI
      renders it with no special-casing. `observe/` and the web assets were not
      touched, which is the point. The standing rails are written OVER the
      folded-in provides, so a provide can never shadow `loss`.
    - THE RULING THAT FALLS OUT OF IT, now stated in `base.py`,
      ARCHITECTURE.md and here: **`provides` is not only the loss-input
      channel.** Declaration-driven emission makes every provide observable for
      free, so an adapter type should provide everything a reader of the run
      would want to WATCH — a posterior's scale, a gate's norm, whatever
      internal state explains its behavior — not merely what some loss
      requires. UN-REQUIRED PROVIDES ARE FIRST-CLASS. plora declares
      `plora_sigma_mean` for exactly this reason (a posterior collapsing to
      zero IS plora turning back into a plain LoRA, and it should be visible in
      the ledger the update it starts), and one test pins the whole property:
      a provide no loss requires still reaches the ledger and still appears in
      `dictionary.json`, with `feeds_loss` False and empty consumers on BOTH
      nodes.

    THE COUNTER IS THE VERSION. `emit` uses the counter it holds, RECORDS it,
    and only then advances; `load` restores it. So the n-th emit of a run always
    draws the n-th ensemble whether it happened in this process or the one that
    crashed — Phase 1's re-emit after a resume reproduces the bundle bytes
    the crashed process had, which is what keeps `bundle_id` (and therefore
    every sealed `Turn`) identical across a kill. This is the one adapter type
    whose payload is not a pure function of its parameters, because the served
    ensemble is a fresh DRAW at each version. The SEED is deliberately NOT in
    the payload: seeds come from the spec (I3), so a warm start's child uses
    its own seed tree rather than inheriting its parent's noise stream.

    `grpo_latent_kl` calls `grpo` rather than restating it (the two cannot
    drift) and adds `BETA * kl / batch.microbatches_in_update`. BETA is a
    module constant at 1e-3, deliberately: the loss's source hashes into
    `run_id`, so sweeping it is an edit producing a different experiment —
    which is what a different beta IS. A knob on the spec would let two runs of
    one run_id disagree about the objective.

    697 tests green (69 new: `tests/test_plora.py` plus solo-host tests in
    `test_host.py` and join-rung tests in `test_fleet.py`), and — a first for
    this repo's local work — the torch-gated half was actually RUN, in a scratch
    venv, rather than only written: 62/62, including the per-row replay math
    against a loop reference, the SVD reconstruction against `torch.linalg.svd`,
    and the gradient reaching mu / log_std / trunk / heads. `test_resume.py`
    is green unmodified. `deploy/plora_l4.py` is authored and UNRUN: three
    hosts on fractional partitions of one L4 (0.30 sampling / 0.20 eval / 0.40
    training) over `LocalTransport`, test-time training on ONE DAPO problem,
    four groups of eight per wave. DELIBERATELY NOT DONE: no parity certificate
    (`logprob_gap` is still the rail); no cross-tenant plora coalescing beyond
    what the row plan already expresses; `hf_weight_reader` is unexercised
    locally, having no HF checkout to read; and nothing in `observe/` learned
    the word plora, because the dictionary is what the UI reads.

65. **THE SCORER: the fourth daemon, and post traffic leaves the gradient's
    critical path.** #47 measured the problem and named it: a teacher column
    computed inside the Trainer's post phase makes every gradient wait on a fan
    of sequential 32B prefills. The fix is not a faster prefill, it is a
    DAEMON — `rlstack/runner/daemons/scorer.py` — and the whole build is four
    rules.
    - THE SPLIT RULE, one function (`split_pipeline`, in `spec/flow.py` beside
      the flow graph, because it is one more query over the same declarations):
      a postprocessor declaring `pools` is SCORER-RUN, a pool-less one is
      TRAINER-INLINE. Declared, never guessed and never timed — `pools` already
      names every pool a processor addresses, the gate already vets it and the
      runner already admits engines for it, so the same declaration answers
      WHICH daemon runs it. Sending traffic is what makes a processor slow;
      slow is what has to leave the critical path. A pipeline with no pooled
      half (DAPO, plora — every campaign running today) plans NO Scorer, and
      the Trainer is byte-for-byte the daemon it was.
    - THE PARTS, in `data/stores/base.py`: `write_postdata_part(update,
      producer, columns)` / `read_postdata_part` → `postdata/<u>.<who>.json`,
      beside the merged file, same atomicity, same refusal once the ledger has
      committed the update. The Trainer still writes the ONE
      `postdata/<u>.json`, so flatten, the ledger's column means, `observe/`
      and the wave browser read exactly what they read before and not one line
      of them changed. The read answers None rather than raising, because it IS
      an await predicate. `_parse_update` grew one `split(".")`, which is the
      whole of making attach sweep a part exactly as it sweeps everything else
      the ledger never committed — and sweeping one costs nothing, since
      scoring is deterministic seedless prefill at a pinned bundle. The
      producer is one dot-free name segment, asserted at the key, for the same
      reason a host name may hold no slash.
    - THE VERSION-PINNING RULE, and it is the ONLY place where running beside
      the Trainer instead of inside it changes what has to be ASKED. The
      Trainer scored inline and therefore always held the current bundle; a
      daemon that may be an update ahead — or, after a crash, behind — holds no
      such thing, so it asks the DATA: policy-pool traffic is scored under the
      bundle THE WAVE'S OWN TURNS RECORDED (I6), restored through the same
      `routes_at` closure that already restores on a miss. Scoring the newest
      bundle instead would make a column depend on when the Scorer got round to
      it, which is the one thing a run directory may never depend on. Two edges
      stated rather than hidden: a wave whose turns pin two bundles is REFUSED
      (blending two policies into one column silently is the wrong number this
      design exists to prevent), and a pipeline that never addresses the policy
      pool needs no pin at all — demanding one would make another run's
      replayed trajectories, whose versions this store does not hold,
      unscoreable by a teacher that never looks at the policy.
    - THE EQUIVALENCE OBLIGATION, which is what the other three are for: a
      run's postdata must be the same bytes whether a column was computed by
      the daemon or inline. It holds because the split changes only WHERE — the
      same `run_pipeline`, the same `derive(master, "post", update, group,
      processor)` seed path, the same order within each half, and `given`:
      wave-order columns sliced back per group into each group's starting
      `data`, so an inline processor consumes a scorer column exactly as it
      consumes a neighbour's. What makes that airtight is a contract that
      already existed — a processor reads its declared `consumes` and nothing
      else, and every declared input is produced earlier — so no processor can
      observe which half of the split it is in. `tests/test_scorer.py` runs the
      A/B directly: the same spec, once with the daemon and once with the split
      patched to "everything inline", two stores, one run_id, byte-identical
      run directories with the part file the only difference. The judge is a
      COIN (p_correct=0.5) deliberately, so the reward column actually depends
      on the seed path; the same A/B runs for the policy-pool (opsd) shape,
      where the pin is load-bearing.
    - THE GATE grew one check, `check_pooled_post_follows_inline`
      (`post-split-order`): a POOLED processor may not consume an INLINE
      processor's column. The two halves meet exactly once, at the part, so the
      pooled half runs first and whole; the inversion is a deadlock by
      declaration and is refused while it is still text. The other direction is
      the normal case (`llm_judge` → `grpo_advantage`) and is what `given`
      exists for. TRAIN pipeline only: eval keeps its single inline path, so
      nothing constrains its order.

    THE PACING FALLS OUT RATHER THAN BEING BUILT. The Scorer's await mirrors
    `Trainer.next_rows` with one difference — it READS `waves/<u>` when the
    Trainer has written it and otherwise `realize`s the same rows WITHOUT
    writing them, because one writer per artifact is the rule. Realize is a
    pure function of the plan and the sealed rollouts, so the two agree byte
    for byte, and realizing rather than waiting is exactly what lets the Scorer
    work an update ahead. How far ahead is not this daemon's policy: realize
    answers None until the Generator has sealed the leaves, so the Scorer
    inherits the lag buffer that paces the Generator (#59) and needs no bound
    of its own. The Trainer's measured `post` phase now covers the WAIT plus
    the arithmetic, which is the number that says whether the Scorer is keeping
    up — a scorer far enough ahead collapses it to the arithmetic alone.

    734 tests green on fakes (29 new in `tests/test_scorer.py`);
    `test_resume.py` and `test_post.py` unmodified and green, and the two
    existing end-to-end pipelines that turn out to be POOLED —
    `("verifier", "teacher_logprobs")` and `("verifier", "hinted_logprobs")` —
    now run through the daemon in `test_post.py` without a line of that file
    changing, which is the equivalence claim asserting itself. Mutation-checked
    three ways: a changed seed phase, a broken group slice and an ignored
    recorded bundle each fail the suite.
    V1 LIMITS, deliberate: ONE Scorer owns the WHOLE pooled half of the TRAIN
    pipeline, IN THE RUNNER'S PROCESS beside whatever engines the routes map
    holds — a scorer standing on its own host is placement work that belongs
    with host adoption, not here. The EVAL pipeline keeps its single inline
    path (the Evaluator is firewalled measurement and its cadence is a modulus,
    not a critical path), so `run_pipeline`'s unbounded eval fan (#53's
    leftover) is still unbounded. NOT DONE, and unchanged by this: teacher
    scoring is still un-batched (one `score_tokens` per turn) and un-cached
    (identical prefixes re-prefill) — this entry moved the work off the
    gradient's path, it did not make the work cheaper. Nothing is proven on
    metal: the fakes suite is the whole evidence, and the number this build was
    designed against is still the one the OPD stress test would produce. A part
    is never deleted at the merge: deletion has exactly two meanings in this
    store and neither is "a reader is done".

66. **THE GATED LATENT KL: accuracy first, then the prior.** Ported from the
    arc-agi-ttt harness, where the ordering is EMERGENT rather than written: a
    group whose completions are all correct has an all-tie, z-scored-to-zero
    advantage, so a constant KL term is simply the only gradient left once the
    task is solved. rlstack's `grpo_latent_kl` already had that emergent half;
    what the TTT campaign wants is the other half made explicit — NO prior
    pull while the policy is still learning to be right. Two primitives, both
    existing shapes:
    - `training/post/group_accuracy.py`: consumes `reward`, produces
      `accuracy` — the group's mean reward, one copy per row. The group is the
      gate's scope on purpose: it is the advantage's baseline scope, so "this
      group is solved" and "this group's advantage is identically zero" are
      the same fact, and the gate hands exactly the dead rows to the prior.
    - `training/losses/grpo_latent_kl_gated.py`: grpo's surrogate verbatim
      (called, not restated) plus `gate * BETA * kl / microbatches_in_update`,
      where the gate is the masked fraction of tokens whose `accuracy` reached
      `FULL_ACCURACY` (1.0, a module constant like BETA and for the same
      reason: the threshold IS the objective, so sweeping it must change
      run_id). The gate is a COLUMN, not a branch in the trainer (I9/#38).
    THE ENDPOINTS ARE EXACT regardless of packing — nothing solved pays no KL,
    everything solved pays the whole BETA and coincides with `grpo_latent_kl`
    to the float (pinned by test). Between them the effective beta is the
    per-microbatch solved fractions averaged, a wave statistic that depends
    mildly on how `pack` split the update — stated in the docstring rather
    than hidden, and accepted: the alternative (a per-update gate) has no
    scope to live in, because a postprocessor sees one group and a loss sees
    one microbatch. deploy/plora_l4.py now runs
    `grpo_latent_kl_gated` with `("final_answer", "group_accuracy",
    "grpo_advantage")` — still ONE task (the screened 4/8 problem), where the
    gate reads plainly: pure GRPO until a group goes 8/8, then that group's
    share of the update becomes compression toward the prior. 740 tests green
    on fakes (6 new); the three numeric assertions (gate closed = grpo
    exactly, gate open = grpo_latent_kl exactly, half solved = BETA/2) are
    torch-gated and run in the image. NOT run on metal yet.

---8<--- cut here ---8<---

68. **THE SUBMISSION DOOR AND THE STANDING FLEET: adopt, the desk, and the
    mixed-family chain — three rulings, one arc, each proven by the failure
    that demanded it.** (Renumbered from a colliding #66 at the
    worktree merge; chronologically it sits between #65 and #67.) The deployment model was always "experiments request
    resources and join matching hosts", and the fleet could PLACE (#43) but
    nothing could DELIVER: `submit` was an in-process method, so a campaign
    could only run where its script already stood. This entry is the missing
    delivery, built outward from one verb.
    - **`Host.adopt(spec_row, routes)` — submit, without the submitter
      in-process.** The frame is the spec's canonical JSON plus placement's
      routes (pool name -> ADDRESS, for pools this host does not serve). The
      schema is derived HOST-SIDE (`schema_for`, a birth fact): an adopted
      spec never ships a schema, so identity is computed where the code that
      will run lives (I3) and a client cannot ship one the metal disagrees
      with. The reply is ACCEPTANCE — {run_id, state} — never completion: the
      ledger is the result channel and peeks are the door, as for any run.
      Custody checks (bind, fit, solo) run fail-fast in the reply; the submit
      gate still runs at Phase 0 inside the tenancy task. RE-ADOPTION IS
      RESUME, the same way resubmission always was. Two fixes the desk's own
      tests forced, both now rules: `dial` resolves an address to a
      TRANSPORT and the HOST wraps it with the pool's declared (base, tp) —
      the venue knows where, the spec knows what — and acceptance rosters the
      tenancy EAGERLY, because a roster written when the background task
      first ticks lets two adopts in one breath both pass check_solo.
      `spec_from_json` (canonical_json's typed inverse, dispatch over the
      closed class table) lives with the WIRE CODECS in runner/remote.py, not
      in canonical.py — the architecture test refused the first placement,
      correctly: spec modules are pure values, and only the wire decodes.
    - **`FleetService` — placement as a service, and the fleet journal's ONE
      WRITER.** The Trainer/ledger pattern applied to the fleet plane: the
      desk holds LISTINGS (descriptions of standing hosts — regimes, address,
      solo — deploy-registered at boot, journaled, rebuilt by `from_journal`
      after any kill), matches the join rung over them, and `submit` ends in
      an adopt at the learner's listing with every other pool's address
      threaded as routes. A placement nothing serves returns BOOT
      INSTRUCTIONS: the standing carve is a venue action (boot a container
      wearing the regimes, list it), because the desk is a CPU process and
      the factories live beside metal it does not have. `delist` and a
      placement-time liveness probe complete it: a dead container is skipped,
      never offered. Single-writer is also the multi-user answer: concurrent
      campaigns serialize through one desk instead of double-reading one
      residual. NOTHING IN rlstack/ SAYS THE VENUE'S NAME — the desk knows
      `connect: address -> RemoteHost`, a campaign is
      `RemoteFleet(transport).submit(spec)`, and the Modal cls transport is
      ~10 lines in the deploy file.
    - **`SiteWrapper` — mixed adapter FAMILIES at one site path nest, not
      collide.** Found by the sweep, not by review: 40 tenants (28 plora, 12
      lora) on ONE shared learner all died in minutes — plora rows routed
      through a lora tenant's wrapper met lora math (`state.b` on a
      PloraState, 20 arms), lora rows through a plora wrapper were asked for
      recorded latents they never drew (12 arms). The hole is as old as #44:
      every site wrapper assumed all state at its path was its own family,
      and no prior tenancy ever mixed families at a path (the stress matrix
      is single-family per site). The fix is a shared chain in replay.py:
      SiteWrapper owns the roster and the nesting — `join_site` finds a
      family's wrapper anywhere in the chain or wraps the head; `leave_site`
      splices it out wherever it sits — and each family's forward applies
      ONLY rows whose routed state is its own, passing the rest through to
      `inner` (the other family's wrapper, or the bare Linear). Pinned by
      tests/test_mixed_families.py: a tenant beside a foreign family is
      BIT-IDENTICAL to the same tenant alone, in both install orders.
    - **The venue lesson, paid for once:** `modal run`'s ephemeral app dies
      with its entrypoint and takes spawned work with it — 32 seeded arms,
      killed at seed time. The standing container is DEPLOYED and every door
      looks it up by name, which is not a workaround but the model itself:
      sweep, via_desk and progress are three processes knocking on one
      standing host.
    - **Proven on metal (deploy/sweep_a100.py):** 40 experiments — plora
      k x prior_std x latent against lora r x lr, two seeds — as tenants of
      ONE shared engine and ONE shared learner on two fractional partitions
      of a single A100, 32 arms through the in-process door and 8 arriving
      from a SEPARATE process as one RemoteFleet frame each, all forty on one
      roster, all forty committing. The observation door queues behind
      training forwards on a saturated loop (progress took minutes at
      40-tenant width) — a stated cost; the store-side read is the observer's
      answer, as everywhere.

67. **FILING: runs spawn into subdirs, and the tree reaches the UI.** The
    store's runs/ was flat and organization leaned entirely on tags; now a
    submission may INDICATE a subdir and the run's directory spawns at
    runs/<subdir>/<run_id>. Three rulings hold it together:
    - **Filing is never identity.** The subdir rides submit/adopt/desk frames
      as a parameter beside the spec — it does not hash, so the same spec
      filed differently is the SAME run (pinned: identical run_id and
      byte-identical ledger at runs/sweeps/arith/<id> and runs/<id>). It
      could not live in the spec without violating I3's "identity is
      content".
    - **One home, for life.** open_run resolves a run by its manifest
      wherever it lives (`run_prefix`, the one seam every peek and every
      handle key routes through, cached because runs never move); a
      different subdir asked at attach is ignored — resubmission is resume,
      not a move. I10's "for life" now includes the address.
    - **Segments are attested** (check_subdir: [A-Za-z0-9._-]+ per segment,
      never "." or ".."), for the host-name reason: a stray separator would
      file a run where no reader looks.
    The observer says where each run lives (runs_data rows carry `subdir`;
    the web index nests a collapsible block per subdir INSIDE each #58
    folder block; the CLI prints `dir <subdir>/` headers). The two axes are
    deliberately distinct and both rendered: a FOLDER is which store (#58, a
    root chosen at birth), a SUBDIR is filing inside one store. Key helpers
    (wave_key and kin) now take the run DIRECTORY, not the id — the one
    signature change, caught by one test.

68. **THE METAL PLANE AND THE REAPER: the desk deduces, the metal enforces,
    and silence gets a janitor.** Until now the standing carve was a human's
    errand (boot instructions in the submit refusal) and a host that died
    without saying delist left a stale listing forever. Samarth's ruling
    closed both, with one invariant named first: **a carve accepted is a
    fraction promised** — space must be booked from the ack, not from the
    build's completion, or two carves in the build window double-book.
    - **MetalService** (runner/fleet.py) is the metal-side end: the
      container that owns a device wears it BY DEFAULT beside its
      HostService routing. It holds ONE registered Metal's books — built
      partitions (hand-built standing hosts enter via `adopt_born`, refused
      without a partition) plus PENDING bookings — and serves `carve`
      (books synchronously before the build's first await; factories run in
      a worker thread because an engine boot is minutes; a failed build
      releases its booking), `decarve` (the venue's `release` unmakes what
      the factories made, the fraction returns to residual), and
      `residual`/`describe` on the ask path. It writes NOTHING to the fleet
      journal: the desk stays that journal's one writer.
    - **The desk commands carves** (the bare provision CALLABLE is
      superseded): metal registration now carries an ADDRESS (`metal` verb —
      the container phones home its own existence beside its host listings;
      journaled; from_journal redials via `connect_metal`), and
      provision_unit asks each registered metal's residual (the DEDUCTION)
      then commands the first that fits (RemoteMetal.carve). A stale
      deduction costs a refusal at the metal's door, never a double-book;
      what no metal holds is still a boot instruction — the standing
      acquire stays a human's. A unit carved for a placement whose later
      unit missed STAYS listed (metal born is metal listed).
    - **Listings carry the capacity VIEW** (partition row + metal name,
      journaled, rebuilt): the desk can deduce, plan, and render — and the
      row is exactly what enforcement must not rely on, because a view
      cannot see in-flight bookings or unilateral deaths. Both of Samarth's
      options landed, with the authority split stated.
    - **reap(probes, wait)**: every listing probed; the silent RETRIED
      (Samarth's "somehow restart" — on Modal the knock itself boots a
      stopped-but-deployed container, so a probe that queues until bring_up
      answers reads recovered, and only a torn-down deployment stays
      silent); the still-silent DECARVED at their metal (a living container
      frees the fraction; a dead one already did, physically) and DELISTED
      with reason="reaped" journaled, so the rebuilt desk agrees.
      fleet_a100 runs it on a modal.Period(15m) schedule and as a manual
      door. Verdicts: alive | recovered | reaped.
    Wire growth: fleet verbs `metal`/`reap`, metal verbs
    `carve`/`decarve`/`residual`/`describe`, RemoteMetal beside RemoteHost/
    RemoteFleet, list frames carry partition/metal, delist carries reason.
    fleet_a100's Metal container books its two standing hosts (residual
    honestly ~0.08) and mints carve addresses; the booking race, the failed
    build, the reaped-carve-frees-metal circle, and the recover-vs-reap
    verdicts are all pinned on fakes (test_fleet_service, 26 tests).

69. **THE BLIND DESK: the desk allocates Demands; specs meet the fleet in
    the campaign layer; Fleet-the-class is gone.** Samarth's ruling, arrived
    at from the side evaluator (pair_eval.py hand-rolled find_listing and a
    third copy of the venue transports because the desk's only door demanded
    an ExperimentSpec with a learner): the desk should not know what an
    experiment IS — its fundamental job is to accept any Demand and try
    allocating it; the host owns what a workload means.
    THE BOUNDARY: rlstack/runner/desk.py imports no spec class (the acid
    test, pinned: a gibberish frame is placed and relayed untouched, the
    HOST refuses it, the desk's reply carries the host's error). The desk's
    vocabulary is Demand rows (demand_rows/demands_from, the wire codec;
    Demand grows `anchor` — where a delivered frame lands), listings, metal,
    addresses. Two doors: `place(demands)` answers addresses (the PURE
    CLIENT's door — an evaluator joins a pool and is thereafter just
    admitted traffic; journaled delivered=False), and `submit(demands,
    frame)` places then DELIVERS the opaque frame to the anchor demand's
    host with routes threaded — routes are derivable from the demand rows
    alone, which is what makes the blind relay possible. One frame keeps
    placement+delivery atomic (no leases, the #43 ruling stands).
    THE CAMPAIGN LAYER (runner/campaign.py) is the only place specs become
    demands: demands_of marks the learner demand anchor (the learner is
    never remote lives HERE now, client-side, where the spec is), frame_for
    carries the canonical row + code claim, and Campaigns is the desk's
    spec-aware sidecar — it owns migrate/sliced_plans (they read stores AND
    specs, so they ride beside the desk in its container, one Transport
    door: Campaigns.serve handles migrate, delegates the rest). RemoteDesk
    .submit(spec) keeps its signature — shaping moved client-side —
    and .resolve(demands) is the pure client's verb (pair_eval's
    serving_pool, deleted).
    THE RENAME (a #55-style vocabulary arc, identity-free — machinery names
    are not hashed): FleetService → Desk, RemoteFleet → RemoteDesk, fleet.py
    → desk.py, the deploy verbs fleet/fleet_ask → desk/desk_ask.
    Fleet-the-class (the in-process ladder) is DELETED, and with it
    Join/Carve/Acquire/Plan-as-data: its unique value was the ladder over
    live hosts, which Desk + MetalService with local factories already are —
    the duplicated join rule (Fleet._covers vs _covers_regimes, a listed
    finding) collapses to ONE `covers()`. "The fleet" survives as the
    collective noun for the plane (the journal key fleet/log.jsonl and
    read_fleet_log/append_fleet_event keep it; renaming bytes on disk
    orphans history). Host/Partition stay two words for two planes (1:1
    stated in ARCHITECTURE.md). Tests: test_fleet_service → test_desk (+
    place/blindness/anchor claims), test_fleet → test_placement (the pure
    vocabulary: demands_of + anchor, placement_units, fraction_for_gb,
    covers). NOT REDEPLOYED with the pair mid-flight: the wire verbs rename,
    and a redeploy would strand the running metal generation and let the
    reaper decarve the live pair's listings — the deploy waits for the
    natural break (the migrate move is the natural break).

70. **MEASUREMENT LEAVES THE RUN: EvalSpec retired, the Evaluator daemon
    dissolved, observation is its own thing.** Samarth's ruling, forced by a
    live incident: the pair's in-run eval was pinned by identity to a
    floor-scoring task set for all 400 updates, and the sanctioned fix (the
    side evaluator) worked precisely BECAUSE eval was firewalled and every
    adapter version restorable — at which point the in-run Evaluator was
    revealed as an identity liability with no compensating power.
    THE CUT: EvalSpec is deleted, ExperimentSpec.eval and Plans.eval are
    gone, the Evaluator daemon and its wiring are gone, validate's eval
    rules are gone, code_hashes no longer reaches eval.post, and migrate's
    eval-slicing deletes itself. A run's identity is its TRAINING loop.
    Old stores are readable history: a pre-#70 canonical row's eval keys
    are extra fields the decoder never touches (pinned by test), and the
    observer still renders legacy eval/ summaries beside the new shape.
    THE REPLACEMENT (runner/measure.py): a Measurement is an observation OF
    a run — name, env, held-out task ids, samples, cadence, post pipeline,
    its own seed — written to measurements/<run_id>/<name>/ (write-once
    manifest + append-only points; ModalVolumeStore persists each point).
    `measure_run` is one idempotent pass any process can run against a
    pool: backfills every missing EVERY-th version from restored bundles
    (KeepRestorable is what licenses reaching into the past), reduces in
    WAVE order (#53's rule carried over, scrambler-pinned), and follows
    the ledger on any cadence. It peeks, never attaches; the run dir gains
    not one byte (pinned). Swapping what a run is measured on mid-run is
    a NON-EVENT: stop one measurer, start another under a NEW name —
    supersede, never rewrite (open_measurement refuses a changed manifest).
    THE EDGES: measurements/ is the store's ONE deletable tree, so the
    observer cache believes its absences instead of blink-guarding them
    (the deliberate carve-out); the UI renders every named measurement as
    its own dashed series beside legacy eval; pair_eval.py's tick is now a
    thin venue wrapper over measure_run writing measurements/<rid>/heldout
    (UI-visible). Vocabulary: RESIDENT vs DAEMON named in ARCHITECTURE.md
    (an Engine/Learner lives on a partition; a Generator/Trainer/Scorer
    watches the store and pokes one) — colocation is transport choice, and
    only the Trainer's is forced (the autograd arc and the seal cannot
    cross a wire).

71. **DECOMMISSION: carve's inverse, client-asked.** The shutdown of the
    gated pair proved the hole: delist is bookkeeping-only (the engine kept
    serving after it), decarve existed only metal-side, and reap composes
    them only for SILENT hosts — so retiring a LIVING host meant killing
    containers by hand. New desk verb `decommission(host, force)`: decarve
    at the listing's metal (release -> engine shutdown, fraction back to
    residual) + delist(reason="decommissioned"), one frame, RemoteDesk
    method included. THE GUARD IS DEPENDENTS, not occupancy: a serve host's
    roster is empty (tenancies live at their anchor), so the desk joins the
    journaled placements' pools against the live rosters' running runs and
    refuses BY NAME; force proceeds. Hand-listed hosts (no metal on the
    listing) only delist; a silent metal delists too (reap's reasoning on
    demand). Reallocation is no new machinery: residual grew, the next
    carve may land there — pinned by the carve-after-decommission test.
    Closes the "no decommission rung" finding. Venue lessons from the same
    shutdown, recorded for the deploy notes: stopping a Modal CONTAINER
    does not stop a SPAWNED call (the serve() input reschedules onto a
    fresh container — the standing shift's kill-switch is the app or the
    call id), and desk-side probes of dead listings each knock-boot the
    metal by name, so delist/decommission BEFORE the container kill.

72. **REROUTE: restart-is-redial.** Taking a host down must not take its
    tenants with it, and no state needs to move to guarantee that — the
    store is the run (resume-equivalence) and adoption is resume, so a
    "reroute" is stop + place + redeliver, nothing copied. Three pieces.
    (1) `Host.stop(run_id)`: the per-tenancy kill, adopt's inverse —
    cancel the adoption task and AWAIT it (daemons unwind structurally
    under the TaskGroup; submit's except path rosters failed + journals
    detach; a pre-try cancellation is normalized by stop itself), so the
    reply means the death is complete and the run_id is free to adopt
    again anywhere. HostService verb + RemoteHost method; stopping
    mid-update costs one redone update, nothing else. (2) THE ARCHIVE:
    `Desk.deliver` (submit's delivery half, now one copy) journals the
    demand ROWS and the opaque FRAME inside the delivered place event —
    still unread, so the desk stays blind — and `Desk.placements()`
    promotes the journal archaeology to a read: latest binding per
    run_id, rows and frame included, `dependents` joins against it.
    (3) `Desk.reroute(run_id, avoiding, park)`: replay the archived
    delivery — place the rows again with `avoiding` off the table
    (find_listing/place_listings grew an `avoid` set; a carve never
    lands there because a carve is a new name), stop the old tenancy at
    whichever listing's roster carries it (`stop_anchored` probes — at
    most one answers), deliver the archived frame to the new placement.
    PLACE-FIRST: a healthy run is never stopped with nowhere to go;
    `park` (decommission's mode — the host dies regardless) stops it
    anyway and journals `parked` with the boot instructions, the run
    waiting whole in the store until a human adds metal and RESUBMITS —
    the revival is the ordinary campaign submit, same run_id.
    `decommission(host, reroute=True)` moves every dependent first and
    tears down after. Pre-archive deliveries refuse the replay with the
    cure named (campaign resubmit). Solo self-rejoin is a known
    non-feature: place-first sees the still-running tenancy occupying
    its own solo host. Pinned by StopTest + RerouteTest (moved run
    finishes on the fresh carve under the same run_id; refused move
    leaves the run running to done; park + revive; unarchived refusal).

73. **THE DSL CAMPAIGN: invented tool languages, the spectrum arms, and the
    reflect loop.** Samarth's thesis under test: tiny trainable surfaces
    (<100k params) leveraging what the model already holds, and language
    itself as the credit channel. The pieces, all on branch dsl-campaign:
    (1) TWO INVENTED DSLs — the Stamp Office (protocol: grab/fold/ink/seal/
    file, invented colors, a color->drawer table) and the Glyph Exchange
    (routing: four units on a one-way ring) — rulebook-in-prompt, verbs
    absent from pretraining, dense milestone ladders (the DAPO
    all-or-nothing lesson), 2 train requests vs a sweeping eval (mol never
    trains; ring directions eval wider than they train). Task builders in
    data/tasks/, graders IN the registered post classes (code_hashes reads
    class source only). (2) SPECTRAL (SVF): one gain per singular direction
    of each matched weight, top-k by |sigma*delta| served through plain
    punica, straight-through backward so all directions compete; dense
    gains ride the payload for resume, materialized peft pair for the
    engine; NO factors artifact (the trainer recomputes the full SVD at
    install). spectral_latent is its plora-style twin — gains GENERATED
    from a latent (posterior + trunk + zero-init per-site heads), served as
    members+mean materialized adapters, recording slatent_eps/member — the
    "does the latent help" ablation. The latent-KL provided channel is
    RENAMED plora_kl -> latent_kl so grpo_latent_kl(_gated) price either
    adapter. plora grew basis="random" (scale-matched random orthonormal
    frame, its own algo id) — the does-the-SVD-frame-matter control.
    (3) THE REFLECT LOOP (iterative SDPO): Derive is the THIRD leaf
    (mint-then-make: a registered TaskMaker turns a sealed trajectory into
    a new task; source refs speak Replay's grammar; generator order
    guarantees the source is sealed), MAKERS is a new registry declared in
    GenSpec.makers and hashed by code_hashes (a NEW SPEC FIELD — canonical
    bytes of every spec change, old rows decode), the reflect maker
    re-serves the WHOLE transcript plus "what went wrong" (chat delimiters
    ride in task meta["chat"]), reflect_retry is the two-turn env
    (critique, injected retry ask, retry), and sdpo is FINAL-TURN behavior
    cloning read off segment_ids — no reward in the loss; the graders ride
    for the observer. Generator pacing gained the intermediate rule: an
    un-referenced rollout is due when the first LATER referenced one is
    (loop plans train only wave 3k). SCoRe's sandbagging caveat stands;
    the Measurement (plain env, iteration-0 behavior) is the honest metric.
    (4) REVERSE PPO IS DEAD: v1's suffix critic provably collapses (causal
    hiddens make every window fully informed -> constant track -> credit
    spikes at EOS); the v2 prefix/RUDDER rewrite was built, then Samarth
    cut the arm — the loss is deleted; value_head KEEPS its new compute
    half (boundary tap + mask hook + zero-init probe, provides "values")
    with no consumer in the zoo. deploy/dsl_a100.py: 2xA100 (one metal
    container, devices=2), desk-carved serve+learn hosts, twelve tenants
    (6 arms x 2 DSLs: grpo / svd / nosvd / spectral / slatent / sdpo), a
    10-minute measurement cron sweeping every run's eval set (phrasing 0)
    under the PLAIN env, roster at measurements/dsl/runs.json.

74. **A RESIDENT IS A PROCESS (ADR 0002): engines and learners as supervised
    children of the metal, pinned and capped by their partition; the
    learner's install is a typed Parameterization.** Samarth's prompt: move
    vLLM to its own process ("live monitorability of which pools are living"),
    and is there a parallel for torch learners. The root, named in the ADR:
    every substrate knob for WHERE (CUDA_VISIBLE_DEVICES) and HOW MUCH
    (torch's per-process allocator cap, vLLM's gpu_memory_utilization) is
    process-granular, and a partition was smaller than a process — so
    `partition.devices` reached nobody who could act on it (ADR 0001 Q4) and
    `set_per_process_memory_fraction` lived in one docstring. The answers,
    all in session: both kinds (Q1); loss bound at install because the
    moments belong to one loss (Q2); emitted bytes on the wire, with NCCL
    between two residents named as the NEXT mode and kept open (Q3);
    building is UNIVERSAL, not a venue's (Q4 — Samarth's correction of the
    draft); a restart goes through resubmit + recarve so metal, desk and
    observer agree (Q7); a learner CAN sleep — the draft mistook an absence
    for an inability (Q8, Q8a: fsdp=1 lands); JSON frames (Q10).
    - **runner/residents.py** (new): `Resident.spawn(birth)` — a spawn-context
      child (never fork; NOT daemonic, a learner has a chorus) that pins
      CUDA_VISIBLE_DEVICES to the partition's devices before torch loads,
      caps a learner's allocator at partition.memory, builds by rlstack's
      universal builders, reports a `hello` (kind, base, tp/fsdp, sleeps,
      devices_seen, pid — the metal refuses a measured device count that is
      not the partition's) and serves frames; `Resident.in_process(birth,
      obj)` is the same door around an already-built object (the fakes
      suite's path, and what makes the two paths one). Frames are JSON-safe
      dicts, request-id multiplexed over one pipe (`PipeTransport`, one
      reader thread): an engine child dispatches `call`s concurrently on its
      own loop so sampling keeps batching (Q5); `ask`s run inline and blocking,
      exactly today's semantics (Q6, revisit later). Door verbs on no
      protocol: hello, sleep/wake, stop. The teardown ladder
      (Teardown/escalate/join_survivors, #53) moved here; ranks.py imports it
      (Q9).
    - **BUILDING IS UNIVERSAL.** `build_engine`/`build_learner` own class,
      base, width, device (`cuda:0` IS the partition's first device once
      pinned; shape n leads a chorus, every follower capped on its own
      device) and fraction. A venue declares `Builds(engine=EngineBuild(...),
      learner=LearnerBuild(...))` — the capacity knobs a partition cannot
      tell you — and nothing else; `FakeEngineBuild`/`FakeLearnerBuild` put
      the fakes in a real process. The recipe lives on the metal, is
      RE-DECLARED at every bring-up from the deploy's constants, and is
      journaled on the `metal` registration row and every `host-up` (Q4a:
      runs stay auto-restartable through resubmit + recarve, and the record of
      HOW is durable). A child reopens the store from a `StoreAddress`
      (`Store.address()` on both backends; `open_store` in
      data/stores/address.py — a mount-only view for a volume store, because
      a resident reads cas blobs and writes nothing).
    - **THE LEARNER'S BOUNDARY.** `Learner.install(tenant, Parameterization)`:
      base, loss BY REGISTRY KEY, entries (adapter type by key, init with the
      per-entry seed already derived, trainable, resolved sites),
      OptimSettings — built by `loop.parameterization_of`, THE place a spec
      becomes an install (`init_seed` moved to loop.py; the derivation is
      byte-identical). `runner/learners/` imports no spec class, pinned in
      test_architecture (#69's acid test, one region over). The chorus
      broadcasts the Parameterization instead of the spec. `RemoteLearner`/
      `LearnerService` are the wire (codecs for TokenBatch / TrainStats /
      Emitted / Parameterization, bytes as base64 in ONE codec, so the NCCL
      carriage later replaces the codec and leaves the verbs alone).
      `EngineService` is the engine-verb half of HostService; HostService
      admits, then forwards through the proxy. `RemotePool` is now also the
      Host's proxy to its own engine child; `attach_residents` treats a
      RemotePool the host attached at birth as local (a remote pool is one
      nobody attached).
    - **THE HOST STAYS THE DOOR.** Arbiter, roster, runner, journal in the
      metal process; `Host(residents=...)`; `_attach_regimes` wires
      evict/wake hooks to the door of every resident whose hello says
      `sleeps` — engine AND learner (Q8): `TorchLearner.sleep`/`wake` move
      the base, every tenant's params and moments to host RAM and empty the
      allocator's cache (fsdp=1; a chorus reports sleeps=false, DTensor
      offload is its own proof). This un-orphans the sleep seam on this
      branch, where nothing wired it. `status()` and `describe()` carry
      `residents` rows; `host-up` carries label + pid; the observer's hosts
      view and the UI's hosts page show them.
    - **A DEAD RESIDENT IS A DEAD HOST (Q7).** The watcher waits on the
      child's SENTINEL — never join(): two threads reaping one child made
      the loser read ECHILD as "alive" through the whole ladder, observed on
      fakes — and `MetalService.resident_exited` decarves the host: siblings
      down the ladder, booking freed, address gone, the death on
      `service.deaths`; the desk's next probe reaps the listing. Nothing
      restarts in place. MetalService takes `builds` + `spawn` instead of
      engine_factory/learner_factory/release; `decarve` is the ladder;
      `shutdown()` is what @modal.exit calls. Four venues converted
      (gsm/dsl/gsm_sweep/fleet a100): typed recipes, `register_metal(...,
      builds=)`, bring_down through the ladder.
    - Tests: 868 (+13). test_residents: codecs; RemoteLearner over
      LocalTransport byte-identical; the in-process door (hooks for both
      kinds, refusal without `sleeps`, status/host-up rows, the pin
      asserted); REAL children — a run through process residents
      byte-identical to in-process with sleep/wake frames crossing to real
      pids, a SIGKILLed resident decarves its host, a birth failing inside
      the child reported through the hello and released, the ladder's rungs
      on stubs (no torch needed now). test_desk's fixture spawns in-process
      residents. Suite ~7 s.
    - UNPROVEN ON METAL, all of it real: the pin on a 2+-device metal, the
      torch cap, learner sleep's move set (adapter tensors outside
      `parameters()` are a stated gap), the follower cap, `open_store` on a
      Modal mount, teardown inside Modal's 30 s grace, and the round-trip
      cost of a TokenBatch frame and of `tokenize` per trajectory (the local
      tokenizer beside RemotePool is now due). Two deviations from the ADR,
      stated: `observe/locate.store_for` is untouched (the address is derived
      from the Store object, not parsed from a locator), and the pipe
      transport lives in residents.py beside the door it serves rather than
      in remote.py. ADR 0001's Q4 is answered here (branch b); its
      alternation sizing (Q7 there) now rests on Q8a's learner sleep.

## Open threads (do NOT treat as settled; flag when your answer touches them)

- TODO (Samarth, settled intent — future, nothing now): BUNDLE LRU EVICTION
  + FAULT-IN. Registrations only grow today (add_bundle is additive forever;
  vLLM already LRU-pages GPU slots/CPU cache from our kept peft dirs — the
  growth is OUR map, disk dirs, prompt-row tensors). The future shape: LRU
  eviction of cold registrations at the engine layer, and on a request
  pinning a VERY OLD bundle_id, fault the weights back in FROM THE CHECKPOINT
  STORE (adapters/<name>@<v>.bin -> compile_bundle reproduces the identical
  content-addressed id — the resume path IS the fault-in path, so eviction
  can never cause wrongness, only a re-registration stall). Two rules: never
  evict under an in-flight pin (I8 immunity); fault-in is transparent at the
  pool-client layer (catch "never registered", re-add, retry). Verify vLLM
  0.28's own max_cpu_loras boundary behavior on the pinned build first.
  Consequence for the join-refusal thread: slots become soft, contention
  becomes the only join currency.

- PLANNED (Samarth-approved, queued behind #48 landing): the OPD stress test
  — Phase A: MATH levels 3-5, numeric-answer subset, few-shot raw-completion
  prompts (no chat template — the v0 constraint), three-host 8B<-32B topology,
  50-100 updates, lag=1, eval/10; pre-registered success: reverse-KL trends
  down, verifier reward non-degrading vs a teacher-baseline probe, gap at the
  kernel floor. Deliberately runs with INLINE teacher scoring to measure the
  scorer-daemon bottleneck at scale (daemon build comes after, informed by
  the number). Phase B after A: a two-turn solve->revise @environment (one
  file) exercising multi-turn seal/flatten + cross-turn teacher scoring.

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

- OPEN (never resolved, only SIDESTEPPED — #61): THE BANK IS REPLICATED, NOT
  SHARDED. FsdpTorchLearner shards the frozen base and nothing else: every
  rank holds the whole bank plus its whole optimizer state (fp32 param + grad
  + two Adam moments) and steps its own copy, so `fsdp` divides the base and
  divides NOTHING a tenant trains. That is ZeRO-3's parameter mechanic on the
  base (where gradient and optimizer sharding are vacuous — the base is
  frozen) and not even ZeRO-1 on the bank. Measured consequence: Samarth's
  asked-for rank-256 all-linear bank on a 14B is 1.03B trainable = 15.3 GiB
  per rank of replicated state, which does not fit beside a 6.88 GiB base
  shard on a 22 GiB card AT ANY WIDTH. The campaign runs only because the
  ruling dropped the bank to r=16 (21M params, 0.31 GiB), which hides the
  gap rather than closing it.
  THE LADDER, in the order the value arrives: (1) ZeRO-1 — shard the fp32
  master and the two moments across ranks, each rank steps its slice of the
  (identical) grads, all-gather the params after the step. Removes 12.3 of
  the 15.3 GiB, touches only optim_step, and leaves #44's additive install
  and the width-free emit bytes alone. (2) Gradient sharding is meaningless
  here without splitting DATA across ranks — today every rank recomputes the
  SAME microbatch, so fsdp is a memory tool, not data parallelism; changing
  that is its own design decision. (3) Full param-sharding of the bank is
  last: it collides with additive install/uninstall and turns emit into a
  distributed gather, which the width-independence invariant would then have
  to be re-proven against.
  Also unresolved from the same measurement: emit writes fp32 and
  RemotePool.add_bundle base64s it into ONE JSON frame, so a 1B bank is
  ~4.1 GB on the wire per update (~5.5 GB/update, ~275 GB over 50) plus a
  4.1 GB peft dir per bundle. Both gate any large-adapter run; neither gates
  r=16.

## Ground rules for you

- Do NOT edit rl-stack-spec.md, rl-stack-design.md, or any artifact — the main
  session owns canon. If your answer implies a spec change, end with a clearly
  marked "PROPOSED SPEC CHANGE:" section stating the exact diff in words.
- Prefer primary sources (the ref/ clones, official docs via WebFetch/WebSearch)
  over memory for version-sensitive claims; today is late August 2026.
- Answer at the mechanism level; use the project's vocabulary (seal, membrane,
  bundle, pool, lowering, wave, delta, site, certificate).
- Your ENTIRE final message is the answer payload — no meta-commentary.
