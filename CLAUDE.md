# rlstack — session handover

Samarth's personal high-throughput RL-for-LLMs harness ("the thin wrapper").
Designed and built across long Claude sessions; everything you need to
continue is in the repo.

## Read before writing code

1. **STYLE.md** — binding. Eight rules; rule 8's folder tree IS the
   architecture, enforced by `tests/test_architecture.py`. Deviations are
   review findings.
2. **ARCHITECTURE.md** — the universal vocabulary reference (#54): every
   noun and verb defined once; docstrings speak it, and a word used with a
   different meaning is a finding.
3. **agent-context/CONTEXT.md** — the decision log. Chronological, numbered;
   later entries supersede earlier ones (#28–#48 cover the current shape:
   the trajectory/wave rename, the loss zoo + stress matrix, the GpuArbiter,
   the multi-tenant Learner, the Host, the operational CLI + observe/, the
   flow graph + dictionary.json, the #38 loss-purity ruling, the scoring
   verb + real opsd, the observer UI + custom panels, #43: the fleet —
   hosts as atomic partitions, join/carve/acquire, the wire — #44: the
   trainer's punica, #45: TP/transport/FSDP on metal, #46: soft prompts +
   the side_attention refusal, #47: real OPD, #48: the Lowering — one
   contract per (adapter type, side) — #54: ARCHITECTURE.md + the docstring
   recode, and #55: the vocabulary rename). If code and an early entry
   disagree, the code plus the latest entry win.
4. **agent-context/rl-stack-spec.md** — the spec canon, v3 (folded through
   #43). Invariants I1–I12. Deltas after the fold-in live in CONTEXT.md
   (incl. #55's terminology: the spec text still says gpuset/kind/llm in
   places — swap at the v4 fold).

## Working norms (Samarth's, stated across sessions)

- Ultra-readable code beats clever code; spec vocabulary goes in executable
  positions. Docstrings state the rule a thing enforces.
- One named function/method per rule; typed records, no meta-dict bags, no
  duck-typing (we own both sides of every interface).
- Spec-shape changes get a new numbered entry in agent-context/CONTEXT.md —
  and fold into rl-stack-spec.md when they change an invariant.
- The fakes suite must stay green: `python3 -m unittest discover -s tests`
  (Python ≥ 3.11 — use python3.13 locally; stdlib-only, torch/vllm/modal
  lazy). Resume-equivalence (`tests/test_resume.py`) is byte-identical run
  dirs — protect it.

## State at handover

- THE VOCABULARY IS RENAMED (#54/#55): ARCHITECTURE.md is the universal
  reference and docstrings speak it. "Kind" (for adapters) is dead — the
  registered class is AdapterType (`@adapter_type`, ADAPTER_TYPES,
  AdapterSpec.adapter_type); Regime/Demand say `capability`;
  Partition.gpuset → `.metal`; the PoolClient param is `client` (not llm);
  TokenBatch.post → `.postdata`; VllmEngine takes max_bundles/max_rank.
  The identity move is ACCEPTED: registry strings + class sources hash
  into run_id, so pre-rename stores are read-only history (the observer
  tolerates old journal keys; one regression test pins that).
- 504 tests green on fakes (torch-gated skips run in the image). Real
  metal is PROVEN through the stress matrix (deploy/stress_l4.py): seven
  concurrent tenants — grpo/ppo/gspo/sft/sdft/replay_distill/self_anchor,
  live + replay + static sources, a judge pool, lag=2 — on one Modal L4
  with ONE shared engine and ONE shared multi-tenant learner, plus
  sleep-sharing kill/resume and cross-container resume, all green. Image
  pinned: vllm 0.28.0 / torch 2.13.0 / transformers 5.16.1. The observer
  UI is deployed (…--rlstack-ui.modal.run), verified live.
- The fleet (#43, I12) is METAL-PROVEN through #45/#47: hosts are ATOMIC
  PURPOSED PARTITIONS (Partition + Regimes, attested at birth, never
  reshaped; >1 regime alternates on the host's own arbiter group);
  sharding is a build fact (Engine.tp / Learner.fsdp, shape-matched at
  submit); runner/fleet.py climbs join → carve (journaled) → acquire
  (human); runner/remote.py is the wire (HostService: admission at the
  serving host; RemotePool: full Engine protocol, byte-identical to local
  on fakes). TP=2/4 inference, score_tokens, the Modal-cls transport
  (deploy/modal_host.py), and FsdpTorchLearner (fsdp=2, kill/resume,
  width-free sealed bytes) are all proven on L4 metal. The learner is
  never remote — the runner goes to it.
- The trainer batches (#44): additive install + row routing (the
  trainer-side punica; rlstack/policy/adapters/replay.py is the seam),
  16/16 metal parity incl. two tenants' deltas in one forward; swap-install
  is gone.
- THE MILESTONE END TEST IS GREEN (#47): true OPD — `opd` is sampled-token
  reverse KL (score-function gradient) over a teacher_logprobs token_level
  column scored by a LIVE Qwen3-32B tp=4 teacher host, student inference
  tp=2, learner fsdp=2, three containers over ModalTransport (run
  c0f65f24362b: KL 0.356→0.285 nats over 4 updates, gap at the kernel
  floor). The old replay-matching loss is renamed replay_distill.
- Experiments are tenants submitted to hosts, each with its own run store
  (one experiment, one store, for life). The GpuArbiter owns admission;
  leases are gone. The observer (rlstack/observe/, `python -m rlstack
  {hosts,runs,gpu,ui}`) reads journals + peeks only — never experiment
  content; the UI renders each run from its own dictionary.json.
- The loss is pure math (#38): requires names data columns only; post
  processors produce everything else (token_level = per-token channel; a
  processor may score through any declared pool, cross-base included).
- SUB-GPU HOSTS ARE METAL-PROVEN (#51/#52, deploy/partition_l4.py): many
  fractional partitions coexist on one device (vLLM budgets against device
  total, so partitions compose additively), real sleep alternation hands
  HBM back (~9.9 GiB on an L4), joins are fraction-free, refusals correct,
  a failed carve leaves neighbors serving. Partitions carry their gpu kind
  (#49, fraction_for_gb is the one GB↔fraction meeting point); carve names
  are unique + journal-safe, factories receive the Partition, VllmEngine
  has an honest sleep seam (#52). Adapter lowerings are ONE contract per
  (kind, side) (#48: demands/attach/apply/align + reaches; vllm_engine.py
  is a mechanism-blind bus). The observer has host pages + hover + the
  open metrics slot (#50). Evaluator samples concurrently with
  order-independent bytes; rank teardown is bounded (#53).

## Quick commands

```
python3.13 -m unittest discover -s tests        # fakes suite (~1s)
modal run deploy/modal_app.py::run_tests        # same suite inside the image
modal run deploy/modal_app.py::run_arith        # small real run via the Host
modal run deploy/modal_app.py::hosts            # observer views on the volume
modal deploy deploy/modal_app.py                # + the observer UI beside the
                                                #   volume (…--rlstack-ui.modal.run)
python3.13 -m rlstack ui <store-root>           # the same UI over a local store
modal run deploy/stress_l4.py                   # the full stress matrix (~1h)
modal run deploy/opd_l4.py                      # OPD 8B←32B, three hosts (~15m)
modal run deploy/fsdp_l4.py                     # the FSDP ladder on 2xL4
```

## Known-open work (deliberate, logged)

- The async scorer daemon: teacher scoring currently rides the Trainer's
  post phase INLINE (sequential 32B prefills block each gradient); the
  scorer is a store-synced daemon that writes postdata ahead of the
  trainer (design settled in conversation + CONTEXT; version-pinning rule:
  post traffic to the POLICY pool pins the wave's recorded policy_version).
  Also un-batched/un-cached teacher scoring, and the notify-deletion TODO.
- Multi-GPU leftovers (#45/#47): host linger/GC back to residual; a
  measured join-refusal signal; the cross-tenant training coalescer (#44
  designed it); streamed sample replies + a local tokenizer beside
  RemotePool; kill/resume of the three-host OPD run (mechanism proven,
  wire untested); the trainer-side cross-tenant batching determinism rule.
- Adapter unruns (#46): soft prompt at tp>1; a soft-prompt tenant over the
  RemotePool wire; an opsd tenant through score_tokens-under-soft-prompt.
  Open ruling for Samarth: per-adapter-type lr scaling (a type's sensible lr ~
  1/sqrt(param count); a mixed-kind bank with empty OptimSpec.overrides is
  arguably a validate warning — the soft-prompt collapse at lr=1e-2 is the
  evidence, CONTEXT #46).
- Parity certificates designed (#25, rlstack_engine/certificates.py) but
  unwired — logprob_gap is the running alarm.
- side_attention rollout half: NOT PROVEN, honestly blocked — vllm 0.28.0
  has no LSE seam on the dense FlashAttention path (#46; the plugin's probe
  names the missing symbols; reachability reports NONE). Replay half IS
  proven (4-D attention mask). Unblocks: FlexAttention score_mod, or a
  vllm bump that plumbs return_softmax_lse.
- Async post daemon ("scorer"), pool-annotated flow graph, eval `terminal`
  bit, S3Store, generation-only runs (algo=None: needs a committing Sealer
  daemon + wave-shape knobs out of Schedule) — designed in CONTEXT, not
  built. The UI (observe/ui.py + page.py + host_series.py, #50): graphs
  with loss-walkback priority, hover raw values, host pages off the
  journals (fleet placement timeline, per-device gpu series, the
  schema-tolerant metrics slot — nothing emits into it yet), run
  dropdown. Named next: distributions, token drill-down, cross-run curve
  comparison, host throughput emission (design in #50).
- Open threads listed at the foot of CONTEXT.md (identity rings, schedule
  split, Wave/ArchiveContext typing).
