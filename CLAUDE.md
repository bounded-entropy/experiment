# rlstack — session handover

Samarth's personal high-throughput RL-for-LLMs harness ("the thin wrapper").
Designed and built across long Claude sessions; everything you need to
continue is in the repo.

## Read before writing code

1. **STYLE.md** — binding. Eight rules; rule 8's folder tree IS the
   architecture, enforced by `tests/test_architecture.py`. Deviations are
   review findings.
2. **agent-context/CONTEXT.md** — the decision log. Chronological, numbered;
   later entries supersede earlier ones (#28–#43 cover the current shape:
   the trajectory/wave rename, the loss zoo + stress matrix, the GpuArbiter,
   the multi-tenant Learner, the Host, the operational CLI + observe/, the
   flow graph + dictionary.json, the #38 loss-purity ruling, the scoring
   verb + real opsd, the observer UI + custom panels, and #43: the fleet —
   hosts as atomic partitions, join/carve/acquire, the wire). If code
   and an early entry disagree, the code plus the latest entry win.
3. **agent-context/rl-stack-spec.md** — the spec canon, v3 (folded through
   #43). Invariants I1–I12. Deltas after the fold-in live in CONTEXT.md.

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

- 401 tests green on fakes. Real metal is PROVEN through the stress
  matrix (deploy/stress_l4.py): seven concurrent tenants — grpo/ppo/gspo/
  sft/sdft/opd/self_anchor, live + replay + static sources, a judge pool,
  lag=2 — on one Modal L4 with ONE shared engine and ONE shared
  multi-tenant learner, plus sleep-sharing kill/resume and cross-container
  resume, all green. An eighth tenant (opsd: hinted scoring through
  VllmEngine.score_tokens) is in the harness but NOT yet executed on metal
  — first-contact expected in the prompt_logprobs indexing. Image pinned:
  vllm 0.28.0 / torch 2.13.0 / transformers 5.16.1. The observer UI is
  deployed (…--rlstack-ui.modal.run), verified live on a 30-step GRPO run.
- The fleet (#43, I12; fakes-proven, real TP/FSDP metal is the next layer):
  a Host is an ATOMIC PURPOSED PARTITION — Partition (gpuset, devices,
  memory fraction) + Regimes (kind × base × shape), attested at birth,
  never reshaped; >1 regime alternates on the host's own arbiter group.
  Sharding is a build fact (Engine.tp / Learner.fsdp, shape-matched at
  submit). runner/fleet.py climbs join (automatic, fraction-free) → carve
  (automatic from residual, journaled in fleet/log.jsonl) → acquire
  (human). runner/remote.py is the wire: HostService (admission at the
  serving host) + LocalTransport (json-round-trips every frame) +
  RemotePool (full Engine protocol; a remote main pool is byte-identical
  to local). The learner is never remote — the runner goes to it. The
  Modal-cls transport and the OPD 8B←32B e2e are NOT yet built.
- Experiments are tenants submitted to hosts, each with its own run store
  (one experiment, one store, for life). The GpuArbiter owns admission;
  leases are gone. The observer (rlstack/observe/, `python -m rlstack
  {hosts,runs,gpu,ui}`) reads journals + peeks only — never experiment
  content; the UI renders each run from its own dictionary.json.
- The loss is pure math (#38): requires names data columns only; post
  processors produce everything else (token_level = per-token channel).
- TorchLearner batches the forward (one padded pass per microbatch,
  numerics identical to the per-doc form) — written for the FSDP milestone,
  not yet run on metal.

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
```

## Known-open work (deliberate, logged)

- The multi-GPU milestone's remaining layers (#43): real TP engines + FSDP
  learner processes; the Modal-cls transport + per-capability host deploy;
  host linger/GC back to residual; a measured join-refusal signal; the end
  test — OPD distilling Qwen 8B from Qwen 32B on L4 hosts (teacher tp-4,
  student inference tp-2, learner fsdp-2), shard configs flipped as host
  build facts.
- Parity certificates designed (#25, rlstack_engine/certificates.py) but
  unwired — logprob_gap is the running alarm. side_attention numerics are B3+.
- Async post daemon ("scorer"), pool-annotated flow graph, eval `terminal`
  bit, S3Store, generation-only runs (algo=None: needs a committing Sealer
  daemon + wave-shape knobs out of Schedule) — designed in CONTEXT, not
  built. The UI (observe/ui.py) exists: graphs with loss-walkback priority;
  distributions, token drill-down, gpu/host pages are the named next
  features.
- Open threads listed at the foot of CONTEXT.md (identity rings, schedule
  split, Wave/ArchiveContext typing).
