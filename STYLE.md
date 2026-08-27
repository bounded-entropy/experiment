# rlstack style

The one reader is Samarth, six months from now, mid-experiment. Optimize for
"open the file, understand it immediately". Concretely:

1. **The spec's vocabulary goes in executable positions, not comments.** A file
   should read as its SPEC.md section executing: `check_sleep_implies_zero_lag`,
   `traj.seal()`, `code_hashes(spec)` — not generic machinery annotated with
   spec references.
2. **Concrete types, no duck-typing.** We own both sides of every interface;
   `isinstance(member, LearnerMember)` over `hasattr(member, "fsdp")`, always.
   No reflection where naming five fields would do.
3. **One named function per rule.** Validation checks, recovery steps, packing
   stages: each gets a name and a docstring stating the rule it enforces, and a
   table (like `validate.CHECKS`) lists them in spec order.
4. **No defensive hardening for hostile strangers.** No freezing ceremony where
   hash-time canonicalization already guarantees identity; no clever exception
   hierarchies; no pickling/`__slots__`/`python -O` contingencies. Enforce real
   invariants (the seal, the ledger) in the simplest way that raises loudly.
5. **Tests in two layers.** `test_examples.py` transcribes SPEC.md §3 and must
   stay readable as documentation; the per-module suites are the regression
   wall and may be exhaustive.
6. **Declarations are typed records** (`LossDef.requires`, not
   `meta["requires"]`). If Phase-0 reads it, it has a field name.
7. Stdlib-only in Phase A; every deferred dependency (numpy, safetensors,
   parquet) sits behind a surface that does not change when it arrives.
8. **The folders are the architecture, and the import graph enforces it**
   (tests/test_architecture.py):

   ```
   spec/           the contract — declarative values, identity, submit gate
   registry.py     the index: name -> typed record; registrations live elsewhere
   client.py       the sampling interface (SampleClient Protocol, pool(name))
                   — neutral ground both worlds may type against
   policy/         the bridge (I2): sites + adapters/ (one file each) + compile
   inference/      the sampling world: Rollout + environments/ (an episode is
                   everything needed to complete and seal a rollout — scoring
                   is NOT here) — never imports training/
   training/       the gradient world: post/ (the pipeline — rewards, judges,
                   advantages are all PostProcessors, one file each) + losses
                   — never imports inference/
   data/           the membrane (trajectory→group→wave, flatten+pack, stores/
                   — Store ABC + one backend per file)
                   — imports no other rlstack package; dumb and estimator-free
   runner/         the substrate: Engine/Learner protocols, the blackboard
                   (signals.py: awaitable store predicates; lease.py: who may
                   occupy the metal), the loop (phases 0-1 + plan_daemons),
                   fakes; the one package allowed to import both worlds
   runner/daemons/ one daemon per GPU responsibility (generator / trainer /
                   evaluator), synchronized ONLY via the store; each daemon's
                   acquisition condition is a named, overridable method
   runner/sources/ where training data comes from — one base (WaveFeed:
                   rows exist in the run's own waves/, or not yet),
                   one file per option (live / replay / static)
   runner/engines/ real inference metal, one file per engine (vllm_engine);
   runner/learners/ real training metal (torch_learner) — both import their
                   heavy deps at module scope and are therefore imported
                   LAZILY, never from the package root (rule 7)
   deploy/         deployment only (I5): images, volumes, venue wiring —
                   nothing semantics-bearing lives here
   rlstack_engine/ (sibling package) code that ships in the ENGINE image:
                   the EnginePlugin contract, BatchView (the one version-
                   pinned metadata shim), slots, certificates. Imports are
                   one-way: it may import rlstack; rlstack names plugins by
                   string only and never imports back.
   ```

   New code goes in the region whose rule it obeys; if it fits none, that is a
   design question, not a filing question.

Agents contributing code follow this file; deviations are review findings.
