# Teaching-chat integration inventory

Source: private teaching experiment branch `codex/fact-writing-references`,
`65c57f6b0bcb657cf35a61a65e6173f2e99740d8` → `e19f1ccf60fc95761112d6c681589aa13b656b17`. This is 98 commits and
334 changed paths. The source working tree has additional uncommitted operational
files; these are deliberately left untouched and are not represented as merged.

The common framework is integrated on top of `b67bb58` (shared adapter definitions
and adapter-neutral checkpoint/publication services). It is not replaced with the
experiment branch's older framework snapshot.

## Framework changes

| Source path under `research/dreams-impl/` | Integration disposition |
| --- | --- |
| `rlstack/client.py` | Typed gradient rows, document sufficient statistics and comparison records; measurement operation moves to the generic training client. |
| `rlstack/runner/interfaces.py` | Generic learner gradient comparison verb. |
| `rlstack/runner/learners/torch_learner.py` | Exact selected-token, equal-document supervised gradients, read-only state boundary, optional fixed-state cache, and no backward on fully frozen replay. |
| `rlstack/runner/learners/gradient_cache.py` | Opt-in bounded reference-gradient and scalar-comparison cache; invalidated on state mutation, sleep and uninstall. |
| `rlstack/runner/learners/fsdp_torch.py` | Explicit refusal of unsupported sharded gradient comparisons. |
| `rlstack/runner/fakes.py` | Explicitly synthetic measurements for protocol tests, with no optimizer mutation. |
| `rlstack/runner/remote.py` | Scalar-only records and nested document groups through existing resident and host learner doors. |
| `rlstack/runner/venues/client.py` | Existing bearer authenticates observer reads; bounded follow retries an exact-run 404 and reports failed/stopped dispositions. |
| `rlstack/policy/adapters/dream_bank_torch.py` | Short-corpus thin-SVD calibration and correction preventing residual covariance from being counted twice; shared tensor state export is retained. |
| `rlstack/runner/fit.py` | Its new exact-document, provenance and cancellation helpers move to `runner/gradient_measurement.py`; the old fitting wrapper is replaced by the normal Trainer integration, owned by the enclosing rewrite. No second optimizer loop is copied. |
| `tests/test_gradient_alignment.py` | CPU direct-autograd oracle, reduction, numerical validity, state-isolation and cache tests. |
| `tests/test_gradient_measurement_remote.py` | Existing learner-wire and host-admission tests. |
| `tests/test_short_calibration.py` | Thin/dense covariance equivalence, full-rank reconstruction and transition tests; torch is now an optional test dependency. |
| `tests/test_venue_client_auth.py` | Observer bearer, exact-run registration retry and terminal status tests. |
| `tests/test_gradient_fit_client.py` | Exact-document/provenance/cancellation helper tests move to `test_gradient_documents.py`; temporary-tenant/cache/journal/custody obligations move to the new generic client tests. |
| `knowledge/0026-measure-gradient-alignment-without-changing-student-state.md` | Historical accepted research decision is preserved in the private branch. Its scientific prompt, corpus references and deployment particulars are not copied into this general framework repository. |
| `knowledge/0028-adapters-declare-site-programs-over-routed-token-rows.md` | Its latest definition/lowering design is already implemented by shared base `b67bb58`; retain that accepted implementation rather than replacing it with a proposal. |
| `agent-context/CONTEXT.md` | Research-only history is preserved privately; general integration behavior is recorded in this repository's current context. |

## Private experiment changes

The other 316 changed paths are private experiment scripts, tests, plans, corpus
and analysis artifacts, run/preservation records, and operational tooling. Their
original versions remain in the private integration branch history at source commit
`e19f1ccf60fc95761112d6c681589aa13b656b17` on `codex/shared-trainer-integration`.
Scientific artifacts remain unchanged. Active scripts and tests that called the
retired fitting interfaces are migrated in that private branch to explicit adapter
specifications and bounded ordinary Replay runs. Their contents are not copied
into this general framework repository.

A complete machine-readable inventory of all 98 source commits and all 334 changed
paths, including each disposition, is retained in that private worktree as
`docs/integration-source-inventory.json`. The shared framework is overlaid into its `research/dreams-impl` subtree while
retaining private recipe modules and their registrations. Retired fitting executor
files and obsolete tests are explicitly removed. Active teaching callers now use
the common Trainer; their reports retain every bounded child run identity and
read exact committed probe and training statistics. Final validation and GPU
qualification limits are recorded with the integration evidence.

## Local validation of the imported changes

- Python 3.13: 5 learner-wire tests, 3 observer authentication/follow tests, and 5
  exact-document/provenance/cancellation helper tests pass.
- Python 3.12 with PyTorch 2.14.0 on CPU: 20 real gradient/cache/oracle tests and
  3 calibration tests pass.
- No GPU allocation, deployment, scientific experiment, live NCCL test, or vLLM
  loading/throughput claim accompanies this integration.
