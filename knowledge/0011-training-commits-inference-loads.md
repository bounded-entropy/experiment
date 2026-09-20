# ADR 0011 — Training commits; inference loads

| | |
|---|---|
| **Date** | 2026-09-13 |
| **Status** | Implemented — design approved in the task before implementation |
| **Author** | Codex with Samarth |
| **Touches** | `runner/`, `spec/validate.py`, tests and architecture documentation |
| **Invariants** | I1 (store boundary), I8 (version pins), I10 (ledger commit) |
| **CONTEXT** | Extends ADR 0006 / CONTEXT #80; implemented in CONTEXT #90 |

## Original prompt

> is the trainer publishing bundles via RPC? hopefully not right? i was hoping that a separate inference daemon would be awaiting the bundle commit to the FS, and then adding the bundle to the inference engine.

> ok let's work in a separate worktree and resolve this. repeat back what youll implement so i can ensure we're on the same page

> no, i was not intending that to be a different daemon. your design is good. please implement and write a small ADR in that branch

## Context / problem

`Trainer.run_forever` writes checkpoint blobs, calls `engine.add_bundle`, then
appends the ledger. It also uses engine tokenization and route restoration.
`loop.py` requires `main` even when `needs_of` declares only a learner.
An inference failure can therefore prevent a replay update from committing.
This is observed in code; GPU failure latency is unmeasured.

## Decision

The Trainer writes adapter and optimizer blobs before the ledger commit and
never calls an inference engine. Injected text is tokenized by the learner
using the same base tokenizer convention as inference (no special tokens),
with the Trainer's existing bounded cache. The existing Generator and Scorer
load their required version from storage and install it on their own pools
before inference. There is no new watcher daemon. Pool requirements come from
actual generation and scoring traffic; no traffic means no engine requirement.

### Interfaces and scope

- Add `Learner.tokenize(tenant, text)` through the existing learner transport;
  Torch/FSDP and fake learners implement it. Generated tokens remain verbatim.
- Remove engine, routes and pool admission from the Trainer constructor.
- Scope inference routes to each daemon's declared pools. Initial and restored
  bundles are installed on demand by inference, never by training setup.
- Keep storage layout, ledger fields, version-selection rules, loss math,
  measurement logging, observer, and deployment configuration unchanged.
- Preserve fake-run bytes and resume equivalence for unchanged specifications.
  Removing an unused topology member still changes spec identity as before.
- No GPU deployment, new retries, independent daemon supervision, or proof of
  live shared-volume visibility is included. Existing run-wide failure handling
  remains: a fatal daemon exception can still terminate its sibling tasks.

```python
Learner.tokenize(tenant: str, text: str) -> tuple[int, ...]
Trainer(..., learner: Learner, initial_version: dict, ...)
routes_at(current: Bundle, *, pools: tuple[str, ...]) -> Routes
```

## Resolved questions

1. **Separate watcher or existing inference daemons?** Existing Generator and
   Scorer. Samarth explicitly confirmed this above.
2. **Engine installation before or after commit?** After commit, owned by
   inference. This is the design Samarth approved. Uncommitted blobs are never
   selected from the ledger; an interrupted update is discarded on resume.
   Initial version zero remains the explicit bootstrap before the first update.
3. **Newest version or exact pin?** Keep existing semantics: generation selects
   the permitted committed version; scoring restores its recorded version.
   Restart/eviction reconstructs the same content-addressed bundle from storage.
   Installing an immutable bundle twice is safe. Training does not wait for it.

## Outcome

Implemented the boundary above, including lazy default-pool resolution for
pool-less postprocessors and restoration of evicted non-policy base bundles.
The old test that forced inference back into the Trainer now compares stored
split-pipeline columns with an unsplit reference pipeline outside training.

Validation: `python3.13 -m unittest discover -s tests` completed 1,399 tests in
28.247 seconds: 1,207 passed, 192 dependency/GPU cases skipped. Eight new
regressions cover a learner-only host with a routed learner, unavailable unused
engines, generation delayed until training completes, uncommitted blobs,
engine installation failure and restart, interrupted learner-only commits,
judge-only replay with base eviction, missing main scoring declarations, and
historical bundle restoration. Existing learner transport tests cover the new
tokenization verb through resident and host doors.

Compared four unchanged specifications against worktree baseline `e5739ba`:
online (23 files), replay (18), judge scoring (27), and policy scoring (27).
Every run identity and every stored file was byte-identical. No GPU deployment
or live shared-volume drill was run; real tokenizer parity and FSDP execution
remain unverified on metal. Workers must include the new learner verb before
running this branch. CONTEXT #90 records the completed change.
