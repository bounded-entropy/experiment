# ADR 0010 — Run directories are explicit

| | |
|---|---|
| **Date** | 2026-09-11 |
| **Status** | Implemented |
| **Author** | Codex with Samarth |
| **Touches** | `data/stores/`, `runner/`, `observe/`, `deploy/` |
| **Invariants** | I3 identity, I10 recovery and committed history |
| **CONTEXT** | Entry 89; supersedes entry 67's implicit directory discovery |

## Original prompt

> "Run `R` originally lives in `runs/books/R`.
>
> 1. Someone resubmits the same experiment with `subdir="membership"`.
> 2. The API promises to resume `runs/books/R`.
> 3. Checking only `runs/membership/R` would miss it and could create a second copy." ok yea we should not support this honestly. it should be assumed that the caller lists the correct subdirectory i think. what are your thoughts on this?

> sure that sounds good. could you make these changes? then, restart the desk + all related services, tear down all metal, etc.. after youre done, could you let the chat [@Design adapter scaling experiments](thread://01a09255-b009-7d50-9f5d-3c25afde9fdd?hostId=local) you're done, and then ask it to start up its experiments?

## Context / problem

`Store.run_prefix` discovers every run on an uncached ID. `ModalVolumeStore`
recursively lists checkpoint contents too: one measured request listed 185,775
entries for 783 manifests in 22.53 seconds. New-run creation can repeat this
discovery before writing its manifest. The caller already supplies its folder.

## Decision

A run reference is its path relative to `runs/`: `subdir/run_id`, or `run_id`
at the root. The supplied subdirectory is authoritative. Opening and peeking
never discovers a different location. Existing-run requests fail at a missing
path; creation explicitly offers a manifest. An explicit resume-only option
also rejects missing paths when a runner offers a manifest for validation.
Submitting again at the same path retains idempotent create-or-resume behavior.
An omitted subdirectory means root, not a wildcard. Parent and replay URIs,
recovery, measurement and UI callers carry the full run reference.

### Touched / untouched

- **Touched:** common store location handling, runner address propagation,
  observer links and explicit discovery, regression tests and documentation.
- **Touched:** desk deployment recovery generation. Only explicit submissions
  in the new generation may opt into automatic recovery after the authorized
  clean restart. Historical parked work remains stored and inspectable.
- **Untouched:** adapter mathematics, scientific run hashes, checkpoint bytes,
  existing directories, and the single-writer/checkpoint commit discipline.

### Promises / non-promises

- Opening a known path and creating a new run perform no global listing.
- A wrong resume path never redirects or creates a run.
- Browsing may explicitly enumerate runs; this cannot prime hidden lookup state.
- A clean restart retains artifacts but does not automatically revive the old
  parked queue. Automatic GPU idle shutdown remains finite.
- No global uniqueness search across folders, store migration, or claim that
  all remaining storage latency disappears.

### Interfaces and sketches

```python
store.run_prefix("family/run_id")  # runs/family/run_id; pure path resolution
store.open_run("run_id", subdir="family")  # existing only, unless manifest offered
store.open_run("run_id", manifest=manifest, subdir="family", create=False)
WarmStart("store://family/run_id@16")
```

## Questions and authorization

1. **Is the caller's directory authoritative, including an omitted root?**
   Accepted by the quoted request and approval of the preceding concrete rules.
2. **May ordinary lookups search other directories?** No, per the same request;
   explicit browsing remains available for historical artifacts.
3. **Do IDs or old committed bytes change?** No. Location is propagated outside
   content identity. Existing resume-equivalence checks remain required.
4. **Can old parked work restart during the service reset?** No. The user asked
   to tear down all metal and hand new experiments to the referenced task; the
   deployment uses a new recovery generation and keeps idle shutdown enabled.

These decisions were approved in the conversation before implementation; no
additional approval or commit was requested. The existing staged work is kept.

## Outcome

Implemented in the shared store, runner, desk/host wire, checkpoint and replay
readers, and observer. Submission receipts carry `run_ref`; observer links
carry `subdir` separately from the store-root selector. The synchronous runner
also now forwards its supplied directory. No historical run data was moved.

Validation: the local suite completed 1,316 tests with 192 dependency/GPU
skips. The deployment image completed the same 1,316 tests on CPU with four
skips. Five browser address checks passed. Tests exercise zero global
enumerations on ordinary remote creation/missing resume, qualified parent and
replay reads, exact resume with unchanged committed ledger bytes, duplicate
basename browsing, and exclusion of historical parked work after reset.

Deployment-specific restart receipts and live measurement logs are kept
outside the framework repository. The deployment accepts an operator-supplied
recovery generation; the library default remains backward compatible.
