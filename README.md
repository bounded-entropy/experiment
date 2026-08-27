# rlstack (Phase A)

Thin RL wrapper. This phase is CPU-only: specs (identity), registries
(declaration + pure compute), Phase-0 validation, data objects, and the store.
Canon: SPEC.md (mirrored from rl-stack-spec.md). Tests: `pytest`.

Layout:
  rlstack/canonical.py   canonical serialization + content hashing + run_id
  rlstack/specs.py       all frozen spec dataclasses (SPEC.md section 2A)
  rlstack/registries.py  @adapter_kind/@env/@reward/@advantage/@loss + source hashes
  rlstack/siteschema.py  SiteSchema interface + FakeSiteSchema (interim grammar)
  rlstack/validate.py    Phase-0 joint validation
  rlstack/data.py        Trajectory/Turn/Wave/TokenBatch/packing (numpy, no torch)
  rlstack/store.py       runs/<id>/ store: manifest, ledger, rollouts, cas, resume
  tests/                 unit tests (CPU, fast, no network)
