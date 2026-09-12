# Working in rlstack

Read `STYLE.md` and `CLAUDE.md` before changing code. Use `ARCHITECTURE.md`
for interfaces and `knowledge/` for accepted decisions. Explain findings in
plain language, distinguish observations from hypotheses, and state material
uncertainty and counterexamples.

## Experiments go through the desk

- Submit every training or generation experiment through the standing desk,
  using the shared client helpers or `RemoteDesk.submit`. This includes
  screens, controls, replications, retries and recovery. Booting registered
  metal with the shared chassis is allowed; a direct Modal `.spawn()` of a
  training function is not a desk submission.
- Keep deployment files to images, volumes, cards and thin wiring to shared
  APIs. Do not build a private scheduler or recovery host in `deploy/`:
  no hand-built residents, per-experiment semaphores, lifecycle loops or
  hard-coded pool routes that bypass desk placement. Use the existing
  `Host`, resident, measurement and observer interfaces.
- Cache isolation, a timeout, a stalled desk or pressure to keep GPUs busy
  does not authorize a bypass. Diagnose the common path and fix or extend
  it there. Check relevant ADR implementation branches and their validation
  status before duplicating infrastructure. If an exception is necessary,
  explain the exact missing capability to the user before taking it.
- Confirm acceptance, the expected run identity, a desk placement and host
  custody, then actual committed progress. Confirm that the observer lists
  the run. A Modal call ID or a hand-written roster alone is insufficient.
- Resume from saved specs and checkpoints after establishing that the old
  owner has stopped. An observation timeout is not proof of process death.
  Keep requests bounded without cancelling a shared synchronous container.
- Use only `MODAL_PROFILE=yu-masala-workspace` for Modal commands. Respect the
  user's GPU ceiling across all apps, including old deployments and screens.
  Release and verify the old allocation before replacing it at that ceiling.
- Automatic GPU idle shutdown must always remain enabled: the user's funds
  are limited. Never set an infinite/`None` idle limit, pin GPU allocations,
  or use keepalives to bypass idle release, including during setup or recovery.
  Verify finite idle limits after allocation and recovery. Fix premature
  shutdown in the shared lifecycle code instead of disabling shutdown.
- Use the standing observer for progress and shared metal measurement doors
  for evaluation. The observer must remain read-only: discover manifests and
  committed artifacts; never create fake host events or attach a run just to
  make it visible. Missing host evidence means unknown liveness, not running.

## Integration and validation

Run directories are explicit (ADR 0010). Pass the correct `subdir` when
submitting; omitted means the root. Reads, parent checkpoints and replay refs
use `subdir/run_id` (for example `store://family/run_id@16`). They never search
other folders. Use `resume=True` on desk/host submissions to require an existing
run; use `create=False` on `Store.open_run` when offering a manifest for resume
validation. Observer links carry `?subdir=...`; use the returned `run_ref` for
follow-up reads. Browsing/listing runs is an explicit operation, not a lookup
fallback. Never treat a missing exact path as permission to rediscover another.

Preserve staged and unstaged user work separately. Inspect the actual checked
out code and commit ancestry; an accepted ADR document does not prove its
implementation is merged. Run the relevant regression tests and the fakes
suite (`python3.13 -m unittest discover -s tests`). Report skipped GPU tests
and an outstanding live drill explicitly; offline tests do not prove live
deployment behavior. Scientific failures and infrastructure failures must
remain separate in reports, with unsuccessful runs retained.
