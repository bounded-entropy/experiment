# ADR 0014 — The checkpoint is the commit, the wire carries the policy, and stop is a disposition

| | |
|---|---|
| **Date** | 2026-09-13 |
| **Status** | Implemented |
| **Author** | Claude (Fable 5.1) with Samarth |
| **Touches** | `data/stores/` (the checkpoint record, the rewind, retention), `runner/daemons/` (the Trainer's two-step commit), `runner/` (loop, restore, host stop/drain, desk dispositions, idle rule, journals), `runner/venues/` (the shared runtimes, both providers), `runner/transports/` (HTTP + blobs), `observe/` (stopped/failed/checkpointed rows), `deploy/` (operator doors, both venues), `spec/validate.py` (two gate checks) |
| **Invariants** | I5 (cadence and delivery mode are placement-time facts, never identity), I8 (the lag window is immune to eviction), I10 (one run, one store; the rewind is a store rule), plus the resume-equivalence property (`tests/test_resume.py`) generalized from "one update" to "up to `every` updates" |
| **CONTEXT** | Entry 92. Extends #59 (one wave, one update), #65 (the Scorer), #74 (ADR 0002), #76 (ADR 0003), #79/#80 (ADR 0006), #84/#85 (ADR 0007/0008), #89 (ADR 0010), #90 (ADR 0011, adopted), #91 (ADRs 0015–0017, the Strange Loop backend published as this ADR's Part E step 1). |

This ADR is written without line wrapping (ADR 0011's standing request).

## Original prompt

> the goal of this chat will be to significantly improve some components of the architecture of rlstack. we are going to unify the strangeloop changes in here as well, and harden it so that it's very robust.
>
> check out [@Assess continual learning ideas] and [@Design adapter scaling experiments] (both of these are Codex chats). one problem that repeatedly arose is that our updates are very slow because sealing trajectories takes a long time. one reason is because for every checkpoint and optimizer state, we are pushing it to the FS. since there exist many experiments running simultaneously, this is a very large amount of overhead on the filesystem.
>
> if you look in my worktrees (or branches), one thing that you should see is the removal of the necessity of the "main" pool for experiments. this is because i wanted to sync exclusively via the file system. i still want this to be a setting, but one thing I should definitively add is allowing the use of localtransport, pipetransport, and remotetransport to actually transfer bundles between learner and engine (this is already supported), while removing the necessity to checkpoint every step. there should be no default to how often checkpointing is necessary, but it should be a deliberate decision made. resumes should still be just as easy as before. the file system latency has been a major headache, since it's been causing hangs and terminations and all sorts of weird things. if we do spaced-out checkpointing for most of our runs, this should solve this headache. it's also been the case that many codex chats have had to go in and actually ensure runs dont respawn after theyve been terminated. when things are willingly terminated, we should definitely have a separate "stopped" flag, so that the desk doesnt try reviving the runs automatically.
>
> speaking of resume, I've noticed that sometimes, my other chats have had to make custom resume functions to resume runs and stop individual runs. i was under the impression that the desk had a campaign manager associated with it that actually kept track of the specific cpu processes that were running experiments, and it could selectively terminate those. hosts/residents also die if nothing hits them for a while, and the metal should also selectively terminate if nothing hits them for a while, etc. etc. these were all primitives ive thought of since the beginning, but it seems like they got lost in translation some while ago.
>
> also, i want all of these to be supported for strangeloop as well. notably, i have a strangeloop branch here where i spawned some runs last night, and im curious what sorts of limitations my agent encountered, and how we can solve them so running stuff is very smooth

## Context / problem

What is true today, on `origin/main` (5d1abf1) unless noted. Line numbers are that tree's.

### 1. Every update is a checkpoint, and the checkpoint is what is slow

The Trainer's commit protocol (`rlstack/runner/daemons/trainer.py:5-12`, executed in `run_forever`, `:144-214`) writes, per update: the wave, the merged postdata, **one adapter blob and one optimizer blob per trainable entry** (`:195-199`), registers the bundle on the engine (`:202`), appends the ledger line (`:203-210`), sweeps retention, journals the clock. Everything is awaited in sequence through `store_work` (`rlstack/runner/signals.py:19-36`), so update `u+1` starts only after `u`'s bytes are durable. There is no cadence knob anywhere: `retention.py:17-18` deliberately refuses an `every:k` knob because retention is about what may be *forgotten*, and nothing was ever about what must be *written*. `restore_tenant` (`rlstack/runner/restore.py:49-71`) reads the ledger tail's `adapters/<name>@<v>.bin` and `optim/<name>@<v>.bin`, which is the reason the blobs exist at every version: the tail is the only resume point.

On a Modal volume the ledger append is the `volume.commit()` (`rlstack/data/stores/modal_volume.py:83-102`, triggered by `_append_line` on `ledger.jsonl`), and a commit persists everything staged, so the 289 MB of blobs the update staged ride that one RPC. Measured, from the Codex threads and the Strange Loop notes:

| where | per update | of which "seal" (blobs + register + ledger + sweep) | source |
|---|---|---|---|
| Modal, continual-latents, one learner | 51–66 s | 19–38 s (median 31 s of 58 s); train 5–17 s | thread A, 2026-09-13 19:04 UTC, `hosts/<host>/log.jsonl` |
| Modal, six-family Q0 fits, ~140 runs on three L4s | median 389 s | 151 s "saving checkpoints / registering / committing"; some updates 15–16 min in that phase | thread B, 2026-09-13 00:27 UTC |
| Strange Loop, lamp reader | 22–35 s seal against 4.2 s train | `research/lamp-prefix-2026-09-13/STRANGELOOP-CHANGES.md` finding 7 |
| Strange Loop, streaming Q0, tenants joining | seal 32 → 59 → 51 → 108 s, train 3 s | `research/streaming-q0-strangeloop-2026-09-13/README.md` finding 2 |
| Strange Loop, same experiment on a raw lease with no per-update publication | under ten minutes per run | lamp finding 12 |

The bytes: a Q0 fit writes "about 96 MB of adapter state plus 193 MB of optimizer state, after training on just one question", and the 96 MB is the *frozen* shared decoder re-emitted at every version (thread A). The Codex agent's own reading, twice corrected, is the right one: the seal timer is not `volume.commit()` alone but everything after the gradient, and every one of those steps shares one worker thread per store (`modal_volume.py:46-55`) — so the ledger commit, the retention sweep's second commit, and the host journal's commit *per event* queue behind each other and behind every other tenant's. Thread B's 15-minute phases and the false "stalled" verdicts (the watchdog's `kill_stalled` journals through the same queue before it kills, `rlstack/runner/host.py:848-870`) are the same queue seen from two sides.

### 2. The bundle already crosses the wire; the store is only the backstop

`Trainer.run_forever` pushes each new bundle with `engine.add_bundle` (`trainer.py:202`), payload bytes included (`rlstack/runner/remote.py:157-163`, base64 in one frame). Consumers pin off the ledger (`Generator.newest_bundle`, `generator.py:117-123`; `Scorer.pinned_bundle`, `scorer.py:103-139`) and ask the engine; only on a miss does `restore_bundle_on` (`restore.py:29-46`) read blobs back from the store. Four transports exist and all carry it: `LocalTransport` (`remote.py:816`), `PipeTransport` (`residents.py:487`), `ModalClsTransport` (`transports/modal_cls.py:29`), and on the Strange Loop worktree `HttpTransport` with a verified blob spool for frames past 1 MiB (`transports/http.py`, `http_blobs.py`, unpublished ADR 0012). So the wire path Samarth asks for is not new; what is new is that it stops being *backed* by a blob at every version.

The `main` pool is still mandatory: `loop.py` raises `no 'main' engine pool` for a learner-only replay run, which cost thread A a cancelled family and a republished plan (`streaming-access-v1` → `-v2`). The unpublished ADR 0011 on `codex/trainer-store-boundary` (`70b8c32`) removes that check, moves `tokenize` onto the Learner, and takes the Trainer off the engine entirely so the store is the *only* channel. That branch is the "sync exclusively via the file system" experiment; it is correct for what it wanted and is the opposite default from what this ADR wants, so it becomes one of two modes rather than being merged as-is.

### 3. There is no "stopped"; parked is a queue, and the queue is retried on every registration

The fleet journal's dispositions are `parked`, `place`, `submit-intent`, `submit-missed`, `submit-replayed`, `release`, `reaped`/`delist`, `knock-refused`, `carve-refused`, `migrate`, `metal`, `recipe`. `Desk.parked()` (`desk.py:1976`) is the retry queue; `retry_parked` (`:2001`) runs on every reap and on every `metal` registration (`:2334`). `Host.stop` (`host.py:567`) cancels the adoption and rosters it **`failed`**. `Desk.stop_anchored` (`desk.py:1623`) is reachable only through `reroute`/`decommission`; `Desk.serve` (`:2280`) has no `stop` verb, `RemoteDesk` has no `stop` method (lamp finding 8), `deploy/desk.py` has no `stop` door. The Codex agent's summary from thread B is exact:

> A worker's `stop(run_id)` interrupted that experiment and preserved its checkpoints. The desk still remembered its unfinished work. If the worker disappeared later, automatic recovery could restart it. 'Parking' an experiment also meant waiting to retry, not cancelling it.

Consequences on the record: thread B's rules-stated runs were revived by the recovery queue after being cancelled, and it shipped a per-subdir "cancellation guard" to the CPU desk from an isolated source copy; thread A wrote `cancel_subdir`, `rollout.py pause/resume`, and `experiment.py resume --arm --avoiding`; both threads ended by writing `STOP_REQUESTED.json` sentinel files because nothing in the desk expresses "stopped". The in-flight working tree's `RECOVERY_GENERATION` and `BOOTABLE_METALS` (ADR 0010's deployment) are the same need met by a deployment constant. On Strange Loop, a tenancy dying of CUDA OOM is journaled `parked` (lamp finding 15), and a pod vanishing mid-run left `chain.sh` polling `done` for six hours with `automatic_recovery: false` (finding 14).

### 4. The idle rule cannot see a dependent, and there is no host-level clock

`listing_busy` (`desk.py:2066-2094`) says yes for a running tenancy, work in flight, or a moved `admitted` counter — all read off the listing's *own* status frame. A pool host whose only client is a training run anchored on *another* host, between waves, shows none of the three, and `release_idle` releases it `force=True` by ruling (`:2121-2136`). Thread A's 06:14 UTC outage was exactly this: "the desk released an 'idle' inference worker while training jobs on another worker still depended on it. Their next calls then failed." The fix was made in an isolated copy and never reached main. The only clock is the metal's (`IDLE_S`, `idle_since` per metal); a carved host on a shared A100 whose runs are all done stays carved until the whole metal goes idle. A registered metal with no listings is idle from its first tick, so an operator has under five minutes between `up` and the first submission (lamp finding 17).

### 5. Strange Loop is a second copy of the lifecycle, unpublished

`codex/strangeloop-backend` has three committed docs commits; the backend itself is 178 dirty paths in that worktree: `stores/strangeloop.py` (552 lines, mount + API, publish = `sync /scratch` + read-back verify per mutation), `runner/venues/{runtime,provider,client}.py` + `venues/{modal,strangeloop}/{provider,desk,worker}.py` (ADR 0013: both providers on one `DeskRuntime`/`MetalRuntime`), `transports/http.py` + `http_blobs.py` (ADR 0012), `exporters/wandb.py`, `deploy/strangeloop.py`. That shared runtime is the right place for everything in this ADR to land once, which is why unification comes first in the plan below. Two ADRs carry the number 0011 (`trainer-store-boundary`: training commits / inference loads; `strangeloop-backend`: the scratch store); the Strange Loop set is renumbered at publication.

## Decision

A run's **commit** stays the ledger line — one append per update, cheap, the observer's and the Generator's truth. A run's **checkpoint** becomes a separate, deliberately spaced durable point: the blobs for the current version plus one line in `checkpoints.jsonl`. **Attach rewinds to the last checkpoint**, not to the ledger tail: ledger lines, waves, postdata and rollouts pinned past the checkpoint are unsealed and regenerate, exactly as one update does today. The cadence is a **submit-time declaration with no default** (`Checkpointing(every=…)`), carried on the frame beside `subdir` and `resume`, never in the spec (I5). **A run need not declare a pool at all**: a learner alone is complete, and its Trainer touches no engine. Where a pool serves the policy, the policy reaches it **over the wire by default** — the `add_bundle` push that exists today, on whichever transport the placement chose — and a non-default **`store` delivery mode** keeps the file-system-only path as a setting, which the gate binds to `every=1`. A deliberate stop **drains**: the Trainer finishes its update, checkpoints, and exits, so stopping loses nothing. `stopped` and `failed` become journaled **dispositions** that outrank `parked`; only a wire death parks, and only a park is retried. The idle rule counts a routed-through dependent as busy, hosts get their own clock, and journals stop persisting per event. All of it lands once, in the shared venue runtimes, for Modal and Strange Loop alike.

### Part A — Checkpointing as a declaration; the rewind

```
runs/<subdir>/<run_id>/
  ledger.jsonl        the commit record: one line per update, as today            (append-only)
  checkpoints.jsonl   the durable points: {"update": u, "versions": {...}}          (append-only, NEW)
  adapters/<n>@<v>.bin, optim/<n>@<v>.bin   exist for CHECKPOINTED versions only
```

- `Checkpointing(every: int)` — checkpoint at every update `u` with `u % every == 0`, **always** at the extent's last update, and **always** on drain. `every=1` is today's behavior to the byte, plus one file. No default anywhere: `Campaigns.submit`, `RemoteDesk.submit`, the venue clients and `run_experiment` all require it; the gate refuses `every < 1`.
- `RunHandle.append_checkpoint(update, versions)` is written after the blobs are durable; `checkpoint_tail()` is what resume reads; `ledger_tail()` keeps its meaning (the committed record) and is what the Generator pins off.
- **The rewind.** `_discard_unsealed` (`base.py:979-1010`) generalizes: with the checkpoint at `c` and its version map `V_c`, it cuts `ledger.jsonl` back to line `c` (the same operation `_repair_ledger` performs on a torn tail, now on a provisional one), deletes `waves/` and `postdata/` above `c`, and deletes `rollouts/` whose recorded policy version exceeds `V_c` — because a rollout sampled at a version that no longer exists would pin a bundle no store can rebuild and no engine will hold after a restart. Rollouts at or below `V_c` are kept, as today. A generation-only run has no checkpoints and is untouched by this rule.
- **Retention** becomes a pure function of `checkpoints.jsonl` instead of the ledger: `KeepRestorable` keeps every checkpointed adapter version and the checkpoint tail's moments. `_refuse_live_version`'s floor is the checkpoint tail.
- **Measurement** (`measure_run`) backfills checkpointed versions only; a version with no blob is not a measurement point. The Q0 protocol in thread B already evaluates at checkpoints.
- The frozen-part emission (the 96 MB decoder at every version) is **not** in this ADR: it is an adapter-contract change (a `frozen` half of `emit`, written once at v0) and gets its own ADR after this one measures how much of the seal it still is.

### Part B — The wire exists only where a pool serves the policy; there need not be one

The first question is not *how* the policy travels but *whether it travels at all*. A run declares its pools in `Topology`; **a run with no pool that serves the policy has no wire, no `main`, no `add_bundle`, no reachability probe, and no engine anywhere in its Trainer** — a learner alone is a complete experiment (replay, SFT, distillation from a static leaf). Today `loop.py` raises `no 'main' engine pool` before Phase 0 for exactly that run (thread A: a cancelled family and a republished plan); that check is deleted, `main` is required only when traffic (an environment, a pooled postprocessor) addresses it (adopted from `codex/trainer-store-boundary`: `validate.py`, `loop.py`, `traffic.py:route`), and the Trainer tokenizes injected spans through `Learner.tokenize` (adopted) so nothing in the training world ever needs an engine.

Where a pool **does** serve the policy, the policy reaches it **over the wire by default**: the Trainer pushes each committed bundle to every such pool, before the ledger line, on whatever transport the placement resolved for that pool (`LocalTransport` in-process, `PipeTransport` to a resident, `ModalClsTransport` across containers, `HttpTransport` on Strange Loop) — the `add_bundle` push `trainer.py:202` already makes, now issued per serving pool rather than to a `main` that must exist. Consumers ask the engine (`knows_bundle`); a miss on a *checkpointed* version faults in from the store (`restore_bundle_on`, unchanged); a miss on an *uncheckpointed* version raises `BundleUnavailable`, an infrastructure death: the tenancy parks and the retry resumes from the checkpoint. The engine's residency (`residency.py`) may never evict a live tenant's newest `max_policy_lag + 1` versions (I8 immunity made explicit), so the only way to reach that miss is a restarted engine — under ADR 0002 a dead host and a park regardless.

`Checkpointing(every, delivery="wire" | "store")` keeps the file-system-only path as the **non-default** setting: under `store` the Trainer pushes nothing and every consumer faults in from the store on every miss, so the gate binds `store` to `every=1` by name. `delivery` is meaningless, and ignored, for a run with no serving pool.

### Part C — Stop is a disposition; drain is a checkpoint

- `Host.stop(run_id, *, drain=True, deadline_s)` — sets the tenancy's `StopRequest`; the Trainer reads it at the top of the update loop, finishes the update in flight, `checkpoint()`s, and returns; the daemons unwind; the roster reads **`stopped`** (today: `failed`). Past `deadline_s` the host cancels as it does now — a cancelled drain loses at most one interval, which resume-equivalence already covers.
- **Dispositions**, journaled by the desk, the one writer:
  - `stopped {run_id, subdir, reason, t}` — written by `Desk.stop(run_id, reason, drain=True)` after `stop_anchored` acknowledges. Outranks `parked`: `parked_rows()` drops a run whose latest event is `stopped`; `strand` skips it; `retry_parked` never sees it. A later `submit-intent` for the same key supersedes the stop — resubmitting is how a run resumes here, and it is the only way a stopped run moves again.
  - `failed {run_id, subdir, error, host, t}` — a tenancy that died of anything other than a wire (`Unreachable`, `WrongEpoch`, `BundleUnavailable`, a `StoreError` on a read) is not retried; the desk learns it from the roster on the next pulse or reap and journals it with the host's error text. A wire death is `parked`, as today, and retried.
  - `Desk.stop_subdir(subdir, reason)` — one `stopped` per current placement under that filing; it does not guard future explicit submissions.
- Operator doors, both venues: `RemoteDesk.stop` / `stop_subdir`, `deploy/desk.py::stop --run <ref>|--subdir <s> [--no-drain]`, `deploy/strangeloop.py stop`, and `python -m rlstack fleet stop --desk <address>` for whatever `transport_for` can reach. `status` lists `stopped` and `failed` beside `parked`, with reasons; the observer renders all three on the run row and on the fleet notes.
- `RECOVERY_GENERATION` retires: "clean restart, nothing old revives" is `stop_subdir` over every open placement, journaled per run, and readable. `BOOTABLE_METALS` stays: it is an allocation authorization, a different question.

### Part D — Idle by dependents; a clock per host; journals that batch; a watchdog that rechecks

- `listing_busy` gains a fourth yes: **a running placement routes through this listing** — read off `placements()` (the archived routes) restricted to runs whose anchor's roster, in the same snapshot, says running. This is the case ADR 0003's "unguarded" ruling did not consider: the evidence is a live anchor, not a stale journal row.
- **Host idle** — a listing that is not busy for `host_idle_s` (default: the metal's own limit) is **decarved** (GB back to residual, listing delisted "idle"); the metal's clock is unchanged and releases the container when every host has gone. One clock shape, two levels. Residents need no clock: a resident is a host's process and dies with its host; an alternating host's inactive resident is already asleep under the arbiter.
- **Registration grace** — a registered metal with no listings starts its clock at registration; a venue's `up` should carry the first submission (`up` + `submit` in one breath, as the Strange Loop chain scripts already do). Stated, not changed.
- **Journals batch** — `hosts/<name>/log.jsonl` appends stage and persist on a cadence (`JOURNAL_FLUSH_S = 30`) or on the next ledger/checkpoint persist, whichever comes first; torn and late tails are already tolerated. `fleet/log.jsonl` stays per-event: it is the desk's truth. The retention sweep's commit folds into the checkpoint's.
- **The watchdog rechecks** — `kill_stalled` probes once more with a short deadline after the stall verdict and before it kills, and journals *after* the kill, never through the store queue first.

### Part E — Unification

In order, because each later part lands once in the shared runtimes:

1. Publish the Strange Loop worktree onto main as reviewed commits: `runner/venues/*`, `transports/http.py` + `http_blobs.py`, `stores/strangeloop.py`, `exporters/wandb.py`, `deploy/strangeloop*`, examples, tests; ADRs renumbered 0011→0015 (scratch store), 0012→0016 (HTTP blobs), 0013→0017 (shared runtimes); `deploy/desk.py` and `deploy/modal_venue.py` move onto `desk_class` / `MetalRuntime`. `StrangeLoopLocalStore` publishes at `_persist` points, not per mutation (it inherits the Modal discipline: a wave is staged, the ledger line publishes).
2. Adopt the `main`-optional and `Learner.tokenize` halves of `codex/trainer-store-boundary` (0011 keeps its number).
3. Parts A–B (store, trainer, loop, restore, retention, gate, tests).
4. Part C (host, desk, remote, deploy doors, observer).
5. Part D.
6. Deploy: desk + UI + one L4 metal on Modal; one A100 lease on Strange Loop; the drill is a spaced-checkpoint run killed mid-interval, resumed, and compared.

Strange Loop findings folded into the parts above: `RemoteDesk.stop` (C), `failed` vs `parked` (C), parked-with-no-live-metal surfaced in status (C), the desk extending its own lease as a `MetalRuntime` duty and refusing a submission its remaining lease cannot finish (E, provider), the exporter waiting on the run's manifest rather than racing the adoption (E), a quarantined store owner named in `status` (E). The scratch platform gaps (native append, reload, conditional write) stay platform follow-ups as ADR 0011-SL records them; with per-update publication reduced to a ledger line and a wave, they stop being on the update's critical path.

### Touched / untouched

- **Touched** — `data/stores/base.py`: `checkpoints.jsonl` verbs, `checkpoint_tail`, the rewind in `_discard_unsealed`, `_refuse_live_version` against the checkpoint tail; `data/stores/retention.py`: the policy reads the checkpoint record; `data/stores/modal_volume.py`, `data/stores/strangeloop.py`: journal persists batch, publication at `_persist` points; `runner/daemons/trainer.py`: `commit()` and `checkpoint()` as two named steps, `StopRequest`, the `wire`/`store` push; `runner/daemons/generator.py`, `scorer.py`: `BundleUnavailable` on an uncheckpointed miss; `runner/loop.py`: `Checkpointing` threaded from the frame, the final checkpoint at the extent, `main` optional; `runner/restore.py`: `restore_tenant` from the checkpoint tail; `runner/residency.py`: the lag-window immunity; `runner/host.py`: `stop(drain=)`, `StopRequest`, `stopped` roster status, the watchdog recheck; `runner/desk.py`: `stop`, `stop_subdir`, `stopped()`, `failed()`, dispositions in `parked_rows`/`strand`, dependents in `listing_busy`, host idle; `runner/remote.py`: `RemoteDesk.stop`/`stop_subdir`, `RemoteHost.stop(drain=)`; `runner/campaign.py`: `Checkpointing` on the frame; `runner/venues/*`: the shared runtimes carry the new clocks and doors; `runner/interfaces.py`: `Learner.tokenize`; `spec/validate.py`: `check_checkpointing`, `check_main_declared_when_addressed`; `observe/views.py`, `observe/web/runs.js`, `fleet.js`: `checkpointed_at`, `stopped`, `failed`; `deploy/desk.py`, `deploy/strangeloop.py`, `rlstack/__main__.py`: the doors; `tests/test_resume.py`: crash at every point under `every=k`, plus the rewind of rollouts; new `tests/test_checkpointing.py`, `test_dispositions.py`, `test_idle_dependents.py`.
- **Untouched** — `spec/specs.py` and `spec/canonical.py`: nothing here enters identity; a run's `run_id` is what it was, and `tests/venue_spec_rows.json` pins that. `policy/`: the bundle, the lowerings, `emit`/`load` are unchanged (the frozen-part split is deferred by name). `inference/`, `training/`: the two worlds see no difference. `data/trajectory.py`, `data/flatten.py`: sealed bytes are sealed bytes. `runner/arbiter.py`: admission is untouched; the host clock reads the arbiter's counters it already exposes. `runner/transports/modal_cls.py`: the push already rides it. The ledger's line format and `manifest.json`: byte-identical to today, so every existing run directory reads under the new code with `checkpoints.jsonl` absent meaning "every line was a checkpoint" (the migration rule for old runs).

### Promises / non-promises

- **Promises** — (1) On fakes, a run under `Checkpointing(every=k)` killed at any point and resumed equals the same run straight, byte for byte, for every `k` the suite tries (1, 2, 3, 7) and at every crash point `test_resume.py` already names plus "after the ledger line, before the checkpoint" and "after the checkpoint line, before the sweep". (2) A run under `every=1` produces today's directory plus `checkpoints.jsonl`. (2a) A spec declaring a learner and no pool is accepted at the gate, runs to its extent, and its Trainer opens no transport to any engine. (3) Blobs are written `⌈n/k⌉` times, not `n` times. (4) `stop` on a live run returns only after a checkpoint at the update boundary exists, and a stopped run is never redelivered by reap, registration, or knock; only a new submit moves it. (5) A running placement's pool is never released idle while its anchor's roster says running. (6) `store` with `every != 1` is refused at the gate, and a missing `Checkpointing` is refused at every submission door. (7) The fakes suite is green in both worktrees before and after each part lands. (8) Every existing run directory still opens, resumes and renders.
- **Non-promises** — this does not make the seal faster on the checkpoint update itself (that update still writes every blob); it does not shrink the blobs (the frozen-part split is deferred); it does not prove Strange Loop's scratch publication survives an abrupt pod death mid-checkpoint (the live drill is the evidence, and it is listed as unproven until run); it does not make the HTTP blob path's throughput over SSH known; it does not remove the platform's own idle scaledown as the backstop; it does not add a wall-clock checkpoint cadence (Q10); it does not turn `failed` into an automatic retry of any kind.

### Interfaces

- **The frame** (`campaign.frame_for`) carries `checkpointing: {every, delivery}` beside `subdir` and `resume`; the desk relays it blind, the host hands it to `run_experiment_async`.
- **The store** gains `append_checkpoint` / `checkpoint_tail` / `read_checkpoints` on `RunHandle`, and `peek_checkpoints` for the observer; `_discard_unsealed` is the rewind.
- **The Trainer** exposes its two steps as two named methods; `sweep_stale` runs at the checkpoint.
- **The host door** takes `stop {run_id, drain, deadline_s}`; the roster status vocabulary is `running | done | stopped | failed`.
- **The desk door** takes `stop {run_id|subdir, reason, drain}`; `status` and `placements` carry the dispositions; `listing_busy` reads the snapshot's placements.
- **The gate** has two new named checks in `validate.CHECKS`.
- **`observe/`** reads `checkpoints.jsonl` for `checkpointed_at`, and the fleet journal for `stopped`/`failed`, and never anything else new.

### Sketches

```python
@dataclass(frozen=True)
class Checkpointing:
    """A PLACEMENT-TIME declaration (I5): what a crash may cost and how the policy travels. Never hashed."""
    every: int                       # blobs + a checkpoints.jsonl line at u % every == 0, at the extent, and on drain
    delivery: Literal["wire", "store"] = "wire"   # the gate binds "store" to every == 1; ignored where no pool serves the policy


def serving_pools(spec: ExperimentSpec) -> tuple[str, ...]:
    """The pool names whose engines serve THIS run's policy — the wire's whole extent. Empty is a complete run: a learner alone."""


class RunHandle:
    def append_checkpoint(self, update: int, versions: Mapping[str, int]) -> None: ...
    def checkpoint_tail(self) -> dict | None: ...
    def _discard_unsealed(self) -> None:
        """REWIND TO THE CHECKPOINT: ledger lines, waves, postdata past its update and rollouts pinned past its versions."""


class Trainer:
    async def commit(self, update: int, wave: Wave, postdata: dict, stats: list[TrainStats]) -> None:
        """The ledger line, after the wire push under `wire` delivery. Cheap; every update."""
    async def checkpoint(self, update: int, emitted: Emitted) -> None:
        """Blobs for every trainable entry, then the checkpoints.jsonl line, then the sweep. At the cadence, the extent, and on drain."""


class Host:
    async def stop(self, run_id: str, *, drain: bool = True, deadline_s: float = BUILD_DEADLINE_S) -> dict:
        """{stopped, run_id, state: "stopped", checkpointed_at: u} — the Trainer drained at an update boundary, or cancelled past the deadline."""


class Desk:
    async def stop(self, run_id: str, reason: str, *, drain: bool = True) -> dict: ...
    async def stop_subdir(self, subdir: str, reason: str, *, drain: bool = True) -> dict: ...
    def stopped(self) -> dict[str, dict]: ...     # latest `stopped` per run, unless a later submit-intent superseded it
    def failed(self) -> dict[str, dict]: ...
    async def listing_busy(self, listing: Listing, seen: dict[str, int], snapshot: Snapshot) -> bool:
        """... or a RUNNING placement routes through it (the fourth yes)."""
```

## Questions

**Q1. Where does the cadence live — the frame (submit-time, beside `subdir`/`resume`) or the spec?**
Recommendation: the frame. Cadence changes what a crash costs and how long an update takes, never a result — I5's definition of a placement fact — and putting it in the spec would give the same experiment a different `run_id` per cadence and break resume across a cadence change. Required at every submission door, no default (your ruling).
If the spec: identity moves with cadence; `tests/venue_spec_rows.json` and every pinned row changes; a resume must repeat the cadence exactly.

> **Samarth:** agree.

**Q2. Does attach REWIND past the checkpoint (discard ledger lines, waves, postdata and rollouts above it, regenerate), or keep committed ledger lines and only redo the training state?**
Recommendation: rewind. Redoing updates `c+1..n` rewrites their ledger lines anyway (the learner state that produced them is gone), so keeping the old lines would be keeping lines the run can no longer stand behind, and keeping rollouts pinned to versions that no longer exist would strand the Scorer on a bundle no store holds. The cost is stated and bounded: a crash costs at most `every` updates of compute, and that is the deliberate trade the cadence buys. On fakes the regenerated bytes are identical, which is the property the suite proves.
If the other branch: the ledger becomes two-valued (committed-and-restorable vs committed-and-not) and every reader has to know which; the rollout hazard needs its own rule.

> **Samarth:** agree.

**Q3. `checkpoints.jsonl` beside the ledger (recommended), or a `checkpoint: true` flag on the ledger line?**
Recommendation: the separate file. It keeps the ledger's bytes exactly today's (every old run reads unchanged, "no checkpoints file" meaning "every line was one"), it lets a checkpoint be taken *after* a commit (drain, on-demand) without rewriting a line, and retention becomes a function of the smaller record.
If the flag: one file fewer, but a drain checkpoint needs a second record anyway and old ledgers need a migration rule.

> **Samarth:** agree.

**Q4. Under `wire` delivery, is an engine miss on an uncheckpointed version an infrastructure death (park, resume from the checkpoint — recommended), or does the Trainer grow a re-emit channel consumers can ask?**
Recommendation: park. A live engine never evicts inside the lag window (I8 made explicit in `residency.py`), so the miss means the engine process restarted, which under ADR 0002 is a dead host and a park regardless. A re-emit channel would be a daemon-to-daemon call, which the blackboard forbids, and it would need to be exact at a commit boundary, which the Trainer alone can know.
If the other branch: a new Trainer verb and a new consumer wait, on every venue.

> **Samarth:** it depends on the semantics of how exactly the remote transport works — if the engine container exposes an endpoint where you add a bundle and that is how the bytes are transferred, then what you said is okay. (Answered in chat 2026-09-13: it is — `add_bundle` is the engine's own admission-free `ask` verb on every transport, and the ledger is what the daemons pin from. Recorded as agree, conditional on Part B's mechanics as written below.)

**Q5. Is `delivery` the only setting, defaulting to `wire`, with `store` the non-default file-system-only path bound to `every=1` — and is "no serving pool" simply the absence of a wire rather than a third mode?**
Recommendation: yes. Your ruling: a pool that serves the policy gets the wire by default; a run with none needs no engine, no `main`, no push, and declares nothing about delivery. `store` stays because you asked for the file-system path to remain a setting, and its binding makes it impossible to ask an engine for a bundle no blob backs.
If `store` is dropped: one fewer branch in the Trainer and its tests retire; the "no serving pool" case is unchanged either way.

> **Samarth:** yeah, we should keep store delivery as a mode. Definitely. (agree)

**Q6. Does `stop` DRAIN by default (finish the update, checkpoint, exit; cancel past a deadline), with `--no-drain` the exception?**
Recommendation: drain by default. A deliberate stop that loses up to `every` updates would make every chain script checkpoint by hand before stopping. The deadline is `BUILD_DEADLINE_S` plus one update's observed duration where the host has one, else the build deadline alone.
If cancel by default: stopping is instant and loses the interval; resume regenerates it.

> **Samarth:** agree. we should definitely checkpoint at a stop unless otherwise stated.

**Q7. Is `failed` (a non-wire death: gate refusal, OOM, a loss raising, a store read error) NEVER retried automatically, with `parked` (a wire death) the only retried disposition?**
Recommendation: yes. A code error retried is a loop; the Codex threads' revived runs were all this. The classification is by exception type at the host, and the error text rides the journal so an operator can tell OOM from a lost pod without pulling a log.
If the other branch: one retry of `failed` on fresh metal before it sticks — it hides flaky infrastructure for one interval and costs one wasted carve per real bug.

> **Samarth:** sure (agree).

**Q8. Retire `RECOVERY_GENERATION` in favor of `stop_subdir` over every open placement (journaled per run), keeping `BOOTABLE_METALS` as the allocation authorization?**
Recommendation: yes. The generation is a deployment constant that says "nothing old moves" without saying which runs; `stopped` rows say it per run, survive restarts by the same journal, and read in `status`.
If kept: two mechanisms answer one question, and a resubmitted old run must also carry the new generation to move.

> **Samarth:** agree, remove this tag; replace with stopped rows.

**Q9. Does a routed-through dependent count as busy for the idle rule (overriding ADR 0003's "unguarded by ruling" for this case), and does a host get its own decarve clock at the metal's limit?**
Recommendation: both. The dependent is evidence from a live anchor roster in the same snapshot, which is stronger than the clock's silence; the outage on 2026-09-13 06:14 UTC is the counterexample the ruling did not have. The host clock is the same rule one level down so a shared A100 returns residual as its carves go quiet.
If the other branch: the pool release stays possible under a between-waves training run; the host clock is dropped and only whole metals ever go.

> **Samarth:** yes, each carved host should also get a clock (agree).

**Q10. Cadence in updates only (recommended), or also a wall-clock `every_seconds`?**
Recommendation: updates only. A wall-clock cadence makes what is restorable depend on how fast the metal was, which is true today of nothing in a run directory; drain and the end-of-extent checkpoint cover "before I stop" and "when it is done".
If both: `Checkpointing(every=k, every_seconds=s)` checkpoints at whichever comes first; the byte-identity promise weakens to "the same set of ledger lines and blobs at or above the slower cadence".

> **Samarth:** yes (agree).

**Q11. Journals persist on a 30-second cadence or at the next ledger/checkpoint persist, never per event — accepting up to 30 s of lost host observability on a crash?**
Recommendation: yes; the fleet journal stays per event. Thread B's own preferred order put this third, and every one of those per-event commits queues ahead of a ledger line on the same volume thread.
If per event stays: no change, and the seal keeps paying for the timing line that measures it.

> **Samarth:** sure that works (agree).

**Q12. Publish the Strange Loop worktree onto main first, renumbered (0011→0015, 0012→0016, 0013→0017), and enable its automatic recovery once `stopped` exists?**
Recommendation: yes to both. Everything in A–D wants to land once in `DeskRuntime`/`MetalRuntime`, and `auto_recover=False` is what left the chain scripts polling for six hours; with `stopped`/`failed` on the record, recovery revives only what was parked by a wire death.
If the other branch: A–D land twice (once in `deploy/desk.py`'s inline class, once in the worktree) and drift again.

> **Samarth:** yes (agree).

**Q13. Measurement backfills checkpointed versions only — accepted?**
Recommendation: yes; a version without a blob is not restorable and so not a measurement point, and the Q0 protocol measures at checkpoints already. `measure_run` reads `checkpoints.jsonl`, not the ledger.
If the other branch: measurement would require `every=1`, which is the case the cadence exists to leave.

> **Samarth:** yes (agree).

**Q14. Defer the frozen-part emission (the 96 MB decoder at every version) to its own ADR, after this one measures the remaining seal?**
Recommendation: defer. It is an adapter-contract change (`emit` returning a frozen half written once at v0 and a trainable half per version) touching every adapter type's two lowerings and `compile_bundle`; folding it here would put the policy bridge inside a lifecycle ADR.
If folded: one more part, `policy/` enters the touched list, and the byte-identity obligations of `load` need their own questions.

> **Samarth:** yes (agree).

**Q15. Does the runner (Generator / Scorer / Trainer for one run) become ITS OWN PROCESS, or stay an asyncio task inside the anchor host's process?**
Today it is a task: `Host.adopt` creates it on the host's own loop (`host.py:547` → `run_experiment_async`, `:444`), anchored on the learner's host by default, `main`'s host without a learner, or any named member (ADR 0006 Part A, `campaign.anchor_demand`). It is not bound to the learner's process — the learner is a child process behind a pipe (ADR 0002) — but it always lives inside SOME host container, and a host loop wedged on its store queue stalls every runner on it (the 15-minute phases in thread B).
Recommendation: keep it a task in this ADR, and open the process split as its own ADR right after: the runner is already a pure client (store + transports to pools and the learner), so the split is "a host with no residents that anchors runners" — a CPU-only regime the desk can place onto and lease like any other — rather than a new kind of thing. Doing it here would put a fourth leased process (runner) into an ADR already moving the commit, the stop and the idle rule.
If split now: `runner` becomes a capability a metal declares, the desk gains a runner-only carve, `Host.stop` becomes a process stop, and the campaign manager you described (the desk selectively terminating the processes running experiments) is literally true.

> **Samarth:** resolve as you see fit. (Resolved per the recommendation: the runner stays a task of its anchor host in this ADR; the process split is the next ADR.)

**Q16. Retire the word "daemon" for Generator / Scorer / Trainer?**
Your reading is right: the hosts are the standing services that take submissions, and these three are clients of them. Recommendation: the collective noun becomes **runner** — "a run is its runners", `RunnerNeed`, `plan_runners`, `runner/daemons/` → `runner/runners/` (or `runner/roles/` if the doubled word grates; the three class names stay). Hosts are not renamed "daemons"; the word simply leaves. ARCHITECTURE.md's "Daemon" and "Resident / daemon" entries are rewritten, and the folder tree in STYLE.md rule 8 with them.
If kept: nothing moves; the vocabulary stays misleading in the place a reader meets it first.

> **Samarth:** resolve as you see fit. (Resolved: the noun is RUNNER, the folder is `runner/roles/` — the three roles of one run — because `runner/runners/` doubles the word in every import; `RunnerNeed`, `plan_runners`; the three class names stay.)

## Outcome

Implemented 2026-09-13 on `claude/adr-0014-commit-cadence` (forked from `origin/main` 5d1abf1), as nine commits in the order Part E prescribed: the `main`-optional commit of ADR 0011 cherry-picked; the Strange Loop worktree published on a sibling branch and merged (ADRs 0015–0017, CONTEXT #91); Parts A+B; Part C; Part D; the Q16 rename; this record. Suite: 1478 tests, 194 skipped (the skips are torch/vllm/modal/live-scratch gated), up from 1295 on the base.

**What the answers changed.** Q2's rewind made `checkpoints.jsonl` the attach boundary and turned `_repair_ledger` into a general record repair plus `_cut_ledger_above`; checkpoint 0 grew moments (a warm start's loaded parent moments must survive a resume at 0 — found by the crash matrix). Q4 needed the `main`-only reading of `serving_pools`: a judge on the policy base is not the policy, and a `PoolMember` cannot say otherwise without moving every pinned spec row. Q6's drain needed one more thing than the recommendation said — a Trainer WAITING for another runner's work must also read the stop (found by the dispositions test hanging), so `RunSignals.wait_once` exists and the Trainer waits through `wait_unless_stopped`. Q8 retired `RECOVERY_GENERATION` from the desk, both runtimes, both providers' configs and the examples. Q9's fourth yes reads the tick's own roster frames, probed once and concurrently, and the host clock is `Desk(host_idle_s=)`; a wire death parks avoiding no host (the anchor answered; what died was a dependency). Q11's batching is a per-store flush timer disarmed by any commit. Q16 renamed the folder to `runner/roles/`.

**What landed beyond the sketches.** `RunReport.stopped` and an honest `completed` for a drained run; `Tenancy.error / wire / ended_t` and `Host.end_tenancy` as the one writer of a tenancy's last word; `Desk.dispositions()` as the single journal pass behind `parked_rows`, `stopped`, `failed`; `Desk.record_deaths` on the reaper's tick, off listings that answered; the watchdog's `stall-recovered` row; `peek_checkpoints`, `RunProgress.checkpointed`, `checkpointed` on observer rows; the `fleet` subcommand of `python -m rlstack`.

**Tests added.** `test_checkpointing.py` (byte-identical resume at every k in {1, 2, 3, 7} across nine crash points of the two-step protocol; the rollout rewind; a pre-0014 directory; drained stops; both deliveries; both kinds of miss), `test_dispositions.py` (a drained stop journals and survives reap, registration and a stray park; resubmission supersedes it and resumes from the drain's checkpoint; `stop_subdir`; an undrained stop; the experiment's own death is `failed` and never retried; a wire death is parked and resumes from the checkpoint; the observer's ranking), `test_idle_dependents.py` (the routed-through pool under a blocked learner stays busy; idle hosts are decarved before their metal is released; journal lines ride the next commit or the timer; the recheck keeps an answering resident and ends a silent one). The two ADR 0011 boundary tests now pin `delivery="store"`, which is exactly the discipline they were written for.

**Deliberately not done.** The Strange Loop worktree's desk custody rewrite (superseded by Part C); `StrangeLoopLocalStore` publishing at `_persist` points (its per-write contract is encoded in its tests and ADR 0015's prose — its own note first); the frozen-part emission (Q14, next ADR); the runner as its own process (Q15, next ADR).

**Unproven on metal, stated as CONTEXT #92 states it:** no live drill yet on either venue — a spaced-checkpoint run killed mid-interval and resumed and compared; a drain through a real `PipeTransport` learner; `BundleUnavailable` from a real engine restart; HTTP blob throughput over SSH. The deploy is the next step, one L4 on Modal first.
