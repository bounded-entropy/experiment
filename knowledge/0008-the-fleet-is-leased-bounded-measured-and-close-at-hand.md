# ADR 0008 — The fleet just works: leased, bounded, measured, and close at hand

| | |
|---|---|
| **Date** | 2026-09-04 |
| **Status** | **Implemented on fakes (2026-09-05; CONTEXT #85); awaiting the drill (Q8).** Accepted 2026-09-04: Q2, Q4, Q5, Q7 agreed; Q3 amended — a LIVE read, bounded and outside the lock; Q1, Q6, Q8, Q9 stood on the recommendation. |
| **Author** | Claude Fable 5.1 (session: ADR 0005 on metal, 2026-09-04) |
| **Touches** | `rlstack/runner/desk.py` (leases, epochs, deadlines, journal-first submit, the parked state), `rlstack/runner/remote.py` (deadline on `Transport.call`/`ask`; the epoch in every frame's address), `rlstack/runner/transports/modal_cls.py` (deadline), `rlstack/runner/host.py` (resident heartbeats and the phase watchdog), `rlstack/runner/learners/*` and `engines/*` (the first-contact report), `rlstack/runner/meters.py` (counters that must advance), `deploy/modal_venue.py` (heartbeat duty, async doors, measurement on the metal, the client-side canonical row), `deploy/concept_steer.py` + the check venues (doors on the client), `rlstack/observe/` (parked and unreachable rendered), `tests/` (fakes for every invariant; the drill on metal), `ARCHITECTURE.md`, CONTEXT #85 |
| **Invariants** | I5 (topology semantics-neutral: nothing here touches a run's identity), I10 (the store is the run: every fact below is journaled before it is acted on), I11 (a run is self-describing: a parked run says why) — and six FLEET invariants introduced here, F1–F6 |
| **CONTEXT** | extends ADR 0001 (the desk), ADR 0003 (idle metal, the knock), ADR 0007 (one desk, addresses carry the venue); records the 2026-09-04 venue day (the chronology and "what goes wrong" notes in the session's scratchpad, to be folded into #85) |

## Original prompt

> in the future, i just want everything in this process (different experiments submitting to desk, deploying on gpu), to be really clean and just work. do you think the codebase is in this state now, or should we go through the open hazards first
>
> i want to have a robust understanding of what could go wrong, even when we migrate to other services (AWS or runpod). it needs to be clear what sorts of issues could occur, and how we could resolve them generally
>
> let's write this hardening ADR, with clean global invariants like you introduced

## Context / problem

ADR 0005's three arms reached their extent on 2026-09-04, but only under a human's hand for six hours. Twenty-odd failures on one venue, on one platform, sort into four root causes — and none of them is Modal's, which is why the fix is also the portability layer.

**The desk believes a picture nobody is obliged to keep true.** It rebuilds the fleet from its journal and treats the rows as facts: on the plane, answers at this address, carries this run. The world drifted while the rows stood, five times in one evening: a released container registered itself and then stopped fetching inputs (a metal on the plane with no one home); a venue redeploy made a metal's name resolve to a container that did not exist yet while the old one kept the GPU; a release left the row's plane flag standing so a door submitted before the reboot had registered; a learner sat twelve minutes inside a device move and nothing noticed; the observer took a dead duplicate's last word over a live tenancy's.

**The wire is trusted to be prompt, exactly-once, and cancellable.** An unbounded residual ask under the placement lock wedged the desk for an hour — status, reap and every submit queued behind one unreachable metal, twice. The stats tick awaited a resident inline through its ten-minute load. A submit input Modal replayed off a shut-down container ran twice and adopted one run on two metals. A client's timeout on a sync door shut the desk container down, three times.

**Numbers were declared where they should have been measured.** The learner's declared 48 then 64 GB per card met a 46 GiB resident plus one 8 GiB fp32 block, three times. The traffic meter read zero on every venue for three days after ADR 0002 moved engines into their own process, and zero was indistinguishable from broken. ADR 0007's mount rewrite of the observer had never been deployed, and when it was, every container died at construction while the old one kept serving a week-old view.

**Control-plane work was put on the platform's scheduler and reflection.** The canonical row, progress and the measurement each ran as an on-demand CPU function, and for an hour Modal scheduled none of them; the desk itself waited fifteen minutes for a CPU worker; the platform imported our classes by `__module__`; every `::status` booted a second desk; and for fifty minutes no GPU pair was schedulable while we held one in a container that could take no inputs.

**Where things run, stated once, because F1 applies per level.** A carved host is an object inside the metal container's one control process; that process holds the `MetalService`, every host on the metal, the door that answers the desk, the stats tick and the keepalive. A run's daemons — Generator, Trainer, Scorer — are asyncio tasks in that same process, one `TaskGroup` per adoption, so stopping a run is cancelling one task. The residents are the processes: `Resident.spawn` forks the engine (itself one process per tensor-parallel rank) and the learner (one per FSDP rank), and a daemon reaches them over a pipe. So the metal heartbeats for the container, the host heartbeats for its residency, and each resident heartbeats to its host through the pipe — three leases, three different things that can die.

What Modal hides, AWS and RunPod will not: a name that routes to the latest version and boots on demand; at-least-once replay after a container dies; a container that quietly stops fetching; capacity waits inside a call. A desk that owns those four itself moves platforms by changing only how a container is created and addressed.

## Decision

Six fleet invariants. Each is a sentence the code can be checked against, on fakes and in a drill on metal.

- **F1 — Leased.** A fact about the fleet is true only while its lease is renewed. Every metal, host and resident registers with an EPOCH (its boot id) and renews a LEASE by heartbeat; the desk believes nothing whose lease has lapsed, lists nothing it has not heard from within the lease, and places nothing on it.
- **F2 — Epoch-addressed.** A frame names the instance it means. Every registration, carve and admitted frame carries the epoch it expects; a container that receives a frame for another epoch refuses it by name. A released container is a lapsed epoch: it refuses every later input and is never reborn.
- **F3 — Bounded.** Every wire verb has a deadline, and no lock is held across the wire. A wait that expires journals `unreachable` on the row it was about, and the placement ladder passes that row over. Placement reads every metal's residual LIVE — concurrently, under one deadline — before it takes its lock, and places against that snapshot; nothing in the desk awaits a metal inside the placement lock.
- **F4 — Idempotent.** Every state-changing verb is idempotent by its key, because every wire is at-least-once. The journal records the INTENT before the act (`submit-intent` with the run id and folder before placement), so a replayed frame finds its own intent and answers with the delivery it already has. Cancellation is the server's decision: doors are async and carry their own deadlines; a client never cancels a fleet input.
- **F5 — Measured.** Every declared resource number is journaled beside its measured counterpart at first contact, and no meter can read zero silently. A resident reports its peak memory after its first forward and the host journals it beside the partition's declaration; a meter carries a request counter that must advance whenever the resident served anything; a build ships only after a smoke run in its own image.
- **F6 — Close at hand.** Control-plane work runs where a leased process already is — the client for anything pure, the desk for anything that needs the store, the metal for anything that needs the engine — never on an on-demand function. Waiting for capacity is a journaled PARKED state with a reason and a bound, never a hang inside a door.

### Touched / untouched

- **Touched** — `desk.py`: `Lease` (epoch, last heartbeat, limit) per metal and host; `heartbeat` verb; `listings`/`metal` rows gain `epoch`, `heard_t`, `lease_s`; `covers` requires a live lease; residuals read live at placement — every metal asked concurrently, one deadline, BEFORE the lock — and the snapshot placed against (the heartbeat's residual feeds the row the observer shows, nothing else); `submit-intent` journaled first, `submit` idempotent per `(folder, run_id)`; `parked` events carry `wants` (regimes, devices, GB) and `since`; the reaper reaps lapsed leases, not just silent probes. `remote.py`: `Transport.call`/`ask` take `deadline_s`; `Address` gains an optional `@epoch`; `HostService`/`EngineService` refuse a mismatched epoch. `modal_cls.py`: the deadline is passed to the Modal call; `door_ask` becomes async. `host.py`: residents heartbeat through their door; a resident that misses its heartbeat inside a phase with a bound is killed and journaled `stalled`. `meters.py`: `requests_served` counter, monotone. `learners/*`, `engines/*`: `first_contact()` → `{"peak_gb": ...}` journaled beside the declaration. `modal_venue.py`: a heartbeat duty per host; async doors; `canonical_row` on the client (the plan bytes go to the cas THROUGH THE DESK's `put_plan` verb, so the client needs no mount); `measure` rides the metal that serves the pool. `observe/`: `parked`, `unreachable`, `stalled` and `epoch` rendered; run status from open residencies (landed 3a02b58).
- **Untouched** — every spec, plan and run identity (nothing here is hashed); the store's seal rules; the daemons; adapters, losses, engines' inference paths; the address grammar's scheme and app/cls/host segments (the epoch is a suffix); ADR 0003's idle rule (a lapsed lease is a stronger reason to release, not a new one).

### Promises / non-promises

- **Promises** — (1) A metal or host whose heartbeat stops is off every listing within one lease, and nothing is placed on it (fakes: a fake clock). (2) A frame carrying a stale epoch is refused by name, and a released container answers no later input (fakes; the chassis stub). (3) No desk verb holds the placement lock across a wire call; every wire call has a deadline; an expired wait journals `unreachable` and placement continues (fakes: a transport that never answers). (4) The same `submit` frame delivered twice yields one placement and two identical replies (fakes). (5) A client cancellation cannot end a desk or metal container: doors are async and their deadlines are their own (chassis stub + the drill). (6) After the first forward, every host journal carries `peak_gb` beside `vram_gb` for each resident; a resident that served requests and reports a zero window fails a test on fakes. (7) `concept_steer::train` reaches its submit with no on-demand function in the path; `::measure` runs on the metal. (8) Waiting for capacity appears on the run page as `parked` with its reason within one reaper tick. (9) THE DRILL, on metal: two venues deployed; two experiments submitted from two sessions; one venue redeployed mid-run; one metal container killed by hand; every run reaches its extent or is parked with a reason, unattended, and the fleet ends released. The ADR is Implemented only after the drill.
- **Non-promises** — multi-node FSDP; an S3 store; a scheduler that waits for capacity better than the platform does (parking is visible, not smart); the learner's memory being smaller (F5 measures; it does not shrink).

### Interfaces

```python
# desk.py
@dataclass
class Lease:
    epoch: str          # the instance's boot id, minted at bring-up
    heard_t: float      # last heartbeat
    lease_s: float      # how long a silence is still belief
    def live(self, now: float) -> bool: ...

class Desk:
    async def heartbeat(self, name: str, epoch: str, residual: list[float] | None) -> dict
    # register_metal / list_host carry `epoch`; a stale epoch is refused by name
    # submit: journal {"event": "submit-intent", "folder", "run_id", "t"} FIRST;
    #         a second frame with the same key answers the archived delivery
    # place: reads residuals from the leases' cache; never awaits a metal under the lock

# remote.py
class Transport(Protocol):
    async def call(self, verb: str, payload: dict, *, deadline_s: float) -> dict
    async def ask(self, verb: str, payload: dict, *, deadline_s: float) -> dict   # async now
class Unreachable(Exception): ...        # a deadline expired; journaled by the caller

# host.py / interfaces.py
class Resident(Protocol):
    def first_contact(self) -> dict      # {"peak_gb": float, ...} after the first forward
# meters.py
TrafficWindow.requests_served: int       # monotone across windows; a served request that reports 0 is a failure
```

### Sketches

```text
# a lease's life
metal boots  --register(epoch=e1)-->  desk: Lease(e1, now, 60s); listed, placeable
every 20s    --heartbeat(e1, residual)-->  heard_t = now; residual on the row (display)
submit       --residual? to every live metal, concurrently, 5 s-->  snapshot; THEN the lock; place
silence 60s  -->  lease lapsed: delisted, not placeable, reaper knocks (boot_for) and journals
container released  -->  epoch e1 retired: every later frame for e1 refused; the knock boots e2

# a submit, twice
frame#1 submit(run X)  -> journal submit-intent(X) -> place -> deliver -> archive -> reply R
frame#2 submit(run X)  -> intent(X) exists -> reply R  (no second placement)

# a venue redeploy, no longer a hazard
old container holds epoch e1; new version's container boots as e2 and registers;
e1's heartbeats keep it listed until it is released — the desk addresses e1 and e2 as two
instances of one metal name, and the venue's `deploy` door releases e1 first anyway (F2)
```

## Questions

**Q1. The lease is 60 s and the heartbeat 20 s, for metal and hosts alike; residents heartbeat to their host at 10 s.**
Recommendation: yes. Long enough to survive a slow tick, short enough that a dead container is off the listings before a campaign door's next submit. The values are desk constants, journaled at registration, and a metal may declare a longer one.
If the other branch (leases as long as the idle limit): a dead metal stays placeable for thirty minutes — today's evening, again.

**Q2. The epoch rides the address as a suffix (`modal://app/cls#host@epoch`) and every frame carries it; a mismatch is refused by the receiving container.**
Recommendation: yes, the address is where the desk already keeps what it addresses, and `parse_address` gains one optional segment. The venue-carrying address of ADR 0007 Q3 stays intact.
If the other branch (epoch only in the registration row): a frame routed to a fresh container of the same name is silently accepted by it — the stranded-address hazard survives.

> **Samarth:** agreed (2026-09-04), after "what exactly does epoch mean": a boot identity a container mints once and that dies with it — a name is not an instance, and the epoch is what lets a frame be refused when it belongs to a world that no longer exists.

**Q3. Residuals are read at heartbeat time and cached; placement never asks the metal.**
Recommendation: yes. The wire call that wedged the desk moves out of the placement lock entirely; the price is a residual up to one heartbeat old, and a carve that finds less than the cache said is refused by the metal and parked, as any carve refusal is today.
If the other branch (a bounded ask under the lock): a 5 s deadline per metal times N metals inside every submit, and a lock still held across I/O.

> **Samarth:** "i feel like placement should keep a live read but it could return with unreachable. what's wrong with that?" (2026-09-04). *Folded, amended: nothing is wrong with it, and it is the stronger check — a live read proves the metal is reachable NOW and sees a decarve the heartbeat would miss. The invariant underneath both branches is that the read is OUTSIDE the lock: a submit asks every live metal concurrently under one deadline, snapshots, then takes the lock and places against the snapshot. Worst case per submit is one deadline, not one per metal; an expired ask is journaled `unreachable` and passed over. The heartbeat's residual is for the row only.*

**Q4. Doors are async and carry their own deadlines; a client never cancels a fleet input.**
Recommendation: yes. On Modal a cancelled sync input on a concurrent container shuts the container down (observed three times); an async method is cancellable without that. `door_ask` becomes async, `Transport.ask` becomes async, and the sync call sites (registration, build facts) go through `asyncio.run` or `to_thread` where they already are.
If the other branch (keep sync doors, forbid client timeouts by rule): a rule a human forgets under stress, as one did tonight.

> **Samarth:** agreed (2026-09-04). *The "concurrent container" is the desk itself — `@modal.concurrent(max_inputs=32)` on one process; a cancelled request on a sync door has no clean interruption, so Modal shuts the container down; on an async door it is one task the loop drops.*

**Q5. The canonical row is computed on the client and the plan bytes reach the cas through a desk verb (`put_plan`), so no venue door needs an on-demand function.**
Recommendation: yes. The row is pure (proved tonight: the local row matched the function's byte for byte); the only reason for the function was the mount. Progress reads come from the observer's API, which the campaign helper already knows how to poll. The measurement rides the metal that serves the pool (a `measure` verb on the metal's plane door).
If the other branch: every door stays hostage to the platform's CPU scheduler, and none of it exists on AWS or RunPod.

> **Samarth:** agreed (2026-09-04).

**Q6. `first_contact` is journaled for every resident, and a served-but-zero window is a test failure, not a quiet tick.**
Recommendation: yes. Three OOMs and three days of zeros were both "declared, not measured". The report is one dict per resident after its first forward or first request; the counter is one integer per window.
If the other branch: the next adapter type or the next platform gets the same three days.

**Q7. A run whose host's lease lapses is PARKED (journaled with what it wants) and rerouted on the next registration, as a reaped listing's runs are today.**
Recommendation: yes — this is ADR 0001 Q5d's parking with a second trigger, and the run page shows it (F6). Resume-equivalence prices the move at one uncommitted update.
If the other branch (fail the run): a platform hiccup fails experiments that the store could have resumed.

> **Samarth:** agreed (2026-09-04), with the clarification recorded in Context: daemons are tasks in the host's process, residents are the processes — a lapsed HOST lease parks its runs; a lapsed RESIDENT lease kills that resident and the host journals `stalled`.

**Q8. The drill is the acceptance test and runs before Status becomes Implemented.**
Recommendation: yes, and it is Samarth's to run: two venues, two sessions, a mid-run redeploy, a container killed by hand, everything unattended. ADR 0007 became "Implemented on fakes" and met the metal the same afternoon; this ADR does not get that status.

**Q9. Non-promises, stated once.** Nothing here makes capacity appear; parking makes waiting visible and bounded. Nothing here shrinks the learner; F5 measures it. Multi-node and S3 remain their own ADRs. Agree?

## Outcome

**Status: Implemented on fakes; awaiting the drill (Q8).** 1125 tests green on
fakes, from 1066 before the ADR. NO METAL WAS RUN: not one line below has been
executed on a GPU, on Modal, or against a real desk. Q8 said this ADR does not
get "Implemented" until the drill runs, and it has not.

Four commits, one per invariant pair, each green before the next:

1. `The fleet is leased and epoch-addressed` (F1, F2) — 1093 tests
2. `The wire is bounded and the submit is idempotent` (F3, F4) — 1105
3. `Measured at first contact` (F5) — 1114
4. `The control plane comes close at hand` (F6) — 1125

### What landed, against each invariant

**F1 — Leased.** `Lease(epoch, heard_t, lease_s)` in `desk.py`, one table over
metal names AND host names because there is one rule. `heartbeat(name, epoch,
residual)` is a desk verb wired through `Campaigns` and `RemoteDesk`;
`register_metal` and `list_host` carry the epoch and open the lease, and
journal the constants (60 s lease, 20 s heartbeat — Q1) on their own rows.
`leased()` is the gate at the join rung and the carve rung; `status()` rows
carry `epoch`, `heard_t`, `lease_s` and the verdict. The reaper reaps lapsed
leases BEFORE it probes anything, concludes those listings without retries,
parks their runs and knocks their metals. The chassis mints an epoch at
bring-up, announces it, and renews at the cadence the registration reply hands
back, re-announcing when a heartbeat is refused. Residents answer a
`heartbeat` door verb; `Host.watch_residents` asks each every 10 s and ends
one that has answered nothing past a 600 s phase bound, journaling `stalled`.

**F2 — Epoch-addressed.** `parse_address` reads an optional `@epoch`;
`with_epoch` composes it and `without_epoch` strips it, because the epoch
names the instance and never the route — every transport dials the route and
STAMPS the epoch into the frame under one reserved key (`@epoch`).
`check_epoch` is the one refusal and `WrongEpoch` names both lives; it is
called at the host's door, the metal's plane door and the resident's door. A
carved host's address carries its container's epoch; a released container is a
lapsed epoch and the commit `1ea2b8a` refusal stands, now naming it.

**F3 — Bounded.** Both `Transport` verbs are coroutines taking `deadline_s`,
and every frame goes through `bounded`; `Unreachable` is what an expired wait
raises and the caller journals `unreachable` on the row it was about.
Placement reads the fleet LIVE and WHOLE before deciding — every leased metal's
residual and every leased listing's status, concurrently, under the desk's own
deadlines (5 s and 10 s), outside every lock — and `decide` is a pure function
of that `Snapshot`, the only thing the one `asyncio.Lock` in the file is held
over. The carve is issued after the lock is released. The audit is a test that
reads the source: one `asyncio.Lock()` in `desk.py`, and no `await`, `async`
or journal write inside `decide`/`join_rung`/`carve_rung`.

**F4 — Idempotent.** `submit-intent` before placement, `submit-replayed` when
a replay is answered from the archive, `submit-missed` when an attempt found
nowhere to go. Doors are async on both containers; the chassis has a `fleet()`
helper and no client-side timeout anywhere near a desk call.

**F5 — Measured.** `first_contact()` on the Engine and Learner protocols,
implemented for the fakes, `TorchLearner` (allocator high-water mark),
`FsdpTorchLearner` (gathered across the chorus by an announced verb) and
`VllmEngine` (weights GB, KV tokens, off vLLM's own config). `ResidentBirth`
carries the GB it was declared at; the host journals `first-contact` once per
resident. `TrafficWindow.requests_served` is monotone; `meter_silent` is the
one reading and `meter-silent` the journal line. `smoke_function` in the
chassis, wired into all three venues, and each header says to run it before
`modal deploy`.

**F6 — Close at hand.** `canonical_row(build, borrow=...)` builds specs on the
client against a throwaway `LocalStore` and ships the cas through the desk's
`put_plan`; `read_cas` is its inverse. `progress`/`ledgers` poll the
observer's `/api/run/<id>?root=`, which now serves `done`.
`MetalService.measure_the_run` is a plane-door verb. `canonical`,
`progress_function` and `measure_once` are deleted. Every `parked` row goes
through one writer and carries `wants` and `since`; the observer reads the
fleet journal (`fleet_notes`) and renders `parked`, `unreachable`, `stalled`
and the epoch on host and run rows.

### Deviations from the ADR, and the ambiguities resolved

1. **The lease gate is `Desk.leased()`, not `covers()`.** The ADR said
   "`covers` requires a live lease". `covers` compares regimes and recipes —
   birth facts of a description — while a lease is what the desk has HEARD,
   which lives with the heartbeats that renew it. Splitting them keeps
   `covers` a pure two-argument rule (and its own tests) and puts the
   liveness question where the table is.
2. **The resident's epoch refusal lives in `Door`, not `EngineService`.** The
   ADR named EngineService; `Door` is the resident's whole door (it owns
   `sleep`, `wake` and `hello`, which reach no service at all), and
   EngineService is also constructed inline by `HostService`, which has
   already checked. One receiver, one check.
3. **The submit key is a digest of the FRAME, not the run id.** The ADR's
   `submit-intent {folder, run_id, t}` needs a run id before placement, and
   computing one means reading the spec, which the desk may not do. A run id
   is a pure function of what the frame carries, so the digest decides the
   same question; the run id joins the record at the `place` event.
4. **Two readings the ADR did not name, chosen by the repo's own rule
   (refuse loudly over waiting silently) and written into the code:** an
   accepted delivery whose run is NOT RUNNING is a resume to place, not a
   replay to answer — otherwise stopping a run and resubmitting it, the way
   every resume on this fleet happens, would be answered with the delivery of
   the run that was stopped; and a second frame arriving while the first is
   still in flight is refused with `in_flight: True` rather than waited for or
   placed again. `IN_FLIGHT_S` (300 s) bounds that refusal so a desk that died
   mid-submit does not block its own retry forever.
5. **`read_cas` is an addition.** Q5 named `put_plan` only, but a teacher's
   rollout plan is one group per prompt in a task set, so a client that must
   BUILD a plan has to READ what the plan indexes. It is `put_plan`'s inverse
   and equally opaque to the desk.
6. **`liveness` moved from the desk's sync door to its async one.** It fans
   out over the wire, and after F3 a verb that goes to the wire belongs where
   it can be bounded and cancelled.
7. **The recontinue lock became a claim set.** The ADR's audit says no lock
   may be held across a wire call; `retry_parked` is nothing but wire calls,
   so each run is CLAIMED for the duration of its own reroute instead. The
   guarantee ("a run must not be adopted twice") is unchanged and two retries
   over different runs now proceed together.
8. **Registration no longer reconciles.** `register_metal` returns whether the
   name was KNOWN and writes only the journal; `reconcile_metal` is the
   separate awaited pass. F2 made most of it free: a listing whose epoch is
   not its metal's current one is a corpse without a probe.
9. **`LocalTransport` and `DoorTransport` answer on a thread.** A `Service`'s
   `answer` is synchronous, and running it inline would hold the loop meant to
   be bounding it — so the deadline could never fire, and a slow in-process
   answer would starve every other bridged frame. Measured: the fakes suite
   went 9.1 s → 11.5 s for the whole of ADR 0008, which is the price of a
   thread hop per in-process ask.
10. **`RESIDENT_STALL_S` is 600 s** — chosen so the unexplained 12-minute
    `.to()` of 2026-09-04 is caught, and generous enough that a 32B's weight
    load is not.
11. **The observer needs its URL.** `RLSTACK_OBSERVER` is an environment
    variable with no default: the workspace is not something the chassis can
    know, and an unset one refuses loudly at the first poll.

### Found on the way, not foreseen by the ADR

- **The desk's `metal` wire verb dropped a registration's declared `idle_s`.**
  `Desk.serve` never passed it to `register_metal`, so every metal took the
  desk's default however loudly its venue declared its own. Invisible in
  practice (the venues declare 1800 s, which is the default) and fixed here.
- **`sleeps = ranks.width == 1`'s cousin**: nothing new, but the same shape —
  `first_contact` on a chorus needs a collective, so it is an ANNOUNCED verb
  like `sleep`, and `follow`'s table gained a fourth entry.

### What stays unproven

Everything on metal, which is Q8. Specifically unobserved: that a Modal
container answers `door`/`door_ask` under the new async signatures and the new
deadlines; that a metal's heartbeat duty keeps a lease alive across a real
20 s tick; that a stale epoch is refused by a real reborn container rather
than merely by a `LocalTransport`; that `torch.cuda.max_memory_allocated`
reports what an OOM is decided against, or that `all_gather_object` inside an
announced `first_contact` does not deadlock a real chorus; that vLLM 0.28's
`vllm_config.model_config.get_weights_bytes()` and `cache_config.num_gpu_blocks`
exist under those names (the read is defensive and reports `unreadable`
instead of crashing, which is the one place in this repo defensive reads are
deliberate); that the observer's `/api/run/<id>` is reachable from a `modal
run` driver; and that `put_plan` puts a plan a metal container can then read
off the same volume. THE DRILL (promise 9) is Samarth's and is unrun.

Recorded as CONTEXT **#85**.
