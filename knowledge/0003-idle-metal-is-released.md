# ADR 0003 — Idle metal is released by the desk

| | |
|---|---|
| **Date** | 2026-09-01 |
| **Status** | Implemented (2026-09-01; CONTEXT #76) |
| **Author** | Claude Fable 5.1 (session: gsm-campaign, during ADR 0001's implementation) |
| **Touches** | `runner/desk.py` (the desk's idle policy, `release`), `runner/remote.py` (one metal verb), `runner/host.py` (two status fields), `deploy/` (per-metal `idle_s`, the venue's scaledown rule), `tests/` |
| **Invariants** | I12 (acquire is a human's — this ADR bears on its inverse and on re-acquire) |
| **CONTEXT** | extends #68 (the metal plane and the reaper), #72 (reroute, park), ADR 0001's supervision loop (Q5–Q5d, CONTEXT #75 when it lands); the entry number lands at implementation. **Implements after ADR 0001**: it edits `register_metal` and `reap`, which 0001 is rewriting. |

## Original prompt

> ok now i want to address a simple concern real quick. i think it should be
> true, that if nothing runs on a piece of metal for some duration (e.g. 30
> min), it should be automatically deallocated. it's true that the desk should
> own this. can we implement this? it should be a property of the desk, with the
> option to override the limit upon a new metal request hitting the desk.
>
> this shouldnt be too bad to implement. could you write a quick ADR for it
> (with only a small number of questions since this is a simple change), and we
> can move forth?

And, answering in session:

> i agree with the four question resolution you had from ADR 0003. could you
> implement … or rather, ask an async opus subagent to implement

## Context / problem

A registered metal today lives until a human tears it down. The metal
container is kept alive by the shift `up` spawns (`gsm_a100.py:363`, a
never-returning `serve` input) or, failing that, by the venue's own scaledown
(`scaledown_window=900`, `gsm_a100.py:329`) — which is Modal's decision, not
the fleet's, and leaves the desk's listings and metal row standing over a
container that is gone (ADR 0001, Q5's corpses). Nothing in the fleet says
"nothing has run here for half an hour, hand the card back."

What the desk already knows, and does not: `Listing.occupied()` (`desk.py:174`)
reads a host's roster over the wire — running tenancies only. The arbiter
counts admitted work in flight per resident (`arbiter.py:37`, `in_flight`) and
`Host.status()` does not report it (`host.py:620`), so a PURE CLIENT sampling
through a host with no tenancy — a measurement cron (`dsl_a100.py:639`,
every 10 minutes), a side evaluator — is invisible to any idleness test the
desk can run today. The reaper already runs on a schedule (`dsl_a100.py:707`,
`modal.Period(minutes=15)`) and already probes every listing; it has the tick
and the wire, and no idle rule to apply.

Release is the acquire rung's inverse (I12: acquire = new metal = money = a
human), and ADR 0001's supervision loop makes the inverse cheap on a lazy
venue: a released metal that is still deployed costs nothing, and a knock
boots it, re-registers it (0001 Q5/Q5a) and makes it carve-able again.

*Unmeasured:* how often an idle metal actually sits past 30 minutes on the
gsm venues. The reaper's journal would say; nobody has asked it.

## Decision

**The desk releases a metal nothing has run on for `idle_s`.** The desk owns
one default (`Desk(idle_s=1800)`), and a metal's registration may override it
for that metal (`register_metal(..., idle_s=)`; `None` = never released, a
pinned metal). At every reaper tick the desk observes each carve-able metal:
IDLE means every listing on it reports no running tenancy AND no admitted
work since the previous tick (Q1), or the metal has no listings. The first
idle observation stamps `idle_since`; a busy one clears it (Q2). Past `idle_s`
the desk RELEASES: every listing on the metal is delisted (reason "released:
idle"), the metal is told `release` — its residents come down the ladder and
its shift ends, so the venue reclaims the container — and a `release` event
is journaled. The row stays inventory (what the fleet owns) and leaves
`metal_remotes` (what the fleet may carve), so `status()` says `released`.
The reaper skips released metals: a released metal is not silent, it is
parked. A placement that nothing holds and that a released metal could hold
KNOCKS it (Q4): bring-up re-registers, the row updates, the carve proceeds —
the automatic re-acquire of metal the fleet already owns.

### Touched / untouched

- **Touched** — `runner/desk.py`: `Desk.idle_s`, `metal_idle_s`, `idle_since`;
  `observe_idle` / `release_idle` (one named method per rule); `release(name)`;
  `register_metal(..., idle_s=)` journaled; `from_journal` replays `release`
  and a later `metal` clears it; `reap` calls the idle sweep and skips
  released metals; `provision_unit` knocks a released metal that could hold
  the unit before falling through to boot instructions.
- **Touched** — `runner/remote.py`: `RemoteMetal.release()` → the `release`
  metal verb; `MetalService.serve("release")` runs `shutdown()` (which 0002
  already has, `desk.py:1055`) and ends the shift.
- **Touched** — `runner/host.py`: `status()` gains `in_flight` and `admitted`
  (a monotone count of admissions since birth) off the arbiter — two fields,
  both already counted.
- **Touched** — `deploy/*.py`: each venue passes `idle_s` for its metals (or
  leaves the desk's default); the venue's own scaledown is set no shorter
  than the desk's limit, or removed, so the desk and not the venue decides
  (Q3).
- **Untouched** — the fleet journal's existing rows, `Metal`, `Partition`,
  `Host` beyond two status fields, every daemon, `spec/`, identity, `observe/`
  (it renders `status()` and already shows `plane`; `released` is one more
  boolean on the same row).

### Promises / non-promises

- **Promises.** A metal with no running tenancy and no admitted work for
  `idle_s` is released within one reaper tick of the limit, its listings
  delisted and its residents down, journaled; a rebuilt desk agrees. A metal
  registered with `idle_s=None` is never released. A pure client's traffic
  keeps a metal alive. A released metal is re-acquired by the first placement
  that needs it, with no human step, on a venue whose knock boots. The suite
  is green; `test_resume.py` untouched.
- **Non-promises.** It does not observe a real release-then-knock on Modal
  (UNPROVEN until seen). It does not release a metal a tenancy is RUNNING on,
  however slowly — idle is the absence of work, not its speed. It does not
  choose which of several fitting released metals to knock (the first by
  name, as `provision_unit` already orders). It does not touch the venue's
  own timers beyond the rule in Q3.

### Interfaces

`metal` verb (fleet) grows `idle_s`; `release` (metal) is new — the third
metal command beside `carve`/`decarve`; `status()` reports `released` per
metal; `Host.status()` grows `in_flight` and `admitted`; the fleet journal
grows one event type, `release: {metal, t, idle_s, listings: [...]}`.
`reap`'s reply grows `released: [metal, ...]`.

### Sketches

```python
class Desk:
    def __init__(self, store, host_for, metal_for=None, boot_for=None, *,
                 idle_s: float | None = 1800.0) -> None: ...

    def observe_idle(self, now: float) -> None:
        """One tick: for every carve-able metal, idle = no listing reports a
        running tenancy and none admitted work since the last tick (or no
        listings at all). First idle stamps idle_since; busy clears it."""

    async def release_idle(self, now: float) -> list[str]:
        """Every metal idle past its limit: delist its listings (reason
        "released: idle"), tell the metal `release`, journal, drop it from
        metal_remotes. Returns the released names."""

    async def release(self, name: str, reason: str) -> dict:
        """The acquire rung inverted, desk-issued. Also a manual door."""

class RemoteMetal:
    async def release(self) -> dict: ...      # residents down the ladder, shift ended
```

## Questions

**Q1. What counts as "nothing runs"?** Tenancies only (what `occupied()`
sees), or tenancies plus admitted traffic?
Recommendation: **both — a metal is idle only if no listing on it reports a
running tenancy AND no listing's `admitted` counter moved since the previous
tick.** A pure client (a measurement cron, an evaluator) has no tenancy and
would otherwise have its host released under it mid-sample; the counter is
two fields on `status()` the arbiter already counts, and comparing counters
across ticks avoids the race of asking "in flight right now?" between two
requests.
If the other branch: tenancies only — simpler, no status change, and a
measurement pass or a long evaluation on a bare inference host is cut off by
the sweep.

> **Samarth:** agree — "i agree with the four question resolution you had"
> (idle = no running tenancy AND no admitted work since the previous tick (the `admitted` counter on `status()`))

**Q2. Where does the idle clock live?** In the desk's memory, or journaled.
Recommendation: **memory.** `idle_since` per metal is set on the first idle
observation and cleared on a busy one; a desk restart forgets it, and the
worst case is one extra tick of idleness before release (the clock restarts
from the rebuild). Journaling every observation is noise in the fleet log for
a timer whose only reader is the next tick.
If the other branch: an `idle` mark per metal in the journal, cleared by a
`busy` mark — exact across restarts, and two new event types the observer
must learn.

> **Samarth:** agree — "i agree with the four question resolution you had"
> (the clock lives in the desk's memory; a restart costs at most one tick)

**Q3. What does release do to the container, and who owns the timer?** Today
the spawned shift keeps the container alive and the venue's
`scaledown_window` would otherwise reclaim it at 15 minutes, before the desk's
30.
Recommendation: **release ends the shift (the `release` verb runs
`MetalService.shutdown()` and returns from the serve loop), so the venue
reclaims the container as a consequence of the desk's decision; and the rule
for deploys is that the venue's idle scaledown is disabled or set no shorter
than the desk's limit, so the desk decides first.** If ADR 0001's Q5b lands
the shift inside the metal's own process, the same verb ends that loop.
If the other branch: the venue's timer stays shorter and reclaims idle metal
on its own; the desk then only cleans up after it (0001 Q5's corpses), and
"the desk owns this" is a description rather than a rule.

> **Samarth:** agree — "i agree with the four question resolution you had"
> (release ends the shift so the venue reclaims the container; deploys set the venue's scaledown no shorter than the desk's limit)

**Q4. Is a released metal re-acquired automatically?** I12 says acquire is a
human's, because new metal costs money.
Recommendation: **yes, by knock, for metal the fleet already owns.** The
human act was the deploy; a released metal is deployed, registered as
inventory, and free. `provision_unit`, finding nothing carve-able that holds
the unit, knocks a released metal whose recorded facts fit; bring-up
re-registers (0001 Q5a), the row updates (0001 Q5), and the carve proceeds.
What no metal — released or live — can hold is still a boot instruction.
I12's wording gains one clause: acquire is a human's for NEW metal.
If the other branch: a released metal is a boot instruction like any other
miss, and every idle period costs a human `up` afterward — the loop this ADR
exists to close stays open at its second half.

> **Samarth:** agree — "i agree with the four question resolution you had"
> (a released metal is re-acquired by knock; I12 gains the word NEW)

## Outcome

**Landed** (CONTEXT #76), on top of ADR 0001, in five commits: the door's two
counters; the metal's third verb; the desk's idle policy; the venues; the
vocabulary.

- **The rule.** `Arbiter.in_flight()` / `admitted()` (a monotone count of
  admissions since birth) on `Host.status()`; `Desk.listing_busy` reads one
  status frame and says busy on a running tenancy, work in flight, or a
  counter that moved; `observe_idle(now)` / `release_idle(now)` are the tick;
  `idle_limit(name)` is the metal's own limit or the desk's
  (`Desk(idle_s=1800.0)`, `IDLE_S`), None PINNED.
- **Release.** `Desk.release(name, reason)` journals the intent first, delists
  every listing, tells the metal, drops it from `metal_remotes` and keeps the
  row in `metal`; `status()` says `released` and `idle_s`; `from_journal`
  replays the event and a later `metal` row clears it. `MetalService.release()`
  + `until_released()` are the metal half — residents down, books emptied, the
  SHIFT LATCH set — with `RemoteMetal.release()` as the client end.
- **The door back.** `provision_unit` = `carve_unit` → `knock_released`
  (`could_hold` against recorded facts) → `carve_unit`; `reacquire` knocks and
  registers the row itself where the container's announce has not landed.
- **What the answers changed** — nothing was overturned; three things the
  questions did not reach were decided in the code and are stated in CONTEXT
  #76: `in_flight` is part of the busy test as well as the counter (one long
  admission spans two ticks and moves no counter); a listing observed for the
  FIRST time is BUSY (release on evidence, never on its absence — one tick's
  cost, which Q2 already priced); and `knock` had to read the plane address
  off the METAL ROW rather than `metal_remotes`, since leaving `metal_remotes`
  is precisely what release means. Two small additions beyond the Touched
  list: the desk serves `release` by hand (`Desk.serve`, `RemoteDesk.release`
  — the sketch's "also a manual door") and `register_metal`'s three-valued
  `idle_s` needed a named absence (`remote.DESK_DEFAULT`).
- **Tests: 889, from 876** (+13). `test_resume.py` untouched and green. The
  promises are pinned one to one: the limit on a fake clock; a pure client's
  traffic through a RemotePool with an empty roster; `idle_s=None` never
  released; the delist + journal + carve-able set + the rebuilt desk; a later
  registration clearing the release; the reaper skipping released metal; a
  placement knocking a released metal and carving (both doors); the shift
  ending on the wire; idempotence.
- **UNPROVEN**, as the non-promises said: everything venue-side. No
  release-then-knock has been seen on Modal — whether the shift's return gets
  the container reclaimed, whether a knock boots a released container and its
  announce lands, what the lingering costs between release and reclaim. Still
  unmeasured: how often idle metal actually sits past 30 minutes on the gsm
  venues; the `release` events on the fleet journal will now say.
