"""Names: which library adapters a wave needs, and the wait until they exist.

A route may name a frozen LIBRARY set, `lib:<name>`, alone or stacked
(`lib:<name>+dreamer`), and `<name>` is a named adapter in the store (ADR
0019): written once by some fit run, under the experiment's subdir, possibly
while the run that needs it is already going. This module is the consumer's
half of that overlap:

    lib_names(roles)     the names a set of role / route strings mention
    wave_lib_names(...)  the names one planned wave mentions, leaf by leaf
    row_lib_names(...)   the names a realized wave's TRAINED rows carry
    names_ready(...)     the wait: returns when every name is present,
                         refuses when one is orphaned, waits otherwise

THE WAIT NEVER STARTS FROM INIT. A consumer of a missing name blocks; it is
released by the bytes, or refused by `NamedAdapterOrphaned` when the run that
promised them ended without writing them. `promised` and `unknown` both wait —
refusing on `unknown` would force submit order and remove the overlap the
names exist for — and every NOTE_EVERY_S the wait says what it is waiting on,
because a wait nobody can see is a hang.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from rlstack.data.plan import EVAL, PlanError, RunPlan, Sample, WavePlan, Waves
from rlstack.data.stores.base import (
    ORPHANED, PRESENT, RunHandle, Store, StoreError, check_name,
)
from rlstack.data.trajectory import Task
from rlstack.runner.meters import HostJournal
from rlstack.runner.signals import RunSignals, store_work

LIB_PREFIX = "lib:"
"""A route part naming a frozen library set: `lib:<name>`."""

STACK = "+"
"""What joins the parts of a stacked route: `lib:<name>+dreamer`. Outside the
name grammar, so a route always cuts cleanly."""

NOTE_EVERY_S = 600.0
"""How often a wait that is still waiting says so."""

ENDED_CHECK_EVERY_S = 60.0
"""How often a wait asks whether a missing name's WRITER ENDED. Presence is
asked every beat (one small read per missing name); whether the writer is
done, stopped or failed costs its plan, its ledger and the fleet journal, so
it is asked on the first beat and then at this pace — an orphan is refused
within a minute, not within a poll."""


class NamedAdapterOrphaned(StoreError):
    """A needed name will never exist: the run that promised it is done,
    stopped or failed, and its bytes are absent. Names the adapter and its
    writer, because the repair is to resubmit that fit."""


# ---------------------------------------------------------------------------
# which names
# ---------------------------------------------------------------------------

def lib_names(roles: Iterable[str]) -> tuple[str, ...]:
    """The `<name>` of every `lib:<name>` part of these role (or route)
    strings, first mention first, each once. A role may be compound
    (`lib:<name>+dreamer`); every part is read. A name outside the store's
    grammar is refused HERE, which is the grammar's gate: plans are cas blobs
    at Phase 0, so the roles that read them are the first to see a name."""
    names: dict[str, None] = {}
    for role in roles:
        for part in str(role).split(STACK):
            if part.startswith(LIB_PREFIX):
                try:
                    names[check_name(part[len(LIB_PREFIX):])] = None
                except StoreError as refused:
                    raise PlanError(f"role {role!r}: {refused}") from None
    return tuple(names)


def wave_lib_names(entry: Waves, tasks: Mapping[str, Task] | None = None) -> tuple[str, ...]:
    """The library names ONE PLANNED WAVE mentions: every leaf's role, and —
    for a Sample leaf whose task is known — the task's `meta["route"]`, which
    is what an answering environment samples under (an `eval` answer under a
    library set can say so nowhere else: its role is taken). A WaveRef
    mentions none; the wave it names carries its routes in its rows."""
    if not isinstance(entry, WavePlan):
        return ()
    said: list[str] = []
    for leaf in entry.leaves():
        said.append(leaf.role)
        if isinstance(leaf, Sample) and tasks is not None and leaf.task_id in tasks:
            route = tasks[leaf.task_id].meta.get("route")
            if route is not None:
                said.append(str(route))
    return lib_names(said)


def row_lib_names(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """The library names a REALIZED wave's trained rows carry, read off each
    turn's `route` fact — the leaf's role where it named a set, the route the
    row was sampled under where it did not. This is what the learner will be
    asked to replay, so it is what must be loaded. An `eval` row is scored
    and never forwarded (`trained_rows`), so its route loads nothing."""
    said: list[str] = []
    for row in rows:
        facts = [turn.get("turn_extras") or {} for turn in row.get("turns", ())]
        if any(extras.get("role") == EVAL for extras in facts):
            continue
        said.extend(str(extras["route"]) for extras in facts if extras.get("route"))
    return lib_names(said)


def check_plan_names(plan: RunPlan) -> None:
    """Every `lib:<name>` a plan's roles mention is a legal name — asked once,
    at a role's birth, so a misspelt name refuses the run before its first
    wave instead of at the wave that reaches it."""
    for index, entry in enumerate(plan.waves, start=1):
        try:
            wave_lib_names(entry)
        except PlanError as refused:
            raise PlanError(f"wave {index}: {refused}") from None


def names_subdir(run: RunHandle) -> str:
    """WHERE A RUN'S NAMES LIVE: the subdir it was filed under (ADR 0010 —
    runs/<subdir>/<run_id>), which is the experiment's folder and therefore
    the one place a fit run and its consumers both know without being told.
    A run filed at the store's root has no experiment folder, so a plan of
    its that mentions a library name is refused by that name."""
    home = run.run_dir or f"runs/{run.run_id}"
    subdir = "/".join(home.split("/")[1:-1])
    if not subdir:
        raise PlanError(
            f"run {run.run_id!r} is filed at the store's root: named adapters "
            f"live under an experiment's subdir (runs/<subdir>/names/), so a "
            f"run whose plans mention lib:<name> must be submitted under one")
    return subdir


# ---------------------------------------------------------------------------
# the wait
# ---------------------------------------------------------------------------

class NamesWait:
    """One wait's memory: which names are still missing, when it last asked
    whether their writers ended, when it last said it was waiting. The
    predicate (`all_present`) is a read of the store like every runner
    condition; this object only paces the expensive half of it."""

    def __init__(self, store: Store, subdir: str, names: Sequence[str],
                 journal: HostJournal | None,
                 clock: Callable[[], float]) -> None:
        self.store = store
        self.subdir = subdir
        self.missing = tuple(dict.fromkeys(names))
        self.journal = journal
        self.clock = clock
        self.began = clock()
        self.noted = self.began
        self.asked_ended: float | None = None

    def all_present(self) -> bool:
        """True once every name is present. Raises NamedAdapterOrphaned for a
        name whose promised writer ended without it."""
        self.missing = tuple(name for name in self.missing
                             if self.store.named_meta(self.subdir, name) is None)
        if not self.missing:
            return True
        now = self.clock()
        if self.asked_ended is None or now - self.asked_ended >= ENDED_CHECK_EVERY_S:
            self.asked_ended = now
            self.refuse_orphans()
        if self.missing and now - self.noted >= NOTE_EVERY_S:
            self.noted = now
            self.note(now)
        return not self.missing

    def refuse_orphans(self) -> None:
        """The full question, per missing name: a name that turned present
        between the two reads leaves the wait; an orphan ends it."""
        still: list[str] = []
        for name in self.missing:
            state = self.store.named_state(self.subdir, name)
            if state == ORPHANED:
                writer = self.store.named_writer(self.subdir, name)
                raise NamedAdapterOrphaned(
                    f"adapter {name!r} under {self.subdir!r} will never exist: "
                    f"its promised writer {writer!r} is done, stopped or failed "
                    f"and wrote no bytes for it — resubmit that fit (a new "
                    f"promise takes the name over)")
            if state != PRESENT:
                still.append(name)
        self.missing = tuple(still)

    def note(self, now: float) -> None:
        """Say what the wait is waiting on: to the host's journal where the
        run has one, and to the log either way."""
        states = {name: self.store.named_state(self.subdir, name)
                  for name in self.missing}
        waited = now - self.began
        print(f"[names] {self.subdir}: waited {waited:.0f}s on "
              f"{len(states)} named adapter(s): {states}", flush=True)
        if self.journal is not None:
            self.journal.append({"event": "names-wait", "t": time.time(),
                                 "subdir": self.subdir, "waited_s": waited,
                                 "missing": states})


async def names_ready(store: Store, subdir: str, names: Sequence[str],
                      signals: RunSignals, *,
                      stopped: Callable[[], bool] | None = None,
                      journal: HostJournal | None = None,
                      clock: Callable[[], float] = time.monotonic) -> None:
    """Return when every name is `present`; raise NamedAdapterOrphaned on
    `orphaned`; keep waiting on `promised` and `unknown`, with a note every
    NOTE_EVERY_S.

    The wait is the blackboard's: the predicate re-reads the store off the
    loop (`store_work`) and sleeps one beat of `signals` between reads, so a
    fit run in this process wakes it at once and one on other metal within a
    poll. `stopped` is the second way out a draining runner needs (ADR 0014,
    Q6): when it answers True the wait returns with names still missing, and
    the caller — who passed it — reads its own stop before using them.
    """
    if not names:
        return
    waiting = NamesWait(store, subdir, names, journal, clock)
    while True:
        if stopped is not None and stopped():
            return
        if await store_work(waiting.all_present):
            return
        await signals.wait_once()
