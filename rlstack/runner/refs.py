"""Refs: where an already-sealed trajectory lives, and how it is read back.

A plan's `Replay` leaf names one trajectory by ref, and this module is the one
place that knows what a ref means. Four locations, one grammar
(`<location>#<index>`, the index picking one row out of a sealed wave):

    self://rollouts/<r>            this run's own generated wave r
    store://<run_id>/waves/<u>     another run's sealed wave
    store://<run_id>/rollouts/<r>  another run's sealed ROLLOUT (ADR 0006
                                   Part B): what a generation-only run
                                   leaves, since only a Trainer realizes
                                   waves/ — a teacher's output, replayed
    cas://<sha>                    a fixed file of sealed trajectory rows

ONLY `self://` CAN ANSWER "NOT YET". The other two are sealed before this run
starts, which is what lets the submit gate check them while a plan is still
just data — a missing parent wave becomes a Phase-0 refusal instead of a
mid-run raise. Reads are cached per location, so a wave assembled out of a
hundred leaves of one file reads that file once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from rlstack.data.plan import PlanError
from rlstack.data.stores.base import RunHandle, Store

SELF = "self://rollouts/"
STORE = "store://"
CAS = "cas://"

STORE_SECTIONS = ("waves", "rollouts")
"""What a store ref may name inside another run: the waves a Trainer realized,
or the rollouts a Generator sealed. Both are sealed-forever artifacts, which is
what lets either be read without attaching to that run."""


@dataclass(frozen=True)
class Ref:
    """A parsed ref: where the rows are, and which one (None = all of them)."""

    location: str
    index: int | None

    @property
    def pending_allowed(self) -> bool:
        """Whether "not yet" is a legal answer here — true only for this run's
        own rollouts, which a Generator is still producing."""
        return self.location.startswith(SELF)


def parse(ref: str) -> Ref:
    """`<location>#<index>` → Ref. The index is optional; a ref without one
    names a whole sealed wave, which is what a WaveRef expands over."""
    location, _, index = ref.partition("#")
    if not location.startswith((SELF, STORE, CAS)):
        raise PlanError(
            f"unknown ref {ref!r}: a sealed trajectory lives at "
            f"{SELF}<r>, {STORE}<run_id>/waves/<u>, "
            f"{STORE}<run_id>/rollouts/<r>, or {CAS}<sha>")
    if index and not index.isdigit():
        raise PlanError(f"ref {ref!r} has a non-numeric row index {index!r}")
    return Ref(location, int(index) if index else None)


class RefReader:
    """Reads refs against one run's store; caches each location's rows.

    The reader is per-run because `self://` means THIS run — the same ref text
    resolves differently in a different run, which is exactly what makes a plan
    portable between them.
    """

    def __init__(self, store: Store, run: RunHandle) -> None:
        self._store = store
        self._run = run
        self._cache: dict[str, list[dict]] = {}

    def rows(self, location: str) -> list[dict] | None:
        """Every sealed row at `location`, or None when it is not sealed yet."""
        if location in self._cache:
            return self._cache[location]
        rows = self._read(location)
        if rows is not None:
            self._cache[location] = rows
        return rows

    def _read(self, location: str) -> list[dict] | None:
        if location.startswith(SELF):
            index = location[len(SELF):]
            if not index.isdigit():
                raise PlanError(
                    f"{location!r}: a self ref names a rollout INDEX, {SELF}<r>")
            try:
                return self._run.read_rollout(int(index))
            except FileNotFoundError:
                return None            # the Generator has not sealed it yet
        if location.startswith(STORE):
            return self._store_rows(location)
        raw = self._store.cas_get(location).decode("utf-8")
        return [json.loads(line) for line in raw.splitlines() if line]

    def _store_rows(self, location: str) -> list[dict]:
        """ANOTHER RUN'S SEALED WAVE OR ROLLOUT, read without attaching to it.

        `waves/<u>` is what that run's Trainer realized; `rollouts/<r>` is what
        its Generator sealed, and is the only thing a generation-only run
        leaves (ADR 0006 Part B — a run with no Trainer never writes waves/).
        Both are sealed forever, so the read is a PEEK: opening the parent
        would attach to it, and attaching sweeps work its ledger has not
        committed — which a reader has no business doing to another run.
        "Not yet" is not an answer here; only `self://` may say that.
        """
        run_id, _, tail = location[len(STORE):].partition("/")
        section, _, index = tail.partition("/")
        if section not in STORE_SECTIONS or not index.isdigit():
            raise PlanError(
                f"{location!r}: a store ref names a run's sealed wave or "
                f"rollout, {STORE}<run_id>/waves/<u> or "
                f"{STORE}<run_id>/rollouts/<r>")
        rows = (self._store.peek_wave(run_id, int(index)) if section == "waves"
                else self._store.peek_rollout(run_id, int(index)))
        if rows is None:
            raise PlanError(
                f"{location!r}: run {run_id!r} has sealed no {section[:-1]} "
                f"{int(index)} — a store ref names data that already exists, "
                f"which is what lets a plan naming it be checked before this "
                f"run starts")
        return rows

    def row(self, ref: str) -> dict | None:
        """One sealed row, or None when its wave is not sealed yet."""
        parsed = parse(ref)
        if parsed.index is None:
            raise PlanError(
                f"{ref!r} names a whole wave; a leaf takes one row "
                f"({ref}#<index>)")
        rows = self.rows(parsed.location)
        if rows is None:
            return None
        if parsed.index >= len(rows):
            raise PlanError(
                f"{ref!r} is out of range: {parsed.location} sealed "
                f"{len(rows)} trajectories")
        return rows[parsed.index]
