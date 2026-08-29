"""Retention: what a run's store may forget, as a class someone implements.

A run's store only ever grows, and most of what it holds stops having a reader
long before the run ends. Measured on run `803578405216` (a 21M-parameter LoRA
over 50 updates):

    optim     50 files   8.0 GiB    160 MiB each — Adam keeps TWO fp32 moments
    adapters  50 files   4.0 GiB     80 MiB each
    waves     50 files    42 MiB
    postdata  50 files      ~0

Two thirds of that is optimizer state, and exactly one blob of it — the ledger
tail's — has a reader. The same run against a 1B-parameter adapter is ~400 GB
of moments, which is the number this file exists for.

The policy is a CLASS, not a config record: no `optim="tail"` string, no
`every:k` knob. Swappable behavior is a base class here the same way it is for
Environment, PostProcessor, AdapterType and the rollout lowering — the contract
is one abstract method, the rule lives in its docstring, and one implementation
per rule. `KeepRestorable` below is the default and, so far, the only one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class RetentionPolicy(ABC):
    """What may be deleted from a run's store, decided from the ledger alone.

    A policy is a PURE FUNCTION OF THE COMMIT RECORD. It is handed the ledger
    and nothing else — no store, no spec, no clock, no tuning arguments — so it
    can be read, reasoned about and tested without a byte on disk, and the same
    ledger always frees the same blobs. Subclass, implement `expendable`, and
    state the rule you enforce in its docstring.

    What a policy may name is a BLOB VERSION: the triple (section, name,
    version) is the address of `adapters/<name>@<v>.bin` or
    `optim/<name>@<v>.bin` and nothing else. The ledger, the manifest, a sealed
    wave and its postdata are unreachable from here — retention cannot weaken
    the append-only guards, by construction rather than by promise.

    Retention changes what is RECOVERABLE, never what was COMPUTED. Nothing
    here is hashed, nothing reaches an ExperimentSpec, and a swept run's
    `run_id` is the run_id it was born with (I3, I11).
    """

    @abstractmethod
    def expendable(self, ledger: Sequence[Mapping[str, Any]],
                   ) -> Iterable[tuple[str, str, int]]:
        """The (section, name, version) blobs no reader can ever want again.

        `ledger` is every committed entry, oldest first; the last is the tail,
        and a run's live state is exactly what the tail names. Naming a blob
        that is already gone is not an error — the sweep skips absent keys — so
        a policy states a standing truth about the run and never has to know
        which sweep it is.
        """


class KeepRestorable(RetentionPolicy):
    """Keep every adapter forever; keep the ledger tail's optimizer moments.

    Everything else — every optimizer blob below the tail — is expendable, and
    that asymmetry is the whole rule. It is the store-side reading of the one
    `runner/restore.py` already states: serving state is immutable and
    versioned, so any HISTORICAL version can be asked for at any moment, while
    training state is mutable and advances every microbatch, so only the LAST
    COMMITTED version exists at all.

    Adapter blobs have several readers that pin a historical version, and they
    pin it forever:

        the evaluator's `bundle_for` — eval measures a version the run has
            long moved past, rebuilding it from the store
        `restore_bundle_on` — a bounded pool evicts, a container restarts, and
            a version comes back with its content-addressed id as the proof
        `WarmStart` naming `store://<run_id>@<v>` — another experiment's
            starting point, which may be any version this run ever committed

    Deleting an adapter would break restore for that version forever, so no
    policy of this class ever names one. Optimizer moments have exactly ONE
    reader, `restore_tenant`, and it reads them only at the tail — a learner is
    restorable only at a commit boundary, because anywhere else would resurrect
    a policy that never existed. Deleting a stale optim blob therefore cannot
    break anything, and keeping it costs two thirds of the run.

    THE ONE CONSEQUENCE, stated: a WarmStart with `optim="load"` naming a
    MID-RUN version of a swept parent finds no moments and refuses loudly. Warm
    starting from the parent's tail — what `extend` does — is unaffected.
    """

    def expendable(self, ledger: Sequence[Mapping[str, Any]],
                   ) -> tuple[tuple[str, str, int], ...]:
        """Every optim version strictly below the tail's, per delta.

        A delta the tail does not name has no known live version, so nothing of
        it is named either: the tail is the only evidence of what is live, and
        absent evidence keeps bytes.
        """
        if not ledger:
            return ()
        live = _versions(ledger[-1])
        stale = {
            ("optim", name, version)
            for entry in ledger
            for name, version in _versions(entry).items()
            if version < live.get(name, 0)
        }
        return tuple(sorted(stale))


def _versions(entry: Mapping[str, Any]) -> dict[str, int]:
    """One ledger entry's {delta: version} map — the policy version it sealed.
    An entry carrying none (or a malformed one) contributes nothing."""
    versions = entry.get("versions")
    if not isinstance(versions, Mapping):
        return {}
    return {str(name): int(version) for name, version in versions.items()}


@dataclass(frozen=True)
class Swept:
    """What one sweep actually freed: the blob versions deleted, in the order
    they were named, and the bytes they occupied. Both are facts about a
    directory at a moment, never about the experiment — nothing here is
    journaled into a run, written to the ledger, or hashed."""

    blobs: tuple[tuple[str, str, int], ...]
    freed: int

    def __len__(self) -> int:
        return len(self.blobs)


DEFAULT_RETENTION = KeepRestorable()
"""The policy the Trainer sweeps with unless it is handed another one."""
