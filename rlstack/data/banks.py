"""A BANK: a growing, versioned set of named adapters (memories) that runs
draw from while it grows.

A publisher (a fit pipeline) seals memories as named adapters (ADR 0019) and
then publishes a BANK VERSION: version v lists the memories it ADDS to
version v-1, each with its adapter's seal (`sha256`) and whatever the
publisher records about it (its group, split, told facts, probe results).
Versions are append-only and write-once (`Store.write_bank_version`), so a
version means the same memories forever and a consumer that logs
(bank, version) — or the memory names it drew — has said exactly what it saw.

    BankMemory      one memory as a version lists it
    BankVersion     one version's row: what it adds, what it RETIRES, its own facts
    publish         the next version, refused unless every added memory is sealed
    bank_at         the memories as of a version (the latest when unasked)
    draw            a seeded draw of n memories from a view, by split

A memory is never deleted: a later version RETIRES it, with a reason (its
facts were superseded, say). A view leaves retired memories out of its
draws and says when and why each left.

A consumer that reads "the latest version" at each update reads `bank_at`
once per update, draws, and puts `view.version` and the drawn names in what
it seals (the task meta), so its logs answer which version it drew from.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rlstack.data.stores.base import Store, StoreError


@dataclass(frozen=True)
class BankMemory:
    """One memory: a named adapter under the bank's subdir, sealed as
    `sha256`, in `split`. `record` is the publisher's free description
    (facts, told set, probes) — carried, never interpreted here."""

    name: str
    sha256: str
    split: str
    record: Mapping[str, Any] = field(default_factory=dict)

    def row(self) -> dict:
        return {"name": self.name, "sha256": self.sha256, "split": self.split,
                "record": dict(self.record)}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "BankMemory":
        return cls(str(row["name"]), str(row["sha256"]), str(row["split"]),
                   dict(row.get("record", {})))


@dataclass(frozen=True)
class BankVersion:
    """Version `version` of a bank: the memories it adds, when it was made
    and by what (`about`: code commit, source shards, timings)."""

    bank: str
    version: int
    created: str
    added: tuple[BankMemory, ...]
    total: int                         # memories in use after this version (retired excluded)
    about: Mapping[str, Any] = field(default_factory=dict)
    retired: Mapping[str, str] = field(default_factory=dict)   # name -> why

    def row(self) -> dict:
        row = {"bank": self.bank, "version": self.version, "created": self.created,
               "added": [m.row() for m in self.added], "total": self.total,
               "about": dict(self.about)}
        if self.retired:
            row["retired"] = dict(self.retired)
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "BankVersion":
        return cls(str(row["bank"]), int(row["version"]), str(row["created"]),
                   tuple(BankMemory.from_row(m) for m in row["added"]), int(row["total"]),
                   dict(row.get("about", {})), dict(row.get("retired", {})))


@dataclass(frozen=True)
class BankView:
    """The bank as of `version`: every memory of versions 1..version still
    in use, in publication order; `retired` names each one a version took
    out, with (that version, why)."""

    bank: str
    version: int
    memories: tuple[BankMemory, ...]
    retired: Mapping[str, tuple[int, str]] = field(default_factory=dict)

    def names(self, split: str | None = None) -> list[str]:
        return [m.name for m in self.memories if split is None or m.split == split]

    def memory(self, name: str) -> BankMemory:
        for m in self.memories:
            if m.name == name:
                return m
        raise KeyError(f"{name!r} is not in bank {self.bank!r} at version {self.version}")


def latest_version(store: Store, subdir: str, bank: str) -> int:
    """The newest written version, 0 for a bank with none."""
    versions = store.bank_versions(subdir, bank)
    return versions[-1] if versions else 0


def publish(store: Store, subdir: str, bank: str, added: Sequence[BankMemory], *,
            created: str, about: Mapping[str, Any] | None = None,
            retired: Mapping[str, str] | None = None) -> BankVersion:
    """Write the next version, adding `added` and retiring `retired` (name
    -> why). EVERY ADDED MEMORY IS SEALED FIRST: each name must be a present
    named adapter under `subdir` whose seal is the `sha256` given, and none
    may already be in the bank, in use or retired — so a version never lists
    an adapter a reader cannot load, nor one twice. Only a memory in use can
    be retired, and only with a reason."""
    retired = dict(retired or {})
    if not added and not retired:
        raise StoreError("a bank version adds or retires at least one memory")
    current = bank_at(store, subdir, bank)
    in_use = {m.name for m in current.memories}
    for name, why in retired.items():
        if name not in in_use:
            raise StoreError(f"memory {name!r} is not in use in bank {bank!r}: nothing to retire")
        if not str(why).strip():
            raise StoreError(f"retiring {name!r} needs a reason")
    held = in_use | set(current.retired)
    for m in added:
        meta = store.named_meta(subdir, m.name)
        if meta is None:
            raise StoreError(f"memory {m.name!r} is not a present named adapter under {subdir!r}")
        if meta.get("sha256") != m.sha256:
            raise StoreError(f"memory {m.name!r} is sealed as {str(meta.get('sha256'))[:12]}, "
                             f"not {m.sha256[:12]}")
        if m.name in held:
            raise StoreError(f"memory {m.name!r} is already in bank {bank!r}")
        held.add(m.name)
    version = BankVersion(bank, current.version + 1, created, tuple(added),
                          len(current.memories) + len(added) - len(retired), dict(about or {}),
                          retired)
    store.write_bank_version(subdir, bank, version.version, version.row())
    return version


def read_version(store: Store, subdir: str, bank: str, version: int) -> BankVersion:
    row = store.read_bank_version(subdir, bank, version)
    if row is None:
        raise StoreError(f"bank {bank!r} has no version {version} under {subdir!r}")
    return BankVersion.from_row(row)


def bank_at(store: Store, subdir: str, bank: str, version: int | None = None) -> BankView:
    """The memories as of `version` (None: the latest; 0: the empty bank)."""
    upto = latest_version(store, subdir, bank) if version is None else version
    memories: list[BankMemory] = []
    retired: dict[str, tuple[int, str]] = {}
    for v in range(1, upto + 1):
        row = read_version(store, subdir, bank, v)
        memories += row.added
        retired.update({name: (v, why) for name, why in row.retired.items()})
    return BankView(bank, upto, tuple(m for m in memories if m.name not in retired), retired)


def draw(view: BankView, n: int, *, seed: str, split: str = "train") -> list[str]:
    """`n` distinct memory names of `split`, drawn uniformly by a generator
    seeded with `seed` and the view's version — so the same (seed, version)
    draws the same names on a resume."""
    names = view.names(split)
    if len(names) < n:
        raise ValueError(f"bank {view.bank!r} v{view.version} holds {len(names)} "
                         f"{split} memories; {n} were asked")
    return random.Random(f"bank-draw:{view.bank}:{view.version}:{seed}").sample(names, n)
