"""Where the stores are: locators, and the roots a TOP directory holds.

store_for(locator): a store is NAMED by a locator, and the reader must run
somewhere that locator resolves.

    /path, file:///path   LocalStore — wherever that filesystem is mounted
                          (a Modal volume IS one, inside a container)
    s3://bucket/prefix    an S3Store subclass, when it lands — resolves
                          anywhere
    modal://volume        does NOT resolve locally, on purpose: run the reader
                          beside the volume instead of pretending to fetch it

FOLDERS ARE STORE ROOTS (#58). A run's organizational location is the
directory its store was constructed with — chosen at birth, never moved,
never renamed. So the observer takes a TOP directory and finds the roots
beneath it: a directory holding runs/, hosts/, fleet/ or annotations.jsonl IS
a root, and discovery never descends into one (everything under a root is the
store's own key tree, not more folders). A top that is itself a root is the
degenerate single-store case — one Root whose folder is "" — and that is
exactly what the deployed observer's /store mount is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rlstack.data.stores.base import Store
from rlstack.data.stores.local import LocalStore

# What makes a directory a store root: any part of the store's key tree.
ROOT_MARKS = ("runs", "hosts", "fleet", "annotations.jsonl")


@dataclass(frozen=True)
class Root:
    """One store root and WHERE it sits: `folder` is its path relative to the
    top directory the reader was given ("" for the top itself).

    The folder is organizational only — it is never hashed, never part of a
    run_id, and never moved. It is, however, the ADDRESS: the same spec
    submitted under two folders yields the same run_id in both, so every link
    the observer emits carries the folder."""

    folder: str
    store: Store


def store_for(locator: str) -> Store:
    if locator.startswith("file://"):
        return LocalStore(locator[len("file://"):])
    if locator.startswith("s3://"):
        raise NotImplementedError(
            f"{locator}: no S3 store backend yet (the Store ABC's byte verbs "
            f"are where it lands — data/stores/, one file per backend)")
    if locator.startswith("modal://"):
        raise NotImplementedError(
            f"{locator} does not resolve outside a container. Run the reader "
            f"beside the volume: `modal run deploy/modal_app.py::hosts`, or "
            f"an observer deployed with the volume mounted.")
    return LocalStore(locator)


def rooted(items: Sequence[Store | Root]) -> list[Root]:
    """Stores or Roots, read as Roots. A bare Store is the degenerate case —
    one root, folder "" — which is how the deployed UI hands its volume in."""
    return [item if isinstance(item, Root) else Root("", item) for item in items]


def is_root(path: Path) -> bool:
    """A directory is a store root when any part of the key tree is in it."""
    return any((path / mark).exists() for mark in ROOT_MARKS)


def roots_under(locator: str) -> list[Root]:
    """Every store root at or beneath one TOP directory, folders relative
    to it.

    Three answers, one rule. The top IS a root (or holds no root at all, a
    directory a fresh store is about to be born in) → the degenerate case,
    one Root("") over the locator exactly as store_for names it. Otherwise
    every root found below, in path order, and NEVER a root inside another:
    the first directory that carries a mark ends the descent, because
    everything under it is that store's key tree.

    Non-local locators keep their old answer — store_for raises for them, and
    a bucket is not a directory to walk."""
    top = Path(locator[len("file://"):] if locator.startswith("file://") else locator)
    if not top.is_dir() or is_root(top):
        return [Root("", store_for(locator))]
    found = _descend(top, top)
    return found or [Root("", store_for(locator))]


def _descend(top: Path, here: Path) -> list[Root]:
    """Roots below `here`, stopping at each one. Hidden directories are not
    ours; a directory the reader may not open is skipped, not fatal."""
    out: list[Root] = []
    try:
        children = sorted(path for path in here.iterdir()
                          if path.is_dir() and not path.name.startswith("."))
    except OSError:
        return out
    for child in children:
        if is_root(child):
            out.append(Root(child.relative_to(top).as_posix(), LocalStore(child)))
        else:
            out.extend(_descend(top, child))
    return out


def roots_for(locators: Sequence[str]) -> list[Root]:
    """The roots a reader was given: one top directory per locator.

    A FOLDER MUST NAME EXACTLY ONE ROOT — it is the address every link
    carries. With one top that is free; with several, each top's own name
    prefixes its folders, and two tops with the same name are refused rather
    than silently collapsed onto one address."""
    if len(locators) == 1:
        return roots_under(locators[0])
    out: list[Root] = []
    seen: set[str] = set()
    for locator in locators:
        prefix = Path(locator.rstrip("/")).name or locator
        if prefix in seen:
            raise ValueError(
                f"two tops named {prefix!r}: a folder addresses one root, so "
                f"give tops with distinct names (or one top above both)")
        seen.add(prefix)
        out.extend(Root(f"{prefix}/{root.folder}" if root.folder else prefix,
                        root.store)
                   for root in roots_under(locator))
    return out
