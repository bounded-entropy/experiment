"""Restore: the store is the backstop for everything a pool or a learner holds.

Residency is not durable and never has to be. A committed version's blobs are
in the store forever, so anything holding that version — an engine's attached
bundle, a learner's params and moments — can be dropped and rebuilt. Both
directions are the same verb, `restore`, and they differ only in which blobs
they read and where the bytes land:

    restore_bundle_on   version map -> servable adapter blobs -> attached on an
                        ENGINE, with the content-addressed id as the proof
                        (policy/compile.py owns the proof itself)
    restore_tenant      version map -> trainable adapter + optim blobs ->
                        loaded into a LEARNER's params and Adam moments

The asymmetry worth knowing: serving state is immutable and versioned, so any
historical version can come back at any moment; training state is mutable and
advances every microbatch, so only the LAST COMMITTED version exists in the
store and a learner is restorable (and evictable) only at a commit boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rlstack.policy.compile import Bundle, ReadBlob, restore_bundle
from rlstack.runner.interfaces import Engine, Learner


class BundleUnavailable(RuntimeError):
    """A pool does not hold a pinned version and no store can rebuild it
    (ADR 0014, Part B): the version was committed but not CHECKPOINTED, so
    its blobs never existed, and the engine that held it is gone. An
    infrastructure death, not the experiment's: the tenancy parks and the
    retry resumes from the checkpoint, where the blobs are."""


def restore_bundle_on(engine: Engine, pin: Bundle, read_blob: ReadBlob,
                      servable: Iterable[str],
                      adapter_types: Mapping[str, str] | None = None) -> None:
    """Make sure this engine can serve `pin`, rebuilding it if it cannot.

    ASK FIRST: the common path is that the pool still holds the bundle, and
    that path must cost one cheap question rather than a store read. On a miss —
    an eviction, a restarted container, a pool that never saw this version —
    the blobs come back, recompile, and re-register. A miss on a version the
    store never checkpointed is `BundleUnavailable` (ADR 0014).

    A pin carries the id and the full version map and no payloads, which is
    exactly what a request carries and what every ledger line records; that is
    what makes this callable from anywhere holding a committed version.
    """
    if engine.knows_bundle(pin.bundle_id):
        return
    try:
        rebuilt = restore_bundle(pin.policy_version, pin.bundle_id,
                                 read_blob, servable, adapter_types)
    except FileNotFoundError as missing:
        raise BundleUnavailable(
            f"bundle {pin.bundle_id} at {dict(pin.policy_version)} is not resident "
            f"on the pool and not checkpointed on the store ({missing}): the pool "
            f"that held it is gone — resume from the last checkpoint") from None
    engine.add_bundle(rebuilt)


def restore_tenant(learner: Learner, tenant: str,
                   policy_version: Mapping[str, int], read_blob: ReadBlob,
                   trainable: Iterable[str]) -> None:
    """Put a tenant's committed training state back on the learner.

    The trainer's own path back, and resume's: the deltas AND the optimizer
    moments, because a learner that keeps training needs the moments and a
    serving engine never does. Only trainable entries move — a frozen delta
    does not advance, so its version 0 blob is already what it will always be.

    Safe only at a commit boundary, and that is the whole rule: the ledger line
    is written after every microbatch of a wave has landed, so the blobs it
    names are an exact copy of the state. Restoring from anywhere else would
    resurrect a policy that never existed.
    """
    names = sorted(trainable)
    learner.load(
        tenant,
        adapters={name: read_blob("adapters", name, policy_version[name])
                  for name in names},
        optim={name: read_blob("optim", name, policy_version[name])
               for name in names},
    )
