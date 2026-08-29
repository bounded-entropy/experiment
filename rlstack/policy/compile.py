"""Bundle compilation: a bank, lowered into the only thing that crosses back.

A Bundle is the content-addressed artifact `sync_weights` hands to the engine —
the single data channel from the training world to the inference world (I2). It
pins the FULL policy version map (trainable and frozen deltas alike) but carries
payloads only for servable deltas: a trainer-only adapter type (value_head)
versions like any other delta yet never ships. A payload is whatever the adapter
type's `emit` produces; identity is the content hash of the version map plus the
payload digests, so identical banks compile to identical bundle ids everywhere.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from rlstack.policy.adapters.base import Mechanism
from rlstack.registry import ADAPTER_TYPES
from rlstack.spec.canonical import content_hash


@dataclass(frozen=True)
class Bundle:
    """One compiled policy: id, full version map, servable payloads.

    `adapter_types` (payload name -> the registered adapter type that lowers it)
    makes the bundle self-describing: an engine routes every payload without ever
    seeing the spec (group_by_adapter_type below).
    """

    bundle_id: str
    policy_version: Mapping[str, int]
    payloads: Mapping[str, bytes] = field(repr=False, default_factory=dict)
    adapter_types: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def pin(cls, bundle_id: str, policy_version: Mapping[str, int]) -> "Bundle":
        """A payload-less Bundle used purely as an ADDRESS: a request pins the
        id, and the serving engine already holds the payloads — add_bundle is
        availability, and it lands before the ledger commit that makes the id
        visible."""
        return cls(bundle_id, dict(policy_version))


def compile_bundle(
    payloads: Mapping[str, bytes],
    policy_version: Mapping[str, int],
    servable: Iterable[str],
    adapter_types: Mapping[str, str] | None = None,
) -> Bundle:
    """Lower emitted payloads into a content-addressed Bundle.

    `payloads` is everything the learner emitted; only names in `servable`
    ship, each labeled with its adapter type from `adapter_types`. bundle_id =
    h(version map ⊕ served payload digests), 12 hex chars, prefixed for
    greppability.
    """
    served = {name: payloads[name] for name in sorted(servable)}
    digest_map = {name: hashlib.sha256(data).hexdigest() for name, data in served.items()}
    bundle_id = "bundle:" + content_hash(
        {"versions": dict(policy_version), "payloads": digest_map}
    )[:12]
    return Bundle(bundle_id=bundle_id, policy_version=dict(policy_version),
                  payloads=served,
                  adapter_types={name: (adapter_types or {})[name] for name in served}
                  if adapter_types else {})


ReadBlob = Callable[[str, str, int], bytes]     # (section, name, version) -> bytes


def restore_bundle(policy_version: Mapping[str, int], bundle_id: str,
                   read_blob: ReadBlob, servable: Iterable[str],
                   adapter_types: Mapping[str, str] | None = None) -> Bundle:
    """Rebuild a committed bundle from the store, and PROVE it is the same one.

    A pinned version map plus the store is all a bundle ever was: read each
    servable delta's blob at its pinned version, recompile, and compare ids.
    Since bundle_id = h(version map ⊕ payload digests), a matching id is proof
    the rebuild IS the policy that was committed — no trust, no drift.

    This is why serving residency need not be durable. An evicted bundle, a
    restarted container and a resumed run are one situation with one answer, and
    the caller that decides WHEN to ask lives in runner/restore.py.

    `read_blob` is a callable rather than a store so this file keeps knowing
    nothing about storage — the caller chooses the run, which is what lets a
    warm start read a parent's blobs through the same door.
    """
    payloads = {name: read_blob("adapters", name, policy_version[name])
                for name in sorted(servable)}
    rebuilt = compile_bundle(payloads, policy_version, servable, adapter_types)
    if rebuilt.bundle_id != bundle_id:
        raise ValueError(
            f"restored bundle {rebuilt.bundle_id} is not the committed "
            f"{bundle_id}: the blobs at this version map do not compile to the "
            f"policy that was sealed under it")
    return rebuilt


def group_by_adapter_type(bundle: Bundle) -> dict[str, dict[str, bytes]]:
    """Route a bundle's payloads to their ADAPTER TYPES — the engine bus's
    dispatch input (#48).

    Each adapter type's rollout lowering receives ALL payloads of its own entries
    and attaches them jointly (punica merges peft fragments; a soft prompt's
    blocks are segments of one virtual prompt). Payloads arrive in bank order,
    since compile_bundle sorts them, so an adapter type that occupies prompt
    positions concatenates in it. A payload with no adapter-type label, or with a
    trainer-only adapter type, is a compile error: a servable payload must be
    routable.
    """
    grouped: dict[str, dict[str, bytes]] = {}
    for name, data in bundle.payloads.items():
        if name not in bundle.adapter_types:
            raise ValueError(
                f"payload {name!r} in {bundle.bundle_id} carries no "
                f"adapter-type label")
        adapter_type = bundle.adapter_types[name]
        if ADAPTER_TYPES.get(adapter_type).instance.serving is None:
            raise ValueError(
                f"payload {name!r} ({adapter_type}) is trainer-only yet "
                f"shipped in {bundle.bundle_id}")
        grouped.setdefault(adapter_type, {})[name] = data
    return grouped


def group_by_mechanism(bundle: Bundle) -> dict[Mechanism, dict[str, bytes]]:
    """The same routing, keyed by the serving MECHANISM instead of the adapter
    type.

    The inventory view: which levers this bundle would be served through. The
    bus itself dispatches by adapter type (group_by_adapter_type), calling each
    adapter type's own rollout lowering.
    """
    grouped: dict[Mechanism, dict[str, bytes]] = {}
    for adapter_type, payloads in group_by_adapter_type(bundle).items():
        serving = ADAPTER_TYPES.get(adapter_type).instance.serving
        grouped.setdefault(serving, {}).update(payloads)
    return grouped
