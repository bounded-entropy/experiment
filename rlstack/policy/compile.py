"""Bundle compilation: a bank, lowered into the only thing that crosses back.

A Bundle is the content-addressed artifact `sync_weights` hands to the engine —
the single data channel from the training world to the inference world (I2). It
pins the FULL policy version map (trainable and frozen deltas alike) but carries
payloads only for servable deltas: a trainer-only kind (value_head) versions
like any other delta yet never ships. A payload is whatever the kind's `emit`
produces; identity is the content hash of the version map plus the payload
digests, so identical banks compile to identical bundle ids everywhere.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from rlstack.policy.adapters.base import Mechanism
from rlstack.registry import ADAPTERS
from rlstack.spec.canonical import content_hash


@dataclass(frozen=True)
class Bundle:
    """One compiled policy: id, full version map, servable payloads.

    `kinds` (payload name -> the registered kind that lowers it) makes the
    bundle self-describing: an engine routes every payload without ever seeing
    the spec (group_by_kind below).
    """

    bundle_id: str
    policy_version: Mapping[str, int]
    payloads: Mapping[str, bytes] = field(repr=False, default_factory=dict)
    kinds: Mapping[str, str] = field(default_factory=dict)

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
    kinds: Mapping[str, str] | None = None,
) -> Bundle:
    """Lower emitted payloads into a content-addressed Bundle.

    `payloads` is everything the learner emitted; only names in `servable`
    ship, each labeled with its kind from `kinds`. bundle_id = h(version map ⊕
    served payload digests), 12 hex chars, prefixed for greppability.
    """
    served = {name: payloads[name] for name in sorted(servable)}
    digest_map = {name: hashlib.sha256(data).hexdigest() for name, data in served.items()}
    bundle_id = "bundle:" + content_hash(
        {"versions": dict(policy_version), "payloads": digest_map}
    )[:12]
    return Bundle(bundle_id=bundle_id, policy_version=dict(policy_version),
                  payloads=served,
                  kinds={name: (kinds or {})[name] for name in served} if kinds else {})


def group_by_kind(bundle: Bundle) -> dict[str, dict[str, bytes]]:
    """Route a bundle's payloads to their KINDS — the engine bus's dispatch
    input (#48).

    Each kind's rollout lowering receives ALL payloads of its own entries and
    attaches them jointly (punica merges peft fragments; a soft prompt's blocks
    are segments of one virtual prompt). Payloads arrive in bank order, since
    compile_bundle sorts them, so a kind that occupies prompt positions
    concatenates in it. A payload with no kind label, or with a trainer-only
    kind, is a compile error: a servable payload must be routable.
    """
    grouped: dict[str, dict[str, bytes]] = {}
    for name, data in bundle.payloads.items():
        if name not in bundle.kinds:
            raise ValueError(
                f"payload {name!r} in {bundle.bundle_id} carries no kind label")
        kind = bundle.kinds[name]
        if ADAPTERS.get(kind).instance.serving is None:
            raise ValueError(
                f"payload {name!r} ({kind}) is trainer-only yet "
                f"shipped in {bundle.bundle_id}")
        grouped.setdefault(kind, {})[name] = data
    return grouped


def group_by_mechanism(bundle: Bundle) -> dict[Mechanism, dict[str, bytes]]:
    """The same routing, keyed by the serving MECHANISM instead of the kind.

    The inventory view: which levers this bundle would be served through. The
    bus itself dispatches by kind (group_by_kind), calling each kind's own
    rollout lowering.
    """
    grouped: dict[Mechanism, dict[str, bytes]] = {}
    for kind, payloads in group_by_kind(bundle).items():
        serving = ADAPTERS.get(kind).instance.serving
        grouped.setdefault(serving, {}).update(payloads)
    return grouped
