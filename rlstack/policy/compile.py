"""Bundle compilation: the bank, lowered into the only thing that crosses back.

A Bundle is the compiled, content-addressed artifact `sync_weights` hands to the
engine — the single data channel from the training world to the inference world
(I2). It pins the FULL policy version map (trainable and frozen deltas alike)
but carries payloads only for servable deltas: a trainer-only kind (value_head)
versions like any other delta yet never ships to the engine.

Phase A/B1 payloads are opaque bytes from Learner.emit; Phase B2 makes them
peft-shaped safetensors that vLLM's add_lora reads directly. The identity rule
is already final: bundle_id is a content hash of the version map plus payload
digests, so identical banks compile to identical ids everywhere.
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

    `kinds` (payload name -> registered adapter kind) makes the bundle
    self-describing: any engine can route each payload to its serving
    mechanism without seeing the spec (group_by_mechanism below).
    """

    bundle_id: str
    policy_version: Mapping[str, int]
    payloads: Mapping[str, bytes] = field(repr=False, default_factory=dict)
    kinds: Mapping[str, str] = field(default_factory=dict)


def compile_bundle(
    payloads: Mapping[str, bytes],
    policy_version: Mapping[str, int],
    servable: Iterable[str],
    kinds: Mapping[str, str] | None = None,
) -> Bundle:
    """Lower emitted payloads into a content-addressed Bundle.

    `payloads` is everything the learner emitted; only names in `servable`
    ship, each labeled with its adapter kind from `kinds`. bundle_id =
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
                  kinds={name: (kinds or {})[name] for name in served} if kinds else {})


def group_by_mechanism(bundle: Bundle) -> dict[Mechanism, dict[str, bytes]]:
    """Route a bundle's payloads to their serving mechanisms.

    THE dispatch input for any engine's add_bundle: each mechanism's consumer
    receives ALL payloads of its adapters and compiles them jointly (punica
    merges peft fragments; side_attention takes prompt rows + bias together).
    A payload with no kind label or a trainer-only kind is a compile error —
    servable payloads must be routable.
    """
    grouped: dict[Mechanism, dict[str, bytes]] = {}
    for name, data in bundle.payloads.items():
        if name not in bundle.kinds:
            raise ValueError(
                f"payload {name!r} in {bundle.bundle_id} carries no kind label")
        serving = ADAPTERS.get(bundle.kinds[name]).instance.serving
        if serving is None:
            raise ValueError(
                f"payload {name!r} ({bundle.kinds[name]}) is trainer-only yet "
                f"shipped in {bundle.bundle_id}")
        grouped.setdefault(serving, {})[name] = data
    return grouped
