"""Canonical serialization, content hashing, and run identity.

Identity is computed, never typed (I3): run_id = h(spec ⊕ registered-code
hashes ⊕ data fingerprint). Everything hashable in the stack — specs, bundles,
certificates — routes through canonical_json, so two objects meaning the same
experiment produce byte-identical JSON no matter how their mappings were
literal-ordered.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any

# The plain JSON tree that _canonicalize produces (dict / list / str / int / float /
# bool / None) before json.dumps sees it.
JSONValue = Any

_TYPE_KEY = "__type__"

# A dataclass field declared with `metadata={IDENTITY: WHEN_SET}` enters the
# canonical form only when it is not None. This is how a spec GROWS a field
# without moving anyone's identity (I3): every spec written before the field
# existed canonicalizes to exactly the bytes it always did, and a spec that
# sets it is a different experiment, as it should be. The wire's decoder
# (runner/remote.py) reads an absent key as the field's default.
IDENTITY = "identity"
WHEN_SET = "when_set"
# A field declared with `metadata={IDENTITY: RESOURCE}` is a RESOURCE, not
# science (ADR 0022): how many replicas serve a pool, how many data-parallel
# groups split a learner's microbatches. It never enters identity — the
# canonical form below drops it, so run_id and the manifest are the same at
# any width and resume-equivalence holds across widths — but it must still
# reach the host that runs the experiment, so the WIRE form (`wire_json`)
# carries it whenever it differs from its default.
RESOURCE = "resource"


def _canonicalize(obj: object, wire: bool = False) -> JSONValue:
    """Lower one object onto the plain JSON tree. Recursive; raises on anything
    unsupported. `wire` keeps the RESOURCE fields that differ from their
    default; the identity form (wire=False) never holds one."""
    # Enum first: IntEnum/StrEnum members are also int/str and would pass through untagged.
    if isinstance(obj, Enum):
        return _canonicalize(obj.value, wire)

    if obj is None:
        return None

    # bool before int — bool is an int subclass.
    if isinstance(obj, bool):
        return bool(obj)

    if isinstance(obj, str):
        return str(obj)

    if isinstance(obj, int):
        return int(obj)

    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise TypeError(
                f"canonical_json: non-finite float {obj!r} is not hashable identity"
            )
        # repr-precision-stable: json uses float.__repr__, the shortest round-tripping form.
        return float(obj)

    # Dataclass INSTANCES (not the classes themselves) carry their class name, so a
    # two structurally identical spec classes never collide.
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, JSONValue] = {_TYPE_KEY: type(obj).__name__}
        for f in sorted(dataclasses.fields(obj), key=lambda f: f.name):
            value = getattr(obj, f.name)
            if value is None and f.metadata.get(IDENTITY) == WHEN_SET:
                continue
            if f.metadata.get(IDENTITY) == RESOURCE and not (
                    wire and value != f.default):
                continue
            out[f.name] = _canonicalize(value, wire)
        return out

    # Any Mapping, MappingProxyType included; keys sorted at dumps time, str-only.
    if isinstance(obj, Mapping):
        mapped: dict[str, JSONValue] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"canonical_json: mapping keys must be str, got {type(key).__name__}"
                )
            mapped[key] = _canonicalize(value, wire)
        return mapped

    if isinstance(obj, (tuple, list)):
        return [_canonicalize(item, wire) for item in obj]

    raise TypeError(f"canonical_json: unsupported type {type(obj).__name__}")


def canonical_json(obj: object) -> str:
    """Deterministic JSON for any spec tree: sorted keys, no whitespace, no NaN."""
    return json.dumps(
        _canonicalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def wire_json(obj: object) -> str:
    """The canonical JSON a FRAME carries: the identity form plus every
    RESOURCE field that is not at its default (ADR 0022). What crosses to a
    host is the whole experiment — how wide to run it included — while
    `canonical_json`, and so run_id and the manifest, never sees the width.
    A spec at every default is byte-for-byte its canonical form."""
    return json.dumps(
        _canonicalize(obj, wire=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(obj: object) -> str:
    """sha256 hexdigest of the canonical JSON — the content address of anything."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def run_id(spec: object, code_hashes: Mapping[str, str], data_fingerprint: str) -> str:
    """I3: h(spec ⊕ registered-code hashes ⊕ data fingerprint), first 12 hex chars.

    Editing a registered function's body changes the code hashes and therefore
    the run; renaming a file changes nothing.
    """
    return content_hash(
        {"spec": spec, "code": code_hashes, "data": data_fingerprint}
    )[:12]

# The tag _canonicalize stamps on every dataclass, exported for the wire's
# decoder (runner/remote.py) — canonical.py itself stays a pure-value module
# and never learns the spec classes' names.
TYPE_KEY = _TYPE_KEY
