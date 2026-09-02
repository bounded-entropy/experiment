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


def _canonicalize(obj: object) -> JSONValue:
    """Lower one object onto the plain JSON tree. Recursive; raises on anything unsupported."""
    # Enum first: IntEnum/StrEnum members are also int/str and would pass through untagged.
    if isinstance(obj, Enum):
        return _canonicalize(obj.value)

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
            out[f.name] = _canonicalize(getattr(obj, f.name))
        return out

    # Any Mapping, MappingProxyType included; keys sorted at dumps time, str-only.
    if isinstance(obj, Mapping):
        mapped: dict[str, JSONValue] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"canonical_json: mapping keys must be str, got {type(key).__name__}"
                )
            mapped[key] = _canonicalize(value)
        return mapped

    if isinstance(obj, (tuple, list)):
        return [_canonicalize(item) for item in obj]

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
