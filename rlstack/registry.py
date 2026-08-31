"""The registries: one table per swappable kind, filled at import.

This module is the MECHANISM only — a `Registry` is a dict with a helpful
KeyError and a duplicate rule, and each entry is a typed record pairing the
DECLARATION (the fields Phase 0 reads) with the COMPUTE (the function or class
itself). The registered things live with their worlds — adapter types in
policy/adapters/, environments in inference/, postprocessors and losses in
training/, yours wherever you define them — so a name exists iff the module
defining it was imported. `code_hashes` collects the source hash of everything
a spec references, which is what makes identity computed (I3): edit a body and
the run_id changes, rename a file and nothing does.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from rlstack.spec.specs import ExperimentSpec


# ---------------------------------------------------------------------------
# the mechanism
# ---------------------------------------------------------------------------

def source_hash(obj: object) -> str:
    """sha256 of the object's source; falls back to module:qualname for builtins."""
    try:
        src = inspect.getsource(obj)  # type: ignore[arg-type]
    except (OSError, TypeError, IndentationError):
        module = getattr(obj, "__module__", "?")
        qualname = getattr(obj, "__qualname__", repr(obj))
        return hashlib.sha256(f"{module}:{qualname}".encode()).hexdigest()
    return hashlib.sha256(src.encode()).hexdigest()


class Registry:
    """Named entries of one registry kind ("loss", "environment", ...).

    Entries are the typed *Def records. Registering a name twice is an error
    unless the source is identical: a re-import is a no-op, but two different
    bodies under one name would make identity ambiguous.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Any] = {}

    def add(self, entry: Any) -> None:
        prior = self._entries.get(entry.name)
        if prior is not None and prior.source_hash != entry.source_hash:
            raise ValueError(
                f"duplicate {self.kind} registration {entry.name!r} with different "
                f"source — identity would be ambiguous"
            )
        self._entries[entry.name] = entry

    def get(self, name: str) -> Any:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} {name!r} (is the module defining it "
                f"imported?); registered {self.kind}s: "
                f"{', '.join(self.names()) or '(none)'}"
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Registry({self.kind!r}, {len(self._entries)} entries)"


ENVS = Registry("environment")      # EnvironmentDef (inference/environments/base.py)
POST = Registry("postprocessor")    # PostDef (training/post/base.py)
LOSSES = Registry("loss")           # LossDef (below)
ADAPTER_TYPES = Registry("adapter_type")  # AdapterTypeDef (policy/adapters/base.py):
                                    # one registered ADAPTER TYPE per entry,
                                    # which an AdapterSpec names by string


# ---------------------------------------------------------------------------
# loss declarations (the only registered functions; everything else is a class)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossDef:
    """Microbatch-scope loss: PURE MATH — fn(PolicyOutputs, TokenBatch) -> Loss.

    `requires` names DATA COLUMNS only: postdata columns the pipeline
    produced, recorded facts (base or bank), or bank-provided forward
    tensors. A loss never causes metal work (I9) — anything that needs a GPU
    to compute (judges, teacher scoring, hinted rescoring) is a
    postprocessor's job, landing in postdata before the loss runs.
    """

    name: str
    fn: Callable[..., Any]
    requires: tuple[str, ...]
    source_hash: str


def loss(name: str, requires: tuple = ()) -> Callable:
    """Register a loss under `name`; `requires` must name data columns, so a
    non-string requirement is refused at registration."""
    def register(fn: Callable) -> Callable:
        bad = [r for r in requires if not isinstance(r, str)]
        if bad:
            raise TypeError(
                f"loss {name!r} requires {bad!r}: requires names data columns "
                f"only — anything needing metal to compute is a post "
                f"processor's job (it lands in postdata before the loss runs)")
        LOSSES.add(LossDef(name, fn, tuple(requires), source_hash(fn)))
        return fn
    return register


# ---------------------------------------------------------------------------
# identity (I3)
# ---------------------------------------------------------------------------

def code_hashes(spec: ExperimentSpec) -> dict[str, str]:
    """Source hash of every registered name the spec references.

    Keys are "<registry kind>:<name>". An unknown name raises the registry's
    KeyError — run the submit gate first to report those as issues instead.
    """
    out: dict[str, str] = {}

    def add(registry: Registry, name: str) -> None:
        out[f"{registry.kind}:{name}"] = registry.get(name).source_hash

    if spec.algo is not None:
        add(LOSSES, spec.algo.loss)
        for name in spec.algo.post:
            add(POST, name)

    if spec.gen is not None:
        for name in spec.gen.envs:
            add(ENVS, name)

    for adapter in spec.policy.bank.values():
        add(ADAPTER_TYPES, adapter.adapter_type)

    return out
