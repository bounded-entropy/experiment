"""The registries (SPEC.md §2B): one table per swappable kind, filled at import.

This module is the MECHANISM only — a `Registry` is a dict with a helpful
KeyError and a duplicate rule, and each entry is a typed record pairing the
DECLARATION (the fields Phase-0 validation reads) with the COMPUTE (the function
itself). The things actually registered live with their worlds: adapters in
policy/adapters/, environments in inference/, postprocessors and losses in
training/, and your own wherever you define them — a name exists iff
the module defining it was imported.

`code_hashes` collects the source hash of everything a spec references; those
hashes feed run identity (I3): edit a body and the run_id changes; rename a
file and nothing does.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from rlstack.spec.specs import ExperimentSpec


# ---------------------------------------------------------------------------
# planned passes — the non-string members of a loss's `requires`. Each names a
# pass the runner plans (memoized per wave) rather than a field the training
# forward already produces.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ref:
    """A pinned reference policy version → planned no_grad pass."""

    version: str


@dataclass(frozen=True)
class Teacher:
    """A scoring pass on a named engine pool (inference-world traffic)."""

    pool: str


@dataclass(frozen=True)
class Probe:
    """A differentiable scoring forward, built by a named probe builder."""

    name: str


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
    """Named entries of one kind ("loss", "env", ...).

    Entries are the typed *Def records. Registering a name twice is an error
    unless the source is identical (a re-import is a no-op; two different
    bodies under one name would make identity ambiguous).
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
ADAPTERS = Registry("adapter")      # AdapterDef (policy/adapters/base.py)


# ---------------------------------------------------------------------------
# loss declarations (the only registered functions; everything else is a class)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossDef:
    """Microbatch-scope loss: pure fn(PolicyOutputs, TokenBatch) -> Loss.

    `requires` mixes PolicyOutputs field names ("values", "ref_logprobs", ...)
    with planned passes (Ref / Teacher / Probe); the runner plans the passes and
    Phase 0 checks the fields against what the bank provides.
    """

    name: str
    fn: Callable[..., Any]
    requires: tuple
    source_hash: str


def loss(name: str, requires: tuple = ()) -> Callable:
    def register(fn: Callable) -> Callable:
        LOSSES.add(LossDef(name, fn, tuple(requires), source_hash(fn)))
        return fn
    return register


# ---------------------------------------------------------------------------
# identity (I3)
# ---------------------------------------------------------------------------

def code_hashes(spec: ExperimentSpec) -> dict[str, str]:
    """Source hash of every registered name the spec references.

    Keys are "<registry kind>:<name>". An unknown name raises the registry's
    KeyError — run Phase-0 validation first to report those as issues.
    """
    out: dict[str, str] = {}

    def add(registry: Registry, name: str) -> None:
        out[f"{registry.kind}:{name}"] = registry.get(name).source_hash

    if spec.algo is not None:
        add(LOSSES, spec.algo.loss)
        for name in spec.algo.post:
            add(POST, name)

    if spec.gen is not None:
        add(ENVS, spec.gen.env)

    if spec.eval is not None:
        if spec.eval.env is not None:
            add(ENVS, spec.eval.env)
        for name in spec.eval.post:
            add(POST, name)

    for adapter in spec.policy.bank.values():
        add(ADAPTERS, adapter.kind)

    return out
