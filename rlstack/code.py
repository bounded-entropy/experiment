"""Explicit experiment imports and the source closure workers must agree on.

An external package is installed by the deployment, never discovered through
ambient registry contents. Its complete Python source tree is a small manifest
in the run identity and in each learner/engine installation.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
from collections.abc import Callable, Mapping, Sequence
from importlib.util import find_spec
from pathlib import Path


_LOADED_SOURCE: dict[str, tuple[object, str]] = {}


def source_manifest(roots: Sequence[str]) -> dict[str, str]:
    """Hash each declared module, or every Python file in its package tree."""
    manifest = {}
    for name in roots:
        spec = find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"declared experiment code {name!r} is not installed")
        origin = Path(spec.origin)
        if spec.submodule_search_locations is None:
            files = [(origin.name, origin)]
        else:
            package = origin.parent
            files = [(path.relative_to(package).as_posix(), path)
                     for path in sorted(package.rglob("*.py"))
                     if "__pycache__" not in path.parts]
        # Importing a subpackage executes its parent initializers too. Keep
        # those tiny files in the closure without hashing unrelated siblings.
        for index in range(1, len(name.split("."))):
            parent = ".".join(name.split(".")[:index])
            parent_spec = find_spec(parent)
            if parent_spec is not None and parent_spec.origin is not None:
                files.append((f"parent/{parent}/__init__.py", Path(parent_spec.origin)))
        digest = hashlib.sha256()
        for relative, path in files:
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        manifest[name] = digest.hexdigest()
    return manifest


def load_code(roots: Sequence[str]) -> dict[str, str]:
    """Load declared roots; changed loaded source requires a fresh process.

    Rehashing changed files cannot reload Python's cached functions. Refusing
    that case prevents old objects from running under a new source identity.
    """
    manifest = source_manifest(roots)
    for name, digest in manifest.items():
        prior = _LOADED_SOURCE.get(name)
        if prior is not None and sys.modules.get(name) is prior[0] and prior[1] != digest:
            raise ValueError(f"experiment code {name!r} changed after import; restart this process before using the new source")
        module = importlib.import_module(name)
        _LOADED_SOURCE[name] = (module, digest)
    if source_manifest(roots) != manifest:
        raise ValueError("experiment code changed while loading; retry in a fresh process")
    return manifest


def verify_code(claimed: Mapping[str, str]) -> None:
    """Refuse a missing package or different worker source before execution."""
    actual = load_code(tuple(claimed))
    if actual != dict(claimed):
        changed = sorted(name for name in claimed if actual.get(name) != claimed[name])
        raise ValueError(f"experiment code mismatch for {', '.join(changed)}; install the pinned package and restart the worker")


def declared_module(module: str, roots: Sequence[str]) -> bool:
    return any(module == root or module.startswith(root + ".") for root in roots)


def require_declared_definition(module: str, roots: Sequence[str]) -> None:
    """Shared primitives are built in; an external definition needs a root."""
    if not module.startswith("rlstack.") and not declared_module(module, roots):
        raise ValueError(f"external definition {module!r} is not covered by ExperimentSpec.code")


def resolve_callable(reference: str) -> Callable:
    """Resolve an explicitly named ordinary function or constructor."""
    module, separator, symbol = reference.partition(":")
    if not separator or not module or not symbol or "." in symbol:
        raise ValueError(f"function {reference!r} must be an explicit 'module:symbol' reference")
    fn = vars(importlib.import_module(module)).get(symbol)
    if not callable(fn):
        raise ValueError(f"{reference!r} must name a callable")
    if fn.__module__ != module:
        raise ValueError(f"function {reference!r} must name its defining module")
    return fn


def resolve_function(reference: str) -> Callable:
    """Resolve one ordinary async function, with no registration or base class."""
    fn = resolve_callable(reference)
    if not inspect.iscoroutinefunction(fn):
        raise ValueError(f"function {reference!r} must name an async function")
    return fn


def require_declared_function(reference: str, code: tuple[str, ...]) -> Callable:
    module = reference.partition(":")[0]
    if not declared_module(module, code):
        raise ValueError(f"function {reference!r} is not covered by ExperimentSpec.code")
    return resolve_function(reference)
