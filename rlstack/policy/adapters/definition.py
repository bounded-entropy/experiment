"""One semantic adapter definition, consumed by replay and rollout compilers.

This module is client-side data: importing definitions never imports a GPU
library. Existing operations retain their proven compute providers. A custom
site operation uses the common providers and needs no per-adapter lifecycle.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rlstack.policy.siteschema import SiteMeta


class SiteKind(StrEnum):
    LINEAR = "linear"
    RESIDUAL = "residual"
    PROMPT = "prompt"
    ATTENTION = "attention"
    ANY = "any"


class Input(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    POSITION = "position"
    REQUEST = "request"


class Operation(StrEnum):
    LOW_RANK = "low_rank_add"
    STEER = "steer_add"
    NORM_STEER = "norm_scaled_steer"
    PROBABILISTIC_LOW_RANK = "probabilistic_low_rank_add"
    SPECTRAL = "spectral_top_k"
    SPECTRAL_LATENT = "spectral_latent"
    PROMPT = "prepend_embeddings"
    ATTENTION_BIAS = "attention_bias"
    VALUE = "value_readout"
    CUSTOM = "custom_site"


class Dim(StrEnum):
    INPUT = "input_dim"
    OUTPUT = "output_dim"


class Init(StrEnum):
    ZEROS = "zeros"
    ONES = "ones"
    NORMAL = "normal"


@dataclass(frozen=True)
class Symbol:
    """A declared implementation symbol; resolution is lazy and explicit."""

    module: str
    name: str

    def load(self):
        return vars(importlib.import_module(self.module))[self.name]


@dataclass(frozen=True)
class Parameter:
    name: str
    shape: tuple[int | Dim, ...]
    init: Init = Init.ZEROS
    std: float = 0.02
    trainable: bool = True
    dtype: str = "float32"

    def __post_init__(self):
        if not self.name or "/" in self.name or ":" in self.name:
            raise ValueError("a site parameter needs a nonempty local name")
        if self.dtype not in ("float32", "float16", "bfloat16", "float64"):
            raise ValueError(f"unsupported site parameter dtype {self.dtype!r}")
        if any(not isinstance(d, Dim) and (type(d) is not int or d < 1)
               for d in self.shape):
            raise ValueError("parameter dimensions are positive integers or Dim")
        if self.std < 0:
            raise ValueError("initialization std must be nonnegative")


@dataclass(frozen=True)
class SiteProgram:
    """Forward/reference implement (inputs, output, parameters, token facts).

    A reference supplies autograd and CPU execution. GPU forward may be a
    Triton launch sequence; it must preserve output shape and never mutate
    inputs. Kernels see current query tokens only, including at decode.
    """

    forward: Symbol
    reference: Symbol | None = None
    inputs: tuple[Input, ...] = (Input.OUTPUT,)
    width: int | None = None
    helpers: tuple[Symbol, ...] = ()

    def __post_init__(self):
        if len(set(self.inputs)) != len(self.inputs):
            raise ValueError("site inputs must be unique")
        if self.width is not None and self.width < 1:
            raise ValueError("boundary width must be positive")


@dataclass(frozen=True)
class ReplayProvider:
    """A compute module implementing build/install/uninstall/emit/load."""

    module: str


@dataclass(frozen=True)
class RolloutProvider:
    """A native or custom rollout implementation, separate from its math."""

    factory: Symbol
    mechanism: str


@dataclass(frozen=True)
class AdapterDefinition:
    operation: Operation | str
    site: SiteKind
    replay: ReplayProvider | None = None
    rollout: RolloutProvider | None = None
    parameters: tuple[Parameter, ...] = ()
    program: SiteProgram | None = None
    scale: float = 1.0

    def __post_init__(self):
        if self.scale != 1.0:
            raise ValueError("providers currently use unit scaling; declare other scales in program state")
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError("site parameter names must be unique")
        if self.operation == Operation.CUSTOM:
            if self.program is None or self.site not in (SiteKind.LINEAR, SiteKind.RESIDUAL):
                raise ValueError("custom programs need a linear or residual site")
            if self.replay is not None or self.rollout is not None:
                raise ValueError("custom programs use the common lowering providers")
            if self.site == SiteKind.RESIDUAL and Input.INPUT in self.program.inputs:
                raise ValueError("residual programs expose the output stream; linear sites also expose input")
        elif self.program is not None or self.replay is None:
            raise ValueError("built-in operations declare their replay provider")

    @property
    def mechanism(self) -> str | None:
        return "site" if self.program is not None else (
            None if self.rollout is None else self.rollout.mechanism)

    @property
    def trainable(self) -> bool:
        return self.program is None or self.program.reference is not None

    def accepts(self, meta: SiteMeta) -> bool:
        if self.site == SiteKind.LINEAR:
            return meta.has_weight
        if self.site == SiteKind.RESIDUAL:
            return meta.is_boundary and not meta.has_weight
        return self.site == SiteKind.ANY or not meta.has_weight

    def resolve(self, sites: tuple[SiteMeta, ...], settings: dict) -> ResolvedAdapter:
        if not sites or any(not self.accepts(site) for site in sites):
            raise ValueError(f"{self.operation}: unsupported or empty site selection")
        if self.program is not None:
            if set(settings) - {"seed", "d"}:
                raise ValueError("site parameters are declared; init accepts only seed and boundary d")
            if self.site == SiteKind.RESIDUAL and self.program.width is None:
                if type(settings.get("d")) is not int or settings["d"] < 1:
                    raise ValueError("residual site programs require a positive boundary d")
        return ResolvedAdapter(self, sites, dict(settings))

    def source_hash(self) -> str:
        """Hash the semantic declaration AND explicitly selected source modules.

        Module paths locate code; their contents, not their paths, identify it.
        This narrow closure applies to custom definitions; legacy registration
        hashes remain unchanged outside their intentionally edited class bodies.
        """
        row = asdict(self)
        sources: list[str] = []
        from importlib.util import find_spec
        from pathlib import Path
        if self.program is None:
            modules = [self.replay.module]
            if self.rollout is not None:
                modules.append(self.rollout.factory.module)
            for module in modules:
                spec = find_spec(module)
                if spec is None or spec.origin is None:
                    raise ValueError(f"missing provider module {module}")
                sources.append(Path(spec.origin).read_text())
        if self.program is not None:
            for symbol in (self.program.forward, self.program.reference, *self.program.helpers):
                if symbol is not None:
                    # Read source without executing/importing the heavy module.
                    spec = find_spec(symbol.module)
                    if spec is None or spec.origin is None:
                        raise ValueError(f"missing program module {symbol.module}")
                    sources.append(Path(spec.origin).read_text())
            row["program"]["forward"] = self.program.forward.name
            row["program"]["reference"] = (None if self.program.reference is None
                                                else self.program.reference.name)
            row["program"]["helpers"] = [symbol.name for symbol in self.program.helpers]
        return hashlib.sha256(json.dumps(
            {"definition": row, "sources": sources}, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ResolvedAdapter:
    definition: AdapterDefinition
    sites: tuple[SiteMeta, ...]
    settings: dict[str, Any]


@dataclass(frozen=True)
class ProgramManifest:
    program_hash: str
    sites: tuple[str, ...]
    settings: dict

    def metadata(self) -> dict[str, str]:
        # One safetensors metadata key avoids nondeterministic multi-key
        # header order; its structured value is canonical JSON.
        return {"rlstack_site": json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))}

    @classmethod
    def from_metadata(cls, metadata):
        row = json.loads(metadata["rlstack_site"])
        return cls(row["program_hash"], tuple(row["sites"]), row["settings"])


def builtin(operation: Operation, site: SiteKind, replay: str,
            rollout: str | None = None, factory: str = "",
            mechanism: str = "punica") -> AdapterDefinition:
    """One shared description around a preserved native compute implementation."""
    return AdapterDefinition(operation, site, ReplayProvider(
        f"rlstack.policy.adapters.{replay}_torch"), None if rollout is None else
        RolloutProvider(Symbol(f"rlstack.policy.adapters.{rollout}_vllm", factory), mechanism))


def native(operation: str, site: SiteKind, replay: str,
           rollout: str | None = None, factory: str = "",
           mechanism: str = "punica") -> AdapterDefinition:
    """An external native definition declares its providers by full module path."""
    return AdapterDefinition(operation, site, ReplayProvider(replay),
        None if rollout is None else RolloutProvider(Symbol(rollout, factory), mechanism))
