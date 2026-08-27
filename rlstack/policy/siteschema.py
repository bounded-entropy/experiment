"""SiteSchema: the trainer's half of the site treaty, as frozen data.

A schema is a pure function of ONE base checkpoint: every attachment point the
checkpoint expresses, under its canonical name — which IS the checkpoint's own
module-tree name (identity plus a prefix), so per-model translation tables do
not exist on this side. A schema is built once by a COMPILER function
(`fake_qwen_schema` below; `hf_schema` walks the real module tree in Phase B2)
and is inert afterwards: plain data, plus ONE concrete resolver implementing
the one canonical pattern grammar. The grammar is canon — `AdapterSpec.site`
hashes into run identity, so its meaning cannot vary by backend.

What the schema deliberately does NOT know:
  - engine reachability — a property of an engine BUILD, not of the model
    graph; each pool's engine self-reports it (Engine.reachability) and the
    runner checks it at Phase 0;
  - adapter-created sites — a soft prompt EXPORTS prompt[:n]; Phase-0
    resolution runs against schema ∪ bank exports (spec/validate.site_space).
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Sequence
from dataclasses import dataclass

from rlstack.spec.canonical import content_hash


@dataclass(frozen=True)
class SiteMeta:
    """One site: the trainer-attestable facts, nothing else.

    `name` is the spec-side address (what a bank entry's pattern matches);
    `path` is the trainer-side address (where install_replay hooks, an entry
    in the base's `named_modules()`). Engine reachability is deliberately
    absent — see the module docstring.
    """

    name: str                       # canonical name, e.g. "layers.3.self_attn.q_proj"
    path: str                       # trainer module path / hookable boundary
    has_weight: bool
    shape: tuple[int, int] | None
    is_boundary: bool


# ---------------------------------------------------------------------------
# the canonical pattern grammar — one grammar, defined here, for every schema
# ---------------------------------------------------------------------------

_RANGE = re.compile(r"(\d+)-(\d+)")


def is_module_name(name: str) -> bool:
    """True for dotted module-tree names; False for non-module sites.

    Non-module names ("prompt[:8]", "queries -> prompt[:8]", "final_hidden",
    "logits") carry no segment structure, so they match exactly only.
    """
    return "." in name and " -> " not in name and "[" not in name


def segment_matches(pattern_segment: str, name_segment: str) -> bool:
    """One dot-segment: numeric range ("0-15") or fnmatch wildcards."""
    numeric = _RANGE.fullmatch(pattern_segment)
    if numeric is not None:
        if not name_segment.isdigit():
            return False
        low, high = int(numeric.group(1)), int(numeric.group(2))
        return low <= int(name_segment) <= high
    return fnmatch.fnmatchcase(name_segment, pattern_segment)


def pattern_matches(pattern: str, name: str) -> bool:
    """The full rule: segment count must agree; each segment must match."""
    pattern_segments = pattern.split(".")
    name_segments = name.split(".")
    if len(pattern_segments) != len(name_segments):
        return False
    return all(segment_matches(p, s)
               for p, s in zip(pattern_segments, name_segments))


def resolve(sites: Sequence[SiteMeta], pattern: str) -> tuple[SiteMeta, ...]:
    """Sites matching `pattern` under the canonical grammar; empty when none.

    Module patterns match module names segment-wise; non-module names match
    exactly only. This function is THE resolver — validate and the runner use
    it over schema ∪ bank exports; SiteSchema.resolve delegates here.
    """
    if not is_module_name(pattern):
        return tuple(site for site in sites if site.name == pattern)
    return tuple(site for site in sites
                 if is_module_name(site.name) and pattern_matches(pattern, site.name))


# ---------------------------------------------------------------------------
# the schema itself: frozen data with the resolver attached
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteSchema:
    """The per-base site catalog: which base it describes, and its sites.

    `base` is checked against `policy.base` at Phase 0 (schema-base-mismatch);
    `fingerprint()` goes into the run manifest, so attaching to a run with a
    silently different schema fails loudly (ManifestMismatch).
    """

    base: str
    sites: tuple[SiteMeta, ...]

    def resolve(self, pattern: str) -> tuple[SiteMeta, ...]:
        return resolve(self.sites, pattern)

    def fingerprint(self) -> str:
        """Content hash of the catalog — same sites, same fingerprint."""
        return content_hash([
            [s.name, s.path, s.has_weight, list(s.shape) if s.shape else None,
             s.is_boundary]
            for s in self.sites
        ])


# ---------------------------------------------------------------------------
# compilers: functions that build a SiteSchema for a base
# ---------------------------------------------------------------------------

_ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJ = ("gate_proj", "up_proj", "down_proj")


def fake_qwen_schema(n_layers: int, *, base: str) -> SiteSchema:
    """A Qwen-shaped toy catalog for `base`: weighted projections + boundaries.

    The CPU stand-in compiler: what hf_schema will read off a real checkpoint
    in Phase B2, hand-built here so Phase-0 validation has real material.
    Note what is absent: prompt[:n] and its attention rectangle are NOT base
    sites — a soft prompt in the bank exports them (Adapter.exports).
    """
    sites: list[SiteMeta] = []
    for n in range(n_layers):
        for proj in _ATTN_PROJ:
            sites.append(SiteMeta(
                name=f"layers.{n}.self_attn.{proj}",
                path=f"model.layers.{n}.self_attn.{proj}",
                has_weight=True, shape=(64, 64), is_boundary=False))
        for proj in _MLP_PROJ:
            sites.append(SiteMeta(
                name=f"layers.{n}.mlp.{proj}",
                path=f"model.layers.{n}.mlp.{proj}",
                has_weight=True, shape=(64, 64), is_boundary=False))
        sites.append(SiteMeta(
            name=f"resid_pre.{n}", path=f"model.layers.{n}",
            has_weight=False, shape=None, is_boundary=True))
    sites.extend([
        SiteMeta(name="logits", path="lm_head",
                 has_weight=False, shape=None, is_boundary=True),
        SiteMeta(name="final_hidden", path="model.norm",
                 has_weight=False, shape=None, is_boundary=True),
    ])
    return SiteSchema(base=base, sites=tuple(sites))


def hf_schema(base: str) -> SiteSchema:
    """Compile the schema for a real HF checkpoint from its config.

    Reads geometry only (no weights): layer count and projection shapes,
    including GQA (k/v project to num_key_value_heads). Canonical names are
    the checkpoint's own tree names — identity plus the "model." prefix.
    Deferred dependency: transformers, imported here only (STYLE rule 7).
    """
    from transformers import AutoConfig  # deferred: engine/trainer images only

    config = AutoConfig.from_pretrained(base)
    hidden = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim",
                           hidden // int(config.num_attention_heads)))
    q_out = int(config.num_attention_heads) * head_dim
    kv_out = int(config.num_key_value_heads) * head_dim
    mlp = int(config.intermediate_size)

    attn_shapes = {"q_proj": (hidden, q_out), "k_proj": (hidden, kv_out),
                   "v_proj": (hidden, kv_out), "o_proj": (q_out, hidden)}
    mlp_shapes = {"gate_proj": (hidden, mlp), "up_proj": (hidden, mlp),
                  "down_proj": (mlp, hidden)}

    sites: list[SiteMeta] = []
    for n in range(int(config.num_hidden_layers)):
        for proj, shape in attn_shapes.items():
            sites.append(SiteMeta(
                name=f"layers.{n}.self_attn.{proj}",
                path=f"model.layers.{n}.self_attn.{proj}",
                has_weight=True, shape=shape, is_boundary=False))
        for proj, shape in mlp_shapes.items():
            sites.append(SiteMeta(
                name=f"layers.{n}.mlp.{proj}",
                path=f"model.layers.{n}.mlp.{proj}",
                has_weight=True, shape=shape, is_boundary=False))
        sites.append(SiteMeta(
            name=f"resid_pre.{n}", path=f"model.layers.{n}",
            has_weight=False, shape=None, is_boundary=True))
    sites.extend([
        SiteMeta(name="logits", path="lm_head",
                 has_weight=False, shape=None, is_boundary=True),
        SiteMeta(name="final_hidden", path="model.norm",
                 has_weight=False, shape=None, is_boundary=True),
    ])
    return SiteSchema(base=base, sites=tuple(sites))
