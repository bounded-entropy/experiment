"""Adapter kinds — the five-member protocol (SPEC.md §2B).

The policy is the only primitive living in both worlds (I2). Each kind says how
its half lowers on each side — rollout lowering into the engine, replay lowering
into the trainer forward — and the compiled bundle is the only data channel
between them. Phase A carries the declaration halves; the compute lands in
Phase B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_KINDS, source_hash

# Closed vocabulary for `serving`: how the rollout lowering reaches the engine.
SERVING_KINDS = (
    "native:punica",         # vLLM multi-LoRA slots
    "native:prompt_embeds",  # vLLM prompt embeddings
    "native:logits",         # logits processor
    "plugin:lse_patch",      # our attention-backend patch (needs engine_plugin)
    None,                    # trainer-only; never served
)


class AdapterKind:
    """Subclass, set the class attributes, implement the methods, register with
    @adapter_kind. Subclasses must construct with no arguments — the decorator
    instantiates one shared instance for Phase-0 predicate calls."""

    engine_plugin: str | None = None       # module the engine image must carry
    serving: str | None = None             # one of SERVING_KINDS
    provides: frozenset[str] = frozenset() # PolicyOutputs fields the replay adds
    records: tuple[str, ...] = ()          # per-token columns the rollout lowering
                                           # writes into the Turn at sampling time

    # `records` and `provides` are mirror images across the membrane: records
    # are FACTS from sampling time (frozen at the seal, never recomputable —
    # e.g. the adapter index drawn at each token); provides are TENSORS from
    # training time (recomputed each forward under current params — e.g. the
    # value head's "values"). install_replay reads this kind's own recorded
    # columns to make the trainer forward faithful; a loss that wants a
    # recorded column in its math names it in `requires`.

    def site_ok(self, meta: SiteMeta) -> bool:
        """Can this kind live at a site with this metadata? Checked at Phase 0."""
        return True

    def params(self, sites: tuple[SiteMeta, ...], init: dict) -> Any:
        """Build the trainable parameterization for the matched sites."""
        raise NotImplementedError

    def install_replay(self, model: Any, params: Any, sites: tuple[SiteMeta, ...]) -> None:
        """Wire the replay lowering into the trainer forward."""
        raise NotImplementedError

    def emit(self, params: Any) -> Any:
        """Lower params into the bundle payload the engine-side consumer reads."""
        raise NotImplementedError

    def parity(self, harness: Any) -> Any:
        """The mandatory rollout/replay numerical parity test."""
        raise NotImplementedError


@dataclass(frozen=True)
class KindDef:
    """A registered adapter kind: the class plus one shared instance."""

    name: str
    cls: type[AdapterKind]
    instance: AdapterKind
    source_hash: str


def adapter_kind(name: str):
    def register(cls: type[AdapterKind]) -> type[AdapterKind]:
        ADAPTER_KINDS.add(KindDef(name, cls, cls(), source_hash(cls)))
        return cls
    return register


# ---------------------------------------------------------------------------
# built-in kinds
# ---------------------------------------------------------------------------

@adapter_kind("lora")
class LoraKind(AdapterKind):
    """Per-matrix low-rank delta; served natively by vLLM multi-LoRA."""

    serving = "native:punica"

    def site_ok(self, meta: SiteMeta) -> bool:
        return meta.has_weight


@adapter_kind("soft_prompt")
class SoftPromptKind(AdapterKind):
    """Learned virtual prompt rows; served natively via prompt_embeds."""

    serving = "native:prompt_embeds"

    def site_ok(self, meta: SiteMeta) -> bool:
        return not meta.has_weight


@adapter_kind("attn_bias")
class AttnBiasKind(AdapterKind):
    """Learned bias on an attention-score rectangle; needs the LSE-merge plugin."""

    serving = "plugin:lse_patch"
    engine_plugin = "rlstack_engine.attn_bias"

    def site_ok(self, meta: SiteMeta) -> bool:
        return not meta.has_weight


@adapter_kind("value_head")
class ValueHeadKind(AdapterKind):
    """Trainer-only scalar head over a hidden boundary; never served."""

    serving = None
    provides = frozenset({"values"})

    def site_ok(self, meta: SiteMeta) -> bool:
        return meta.is_boundary and not meta.has_weight
