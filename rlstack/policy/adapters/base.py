"""The Adapter contract — the five-member protocol (SPEC.md §2B).

The policy is the only primitive living in both worlds (I2). An adapter says
how its half lowers on each side — rollout lowering into the engine, replay
lowering into the trainer — and the compiled bundle is the only data channel
between them. Each adapter is registered as a CLASS in its own file under this
folder:

    @adapter("my_adapter")
    class MyAdapter(Adapter):
        serving = Mechanism.PUNICA       # or None for trainer-only
        provides = frozenset({...})      # training-time tensors the replay adds
        records = ("...",)               # sampling-time facts the rollout writes
        def site_ok(self, meta): ...
        # params / install_replay / emit / parity land with Phase B compute

`records` and `provides` are mirror images across the membrane: records are
FACTS from sampling time (frozen at the seal, never recomputable — e.g. the
adapter index drawn at each token); provides are TENSORS from training time
(recomputed each forward — e.g. the value head's "values"). install_replay
reads its own recorded columns to make the trainer forward faithful; a loss
that wants a recorded column in its math names it in `requires`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTERS, source_hash
from rlstack.spec.specs import AdapterSpec


class Mechanism(StrEnum):
    """How an engine reaches a site: the CLOSED set of serving levers.

    Native levers (PUNICA, PROMPT_EMBEDS, LOGITS) are maintained by the engine
    itself; SIDE_ATTENTION is ours, shipped as an engine plugin (rlstack_engine)
    that must re-earn per-request selection, cache correctness, and parity on
    every build. Whether a lever reaches a given site on a given BUILD is the
    engine's to answer (Engine.reachability), never a static table here.
    """

    PUNICA = "punica"                 # per-token weight deltas (vLLM multi-LoRA)
    PROMPT_EMBEDS = "prompt_embeds"   # virtual rows before the prompt
    LOGITS = "logits"                 # logits processor
    SIDE_ATTENTION = "side_attention" # our LSE-merge plugin (soft prompt + bias)
    NONE = "none"                     # inventory answer only: not reachable


class Adapter:
    """Subclass, set the class attributes, implement the methods, register with
    @adapter. Subclasses must construct with no arguments — the decorator
    instantiates one shared instance for Phase-0 predicate calls."""

    engine_plugin: str | None = None          # module the engine image must carry
    serving: Mechanism | None = None          # None: trainer-only, never served
    provides: frozenset[str] = frozenset()    # PolicyOutputs fields the replay adds
    records: tuple[str, ...] = ()             # per-token columns the rollout writes

    def site_ok(self, meta: SiteMeta) -> bool:
        """Can this adapter live at a site with this metadata? Checked at Phase 0."""
        return True

    def exports(self, spec: AdapterSpec) -> tuple[SiteMeta, ...]:
        """Sites this bank entry CREATES — points the base checkpoint does not
        have (a soft prompt exports its prompt[:n] positions). Phase-0
        resolution runs against schema ∪ every entry's exports."""
        return ()

    def params(self, sites: tuple[SiteMeta, ...], init: dict) -> Any:
        """Build the trainable parameterization for the matched sites."""
        raise NotImplementedError

    def install_replay(self, model: Any, params: Any, sites: tuple[SiteMeta, ...]) -> None:
        """Wire the replay lowering into the trainer forward."""
        raise NotImplementedError

    def uninstall_replay(self, model: Any, params: Any, sites: tuple[SiteMeta, ...]) -> None:
        """install_replay's exact inverse: restore the module tree so another
        tenant's adapters can install (the trainer-side mirror of the engine
        evicting a bundle). A kind without this cannot swap-share a learner."""
        raise NotImplementedError(
            f"{type(self).__name__} has no uninstall_replay: it cannot share "
            f"a multi-tenant learner (each tenant needs exclusive install)")

    def emit(self, params: Any) -> Any:
        """Lower params into the bundle payload the engine-side consumer reads."""
        raise NotImplementedError

    def load(self, params: Any, payload: bytes) -> None:
        """emit's inverse: restore params in place from an emitted payload
        (resume and warm-start walk through here)."""
        raise NotImplementedError

    def parity(self, harness: Any) -> Any:
        """The mandatory rollout/replay numerical parity test."""
        raise NotImplementedError


@dataclass(frozen=True)
class AdapterDef:
    """A registered adapter: the class plus one shared instance."""

    name: str
    cls: type[Adapter]
    instance: Adapter
    source_hash: str


def adapter(name: str):
    def register(cls: type[Adapter]) -> type[Adapter]:
        ADAPTERS.add(AdapterDef(name, cls, cls(), source_hash(cls)))
        return cls
    return register
