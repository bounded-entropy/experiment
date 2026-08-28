"""The adapter-type contract: what a registered adapter class declares and computes.

The class registers an ADAPTER TYPE — `AdapterSpec.adapter_type` names it by
string, and a configured bank entry is an adapter. Because the policy is the
only primitive living in both worlds (I2), an adapter type ships BOTH
lowerings, each in its own file beside this one and imported lazily so the
client library stays stdlib-clean: the replay lowering in
`<adapter_type>_torch.py` (entered through install_replay), the rollout
lowering in `<adapter_type>_vllm.py` (built by rollout_lowering, contract in
adapters/rollout.py). Parity — the exam that binds the pair — is then
reviewable in one directory.

`records` and `provides` are mirror images across the membrane: records are
sampling-time FACTS, frozen at the seal and never recomputable; provides are
training-time TENSORS, recomputed by each forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES, source_hash
from rlstack.spec.specs import AdapterSpec

if TYPE_CHECKING:                       # the seam imports this module back
    from rlstack.policy.adapters.rollout import RolloutLowering, ServingBuild


class Mechanism(StrEnum):
    """How an engine reaches a site: the CLOSED set of serving levers.

    PUNICA, PROMPT_EMBEDS and LOGITS are the engine's own; SIDE_ATTENTION is
    ours, shipped as an engine plugin that must re-earn per-request selection,
    cache correctness and parity on every build. Whether a lever reaches a given
    site on a given BUILD is the engine's to answer (Engine.reachability), never
    a static table here (I7).
    """

    PUNICA = "punica"                 # per-token weight deltas (vLLM multi-LoRA)
    PROMPT_EMBEDS = "prompt_embeds"   # virtual rows before the prompt
    LOGITS = "logits"                 # logits processor
    SIDE_ATTENTION = "side_attention" # our LSE-merge plugin (soft prompt + bias)
    NONE = "none"                     # inventory answer only: not reachable


class AdapterType:
    """The registered class: one ADAPTER TYPE. Subclass, set the class
    attributes, implement the methods, register with @adapter_type. Subclasses
    must construct with no arguments — the decorator instantiates one shared
    instance for Phase-0 predicate calls."""

    engine_plugin: str | None = None          # module the engine image must carry
    serving: Mechanism | None = None          # None: trainer-only, never served
    provides: frozenset[str] = frozenset()    # PolicyOutputs fields the replay adds
    records: tuple[str, ...] = ()             # per-token columns the rollout writes

    def site_ok(self, meta: SiteMeta) -> bool:
        """Can this adapter type live at a site with this metadata? Checked at
        Phase 0."""
        return True

    def exports(self, spec: AdapterSpec) -> tuple[SiteMeta, ...]:
        """Sites this bank entry CREATES — points the base checkpoint does not
        have (a soft prompt exports its prompt[:n] positions). Phase-0
        resolution runs against schema ∪ every entry's exports."""
        return ()

    def params(self, sites: tuple[SiteMeta, ...], init: dict) -> Any:
        """Build the trainable parameterization for the matched sites."""
        raise NotImplementedError

    def rollout_lowering(self, build: "ServingBuild") -> "RolloutLowering":
        """Build this adapter type's ROLLOUT lowering for one engine build
        (#48).

        install_replay's twin on the other side of the bridge: that one wires
        the adapter type into a trainer forward, this one hands the engine the
        verbs (demands / attach / apply / align, plus reaches) it serves the
        adapter type through. A trainer-only adapter type (serving None) has
        none, and says so.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no rollout lowering: it is trainer-"
            f"only (serving is None) and no engine ever hears about it")

    def install_replay(self, model: Any, params: Any, sites: tuple[SiteMeta, ...]) -> None:
        """Wire the replay lowering into the trainer forward. Additive: every
        installed tenant stays wired (I8)."""
        raise NotImplementedError

    def uninstall_replay(self, model: Any, params: Any, sites: tuple[SiteMeta, ...]) -> None:
        """install_replay's exact inverse: restore the module tree when a
        tenant leaves (the trainer-side mirror of the engine evicting a
        bundle). An adapter type without it cannot share a multi-tenant
        learner."""
        raise NotImplementedError(
            f"{type(self).__name__} has no uninstall_replay: install is "
            f"ADDITIVE, so without its inverse a tenant could never be REMOVED "
            f"from a shared learner")

    def emit(self, params: Any) -> Any:
        """Lower params into the bundle payload the engine-side consumer reads."""
        raise NotImplementedError

    def load(self, params: Any, payload: bytes) -> None:
        """emit's inverse: restore params in place from an emitted payload
        (resume and warm-start walk through here)."""
        raise NotImplementedError

    def parity(self, harness: Any) -> Any:
        """The mandatory numerical exam binding this adapter type's two lowerings
        (I7). Declared here and unwired; the running parity alarm is
        logprob_gap."""
        raise NotImplementedError


@dataclass(frozen=True)
class AdapterTypeDef:
    """A registered adapter type: the class, one shared instance, and the
    source hash that carries an edit to its body into run identity (I3)."""

    name: str
    cls: type[AdapterType]
    instance: AdapterType
    source_hash: str


def adapter_type(name: str):
    def register(cls: type[AdapterType]) -> type[AdapterType]:
        ADAPTER_TYPES.add(AdapterTypeDef(name, cls, cls(), source_hash(cls)))
        return cls
    return register
