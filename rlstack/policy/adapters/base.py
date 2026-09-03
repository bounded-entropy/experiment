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

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar

from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES, source_hash
from rlstack.spec.specs import AdapterSpec

if TYPE_CHECKING:                       # the seam imports this module back
    from rlstack.policy.adapters.rollout import (
        Request, RolloutLowering, ServingBuild,
    )


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


@dataclass(frozen=True)
class Directive:
    """A per-request instruction to ONE adapter type, passed by the CALLER of
    `sample` / `score` (ADR 0004, Q2).

    A spec fixes what an adapter IS; a directive says what it does for THIS
    request — which positions a steering vector covers, say. It is a typed
    record the adapter type declares (`AdapterType.directive`), never a
    mapping: the wire encodes it by adapter type name and the adapter type
    decodes its own. What a directive made the rollout do is RECORDED at the
    seal (`record_directive`), so replay reproduces it from the turn's facts
    rather than from the caller's memory (I6).
    """

    adapter_type: ClassVar[str]


class AdapterType:
    """The registered class: one ADAPTER TYPE. Subclass, set the class
    attributes, implement the methods, register with @adapter_type. Subclasses
    must construct with no arguments — the decorator instantiates one shared
    instance for Phase-0 predicate calls."""

    engine_plugin: str | None = None          # module the engine image must carry
    records: tuple[str, ...] = ()             # per-token columns the rollout writes
    serving: Mechanism | None = None          # None: trainer-only, never served
    directive: type[Directive] | None = None  # the per-request record it accepts

    provides: frozenset[str] = frozenset()
    """Training-forward tensors this adapter type computes, by name.

    NOT ONLY THE LOSS-INPUT CHANNEL. A declared provide is emitted per update
    into the ledger's train block and described in the run's own
    dictionary.json (spec/flow.py gives each one a forward node AND a
    per-update stat twin), so an adapter type should provide everything a
    reader of the run would want to WATCH — a posterior's scale, a gate's norm,
    whatever internal state explains the adapter's behavior — and not merely
    what some loss happens to require. Observability is free once the name is
    declared, and a provide nothing requires is first-class: it costs one
    recomputed tensor and buys a curve.
    """

    def site_ok(self, meta: SiteMeta) -> bool:
        """Can this adapter type live at a site with this metadata? Checked at
        Phase 0."""
        return True

    def exports(self, spec: AdapterSpec) -> tuple[SiteMeta, ...]:
        """Sites this bank entry CREATES — points the base checkpoint does not
        have (a soft prompt exports its prompt[:n] positions). Phase-0
        resolution runs against schema ∪ every entry's exports."""
        return ()

    def directive_for(self, request: "Request") -> Directive | None:
        """THIS adapter type's directive among the request's, or None.

        A request carries at most one per adapter type — two would be two
        instructions for one lever — so a second is refused here, by the
        adapter type that would have had to choose.
        """
        if self.directive is None:
            return None
        mine = [d for d in request.directives if isinstance(d, self.directive)]
        if len(mine) > 1:
            raise ValueError(
                f"a request carries {len(mine)} {self.directive.__name__} "
                f"directives; one adapter type takes at most one per request")
        return mine[0] if mine else None

    def record_directive(self, directive: Directive | None,
                         request: "Request") -> Mapping[str, Any]:
        """What the rollout RECORDS about this request's directive — the
        turn facts replay reads back (I6), the same on every engine.

        Called per attached adapter type per request, directive or not: an
        adapter type whose behavior has a default worth recording (a window
        that defaults to "every position") records it here too, so replay
        never has to know what the default was. Default: nothing.
        """
        return {}

    def params(self, sites: tuple[SiteMeta, ...], init: dict) -> Any:
        """Build the trainable parameterization for the matched sites."""
        raise NotImplementedError

    def provide(self, params: Any) -> Mapping[str, Any]:
        """The COMPUTE half of the `provides` declaration: training-forward
        tensors, recomputed by every pass.

        Called inside the routed forward, once per microbatch, and merged into
        PolicyOutputs.provided under the declared names — which is how a loss
        may `require` one (a latent KL, a value head's values) without the
        runner planning any work for it (I9). Grad flows: what comes back here
        is part of the same graph the objective backwards through. An adapter
        type declaring nothing provides nothing.

        Every returned tensor is ALSO summarized to one float for the update's
        ledger line, by the same rule for all of them: a 0-dim tensor is its own
        value, anything else is its mean. Return something that means something
        under that rule — see the `provides` note above on declaring what a
        reader should watch, not only what a loss requires.
        """
        return {}

    def param_groups(self, params: Any) -> Mapping[str, list]:
        """Named optimizer groups for one bank entry — how OptimSpec.overrides
        addresses PARTS of an adapter.

        The default is the whole entry under the empty name: one group, exactly
        the optimizer this learner always built. An adapter type whose pieces
        want different treatment (a hypernet that should decay against a
        posterior that must not) names them here, and `entry.group` in the
        overrides reaches one of them.
        """
        return {"": params.parameters()}

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
