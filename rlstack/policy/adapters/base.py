"""The adapter-type contract: what a registered adapter class declares and computes.

The class registers an ADAPTER TYPE — `AdapterSpec.adapter_type` names it by
string, and a configured bank entry is an adapter. Because the policy is the
only primitive living in both worlds (I2), both lowerings consume one shared
AdapterDefinition. Existing per-side modules are lazy compute providers;
custom site programs use common providers generated from their declaration.
Parity binds the implementations of that one definition (ADR 0028).

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
from rlstack.policy.state import AdapterState
from rlstack.policy.adapters.definition import AdapterDefinition

if TYPE_CHECKING:                       # the seam imports this module back
    from rlstack.policy.adapters.rollout import (
        Request, RolloutLowering, ServingBuild,
    )


class Mechanism(StrEnum):
    """How an engine reaches a site: the CLOSED set of serving levers.

    PUNICA, PROMPT_EMBEDS and LOGITS are the engine's own; SIDE_ATTENTION and
    RESIDUAL are ours, shipped as engine plugins that must re-earn per-request
    selection, cache correctness and parity on every build. Whether a lever
    reaches a given site on a given BUILD is the engine's to answer
    (Engine.reachability), never a static table here (I7).
    """

    PUNICA = "punica"                 # per-token weight deltas (vLLM multi-LoRA)
    PROMPT_EMBEDS = "prompt_embeds"   # virtual rows before the prompt
    LOGITS = "logits"                 # logits processor
    SIDE_ATTENTION = "side_attention" # our LSE-merge plugin (soft prompt + bias)
    RESIDUAL = "residual"             # our hook: a per-slot vector added at a
    #                                   residual boundary (ADR 0004)
    NONE = "none"                     # inventory answer only: not reachable
    SITE = "site"                     # a declared token-local GPU program


Reachability = Mapping[str, Mechanism | frozenset[Mechanism]]


def encode_reachability(reachability: Reachability) -> dict:
    """Preserve legacy scalar rows; multiple providers use sorted name lists."""
    return {name: (value.name if isinstance(value, Mechanism)
                   else sorted(mechanism.name for mechanism in value))
            for name, value in reachability.items()}


def decode_reachability(row: Mapping) -> Reachability:
    return {name: (Mechanism[value] if isinstance(value, str)
                   else frozenset(Mechanism[mechanism] for mechanism in value))
            for name, value in row.items()}


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

    definition: AdapterDefinition | None = None
    adapter_name: str = ""
    engine_plugin: str | None = None          # module the engine image must carry
    records: tuple[str, ...] = ()             # per-token columns the rollout writes
    serving: Mechanism | None = None          # None: trainer-only, never served
    directive: type[Directive] | None = None  # the per-request record it accepts

    provides: frozenset[str] = frozenset()
    def code_modules(self) -> tuple[str, ...]:
        """The declaration and every explicitly selected compute provider."""
        modules = [type(self).__module__]
        definition = self.definition
        if definition is not None:
            if definition.replay is not None:
                modules.append(definition.replay.module)
            if definition.rollout is not None:
                modules.append(definition.rollout.factory.module)
            if definition.program is not None:
                for symbol in (definition.program.forward, definition.program.reference, *definition.program.helpers):
                    if symbol is not None:
                        modules.append(symbol.module)
        return tuple(dict.fromkeys(modules))

    named_states: bool = False
    """Whether this adapter can attach separately named frozen state slots."""

    def named_dependencies(self, routes: Sequence[str]) -> tuple[str, ...]:
        """Named frozen states referenced by this adapter's own route grammar."""
        return ()

    def named_state_route(self, name: str) -> str:
        """The adapter-owned slot address corresponding to one stored name."""
        raise NotImplementedError("this adapter has no named state slots")

    def directive_dependencies(self, directive: Directive) -> tuple[str, ...]:
        return ()

    clips_by_group: bool = False
    """WHOSE GRADIENT NORM IS ONE NORM (ADR 0019): False clips an entry's
    parameters together, as one model; True clips each optimizer group by
    itself. An entry whose groups are INDEPENDENT fits sharing a forward (a
    dream_bank's lanes) must say True, or one lane's large gradient scales
    its neighbours' steps and K lanes stop equalling K tenants."""
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
        return self.definition is None or self.definition.accepts(meta)

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
        if self.definition is None:
            raise NotImplementedError
        from rlstack.policy.adapters.lowering import compile_replay
        return compile_replay(self.definition).build(self.definition.resolve(sites, init))

    def initial_params(self, sites: tuple[SiteMeta, ...],
                       init: Mapping[str, Any]) -> Any:
        """VERSION ZERO OF ONE BANK ENTRY, said in one place: `params` over the
        entry's resolved sites and its init mapping — whose per-entry seed was
        already derived by the runner (loop.init_seed), so the same entry is
        the same object wherever it is built.

        The learner's install builds its v0 here (learners/torch_learner.py)
        and a run with NO learner emits it through `initial_payload` at Phase
        1 (ADR 0006 Part B, Q6). One init function per adapter type is what
        makes those two paths the same bytes rather than two implementations
        that happen to agree.
        """
        return self.params(sites, dict(init))

    def initial_payload(self, sites: tuple[SiteMeta, ...],
                        init: Mapping[str, Any]) -> bytes:
        """That version zero, lowered to the payload an engine serves:
        `emit(initial_params(sites, init))`.

        THE INIT FUNCTION a run with no learner calls. Its bytes are the ones
        a learner would have emitted for the same entry at version 0, so the
        bundle they compile into is the same content-addressed bundle id, and
        a generation-only run serves exactly what a training run's Phase 1
        would have served.
        """
        return self.emit(self.initial_params(sites, init))

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
        if self.definition is not None:
            from rlstack.policy.adapters.lowering import compile_rollout
            return compile_rollout(self.definition, build, self.adapter_name)
        raise NotImplementedError(
            f"{type(self).__name__} has no rollout lowering: it is trainer-"
            f"only (serving is None) and no engine ever hears about it")

    @property
    def supports_mixed_routes(self) -> bool:
        """Whether this replay provider can select several row slots at once.

        False providers still train routed entries in uniform microbatches.
        Transport and checkpoint support do not depend on this capability.
        """
        from rlstack.policy.adapters.definition import Operation
        return self.definition is not None and (
            self.definition.program is not None or self.definition.operation in (
                Operation.LOW_RANK, Operation.STEER, Operation.NORM_STEER))

    def install_replay(self, model: Any, params: Any, sites: tuple[SiteMeta, ...]) -> None:
        """Wire the replay lowering into the trainer forward. Additive: every
        installed tenant stays wired (I8)."""
        if self.definition is None:
            raise NotImplementedError
        from rlstack.policy.adapters.lowering import compile_replay
        compile_replay(self.definition).install(model, params)

    def uninstall_replay(self, model: Any, params: Any, sites: tuple[SiteMeta, ...]) -> None:
        """install_replay's exact inverse: restore the module tree when a
        tenant leaves (the trainer-side mirror of the engine evicting a
        bundle). An adapter type without it cannot share a multi-tenant
        learner."""
        if self.definition is not None:
            from rlstack.policy.adapters.lowering import compile_replay
            compile_replay(self.definition).uninstall(model, params)
            return
        raise NotImplementedError(
            f"{type(self).__name__} has no uninstall_replay: install is "
            f"ADDITIVE, so without its inverse a tenant could never be REMOVED "
            f"from a shared learner")

    def emit(self, params: Any) -> Any:
        """Lower params into the bundle payload the engine-side consumer reads."""
        if self.definition is None:
            raise NotImplementedError
        from rlstack.policy.adapters.lowering import compile_replay
        return compile_replay(self.definition).emit(params)

    def export_state(self, params: AdapterState, version: int):
        """Tensor state at an explicit version, independent of carriage.

        This conversion owns tensor names and semantic metadata, never NCCL
        or base64. It is pure: checkpoint cadence cannot advance ensemble
        draws. Training-only entries use it for checkpointing as well.
        """
        return params.export_state(version)

    def load(self, params: Any, payload: bytes) -> None:
        """emit's inverse: restore params in place from an emitted payload
        (resume and warm-start walk through here)."""
        if self.definition is None:
            raise NotImplementedError
        from rlstack.policy.adapters.lowering import compile_replay
        compile_replay(self.definition).load(params, payload)

    def load_set(self, params: Any, route: str, payload: bytes | None,
                 optimizer: Any) -> None:
        """Start ONE named set of this entry over (ADR 0019): from `payload`
        (a named payload — one LoRA set as `lora_torch.emit` writes it), or
        from its own deterministic init when None; either way that set's
        moments in `optimizer` are reset. Only an adapter type that holds
        sets by route has this verb."""
        raise ValueError(f"{type(self).__name__} holds no sets by route: "
                         f"load_set({route!r}) has nothing to address")

    def emit_set(self, params: Any, route: str) -> bytes:
        """ONE set of this entry as a named payload (ADR 0019)."""
        raise ValueError(f"{type(self).__name__} holds no sets by route: "
                         f"emit_set({route!r}) has nothing to address")

    def drop_set(self, params: Any, route: str) -> None:
        """Forget one LIBRARY set (`lib:<name>`) of this entry (ADR 0019)."""
        raise ValueError(f"{type(self).__name__} holds no sets by route: "
                         f"drop_set({route!r}) has nothing to address")

    def after_step(self, params: Any) -> None:
        """Called by the learner after every optimizer step on this entry —
        the one place a parameterization may put itself back on a manifold
        the step left (a norm-scaled steer's direction back onto the unit
        sphere). A no-op for a type whose parameters are free."""

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
        cls.adapter_name = name
        definition = cls.definition
        digest = source_hash(cls)
        if definition is not None:
            cls.serving = None if definition.mechanism is None else Mechanism(definition.mechanism)
            import hashlib
            from rlstack.policy.adapters.identity_compat import EXTRACTED_DEFINITIONS
            combined = hashlib.sha256((digest + definition.source_hash()).encode()).hexdigest()
            digest = EXTRACTED_DEFINITIONS.get(combined, combined)
        ADAPTER_TYPES.add(AdapterTypeDef(name, cls, cls(), digest))
        return cls
    return register
