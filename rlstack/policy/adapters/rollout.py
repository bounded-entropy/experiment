"""The rollout lowering contract: one adapter type's part of ONE engine request.

The trainer routes installed deltas per ROW of a padded forward
(adapters/replay.py); the engine routes its levers per REQUEST — the same
bridge, the other side of it. One RolloutLowering per (adapter type, build),
and its verbs: demands (what the build must pay), reaches (what that payment
buys at a site), attach (make one bundle's state resident, additively), apply
(this adapter type's contribution to one request), align (prompt positions it
occupies, SUMMED across a bundle's adapter types). A `claims` declaration lets
add_bundle refuse two adapter types claiming one request lever while the bundle
is still just an id.

The engine that owns these is a BUS — it loops the adapter types, calls the
verbs, merges the levers and sums the alignments — so "native vs plugin" is a
demands() difference and nothing else. This file is the contract; each adapter
type's own *_vllm.py is the compute.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.siteschema import SiteMeta


@dataclass(frozen=True)
class ServingBuild:
    """The engine build facts a rollout lowering is allowed to know.

    Everything here is fixed before the first bundle arrives and never changes
    afterwards — which is the point: reachability, sizing and dtype are BUILD
    facts, so a lowering reads them once at construction instead of asking the
    engine mid-flight.
    """

    base: str                  # the checkpoint this metal serves
    config: Any                # the base's HF config (widths, dtype)
    workdir: Path              # scratch this build's lowerings may write into
    max_bundles: int           # how many bundles' state may be resident at once
    max_rank: int              # the widest delta rank this build serves
    max_members: int = 0       # widest ENSEMBLE one bundle may be served as
    # How this build reads a content-addressed object ("cas://<sha>" -> bytes).
    # An adapter type whose state is too large to ride in a payload ships the
    # ADDRESS instead and resolves it here; a build handed no reader cannot
    # serve such an adapter type and refuses it at construction, exactly as it
    # refuses a plugin it does not install.
    cas: Callable[[str], bytes] | None = None


@dataclass(frozen=True)
class BuildDemands:
    """What the BUILD must pay for one adapter type to be servable at all.

    `engine_args` are folded into the engine's construction arguments; `plugin`
    is a module the engine IMAGE must carry and install (named by string — the
    import is one-way, rlstack never imports rlstack_engine). A build that
    cannot pay a demand refuses the adapter type at construction.
    """

    engine_args: Mapping[str, Any] = field(default_factory=dict)
    plugin: str | None = None


@dataclass(frozen=True)
class Request:
    """ONE unit of engine work, as the adapter types see it.

    The real tokens of the request, in order — everything an adapter type needs
    to place itself relative to them. It is deliberately NOT the assembled vLLM
    prompt: an adapter type shapes its own contribution and the bus merges, so
    no adapter type has to understand another's lever.

    `seed` is the request's own seed off the episode's sequence, or None for
    score traffic, which draws nothing and is deterministic by contract. An
    adapter type that must CHOOSE something per request draws it from here —
    the same seed tree the rest of the run derives from — so the choice is
    reproducible and the seedless case is a stated branch, never a silent RNG.
    """

    token_ids: tuple[int, ...]
    seed: int | None = None


@dataclass(frozen=True)
class Levers:
    """One adapter type's contribution to one request — apply's answer.

    `prompt` is the request's prompt FORM when this adapter type shapes it (one
    that adds positions must, because the positions are in the prompt);
    `kwargs` are the generate() keywords that select this adapter type's state
    per request. An adapter type contributes one or the other or neither, never
    a bag of untyped extras.

    `turn_extras` is the RECORDING half of the same answer: the sampling-time
    facts this adapter type's choice for this request produced, which the
    engine folds into the FinishEvent and the seal freezes into
    Turn.turn_extras (I6). A fact recorded here is one replay cannot re-derive
    — the draw already happened — so it is data, exactly like the token ids.
    """

    prompt: Any | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    turn_extras: Mapping[str, Any] = field(default_factory=dict)

    def merged_with(self, other: "Levers") -> "Levers":
        """Fold one adapter type's contribution into the request so far.

        A later adapter type's prompt form replaces the earlier one, and its
        keywords and recorded facts JOIN — which is only ever unambiguous
        because add_bundle already refused a bundle whose adapter types claim
        the same lever (check_levers_compose below), so this fold never has to
        choose a winner.
        """
        return Levers(
            prompt=self.prompt if other.prompt is None else other.prompt,
            kwargs={**self.kwargs, **other.kwargs},
            turn_extras={**self.turn_extras, **other.turn_extras})


@dataclass(frozen=True)
class Alignment:
    """How much of the prompt is NOT the request's own tokens.

    `positions` is what this adapter type's state occupies in front of them, so
    an answer read off the prompt (score_tokens' scored suffix) starts at the sum
    of every attached adapter type's positions plus the context length — the
    rollout twin of the replay boundary's logit trim.
    """

    positions: int = 0


class RolloutLowering:
    """One adapter type's half of the bridge on the ENGINE side.

    Subclass per (adapter type, engine family), set the three class attributes,
    implement the verbs; the adapter type's declaration half builds one per
    engine through AdapterType.rollout_lowering(build). One instance per engine
    build, so per-build bookkeeping (id allocation, scratch dirs) lives here and
    nowhere else.
    """

    adapter_type: ClassVar[str]              # the adapter type it serves
    mechanism: ClassVar[Mechanism]           # the lever it serves through
    claims: ClassVar[tuple[str, ...]] = ()   # request parts apply() writes:
    #                                          "prompt", or a generate() keyword

    def __init__(self, build: ServingBuild) -> None:
        self.build = build

    def demands(self) -> BuildDemands:
        """What this build must pay to serve the adapter type (engine args,
        plugins)."""
        raise NotImplementedError

    def reaches(self, meta: SiteMeta) -> bool:
        """Does the payment above buy this site? An engine's reachability
        inventory is the union of its served adapter types' answers (I7)."""
        raise NotImplementedError

    def attach(self, bundle_id: str, payloads: Mapping[str, bytes]) -> Any:
        """Make one bundle's state resident and return it; the engine holds it
        and hands it back to apply and align. All of THIS adapter type's
        payloads arrive together, in bank order — an adapter type compiles its
        own entries jointly."""
        raise NotImplementedError

    def apply(self, attached: Any, request: Request) -> Levers:
        """This adapter type's contribution to one request."""
        raise NotImplementedError

    def align(self, attached: Any) -> Alignment:
        """Prompt positions this adapter type's state occupies. Default: none —
        a lever that does not add positions leaves the geometry alone."""
        return Alignment()

    def detach(self, attached: Any) -> None:
        """attach's exact inverse: release everything it made resident for one
        bundle — scratch on disk, an id, a tensor — so a BOUNDED pool can evict.

        The replay side has had this since #3 (uninstall_replay); the serving
        side did not, and that is precisely why no eviction policy could exist
        here: nothing could release what attach created, so residency only ever
        grew. An adapter type without detach is honestly un-evictable and says
        so, exactly as one without uninstall_replay cannot share a multi-tenant
        learner.

        Detaching is never a loss. A bundle is a committed policy version, and
        restore rebuilds it from the store with the content-addressed id as
        proof it is the same one.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no detach: what attach made resident "
            f"cannot be released, so a pool serving this adapter type can only "
            f"grow")


def check_levers_compose(bundle_id: str,
                         lowerings: Sequence[RolloutLowering]) -> None:
    """THE COMPOSITION RULE: one request carries one prompt form and one value
    per keyword, so no two of a bundle's adapter types may claim the same
    lever.

    soft_prompt + lora compose because they claim different things; a SECOND
    prompt-shaping adapter type does not. The refusal belongs HERE — at
    add_bundle, loudly, while the bundle is still just an id — never at sample
    time, where it would mean a request served the wrong policy.
    """
    claimed: dict[str, str] = {}
    for lowering in lowerings:
        for claim in lowering.claims:
            if claim in claimed:
                raise ValueError(
                    f"{bundle_id}: {lowering.adapter_type!r} and "
                    f"{claimed[claim]!r} both claim the request's {claim!r}, "
                    f"and one request carries one — this engine cannot express "
                    f"that bank")
            claimed[claim] = lowering.adapter_type
