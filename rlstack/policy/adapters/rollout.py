"""The rollout seam: what one adapter kind contributes to ONE engine request.

The trainer batches documents into a padded forward and indexes the installed
deltas per ROW (adapters/replay.py); the engine batches requests that pin
different bundles and indexes its levers per REQUEST — the same bridge, the
other side of it (I2, I8). This module is the serving half: one RolloutLowering
per kind, built once per engine build, so no mechanism knowledge has to live in
the engine at all.

FOUR VERBS, read off what serving a mechanism actually costs (#48, completing
the #3 rule that every kind ships BOTH lowerings):

    demands   what the BUILD must pay before this kind can be served — engine
              args, a plugin's presence. A build that cannot pay does not
              serve the kind, and says so at construction rather than at the
              first request.
    attach    make one bundle's state resident: payloads -> the object apply
              and align read. ADDITIVE and once per bundle, the mirror of
              install_replay (I8).
    apply     contribute to ONE unit of work — a request here, a row of the
              padded forward on the trainer side. The units DIFFER by side
              (our plan batches the trainer; vLLM's scheduler batches the
              engine) and the contract does not pretend they are the same.
    align     how many prompt positions this kind's state occupies, so an
              answer read off the prompt (score_tokens) is found where the
              real tokens actually start. SUMMED across a bundle's kinds.

The engine that owns these is a BUS: it loops the bundle's kinds, calls the
verbs, merges the levers and sums the alignments. "Native vs plugin" stops
being a category — it is a demands() difference and nothing else.

torch and vLLM are deliberately absent here: this file is the contract, each
kind's own *_vllm.py is the compute, and a kind imports its heavy half lazily
(STYLE rule 7 — the lora_torch precedent).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    facts (#25b, #43), so a lowering reads them once at construction instead of
    asking the engine mid-flight.
    """

    base: str                  # the checkpoint this metal serves
    config: Any                # the base's HF config (widths, dtype)
    workdir: Path              # scratch this build's lowerings may write into
    max_bundles: int           # how many bundles' state may be resident at once
    max_rank: int              # the widest delta rank this build serves


@dataclass(frozen=True)
class BuildDemands:
    """What the BUILD must pay for one kind to be servable at all.

    `engine_args` are folded into the engine's construction arguments; `plugin`
    is a module the engine IMAGE must carry and install (named by string — the
    import is one-way, rlstack never imports rlstack_engine). A build that
    cannot pay a demand refuses the kind at construction.
    """

    engine_args: Mapping[str, Any] = field(default_factory=dict)
    plugin: str | None = None


@dataclass(frozen=True)
class Request:
    """ONE unit of engine work, as the kinds see it.

    The real tokens of the request, in order — everything a kind needs to place
    itself relative to them. It is deliberately NOT the assembled vLLM prompt:
    a kind shapes its own contribution and the bus merges, so no kind has to
    understand another kind's lever.
    """

    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class Levers:
    """One kind's contribution to one request — apply's answer.

    `prompt` is the request's prompt FORM when this kind shapes it (a kind that
    adds positions must, because the positions are in the prompt); `kwargs` are
    the generate() keywords that select this kind's state per request. A kind
    contributes one or the other or neither, never a bag of untyped extras.
    """

    prompt: Any | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def merged_with(self, other: "Levers") -> "Levers":
        """Fold one kind's contribution into the request assembled so far.

        A later kind's prompt form replaces the earlier one and its keywords
        join — which is only ever unambiguous because add_bundle already
        refused a bundle whose kinds claim the same lever (check_levers_compose
        below), so this fold never has to choose a winner.
        """
        return Levers(
            prompt=self.prompt if other.prompt is None else other.prompt,
            kwargs={**self.kwargs, **other.kwargs})


@dataclass(frozen=True)
class Alignment:
    """How much of the prompt is NOT the request's own tokens.

    `positions` is what this kind's state occupies in front of them, so the
    answer geometry above (score_tokens' scored suffix) is the sum of every
    attached kind's positions plus the context length — the rollout twin of the
    replay boundary's logit trim (#46).
    """

    positions: int = 0


class RolloutLowering:
    """One kind's half of the bridge on the ENGINE side (#3's rollout lowering).

    Subclass per (kind, engine family), set the three class attributes,
    implement the verbs; the kind's declaration half builds one per engine
    through Adapter.rollout_lowering(build). One instance per engine build, so
    per-build bookkeeping (id allocation, scratch dirs) lives here and nowhere
    else.
    """

    kind: ClassVar[str]                      # the registered adapter kind
    mechanism: ClassVar[Mechanism]           # the lever it serves through
    claims: ClassVar[tuple[str, ...]] = ()   # request parts apply() writes:
    #                                          "prompt", or a generate() keyword

    def __init__(self, build: ServingBuild) -> None:
        self.build = build

    def demands(self) -> BuildDemands:
        """What this build must pay to serve the kind (engine args, plugins)."""
        raise NotImplementedError

    def reaches(self, meta: SiteMeta) -> bool:
        """Does the payment above buy this site? The engine's reachability
        inventory is the union of its served kinds' answers (I7, #25b)."""
        raise NotImplementedError

    def attach(self, bundle_id: str, payloads: Mapping[str, bytes]) -> Any:
        """Make one bundle's state resident and return it; the engine holds it
        and hands it back to apply and align. All of THIS kind's payloads
        arrive together, in bank order — a kind compiles its own jointly."""
        raise NotImplementedError

    def apply(self, attached: Any, request: Request) -> Levers:
        """This kind's contribution to one request."""
        raise NotImplementedError

    def align(self, attached: Any) -> Alignment:
        """Prompt positions this kind's state occupies. Default: none — a
        lever that does not add positions leaves the geometry alone."""
        return Alignment()


def check_levers_compose(bundle_id: str,
                         lowerings: Sequence[RolloutLowering]) -> None:
    """#48's COMPOSITION RULE: one request carries one prompt form and one
    value per keyword, so no two of a bundle's kinds may claim the same lever.

    soft_prompt + lora compose because they claim different things. A SECOND
    prompt-shaping kind does not, and the refusal belongs HERE — at
    registration, loudly, while the bundle is still just an id — never at
    sample time, where it would mean a request served the wrong policy.
    """
    claimed: dict[str, str] = {}
    for lowering in lowerings:
        for claim in lowering.claims:
            if claim in claimed:
                raise ValueError(
                    f"{bundle_id}: {lowering.kind!r} and {claimed[claim]!r} "
                    f"both claim the request's {claim!r}, and one request "
                    f"carries one — this engine cannot express that bank")
            claimed[claim] = lowering.kind
