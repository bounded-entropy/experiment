"""A bank of LoRA sets inside ONE bank entry: a dreamer and N memories, with
an optional frozen covariance anchor (ADR 0018).

Every set is a plain per-matrix low-rank delta at the same sites; which set
a ROW carries is the row's `route` fact, which set a REQUEST carries is a
`Route` directive. The dreamer is the default on both sides — a request with
no directive samples under it, a row with no fact trains it — so with zero
memories and no anchor this entry is a `lora` entry in bytes and numerics.

A route may also name a LIBRARY set (`lib:<name>`, ADR 0019): a named adapter
loaded beside the entry's own sets, frozen, outside every optimizer group and
outside the entry's payload. And a route may STACK exactly two parts
(`lib:<name>+dreamer`): the row runs under the SUM of the parts' deltas, at
most one of them trainable. `Route.name` stays one string; `parts_of` is the
one place it is taken apart.

The anchor prices what a memory's delta does to the base's activations on
protected text: `anchor_penalty` is provided to the loss once per forward.
In `calibrate` mode the same entry accumulates those activations' covariance
instead, and its checkpoint IS the anchor a stream run warm-starts from.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rlstack.policy.adapters.base import AdapterType, Directive, Mechanism, adapter_type
from rlstack.policy.siteschema import SiteMeta

DREAMER = "dreamer"
BASE = "base"                       # no delta at all: the bare base
MEMORY_PREFIX = "memory:"
LIB_PREFIX = "lib:"                 # a frozen library set, by its store name
STACK = "+"                         # joins the two parts of a stacked route
ROUTE_RECORD = "route"              # the turn fact replay routes by
MODES = ("stream", "calibrate")
LIBRARY_NAME = re.compile(r"[A-Za-z0-9._/-]+")
"""The store's grammar for an adapter name (ADR 0019): no `+` and no `:`, which
is what lets a route be split on them without quoting."""


def memory_route(index: int) -> str:
    """The route name of memory `index`: zero-padded so the names sort."""
    return f"{MEMORY_PREFIX}{index:02d}"


def is_memory(route: str) -> bool:
    return route.startswith(MEMORY_PREFIX)


def set_routes(memories: int) -> tuple[str, ...]:
    """Every set this entry holds, in a fixed order: the dreamer first."""
    return (DREAMER,) + tuple(memory_route(i) for i in range(memories))


def library_route(name: str) -> str:
    """The route part naming the library set `name`."""
    return f"{LIB_PREFIX}{name}"


def is_library(part: str) -> bool:
    """Is this route part a frozen library set (`lib:<name>`)?"""
    return part.startswith(LIB_PREFIX)


def library_name(part: str) -> str:
    """The store name inside a `lib:<name>` part."""
    if not is_library(part):
        raise ValueError(f"route part {part!r} names no library set")
    return part[len(LIB_PREFIX):]


def parts_of(route: str) -> tuple[str, ...]:
    """A route's parts, in the order written: one for a plain route, two for a
    stacked one. Splitting only — `check_route` is the grammar."""
    return tuple(route.split(STACK))


def check_library_name(name: str) -> None:
    """A library part carries a store name: `[A-Za-z0-9._/-]+`, no leading
    slash, no `..` — the same grammar the store writes names under."""
    if (LIBRARY_NAME.fullmatch(name) is None or name.startswith("/")
            or ".." in name.split("/")):
        raise ValueError(
            f"library name {name!r} is not an adapter name: [A-Za-z0-9._/-]+ "
            f"with no leading slash and no '..'")


def check_part(part: str, memories: int) -> None:
    """One part names a set this entry holds, or a library set by name."""
    if is_library(part):
        check_library_name(library_name(part))
        return
    if part in set_routes(memories):
        return
    raise ValueError(f"route {part!r} names no set of a dream_bank with "
                     f"{memories} memories (routes: dreamer, memory:00.., "
                     f"lib:<name>, <part>+<part>, base)")


def check_route(route: str, memories: int) -> None:
    """THE ROUTE GRAMMAR, both sides' (the engine serves by it, replay routes
    by it): a set this entry holds, the bare base, a library set, or a stack
    of EXACTLY two distinct parts — and `base`, being no delta, stacks with
    nothing."""
    if route == BASE:
        return
    parts = parts_of(route)
    if len(parts) == 1:
        check_part(route, memories)
        return
    if len(parts) != 2:
        raise ValueError(f"route {route!r} stacks {len(parts)} parts; a "
                         f"stacked route has exactly two")
    for part in parts:
        if part == BASE:
            raise ValueError(f"route {route!r} stacks `base`, which is no "
                             f"delta; name the other part alone")
        check_part(part, memories)
    if parts[0] == parts[1]:
        raise ValueError(f"route {route!r} stacks a set on itself")


def check_trains_at_most_one(route: str) -> None:
    """The TRAINING-ROW rule on top of the grammar: every `lib:` part is
    frozen, and at most one part of a row's route is trainable — a row's
    gradient belongs to one set."""
    trainable = [part for part in parts_of(route) if not is_library(part)]
    if len(trainable) > 1:
        raise ValueError(f"route {route!r} stacks two trainable sets "
                         f"{trainable}; at most one part of a training row "
                         f"trains")


def library_names_of(route: str) -> tuple[str, ...]:
    """The store names of a route's library parts, in the order written."""
    return tuple(library_name(part) for part in parts_of(route) if is_library(part))


def route_of(turns: Sequence[Mapping[str, object]]) -> str:
    """The row's route: its first turn fact naming one, else the dreamer."""
    for turn in turns:
        if ROUTE_RECORD in turn:
            return str(turn[ROUTE_RECORD])
    return DREAMER


@dataclass(frozen=True)
class Route(Directive):
    """Which set THIS request runs under (ADR 0004 Q2's directive shape):
    `dreamer` (the default), `memory:<jj>`, `base` for no delta, a library
    set `lib:<name>`, or two parts stacked `<part>+<part>` (ADR 0019)."""

    adapter_type = "dream_bank"
    name: str = DREAMER


@adapter_type("dream_bank")
class DreamBank(AdapterType):
    serving = Mechanism.PUNICA
    directive = Route
    records = (ROUTE_RECORD,)
    clips_by_group = True   # every route is its own fit: a lane never scales another's step
    provides = frozenset({"anchor_penalty", "anchor_raw", "memory_delta_norm",
                          "dreamer_delta_norm"})

    def site_ok(self, meta: SiteMeta) -> bool:
        return meta.has_weight

    def record_directive(self, directive: Directive | None,
                         request: Any) -> Mapping[str, Any]:
        """The route a request ran under, recorded whether or not the caller
        said one — replay reads the fact, never the caller's memory (I6)."""
        route = DREAMER if directive is None else directive.name
        return {ROUTE_RECORD: route}

    # compute halves — lazy, from here only (STYLE rule 7)

    def rollout_lowering(self, build):
        from rlstack.policy.adapters import dream_bank_vllm
        return dream_bank_vllm.DreamBankRollout(build)

    def params(self, sites: tuple[SiteMeta, ...], init: dict):
        from rlstack.policy.adapters import dream_bank_torch
        return dream_bank_torch.build(sites, init)

    def install_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import dream_bank_torch
        dream_bank_torch.install(model, params)

    def uninstall_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import dream_bank_torch
        dream_bank_torch.uninstall(model, params)

    def provide(self, params) -> Mapping[str, Any]:
        from rlstack.policy.adapters import dream_bank_torch
        return dream_bank_torch.provide(params)

    def param_groups(self, params) -> Mapping[str, list]:
        from rlstack.policy.adapters import dream_bank_torch
        return dream_bank_torch.param_groups(params)

    def emit(self, params) -> bytes:
        from rlstack.policy.adapters import dream_bank_torch
        return dream_bank_torch.emit(params)

    def load(self, params, payload: bytes) -> None:
        from rlstack.policy.adapters import dream_bank_torch
        dream_bank_torch.load(params, payload)

    # the named-set verbs (ADR 0019) — one route of the entry at a time

    def load_set(self, params, route: str, payload: bytes | None, optimizer) -> None:
        from rlstack.policy.adapters import dream_bank_torch
        dream_bank_torch.load_set(params, route, payload, optimizer)

    def emit_set(self, params, route: str) -> bytes:
        from rlstack.policy.adapters import dream_bank_torch
        return dream_bank_torch.emit_set(params, route)

    def drop_set(self, params, route: str) -> None:
        from rlstack.policy.adapters import dream_bank_torch
        dream_bank_torch.drop_set(params, route)
