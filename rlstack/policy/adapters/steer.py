"""Steering vectors: one learned vector per residual boundary, ADDED to the
stream at every position of a request — or inside the window the caller
asked for (ADR 0004).

The cheapest intervention the policy can carry: `d` numbers per matched
boundary (`resid_pre.<n>`, `final_hidden`), zero as its exact identity, and
a gradient from the first step. Served on the RESIDUAL lever — a hook in the
engine image that adds each request's own vector at the boundary, per token,
inside one fused batch (I8) — and replayed by a site wrapper that adds the
routed row's vector at the same boundary. Both sides add the same numbers at
the same positions; the parity rail (logprob_gap) measures the bf16 rest.

WHICH positions is a property of the REQUEST: a caller passes a
`SteerWindow` with `sample`, the rollout resolves it against the request it
sees and RECORDS the resolved window as a turn fact (I6), and replay reads
that fact back — so no directive ever asks replay to remember anything. The
default, no directive, is every position: prompt, completion, every decode
step — "steer constantly on decode". A row with NO record gets that same
default at replay: it is a trajectory this policy never sampled (a teacher's
sealed set, a fixed cas file), not a recording bug (ADR 0005, Q3).

init: d (the boundary's width — a boundary site has no shape, so the spec
states it), tie (one vector shared across every matched boundary, else one
per boundary), init_std (0.0: start at the identity; seeded per site when
not).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rlstack.policy.adapters.base import (
    AdapterType, Directive, Mechanism, adapter_type,
)
from rlstack.policy.adapters.rollout import Request
from rlstack.policy.siteschema import SiteMeta

STEER_RECORD = "steer_window"     # the per-turn fact: [start, end], slice coordinates
ALPHA_PROVIDED = "alpha"          # nsteer's provided scalar: the fraction of ||h_t|| injected
# The request's extra_args, read by the engine image's hook: the bundle's
# steer FILE (its address on this build) and the resolved window.
STEER_FILE = "rlstack_steer"
STEER_START = "rlstack_steer_start"
STEER_END = "rlstack_steer_end"


@dataclass(frozen=True)
class SteerWindow(Directive):
    """Which positions of THIS request the steer applies to, in the request's
    REAL-token coordinates: 0 is the first prompt token, len(prompt) the first
    generated one, `end` None runs to the end of generation. Absent: every
    position. `SteerWindow(start=len(prompt))` is completion-only."""

    adapter_type = "steer"
    start: int = 0
    end: int | None = None


def resolve_window(directive: SteerWindow | None,
                   request: Request) -> tuple[int, int | None]:
    """The caller's window in the ENGINE's coordinates: offset by the
    positions other adapter types put in front of the real tokens (a soft
    prompt's rows are in the batch slice too), so the hook and the replay
    wrapper — whose rows carry the same prefix — add at the same places.

    A window that cannot fit is refused here, with the request named, never
    clamped: a clamped window would record something the caller did not ask
    for.
    """
    window = SteerWindow() if directive is None else directive
    if window.start < 0 or (window.end is not None and window.end < window.start):
        raise ValueError(
            f"steer window [{window.start}, {window.end}) cannot fit any "
            f"request: start must be >= 0 and end >= start")
    end = None if window.end is None else window.end + request.occupied
    return window.start + request.occupied, end


def window_record(start: int, end: int | None) -> dict[str, Any]:
    """The turn fact both engines seal: the resolved window, slice
    coordinates, `end` None for open."""
    return {STEER_RECORD: [start, end]}


@adapter_type("steer")
class Steer(AdapterType):
    serving = Mechanism.RESIDUAL
    records = (STEER_RECORD,)
    directive = SteerWindow

    def site_ok(self, meta: SiteMeta) -> bool:
        """A residual boundary: unweighted and boundary — the value head's
        predicate. Which boundaries a BUILD reaches is the engine's answer."""
        return meta.is_boundary and not meta.has_weight

    def record_directive(self, directive: Directive | None,
                         request: Request) -> Mapping[str, Any]:
        """The resolved window, default included, so replay never has to know
        what the default was."""
        start, end = resolve_window(directive, request)
        return window_record(start, end)

    # compute halves — both import torch, so both load lazily, from here
    # only (rule 7)

    def rollout_lowering(self, build):
        from rlstack.policy.adapters import steer_vllm
        return steer_vllm.SteerRollout(build)

    def params(self, sites: tuple[SiteMeta, ...], init: dict):
        from rlstack.policy.adapters import steer_torch
        return steer_torch.build(sites, init)

    def install_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import steer_torch
        steer_torch.install(model, params)

    def uninstall_replay(self, model, params, sites: tuple[SiteMeta, ...]) -> None:
        from rlstack.policy.adapters import steer_torch
        steer_torch.uninstall(model, params)

    def emit(self, params) -> bytes:
        from rlstack.policy.adapters import steer_torch
        return steer_torch.emit(params)

    def load(self, params, payload: bytes) -> None:
        from rlstack.policy.adapters import steer_torch
        steer_torch.load(params, payload)


@adapter_type("nsteer")
class NSteer(Steer):
    """A NORM-SCALED steering vector: the steer family with `alpha`.

    The injection at a token is `alpha * ||h_t|| * v / ||v||` — a fixed
    fraction of the live residual norm at that token, along a learned unit
    direction (the paper's calibration: "the injection norm is always the
    same fraction of the norm of the activation it perturbs"). The residual's
    norm differs across layers, models and tokens, so a fixed-norm add does
    not mean the same thing twice; this one does. `alpha` is the entry's, in
    its init and so in run identity — another fraction is another run.

    What is learned is the DIRECTION and, when `train_alpha` (the default),
    the FRACTION: alpha is a log-parameter — positive by construction — in
    the same optimizer as the direction, and each version's payload writes
    its current value into its metadata, which is where a plot of the
    coefficient over training reads it. The direction lives ON THE UNIT
    SPHERE when `unit` (the default): drawn at unit length, and put back on
    the sphere after every optimizer step (`after_step`), so the parameter
    is the direction and nothing else. Version 0 is therefore a seeded
    random unit direction at alpha, NOT the identity: a norm-scaled steer
    has no zero.

    Everything else is the steer's: the window directive and its record,
    the residual lever, the engine hook (which reads alpha off the fused
    file's metadata and scales per token), the replay wrapper.
    """

    def rollout_lowering(self, build):
        from rlstack.policy.adapters import steer_vllm
        return steer_vllm.NSteerRollout(build)

    provides = frozenset({ALPHA_PROVIDED})

    def provide(self, params) -> Mapping[str, Any]:
        """The fraction this forward injects, as a tensor — the learner
        summarizes it into the ledger's train block every update (`provides`
        is an observability channel), which is where a plot of alpha over
        training reads it; with train_alpha it is the parameter's exp and
        moves, without it the constant."""
        fraction = params.fraction()
        return {} if fraction is None else {ALPHA_PROVIDED: fraction}

    def after_step(self, params) -> None:
        from rlstack.policy.adapters import steer_torch
        steer_torch.project(params)

