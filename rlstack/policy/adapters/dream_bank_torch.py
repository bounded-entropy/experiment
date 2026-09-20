"""dream_bank's replay lowering: the sets, the routing, the anchor, calibration.

Each set is a `lora_torch.LoraState` — the one-site math is LoRA's, called,
not restated. What is this family's own is the ROUTING (a row's `route`
fact picks its set; `base` picks none), the ANCHOR penalty (a frozen
rank-k-plus-diagonal factorization of the base's input covariance per site,
priced against every memory's delta), and CALIBRATION (accumulating that
covariance from the activations the forward sees, emitted as the anchor).

The payload is one artifact (plora_factors.pack_artifact's head-plus-
safetensors form): every set's (A, B) under a `<route>/` prefix in peft key
form — so the engine side splits it into peft fragments by prefix — and the
anchor's factors under `anchor/`. torch at module scope: this file loads from
the adapter type's lazy imports and the learner, never from the package root.

ADR 0019 adds the NAMED-SET verbs and STACKING. `load_set` / `emit_set` move
ONE set as a named payload — exactly `lora_torch.emit`'s bytes for one LoRA
state, so the store, the engine and this file read one format. A `lib:<name>`
route is a LIBRARY set: a frozen LoraState held beside the entry's own sets,
in no optimizer group, no norm, no penalty and no payload. A stacked route
`<part>+<part>` gives its rows the SUM of the parts' deltas, every library
part through detached factors, so a gradient flows THROUGH a library delta to
the layers below it and never INTO it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from rlstack.policy.adapters import lora_torch
from safetensors.torch import load as st_load

from rlstack.policy.adapters.dream_bank import (
    BASE, DREAMER, MODES, check_route, check_trains_at_most_one, is_library,
    is_memory, parts_of, route_of, set_routes,
)
from rlstack.policy.adapters.lora_torch import PEFT_PREFIX, LoraState
from rlstack.policy.adapters.plora_factors import pack_artifact, unpack_artifact
from rlstack.policy.adapters.replay import ReplayRows, SiteWrapper, join_site, leave_site
from rlstack.policy.siteschema import SiteMeta

FORMAT = "dream_bank/1"
ANCHOR_RIDGE = 1e-3          # C += ridge * tr(C)/d * I before the eigendecomposition


@dataclass
class Anchor:
    """The frozen factors of one protected covariance per path:
    C ≈ U diag(e) Uᵀ + delta·I, with U [d_in, k], e [k], delta the mean of
    the dropped eigenvalues. `captured` is Σe / tr(C): how much of the
    covariance the k directions hold — the calibrate run's manifest number."""

    u: dict[str, torch.Tensor]
    e: dict[str, torch.Tensor]
    delta: dict[str, float]
    captured: dict[str, float] = field(default_factory=dict)
    count: int = 0


@dataclass
class DreamBankState:
    """One entry's state: the sets by route, the anchor, the mode — and the
    LIBRARY sets (`lib:<name>` -> a frozen LoraState), which are deliberately
    not in `sets`: everything that walks `sets` (parameters, param groups,
    provide, emit) is about what this entry OWNS and trains."""

    r: int
    memories: int
    lam: float
    anchor_rank: int
    mode: str
    paths: tuple[str, ...]
    sets: dict[str, LoraState]               # route -> its LoRA state
    anchor: Anchor | None = None
    # calibrate mode: Σ x xᵀ per path in fp32, and the token count
    sums: dict[str, torch.Tensor] = field(default_factory=dict)
    count: int = 0
    # what a set's re-init needs (load_set with no payload): the entry's seed
    # and the sites its sets were built over
    seed: int = 0
    sites: tuple[SiteMeta, ...] = ()
    libraries: dict[str, LoraState] = field(default_factory=dict)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [p for route in set_routes(self.memories)
                for p in self.sets[route].parameters()]


# ---------------------------------------------------------------------------
# build / install
# ---------------------------------------------------------------------------

def _set_seed(seed: int, route: str) -> int:
    digest = hashlib.sha256(f"{seed}:{route}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build(sites: tuple[SiteMeta, ...], init: dict) -> DreamBankState:
    """One LoRA state per set, each seeded off the entry's seed and its route
    name; B = 0 everywhere, so every set starts at the base."""
    mode = str(init.get("mode", "stream"))
    if mode not in MODES:
        raise ValueError(f"dream_bank mode must be one of {MODES}, got {mode!r}")
    memories = int(init.get("memories", 0))
    if memories < 0:
        raise ValueError("dream_bank needs a non-negative memory count")
    seed = int(init.get("seed", 0))
    r = int(init["r"])
    sets = {route: lora_torch.build(sites, {"r": r, "seed": _set_seed(seed, route)})
            for route in set_routes(memories)}
    return DreamBankState(
        r=r, memories=memories, lam=float(init.get("lam", 0.0)),
        anchor_rank=int(init.get("anchor_rank", 128)), mode=mode,
        paths=tuple(meta.path for meta in sites), sets=sets,
        seed=seed, sites=tuple(sites))


def install(model: torch.nn.Module, state: DreamBankState) -> None:
    """Every set's tensors to the site's device, one wrapper per path."""
    for path in state.paths:
        parent, leaf = lora_torch.leaf_module(model, path)
        device = next(getattr(parent, leaf).parameters()).device
        for lora in state.sets.values():
            lora.a[path].data = lora.a[path].data.to(device)
            lora.b[path].data = lora.b[path].data.to(device)
        if state.anchor is not None:
            state.anchor.u[path] = state.anchor.u[path].to(device)
            state.anchor.e[path] = state.anchor.e[path].to(device)
        join_site(model, path, DreamBankSite, state)


def uninstall(model: torch.nn.Module, state: DreamBankState) -> None:
    for path in state.paths:
        leave_site(model, path, DreamBankSite, state)


def param_groups(state: DreamBankState) -> Mapping[str, list]:
    """ONE named group per route — `dreamer`, `memory:00`, … — so a step may
    scale each set's learning rate on its own (`optim_step(lr_scales=...)`,
    ADR 0019: every fit lane decays on its own schedule) and
    OptimSpec.overrides can still give the dreamer its own (`pi.dreamer`).
    The names sort in `set_routes` order, which is what keeps a resumed
    optimizer's groups addressable by index; an override keyed `pi.memory`
    reaches every memory group through the learner's family rule. Library
    sets are in no group: nothing steps them."""
    return {route: list(state.sets[route].parameters())
            for route in set_routes(state.memories)}


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def routes_of(rows: ReplayRows, memories: int) -> list[str]:
    """Every row's route, checked against the sets this entry holds and
    against the training-row rule (at most one trainable part)."""
    if rows.facts is None:
        return [DREAMER] * int(rows.index.shape[0])
    routes = [route_of(turns) for turns in rows.facts]
    for route in routes:
        check_route(route, memories)
        check_trains_at_most_one(route)
    return routes


def lora_of(state: DreamBankState, part: str) -> LoraState:
    """The LoRA state one route part names: an entry set, or a library set
    that `load_set` installed — a library nobody loaded is refused here, at
    the forward that would have silently run without it."""
    if not is_library(part):
        return state.sets[part]
    if part not in state.libraries:
        raise ValueError(
            f"route part {part!r} names a library set this entry does not "
            f"hold (loaded: {sorted(state.libraries) or 'none'}); load_set "
            f"installs one")
    return state.libraries[part]


def part_delta(x: torch.Tensor, state: DreamBankState, part: str,
               path: str) -> torch.Tensor:
    """One part's delta over every row of `x`. An entry set goes through
    LoRA's own whole-batch expression; a LIBRARY set goes through DETACHED
    factors — the delta still depends on `x`, so the gradient reaches the
    trainable deltas in the layers below, and never the library's (A, B)."""
    lora = lora_of(state, part)
    if not is_library(part):
        return lora_torch._whole_batch_delta(x, lora, path)
    a, b = lora.a[path].detach(), lora.b[path].detach()
    return (x.to(a.dtype) @ a.T) @ b.T


class DreamBankSite(SiteWrapper):
    """inner(x) + the delta of each row's own set; `base` rows get none, and
    a stacked row gets the sum of its two parts' deltas."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = self.plan.rows
        slot = rows.uniform()
        if slot is None:
            raise ValueError("dream_bank routes rows within one tenant's forward")
        state = slot.get(self.path)
        if not isinstance(state, DreamBankState):
            return self.inner(x)                     # the transparent case
        if state.mode == "calibrate":
            accumulate(state, self.path, x)
        routes = routes_of(rows, state.memories)
        distinct = set(routes)
        if distinct == {BASE}:
            return self.inner(x)
        if len(distinct) == 1:
            parts = parts_of(routes[0])
            delta = part_delta(x, state, parts[0], self.path)
            for part in parts[1:]:
                delta = delta + part_delta(x, state, part, self.path)
            return self.inner(x) + delta.to(x.dtype)
        if x.dim() != 3 or x.shape[0] != len(routes):
            raise ValueError(
                f"site {self.path}: per-row routing needs [rows, tokens, in] "
                f"activations over {len(routes)} rows, got {tuple(x.shape)}")
        # per-PART: the rows of one set together, ONE pair of matmuls per
        # distinct part (a `base` row gets no delta; a stacked row is in two
        # parts' groups and receives both). Gathering every row's own (A, B)
        # instead — 128 stacked copies per site, three passes per microbatch —
        # made a 7B train step 3× slower per token (2026-09-17).
        out = self.inner(x)
        first = state.sets[DREAMER].a[self.path]
        delta = torch.zeros(x.shape[0], x.shape[1], out.shape[-1],
                            dtype=first.dtype, device=x.device)
        row_parts = [() if route == BASE else parts_of(route) for route in routes]
        for part in sorted({part for parts in row_parts for part in parts}):
            index = torch.tensor([i for i, parts in enumerate(row_parts) if part in parts],
                                 device=x.device)
            piece = part_delta(x.index_select(0, index), state, part, self.path)
            delta = delta.index_add(0, index, piece.to(delta.dtype))
        return out + delta.to(x.dtype)


# ---------------------------------------------------------------------------
# the anchor: penalty and calibration
# ---------------------------------------------------------------------------

def set_penalty(lora: LoraState, anchor: Anchor, paths: Sequence[str]) -> torch.Tensor:
    """½ tr(ΔW C ΔWᵀ) for one set, summed over paths, through the factors:
    with ΔW = B A, M = A U and G = BᵀB,
        tr(ΔW U diag(e) Uᵀ ΔWᵀ) = tr( (M diag(e) Mᵀ) G ),
        δ ‖ΔW‖²_F           = δ tr( (A Aᵀ) G ),
    so nothing of size d_in × d_out is ever formed."""
    total = None
    for path in paths:
        a, b = lora.a[path], lora.b[path]
        u, e, delta = anchor.u[path], anchor.e[path], anchor.delta[path]
        m = a @ u.to(a.dtype)                                           # [r, k]
        g = b.T @ b                                                     # [r, r]
        term = 0.5 * (((m * e.to(a.dtype)) @ m.T) * g).sum() \
             + 0.5 * delta * ((a @ a.T) * g).sum()
        total = term if total is None else total + term
    return total


def delta_norm(lora: LoraState, paths: Sequence[str]) -> torch.Tensor:
    """‖ΔW‖_F over the set's paths, via tr((AAᵀ)(BᵀB))."""
    total = None
    for path in paths:
        a, b = lora.a[path], lora.b[path]
        term = ((a @ a.T) * (b.T @ b)).sum()
        total = term if total is None else total + term
    return total.clamp(min=0).sqrt()


def provide(state: DreamBankState) -> Mapping[str, torch.Tensor]:
    """The penalty (λ times the memories' anchored quadratic form; zero with no
    anchor), the same form UNSCALED (`anchor_raw`, watched so a λ=0 arm still
    reports the magnitude a λ is chosen against), and two watched norms."""
    memory_routes = [route for route in set_routes(state.memories) if is_memory(route)]
    dreamer = state.sets[DREAMER]
    zero = dreamer.a[state.paths[0]].new_zeros(())
    raw = zero
    if state.anchor is not None and memory_routes:
        raw = sum(set_penalty(state.sets[route], state.anchor, state.paths)
                  for route in memory_routes)
    penalty = state.lam * raw if state.lam else zero
    memory_norm = (sum(delta_norm(state.sets[route], state.paths)
                       for route in memory_routes) / len(memory_routes)
                   if memory_routes else zero)
    return {"anchor_penalty": penalty,
            "anchor_raw": raw.detach(),
            "memory_delta_norm": memory_norm.detach(),
            "dreamer_delta_norm": delta_norm(dreamer, state.paths).detach()}


def accumulate(state: DreamBankState, path: str, x: torch.Tensor) -> None:
    """Calibrate mode: Σ x xᵀ over every token the site sees, in fp32."""
    flat = x.detach().reshape(-1, x.shape[-1]).float()
    if path not in state.sums:
        state.sums[path] = torch.zeros(flat.shape[1], flat.shape[1],
                                       dtype=torch.float32, device=flat.device)
    state.sums[path] += flat.T @ flat
    if path == state.paths[0]:
        state.count += flat.shape[0]


def factorize(state: DreamBankState) -> Anchor:
    """The accumulated covariance per path, ridged and eigendecomposed: the
    top-k directions and eigenvalues, the residual as the mean dropped one."""
    if not state.sums or state.count == 0:
        raise ValueError("dream_bank calibrate: no activations accumulated yet")
    u, e, delta, captured = {}, {}, {}, {}
    for path in state.paths:
        c = state.sums[path] / state.count
        d = c.shape[0]
        c = c + (ANCHOR_RIDGE * c.trace() / d) * torch.eye(d, device=c.device)
        values, vectors = torch.linalg.eigh(c)              # ascending
        k = min(state.anchor_rank, d)
        top = torch.arange(d - k, d, device=c.device)
        e[path] = values[top].flip(0).contiguous()
        u[path] = vectors[:, top].flip(1).contiguous()
        dropped = values[: d - k]
        delta[path] = float(dropped.mean()) if len(dropped) else 0.0
        captured[path] = float(e[path].sum() / values.sum())
    return Anchor(u=u, e=e, delta=delta, captured=captured, count=state.count)


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------

def emit(state: DreamBankState) -> bytes:
    """Every set's (A, B) under its route prefix in peft key form, and the
    anchor's factors — a calibrate run's anchor is factorized here, at emit."""
    anchor = factorize(state) if state.mode == "calibrate" and state.sums else state.anchor
    tensors: dict[str, torch.Tensor] = {}
    for route in set_routes(state.memories):
        lora = state.sets[route]
        for path in state.paths:
            tensors[f"{route}/{PEFT_PREFIX}{path}.lora_A.weight"] = lora.a[path].data.cpu()
            tensors[f"{route}/{PEFT_PREFIX}{path}.lora_B.weight"] = lora.b[path].data.cpu()
    meta: dict[str, object] = {
        "format": FORMAT, "r": state.r, "memories": state.memories, "lam": state.lam,
        "anchor_rank": state.anchor_rank, "mode": state.mode, "paths": list(state.paths),
        "routes": list(set_routes(state.memories)), "anchor": anchor is not None,
    }
    if anchor is not None:
        meta["anchor_count"] = anchor.count
        meta["anchor_delta"] = {path: anchor.delta[path] for path in state.paths}
        meta["anchor_captured"] = {path: anchor.captured.get(path, 0.0) for path in state.paths}
        for path in state.paths:
            tensors[f"anchor/{path}/u"] = anchor.u[path].detach().to(torch.bfloat16).cpu()
            tensors[f"anchor/{path}/e"] = anchor.e[path].detach().float().cpu()
    return pack_artifact(meta, tensors)


def load(state: DreamBankState, payload: bytes) -> None:
    """emit's inverse. The sets present in the payload are restored in place
    (a route the payload lacks keeps its init); an anchor in the payload
    becomes this state's anchor. A calibrate checkpoint loaded into a stream
    entry is exactly the warm start ADR 0018 describes."""
    meta, tensors = unpack_artifact(payload)
    if meta.get("format") != FORMAT:
        raise ValueError(f"not a dream_bank payload: {meta.get('format')!r}")
    if int(meta["r"]) != state.r or list(meta["paths"]) != list(state.paths):
        raise ValueError("dream_bank payload does not match this entry's rank or sites")
    device = state.sets[DREAMER].a[state.paths[0]].device
    for route in set_routes(state.memories):
        if route not in meta["routes"]:
            continue
        lora = state.sets[route]
        for path in state.paths:
            lora.a[path].data.copy_(tensors[f"{route}/{PEFT_PREFIX}{path}.lora_A.weight"])
            lora.b[path].data.copy_(tensors[f"{route}/{PEFT_PREFIX}{path}.lora_B.weight"])
    if meta.get("anchor"):
        state.anchor = Anchor(
            u={path: tensors[f"anchor/{path}/u"].to(device) for path in state.paths},
            e={path: tensors[f"anchor/{path}/e"].to(device) for path in state.paths},
            delta={path: float(v) for path, v in meta["anchor_delta"].items()},
            captured={path: float(v) for path, v in meta.get("anchor_captured", {}).items()},
            count=int(meta.get("anchor_count", 0)))


# ---------------------------------------------------------------------------
# named sets (ADR 0019): one route at a time, as a named payload
# ---------------------------------------------------------------------------

def check_named_payload(state: DreamBankState, route: str,
                        tensors: Mapping[str, torch.Tensor]) -> int:
    """A named payload fits this entry when it covers exactly the entry's
    paths with factors of the sites' shapes — and, for a set the entry
    TRAINS, at the entry's rank (its moments and its peers are that rank). A
    library set may be any rank: it is only ever added. Returns the rank."""
    suffix = ".lora_A.weight"
    paths = sorted(key[len(PEFT_PREFIX):-len(suffix)] for key in tensors
                   if key.startswith(PEFT_PREFIX) and key.endswith(suffix))
    if paths != sorted(state.paths) or len(tensors) != 2 * len(paths):
        raise ValueError(
            f"named payload for {route!r} covers paths {paths}, but this "
            f"entry's sites are {sorted(state.paths)}")
    ranks = sorted({int(tensors[f"{PEFT_PREFIX}{path}{suffix}"].shape[0])
                    for path in state.paths})
    if len(ranks) != 1 or (not is_library(route) and ranks[0] != state.r):
        raise ValueError(
            f"named payload for {route!r} has rank {ranks}, but this entry's "
            f"sets are rank {state.r}")
    dreamer = state.sets[DREAMER]
    for path in state.paths:
        a = tensors[f"{PEFT_PREFIX}{path}.lora_A.weight"]
        b = tensors[f"{PEFT_PREFIX}{path}.lora_B.weight"]
        d_in, d_out = dreamer.a[path].shape[1], dreamer.b[path].shape[0]
        if tuple(a.shape) != (ranks[0], d_in) or tuple(b.shape) != (d_out, ranks[0]):
            raise ValueError(
                f"named payload for {route!r} at {path}: A {tuple(a.shape)} / "
                f"B {tuple(b.shape)} do not fit a site of shape "
                f"({d_in} -> {d_out}) at rank {ranks[0]}")
    return ranks[0]


def check_set_route(state: DreamBankState, route: str) -> None:
    """The named-set verbs address ONE set: an entry set or a library part —
    never `base` (no set) and never a stack (two)."""
    check_route(route, state.memories)
    if route == BASE or len(parts_of(route)) != 1:
        raise ValueError(f"{route!r} is not one set: the named-set verbs take "
                         f"`dreamer`, `memory:NN` or `lib:<name>`")


def library_state(state: DreamBankState, tensors: Mapping[str, torch.Tensor],
                  rank: int) -> LoraState:
    """A named payload as a FROZEN LoraState beside the entry's own sets: on
    each site's device, in the sets' dtype, requiring no gradient."""
    dreamer = state.sets[DREAMER]
    def frozen(path: str, leaf: str) -> torch.nn.Parameter:
        like = dreamer.a[path]
        tensor = tensors[f"{PEFT_PREFIX}{path}.{leaf}.weight"]
        return torch.nn.Parameter(tensor.to(device=like.device, dtype=like.dtype),
                                  requires_grad=False)
    return LoraState(r=rank,
                     a={path: frozen(path, "lora_A") for path in state.paths},
                     b={path: frozen(path, "lora_B") for path in state.paths})


def reset_moments(lora: LoraState, optimizer: torch.optim.Optimizer | None) -> None:
    """A set that starts over starts with no history: its Adam moments (and
    step count) leave the optimizer, and any gradient it had accumulated
    goes with them. The other sets' state is untouched."""
    for parameter in lora.parameters():
        parameter.grad = None
        if optimizer is not None:
            optimizer.state.pop(parameter, None)


def load_set(state: DreamBankState, route: str, payload: bytes | None,
             optimizer: torch.optim.Optimizer | None) -> None:
    """Start ONE set over, IN PLACE (the optimizer and the site wrappers hold
    these Parameters by identity).

    `payload` None re-initializes an entry set from its own deterministic
    seed — the bytes `build` gave it. A named payload is copied in after
    `check_named_payload`. Either way the set's moments are reset. A
    `lib:<name>` route installs (or replaces) a frozen library set instead:
    outside the optimizer, outside `emit`, and it needs a payload — a library
    has no init of its own."""
    check_set_route(state, route)
    if is_library(route):
        if payload is None:
            raise ValueError(f"library set {route!r} has no init of its own: "
                             f"load_set needs its named payload")
        tensors = st_load(payload)
        state.libraries[route] = library_state(
            state, tensors, check_named_payload(state, route, tensors))
        return
    lora = state.sets[route]
    if payload is None:
        fresh = lora_torch.build(state.sites, {"r": state.r,
                                               "seed": _set_seed(state.seed, route)})
        for path in state.paths:
            lora.a[path].data.copy_(fresh.a[path].data)
            lora.b[path].data.copy_(fresh.b[path].data)
    else:
        check_named_payload(state, route, st_load(payload))
        lora_torch.load(lora, payload)
    reset_moments(lora, optimizer)


def emit_set(state: DreamBankState, route: str) -> bytes:
    """ONE set as a named payload: `lora_torch.emit` of that set, nothing of
    this family in the bytes — so whatever reads a LoRA reads it."""
    check_set_route(state, route)
    return lora_torch.emit(lora_of(state, route))


def drop_set(state: DreamBankState, route: str) -> None:
    """Forget one LIBRARY set. Only a library can be dropped — an entry set
    is the entry — and a library nobody holds is already dropped."""
    check_set_route(state, route)
    if not is_library(route):
        raise ValueError(f"{route!r} is a set this entry owns; only a "
                         f"library set (lib:<name>) can be dropped")
    state.libraries.pop(route, None)


def split_sets(payload: bytes) -> tuple[dict, dict[str, dict[str, torch.Tensor]]]:
    """The payload's sets as peft-keyed fragments by route — what the engine
    side attaches one adapter per route from. The anchor never ships."""
    meta, tensors = unpack_artifact(payload)
    fragments: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in tensors.items():
        prefix, _, rest = key.partition("/")
        if prefix == "anchor":
            continue
        fragments.setdefault(prefix, {})[rest] = value
    return meta, fragments
