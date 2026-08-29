"""plora's replay lowering: each matched site's Linear wrapped so its forward
returns inner(x) + ((x A^T) C_r^T) U^T, with the core C GENERATED per row from
that row's recorded latent noise.

Three things live here and nowhere else: the trainable state (a posterior over
one latent, a hypernet, one zero-initialized head per site), the materialization
that turns a latent into cores and cores into peft's lora_B — imported verbatim
by plora_vllm, so both lowerings compute the delta with ONE piece of code — and
the analytic KL the loss prices.

WHAT IS ROUTED AND WHAT IS RECORDED. The wrapper is row-aware exactly as
LoraSite is: install is additive, a row whose slot has no plora here sees the
base, and which state a row uses is the row plan's business. plora adds a
second per-row fact on top of that — the NOISE the rollout drew — which the
learner threads through as ReplayRows.facts. It is read, never re-derived: the
draw already happened, and re-drawing would be a different policy.

torch is imported at module scope — this file loads only from the adapter
type's methods (STYLE rule 7).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from rlstack.policy.adapters import plora_factors
from rlstack.policy.adapters.plora import (
    EPS_RECORD, KL_PROVIDED, SIGMA_PROVIDED,
)
from rlstack.policy.adapters.replay import ReplayRows, RowPlan, leaf_module, row_plan
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.seeds import derive

BLOCKS = 2                  # residual blocks in the hypernet trunk
RMS_EPS = 1e-6
PEFT_PREFIX = "base_model.model."    # peft is punica's format; plora_vllm writes it


# ---------------------------------------------------------------------------
# the trainable state
# ---------------------------------------------------------------------------

def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Pre-norm, without a gain. Deliberately parameter-free: a learnable scale
    here would be one more thing to seed and one more path for the mean delta
    to leave the latent, and the block that follows has a free scale anyway."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)


class Hypernet(torch.nn.Module):
    """latent -> hidden: one projection and BLOCKS pre-norm residual blocks,
    `x + W_out SiLU(W_in RMSNorm(x))`.

    EVERY WEIGHT HERE IS BIAS-FREE, and that is the design, not thrift. A bias
    anywhere in this path would let the network emit a nonzero core at z = 0 —
    a mean delta the posterior never paid for, since the KL prices only mu. Keep
    the map linear-through-the-origin in that sense and the whole delta is
    something the latent explains.

    Parameters are plain drawn tensors rather than nn.Linear because
    constructing a Linear draws from the GLOBAL RNG, and no module in this run
    touches global RNG state: every value below comes from one generator seeded
    off the seed tree.
    """

    def __init__(self, latent: int, hidden: int,
                 generator: torch.Generator) -> None:
        super().__init__()
        self.enter = _drawn(hidden, latent, generator)
        self.inner = torch.nn.ParameterList(
            [_drawn(2 * hidden, hidden, generator) for _ in range(BLOCKS)])
        self.outer = torch.nn.ParameterList(
            [_drawn(hidden, 2 * hidden, generator) for _ in range(BLOCKS)])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z @ self.enter.T
        for inner, outer in zip(self.inner, self.outer):
            x = x + torch.nn.functional.silu(rms_norm(x) @ inner.T) @ outer.T
        return x


def _drawn(out_features: int, in_features: int,
           generator: torch.Generator) -> torch.nn.Parameter:
    """One bias-free weight, N(0, 1/in_features), off the given generator."""
    return torch.nn.Parameter(
        torch.randn(out_features, in_features, generator=generator,
                    dtype=torch.float32) / math.sqrt(in_features))


@dataclass
class PloraState:
    """One bank entry's plora: a posterior over ONE latent, the hypernet that
    reads it, one head per site, and the frozen directions the heads steer.

    `u` and `a` are NOT parameters and NOT part of the payload: they are the
    site's own top-k singular directions, identical at every policy version, so
    they are resolved once at install and shipped to the engine as a
    content-addressed artifact instead (plora_factors). `version` is the noise
    counter — see emit().
    """

    k: int
    latent: int
    members: int
    prior_std: float
    hidden: int
    factors: str                            # "cas://<sha>": the frozen half
    seed: int                               # this entry's init seed
    paths: tuple[str, ...]                  # matched sites, in resolution order
    mu: torch.nn.Parameter                  # [latent]
    log_std: torch.nn.Parameter             # [latent]
    trunk: Hypernet
    heads: dict[str, torch.nn.Parameter]    # path -> [k*k, hidden], zero-init
    u: dict[str, torch.Tensor] = field(default_factory=dict)   # path -> [out, k]
    a: dict[str, torch.Tensor] = field(default_factory=dict)   # path -> [k, in]
    version: int = 0

    def parameters(self) -> list[torch.nn.Parameter]:
        return [self.mu, self.log_std, *self.mapper()]

    def mapper(self) -> list[torch.nn.Parameter]:
        """The deterministic half: everything that turns a latent into cores."""
        return [*self.trunk.parameters(), *self.heads.values()]


def _draw_seed(seed: int, path: str) -> int:
    """The per-draw-site seed, same shape as every other adapter type's."""
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build(sites: tuple[SiteMeta, ...], init: dict) -> PloraState:
    """The identity element, seeded: mu = 0, log_std = log(prior_std), heads = 0.

    Those three choices are one choice. Zero heads make every core zero, so
    Delta = U 0 A = 0 and version 0 IS the base for EVERY latent — the same
    starting point lora's B = 0 gives. mu = 0 with log_std = log(prior_std)
    makes q equal p exactly, so KL(q||p) starts at 0 and the loss's KL term
    begins as a term that is not yet pushing anything. Only the trunk is drawn,
    once, off the entry's seed.
    """
    k, latent = int(init["k"]), int(init["latent"])
    hidden, members = int(init["hidden"]), int(init["members"])
    prior_std = float(init["prior_std"])
    seed = int(init.get("seed", 0))
    for meta in sites:
        if meta.shape is None:
            raise ValueError(f"plora needs a weighted site, got {meta.name}")
    trunk = Hypernet(latent, hidden, torch.Generator().manual_seed(
        _draw_seed(seed, "plora.trunk")))
    return PloraState(
        k=k, latent=latent, members=members, prior_std=prior_std, hidden=hidden,
        factors=str(init["factors"]), seed=seed,
        paths=tuple(meta.path for meta in sites),
        mu=torch.nn.Parameter(torch.zeros(latent, dtype=torch.float32)),
        log_std=torch.nn.Parameter(
            torch.full((latent,), math.log(prior_std), dtype=torch.float32)),
        trunk=trunk,
        heads={meta.path: torch.nn.Parameter(
            torch.zeros(k * k, hidden, dtype=torch.float32)) for meta in sites})


# ---------------------------------------------------------------------------
# the latent, the cores, the delta — the math BOTH lowerings use
# ---------------------------------------------------------------------------

def reparameterized_latent(state: PloraState,
                           eps: torch.Tensor) -> torch.Tensor:
    """z = mu + exp(log_std) * eps — THE reparameterization.

    `eps` is what the rollout drew and sealed; the posterior is what the trainer
    currently holds. Composing them here is what puts mu and log_std on the
    gradient path of a sample that was taken before either had its present
    value, which is the entire reason the NOISE is recorded and z is not.
    """
    return state.mu + torch.exp(state.log_std) * eps


def materialize_core(state: PloraState,
                     z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Every site's k x k core for one latent (or a batch of them): the trunk
    once, then one head per site. `z` is [latent] or [rows, latent]; a core is
    [k, k] or [rows, k, k]."""
    hidden = state.trunk(z)
    return {path: _head_core(state, path, hidden) for path in state.paths}


def _head_core(state: PloraState, path: str,
               hidden: torch.Tensor) -> torch.Tensor:
    """One site's core out of the shared trunk activation."""
    return (hidden @ state.heads[path].T).reshape(
        *hidden.shape[:-1], state.k, state.k)


def materialize_b(state: PloraState,
                  cores: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """B = U C: the peft `lora_B` a core becomes.

    peft applies inner(x) + x A^T B^T with scaling alpha/r, and plora emits
    alpha == k so the scaling is exactly 1 — so writing lora_A = A and
    lora_B = U C makes the engine compute x A^T C^T U^T, which is the replay
    expression below, term for term. lora_A never changes between versions;
    this is the only moving part of a plora bundle.
    """
    return {path: state.u[path].to(cores[path].dtype) @ cores[path]
            for path in cores}


def analytic_kl(mu: torch.Tensor, log_std: torch.Tensor,
                prior_std: float) -> torch.Tensor:
    """KL(N(mu, diag(sigma^2)) || N(0, prior_std^2 I)), in closed form:
    sum_i [ log(prior_std/sigma_i) + (sigma_i^2 + mu_i^2)/(2 prior_std^2) - 1/2 ].

    Analytic, not sampled, because it can be: both sides are diagonal Gaussians,
    so the estimator would only add variance to a quantity we can write down.
    Exactly 0 at init (mu = 0, sigma = prior_std), which is what makes the KL
    term in a loss start at zero rather than at an arbitrary offset.
    """
    variance = torch.exp(2.0 * log_std)
    return (math.log(prior_std) - log_std
            + (variance + mu.pow(2)) / (2.0 * prior_std ** 2)
            - 0.5).sum()


def provide(state: PloraState) -> dict[str, Any]:
    """The declared provided tensors, recomputed by this forward."""
    return {KL_PROVIDED: analytic_kl(state.mu, state.log_std, state.prior_std),
            SIGMA_PROVIDED: torch.exp(state.log_std).mean()}


def param_groups(state: PloraState) -> dict[str, list]:
    """`mapper` and `posterior` — the two halves an OptimSpec may address
    separately (torch_learner._group_settings reads "entry.mapper")."""
    return {"mapper": state.mapper(), "posterior": [state.mu, state.log_std]}


# ---------------------------------------------------------------------------
# the row-aware site
# ---------------------------------------------------------------------------

class PloraSite(torch.nn.Module):
    """inner(x) + ((x A^T) C^T) U^T with the core generated from the ROW's own
    recorded latent — the module that replaces a matched Linear.

    LoraSite's structure exactly: one wrapper per site serving every installed
    state, `installed` the roster that makes install additive and tells
    uninstall when the Linear returns, and a transparent case for a row whose
    slot has no plora here. The one thing added is that a row's delta depends on
    a FACT as well as a slot, so the fast path needs both to be uniform.
    """

    def __init__(self, inner: torch.nn.Module, path: str, plan: RowPlan) -> None:
        super().__init__()
        self.inner = inner
        self.path = path
        self.plan = plan
        self.installed: list[PloraState] = []

    def add(self, state: PloraState) -> None:
        """Additive install: this state's delta becomes routable here."""
        if any(present is state for present in self.installed):
            raise RuntimeError(f"install at {self.path}: this state is already "
                               f"installed — install/uninstall out of balance")
        self.installed.append(state)

    def drop(self, state: PloraState) -> None:
        """install's inverse at one site; the caller unwraps when empty."""
        kept = [present for present in self.installed if present is not state]
        if len(kept) == len(self.installed):
            raise RuntimeError(f"uninstall at {self.path}: this state was never "
                               f"installed — install/uninstall out of balance")
        self.installed = kept

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A row's delta is generated from its slot's hypernet and its own
        recorded noise — and a slot that carries no plora here gets the base."""
        rows = self.plan.rows
        one = rows.uniform()
        if one is not None and self.path not in one:
            return self.inner(x)              # the transparent case
        return self.inner(x) + self._delta(x, rows).to(x.dtype)

    def _delta(self, x: torch.Tensor, rows: ReplayRows) -> torch.Tensor:
        """One site's contribution, on whichever path this forward needs.

        THE PARITY ANCHOR: one slot AND one latent is the plain three-GEMM
        expression over the whole batch, so a single-tenant microbatch of one
        trajectory is bit-identical however many tenants share the learner —
        the same guarantee ReplayRows.uniform() buys lora, with the extra
        condition plora's per-row fact adds.
        """
        states = _states_at(rows, self.path)
        noise = _row_noise(rows, states, self.path)
        shared = _one_noise(noise)
        if len(states) == 1 and shared is not None:
            return _whole_batch_delta(x, states[0], self.path, shared)
        return _per_row_delta(x, states, rows, noise, self.path)


def _states_at(rows: ReplayRows, path: str) -> list[PloraState]:
    """Each slot's plora state at this site, in slot order.

    Every slot of a MIXED forward must carry this path, for lora's reason: the
    transparent case is uniform-only, because zero-padding one row's delta is
    the coalescer's admission rule and not a silent fallback here.
    """
    missing = [i for i, slot in enumerate(rows.slots) if path not in slot]
    if missing:
        raise ValueError(
            f"site {path}: slots {missing} carry no delta here, so a mixed "
            f"forward cannot route this site (a uniform forward serves them "
            f"the base; admission is the coalescer's job)")
    return [slot[path] for slot in rows.slots]


def _row_noise(rows: ReplayRows, states: Sequence[PloraState],
               path: str) -> torch.Tensor:
    """[rows, latent]: the noise each row's trajectory was SAMPLED with.

    ONE LATENT PER TRAJECTORY is the rule this enforces. A document's turns are
    the requests of one episode, all served by the same drawn member, so turns
    that disagree mean the recording and the routing have come apart — and
    averaging two latents would silently train a policy that never sampled
    anything. It raises instead.
    """
    if rows.facts is None:
        raise ValueError(
            f"site {path}: a plora replay forward needs the noise the rollout "
            f"drew ({EPS_RECORD!r} in each turn's extras) and this batch "
            f"recorded none — the latent was SAMPLED, so it is not "
            f"re-derivable from the parameters")
    latent = states[0].latent
    return torch.tensor(
        [_one_row_noise(turns, row, latent, path)
         for row, turns in enumerate(rows.facts)],
        dtype=torch.float32, device=rows.index.device)


def _one_row_noise(turns: Sequence[Mapping[str, Any]], row: int, latent: int,
                   path: str) -> list[float]:
    """One row's latent noise, out of its turns' recorded extras."""
    drawn = [tuple(float(v) for v in turn[EPS_RECORD])
             for turn in turns if EPS_RECORD in turn]
    if not drawn:
        raise ValueError(
            f"site {path}: row {row} recorded no {EPS_RECORD!r} — every turn "
            f"served by a plora bundle records the noise its member used")
    distinct = set(drawn)
    if len(distinct) > 1:
        raise ValueError(
            f"site {path}: row {row}'s turns recorded {len(distinct)} different "
            f"latents — one trajectory is one draw, so this row's recording and "
            f"its routing disagree")
    eps = drawn[0]
    if len(eps) != latent:
        raise ValueError(
            f"site {path}: row {row} recorded a {len(eps)}-wide latent but this "
            f"entry declares latent={latent}")
    return list(eps)


def _one_noise(noise: torch.Tensor) -> torch.Tensor | None:
    """The one noise vector every row carries, or None when they differ."""
    if noise.shape[0] == 0:
        return None
    first = noise[0]
    return first if bool((noise == first).all()) else None


def _whole_batch_delta(x: torch.Tensor, state: PloraState, path: str,
                       eps: torch.Tensor) -> torch.Tensor:
    """Every row on one latent: ((x A^T) C^T) U^T in one GEMM triple — the
    plain single-trajectory expression, kept verbatim so a uniform microbatch is
    bit-identical whatever else shares the learner."""
    core = _head_core(state, path,
                      state.trunk(reparameterized_latent(state, eps)))
    a = state.a[path].to(core.dtype)
    u = state.u[path].to(core.dtype)
    return ((x.to(core.dtype) @ a.T) @ core.T) @ u.T


def _per_row_delta(x: torch.Tensor, states: Sequence[PloraState],
                   rows: ReplayRows, noise: torch.Tensor,
                   path: str) -> torch.Tensor:
    """Rows on different latents (or different slots): generate every row's
    core, then three batched GEMMs — punica's shape, written in stock torch.

    The cores are gathered exactly as lora gathers (A, B): every slot's core is
    computed for every row and the row's own is selected by index. That costs
    slots x rows tiny hypernet passes instead of a scatter, which is the right
    trade because the hypernet is a few hundred KB against a base forward, and
    because it keeps the expression differentiable and one line long.
    """
    ranks = sorted({state.k for state in states})
    if len(ranks) > 1:
        raise ValueError(
            f"site {path}: one forward's slots must agree on rank, got {ranks} "
            f"(punica zero-pads its slot bank to max_rank; the coalescer will)")
    if x.dim() != 3 or x.shape[0] != rows.index.shape[0]:
        raise ValueError(
            f"site {path}: per-row deltas need [rows, tokens, in] activations "
            f"over the plan's {rows.index.shape[0]} rows, got {tuple(x.shape)}")
    picked = torch.arange(noise.shape[0], device=rows.index.device)
    cores = torch.stack([                                        # [S, R, k, k]
        _head_core(state, path,
                   state.trunk(reparameterized_latent(state, noise)))
        for state in states])[rows.index, picked]                # [R, k, k]
    a = torch.stack([state.a[path] for state in states])[rows.index]  # [R,k,in]
    u = torch.stack([state.u[path] for state in states])[rows.index]  # [R,out,k]
    activations = x.to(cores.dtype)
    return torch.bmm(
        torch.bmm(torch.bmm(activations, a.to(cores.dtype).transpose(1, 2)),
                  cores.transpose(1, 2)),
        u.to(cores.dtype).transpose(1, 2))


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------

def install(model: torch.nn.Module, state: PloraState) -> None:
    """Wrap the Linear once at each path, RESOLVE the frozen directions, then
    ADD this state to the site.

    The factors are recomputed here from the base's own weight rather than read
    from the artifact, and that is deliberate: the trainer is already holding
    the checkpoint, `top_svd_factors` is a pinned pure function of (weight, k),
    and the artifact exists precisely for the side that ISN'T holding it — the
    engine, whose copy lives inside vLLM. ALGO_ID is what makes the two agree;
    an artifact built by another recipe is refused where it is read.

    Installation is additive (I8): a second tenant at the same site joins the
    PloraSite it finds. Placement happens here — the posterior, the hypernet and
    the frozen directions all move to the device the site's weight lives on.
    """
    plan = row_plan(model)
    for path in state.paths:
        parent, leaf = leaf_module(model, path)
        site = getattr(parent, leaf)
        if not isinstance(site, PloraSite):
            site = PloraSite(site, path, plan)
            setattr(parent, leaf, site)
        weight = _site_weight(site.inner, path)
        u, a = plora_factors.top_svd_factors(weight, state.k)
        state.u[path] = u.to(weight.device)
        state.a[path] = a.to(weight.device)
        _place(state, weight.device)
        site.add(state)


def _site_weight(inner: torch.nn.Module, path: str) -> torch.Tensor:
    """The matrix this site's frozen directions come from."""
    weight = getattr(inner, "weight", None)
    if weight is None:
        raise ValueError(
            f"plora at {path}: the site's module has no `weight` to factor — "
            f"site_ok admits weighted matrices only")
    return weight


def _place(state: PloraState, device: torch.device) -> None:
    """Move the trained half beside the base, before any optimizer or load
    exists (the same rule lora's install keeps)."""
    for parameter in state.parameters():
        parameter.data = parameter.data.to(device)


def uninstall(model: torch.nn.Module, state: PloraState) -> None:
    """install's exact inverse: drop this state from each site, and unwrap the
    PloraSite back to its Linear when the last state leaves. The params object
    survives untouched — its tensors are held by reference — so re-install
    restores identical numerics."""
    for path in state.paths:
        parent, leaf = leaf_module(model, path)
        site = getattr(parent, leaf)
        if not isinstance(site, PloraSite):
            raise RuntimeError(
                f"uninstall at {path}: expected PloraSite, found "
                f"{type(site).__name__} — install/uninstall out of balance")
        site.drop(state)
        if not site.installed:
            setattr(parent, leaf, site.inner)


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------

def noise_for(seed: int, version: int, members: int,
              latent: int) -> torch.Tensor:
    """[members, latent]: the ensemble this policy version is served as.

    One generator per member off the seed tree, so the draw is a pure function
    of (entry seed, version, member) — the same version always ships the same
    ensemble, on any machine, in any process, which is what lets the payload
    hash into a bundle id and a resumed run re-emit byte-identical bytes.
    """
    return torch.stack([
        torch.randn(latent, dtype=torch.float32,
                    generator=torch.Generator().manual_seed(
                        derive(seed, "plora", version, member)))
        for member in range(members)])


def emit(state: PloraState) -> bytes:
    """Lower the trained half into the bundle payload — and DRAW the ensemble.

    What ships is kilobytes: the posterior, the hypernet, one head per site, and
    the `members` noise vectors this version is served as. The frozen directions
    do NOT ship; the payload carries their cas ADDRESS, and the engine resolves
    it once per build.

    THE COUNTER IS THE VERSION, and that is what makes resume work. emit uses
    the counter it holds, RECORDS it, and only then advances, while load
    restores it — so the n-th emit of a run always draws the n-th ensemble
    whether it happened in this process or in the one that crashed. This is the
    one adapter type whose payload is not a pure function of its parameters
    alone, because the served ensemble is a fresh DRAW at each version, and the
    counter is what keeps that draw reproducible.
    """
    tensors = {"posterior.mu": state.mu.data.cpu(),
               "posterior.log_std": state.log_std.data.cpu(),
               "noise": noise_for(state.seed, state.version, state.members,
                                  state.latent)}
    for key, value in state.trunk.state_dict().items():
        tensors[f"trunk.{key}"] = value.cpu()
    for path, head in state.heads.items():
        tensors[f"heads.{path}"] = head.data.cpu()
    payload = plora_factors.pack_artifact({
        "k": state.k, "latent": state.latent, "members": state.members,
        "prior_std": state.prior_std, "hidden": state.hidden,
        "factors": state.factors, "paths": list(state.paths),
        "version": state.version}, tensors)
    state.version += 1
    return payload


def resident(payload: bytes,
             factors: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
             ) -> PloraState:
    """The state a SERVING side rebuilds: one payload plus the frozen
    directions it resolved, and nothing training-only.

    The engine materializes its ensemble with the SAME functions the trainer
    differentiates through (materialize_core, materialize_b) — which is what
    makes plora's two lowerings one piece of arithmetic rather than two that
    have to be kept in agreement. The shapes are built empty and then filled
    through `load`, so there is exactly one restore path.
    """
    meta, _ = plora_factors.unpack_artifact(payload)
    latent, hidden, k = int(meta["latent"]), int(meta["hidden"]), int(meta["k"])
    paths = tuple(meta["paths"])
    state = PloraState(
        k=k, latent=latent, members=int(meta["members"]),
        prior_std=float(meta["prior_std"]), hidden=hidden,
        factors=str(meta["factors"]), seed=0, paths=paths,
        mu=torch.nn.Parameter(torch.zeros(latent, dtype=torch.float32)),
        log_std=torch.nn.Parameter(torch.zeros(latent, dtype=torch.float32)),
        trunk=Hypernet(latent, hidden, torch.Generator().manual_seed(0)),
        heads={path: torch.nn.Parameter(
            torch.zeros(k * k, hidden, dtype=torch.float32)) for path in paths})
    load(state, payload)
    state.u = {path: factors[path][0] for path in paths}
    state.a = {path: factors[path][1] for path in paths}
    return state


def served_noise(payload: bytes) -> torch.Tensor:
    """The [members, latent] ensemble this payload was emitted with — read
    back rather than redrawn, so the engine serves the draw the version was
    SEALED with even if it never sees the seed that made it."""
    return plora_factors.unpack_artifact(payload)[1]["noise"]


def peft_config(base: str, k: int, target_modules: list[str]) -> str:
    """adapter_config.json for one materialized member.

    alpha == r == k, so peft's alpha/r scaling is exactly 1 and the engine
    computes x A^T B^T verbatim — the replay expression with B = U C folded in
    (materialize_b). The rank IS plora's k because a materialized member is an
    ordinary rank-k LoRA; nothing about the ensemble survives into this file's
    format, which is the point.
    """
    import json

    return json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base,
        "r": k, "lora_alpha": k, "lora_dropout": 0.0, "bias": "none",
        "target_modules": sorted(set(target_modules)),
    }, indent=2)


def member_tensors(state: PloraState, z: torch.Tensor) -> dict[str, torch.Tensor]:
    """One member as peft's own key space: lora_A = A (frozen, the same at
    every version) and lora_B = U C for this member's latent.

    Both are written in the FACTORS' dtype, so one member's directory is
    homogeneous: the cores are generated in fp32 and A arrives as whatever the
    artifact stored, and a peft directory holding one of each would be an
    inconsistency the engine silently resolves on load.
    """
    b = materialize_b(state, materialize_core(state, z))
    tensors: dict[str, torch.Tensor] = {}
    for path in state.paths:
        dtype = state.a[path].dtype
        tensors[f"{PEFT_PREFIX}{path}.lora_A.weight"] = state.a[path].cpu()
        tensors[f"{PEFT_PREFIX}{path}.lora_B.weight"] = b[path].to(dtype).cpu()
    return tensors


def load(state: PloraState, payload: bytes) -> None:
    """emit's inverse, in place — including the noise counter, so the next emit
    continues the run's ensemble sequence rather than restarting it."""
    meta, tensors = plora_factors.unpack_artifact(payload)
    if int(meta["latent"]) != state.latent or int(meta["k"]) != state.k:
        raise ValueError(
            f"this payload carries k={meta['k']}, latent={meta['latent']}; the "
            f"entry it is being loaded into declares k={state.k}, "
            f"latent={state.latent}")
    state.mu.data.copy_(tensors["posterior.mu"])
    state.log_std.data.copy_(tensors["posterior.log_std"])
    state.trunk.load_state_dict(
        {key[len("trunk."):]: value for key, value in tensors.items()
         if key.startswith("trunk.")})
    for path, head in state.heads.items():
        head.data.copy_(tensors[f"heads.{path}"])
    state.version = int(meta["version"])
