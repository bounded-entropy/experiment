"""The specs: frozen dataclasses, declarative, and they ARE identity (I3).

A spec is a value. It never does setup and never touches a GPU; a runner
resolves it at Phase 0. Mapping-typed fields are plain dicts because hashing is
what makes identity order-independent — canonical_json sorts every mapping, so
two banks written in different literal orders are the same experiment. Treat
specs as immutable all the way down: never mutate a dict after passing it in.

Reading order below: inference world → the bridge → training world → topology
→ run level → deploy-time → sugar.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# inference world
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingSpec:
    """Behavior-policy knobs. Science, not config: it hashes into run_id."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 1024


@dataclass(frozen=True)
class GenSpec:
    """Everything needed for the policy to complete rollouts: the env, its
    tasks, the sampling knobs. Scoring is NOT here — that is postprocessing."""

    env: str                      # registered @environment
    tasks: str                    # content-addressed: "cas://<sha>/..."
    sampling: SamplingSpec = SamplingSpec()


@dataclass(frozen=True)
class EvalSpec:
    """Firewalled measurement: immutable bundle versions + held-out tasks only."""

    tasks: str
    every: int = 10               # run after every N optim updates
    env: str | None = None        # None → gen.env
    post: tuple[str, ...] = ()    # scoring pipeline over eval trajectories
    n_samples: int = 1
    pool: str = "main"            # which engine pool carries eval traffic


# ---------------------------------------------------------------------------
# the bridge (I2: the policy is the only two-world primitive)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdapterSpec:
    """One adapter: a configured bank entry — one typed intervention of one
    registered kind at one site pattern."""

    kind: str                     # registered @adapter: "lora", "soft_prompt", ...
    site: str                     # canonical name, resolved against the SiteSchema
    init: Mapping[str, object] = field(default_factory=dict)
    trainable: bool = True


@dataclass(frozen=True)
class PolicySpec:
    """A base plus a bank of named adapters: the one primitive that lives in
    both worlds (I2), and so the one carrying a parity obligation."""

    base: str                     # HF id + pinned revision
    bank: Mapping[str, AdapterSpec]


# ---------------------------------------------------------------------------
# training world
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectorySource:
    """What training consumes — always sealed store data (I1).

    "live" means this run's own gen output; "store://<run>/..." and "cas://<sha>"
    replay another run's sealed data (off-policy distillation, SFT) through the
    exact same training contract.
    """

    source: str

    def __post_init__(self) -> None:
        ok = self.source == "live" or self.source.startswith(("store://", "cas://"))
        if not ok:
            raise ValueError(
                f"TrajectorySource.source must be 'live', 'store://...', or "
                f"'cas://...', got {self.source!r}"
            )


@dataclass(frozen=True)
class OptimSpec:
    """Optimizer plus per-adapter overrides (param groups keyed by bank name)."""

    name: str
    lr: float
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    overrides: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


@dataclass(frozen=True)
class Schedule:
    """The wave shape. Mostly statistical knobs (they change the estimator),
    one engineering knob (microbatch_tokens changes only how compute is
    chunked). max_policy_lag is the lag BUFFER — how stale a behavior policy
    the trainer tolerates, 0 meaning strict alternation."""

    group_size: int
    trajectories_per_wave: int
    n_updates: int
    epochs_per_wave: int = 1
    microbatch_tokens: int = 16384
    max_policy_lag: int = 0


@dataclass(frozen=True)
class AlgoSpec:
    """The training world, by registered name; declarations travel with the names (I4).

    `post` is the ORDERED postprocessing pipeline: per-group processors run
    after the seal (rewards, judges, advantages, ...), their columns land in
    postdata, and the loss `requires` the ones it uses.
    """

    loss: str                     # registered @loss
    post: tuple[str, ...]         # registered @postprocessor names, in order
    optim: OptimSpec
    schedule: Schedule


# ---------------------------------------------------------------------------
# topology — semantics-neutral (I5): moving members between groups never
# changes results, only throughput
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuSet:
    """Pure device demand: what, never where (I5) — placement decides that.
    Literal ids are only for pinning a dedicated daemon."""

    n: int | None = None
    nodes: int = 1
    ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PoolMember:
    """A pool's declared capacity: the name traffic routes to, serving sample
    and score under it. `tp` is the width the engine handed for it must be
    BUILT at, never a request."""

    name: str
    base: str | None = None       # None → the policy base
    tp: int = 1
    n: int = 1
    fraction: float | None = None # share of the group's GPU memory


@dataclass(frozen=True)
class LearnerMember:
    """The differentiable fwd/bwd workload; `fsdp` is likewise a build width
    the handed learner must already have."""

    fsdp: int = 1
    fraction: float | None = None


Member = PoolMember | LearnerMember


@dataclass(frozen=True)
class GpuGroup:
    """The unit of colocation: members co-resident on one GpuSet.

    The member vocabulary is closed at pool + learner; everything else
    (rollout, eval, judge, teacher) is traffic routed to named pools.
    sharing="sleep" makes it an exclusive group: the learner alternates with
    the engines on the same memory, and therefore implies max_policy_lag == 0.
    """

    gpus: GpuSet
    members: tuple[Member, ...]
    sharing: str = "concurrent"

    def __post_init__(self) -> None:
        if self.sharing not in ("concurrent", "sleep"):
            raise ValueError(
                f"GpuGroup.sharing must be 'concurrent' or 'sleep', got {self.sharing!r}"
            )


@dataclass(frozen=True)
class GpuConfig:
    """All groups. Each pool name appears once; feasibility is checked at submit."""

    groups: tuple[GpuGroup, ...]


# ---------------------------------------------------------------------------
# run level
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Seeds:
    """The root of the seed tree: every random draw in a run is h(master,
    *path), so no module ever touches global RNG state and any part of a run
    can be regenerated in isolation."""

    master: int


@dataclass(frozen=True)
class WarmStart:
    """Start a NEW experiment from another run's sealed state.

    Hashes into run_id; the parent is recorded in the manifest — lineage
    without ceremony. `map` renames adapters on the way in: source bank name →
    name in THIS spec's bank.
    """

    policy: str                   # "store://<run_id>@<version>" or "cas://<sha>"
    optim: str = "fresh"          # "load" restores Adam moments; "fresh" starts over
    map: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.optim not in ("load", "fresh"):
            raise ValueError(
                f"WarmStart.optim must be 'load' or 'fresh', got {self.optim!r}"
            )
        if not self.policy.startswith(("store://", "cas://")):
            raise ValueError(
                f"WarmStart.policy must be sealed state ('store://...' or "
                f"'cas://...'), got {self.policy!r}"
            )


@dataclass(frozen=True)
class ExperimentSpec:
    """The whole experiment as one value: the unit of identity, of store
    ownership, and of a run directory."""

    policy: PolicySpec            # the bridge (I2)
    gen: GenSpec | None           # inference world; None = pure-offline run
    trajectories: TrajectorySource   # the training world's ONLY input (I1)
    algo: AlgoSpec | None         # training world; None = generation-only run
    gpu_config: GpuConfig         # semantics-neutral (I5)
    seeds: Seeds
    init: WarmStart | None = None
    eval: EvalSpec | None = None
    tier: str = "lab"             # "lab" | "release"

    def __post_init__(self) -> None:
        if self.tier not in ("lab", "release"):
            raise ValueError(
                f"ExperimentSpec.tier must be 'lab' or 'release', got {self.tier!r}"
            )


# ---------------------------------------------------------------------------
# deploy-time — OUTSIDE the spec; never hashes into run_id
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BackendProfile:
    """Where a run executes. The manifest records which profile ran it."""

    kind: str                     # "local" | "modal" | "skypilot"
    gpu: str = ""
    nodes: int = 1
    idle: str = "keep"
    extra: Mapping[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# sugar — plain constructors, nothing hidden
# ---------------------------------------------------------------------------

def gpus(n: int | None = None, nodes: int = 1, ids: tuple[str, ...] | None = None) -> GpuSet:
    """gpus(6) | gpus(16, nodes=2) | gpus(ids=("0", "1"))."""
    return GpuSet(n=n, nodes=nodes, ids=ids)


def pool(name: str, base: str | None = None, tp: int = 1, n: int = 1,
            fraction: float | None = None) -> PoolMember:
    """One declared pool, served by this group's metal."""
    return PoolMember(name=name, base=base, tp=tp, n=n, fraction=fraction)


def learner(fsdp: int = 1, fraction: float | None = None) -> LearnerMember:
    """The differentiable member."""
    return LearnerMember(fsdp=fsdp, fraction=fraction)


def lora(site: str, r: int, tie: bool = False) -> AdapterSpec:
    """Per-matrix low-rank delta; tie= shares one (A, B) across all matched sites."""
    return AdapterSpec(kind="lora", site=site, init={"r": r, "tie": tie})


def soft_prompt(site: str, n: int, d: int) -> AdapterSpec:
    """n virtual tokens of width d, served through the native prompt_embeds
    mechanism."""
    return AdapterSpec(kind="soft_prompt", site=site, init={"n": n, "d": d})


def attn_bias(site: str, **init: object) -> AdapterSpec:
    """Learned bias on an attention-score rectangle: the one kind served by a
    mechanism of ours (side_attention, an engine plugin) rather than a native
    one. Its replay half is proven; its rollout half is not, on the pinned
    engine build."""
    return AdapterSpec(kind="attn_bias", site=site, init=init)
