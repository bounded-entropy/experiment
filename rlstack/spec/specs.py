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
    """What the policy may draw on to complete rollouts: the environments it
    MAY run, the task sets it MAY draw from, and the sampling knobs. WHICH task
    runs under WHICH environment is the plan's business, not the spec's.

    `envs` is a declaration, not a use: identity is a pure function of the spec
    value (I3), so the environments whose source hashes into run_id have to be
    nameable without fetching a plan. The submit gate refuses any leaf naming
    an environment not declared here. `makers` is the same declaration for
    the task makers Derive leaves name (mint-then-make: the reflect loop's
    content derivation is registered code, so it hashes like an environment).
    Scoring is NOT here — that is postprocessing."""

    envs: tuple[str, ...]         # registered @environment names
    tasks: tuple[str, ...]        # content-addressed: "cas://<sha>/..." each
    sampling: SamplingSpec = SamplingSpec()
    makers: tuple[str, ...] = ()  # registered @task_maker names


# ---------------------------------------------------------------------------
# the bridge (I2: the policy is the only two-world primitive)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdapterSpec:
    """One adapter: a configured bank entry — one typed intervention of one
    registered adapter type at one site pattern."""

    adapter_type: str             # registered @adapter_type: "lora", "soft_prompt", ...
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
class Plans:
    """The shape of the run, by reference: content-addressed plans.

    `train` is what the Trainer consumes, one wave per update; None means
    nothing trains (a generation-only run — ADR 0006 Part B). `rollout` is
    what the Generator makes, one wave per rollout index; None means the run
    samples nothing (every train leaf is already sealed elsewhere). At least
    one of them exists, or the run has no work and no length — the gate says
    so (check_plans_declare_an_extent). Measurement has no plan here: it is
    not part of the run (a Measurement follows the ledger from outside,
    observe-side).

    Each uri's sha IS the plan's content hash, so a plan hashes into run_id
    exactly as if it were written inline, while the spec stays readable.
    """

    train: str | None = None      # "cas://<sha>"
    rollout: str | None = None

    @property
    def extent(self) -> str:
        """WHICH PLAN IS THE RUN'S LENGTH: the train plan when one exists —
        one wave is one gradient update, so a run is done when its updates
        are — else the rollout plan, whose last sealed wave ends a run that
        only generates.

        A derived property and never a field: the extent is readable off the
        two uris, so no hashed record gains anything and every existing run's
        identity is exactly what it was (ADR 0006 Part B, Q4).
        """
        return "train" if self.train is not None else "rollout"


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
    """How a wave is CONSUMED — its shape is the plan's (#59).

    group_size, trajectories_per_wave and n_updates left with the plan, which
    states all three by construction; epochs_per_wave left with the invariant
    it violated: ONE WAVE IS ONE GRADIENT UPDATE, so training a wave twice is
    two updates whose plans name the same trajectories, and the ledger keeps
    one commit per wave either way.

    What is left is one engineering knob and one statistical one:
    microbatch_tokens changes only how compute is chunked (a wave's documents
    are packed into forwards under this token budget and their gradients
    accumulate into a single step), and max_policy_lag is the lag BUFFER — how
    stale a behavior policy the trainer tolerates, 0 meaning strict
    alternation."""

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
# topology — semantics-neutral (I5): moving members between hosts never
# changes results, only throughput. One HostSpec is one host (ADR 0001).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PoolMember:
    """A pool's declared capacity: the name traffic routes to, serving sample
    and score under it. `tp` is the width the engine handed for it must be
    BUILT at, never a request. `vram_gb` is the memory it needs, in GB, TOTAL
    across its tp shards (the per-device need is vram_gb / tp, so the number
    is invariant under re-sharding, as I5 requires); None means a whole
    device per shard. GB, never a fraction: 0.625 of an L4 is 15 GB and of
    an H100 is 50 GB, and the metal that lands it knows which card it is."""

    name: str
    base: str | None = None       # None → the policy base
    tp: int = 1
    vram_gb: float | None = None  # TOTAL across shards; None = a whole device per shard


@dataclass(frozen=True)
class LearnerMember:
    """The differentiable fwd/bwd workload; `fsdp` is likewise a build width
    the handed learner must already have, and `vram_gb` is total across the
    fsdp shards, None a whole device per shard."""

    fsdp: int = 1
    vram_gb: float | None = None


Member = PoolMember | LearnerMember


@dataclass(frozen=True)
class HostSpec:
    """ONE HOST: a single placement unit, carved as one Partition and listed
    as one Host. One member is a dedicated host; several members ALTERNATE on
    its partition — one resident live at a time (the host's own exclusive
    arbiter group), so a learner alternating with an engine serializes
    generation and training and implies max_policy_lag == 0. Two workloads
    that should run side by side are two HostSpecs, two carves, two honest
    bookings — never one host wearing both at once.

    The member vocabulary is closed at pool + learner; everything else
    (rollout, eval, judge, teacher) is traffic routed to named pools.
    """

    members: tuple[Member, ...]


@dataclass(frozen=True)
class Topology:
    """All hosts. Each pool name appears once; feasibility is checked at
    submit, and what fits WHERE is placement's question, answered against a
    real residual — never here."""

    hosts: tuple[HostSpec, ...]


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
    plans: Plans                  # the shape: which trajectories, which wave
    algo: AlgoSpec | None         # training world; None = generation-only run
    topology: Topology            # semantics-neutral (I5)
    seeds: Seeds
    init: WarmStart | None = None
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

    kind: str                     # a venue name, e.g. "local", "modal", "aws"
    gpu: str = ""
    nodes: int = 1
    idle: str = "keep"
    extra: Mapping[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# sugar — plain constructors, nothing hidden
# ---------------------------------------------------------------------------

def pool(name: str, base: str | None = None, tp: int = 1,
         vram_gb: float | None = None) -> PoolMember:
    """One declared pool, served by its host's metal."""
    return PoolMember(name=name, base=base, tp=tp, vram_gb=vram_gb)


def learner(fsdp: int = 1, vram_gb: float | None = None) -> LearnerMember:
    """The differentiable member."""
    return LearnerMember(fsdp=fsdp, vram_gb=vram_gb)


def lora(site: str, r: int, tie: bool = False) -> AdapterSpec:
    """Per-matrix low-rank delta; tie= shares one (A, B) across all matched sites."""
    return AdapterSpec(adapter_type="lora", site=site, init={"r": r, "tie": tie})


def plora(site: str, k: int, latent: int = 32, members: int = 8,
          prior_std: float = 0.05, hidden: int = 128,
          factors: str | None = None, basis: str = "svd",
          basis_seed: int = 0) -> AdapterSpec:
    """Probabilistic LoRA: a rank-k delta whose k x k core is GENERATED from a
    latent draw, so the policy is a distribution over adapters rather than one.

    Each matched weight M is factored once, offline, into frozen directions
    (`factors`, a "cas://<sha>" artifact); what trains is a small hypernet
    mapping a latent z to each site's core, plus the latent's own posterior
    N(mu, diag(exp(log_std)^2)) against a N(0, prior_std^2) prior. `members`
    is how many draws the engine serves as one ensemble; a request picks one
    by its seed, and score traffic gets the posterior mean.

    `basis` picks the frozen directions' recipe: "svd" is M's own top-k
    singular directions; "random" is the control — the same top-k singular
    VALUES steering random orthonormal directions drawn off `basis_seed`, so
    the two arms differ in the basis and nothing else. The artifact named by
    `factors` must be built with the same recipe (build_factors takes both).
    """
    return AdapterSpec(adapter_type="plora", site=site, init={
        "k": k, "latent": latent, "members": members,
        "prior_std": prior_std, "hidden": hidden, "factors": factors,
        "basis": basis, "basis_seed": basis_seed})


def soft_prompt(site: str, n: int, d: int) -> AdapterSpec:
    """n virtual tokens of width d, served through the native prompt_embeds
    mechanism."""
    return AdapterSpec(adapter_type="soft_prompt", site=site, init={"n": n, "d": d})


def steer(site: str, d: int, tie: bool = False,
          init_std: float = 0.0) -> AdapterSpec:
    """A steering vector of width d at every matched residual boundary
    ("resid_pre.8-20", "final_hidden"), added at every position of a request
    or inside the window the caller passes per request (SteerWindow). tie=
    shares one vector across the range; init_std=0.0 starts at the identity.
    Served on the residual lever, our hook in the engine image (ADR 0004)."""
    return AdapterSpec(adapter_type="steer", site=site,
                       init={"d": d, "tie": tie, "init_std": init_std})


def attn_bias(site: str, **init: object) -> AdapterSpec:
    """Learned bias on an attention-score rectangle: the one adapter type served
    by a mechanism of ours (side_attention, an engine plugin) rather than a
    native one. Its replay half is proven; its rollout half is not, on the
    pinned engine build."""
    return AdapterSpec(adapter_type="attn_bias", site=site, init=init)
