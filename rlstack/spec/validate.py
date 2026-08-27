"""Phase-0 joint validation (I4): everything checkable before a GPU is touched.

One function per rule, named for what it enforces, run in the order CHECKS lists
them. Each check is independent and small; together they cover registry
resolution, declaration wiring (loss.requires vs the bank and pipeline,
post-pipeline wiring), site resolution against schema ∪ bank exports, topology
feasibility, and spec coherence. `validate` returns EVERY issue found (never
just the first); `validate_or_raise` is the submit gate.

One check deliberately lives OUTSIDE the CHECKS table: site reachability
(check_sites_reachable_on) needs an engine build's self-reported inventory,
so the runner runs it at Phase 0 per serving pool.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.siteschema import SiteMeta, SiteSchema, resolve
from rlstack.registry import (
    ADAPTERS,
    ENVS,
    LOSSES,
    POST,
    Probe,
    Ref,
    Teacher,
)
from rlstack.spec.specs import PoolMember, ExperimentSpec, LearnerMember

# PolicyOutputs fields the training forward can always produce, with no bank help.
BASE_PROVIDES = frozenset({"ref_logprobs", "entropies", "hidden_states"})

# Rollout facts every trajectory records regardless of the bank (I6).
BASE_RECORDS = frozenset({"behavior_logprobs", "finish"})

# Loss requirements that name a planned pass (satisfied by the runner, not the bank).
PLANNED_PASSES = (Ref, Teacher, Probe)


@dataclass(frozen=True)
class ValidationIssue:
    """One submit-time failure: stable code, dotted spec path, human message."""

    code: str
    path: str
    message: str


class SpecError(Exception):
    """Raised at submit with every issue found."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = list(issues)
        detail = "; ".join(f"[{i.code}] {i.path}: {i.message}" for i in self.issues)
        super().__init__(f"{len(self.issues)} spec issue(s): {detail}")


# ---------------------------------------------------------------------------
# small shared lookups
# ---------------------------------------------------------------------------

def _issue(code: str, path: str, message: str) -> ValidationIssue:
    return ValidationIssue(code, path, message)


def _declared_pools(spec: ExperimentSpec) -> dict[str, str]:
    """Engine pool name -> the spec path that declares it."""
    pools: dict[str, str] = {}
    for gi, group in enumerate(spec.gpu_config.groups):
        for mi, member in enumerate(group.members):
            if isinstance(member, PoolMember) and member.name not in pools:
                pools[member.name] = f"gpu_config.groups[{gi}].members[{mi}]"
    return pools


# ---------------------------------------------------------------------------
# the checks, in spec order
# ---------------------------------------------------------------------------

def check_names_are_registered(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Every name the spec mentions resolves in its registry.

    This check owns all the unknown-* codes; later checks silently skip names
    it has already reported.
    """
    issues = []

    def look(registry, name: str, code: str, path: str) -> None:
        if name not in registry:
            try:
                registry.get(name)  # raises with the helpful "registered: ..." message
            except KeyError as exc:
                issues.append(_issue(code, path, str(exc.args[0])))

    if spec.algo is not None:
        look(LOSSES, spec.algo.loss, "unknown-loss", "algo.loss")
        for i, name in enumerate(spec.algo.post):
            look(POST, name, "unknown-post", f"algo.post[{i}]")
    if spec.gen is not None:
        look(ENVS, spec.gen.env, "unknown-env", "gen.env")
    if spec.eval is not None:
        if spec.eval.env is not None:
            look(ENVS, spec.eval.env, "unknown-env", "eval.env")
        for i, name in enumerate(spec.eval.post):
            look(POST, name, "unknown-post", f"eval.post[{i}]")
    for entry_name, adapter in spec.policy.bank.items():
        look(ADAPTERS, adapter.kind, "unknown-adapter",
             f"policy.bank.{entry_name}.kind")
    return issues


def check_loss_requires_are_provided(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Every string the loss requires is either PROVIDED (a training-forward
    field: from the base forward or a kind's replay lowering) or RECORDED (a
    sampling-time fact: a base column or a kind's `records`). Planned passes
    (Ref/Teacher/Probe) are the runner's job."""
    if spec.algo is None or spec.algo.loss not in LOSSES:
        return []
    available = set(BASE_PROVIDES) | set(BASE_RECORDS)
    for adapter in spec.policy.bank.values():
        if adapter.kind in ADAPTERS:
            kind = ADAPTERS.get(adapter.kind).instance
            available |= set(kind.provides)
            available |= set(kind.records)
    for name in spec.algo.post:            # the pipeline's postdata columns
        if name in POST:
            available |= set(POST.get(name).produces)
    issues = []
    for req in LOSSES.get(spec.algo.loss).requires:
        if isinstance(req, PLANNED_PASSES):
            continue
        if req not in available:
            issues.append(_issue(
                "unsatisfied-requires", "algo.loss",
                f"loss {spec.algo.loss!r} requires {req!r}, which nothing in the "
                f"bank provides or records; available: {', '.join(sorted(available))}"))
    return issues


def check_post_pipelines_are_wired(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Each post pipeline is internally consistent, in order: every processor's
    `consumes` is produced EARLIER in the same pipeline, and every column has
    exactly one producer."""
    issues = []
    pipelines = []
    if spec.algo is not None:
        pipelines.append(("algo.post", spec.algo.post))
    if spec.eval is not None:
        pipelines.append(("eval.post", spec.eval.post))
    for path, pipeline in pipelines:
        owner: dict[str, str] = {}  # column -> processor that produces it
        for i, name in enumerate(pipeline):
            if name not in POST:
                continue  # unknown-post already reported
            pdef = POST.get(name)
            for want in pdef.consumes:
                if want not in owner:
                    issues.append(_issue(
                        "post-unwired", f"{path}[{i}]",
                        f"postprocessor {name!r} consumes {want!r}, which nothing "
                        f"earlier in the pipeline produces; produced so far: "
                        f"{', '.join(sorted(owner)) or '(none)'}"))
            for column in pdef.produces:
                if column in owner:
                    issues.append(_issue(
                        "post-collision", f"{path}[{i}]",
                        f"postprocessor {name!r} produces {column!r}, already "
                        f"produced by {owner[column]!r} — one owner per column"))
                else:
                    owner[column] = name
    return issues


def site_space(spec: ExperimentSpec, schema: SiteSchema) -> tuple[SiteMeta, ...]:
    """The full site space of this experiment: schema ∪ every entry's exports.

    The schema is a pure function of the base checkpoint; a bank entry may
    CREATE sites the checkpoint does not have (a soft prompt exports its
    prompt[:n] positions). Resolution — here and at Phase 1 — runs against
    the union. Unknown kinds contribute nothing (unknown-adapter reports them).
    """
    exported: list[SiteMeta] = []
    for adapter in spec.policy.bank.values():
        if adapter.kind in ADAPTERS:
            exported.extend(ADAPTERS.get(adapter.kind).instance.exports(adapter))
    return schema.sites + tuple(exported)


def check_schema_describes_the_base(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """The schema handed in must be compiled from this spec's base."""
    if schema.base == spec.policy.base:
        return []
    return [_issue(
        "schema-base-mismatch", "policy.base",
        f"spec base {spec.policy.base!r} but the schema describes {schema.base!r}")]


def check_sites_resolve(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Every bank entry's site pattern matches at least one site in the space."""
    space = site_space(spec, schema)
    issues = []
    for entry_name, adapter in spec.policy.bank.items():
        if not resolve(space, adapter.site):
            issues.append(_issue(
                "site-no-match", f"policy.bank.{entry_name}.site",
                f"site pattern {adapter.site!r} matches no schema entry and no "
                f"bank export"))
    return issues


def check_kinds_accept_their_sites(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Every matched site passes its kind's site_ok predicate."""
    space = site_space(spec, schema)
    issues = []
    for entry_name, adapter in spec.policy.bank.items():
        if adapter.kind not in ADAPTERS:
            continue
        kind = ADAPTERS.get(adapter.kind).instance
        rejected = [m.name for m in resolve(space, adapter.site)
                    if not kind.site_ok(m)]
        if rejected:
            issues.append(_issue(
                "site-predicate-failed", f"policy.bank.{entry_name}.site",
                f"kind {adapter.kind!r} rejects {len(rejected)} matched site(s): "
                f"{', '.join(rejected[:4])}"))
    return issues


def check_sites_reachable_on(
    spec: ExperimentSpec,
    schema: SiteSchema,
    pool: str,
    reachability: Mapping[str, Mechanism],
) -> list[ValidationIssue]:
    """A served kind's mechanism must be how `pool`'s engine reaches every
    matched site.

    Reachability is a property of an engine BUILD (kernel coverage, fusion
    maps, installed plugins), not of the model graph — so this check is NOT in
    CHECKS: the runner asks each serving pool's engine for its self-reported
    inventory (Engine.reachability) at Phase 0 and calls this with the answer.
    Trainer-only kinds (serving None) are never served: nothing to check.
    """
    space = site_space(spec, schema)
    issues = []
    for entry_name, adapter in spec.policy.bank.items():
        if adapter.kind not in ADAPTERS:
            continue
        serving = ADAPTERS.get(adapter.kind).instance.serving
        if serving is None:
            continue
        unreachable = [m.name for m in resolve(space, adapter.site)
                       if reachability.get(m.name, Mechanism.NONE) != serving]
        if unreachable:
            issues.append(_issue(
                "site-unreachable", f"policy.bank.{entry_name}.site",
                f"kind {adapter.kind!r} is served via {serving!r} but the "
                f"{pool!r} engine build does not reach {len(unreachable)} "
                f"matched site(s) that way: {', '.join(unreachable[:4])}"))
    return issues


def check_groups_exist(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """The topology declares at least one group."""
    if spec.gpu_config.groups:
        return []
    return [_issue("no-groups", "gpu_config", "gpu_config declares no groups")]


def check_pool_names_are_unique(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Each engine pool name appears in exactly one group."""
    issues = []
    first: dict[str, str] = {}
    for gi, group in enumerate(spec.gpu_config.groups):
        for mi, member in enumerate(group.members):
            if not isinstance(member, PoolMember):
                continue
            path = f"gpu_config.groups[{gi}].members[{mi}]"
            if member.name in first:
                issues.append(_issue(
                    "duplicate-pool", path,
                    f"engine pool {member.name!r} already declared at {first[member.name]}"))
            else:
                first[member.name] = path
    return issues


def check_sleep_groups_have_one_learner(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """sharing='sleep' means the learner alternates with the engines — exactly one learner."""
    issues = []
    for gi, group in enumerate(spec.gpu_config.groups):
        if group.sharing != "sleep":
            continue
        n_learners = sum(1 for m in group.members if isinstance(m, LearnerMember))
        if n_learners != 1:
            issues.append(_issue(
                "bad-sleep-group", f"gpu_config.groups[{gi}]",
                f"sharing='sleep' alternates one learner with the engines, "
                f"but this group has {n_learners} learner member(s)"))
    return issues


def check_sleep_implies_zero_lag(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Sleep serializes generation and training, so sampled waves are never stale."""
    has_sleep = any(g.sharing == "sleep" for g in spec.gpu_config.groups)
    if not has_sleep or spec.algo is None:
        return []
    lag = spec.algo.schedule.max_policy_lag
    if lag == 0:
        return []
    return [_issue(
        "sleep-lag-conflict", "algo.schedule.max_policy_lag",
        f"sharing='sleep' implies lag 0, but max_policy_lag={lag}")]


def check_fractions_fit(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """When every member of a group states its memory fraction, they must fit in 1.0.

    (A group with any unstated fraction is left to Phase-1 memory probing.)
    """
    issues = []
    for gi, group in enumerate(spec.gpu_config.groups):
        fractions = [m.fraction for m in group.members]
        if not fractions or any(f is None for f in fractions):
            continue
        total = sum(fractions)
        if total > 1.0 + 1e-9:
            issues.append(_issue(
                "fraction-overflow", f"gpu_config.groups[{gi}]",
                f"member fractions sum to {total:.3f} > 1.0"))
    return issues


def check_traffic_routes_to_declared_pools(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Gen traffic needs a pool named 'main'; eval traffic needs its named pool."""
    pools = _declared_pools(spec)
    issues = []
    if spec.gen is not None and "main" not in pools:
        issues.append(_issue(
            "main-pool-missing", "gpu_config",
            f"gen traffic routes to the pool named 'main', which no group declares; "
            f"pools: {', '.join(sorted(pools)) or '(none)'}"))
    if spec.eval is not None and spec.eval.pool not in pools:
        issues.append(_issue(
            "eval-pool-missing", "eval.pool",
            f"eval traffic routes to pool {spec.eval.pool!r}, which no group "
            f"declares; pools: {', '.join(sorted(pools)) or '(none)'}"))
    return issues


def check_post_pools_are_declared(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Every pool a pipeline processor samples from (PostDef.pools) must be a
    declared engine pool — a judge's traffic is vetted at submit, never
    discovered as a KeyError mid-update. "main" needs no declaring here: the
    runner requires it unconditionally."""
    pools = _declared_pools(spec)
    pipelines = []
    if spec.algo is not None:
        pipelines.append(("algo.post", spec.algo.post))
    if spec.eval is not None:
        pipelines.append(("eval.post", spec.eval.post))
    issues = []
    for field, pipeline in pipelines:
        for i, name in enumerate(pipeline):
            if name not in POST:
                continue                    # unknown-post already reported
            for pool in POST.get(name).pools:
                if pool != "main" and pool not in pools:
                    issues.append(_issue(
                        "post-pool-missing", f"{field}[{i}]",
                        f"postprocessor {name!r} samples from pool {pool!r}, "
                        f"which no group declares; pools: "
                        f"{', '.join(sorted(pools)) or '(none)'}"))
    return issues


def traffic_pools(spec: ExperimentSpec) -> set[str]:
    """Every pool this spec's traffic can route to at run time: "main" (gen
    and the pipelines' default client), eval.pool, and each pipeline
    processor's declared pools. The loop holds the engine map it was handed
    against this set before any daemon starts."""
    pools = {"main"}
    if spec.eval is not None:
        pools.add(spec.eval.pool)
        pools.update(p for name in spec.eval.post if name in POST
                     for p in POST.get(name).pools)
    if spec.algo is not None:
        pools.update(p for name in spec.algo.post if name in POST
                     for p in POST.get(name).pools)
    return pools


def check_post_pools_can_coreside(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """A pipeline holds every pool it samples from CO-RESIDENT for its whole
    run — but two pools in one sharing="sleep" group alternate on the same
    memory by declaration, so no admission order can satisfy that pipeline.
    Refused at submit (the arbiter would raise at runtime, later and louder).
    The algo pipeline's set is its processors' declared pools (the trainer
    admits exactly those); the eval pipeline additionally holds eval.pool —
    the evaluator runs episodes and scoring under one admission."""
    sleep_group: dict[str, int] = {}
    for gi, group in enumerate(spec.gpu_config.groups):
        if group.sharing != "sleep":
            continue
        for member in group.members:
            if isinstance(member, PoolMember):
                sleep_group[member.name] = gi
    if not sleep_group:
        return []

    pipelines = []
    if spec.algo is not None:
        pipelines.append(("algo.post", spec.algo.post, ()))
    if spec.eval is not None:
        pipelines.append(("eval.post", spec.eval.post, (spec.eval.pool,)))
    issues = []
    for field, pipeline, held in pipelines:
        sampled = set(held) | {pool for name in pipeline if name in POST
                               for pool in POST.get(name).pools}
        by_group: dict[int, list[str]] = {}
        for pool in sorted(sampled):
            if pool in sleep_group:
                by_group.setdefault(sleep_group[pool], []).append(pool)
        for gi, members in sorted(by_group.items()):
            if len(members) > 1:
                issues.append(_issue(
                    "post-pools-conflict", field,
                    f"pipeline needs pools {members} co-resident, but they "
                    f"alternate in sleep group gpu_config.groups[{gi}] — an "
                    f"alternation set cannot serve one pipeline"))
    return issues


def check_live_trajectories_have_gen(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """source='live' consumes this run's own sealed gen output — gen must exist."""
    if spec.trajectories.source == "live" and spec.gen is None:
        return [_issue(
            "live-without-gen", "trajectories.source",
            "source='live' consumes this run's own sealed gen output, but gen is None")]
    return []


def check_eval_tasks_are_held_out(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Eval is firewalled measurement; measuring on the training tasks is not."""
    if spec.eval is None or spec.gen is None or spec.eval.tasks != spec.gen.tasks:
        return []
    return [_issue(
        "eval-train-overlap", "eval.tasks",
        f"eval tasks {spec.eval.tasks!r} are the training tasks; eval is "
        f"firewalled measurement over held-out tasks")]


def check_schedule_is_sane(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Counts are positive; the lag bound is non-negative."""
    if spec.algo is None:
        return []
    s = spec.algo.schedule
    issues = []
    for name, value in [("group_size", s.group_size),
                        ("trajectories_per_wave", s.trajectories_per_wave),
                        ("n_updates", s.n_updates),
                        ("epochs_per_wave", s.epochs_per_wave),
                        ("microbatch_tokens", s.microbatch_tokens)]:
        if value < 1:
            issues.append(_issue(
                "bad-schedule", f"algo.schedule.{name}",
                f"{name} must be a positive int, got {value}"))
    if s.max_policy_lag < 0:
        issues.append(_issue(
            "bad-schedule", "algo.schedule.max_policy_lag",
            f"max_policy_lag must be >= 0, got {s.max_policy_lag}"))
    return issues


def check_warm_start_map_targets_this_bank(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """WarmStart.map is source-delta-name -> THIS bank's name; the VALUES must exist here."""
    if spec.init is None:
        return []
    issues = []
    for src, dst in spec.init.map.items():
        if dst not in spec.policy.bank:
            issues.append(_issue(
                "warmstart-unknown-delta", f"init.map.{src}",
                f"warm start map entry {src!r} -> {dst!r}: {dst!r} names no delta "
                f"in this bank; bank: {', '.join(sorted(spec.policy.bank)) or '(none)'}"))
    return issues


CHECKS = (
    check_names_are_registered,
    check_loss_requires_are_provided,
    check_post_pipelines_are_wired,
    check_schema_describes_the_base,
    check_sites_resolve,
    check_kinds_accept_their_sites,
    check_groups_exist,
    check_pool_names_are_unique,
    check_sleep_groups_have_one_learner,
    check_sleep_implies_zero_lag,
    check_fractions_fit,
    check_traffic_routes_to_declared_pools,
    check_post_pools_are_declared,
    check_post_pools_can_coreside,
    check_live_trajectories_have_gen,
    check_eval_tasks_are_held_out,
    check_schedule_is_sane,
    check_warm_start_map_targets_this_bank,
)


def validate(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Run every check; an empty list means the spec may be submitted."""
    issues: list[ValidationIssue] = []
    for check in CHECKS:
        issues.extend(check(spec, schema))
    return issues


def validate_or_raise(spec: ExperimentSpec, schema: SiteSchema) -> None:
    """The submit gate: raise SpecError carrying every issue found."""
    issues = validate(spec, schema)
    if issues:
        raise SpecError(issues)
