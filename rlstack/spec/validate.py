"""Phase-0 joint validation: everything checkable before a GPU is touched (I4).

One named function per rule, run in the order CHECKS lists them. Together they
cover registry resolution, declaration wiring (a loss's requires against the
bank and the pipeline, the post pipeline's own order), site resolution against
schema ∪ bank exports, topology feasibility, and spec coherence. `validate`
returns EVERY issue found, never just the first; `validate_or_raise` is the
submit gate.

Three checks deliberately live OUTSIDE the table because they consult live
metal rather than spec values — check_sites_reachable_on (reachability is a
BUILD fact), check_pools_serve_their_base and check_members_match_their_shape.
The runner calls those at submit, once it holds the engines and the learner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence

from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.siteschema import SiteMeta, SiteSchema, resolve
from rlstack.registry import (
    ADAPTER_TYPES,
    ENVS,
    LOSSES,
    MAKERS,
    POST,
)
from rlstack.spec.specs import PoolMember, ExperimentSpec, LearnerMember

# The base records set lives with the flow graph (spec/flow.py), the one
# canonical walk over the data declarations; re-exported here because it is
# part of the validation vocabulary.
from rlstack.spec.flow import (  # noqa: E402,F401
    BASE_RECORDS, PipelineSplit, flow_graph, split_pipeline,
)

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
    for hi, host in enumerate(spec.topology.hosts):
        for mi, member in enumerate(host.members):
            if isinstance(member, PoolMember) and member.name not in pools:
                pools[member.name] = f"topology.hosts[{hi}].members[{mi}]"
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
        for index, name in enumerate(spec.gen.envs):
            look(ENVS, name, "unknown-env", f"gen.envs[{index}]")
        for index, name in enumerate(spec.gen.makers):
            look(MAKERS, name, "unknown-maker", f"gen.makers[{index}]")
    for entry_name, adapter in spec.policy.bank.items():
        look(ADAPTER_TYPES, adapter.adapter_type, "unknown-adapter",
             f"policy.bank.{entry_name}.adapter_type")
    return issues


def check_loss_requires_are_provided(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Every string the loss requires is PROVIDED (a training-forward tensor,
    from the base forward or an adapter type's replay lowering), RECORDED (a
    sampling-time fact: a base column or an adapter type's `records`), or
    PRODUCED by the post pipeline — a query on the flow graph. The loss is pure math
    (I9): requires names data columns, never work the runner must plan."""
    if spec.algo is None or spec.algo.loss not in LOSSES:
        return []
    graph = flow_graph(spec)
    available = graph.available_to_loss()
    return [_issue(
        "unsatisfied-requires", "algo.loss",
        f"loss {spec.algo.loss!r} requires {req!r}, which nothing in the "
        f"bank provides or records; available: {', '.join(sorted(available))}")
        for req in graph.unsatisfied_requires()]


def check_post_pipelines_are_wired(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Each post pipeline is internally consistent, in order: every processor's
    `consumes` is produced EARLIER in the same pipeline, and every column has
    exactly one producer. Both rules are queries on the flow graph."""
    graph = flow_graph(spec)
    issues = []
    phases = []
    if spec.algo is not None:
        phases.append(("algo.post", "post"))
    for path, phase in phases:
        for i, name, want, produced in graph.missing_consumes(phase):
            issues.append(_issue(
                "post-unwired", f"{path}[{i}]",
                f"postprocessor {name!r} consumes {want!r}, which nothing "
                f"earlier in the pipeline produces; produced so far: "
                f"{', '.join(produced) or '(none)'}"))
        for i, name, column, prior in graph.column_collisions(phase):
            issues.append(_issue(
                "post-collision", f"{path}[{i}]",
                f"postprocessor {name!r} produces {column!r}, already "
                f"produced by {prior!r} — one owner per column"))
    return issues


def check_pooled_post_follows_inline(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """The SPLIT must respect produces→consumes: a POOLED processor may not
    consume a column an INLINE one produces.

    The two halves of a train pipeline meet exactly once — at the postdata part
    the Scorer writes and the Trainer awaits — so the pooled half runs FIRST
    and whole. A pooled processor consuming inline output would be waiting on a
    column written after it: a deadlock by declaration, refused here while it
    is still text. The other direction is the normal case and is fine, because
    the Trainer awaits the part before running its own half and hands those
    columns in as `given`.

    The TRAIN pipeline only. Eval keeps its single inline path (the Evaluator
    is firewalled measurement and plans no Scorer), so nothing constrains its
    order.
    """
    if spec.algo is None:
        return []
    split = split_pipeline(spec.algo.post)
    if not split.pooled:
        return []
    inline_owner = {column: name for name in split.inline if name in POST
                    for column in POST.get(name).produces}
    issues = []
    for index, name in enumerate(spec.algo.post):
        if name not in split.pooled or name not in POST:
            continue
        for want in POST.get(name).consumes:
            if want in inline_owner:
                issues.append(_issue(
                    "post-split-order", f"algo.post[{index}]",
                    f"postprocessor {name!r} sends pool traffic, so the Scorer "
                    f"runs it — but it consumes {want!r}, produced by "
                    f"{inline_owner[want]!r}, which runs INLINE in the Trainer "
                    f"after the Scorer's whole half. Move the producer onto a "
                    f"pool, or the consumer off one"))
    return issues


def site_space(spec: ExperimentSpec, schema: SiteSchema) -> tuple[SiteMeta, ...]:
    """The full site space of this experiment: schema ∪ every bank entry's
    exports.

    The schema is a pure function of the base checkpoint; a bank entry may
    CREATE sites the checkpoint does not have (a soft prompt exports its
    prompt[:n] positions). Resolution — here and at Phase 1 — runs against
    the union. Unknown adapter types contribute nothing (unknown-adapter reports
    them).
    """
    exported: list[SiteMeta] = []
    for adapter in spec.policy.bank.values():
        if adapter.adapter_type in ADAPTER_TYPES:
            exported.extend(ADAPTER_TYPES.get(adapter.adapter_type).instance.exports(adapter))
    return schema.sites + tuple(exported)


def check_schema_describes_the_base(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """The schema handed in must be compiled from this spec's base."""
    if schema.base == spec.policy.base:
        return []
    return [_issue(
        "schema-base-mismatch", "policy.base",
        f"spec base {spec.policy.base!r} but the schema describes {schema.base!r}")]


def check_sites_resolve(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Every adapter's site pattern matches at least one site in the space."""
    space = site_space(spec, schema)
    issues = []
    for entry_name, adapter in spec.policy.bank.items():
        if not resolve(space, adapter.site):
            issues.append(_issue(
                "site-no-match", f"policy.bank.{entry_name}.site",
                f"site pattern {adapter.site!r} matches no schema entry and no "
                f"bank export"))
    return issues


def check_adapter_types_accept_their_sites(
        spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Every matched site passes its adapter type's site_ok predicate."""
    space = site_space(spec, schema)
    issues = []
    for entry_name, adapter in spec.policy.bank.items():
        if adapter.adapter_type not in ADAPTER_TYPES:
            continue
        adapter_type = ADAPTER_TYPES.get(adapter.adapter_type).instance
        rejected = [m.name for m in resolve(space, adapter.site)
                    if not adapter_type.site_ok(m)]
        if rejected:
            issues.append(_issue(
                "site-predicate-failed", f"policy.bank.{entry_name}.site",
                f"adapter type {adapter.adapter_type!r} rejects {len(rejected)} "
                f"matched site(s): {', '.join(rejected[:4])}"))
    return issues


def check_sites_do_not_overlap(
        spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """THE BANK RULE, at the gate: a site carries at most one delta per
    tenant, so no two bank entries may resolve to a common site.

    Stated at the replay seam since #44 (adapters/replay.py) and, until ADR
    0004, enforced nowhere: two entries at one path both installed and both
    applied — two policies wearing one version number, summed silently, for
    every adapter type. The fix for a spec is one entry (a wider pattern, or
    `tie`); across TENANTS nothing here applies — that is I8, and the site
    wrapper's roster. Reported once per pair, at the later entry.
    """
    space = site_space(spec, schema)
    matched = {name: {meta.name for meta in resolve(space, adapter.site)}
               for name, adapter in spec.policy.bank.items()}
    issues = []
    names = list(spec.policy.bank)
    for later_at, later in enumerate(names):
        for earlier in names[:later_at]:
            shared = sorted(matched[earlier] & matched[later])
            if shared:
                issues.append(_issue(
                    "site-overlap", f"policy.bank.{later}.site",
                    f"entries {earlier!r} and {later!r} both resolve to "
                    f"{len(shared)} site(s) ({', '.join(shared[:4])}) — a "
                    f"site carries at most one delta per tenant"))
    return issues


def _plora_entries(spec: ExperimentSpec) -> list[tuple[str, Mapping]]:
    """Every bank entry of adapter type "plora", as (name, init).

    Named once because two checks read it, and spelled by ADAPTER TYPE rather
    than by duck-typing the init: an adapter type is a registered string, so
    the gate can ask about one by name without importing its compute half
    (which would drag torch into Phase 0).
    """
    return [(name, adapter.init) for name, adapter in spec.policy.bank.items()
            if adapter.adapter_type == "plora"]


def check_plora_entries_name_their_factors(
        spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """A plora entry's FROZEN half is a content-addressed artifact, named here.

    The trained half is tiny and travels in the bundle; the singular directions
    it steers are megabytes per site and identical at every version, so they
    live in the CAS and the spec carries the address. An address is not
    optional: without it nothing the engine can serve exists, and identity
    (I3) would silently cover two different factorizations under one run_id.
    """
    issues = []
    for name, init in _plora_entries(spec):
        uri = init.get("factors")
        if not isinstance(uri, str) or not uri.startswith("cas://"):
            issues.append(_issue(
                "plora-factors-missing", f"policy.bank.{name}.init.factors",
                f"plora entry {name!r} carries factors={uri!r}; it must name a "
                f"content-addressed artifact ('cas://<sha>') built by "
                f"plora_factors.build_factors for this base and this k"))
    return issues


def check_plora_shapes_are_positive(
        spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """k, latent and members are counts, and a count of zero is not a smaller
    policy — it is no policy at all (a rank-0 delta, an empty latent, an
    ensemble nobody can be drawn from). Refused here, where they are still
    plain ints, rather than at the first eigh."""
    issues = []
    for name, init in _plora_entries(spec):
        for field, floor in (("k", 1), ("latent", 1), ("members", 1),
                             ("hidden", 1)):
            value = init.get(field)
            if not isinstance(value, int) or isinstance(value, bool) \
                    or value < floor:
                issues.append(_issue(
                    "plora-bad-shape", f"policy.bank.{name}.init.{field}",
                    f"plora entry {name!r} declares {field}={value!r}; it must "
                    f"be an int >= {floor}"))
        prior_std = init.get("prior_std")
        if not isinstance(prior_std, (int, float)) or prior_std <= 0:
            issues.append(_issue(
                "plora-bad-shape", f"policy.bank.{name}.init.prior_std",
                f"plora entry {name!r} declares prior_std={prior_std!r}; the "
                f"prior is N(0, prior_std^2) and a non-positive scale has no "
                f"KL to the posterior"))
    return issues


def check_sites_reachable_on(
    spec: ExperimentSpec,
    schema: SiteSchema,
    pool: str,
    reachability: Mapping[str, Mechanism],
) -> list[ValidationIssue]:
    """A served adapter type's mechanism must be how `pool`'s engine reaches
    every matched site.

    Reachability is a BUILD fact — kernel coverage, fusion maps, installed
    plugins — not a property of the model graph, so this check is NOT in
    CHECKS: the runner asks each serving pool's engine for its self-reported
    inventory (Engine.reachability) at Phase 0 and calls this with the answer.
    Trainer-only adapter types (serving None) are never served: nothing to check.
    """
    space = site_space(spec, schema)
    issues = []
    for entry_name, adapter in spec.policy.bank.items():
        if adapter.adapter_type not in ADAPTER_TYPES:
            continue
        serving = ADAPTER_TYPES.get(adapter.adapter_type).instance.serving
        if serving is None:
            continue
        unreachable = [m.name for m in resolve(space, adapter.site)
                       if reachability.get(m.name, Mechanism.NONE) != serving]
        if unreachable:
            issues.append(_issue(
                "site-unreachable", f"policy.bank.{entry_name}.site",
                f"adapter type {adapter.adapter_type!r} is served via {serving!r} "
                f"but the {pool!r} engine build does not reach {len(unreachable)} "
                f"matched site(s) that way: {', '.join(unreachable[:4])}"))
    return issues


def check_hosts_exist(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """The topology declares at least one host."""
    if spec.topology.hosts:
        return []
    return [_issue("no-hosts", "topology", "topology declares no hosts")]


def check_pool_names_are_unique(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Each engine pool name appears on exactly one host."""
    issues = []
    first: dict[str, str] = {}
    for hi, host in enumerate(spec.topology.hosts):
        for mi, member in enumerate(host.members):
            if not isinstance(member, PoolMember):
                continue
            path = f"topology.hosts[{hi}].members[{mi}]"
            if member.name in first:
                issues.append(_issue(
                    "duplicate-pool", path,
                    f"engine pool {member.name!r} already declared at {first[member.name]}"))
            else:
                first[member.name] = path
    return issues


def alternates_the_learner(host) -> bool:
    """Does this HostSpec make the learner ALTERNATE with an engine? A
    multi-member host is one partition with one resident live at a time
    (ADR 0001, Q1: "alternate" — the word "sleep" survives only as vLLM's
    build fact); when the learner is one of those members, generation and
    training take turns on the same memory."""
    return (len(host.members) > 1
            and any(isinstance(m, LearnerMember) for m in host.members)
            and any(isinstance(m, PoolMember) for m in host.members))


def check_alternation_implies_zero_lag(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """A learner alternating with an engine serializes generation and
    training, so sampled waves are never stale and a lag buffer is a
    contradiction. Two pools alternating on a host of their own bind
    nothing here: the learner elsewhere keeps training while they switch."""
    if spec.algo is None:
        return []
    if not any(alternates_the_learner(h) for h in spec.topology.hosts):
        return []
    lag = spec.algo.schedule.max_policy_lag
    if lag == 0:
        return []
    return [_issue(
        "alternation-lag-conflict", "algo.schedule.max_policy_lag",
        f"a learner alternating with an engine on one host implies lag 0, "
        f"but max_policy_lag={lag}")]


def check_traffic_routes_to_declared_pools(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Gen traffic needs a pool named 'main'; eval traffic needs its named pool."""
    pools = _declared_pools(spec)
    issues = []
    if spec.gen is not None and "main" not in pools:
        issues.append(_issue(
            "main-pool-missing", "topology",
            f"gen traffic routes to the pool named 'main', which no host declares; "
            f"pools: {', '.join(sorted(pools)) or '(none)'}"))
    return issues


def check_pools_serve_their_base(spec: ExperimentSpec, engine_map) -> list[ValidationIssue]:
    """The deploy hands metal; nothing else guarantees it hands the RIGHT
    metal. Each mapped pool's engine must serve that pool's declared base
    (PoolMember.base, defaulting to the policy base); engines reporting base
    None — fakes standing in for metal — serve anything. Outside CHECKS
    because it consults live engine objects, so the loop runs it at submit."""
    declared_base: dict[str, str] = {}
    for host in spec.topology.hosts:
        for member in host.members:
            if isinstance(member, PoolMember):
                declared_base[member.name] = member.base or spec.policy.base
    issues = []
    for name, engine in sorted(engine_map.items()):
        expected = declared_base.get(name)
        served = getattr(engine, "base", None)
        if expected is not None and served is not None and served != expected:
            issues.append(_issue(
                "pool-base-mismatch", f"topology({name})",
                f"pool {name!r} declares base {expected!r} but the engine "
                f"handed for it serves {served!r}"))
    return issues


def check_members_match_their_shape(spec: ExperimentSpec, engine_map,
                                    learner) -> list[ValidationIssue]:
    """Sharding is a BUILD fact, not a request: each mapped pool's engine must
    be BUILT at the pool's declared tensor-parallel width, and the learner at
    the LearnerMember's declared fsdp width. Switching shards means handing
    different metal, never a spec that quietly runs unsharded. Like
    check_pools_serve_their_base, this consults live metal, so the loop runs
    it at submit."""
    declared_tp: dict[str, int] = {}
    for host in spec.topology.hosts:
        for member in host.members:
            if isinstance(member, PoolMember):
                declared_tp[member.name] = member.tp
    issues = []
    for name, engine in sorted(engine_map.items()):
        expected = declared_tp.get(name)
        if expected is not None and engine.tp != expected:
            issues.append(_issue(
                "pool-shape-mismatch", f"topology({name})",
                f"pool {name!r} declares tp={expected} but the engine handed "
                f"for it is built tp={engine.tp}"))
    for host in spec.topology.hosts:
        for member in host.members:
            if not isinstance(member, LearnerMember):
                continue
            if learner is None:
                issues.append(_issue(
                    "learner-missing", "topology(learner)",
                    "the spec declares a learner but none was handed to the "
                    "run: a declared LearnerMember is training metal the run "
                    "expects to reach, locally or over the wire"))
            elif learner.fsdp != member.fsdp:
                issues.append(_issue(
                    "learner-shape-mismatch", "topology(learner)",
                    f"the spec declares fsdp={member.fsdp} but the learner "
                    f"handed is built fsdp={learner.fsdp}"))
    return issues


def check_post_pools_are_declared(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Every pool a pipeline processor addresses (PostDef.pools) must be a
    declared pool — a judge's traffic is vetted at submit, never discovered as
    a KeyError mid-update. "main" needs no declaring here: the runner requires
    it unconditionally."""
    pools = _declared_pools(spec)
    pipelines = []
    if spec.algo is not None:
        pipelines.append(("algo.post", spec.algo.post))
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
                        f"which no host declares; pools: "
                        f"{', '.join(sorted(pools)) or '(none)'}"))
    return issues


def traffic_pools(spec: ExperimentSpec) -> set[str]:
    """Every pool this spec's traffic can address at run time: "main" (gen and
    the pipelines' default client) and each pipeline processor's declared
    pools. The loop holds the engine map it was handed against this set
    before any daemon starts."""
    pools = {"main"}
    if spec.algo is not None:
        pools.update(p for name in spec.algo.post if name in POST
                     for p in POST.get(name).pools)
    return pools


def check_post_pools_can_coreside(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """A pipeline holds every pool it addresses CO-RESIDENT for its whole run —
    but two pools on one multi-member host ALTERNATE on the same memory by
    declaration, so no admission order can satisfy that pipeline. Refused at
    submit; the arbiter would raise at runtime, later and louder. The algo
    pipeline's set is its processors' declared pools (the Scorer admits exactly
    those — they are its whole half by the split rule)."""
    alternating_host: dict[str, int] = {}
    for hi, host in enumerate(spec.topology.hosts):
        if len(host.members) < 2:
            continue
        for member in host.members:
            if isinstance(member, PoolMember):
                alternating_host[member.name] = hi
    if not alternating_host:
        return []

    pipelines = []
    if spec.algo is not None:
        pipelines.append(("algo.post", spec.algo.post, ()))
    issues = []
    for field, pipeline, held in pipelines:
        sampled = set(held) | {pool for name in pipeline if name in POST
                               for pool in POST.get(name).pools}
        by_host: dict[int, list[str]] = {}
        for pool in sorted(sampled):
            if pool in alternating_host:
                by_host.setdefault(alternating_host[pool], []).append(pool)
        for hi, members in sorted(by_host.items()):
            if len(members) > 1:
                issues.append(_issue(
                    "post-pools-conflict", field,
                    f"pipeline needs pools {members} co-resident, but they "
                    f"alternate on topology.hosts[{hi}] — an alternating "
                    f"host cannot serve one pipeline"))
    return issues


def check_plans_declare_an_extent(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """A run has a LENGTH: either a train plan (one wave per update) or a
    rollout plan (one wave per rollout). A spec declaring neither describes a
    run with no work and nothing to be done, which nothing could ever call
    finished (ADR 0006 Part B)."""
    if spec.plans.train is not None or spec.plans.rollout is not None:
        return []
    return [_issue(
        "no-extent", "plans",
        "plans declares neither a train plan nor a rollout plan, so the run "
        "has no extent: nothing to train, nothing to sample, no length")]


def check_learnerless_bank_is_frozen(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """A TRAINABLE bank entry needs a learner to train it, and the topology is
    where a run says it has one.

    A spec declaring no LearnerMember runs without training metal (ADR 0006
    Part B: a generation-only run is a Generator and nothing else), and its
    bank entries are built at their init and never move — so a trainable
    entry there is a contradiction, refused here as text rather than
    surfacing mid-run as `None.install`. Freeze the entry, or declare the
    learner that would train it.
    """
    if any(isinstance(member, LearnerMember)
           for host in spec.topology.hosts for member in host.members):
        return []
    return [_issue(
        "trainable-without-learner", f"policy.bank.{name}",
        f"bank entry {name!r} is trainable but the topology declares no "
        f"learner: nothing in this run could train it")
        for name in sorted(spec.policy.bank)
        if spec.policy.bank[name].trainable]


def check_train_plan_and_algo_agree(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """A TRAIN PLAN AND AN ALGO ARE ONE DECLARATION, read from two ends: the
    plan is what the Trainer consumes, one wave per update, and the Trainer is
    what an algo brings. Either without the other is a run half-described —
    a plan no daemon would consume (and a length nothing could reach), or a
    Trainer with nothing to train on.

    The mirror of check_a_rollout_plan_has_gen: what a daemon needs, the spec
    must declare (ADR 0006 Part B).
    """
    if spec.plans.train is not None and spec.algo is None:
        return [_issue(
            "train-without-algo", "plans.train",
            "a train plan is consumed by the Trainer, but algo is None: with "
            "no loss, no optimizer and no pipeline there is nothing to run "
            "its waves through — drop the train plan (a generation-only run) "
            "or declare the algo that would consume it")]
    if spec.algo is not None and spec.plans.train is None:
        return [_issue(
            "algo-without-train-plan", "plans.train",
            "algo declares a Trainer, but there is no train plan for it to "
            "consume: a run's updates are its train plan's waves, so a "
            "training run without one has no work and no length")]
    return []


def check_a_rollout_plan_has_gen(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """A rollout plan MAKES trajectories, which takes an environment and tasks
    to make them from — so gen must exist to declare both."""
    if spec.plans.rollout is not None and spec.gen is None:
        return [_issue(
            "rollout-without-gen", "plans.rollout",
            "a rollout plan samples, but gen is None: nothing declares which "
            "environments may run or which task sets they may draw from")]
    return []


def check_schedule_is_sane(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Counts are positive; the lag bound is non-negative."""
    if spec.algo is None:
        return []
    s = spec.algo.schedule
    issues = []
    for name, value in [("microbatch_tokens", s.microbatch_tokens)]:
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
    """WarmStart.map is source-bank-name -> THIS bank's name; the VALUES must
    name adapters that exist here."""
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
    check_pooled_post_follows_inline,
    check_schema_describes_the_base,
    check_sites_resolve,
    check_adapter_types_accept_their_sites,
    check_sites_do_not_overlap,
    check_plora_entries_name_their_factors,
    check_plora_shapes_are_positive,
    check_hosts_exist,
    check_pool_names_are_unique,
    check_alternation_implies_zero_lag,
    check_traffic_routes_to_declared_pools,
    check_post_pools_are_declared,
    check_post_pools_can_coreside,
    check_learnerless_bank_is_frozen,
    check_plans_declare_an_extent,
    check_train_plan_and_algo_agree,
    check_a_rollout_plan_has_gen,
    check_schedule_is_sane,
    check_warm_start_map_targets_this_bank,
)


def validate(spec: ExperimentSpec, schema: SiteSchema) -> list[ValidationIssue]:
    """Run every check in CHECKS order; an empty list means the spec may be
    submitted."""
    issues: list[ValidationIssue] = []
    for check in CHECKS:
        issues.extend(check(spec, schema))
    return issues


def validate_or_raise(spec: ExperimentSpec, schema: SiteSchema) -> None:
    """The gate itself: raise SpecError carrying every issue found."""
    issues = validate(spec, schema)
    if issues:
        raise SpecError(issues)
