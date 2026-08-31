"""The flow graph: THE canonical walk over a spec's data declarations.

flow_graph(spec) builds the graph of every named artifact a run will contain —
pipeline columns, recorded facts, forward-recomputed tensors, the standing
rails — with edges taken from the declarations verbatim (produces / consumes /
requires). Everything that reasons over those declarations queries THIS object:
the submit gate's pipeline checks and the run's own dictionary.json (I11) are
two consumers of one walk, so the graph cannot drift from the semantics without
validation failing with it, and a new declaration kind gets a node here once.

The loss is pure math: its `requires` may only name nodes of this graph — post
columns, records, bank-provided forward tensors — never a pass the runner would
have to plan.

`split_pipeline` lives here for the same reason: it is one more query over the
same declarations, read by the submit gate and by the runner, and putting it
anywhere else would make the spec depend on the runtime that obeys it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rlstack.registry import ADAPTER_TYPES, LOSSES, POST
from rlstack.spec.specs import ExperimentSpec

# Rollout facts every trajectory records regardless of the bank (I6).
BASE_RECORDS = frozenset({"behavior_logprobs", "finish"})

# What every loss reports into the ledger, regardless of family.
RAILS = ("loss", "mean_ratio", "logprob_gap", "grad_norm")

# Trainer bookkeeping in the same ledger summary — stored, plottable,
# panel-addressable, but not the loss's rails.
TRAIN_STATS = ("tokens", "microbatches")


@dataclass(frozen=True)
class PipelineSplit:
    """One post pipeline cut by WHERE its processors run.

    `pooled` addresses pools, so it is the Scorer daemon's half, run beside the
    metal it talks to; `inline` touches no pool, so it is arithmetic over
    columns and stays in the Trainer's own post phase. Each half keeps the
    pipeline's declared order, and their concatenation is a permutation of it —
    nothing is dropped and nothing is run twice.
    """

    pooled: tuple[str, ...]
    inline: tuple[str, ...]


def split_pipeline(pipeline: Sequence[str]) -> PipelineSplit:
    """THE SPLIT RULE: a processor declaring `pools` is SCORER-RUN, a pool-less
    one is TRAINER-INLINE.

    Declared, never guessed. `pools` already names every pool a processor sends
    traffic to — the submit gate vets it and the runner admits engines for it —
    so the same declaration answers WHICH daemon runs it: sending traffic is
    what makes a processor slow, and slow is what has to leave the gradient's
    critical path. A pipeline with no pooled half plans no Scorer at all and
    the Trainer's post phase is exactly what it always was.

    An unregistered name is inline: `check_names_are_registered` owns that
    failure, so this walk never raises.
    """
    pooled: list[str] = []
    inline: list[str] = []
    for name in pipeline:
        (pooled if name in POST and POST.get(name).pools else inline).append(name)
    return PipelineSplit(tuple(pooled), tuple(inline))


@dataclass(frozen=True)
class FlowNode:
    """One named data artifact of a run.

    kind: "column" (a post pipeline output) | "record" (a sampling-time
    fact, frozen at the seal) | "provided" (a training-forward tensor,
    recomputed each pass) | "rail" (a loss's standing report) | "stat"
    (trainer bookkeeping in the same ledger summary).
    phase: where it lives — "post" (postdata/<u>), "eval" (eval/<u>),
    "wave" (per token in waves/<u>), "forward" (never stored), "train"
    (the ledger's train summary).
    feeds_loss: transitively reachable into the loss's requires — the UI's
    primary-panel bit; measurement-only artifacts have it False.
    """

    name: str
    kind: str
    phase: str
    producer: str
    consumers: tuple[str, ...]
    feeds_loss: bool
    stored: bool
    granularity: str = "trajectory"     # trajectory | token | update


@dataclass(frozen=True)
class FlowGraph:
    """One spec's data artifacts, in pipeline order, plus the queries the
    submit gate and dictionary.json are built from."""

    nodes: tuple[FlowNode, ...]
    loss: str | None
    lag: int
    post_pipeline: tuple[str, ...]
    eval_pipeline: tuple[str, ...]

    # ---- queries (validate's pipeline rules are these, rendered) ------------

    def missing_consumes(self, phase: str) -> tuple[tuple[int, str, str, tuple[str, ...]], ...]:
        """(index, processor, column, produced-so-far) for every consumes
        with no earlier producer in that phase's pipeline."""
        out = []
        produced: list[str] = []
        for i, name in enumerate(self._pipeline(phase)):
            if name not in POST:
                continue
            pdef = POST.get(name)
            for want in pdef.consumes:
                if want not in produced:
                    out.append((i, name, want, tuple(sorted(produced))))
            produced.extend(c for c in pdef.produces if c not in produced)
        return tuple(out)

    def column_collisions(self, phase: str) -> tuple[tuple[int, str, str, str], ...]:
        """(index, processor, column, prior-owner) for every column produced
        twice within one phase — one owner per column."""
        out = []
        owner: dict[str, str] = {}
        for i, name in enumerate(self._pipeline(phase)):
            if name not in POST:
                continue
            for column in POST.get(name).produces:
                if column in owner:
                    out.append((i, name, column, owner[column]))
                else:
                    owner[column] = name
        return tuple(out)

    def available_to_loss(self) -> frozenset[str]:
        """Every name a loss's string requires may resolve against."""
        return frozenset(
            node.name for node in self.nodes
            if node.phase in ("post", "wave", "forward"))

    def unsatisfied_requires(self) -> tuple[str, ...]:
        """String requirements of the loss that nothing provides, records,
        or produces."""
        if self.loss is None or self.loss not in LOSSES:
            return ()
        available = self.available_to_loss()
        return tuple(req for req in LOSSES.get(self.loss).requires
                     if req not in available)

    # ---- the run's self-description -----------------------------------------

    def to_json(self) -> dict:
        """dictionary.json: what this run's store will contain and why.
        Derived, never identity — a UI walks it instead of re-deriving any
        declaration, so it needs no registry and cannot skew."""
        return {
            "columns": [{
                "name": n.name, "kind": n.kind, "phase": n.phase,
                "producer": n.producer, "consumers": list(n.consumers),
                "feeds_loss": n.feeds_loss, "stored": n.stored,
                "granularity": n.granularity,
            } for n in self.nodes],
            "loss": self.loss,
            "rails": list(RAILS),
            "max_policy_lag": self.lag,
            "post_pipeline": list(self.post_pipeline),
            "eval_pipeline": list(self.eval_pipeline),
        }

    def _pipeline(self, phase: str) -> tuple[str, ...]:
        if phase == "post":
            return self.post_pipeline
        if phase == "eval":
            return self.eval_pipeline
        raise ValueError(f"no pipeline phase {phase!r}")


def flow_graph(spec: ExperimentSpec) -> FlowGraph:
    """The one canonical walk. Everything else is a query on its result."""
    loss = spec.algo.loss if spec.algo is not None else None
    post_pipeline = tuple(spec.algo.post) if spec.algo is not None else ()
    # measurement left the spec (#70): the eval phase survives as a
    # dictionary dimension for PRE-#70 runs, and is empty ever after
    eval_pipeline: tuple[str, ...] = ()

    requires = frozenset(
        LOSSES.get(loss).requires
        if loss is not None and loss in LOSSES else ())
    feeding = _transitively_feeding(post_pipeline, requires)

    nodes: list[FlowNode] = []
    for phase, pipeline in (("post", post_pipeline), ("eval", eval_pipeline)):
        owner: dict[str, str] = {}
        for name in pipeline:
            if name not in POST:
                continue
            for column in POST.get(name).produces:
                if column in owner:
                    continue  # collision: first owner stands, query reports it
                owner[column] = name
                consumers = tuple(
                    later for later in pipeline
                    if later in POST and column in POST.get(later).consumes)
                if phase == "post" and column in requires:
                    consumers = consumers + (f"loss:{loss}",)
                nodes.append(FlowNode(
                    name=column, kind="column", phase=phase,
                    producer=f"postprocessor:{name}", consumers=consumers,
                    feeds_loss=(phase == "post" and column in feeding),
                    stored=True,
                    granularity=("token" if column in POST.get(name).token_level
                                 else "trajectory")))

    for record in sorted(BASE_RECORDS):
        nodes.append(FlowNode(
            name=record, kind="record", phase="wave", producer="base",
            consumers=(f"loss:{loss}",) if record in requires else (),
            feeds_loss=record in requires, stored=True, granularity="token"))
    for entry, adapter_spec in sorted(spec.policy.bank.items()):
        if adapter_spec.adapter_type not in ADAPTER_TYPES:
            continue
        adapter_type = ADAPTER_TYPES.get(adapter_spec.adapter_type).instance
        for record in adapter_type.records:
            nodes.append(FlowNode(
                name=record, kind="record", phase="wave",
                producer=f"adapter:{adapter_spec.adapter_type}",
                consumers=(f"loss:{loss}",) if record in requires else (),
                feeds_loss=record in requires, stored=True))
        for provided in sorted(adapter_type.provides):
            # TWO nodes per provide, and the second is not decoration. The
            # forward node is the tensor itself — recomputed, never stored, and
            # what a loss's requires resolves against. The STAT TWIN is the
            # per-update float the Trainer means into the ledger (TrainStats
            # .provided), which exists whether or not any loss requires the
            # tensor: an adapter type declares what a reader should WATCH, not
            # only what a loss may eat, and this is what makes the run's own
            # dictionary describe it with no special-casing anywhere (I11).
            nodes.append(FlowNode(
                name=provided, kind="provided", phase="forward",
                producer=f"adapter:{adapter_spec.adapter_type}",
                consumers=(f"loss:{loss}",) if provided in requires else (),
                feeds_loss=provided in requires, stored=False))
            nodes.append(FlowNode(
                name=provided, kind="stat", phase="train",
                producer=f"adapter:{adapter_spec.adapter_type}",
                consumers=(), feeds_loss=False, stored=True,
                granularity="update"))

    if loss is not None:
        for rail in RAILS:
            nodes.append(FlowNode(
                name=rail, kind="rail", phase="train",
                producer=f"loss:{loss}", consumers=(), feeds_loss=False,
                stored=True, granularity="update"))
        for stat in TRAIN_STATS:
            nodes.append(FlowNode(
                name=stat, kind="stat", phase="train", producer="trainer",
                consumers=(), feeds_loss=False, stored=True,
                granularity="update"))

    return FlowGraph(nodes=tuple(nodes), loss=loss,
                     lag=(spec.algo.schedule.max_policy_lag
                          if spec.algo is not None else 0),
                     post_pipeline=post_pipeline,
                     eval_pipeline=eval_pipeline)


def _transitively_feeding(pipeline: tuple[str, ...],
                          requires: frozenset[str]) -> frozenset[str]:
    """Columns reachable BACKWARD from the loss's requires through the
    pipeline's produces→consumes chains: reward feeds grpo iff something the
    loss requires is derived from it."""
    feeding = set(requires)
    changed = True
    while changed:
        changed = False
        for name in pipeline:
            if name not in POST:
                continue
            pdef = POST.get(name)
            if any(column in feeding for column in pdef.produces):
                for upstream in pdef.consumes:
                    if upstream not in feeding:
                        feeding.add(upstream)
                        changed = True
    return frozenset(feeding)
