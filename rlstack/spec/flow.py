"""The flow graph: THE canonical walk over a spec's data declarations.

One constructor — flow_graph(spec) — builds the graph of every named data
artifact a run will contain: pipeline columns (postdata/, eval/), recorded
facts (waves/, I6), forward-recomputed tensors, and the standing rails
(ledger train summary). Edges are the declarations verbatim: PostDef
produces/consumes, LossDef requires, adapter records/provides.

Everything that reasons over these declarations queries THIS object, by
design: the submit gate's pipeline checks (validate.py) and the run's
self-description (dictionary.json, written at run creation, what a UI
renders) are two consumers of one walk — the graph cannot drift from the
semantics without validation failing with it. A new declaration kind gets a
node or edge here ONCE and every consumer sees it.

Unknown registered names contribute nothing (check_names_are_registered
reports them); planned-pass requirements (Ref/Teacher/Probe) are the
runner's to satisfy and are not graph nodes in v0.
"""

from __future__ import annotations

from dataclasses import dataclass

from rlstack.registry import ADAPTERS, LOSSES, POST
from rlstack.spec.specs import ExperimentSpec

# PolicyOutputs fields the training forward can always produce, with no bank help.
BASE_PROVIDES = frozenset({"ref_logprobs", "entropies", "hidden_states"})

# Rollout facts every trajectory records regardless of the bank (I6).
BASE_RECORDS = frozenset({"behavior_logprobs", "finish"})

# What every loss reports into the ledger, regardless of family.
RAILS = ("loss", "mean_ratio", "logprob_gap", "grad_norm")


@dataclass(frozen=True)
class FlowNode:
    """One named data artifact of a run.

    kind: "column" (a post pipeline output) | "record" (a sampling-time
    fact, frozen at the seal) | "provided" (a training-forward tensor,
    recomputed each pass) | "rail" (a loss's standing report).
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


@dataclass(frozen=True)
class FlowGraph:
    """The spec's data artifacts plus the queries validate and the
    dictionary are built from. Node order is pipeline order."""

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
                     if isinstance(req, str) and req not in available)

    # ---- the run's self-description -----------------------------------------

    def to_json(self) -> dict:
        """dictionary.json: what this run's store will contain and why —
        a UI walks this instead of re-deriving any declaration."""
        return {
            "columns": [{
                "name": n.name, "kind": n.kind, "phase": n.phase,
                "producer": n.producer, "consumers": list(n.consumers),
                "feeds_loss": n.feeds_loss, "stored": n.stored,
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
    eval_pipeline = tuple(spec.eval.post) if spec.eval is not None else ()

    requires = frozenset(
        req for req in (LOSSES.get(loss).requires
                        if loss is not None and loss in LOSSES else ())
        if isinstance(req, str))
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
                    stored=True))

    for record in sorted(BASE_RECORDS):
        nodes.append(FlowNode(
            name=record, kind="record", phase="wave", producer="base",
            consumers=(f"loss:{loss}",) if record in requires else (),
            feeds_loss=record in requires, stored=True))
    for provided in sorted(BASE_PROVIDES):
        nodes.append(FlowNode(
            name=provided, kind="provided", phase="forward", producer="base",
            consumers=(f"loss:{loss}",) if provided in requires else (),
            feeds_loss=provided in requires, stored=False))

    for entry, adapter_spec in sorted(spec.policy.bank.items()):
        if adapter_spec.kind not in ADAPTERS:
            continue
        kind = ADAPTERS.get(adapter_spec.kind).instance
        for record in kind.records:
            nodes.append(FlowNode(
                name=record, kind="record", phase="wave",
                producer=f"adapter:{adapter_spec.kind}",
                consumers=(f"loss:{loss}",) if record in requires else (),
                feeds_loss=record in requires, stored=True))
        for provided in sorted(kind.provides):
            nodes.append(FlowNode(
                name=provided, kind="provided", phase="forward",
                producer=f"adapter:{adapter_spec.kind}",
                consumers=(f"loss:{loss}",) if provided in requires else (),
                feeds_loss=provided in requires, stored=False))

    if loss is not None:
        for rail in RAILS:
            nodes.append(FlowNode(
                name=rail, kind="rail", phase="train",
                producer=f"loss:{loss}", consumers=(), feeds_loss=False,
                stored=True))

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
