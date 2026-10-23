# ADR 0030 — Experiments own behavior outside the framework

| | |
|---|---|
| **Date** | 2026-10-23 |
| **Status** | Accepted |
| **Author** | Codex integration subagent |
| **Touches** | `spec/`, `registry.py`, `data/plan.py`, `inference/`, `training/`, `policy/adapters/`, `runner/`, worker code loading, external recipe repositories |
| **Invariants** | I1, I2, I3, I4, I5, I8, I9, I10, I11 |
| **CONTEXT** | Extends ADRs 0028 and 0029; supersedes mandatory environment/task-maker declarations |

## Original prompt

> im thinking of the fact that it's kind of inconvenient to have multiple experiments in the same codebase/version of rlstack. the reason for this is because there are probably different rewards/post processors or whatever implemented for each experiment which are unique to each experiment, but implemented in the general library that everyone can access. the actual way this should be done is that the rewards, post processors, etc. should be probably all declared in one experiment folder per experiment, but obv they should import from the class. im just saying that they should be declared that deep into rlstack. does that make sense? i think that the disadvantages arising from having to reimplement specific loss functions instead of using them again is less than the advantage of having this cleaner code. or can you think of a good compromise?

> yea this sorta reflects the problems i was talking about. the main rlstack repo should just consist of the general frameworks, and all the experiment-specific code should be somewhere entirely different. could you fix this and port the existing experiments?

> this includes experiments and makers and stuff like that. actually, before you do that, could you give me an example of a maker

> do we need to abstract that? i feel like that should be smth that's the burden of the person writing the experiment, so i feel like that shouldnt live in rlstack (and i dont even think we need to write a general class to have this porting over functionality yk). same thing for environments

> ok can you do all of this?

## Context / problem

The shared package eagerly imports ten registered Environment classes and a registered Reflect TaskMaker. Experiment-specific rewards, prompt formats, losses and task builders are mixed into the same responsibility folders as general primitives. DreamBank's default `dreamer`/`memory` roles are a recipe-specific adapter composition. The private repository adds more definitions inside a vendored framework subtree. Importing rlstack therefore exposes behavior that belongs to unrelated experiments. The existing class source hash also misses imported prompt constants/helpers, and generation currently looks up undeclared names through ambient registries. Throughput effects of this separation are unmeasured; this decision concerns ownership, explicit execution and source identity.

## Decision

Keep general services, typed records and reusable numerical primitives in rlstack. Move experiment implementations, interaction functions, prompt/task builders, reward/loss compositions and launch recipes to separate versioned experiment locations. Public recipes are ported to the separate `rlstack-recipes` repository/package; private teaching code remains in its existing private repository, outside its framework namespace, with an exact framework pin. Do not duplicate tested numerical primitives merely to move composition. Do not introduce an experiment base class, a new plugin hierarchy or another workflow runner.

### Touched / untouched

- **Touched** — `inference/environments/`, `inference/makers/`, registry/spec declarations and eager imports: remove mandatory Environment and TaskMaker classes and their registries.
- **Touched** — `data/plan.py`, assembly, validation and wire codecs: Sample names a plain function and a Task; Continue names a plain function and an already sealed source trajectory. The function owns task derivation and interaction together. Replay and WaveRef retain their existing meaning.
- **Touched** — explicit code loading: an experiment declares external module/package roots; their complete Python source trees enter identity and are verified wherever that code executes. Learner parameterizations and serving bundles carry these declarations. Long-lived workers cannot satisfy a missing declaration merely because a prior run happened to import that experiment.
- **Touched** — experiment implementations/tests/launchers: move to external versioned sources, rewrite interactions as ordinary async functions, migrate active callers and pin the common framework. Produce a path-by-path disposition inventory.
- **Touched** — reusable math extracted from mixed files: sequence-mean SFT, routing/state/linear algebra and standard objectives remain shared; experiment-specific coefficient choices, semantic columns, prompts and scoring rules move.
- **Untouched** — normal Trainer, independent Adam, sealed records, checkpoint cadence, raw tensor NCCL/base64 and store commit protocols from ADR 0029: the boundary move does not add execution paths.
- **Untouched** — historical scientific records and source branches: old plan records remain readable, but old environment/maker plans are not silently executed under new semantics. Migrated function plans have new identities.
- **Untouched** — generic infrastructure qualification/deployment examples that only demonstrate services: they may remain, using test/demo-owned functions rather than research behavior.

### Promises / non-promises

- **Promises** — importing rlstack imports no particular experiment package; removing the companion package does not break framework imports or its own test suite.
- **Promises** — every generation function used by a plan is explicitly declared and has source identity; undeclared functions, legacy environment/maker execution and mismatched worker code fail before work starts.
- **Promises** — derived work keeps an explicit sealed-source dependency and deterministic per-leaf seed; crash recovery reuses sealed waves and invokes only unsealed work. Framework sealing and validation remain authoritative.
- **Promises** — every moved active recipe has versioned source and migrated imports/entrypoints; scientific artifacts remain intact and private material is not published to the public framework.
- **Non-promises** — migrated plans retain old run IDs. They deliberately change vocabulary/code ownership and receive new identities. Historical read compatibility does not imply historical executor compatibility.
- **Non-promises** — automatic source upload or dependency installation on workers. Install the declared experiment package and pin it in the image; a code mismatch is a deployment error, not permission to use ambient definitions.
- **Non-promises** — numerical/GPU parity or speed beyond the tests and separate hardware evidence actually recorded.

### Interfaces

`GenSpec.functions` lists qualified `module:symbol` references. `Sample(task_id, function)` invokes `await function(client, task)`; `Continue(source, function)` invokes `await function(client, sealed_source)`. Both return the existing Rollout; the shared runner seals it. Experiment functions may construct Tasks, call tools, choose pools, retry semantically and choose stop conditions. The framework owns pool services, deterministic request seeds, accurate recording and durable dependencies. No registered maker or environment object is involved.

`ExperimentSpec.code` declares external import roots. Package-source hashes include helper modules and constants, and travel with compiled learner/serving state. Existing PostProcessor/Loss/AdapterType interfaces remain useful contracts; their experiment implementations are loaded only from declared code roots, and references must be covered by those declarations.

```python
GenSpec(functions=("my_experiment.interaction:answer",), tasks=(tasks_uri,))
Sample("task-17", "my_experiment.interaction:answer")
Continue("self://rollouts/1#0", "my_experiment.interaction:reflect")
ExperimentSpec(..., code=("my_experiment",))

async def reflect(client, source):
    task = make_my_next_task(source)
    return await my_interaction(client, task)
```

## Questions and authorization

**Q1. Move only implementations, or also remove the Environment/TaskMaker hierarchy?** The user explicitly rejected needing a general class for either and then authorized “all of this.” Remove both mandatory hierarchies; retain only concrete function invocation and dependency records. This is the user's settled answer, not an inferred approval for a new abstraction.

**Q2. Keep recipes in an experiments folder inside the framework?** The user explicitly requested “somewhere entirely different.” Use separate versioned repositories/packages; preserve private ownership. Creating a new public remote or publishing private artifacts is unnecessary and is not part of this decision.

**Q3. Duplicate numerical losses or retain shared mathematical primitives?** The accepted discussion recommends reusing subtle tested masking/normalization/numerical components while keeping experimental choices local. Keep generic primitives and useful loss/post contracts; move compositions and semantic policies. Each retained implementation must have a semantic rationale in the inventory.

**Q4. What preserves recovery and code identity when classes disappear?** Plain function references plus sealed-source dependencies and explicit module source declarations preserve the concrete existing requirements without moving the old hierarchy elsewhere. New plan vocabulary changes run IDs; old records remain read-only. A killed unsealed episode is retried with its deterministic seed; a sealed wave is reused. Experiment authors remain responsible for making external side effects idempotent. No new scheduling daemon or source transport is introduced.

These answers record the user's explicit direction and its ordinary implementation consequences. The already authorized rewrite continues without asking the user to approve the same boundary again.

## Outcome

Implementation pending on a branch stacked after the validated core milestone `b01102adf95c97bee6d7b31856c197836ae7fc75`. Core experiment launches can pin that milestone while the broader migration is validated independently. Final inventories, exact repository refs, local checks and separate hardware limits will be recorded here.
