# The framework and experiment boundary

rlstack supplies generation/scoring services, adapter execution and routing,
sealed records, normal training, checkpoints, storage and observation. An
experiment owns its prompts, task construction, interactions, tools, semantic
retries, rewards and objective composition. Experiment code lives in a separate
Python package and repository. There is no required experiment base class,
Environment class, TaskMaker class or registration for interaction functions.

The public recipe ports live in the separate `rlstack-recipes` repository,
currently checked out beside the framework at
`/Users/samarth/Coding/rlstack-recipes`. Its `rlstack_recipes` package groups
behavior by experiment: concept steering, math, distillation, Dreamer, factual
adapters and evaluation. Private teaching recipes remain in the private
Dreamer repository as `dreamer_recipes`. That repository imports a pinned shared
framework; it does not maintain another active `rlstack` package.

## Plain interaction functions

```python
# my_experiment/interactions.py
from rlstack import Message, Role, Rollout

async def answer(client, task):
    messages = [Message(Role.USER, task.prompt)]
    turn = await client.sample(messages)
    return Rollout(task, [*messages, turn.message], [turn])

async def reflect(client, source):
    next_task = my_next_task(source)
    return await answer(client, next_task)
```

The author owns `my_next_task` and the interaction. The framework invokes the
function, checks/seals the returned Rollout, and retains deterministic request
seeds. Per-call sampling overrides are explicit:
`client.sample(messages, sampling=SamplingSpec(temperature=0, max_tokens=64))`.
The effective sampling parameters, stop strings and request seed are recorded
with each generated turn and in bounded call telemetry.

```python
spec = ExperimentSpec(
    ...,
    code=("my_experiment",),
    gen=GenSpec(functions=("my_experiment.interactions:answer",
                           "my_experiment.interactions:reflect"),
                tasks=(tasks_uri,)),
)
Sample("task-17", "my_experiment.interactions:answer")
Continue("self://rollouts/1#0", "my_experiment.interactions:reflect")
```

A Sample passes its Task to the function. A Continue passes an already sealed
Trajectory, so deriving the next task stays ordinary experiment Python. Continue
retains the durable source reference and may read only an earlier self rollout.
A resumed run reuses sealed waves and invokes only unfinished work. Replay and
WaveRef retain their existing meanings.

Every function in the actual generation plan must be declared in
`GenSpec.functions` and covered by `ExperimentSpec.code`. An imported function
that was omitted from either declaration is refused before generation or learner
installation. A plan can declare `dependencies=("adapter/name",)` on a Sample or
Continue when the interaction reads named frozen state. The generator waits
before holding GPU admission. It never infers those names from task metadata.
Adapter-specific interpretation of trained row routes belongs to that adapter's
`named_dependencies` / `named_state_route` methods, not to the Trainer.

## Explicit imports and source identity

`ExperimentSpec.code` names module or package roots installed on every worker
that needs them. Importing those roots registers their useful loss,
postprocessor and adapter definitions through the existing contracts. The
framework never imports a particular experiment package on its own. External
registered definitions and every explicitly selected adapter provider must be
covered by the declaration, even if a previous run imported them already.
Use experiment-prefixed registry names for new definitions when unrelated
experiments might choose the same short name: the existing registry still
refuses two different bodies under one name. Existing recipe names are retained
where they are already distinct.

The source manifest hashes every Python file beneath a declared package,
including helpers and constants, plus parent package initializers. Declare
additional first-party helper packages explicitly when they live outside that
tree; the loader does not infer a transitive import graph. Declaring a narrower experiment subpackage
avoids tying unrelated experiments together. The host compares complete claimed
source identities. Learner installations, serving bundles, raw tensor
publication and custom site workers carry/verify the external source manifest.
The manifest also enters the bundle identity, so an inference cache cannot
confuse identical weights executed by different experiment code.

Installed source is immutable for a resident's lifetime. If local files change
after import, start a fresh process: rehashing files cannot update Python's
already imported functions. The loader refuses that mismatch. This is explicit
source verification, not automatic source upload or hot reloading.

Install/pin the same framework and experiment packages on all workers. Generic
Modal image helpers accept `sources=("my_experiment",)`; the Strange Loop source
bundle accepts explicit top-level `source_packages`. A missing package or source
mismatch fails rather than borrowing ambient registrations. The recipes retain
thin venue wiring and submit through the normal desk.

## What remains shared

Shared loss code retains token NLL, sequence-mean NLL, independently normalized
SFT, PPO/GRPO/GSPO, generic teacher reverse-KL and recorded-behavior distillation.
Shared processors retain generic column arithmetic, group normalization and
policy/teacher token scoring. Low-rank, steering, prompt, spectral and custom
site execution math remains shared. These are reusable mathematical/execution
operations; prompts, answer parsing, correctness checks, judges, fixed research
coefficient choices and semantic combinations live with the recipes that choose
them. Moving a composition calls the existing mathematical primitive rather than
copying its subtle masking or normalization.

DreamBank's dreamer/memory/library composition and task-indexed factual adapter
variants are external recipe adapters. They still use the same state,
checkpoint, routing and weight-publication services as all other adapters.
Their native providers are declared by full module path. No adapter-specific
transport or fitting service is reintroduced.

## Migration and existing runs

Migrated function plans have new identities. Old environment/maker plan rows
round-trip through read-only historical records; executing them is refused with
a migration error. Historical scientific records and artifacts are untouched.
Use the original pinned code to inspect or deliberately resume an old run; a
new function plan does not silently reinterpret it.

The validated core before this boundary migration remains available at
`b01102adf95c97bee6d7b31856c197836ae7fc75`. Hardware results obtained on that
commit qualify that exact core: short ordinary SFT and tensor transport. They do
not qualify the complete external recipe migration, online RL, live vLLM model
serving or custom Triton parity. Final migration inventory and local validation
are recorded alongside ADR 0030.
