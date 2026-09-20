# rlstack

rlstack is a research harness for training and evaluating language-model adapters.
It connects generation, scoring, learning, checkpointing, and GPU scheduling
through shared interfaces. The same run specification can use local test doubles
or real PyTorch learners and vLLM inference workers.

This repository contains the reusable stack. Experiment-specific datasets,
dreamer/SQuAD/SEAL recipe builders, launch configurations, and research results
are maintained separately.

## The basic pieces

| Piece | What it does |
| --- | --- |
| **Spec** | Declares the base model, adapters, generation settings, loss, plans, seeds, and host topology. |
| **Plan** | Says which tasks to sample or replay at each step, or which supervised fit jobs to run. |
| **Environment** | Turns a task into a rollout, recording the generated tokens and their sampling probabilities. |
| **Postprocessor** | Scores completed trajectories and produces columns such as rewards or advantages. |
| **Loss** | Computes the training objective from recorded data and the learner's model outputs. |
| **Engine / learner** | Serve inference and perform training behind common protocols. |
| **Desk / host** | Place runs on available GPU resources and supervise their workers. |
| **Store** | Keeps input data, rollouts, metrics, named adapters, and checkpoints. |

A typical RL update samples a wave of trajectories, scores them, trains on the
recorded data, and commits the resulting policy and metrics. Generation-only
runs and supervised fit runs use the same storage and scheduling infrastructure.

Run identity is derived from the spec and the code and content it names.
Operational settings such as checkpoint cadence are declared separately. Run
directories are explicit: a reference such as `family/run_id` names exactly that
run, without searching other folders.

## Capabilities

- Adapter implementations include LoRA, learned task bases, spectral adapters,
  steering, soft prompts, and a routed adapter bank (`dream_bank`).
- Named adapters can be saved, reused by other runs, and stacked with a trainable
  adapter. Fit jobs can share a learner while keeping separate adapter state,
  optimizer schedules, and document normalization.
- Postprocessors can use the fit/fork API to evaluate the effect of training a
  copy of an adapter. The API does not prescribe an experiment's reward.
- Checkpoints support resume and configurable durability cadence. Deliberate
  stops drain and checkpoint before being recorded as stopped.
- Local, Modal Volume, and Strange Loop scratch stores share one interface.
  Content-addressed objects and named adapter payloads are hash-verified on read;
  Strange Loop can additionally use a bounded pod-local cache and visible
  mounted bytes, with an authoritative API fallback.
- HTTP services support separate inference and training workers, including
  temporary blob transfer for large values. Modal and Strange Loop use shared
  desk and worker lifecycle code.
- The read-only observer displays saved run metrics and host activity. A separate
  W&B exporter reads committed training metrics or sealed generation waves.
  Fit traces retain document loss components, probe results, and phase timings.

## Get started locally

Python 3.11 or newer is required; development checks use Python 3.13.

```sh
git clone https://github.com/bounded-entropy/experiment.git rlstack
cd rlstack
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests
```

The test suite exercises the shared interfaces with deterministic fake engines
and learners. Tests requiring unavailable PyTorch, Transformers, CUDA, or live
provider access are skipped. Passing local tests does not establish GPU or
cross-pod behavior.

For small executable examples, start with [spec construction](tests/test_examples.py),
[generation and training](tests/test_loop.py), [fit jobs](tests/test_fit.py), and
[checkpoint/resume behavior](tests/test_resume.py). These fixtures run without
renting GPUs. Real model workers need the compatible dependencies and cached
weights described by their deployment configuration.

Inspect an existing local store with:

```sh
python -m rlstack --help
python -m rlstack ui /absolute/path/to/store
```

## Run on GPUs

Submit experiments through the standing desk. A submission declares its exact
subdirectory and `Checkpointing(every=..., delivery=...)`; there is no implicit
checkpoint cadence. `delivery="wire"` sends policy updates to serving pools;
`delivery="store"` requires a checkpoint at every update.

- **Modal:** [desk deployment](deploy/desk.py) and [venue wiring](deploy/modal_venue.py).
- **Strange Loop:** [setup and operations](examples/strangeloop-desk.md),
  [local desk configuration](examples/strangeloop-config.json), and
  [hosted desk configuration](examples/strangeloop-hosted.json).
- **HTTP services:** [transport guide](examples/http-services.md).

Set credentials and provider-specific paths in your own configuration, choose a
GPU ceiling, and keep finite idle release enabled. The deployment files build
images and wire shared APIs; experiment logic belongs in the caller's specs,
plans, environments, and postprocessors.

## Repository map

| Directory | Responsibility |
| --- | --- |
| `rlstack/spec/` | Specs, validation, identity, and execution flow. |
| `rlstack/data/` | Trajectories, packing, task files, and storage backends. |
| `rlstack/policy/` | Adapter definitions and their training/inference implementations. |
| `rlstack/inference/` | Rollouts and environments. |
| `rlstack/training/` | Losses and postprocessors. |
| `rlstack/runner/` | Run roles, fit APIs, engines, learners, scheduling, transports, and exporters. |
| `rlstack/observe/` | Read-only metrics and the browser UI. |
| `rlstack_engine/` | Plugins and integration code for inference workers. |
| `deploy/`, `examples/` | Deployment wiring and configuration examples. |
| `tests/` | Interface, regression, resume, and optional hardware checks. |
| `knowledge/` | Architecture decision records, including validation limits. |

Before changing the stack, read [AGENTS.md](AGENTS.md), [STYLE.md](STYLE.md),
and [ARCHITECTURE.md](ARCHITECTURE.md). The [decision log](agent-context/CONTEXT.md)
records implementation history; some historical entries describe experiments
whose recipe code is intentionally outside this repository.
