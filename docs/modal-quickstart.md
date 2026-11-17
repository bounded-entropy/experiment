# Modal quickstart

How to stand rlstack up in a Modal workspace, run an experiment through the
desk, and watch it. Experiment code (interaction functions, graders, losses,
venue files) lives in a separate recipe package; this page covers only the
framework side. Qualification evidence is CONTEXT #103.

## 1. Name the workspace on every command

Every deploy and door refuses to run unless the command names its workspace,
and the name must match the profile the Modal SDK resolves:

```sh
export WS=<your-workspace>          # a Modal profile in ~/.modal.toml
MODAL_PROFILE=$WS modal app list
```

Never use `modal profile activate`: it changes the machine default silently.
Each workspace has its own desk, store volume (`rlstack-store`) and observer.
Nothing moves data between workspaces.

## 2. Deploy the desk and the observer (once per workspace)

```sh
MODAL_PROFILE=$WS RLSTACK_BOOTABLE_METALS=<metal>,<metal> modal deploy deploy/desk.py
MODAL_PROFILE=$WS RLSTACK_OBSERVER_APP=rlstack-observer modal deploy deploy/ui.py
```

- `RLSTACK_BOOTABLE_METALS` lists the metal names (from venue files) that the
  desk may boot. To add a metal, redeploy with the full new list.
- `RLSTACK_OBSERVER_APP` sets the observer's app name and URL label, which
  defaults to `rlstack-ui`. Use another name if that app already exists. The
  URL is `https://<workspace>--<app>.modal.run`; with no auth on it, treat
  it as unlisted, not private.
- The desk runs as a standing CPU container. GPU metal releases itself after
  its finite idle limit; keep that limit enabled.

## 3. A venue file and a submission

A venue file in the recipe package declares one metal: the image
(`gpu_image_for(sources=(...))`, which carries vLLM 0.28, torch 2.13,
transformers 5.16 and flash-linear-attention), the GPU string (for example
`"RTX-PRO-6000"`), the builds it serves, and doors for `up`, `smoke` and
`submit`. Import the chassis as top-level `modal_venue`, which is how it ships
into containers. Submit through the desk with an explicit subdirectory and a
checkpoint cadence:

```python
from rlstack.runner.checkpointing import Checkpointing
checkpointing = Checkpointing(every=5, delivery="wire")
```

Then confirm, in this order: the submission was accepted, the run id and
`run_ref` came back, the desk placed it, the host journaled an attach, the
ledger has committed updates, and the observer lists the run.

## 4. Operate

```sh
MODAL_PROFILE=$WS modal run deploy/desk.py::status                  # listings, metals, runs
MODAL_PROFILE=$WS modal run deploy/desk.py::stop --run <subdir>/<id>  # drain, checkpoint, stop
MODAL_PROFILE=$WS modal run deploy/desk.py::dispositions            # parked / stopped / failed
MODAL_PROFILE=$WS modal run deploy/desk.py::sweep                   # release every metal (guarded)
```

To resume, resubmit the same spec with `resume=True`. The desk re-adopts the
run id and continues from the last checkpoint. Never cancel a `modal run`
driver mid-call: the cancellation reaches the metal container and kills it.

## What is qualified (CONTEXT #103)

| Shape | Status |
| --- | --- |
| Online GRPO through the desk, one card, two tenants (Qwen3-0.6B) | qualified: commit, checkpoint, stop, resume (byte-identical ledger), observer |
| Qwen3.5-4B LoRA r16, punica serving + torch learner, one RTX PRO 6000 | qualified for 4 updates: `logprob_gap` 0.010–0.014, `mean_ratio` ≈ 1.000 |
| Multi-GPU metal (TP / FSDP / dp), NCCL weight transfer, sleep mode, crash recovery, `prompt_logprobs` scoring | not yet qualified |

One known quirk: per-process GPU memory bookings are empty on Modal, because
`nvidia-smi` sees no processes in the sandbox (card-level stats work).

Placement keeps a run on one metal where it can. Every unit after a run's
first prefers the metal the first landed on, joining a listing there or carving
there. It moves to another metal only when that one has no room, and the run is
then reached through the desk's relay (CONTEXT #104).
