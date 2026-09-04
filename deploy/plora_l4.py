"""plora on ONE L4: test-time training on a single DAPO problem, three hosts
deep on one device.

    modal run deploy/plora_l4.py::factors          # build + cas_put the frozen half
    modal run deploy/plora_l4.py::shakeout         # 2 updates end to end
    modal run deploy/plora_l4.py::full --go        # the 30-update campaign

TWO THINGS THIS FILE EXISTS TO EXERCISE. First plora itself: a rank-16 delta
whose k x k cores are generated from a 32-wide latent, served as an ensemble of
8 drawn members plus the posterior mean, with the noise recorded at rollout and
the reparameterized gradient reaching mu and log_std at replay. Second the
sub-GPU fleet (#51/#52): three hosts on FRACTIONAL partitions of device 0 —
sampling 0.30, eval 0.20, training 0.40 — each with its own arbiter, admitting
its own traffic, reached over the wire even though all three share a process.
vLLM budgets against the device total, so fractional partitions compose
additively; the wire is LocalTransport, which json-round-trips both ways, so
nothing above `submit` can tell these hosts apart from three containers.

The frozen half is a SEPARATE ARTIFACT, and the two entry points below are why:
factoring 0.6B's attention stack is one job that produces bytes, and the
campaign then names those bytes by content. Paste the printed uri into FACTORS.

Everything semantics-bearing is in the spec — the plans, the bank, the loss —
and everything else here is venue (I5).
"""

import modal

app = modal.App("rlstack-plora-l4")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)
hf_cache = modal.Volume.from_name("rlstack-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
          "OMP_NUM_THREADS": "1",
          "HF_HOME": "/hf",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("rlstack", "rlstack_engine")
)

BASE = "Qwen/Qwen3-0.6B"
STORE = "modal://rlstack-store"
# the task sets built by deploy/tasks_dapo.py (#60), reused verbatim
TRAIN_TASKS = "cas://09499d32b51e5e1b2a644b1c65e01b44aa42ff1a5bfac78ead41f98f89f09c93"
EVAL_TASKS = "cas://82ae4626dbb59a2c50e2b13cbe7250c5f1ddd02dfb81edc7495efb77759d420b"

SITE = "layers.*.self_attn.*"
K = 16                  # the frozen rank: how many singular directions steer
LATENT = 32             # width of z
MEMBERS = 8             # drawn adapters one version is served as
PRIOR_STD = 0.05
HIDDEN = 128

# The frozen half, by content. `factors` below prints this; paste it here.
FACTORS = "cas://e91b9d39b5144b03092cf31fffc2f6121ed60deb761bc8bdc8cda709944ea26d"

# THE one problem this run trains on — screened, not arbitrary: the base
# passes it 4/8 at temperature 1.0 (96 tasks x 8 samples; 87 were all-wrong,
# none all-right), so its groups start with maximal advantage variance instead
# of the all-or-nothing zero the first draw of the task file gave (#60's
# blocker, met again at 0.6B scale).
TRAIN_TASK_ID = "dapo-math-17k/a6d38312-86c7-4022-b8d2-adcf19fa0c3a"

GROUPS_PER_WAVE = 4     # four independent baselines over the SAME problem
GROUP_SIZE = 8          # completions per baseline
MAX_TOKENS = 1024
EVAL_EVERY = 5

# The partition treaty on one L4 (24 GiB), in GB (ADR 0001): sampling, eval,
# training. They sum under the card because vLLM budgets its reservation
# against the DEVICE total, so partitions of one device compose additively
# (#51); the fractions the substrates take are derived against the MEASURED
# card at bring-up (fraction_for_gb, the one crossing).
MAIN_GB = 7.2
EVAL_GB = 4.8
LEARNER_GB = 9.6


# ---------------------------------------------------------------------------
# the frozen half: one job, one artifact
# ---------------------------------------------------------------------------

@app.function(image=image, gpu="L4", volumes={"/store": store_volume, "/hf": hf_cache},
              timeout=3600)
def factors(k: int = K) -> str:
    """Factor every matched site once and put the bytes in the CAS.

    No model is built: the reader walks the checkpoint's safetensors shards and
    slices out one weight at a time, so this costs one matrix of memory rather
    than a whole base. The artifact stamps (algo, base, k), which is what lets
    the engine refuse a mismatched one at read rather than serving the wrong
    coordinate system silently.
    """
    from rlstack import ModalVolumeStore
    from rlstack.policy.adapters.plora_factors import build_factors, hf_weight_reader
    from rlstack.policy.siteschema import hf_schema, resolve

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    sites = resolve(hf_schema(BASE).sites, SITE)
    payload = build_factors(BASE, sites, k, hf_weight_reader(BASE))
    uri = store.cas_put(payload)
    store_volume.commit()
    print(f"[factors] {len(sites)} sites, k={k}, {len(payload) / 2**20:.1f} MiB")
    print(f"[factors] FACTORS = {uri!r}")
    return uri


# ---------------------------------------------------------------------------
# the science: three plans and one spec
# ---------------------------------------------------------------------------

def rollout_plan(task_id, updates):
    """Every wave is the SAME problem, four times over.

    Test-time training: a group is the advantage's baseline scope, not a task,
    so four groups of eight completions of one problem are four independent
    baselines over one question — which is what makes an all-or-nothing group
    less likely to zero the whole update than one group of thirty-two would.
    """
    from rlstack import GroupPlan, RunPlan, Sample, WavePlan
    wave = WavePlan(tuple(
        GroupPlan(f"{task_id}#{group}",
                  tuple(Sample(task_id, "dapo_math") for _ in range(GROUP_SIZE)))
        for group in range(GROUPS_PER_WAVE)))
    return RunPlan((wave,) * updates)


def train_plan(updates):
    """Update u trains on rollout u, whole: plain on-policy GRPO."""
    from rlstack import RunPlan, WaveRef
    return RunPlan(tuple(WaveRef(f"self://rollouts/{u}")
                         for u in range(1, updates + 1)))


def spec_for(store, task_id, held_out, updates, master):
    """The experiment as one value."""
    from rlstack import (AlgoSpec, ExperimentSpec, GenSpec, Topology,
                         HostSpec, LearnerMember, OptimSpec, Plans,
                         PolicySpec, PoolMember, SamplingSpec, Schedule, Seeds,
                         encode, plora)

    plans = Plans(
        train=store.cas_put(encode(train_plan(updates))),
        rollout=store.cas_put(encode(rollout_plan(task_id, updates))))
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank={"pi": plora(
            SITE, k=K, latent=LATENT, members=MEMBERS, prior_std=PRIOR_STD,
            hidden=HIDDEN, factors=FACTORS)}),
        gen=GenSpec(envs=("dapo_math",), tasks=(TRAIN_TASKS, EVAL_TASKS),
                    sampling=SamplingSpec(temperature=1.0, max_tokens=MAX_TOKENS)),
        plans=plans,
        algo=AlgoSpec(
            # the ungated latent KL: the prior's pull from the first update
            # (the accuracy-gated variant lives with the campaign that uses it)
            loss="grpo_latent_kl",
            post=("final_answer", "grpo_advantage"),
            # THE OVERRIDE IS THE POINT of the dotted grammar: the hypernet is
            # an ordinary network and may be decayed; the posterior is a
            # distribution's parameters, where decay would be a second,
            # unstated prior pulling log_std toward a scale of 1.
            optim=OptimSpec("adamw", lr=3e-4, weight_decay=0.0,
                            overrides={"pi.mapper": {"weight_decay": 1e-2}}),
            # 512, not 2048: the learner's logprob pass holds full-vocab fp32
            # logits, so a microbatch's memory is width x 151936 floats and the
            # 0.40 L4 partition OOMs in the backward at two ~1300-token
            # documents per forward (measured: a 4.45 GiB allocation against
            # 4.25 free, five updates in). One document per forward fits; #58's
            # rule that the knob must bite still holds — completions here are
            # up to ~1300 tokens.
            schedule=Schedule(microbatch_tokens=512, max_policy_lag=1)),
        topology=Topology(hosts=(
            HostSpec((PoolMember("main", tp=1, vram_gb=MAIN_GB),)),
            HostSpec((PoolMember("eval", tp=1, vram_gb=EVAL_GB),)),
            HostSpec((LearnerMember(fsdp=1, vram_gb=LEARNER_GB),)))),
        seeds=Seeds(master=master))


# ---------------------------------------------------------------------------
# the venue: three hosts, one device, one process
# ---------------------------------------------------------------------------

def serving_host(name: str, store, gb: float, metal, cas_get):
    """One sampling partition: its own engine, its own arbiter, its own
    journal, born onto its share of device 0 — `gb` of the MEASURED card,
    converted once here (fraction_for_gb).

    `max_members` is what plora spends: a resident bundle is MEMBERS + 1 punica
    adapters, so the slot budget is counted in members here and the lowering's
    demands() multiply it out. `cas_get` is the other build fact plora needs —
    the frozen factors travel by address, and a build with no way to resolve one
    refuses the adapter type at construction.
    """
    from rlstack.runner.desk import fraction_for_gb
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack.runner.host import Host, Partition, Regime

    fraction = fraction_for_gb(gb, metal)
    engine = VllmEngine(BASE, tp=1, gpu_memory_utilization=fraction,
                        max_model_len=2048, max_bundles=2, max_rank=K,
                        max_members=MEMBERS, cas_get=cas_get,
                        serves=("plora",))
    return Host(name, engines=(engine,), learner=None, store=store,
                partition=Partition(metal.name, (0,), fraction, metal.gpu),
                regimes=(Regime(f"serve-{name}", "inference", BASE, 1),))


def _run(task_id: str, held_out: list[str], updates: int, master: int,
         label: str) -> dict:
    """Drive the experiment from the learner's host — the default anchor —
    and reach the two pools over the wire, because they are other partitions
    even though they are in this process."""
    import asyncio

    from rlstack import ModalVolumeStore
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.desk import MetalService, fraction_for_gb
    from rlstack.runner.host import Host, Partition, Regime
    from rlstack.runner.learners.torch_learner import TorchLearner
    from rlstack.runner.remote import HostService, LocalTransport, RemotePool

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    spec = spec_for(store, task_id, held_out, updates, master)
    metal = MetalService.measure("modal-l4")        # the card, read, not typed

    pools = {}
    for name, gb in (("main", MAIN_GB), ("eval", EVAL_GB)):
        host = serving_host(f"plora-{name}", store, gb, metal, store.cas_get)
        pools[name] = RemotePool(LocalTransport(HostService(host)),
                                 base=BASE, tp=1)

    learner = TorchLearner()
    host = Host("plora-learner", engines=(), learner=learner, store=store,
                partition=Partition(metal.name, (0,),
                                    fraction_for_gb(LEARNER_GB, metal), metal.gpu),
                regimes=(Regime("train-fsdp1", "training", BASE, 1),),
                # one problem, one policy, one tenant: this partition's purpose
                # is this experiment, and a second tenant would only take
                # throughput from it
                solo=True)
    report = asyncio.run(host.submit(spec, hf_schema(BASE), store,
                                     remotes=pools))
    store_volume.commit()
    print(f"[{label}] run_id={report.run_id} "
          f"{report.extent}={report.completed}")
    return {"run_id": report.run_id, "completed": report.completed}


def _tasks(store, n_held_out: int = 4):
    """The one problem this run trains on (TRAIN_TASK_ID, screened above), and
    the control set."""
    from rlstack.data.tasks import load_tasks

    if not any(t.id == TRAIN_TASK_ID for t in load_tasks(store, TRAIN_TASKS)):
        raise ValueError(f"{TRAIN_TASK_ID!r} is not in {TRAIN_TASKS}")
    held_out = [t.id for t in load_tasks(store, EVAL_TASKS)][:n_held_out]
    return TRAIN_TASK_ID, held_out


@app.function(image=image, gpu="L4", volumes={"/store": store_volume, "/hf": hf_cache},
              timeout=7200)
def shakeout(master: int = 11) -> dict:
    """Two updates end to end: three partitions on one device, an ensemble
    served, a latent recorded, one real gradient through mu and log_std."""
    from rlstack import ModalVolumeStore

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    train_id, held_out = _tasks(store)
    return _run(train_id, held_out, updates=2, master=master, label="shakeout")


@app.function(image=image, gpu="L4", volumes={"/store": store_volume, "/hf": hf_cache},
              timeout=86400)
def full(master: int = 11, updates: int = 30, go: bool = False) -> dict:
    """The campaign. Gated: real money, and the shakeout's numbers decide."""
    from rlstack import ModalVolumeStore

    if not go:
        raise SystemExit("refusing to spend: pass --go once the shakeout is read")
    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    train_id, held_out = _tasks(store)
    return _run(train_id, held_out, updates=updates, master=master, label="full")
