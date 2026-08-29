"""The plora-vs-lora sweep: ~40 tenants on ONE A100, two doors in.

    modal run deploy/sweep_a100.py::sweep            # boot, seed 32 arms in-process
    modal run deploy/sweep_a100.py::via_desk         # the OTHER 8, from a NEW
                                                     # process, through the desk
    modal run deploy/sweep_a100.py::progress         # committed updates per arm

WHAT THE SWEEP VARIES (the science): how the STARTING DISTRIBUTION
(plora's prior_std) and the SIZE of the delta (plora's k / lora's r, plus
plora's latent width) change one-task GRPO — 28 plora arms against 12 plain
lora arms, two seeds each, all on the screened DAPO problem. BETA is the
loss-file constant and identical for every plora arm, deliberately.

WHAT THE VENUE PROVES (the plumbing): multi-tenancy at width — every arm is
a tenant of ONE shared engine and ONE shared learner on two fractional
partitions of one device — and the STANDING FLEET end to end: the container
stands a serving host, a learner host, and a FleetService listing both;
`sweep` seeds most arms through the classic in-process submit, and
`via_desk` sends the rest from a DIFFERENT PROCESS as one frame each
(RemoteFleet -> desk -> place -> adopt), landing on the SAME roster. The
desk and the hosts know no venue word; ClsTransport below is this venue's
whole contribution (I5).
"""

import json

import modal

app = modal.App("rlstack-sweep-a100")

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
TRAIN_TASKS = "cas://09499d32b51e5e1b2a644b1c65e01b44aa42ff1a5bfac78ead41f98f89f09c93"
# the screened problem (base pass rate 4/8 at temperature 1.0 on this base)
TRAIN_TASK_ID = "dapo-math-17k/a6d38312-86c7-4022-b8d2-adcf19fa0c3a"

SITE = "layers.*.self_attn.*"
KS = (4, 8, 16)             # every plora k the grid uses -> one factors artifact each
MEMBERS = 4
UPDATES = 20
GROUPS_PER_WAVE = 2
GROUP_SIZE = 8
MAX_TOKENS = 512

SERVE_FRACTION = 0.42
LEARN_FRACTION = 0.50

SERVE_ADDRESS = "sweep://serve"      # opaque strings: the desk and the hosts
LEARN_ADDRESS = "sweep://train"      # never learn what they mean


# ---------------------------------------------------------------------------
# the science: the grid, and one spec per arm
# ---------------------------------------------------------------------------

def grid() -> list[dict]:
    """40 arms: 28 plora (k x prior_std, plus a latent-width probe) and 12
    lora (r x lr), two seeds each. Every arm is one row of plain values —
    what via_desk ships and what an arm's name is derived from."""
    arms: list[dict] = []
    for seed in (11, 12):
        for k in KS:
            for prior in (0.01, 0.03, 0.1, 0.3):
                arms.append({"family": "plora", "k": k, "prior_std": prior,
                             "latent": 32, "lr": 3e-4, "seed": seed})
        for latent in (16, 64):
            arms.append({"family": "plora", "k": 8, "prior_std": 0.05,
                         "latent": latent, "lr": 3e-4, "seed": seed})
        for r in (4, 8, 16):
            for lr in (1e-4, 3e-4):
                arms.append({"family": "lora", "r": r, "lr": lr, "seed": seed})
    return arms


def spec_for(arm: dict, factors: dict[str, str], store):
    """One arm as one ExperimentSpec: the same one-task plans, the same
    schedule, the bank and optimizer the arm names."""
    from rlstack import (
        AlgoSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup, GpuSet,
        GroupPlan, LearnerMember, OptimSpec, Plans, PolicySpec, PoolMember,
        RunPlan, Sample, SamplingSpec, Schedule, Seeds, WavePlan, WaveRef,
        encode, lora, plora,
    )

    wave = WavePlan(tuple(
        GroupPlan(f"{TRAIN_TASK_ID}#{g}",
                  tuple(Sample(TRAIN_TASK_ID, "dapo_math")
                        for _ in range(GROUP_SIZE)))
        for g in range(GROUPS_PER_WAVE)))
    plans = Plans(
        train=store.cas_put(encode(RunPlan(tuple(
            WaveRef(f"self://rollouts/{u}") for u in range(1, UPDATES + 1))))),
        rollout=store.cas_put(encode(RunPlan((wave,) * UPDATES))))

    if arm["family"] == "plora":
        bank = {"pi": plora(SITE, k=arm["k"], latent=arm["latent"],
                            members=MEMBERS, prior_std=arm["prior_std"],
                            factors=factors[str(arm["k"])])}
        loss = "grpo_latent_kl"
        overrides = {"pi.mapper": {"weight_decay": 1e-2}}
    else:
        bank = {"pi": lora(SITE, r=arm["r"])}
        loss = "grpo"
        overrides = {}
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank=bank),
        gen=GenSpec(envs=("dapo_math",), tasks=(TRAIN_TASKS,),
                    sampling=SamplingSpec(temperature=1.0,
                                          max_tokens=MAX_TOKENS)),
        plans=plans,
        algo=AlgoSpec(loss=loss, post=("final_answer", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=arm["lr"],
                                      weight_decay=0.0, overrides=overrides),
                      schedule=Schedule(microbatch_tokens=512,
                                        max_policy_lag=1)),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus=GpuSet(n=1), members=(
                PoolMember("main", tp=1, fraction=SERVE_FRACTION),)),
            GpuGroup(gpus=GpuSet(n=1), members=(
                LearnerMember(fsdp=1, fraction=LEARN_FRACTION),)))),
        seeds=Seeds(master=arm["seed"]))


# ---------------------------------------------------------------------------
# the venue: two hosts + the desk in one container, and its transport
# ---------------------------------------------------------------------------

@app.cls(image=image, gpu="A100-80GB",
         volumes={"/store": store_volume, "/hf": hf_cache},
         timeout=86400, scaledown_window=1800, max_containers=1)
@modal.concurrent(max_inputs=64)
class SweepMetal:
    """One A100 wearing the whole standing fleet: a serving partition, a
    learner partition, and the desk listing both. `fleet`/`fleet_ask` are the
    desk's two verbs on the wire; `run_arms` is the classic in-process door
    for the bulk of the grid."""

    @modal.enter()
    def bring_up(self) -> None:
        from rlstack import ModalVolumeStore
        from rlstack.policy.siteschema import hf_schema
        from rlstack.runner.engines.vllm_engine import VllmEngine
        from rlstack.runner.fleet import FleetService
        from rlstack.runner.host import Host, Partition, Regime
        from rlstack.runner.learners.torch_learner import TorchLearner
        from rlstack.runner.remote import HostService, LocalTransport, RemoteHost

        self.store = ModalVolumeStore("/store", volume=store_volume,
                                      locator=STORE)
        self.factors = ensure_factors(self.store)
        ensure_tasks(self.store)

        # serves BOTH adapter types; plora's demands are the wider sizing and
        # are folded last, which the serves order states on purpose
        engine = VllmEngine(BASE, tp=1, gpu_memory_utilization=SERVE_FRACTION,
                            max_model_len=1536, max_bundles=64, max_rank=16,
                            max_members=MEMBERS, cas_get=self.store.cas_get,
                            serves=("lora", "plora"))
        self.serve_host = Host(
            "sweep-serve", engines=(engine,), learner=None, store=self.store,
            partition=Partition("modal-a100", (0,), SERVE_FRACTION, "A100-80GB"),
            regimes=(Regime("serve-tp1", "inference", BASE, 1),))
        transports = {
            SERVE_ADDRESS: LocalTransport(HostService(self.serve_host))}
        self.learn_host = Host(
            "sweep-train", engines=(), learner=TorchLearner(), store=self.store,
            partition=Partition("modal-a100", (0,), LEARN_FRACTION, "A100-80GB"),
            regimes=(Regime("train-fsdp1", "training", BASE, 1),),
            schema_for=hf_schema,
            dial=lambda address: transports[address])
        transports[LEARN_ADDRESS] = LocalTransport(HostService(self.learn_host))

        self.desk = FleetService(
            self.store,
            connect=lambda address: RemoteHost(transports[address]))
        self.desk.list_host("sweep-serve", self.serve_host.regimes,
                            SERVE_ADDRESS)
        self.desk.list_host("sweep-train", self.learn_host.regimes,
                            LEARN_ADDRESS)
        self.schema = hf_schema(BASE)
        print(f"[sweep] up: {BASE} on one A100, "
              f"factors {sorted(self.factors)} listed 2 hosts")

    @modal.method()
    async def fleet(self, verb: str, payload: dict) -> dict:
        return await self.desk.serve(verb, payload)

    @modal.method()
    def fleet_ask(self, verb: str, payload: dict) -> dict:
        return self.desk.answer(verb, payload)

    @modal.method()
    def build_rows_here(self, arms: list[dict]) -> list[dict]:
        """Arms to canonical spec rows, INSIDE the container: the plans' cas
        blobs land on the store this container reads, visible immediately —
        a separate writer's commit would not be until a reload."""
        from rlstack.spec.canonical import canonical_json

        rows = [json.loads(canonical_json(spec_for(arm, self.factors,
                                                   self.store)))
                for arm in arms]
        store_volume.commit()
        return rows

    @modal.method()
    def roster(self) -> dict:
        """Progress, from the one process that holds it: each tenancy's
        status plus its committed-update count off the ledger."""
        out = {}
        for rid, tenancy in sorted(self.learn_host.roster.items()):
            entries = self.store.peek_ledger(rid)
            out[rid] = {"status": tenancy.status,
                        "committed": int(entries[-1]["update"]) if entries else 0}
        return out

    @modal.method()
    async def run_arms(self, rows: list[dict]) -> dict:
        """The classic door, at width: every row submitted in-process to the
        learner host, all concurrently, and the method holds the container
        open until every tenancy — THESE and any the desk adopted — is done."""
        import asyncio

        from rlstack.runner.remote import RemotePool, spec_from_json

        async def one(row: dict) -> tuple[str, int]:
            pool = RemotePool(
                self.learn_host.dial(SERVE_ADDRESS), base=BASE, tp=1)
            spec = spec_from_json(row)
            report = await self.learn_host.submit(
                spec, self.schema, remotes={"main": pool})
            return report.run_id, report.updates_completed

        stats_task = asyncio.create_task(self.serve_host.run_stats())
        done = await asyncio.gather(*(one(row) for row in rows),
                                    return_exceptions=True)
        finished = [d for d in done if not isinstance(d, BaseException)]
        failed = [repr(d) for d in done if isinstance(d, BaseException)]
        while any(t.status == "running"
                  for t in self.learn_host.roster.values()):
            await asyncio.sleep(20)          # desk-adopted arms still going
        stats_task.cancel()
        store_volume.commit()
        return {"finished": finished, "failed": failed,
                "roster": {rid: t.status
                           for rid, t in self.learn_host.roster.items()}}

    @modal.exit()
    def bring_down(self) -> None:
        self.serve_host.engine_for(BASE, 1).shutdown()


class ClsTransport:
    """This venue's whole contribution: the Transport contract over a Modal
    cls handle. The desk and the hosts never see it (I5)."""

    def __init__(self, handle) -> None:
        self.handle = handle

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.handle.fleet.remote.aio(verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return self.handle.fleet_ask.remote(verb, payload)


# ---------------------------------------------------------------------------
# helpers the container runs at boot (all content-addressed: reruns are free)
# ---------------------------------------------------------------------------

def ensure_factors(store) -> dict[str, str]:
    """One frozen artifact per k the grid uses, built if this store has not
    seen it. Content addressing makes this idempotent across workspaces."""
    from rlstack.policy.adapters.plora_factors import (
        build_factors, hf_weight_reader,
    )
    from rlstack.policy.siteschema import hf_schema, resolve

    sites = resolve(hf_schema(BASE).sites, SITE)
    reader = hf_weight_reader(BASE)
    return {str(k): store.cas_put(build_factors(BASE, sites, k, reader))
            for k in KS}


def ensure_tasks(store) -> None:
    """The DAPO task sets, present on THIS store (seed 17: byte-identical
    uris on every workspace)."""
    try:
        store.cas_get(TRAIN_TASKS)
    except FileNotFoundError:
        from rlstack.__main__ import build_task_sets
        build_task_sets(store, "dapo_math", {"train": 0.98, "eval": 0.02}, 17)


# ---------------------------------------------------------------------------
# the two doors
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def sweep(desk_arms: int = 8) -> None:
    """Seed the grid: the LAST `desk_arms` arms are left for via_desk (the
    separate-process proof); everything else goes through the classic
    in-process door now."""
    arms = grid()
    handle = SweepMetal()
    rows = handle.build_rows_here.remote(arms[:-desk_arms])
    print(f"[sweep] {len(arms)} arms; {len(rows)} in-process now, "
          f"{desk_arms} held for via_desk")
    call = handle.run_arms.spawn(rows)
    print(f"[sweep] container seeded: {call.object_id}")
    print("[sweep] now run: modal run deploy/sweep_a100.py::via_desk")


@app.local_entrypoint()
async def via_desk(desk_arms: int = 8) -> None:
    """THE PROOF: a different process, one frame per experiment, through the
    desk — placed over the listings, adopted at the learner host, landing on
    the same roster as the in-process arms."""
    from rlstack.runner.remote import RemoteFleet

    handle = SweepMetal()
    rows = handle.build_rows_here.remote(grid()[-desk_arms:])
    fleet = RemoteFleet(ClsTransport(handle))
    print(f"[via_desk] status before: "
          f"{sorted(fleet.status()['listings'])}")
    for row in rows:
        reply = await fleet.submit(row)
        print(f"[via_desk] {reply.get('run_id')} accepted="
              f"{reply.get('accepted')} host={reply.get('host')} "
              f"{reply.get('error', '')}")


@app.local_entrypoint()
def progress() -> None:
    """Committed updates per arm, straight off the roster and the ledgers."""
    print(json.dumps(SweepMetal().roster.remote(), indent=2))
