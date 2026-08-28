"""The wire on real venues: a host in ITS OWN container, reached by Modal.

    modal run deploy/modal_host.py                # a whole arith run, remote pool
    modal run deploy/modal_host.py::describe      # what the served host serves

The venue model (Samarth's, settled): POOL TRAFFIC — sample and score, the
ms-latency RPC — rides a real transport; the STORE PLANE — ledger, waves,
journals — rides the Modal volume exactly as it already does. So this file
carries two halves of one wire and nothing else:

    ServedHost      the host-service container: it BUILDS the metal (a
                    VllmEngine at some (base, tp)), wears it as a Host born
                    with that Partition and Regime (#43), and exposes its
                    HostService's two verbs as Modal methods. Admission stays
                    here, at the partition, in this host's own arbiter — a
                    remote experiment is one more source of admitted work.
    ModalTransport  the client end: `call` is `.remote.aio(...)`, `ask` is
                    `.remote(...)`. That is the whole implementation. It is
                    thin because HostService already speaks JSON-safe dict
                    frames and LocalTransport already proved every frame
                    survives a round trip — the transport carries frames, it
                    never learns what is in them.

The runner cannot tell this from local metal: the driver below runs a
complete GRPO run whose "main" pool is a RemotePool over ModalTransport, with
the learner local to the driver (the learner is NEVER remote — the runner
goes to it) and both containers' stores on the one volume.

Deployment only (I5): venue wiring, nothing semantics-bearing.
Image pins: keep in sync with deploy/modal_app.py.

NOTE (modal 1.5): no `from __future__ import annotations` in this file — it
stringifies class annotations, and modal.parameter validates its fields by
their annotation OBJECT ("'str' object has no attribute '__name__'").
"""

import modal

from probe import arith_tasks

app = modal.App("rlstack-remote")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("probe", "rlstack", "rlstack_engine")
)

BASE = "Qwen/Qwen3-0.6B"
STORE = "modal://rlstack-store"


# ---------------------------------------------------------------------------
# the serving end: one host, one container
# ---------------------------------------------------------------------------

@app.cls(image=image, gpu="L4", volumes={"/store": store_volume},
         timeout=3600, scaledown_window=120, max_containers=1)
@modal.concurrent(max_inputs=64)
class ServedHost:
    """One inference partition, contactable from outside its process.

    The container IS the host: it is born with its Partition and its Regime,
    the Host constructor attests the engine it was handed against them, and
    `host-up` lands in the journal on the volume — so the observer's `hosts`
    view sees a remote partition exactly as it sees a local one. Capability is
    a birth fact (#43): a container serving (base, tp) never becomes another.

    max_containers=1 because a host is one partition, not an autoscaling
    pool — a second container would be a second host with its own KV cache and
    its own adapter registrations, and no experiment asked for one.
    Concurrency is the point of the whole exercise: `max_inputs` lets many
    sample requests sit in flight so vLLM's scheduler batches them, which is
    what makes a remote pool worth having.
    """

    base: str = modal.parameter(default=BASE)
    tp: int = modal.parameter(default=1)
    host_name: str = modal.parameter(default="modal-inference")

    @modal.enter()
    def bring_up(self) -> None:
        from rlstack import ModalVolumeStore
        from rlstack.runner.engines.vllm_engine import VllmEngine
        from rlstack.runner.host import Host, Partition, Regime
        from rlstack.runner.remote import HostService

        store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
        engine = VllmEngine(self.base, tp=self.tp,
                            gpu_memory_utilization=0.60, max_model_len=512,
                            max_loras=8, max_lora_rank=16)
        # the vLLM engine itself materializes inside the running loop, on the
        # first admitted verb (AsyncLLMEngine wants a loop) — the enter hook
        # builds everything that does not
        self.host = Host(
            self.host_name, engines=(engine,), learner=None, store=store,
            partition=Partition("modal-l4", tuple(range(self.tp)), 0.60),
            regimes=(Regime(f"serve-tp{self.tp}", "inference", self.base,
                            self.tp),))
        self.service = HostService(self.host)
        print(f"[host {self.host_name}] up: {self.base} tp={self.tp}")

    @modal.method()
    async def call(self, verb: str, payload: dict) -> dict:
        """The admitted verbs (sample_tokens / score_tokens), async because
        they occupy the GPU and the host's arbiter admits them one regime at
        a time."""
        return await self.service.serve(verb, payload)

    @modal.method()
    def ask(self, verb: str, payload: dict) -> dict:
        """The admission-free verbs (add_bundle / reachability / tokenize) —
        additive registration and build facts, so no admission and no loop."""
        return self.service.answer(verb, payload)

    @modal.method()
    def status(self) -> dict:
        """What this partition serves and who is on it — the same dict the
        local `hosts` view renders, fetched across the wire."""
        return {"describe": self.service.describe(),
                "status": {k: v for k, v in self.host.status().items()
                           if k != "partition"}}


# ---------------------------------------------------------------------------
# the client end
# ---------------------------------------------------------------------------

class ModalTransport:
    """A Transport whose wire is a Modal method call.

    The two verbs map one-to-one onto Modal's two calling conventions, and
    that is the entire class: `call` is awaited from the runner's own loop
    (`.remote.aio`), `ask` is called from sync call sites (`.remote`) —
    add_bundle, reachability and tokenize are sync in the Engine protocol
    because, by the multi-tenancy invariant, they never disturb traffic.

    Frames cross as JSON-safe dicts, which is a contract HostService already
    keeps and LocalTransport already enforces; nothing here inspects, encodes
    or repairs a frame, and if anything ever needs to, the fix belongs in
    runner/remote.py where both ends can see it.

    `ask` is called from inside the runner's event loop (Phase 1 registers a
    bundle; flatten tokenizes each message), and modal warns about a blocking
    interface in an async context every time. It works, it is the contract —
    an admission-free verb blocks nothing but its own caller — and
    ModalVolumeStore.commit() has always crossed the same way.
    """

    def __init__(self, partition: ServedHost) -> None:
        self.partition = partition

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.partition.call.remote.aio(verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return self.partition.ask.remote(verb, payload)


# ---------------------------------------------------------------------------
# the driver: a complete run whose main pool lives in another container
# ---------------------------------------------------------------------------



def arith_spec(store, *, n_updates: int, master: int):
    from rlstack import (
        AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup,
        OptimSpec, PolicySpec, SamplingSpec, Schedule, Seeds, TrajectorySource,
        gpus, learner, lora, pool,
    )

    return ExperimentSpec(
        policy=PolicySpec(base=BASE,
                          bank={"pi": lora("layers.*.self_attn.*", r=16)}),
        gen=GenSpec(env="math_single_turn", tasks=store.cas_put(arith_tasks(64, 0)),
                    sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                          max_tokens=12)),
        trajectories=TrajectorySource("live"),
        algo=AlgoSpec(loss="grpo", post=("verifier", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=1e-4),
                      schedule=Schedule(group_size=4, trajectories_per_wave=8,
                                        n_updates=n_updates,
                                        microbatch_tokens=2048)),
        eval=EvalSpec(tasks=store.cas_put(arith_tasks(16, 1)), every=2,
                      n_samples=2, post=("verifier",)),
        # the pool is declared exactly as it would be locally: the spec never
        # says WHERE (#43) — placement decided that the main pool lives on
        # another partition, and the spec is untouched by the fact
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=1), (pool("main", fraction=0.60),
                                 learner(fraction=0.40))),)),
        seeds=Seeds(master=master),
    )


@app.function(image=image, gpu="L4", volumes={"/store": store_volume},
              timeout=3600)
def run_arith_remote(n_updates: int = 3, master: int = 41) -> dict:
    """The learner's container drives; the main pool is a whole other host.

    This is the shape the fleet places into (#43): the runner goes to the
    learner's host, and every pool that landed elsewhere is a RemotePool over
    a transport. Only the transport differs from the in-process case — which
    is the claim being tested.
    """
    import asyncio
    import time

    from rlstack import ModalVolumeStore, RemotePool
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.host import Host, Partition, Regime
    from rlstack.runner.learners.torch_learner import TorchLearner

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    spec = arith_spec(store, n_updates=n_updates, master=master)

    served = ServedHost(base=BASE, tp=1, host_name="modal-inference")
    main = RemotePool(ModalTransport(served), base=BASE, tp=1)
    print("[wire] warming the serving container:", served.ask.remote(
        "tokenize", {"base": BASE, "tp": 1, "text": "warm"}))

    learner_host = Host(
        "modal-learner", engines=(), learner=TorchLearner(), store=store,
        partition=Partition("modal-l4", (0,), 0.40),
        regimes=(Regime("learner-fsdp1", "training", BASE, 1),))

    started = time.time()
    report = asyncio.run(learner_host.submit(spec, hf_schema(BASE),
                                             remotes={"main": main}))
    elapsed = time.time() - started
    store_volume.commit()

    run = store.open_run(report.run_id)
    entries = run.read_ledger()
    print(f"\nrun_id={report.run_id}  updates={report.updates_completed}  "
          f"resumed_from={report.resumed_from}  wall={elapsed:.0f}s")
    for entry in entries:
        print(f"  update {entry['update']}: reward {entry['post']['reward']:.3f}  "
              f"loss {entry['train']['loss']:+.4f}  "
              f"gap {entry['train']['logprob_gap']:.4f}  "
              f"grad {entry['train']['grad_norm']:.3f}")
    print("\n[served host]", served.status.remote())
    print("[learner host]", learner_host.status())

    gaps = [e["train"]["logprob_gap"] for e in entries]
    checks = {
        "ledger complete": [int(e["update"]) for e in entries] == list(
            range(1, n_updates + 1)),
        "logprob_gap bounded": bool(gaps) and max(gaps) < 0.15,
        "evals present": all(run.has_eval(u)
                             for u in range(1, n_updates + 1) if u % 2 == 0),
        "remote pool journaled": any(
            "main" in row.get("remotes", [])
            for row in _host_journal(store, "modal-learner")),
    }
    print("\n[checks]", checks)
    return {"run_id": report.run_id, "wall_s": round(elapsed, 1),
            "max_gap": max(gaps) if gaps else None,
            "failed": [name for name, ok in checks.items() if not ok]}


def _host_journal(store, host: str) -> list[dict]:
    import json

    path = store.path_of(f"hosts/{host}/log.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@app.function(image=image, timeout=600)
def describe() -> dict:
    """The advertisement the fleet matches on, fetched over the wire."""
    served = ServedHost(base=BASE, tp=1, host_name="modal-inference")
    return served.status.remote()


@app.local_entrypoint()
def main(n_updates: int = 3) -> None:
    result = run_arith_remote.remote(n_updates)
    print("\n[remote-pool run]", result)
    if result["failed"]:
        raise SystemExit(f"remote-pool checks FAILED: {result['failed']}")
    print("\nALL REMOTE-POOL CHECKS PASSED")
