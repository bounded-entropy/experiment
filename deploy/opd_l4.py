"""OPD on real metal: an 8B student distilled from a 32B teacher, three hosts.

    modal run deploy/opd_l4.py::probe    # the teacher host ALONE, one score
    modal run deploy/opd_l4.py           # the whole run, three containers

THE SHAPE (#43's per-capability hosts, executed). Nothing here coordinates
anything: each container is born one atomic purposed partition, and the run
reaches the two it does not live on through the wire.

    teacher    Qwen3-32B, tp=4 on L4:4 — INFERENCE ONLY, no learner, no
               adapters (a non-policy pool gets a payload-free bundle, so the
               frozen base is what serves). 32B in bf16 is ~65 GiB of
               weights; four L4s is the smallest metal that holds it.
    student    Qwen3-8B, tp=2 on L4:2 — the sampler. It lives on its OWN
               partition because #45 measured that it cannot share the
               learner's: an 8B engine wants its whole 15.3 GiB where the
               learner's shard already is.
    learner    Qwen3-8B, FsdpTorchLearner fsdp=2 on L4:2, the runner beside
               it (the learner is NEVER remote). Both pools are RemotePools
               over ModalTransport; the store plane rides the volume.

WHAT THE RUN PROVES, in the order the checks assert it:
    the wire      two containers advertise (base, tp) and answer verbs
    the vocab     the same token ids the student sampled are scored by a
                  DIFFERENT model — which is only meaningful because the two
                  share a tokenizer, so `probe` checks that first
    the column    teacher_logprobs lands in postdata, one float per generated
                  token, and the run's own dictionary.json shows it at token
                  granularity feeding loss:opd
    the rails     logprob_gap stays at the kernel floor (the STUDENT pool
                  served exactly the adapters the local trainer recomputed —
                  the teacher is a different model and never enters this rail)

Cost is the reason the schedule is tiny: eight L4s are live at once, and this
run exists to prove plumbing, not convergence.

WHAT THE METAL SAID (2026-08-28, run c0f65f24362b, 4 updates, 6/6 checks).
Loads: the student 8.17 GiB per device in 81s with 10.28 GiB left for KV; the
teacher 16.5 GiB per device in 52s with 3.03 GiB left (49,664 tokens) — 32B at
tp=4 fits an L4:4 with room to prefill, which was the open question. Warming
both partitions CONCURRENTLY took 237s; the four updates took 386s.

    update 1: reward 0.750  loss +0.3559  ratio 0.9968  gap 0.0216
    update 2: reward 0.500  loss +0.3041  ratio 1.0001  gap 0.0201
    update 3: reward 0.500  loss +0.3146  ratio 1.0001  gap 0.0185
    update 4: reward 0.500  loss +0.2848  ratio 0.9974  gap 0.0165

`loss` IS the per-token reverse KL in nats, so that column is the distillation
signal itself: 0.356 → 0.285 over four steps, i.e. the student moved toward the
teacher. At update 1 the LoRA is still B=0, so 0.356 nats/token is the bare
8B-vs-32B distance on these completions. Over all 384 scored tokens the teacher
mean is -1.3357 and the student's recorded mean -1.0188 (KL +0.3169) — the
teacher is LESS confident on the student's draws than the student is, which is
what sampling from the student guarantees.

`gap` is a different rail and stays at 0.0165-0.0216 — the student pool served
exactly the adapters the local trainer recomputed, across a wire, with a
sharded learner. #45's scoring floor enters the TEACHER column instead, as
prefill-vs-decode kernel noise; it is bounded by that same ~0.02 and the
teacher pool carries no adapter at all, so it is ~6% of a 0.32-nat signal and
never an alignment error (::probe's shift-free evidence: four nats between the
true and a wrong continuation).

Deployment only (I5): wiring and measurement, nothing semantics-bearing.
Image pins: keep in sync with deploy/modal_app.py.

NOTE (modal 1.5): no `from __future__ import annotations` in this file — it
stringifies class annotations, and modal.parameter validates its fields by
their annotation OBJECT (deploy/modal_host.py carries the same note).
"""

import modal

from probe import CHECKS, arith_tasks, check

app = modal.App("rlstack-opd")

store_volume = modal.Volume.from_name("rlstack-store", create_if_missing=True)
# 32B is ~65 GiB on the wire from HF; a cache volume makes the second
# container start (and every retry) cheap. Deployment convenience only.
hf_cache = modal.Volume.from_name("rlstack-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0",
          # #45's deployment fact: vLLM's TP workers segfault in libgomp on
          # their first OpenMP-parallel CPU op in this image; spawned workers
          # that never form a thread team do not reach it.
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
          "OMP_NUM_THREADS": "1",
          # every container's HF cache is the same volume: the 32B is
          # downloaded once, ever (hf_hub 1.x transfers over Xet by default —
          # HF_HUB_ENABLE_HF_TRANSFER is retired and warns if set)
          "HF_HOME": "/hf"})
    .add_local_python_source("probe", "rlstack", "rlstack_engine")
)

STUDENT = "Qwen/Qwen3-8B"
TEACHER = "Qwen/Qwen3-32B"
STORE = "modal://rlstack-store"
VOLUMES = {"/store": store_volume, "/hf": hf_cache}



# ---------------------------------------------------------------------------
# the serving end: one capability per container
# ---------------------------------------------------------------------------

def serve_one_partition(name, base, tp, memory):
    """Build the metal, wear it as a Host, expose it: the body both serving
    classes share. The container IS the host — born with its Partition and
    its one inference Regime, attesting the engine it was handed against them
    (#43), and journaling host-up to the volume so the observer's `hosts`
    view sees a remote partition exactly as it sees a local one."""
    from rlstack import ModalVolumeStore
    from rlstack.runner.engines.vllm_engine import VllmEngine
    from rlstack.runner.host import Host, Partition, Regime
    from rlstack.runner.remote import HostService

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    engine = VllmEngine(base, tp=tp, gpu_memory_utilization=memory,
                        max_model_len=512, max_loras=8, max_lora_rank=16)
    host = Host(name, engines=(engine,), learner=None, store=store,
                partition=Partition("modal-l4", tuple(range(tp)), memory, "L4"),
                regimes=(Regime(f"serve-tp{tp}", "inference", base, tp),))
    print(f"[host {name}] up: {base} tp={tp} mem={memory}")
    return host, HostService(host)


@app.cls(image=image, gpu="L4:4", volumes=VOLUMES, timeout=7200,
         scaledown_window=300, max_containers=1, cpu=8.0, memory=65536)
@modal.concurrent(max_inputs=32)
class TeacherHost:
    """32B across four L4s, frozen. It trains nothing and owns no learner:
    a teacher is an inference capability, and the fleet addresses it by
    (base, tp) like any other."""

    base: str = modal.parameter(default=TEACHER)
    tp: int = modal.parameter(default=4)
    host_name: str = modal.parameter(default="modal-teacher-32b")

    @modal.enter()
    def bring_up(self):
        # 16.4 GiB of weights per device out of 22: the budget has to be
        # generous or there is no KV cache left to prefill into
        self.host, self.service = serve_one_partition(
            self.host_name, self.base, self.tp, 0.90)

    @modal.exit()
    def persist_cache(self):
        """The 65 GiB this container pulled from HF outlives it: the next
        start (a retry, the run after the probe) reads the volume instead."""
        hf_cache.commit()

    @modal.method()
    async def call(self, verb: str, payload: dict) -> dict:
        return await self.service.serve(verb, payload)

    @modal.method()
    def ask(self, verb: str, payload: dict) -> dict:
        return self.service.answer(verb, payload)

    @modal.method()
    def status(self) -> dict:
        return {"describe": self.service.describe(),
                "status": self.host.status()}


@app.cls(image=image, gpu="L4:2", volumes=VOLUMES, timeout=7200,
         scaledown_window=300, max_containers=1, cpu=8.0, memory=32768)
@modal.concurrent(max_inputs=64)
class StudentHost:
    """The sampler: 8B at tp=2, serving this run's LoRA bundles. It is a
    separate host from the learner because #45 measured that the two do not
    fit on one 2xL4 — the fleet's answer to that is this file."""

    base: str = modal.parameter(default=STUDENT)
    tp: int = modal.parameter(default=2)
    host_name: str = modal.parameter(default="modal-student-8b")

    @modal.enter()
    def bring_up(self):
        self.host, self.service = serve_one_partition(
            self.host_name, self.base, self.tp, 0.85)

    @modal.exit()
    def persist_cache(self):
        hf_cache.commit()

    @modal.method()
    async def call(self, verb: str, payload: dict) -> dict:
        return await self.service.serve(verb, payload)

    @modal.method()
    def ask(self, verb: str, payload: dict) -> dict:
        return self.service.answer(verb, payload)

    @modal.method()
    def status(self) -> dict:
        return {"describe": self.service.describe(),
                "status": self.host.status()}


class ModalTransport:
    """The client end, verbatim from deploy/modal_host.py: `call` is
    `.remote.aio`, `ask` is `.remote`. HostService already speaks JSON-safe
    dict frames, so a real transport has nothing to serialize."""

    def __init__(self, partition):
        self.partition = partition

    async def call(self, verb: str, payload: dict) -> dict:
        return await self.partition.call.remote.aio(verb, payload)

    def ask(self, verb: str, payload: dict) -> dict:
        return self.partition.ask.remote(verb, payload)


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------



def opd_spec(store, n_updates, master):
    """One on-policy-distillation experiment, declared and nothing more.

    The spec says WHAT: a student that samples, a teacher pool that scores,
    a loss that reads the resulting column. It never says WHERE — three
    capability demands (a tp=2 sampler, a tp=4 teacher, an fsdp=2 learner)
    that placement satisfies with three per-capability hosts."""
    from rlstack import (
        AlgoSpec, EvalSpec, ExperimentSpec, GenSpec, GpuConfig, GpuGroup,
        OptimSpec, PolicySpec, SamplingSpec, Schedule, Seeds, TrajectorySource,
        gpus, learner, lora, pool,
    )

    return ExperimentSpec(
        policy=PolicySpec(base=STUDENT,
                          bank={"pi": lora("layers.*.self_attn.*", r=16)}),
        gen=GenSpec(env="math_single_turn",
                    tasks=store.cas_put(arith_tasks(64, 0)),
                    sampling=SamplingSpec(temperature=1.0, top_p=1.0,
                                          max_tokens=12)),
        trajectories=TrajectorySource("live"),
        algo=AlgoSpec(loss="opd", post=("verifier", "teacher_logprobs"),
                      optim=OptimSpec("adamw", lr=1e-4),
                      schedule=Schedule(group_size=4, trajectories_per_wave=8,
                                        n_updates=n_updates,
                                        microbatch_tokens=2048)),
        eval=EvalSpec(tasks=store.cas_put(arith_tasks(8, 1)), every=2,
                      n_samples=1, post=("verifier",)),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=2), (pool("main", tp=2, fraction=0.85),)),
            GpuGroup(gpus(n=4), (pool("teacher", base=TEACHER, tp=4,
                                      fraction=0.90),)),
            GpuGroup(gpus(n=2), (learner(fsdp=2, fraction=0.90),)),
        )),
        seeds=Seeds(master=master),
    )


# ---------------------------------------------------------------------------
# probe: the teacher host alone, before eight L4s are ever live at once
# ---------------------------------------------------------------------------

PROBE_PROMPT = "What is 47+58? The answer is"


@app.function(image=image, volumes={"/hf": hf_cache}, timeout=5400)
def probe():
    """One score_tokens call against 32B at tp=4, and the precondition that
    makes the whole channel meaningful.

    THE TOKENIZER IS THE PRECONDITION: the ids crossing the wire are the
    STUDENT's draws, so a teacher with another vocabulary would be scoring
    different text. Checked here, on the cheap, before any run.

    THE SCORE IS THE SIGNAL: the teacher is asked for the same continuation
    twice — the true sum and a wrong one. A frozen 32B that is actually
    reading prefers the true one, and by a wide margin; identical scores
    would mean the prefill never saw the context.

    OBSERVED (2026-08-28, one L4:4, twice on two separate cold starts): the
    tokenizers agree exactly (12 ids); ' 105' scores -0.0814 per token and
    ' 731' -4.1414 — four nats, so the prefill demonstrably read the context.
    The per-token scores came back BIT-IDENTICAL across both container
    lifetimes ([-0.3004, -0.0213, -0.0026, -0.0013]), which is the seedless-
    determinism claim of the scoring verb (#40) tested the only way that
    counts. Cold start: 16.5 GiB of weights per device, 308s to load
    unauthenticated from HF the first time and 52s from the cache volume
    after, 3.03 GiB left for KV, ~30s to init the engine."""
    import asyncio

    from transformers import AutoTokenizer

    from rlstack import Bundle, Message, RemotePool, Role

    teacher_pool = TeacherHost()
    student_vocab = AutoTokenizer.from_pretrained(STUDENT)

    print("[probe] waking the teacher host (32B loads on its first verb)")
    teacher = RemotePool(ModalTransport(teacher_pool), base=TEACHER, tp=4)
    theirs = teacher.tokenize(PROBE_PROMPT)
    ours = tuple(student_vocab.encode(PROBE_PROMPT, add_special_tokens=False))
    check("the teacher shares the student's tokenizer", theirs == ours,
          f"{len(ours)} ids, first {list(ours[:5])}")

    warm = Bundle("bundle:probe", {})
    teacher.add_bundle(warm)
    context = (Message(Role.USER, PROBE_PROMPT),)

    def ids_of(text):
        return tuple(student_vocab.encode(text, add_special_tokens=False))

    async def both():
        async def score(text):
            return await teacher.score_tokens(context, ids_of(text),
                                              warm.bundle_id)
        return await asyncio.gather(score(" 105"), score(" 731"))

    true_scores, wrong_scores = asyncio.run(both())
    check("one score per token", len(true_scores) == len(ids_of(" 105")),
          f"{len(true_scores)} scores")
    check("every score is a logprob", all(s < 0.0 for s in true_scores),
          f"{[round(s, 4) for s in true_scores]}")
    true_mean = sum(true_scores) / len(true_scores)
    wrong_mean = sum(wrong_scores) / len(wrong_scores)
    check("the teacher prefers the true continuation", true_mean > wrong_mean,
          f"' 105' {true_mean:+.4f} vs ' 731' {wrong_mean:+.4f}")
    print("\n[teacher host]", teacher_pool.status.remote())

    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    print(f"\n[probe] {sum(ok for _, ok, _ in CHECKS)} passed, {len(failed)} "
          f"failed: {failed}")
    return {"failed": failed, "true_mean": true_mean, "wrong_mean": wrong_mean}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def warm(pools):
    """Force both containers to BUILD their engines, CONCURRENTLY.

    vLLM's AsyncLLMEngine materializes on the first admitted verb, so without
    this the run's first post phase would sit through a 32B load. Concurrently
    because the two partitions are billed the whole time either is loading —
    the student must not wait out the teacher's 65 GiB. It is also the wire's
    first honest exercise: an admission-free add_bundle and an admitted score,
    both across a real transport."""
    import asyncio
    import time

    from rlstack import Bundle, Message, Role

    context = (Message(Role.USER, PROBE_PROMPT),)

    async def one(label, pool):
        bundle = Bundle(f"bundle:warm:{label}", {})
        pool.add_bundle(bundle)
        started = time.time()
        scores = await pool.score_tokens(context, (16, 17, 18),
                                         bundle.bundle_id)
        took = time.time() - started
        print(f"[warm] {label} ready in {took:.0f}s, "
              f"scores {[round(s, 3) for s in scores]}")
        return took

    async def all_of_them():
        return await asyncio.gather(*(one(label, pool)
                                      for label, pool in pools.items()))

    return max(asyncio.run(all_of_them()))


def teacher_column_report(run, updates):
    """What the teacher actually said, read back out of the store.

    Two claims and one measurement. TOKEN ALIGNMENT: every trajectory's
    column has one float per generated token of that trajectory's sealed
    turns — the thing run_pipeline validates at write time, re-checked here
    from the bytes. THE SIGNAL: the mean per-token (behavior − teacher), the
    RECORD's reading of the reverse KL the loss minimizes (the trainer's own
    logprobs differ from the record by exactly logprob_gap, which is the
    kernel floor here). It measures the distance between two DIFFERENT
    models on the student's own draws, and has nothing to do with the gap
    rail, which compares the student to itself."""
    aligned = True
    teacher_all, behavior_all, kl_all = [], [], []
    for u in updates:
        postdata = run.read_postdata(u)
        rows = run.read_wave(u)
        column = postdata["teacher_logprobs"]
        aligned = aligned and len(column) == len(rows)
        for vector, row in zip(column, rows):
            behavior = [lp for turn in row["turns"]
                        for lp in turn["behavior_logprobs"]]
            aligned = aligned and len(vector) == len(behavior)
            teacher_all.extend(vector)
            behavior_all.extend(behavior)
            kl_all.extend(b - t for b, t in zip(behavior, vector))

    def mean(values):
        return sum(values) / len(values) if values else float("nan")

    check("teacher_logprobs is token-aligned in every update", aligned,
          f"{len(teacher_all)} scored tokens over {len(updates)} updates")
    print(f"    teacher mean logprob   {mean(teacher_all):+.4f}")
    print(f"    student mean logprob   {mean(behavior_all):+.4f}")
    print(f"    per-token reverse KL   {mean(kl_all):+.4f} nats "
          f"(student − teacher, the objective)")
    return {"tokens_scored": len(teacher_all),
            "teacher_mean": round(mean(teacher_all), 5),
            "student_mean": round(mean(behavior_all), 5),
            "reverse_kl": round(mean(kl_all), 5)}


@app.function(image=image, gpu="L4:2", volumes=VOLUMES, timeout=7200,
              cpu=8.0, memory=65536)
def opd_run(n_updates: int = 2, master: int = 71) -> dict:
    """The learner's container drives; both pools are somebody else's metal."""
    import asyncio
    import json
    import time

    import torch

    from rlstack import ModalVolumeStore, RemotePool
    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.host import Host, Partition, Regime
    from rlstack.runner.learners.fsdp_torch import lead_fsdp_learner

    store = ModalVolumeStore("/store", volume=store_volume, locator=STORE)
    spec = opd_spec(store, n_updates=n_updates, master=master)
    schema = hf_schema(STUDENT)

    student_pool = RemotePool(ModalTransport(StudentHost()), base=STUDENT, tp=2)
    teacher_pool = RemotePool(ModalTransport(TeacherHost()), base=TEACHER, tp=4)
    print(f"[pins] torch={torch.__version__} "
          f"cuda_devices={torch.cuda.device_count()}")
    warmed = warm({"student-8b-tp2": student_pool,
                   "teacher-32b-tp4": teacher_pool})

    learner = lead_fsdp_learner(2)
    host = Host("modal-opd-learner", engines=(), learner=learner, store=store,
                partition=Partition("modal-l4", (0, 1), 0.90, "L4"),
                regimes=(Regime("learner-fsdp2", "training", STUDENT, 2),))
    print(f"[chorus] rank 0 of 2, learner.fsdp={learner.fsdp}")

    started = time.time()
    try:
        report = asyncio.run(host.submit(
            spec, schema, remotes={"main": student_pool,
                                   "teacher": teacher_pool}))
    finally:
        learner.stop()
    elapsed = time.time() - started
    store_volume.commit()

    run = store.open_run(report.run_id)
    entries = run.read_ledger()
    print(f"\nrun_id={report.run_id}  updates={report.updates_completed}  "
          f"wall={elapsed:.0f}s (+{warmed:.0f}s warming)")
    for entry in entries:
        print(f"  update {entry['update']}: reward {entry['post']['reward']:.3f}  "
              f"loss {entry['train']['loss']:+.4f}  "
              f"ratio {entry['train']['mean_ratio']:.4f}  "
              f"gap {entry['train']['logprob_gap']:.4f}  "
              f"grad {entry['train']['grad_norm']:.3f}")

    updates = [int(e["update"]) for e in entries]
    check("ledger complete", updates == list(range(1, n_updates + 1)),
          f"{len(updates)}/{n_updates}")
    gaps = [e["train"]["logprob_gap"] for e in entries]
    check("logprob_gap at the kernel floor", bool(gaps) and max(gaps) < 0.15,
          f"max {max(gaps):.4f}" if gaps else "no updates")
    column = teacher_column_report(run, updates)

    dictionary = store.peek_dictionary(report.run_id)
    node = [c for c in dictionary["columns"]
            if c["name"] == "teacher_logprobs" and c["phase"] == "post"][0]
    check("the run describes its own teacher channel",
          node["granularity"] == "token" and node["feeds_loss"]
          and "loss:opd" in node["consumers"], json.dumps(node))
    check("evals present",
          all(run.has_eval(u) for u in updates if u % 2 == 0))
    check("both pools journaled as remote", any(
        sorted(row.get("remotes", [])) == ["main", "teacher"]
        for row in _host_journal(store, "modal-opd-learner")))

    print("\n[learner host]", host.status())
    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    print(f"\n[opd checks] {sum(ok for _, ok, _ in CHECKS)} passed, "
          f"{len(failed)} failed: {failed}")
    return {"run_id": report.run_id, "wall_s": round(elapsed, 1),
            "warm_s": round(warmed, 1),
            "max_gap": max(gaps) if gaps else None,
            "rewards": [e["post"]["reward"] for e in entries],
            **column, "failed": failed}


def _host_journal(store, host):
    import json

    path = store.path_of(f"hosts/{host}/log.jsonl")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


@app.function(image=image, timeout=600)
def describe() -> dict:
    """What the two serving partitions advertise, fetched over the wire."""
    return {"teacher": TeacherHost().status.remote(),
            "student": StudentHost().status.remote()}


@app.local_entrypoint()
def main(n_updates: int = 2) -> None:
    result = opd_run.remote(n_updates)
    print("\n[opd run]", result)
    if result["failed"]:
        raise SystemExit(f"OPD checks FAILED: {result['failed']}")
    print("\nALL OPD CHECKS PASSED")
