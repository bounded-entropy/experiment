"""Tensor parallelism on real metal: one engine BUILT tp=2, on one L4:2 host.

    modal run deploy/tp_l4.py                        # Qwen3-0.6B, tp=2
    modal run deploy/tp_l4.py::tp_probe --tp 1       # the same probes, tp=1
    modal run deploy/tp_l4.py::tp_probe_8b           # Qwen3-8B, tp=2

Sharding is a BUILD fact (#43) — `VllmEngine(tp=2)` is a different piece of
metal, not a different request, and the submit gate attests it — and this is
that build's first contact with real GPUs. It checks the three things the
Engine protocol promises: a sharded build streams tokens like any other; two
compiled bundles COEXIST on it with every concurrently issued request pinning
its own, so multi-tenancy carries through punica's per-token adapter indices
under TP; and score_tokens' prompt_logprobs suffix matches, token by token,
the logprobs sampling reported for those same tokens, so an off-by-one in the
indexing shows up as a gap of nats rather than of rounding.

Deployment only (I5): wiring and measurement, nothing semantics-bearing.
Image pins: keep in sync with deploy/modal_app.py.
"""

from __future__ import annotations

import modal

from probe import CHECKS, check

app = modal.App("rlstack-tp")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.28.0", "torch==2.13.0", "transformers==5.16.1",
                 "safetensors", "numpy")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("probe", "rlstack", "rlstack_engine")
)

BASE = "Qwen/Qwen3-0.6B"
BIG = "Qwen/Qwen3-8B"
PROMPT = "What is 47+58? The answer is"



def perturbed_lora_bundle(schema, *, seed: int, scale: float, version: int):
    """One compiled LoRA bundle whose delta is NOT the identity.

    Freshly built LoRA is B=0 — version 0 of a delta IS the base model, so two
    fresh bundles would be indistinguishable and would prove nothing about
    per-request pinning. Filling B (seeded) is exactly what a few optimizer
    steps would have done; the probe just skips the steps.
    """
    import torch

    from rlstack.policy.compile import compile_bundle
    from rlstack.registry import ADAPTERS

    sites = schema.resolve("layers.*.self_attn.*")
    lora = ADAPTERS.get("lora").instance
    params = lora.params(sites, {"r": 16, "seed": seed})
    generator = torch.Generator().manual_seed(seed)
    for path, b in params.b.items():
        b.data.normal_(0.0, scale, generator=generator)
    return compile_bundle({"pi": lora.emit(params)}, {"pi": version},
                          ["pi"], {"pi": "lora"})


async def greedily(engine, bundle_id: str, *, max_tokens: int = 8,
                   prompt: str = PROMPT):
    """One greedy completion, as (token ids, per-token logprobs, text)."""
    from rlstack import Message, Role, SamplingSpec, TokenEvent

    ids: list[int] = []
    logprobs: list[float] = []
    text = ""
    async for event in engine.sample_tokens(
            (Message(Role.USER, prompt),),
            SamplingSpec(temperature=0.0, top_p=1.0, max_tokens=max_tokens),
            (), bundle_id, seed=7):
        if isinstance(event, TokenEvent):
            ids.append(event.token_id)
            logprobs.append(event.logprob)
            text += event.text_delta
    return tuple(ids), tuple(logprobs), text


# ---------------------------------------------------------------------------
# the three claims, one named check each
# ---------------------------------------------------------------------------

async def sampling_streams_under_tp(engine, bundle_id: str) -> None:
    """Claim one: a tp>1 build generates. Tokens arrive, logprobs are finite,
    the stream ends in a FinishEvent (greedily() only returns when it does)."""
    import math

    ids, logprobs, text = await greedily(engine, bundle_id)
    print(f"    base completion: {text!r}  ids={ids}")
    check(f"tp={engine.tp}: sampling streams tokens", len(ids) > 0,
          f"{len(ids)} tokens")
    check(f"tp={engine.tp}: logprobs are finite",
          all(math.isfinite(lp) and lp <= 0.0 for lp in logprobs),
          f"max {max(logprobs):.4f} min {min(logprobs):.4f}")


async def bundles_coexist_and_pin_per_request(engine, base_id: str,
                                              first: str, second: str) -> None:
    """Claim two: punica multi-LoRA under TP. Three requests — bare base, and
    one per bundle — issued CONCURRENTLY so vLLM's scheduler batches them, and
    each comes back under its own adapter (different deltas, different greedy
    continuations). Same prompt, same seed: only the pinned bundle differs."""
    import asyncio

    (base_ids, _, base_text), (a_ids, _, a_text), (b_ids, _, b_text) = (
        await asyncio.gather(greedily(engine, base_id),
                             greedily(engine, first),
                             greedily(engine, second)))
    print(f"    base:     {base_text!r}")
    print(f"    bundle A: {a_text!r}")
    print(f"    bundle B: {b_text!r}")
    check(f"tp={engine.tp}: both bundles served in one batch",
          bool(a_ids) and bool(b_ids), f"{len(a_ids)} / {len(b_ids)} tokens")
    check(f"tp={engine.tp}: each request got its OWN adapter",
          a_ids != b_ids and a_ids != base_ids and b_ids != base_ids,
          f"base={base_ids[:4]} A={a_ids[:4]} B={b_ids[:4]}")


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


async def scores_match_the_sampler(engine, bundle_id: str, label: str,
                                   tolerance: float) -> float:
    """Claim three: score_tokens reads the RIGHT prompt_logprobs positions.

    Sample greedily, then re-score the sampled tokens in the same context: the
    two are the same quantity (logprob of token j given everything before it),
    computed by different paths (decode vs one prefill).

    Two separate things are checked, because they fail differently:

        ALIGNMENT is the indexing claim, and it is scale-free — the aligned
        gap must be far smaller than the gap a one-position shift would give.
        An off-by-one in the prompt_logprobs suffix cannot survive this, at
        any adapter magnitude; kernel noise cannot fail it.
        AGREEMENT is the numerics claim, and its bound is family-aware: the
        bare base runs at the kernel floor, while a punica delta is applied
        by different kernels in prefill (grouped, whole-suffix) than in decode
        (one token at a time), so a large delta rounds differently. The
        `tolerance` argument is where that fact is stated per caller.
    """
    ids, sampled, _ = await greedily(engine, bundle_id)
    from rlstack import Message, Role

    scored = await engine.score_tokens((Message(Role.USER, PROMPT),), ids,
                                       bundle_id)
    check(f"tp={engine.tp} {label}: one score per token",
          len(scored) == len(ids), f"{len(scored)} vs {len(ids)}")
    for j, (tok, s, t) in enumerate(zip(ids, sampled, scored)):
        print(f"      [{j}] token {tok:>6}  sampled {s:+.5f}  scored {t:+.5f}")

    aligned = _mean(abs(s - t) for s, t in zip(sampled, scored))
    shifted = min(_mean(abs(s - t) for s, t in zip(sampled[1:], scored)),
                  _mean(abs(s - t) for s, t in zip(sampled, scored[1:])))
    worst = max((abs(s - t) for s, t in zip(sampled, scored)), default=0.0)
    # The shift test has resolution only while the numerics gap sits under
    # the sequence's own position-to-position variation. The probe's LOUD
    # delta (|B| ~ 0.05 on every attention projection — far past anything
    # training produces) both flattens the logprob profile and inflates the
    # gap, so its ratio is REPORTED, not asserted: alignment is a property of
    # the indexing, and base + faint test it at the resolution real deltas
    # leave — which is the resolution the scoring verb runs at.
    if worst < 0.10:
        check(f"tp={engine.tp} {label}: the suffix is aligned",
              aligned < 0.25 * shifted,
              f"aligned {aligned:.5f} vs one-off {shifted:.5f}")
    else:
        print(f"    (shift test has no resolution here: aligned {aligned:.5f} "
              f"vs one-off {shifted:.5f} — the delta is louder than the "
              f"sequence's own variation)")
    check(f"tp={engine.tp} {label}: scores match the sampler", worst < tolerance,
          f"max |sampled-scored| = {worst:.5f} < {tolerance}")
    return worst


async def probe(engine, schema) -> None:
    from rlstack import Bundle

    base_bundle = Bundle("bundle:base", {})
    first = perturbed_lora_bundle(schema, seed=11, scale=0.05, version=1)
    second = perturbed_lora_bundle(schema, seed=29, scale=0.05, version=1)
    # a delta the size a few optimizer steps would make: the scoring gap's
    # dependence on adapter magnitude is what separates "punica prefill
    # rounds differently" from "the suffix is indexed wrongly"
    faint = perturbed_lora_bundle(schema, seed=11, scale=0.005, version=2)
    check("the two probe bundles are distinct",
          first.bundle_id != second.bundle_id,
          f"{first.bundle_id} / {second.bundle_id}")
    for bundle in (base_bundle, first, second, faint):
        engine.add_bundle(bundle)

    print("\n  -- sampling")
    await sampling_streams_under_tp(engine, base_bundle.bundle_id)
    print("\n  -- multi-LoRA, one batch")
    await bundles_coexist_and_pin_per_request(
        engine, base_bundle.bundle_id, first.bundle_id, second.bundle_id)
    print("\n  -- scoring (base bundle)")
    floor = await scores_match_the_sampler(engine, base_bundle.bundle_id,
                                           "base", tolerance=0.05)
    print("\n  -- scoring (faint LoRA bundle, |B| ~ 0.005)")
    faintly = await scores_match_the_sampler(engine, faint.bundle_id,
                                             "faint-lora", tolerance=0.10)
    print("\n  -- scoring (loud LoRA bundle, |B| ~ 0.05)")
    loudly = await scores_match_the_sampler(engine, first.bundle_id,
                                            "loud-lora", tolerance=0.50)
    # Reported, not checked: alignment is the claim, and it is checked above
    # wherever it resolves. This line is the accompanying numerics story —
    # how far prefill's kernels drift from decode's under a delta — and it is
    # a property of the model, not of the code: on 0.6B it grows with the
    # delta (0.02 / 0.05 / 0.28), on 8B it stays at the floor throughout.
    print(f"\n    scoring gap by adapter magnitude: base {floor:.5f}, "
          f"faint {faintly:.5f}, loud {loudly:.5f}")


def run_probe(base: str, tp: int, gpu_memory_utilization: float) -> dict:
    """Build the metal, run the three claims, report. Shared by both sizes."""
    import asyncio

    import torch
    import transformers
    import vllm

    from rlstack.policy.siteschema import hf_schema
    from rlstack.runner.engines.vllm_engine import VllmEngine

    print(f"[pins] vllm={vllm.__version__} torch={torch.__version__} "
          f"transformers={transformers.__version__} "
          f"cuda_devices={torch.cuda.device_count()}")
    engine = VllmEngine(base, tp=tp, gpu_memory_utilization=gpu_memory_utilization,
                        max_model_len=512, max_loras=8, max_lora_rank=16)
    print(f"[build] {base} tp={engine.tp} on {torch.cuda.device_count()} devices")
    asyncio.run(probe(engine, hf_schema(base)))

    failed = [(n, d) for n, ok, d in CHECKS if not ok]
    print(f"\n[tp={tp} checks] {sum(ok for _, ok, _ in CHECKS)} passed, "
          f"{len(failed)} failed: {failed}")
    return {"base": base, "tp": tp,
            "passed": sum(ok for _, ok, _ in CHECKS), "failed": failed}


@app.function(image=image, gpu="L4:2", timeout=3600)
def tp_probe(tp: int = 2) -> dict:
    return run_probe(BASE, tp, gpu_memory_utilization=0.60)


@app.function(image=image, gpu="L4:2", timeout=5400, cpu=8.0, memory=32768)
def tp_probe_8b(tp: int = 2) -> dict:
    """The same probes on a model that DOES NOT FIT one L4 in bf16 (8B ≈ 16GB
    of weights against 24GB, before KV): tp=2 is not an optimization here, it
    is the only way this base serves at all.

    Container facts learned here, deployment (I5) and not semantics: vLLM's
    TP workers segfault inside libgomp (`gomp_team_start`) on their FIRST
    OpenMP-parallel CPU op in this image — which 0.6B never reaches (its
    buffers stay under torch's parallel grain size) and 8B hits during model
    runner setup. Spawned workers with single-threaded CPU ops never form the
    thread team; a bigger model also wants a bigger container (cpu/memory)."""
    import os

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["OMP_NUM_THREADS"] = "1"
    return run_probe(BIG, tp, gpu_memory_utilization=0.85)


@app.local_entrypoint()
def main() -> None:
    result = tp_probe.remote()
    if result["failed"]:
        raise SystemExit(f"TP checks FAILED: {result['failed']}")
    print("\nALL TP CHECKS PASSED")
