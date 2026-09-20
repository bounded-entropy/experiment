"""Opt-in wire integration fixture: real CUDA learner and optional vLLM engine.

This is a protocol test server, not an experiment runner. Experiments must still
be submitted through the desk. It exits after 15 minutes and never leases GPUs.
Run with the experiment image's Python; serve only behind authenticated SSH.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import tempfile


async def serve(port: int, base: str | None) -> None:
    import torch
    from rlstack import Host, HostService, LocalStore
    from rlstack.runner.learners.torch_learner import TorchLearner
    from rlstack.runner.transports.http import HttpServer

    if not torch.cuda.is_available():
        raise RuntimeError("This opt-in probe requires a real CUDA GPU")

    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(23, 8)
            self.head = torch.nn.Linear(8, 23, bias=False)

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            from types import SimpleNamespace
            return SimpleNamespace(logits=self.head(self.embed(input_ids)))

    torch.manual_seed(7)
    learner = TorchLearner(device="cuda", dtype=torch.float32, checkpoint_activations=False)
    learner._model = TinyLM().cuda().eval()
    learner._model.requires_grad_(False)
    learner._base = "http-probe-tiny"
    engines = ()
    if base:
        from rlstack.runner.engines.vllm_engine import VllmEngine
        engines = (VllmEngine(base, gpu_memory_utilization=.2,
                              max_model_len=256, serves=(), enforce_eager=True),)
    with tempfile.TemporaryDirectory() as directory:
        host = Host("http-probe", engines=engines, learner=learner, store=LocalStore(directory))
        service = HostService(host)
        async with HttpServer(lambda name: service, port=port) as server:
            print(json.dumps({"ready": True, "pid": os.getpid(), "endpoint": server.endpoint,
                              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                              "base": base}), flush=True)
            await asyncio.sleep(900)


def client(endpoint: str, base: str | None) -> dict:
    import math
    from rlstack import Bundle, Message, Role, SamplingSpec
    from rlstack.data.flatten import TokenBatch
    from rlstack.policy.siteschema import SiteMeta
    from rlstack.runner.interfaces import EntryInstall, OptimSettings, Parameterization
    from rlstack.runner.remote import RemoteLearner, RemotePool, transport_for

    learner = RemoteLearner(transport_for(endpoint), admitted=True)
    parameterization = Parameterization(
        base="http-probe-tiny", loss="sft",
        entries=(EntryInstall("pi", "lora", {"r": 2, "seed": 5}, True,
                             (SiteMeta("head", "head", True, (8, 23), False),)),),
        optim=OptimSettings("adamw", .01, (.9, .95), 0., {}))
    for tenant in ("a", "b"):
        learner.install(tenant, parameterization)
    before_a, before_b = learner.emit("a"), learner.emit("b")
    batch = TokenBatch(token_ids=(1, 2, 3, 4), loss_mask=(0, 1, 1, 1),
                       behavior_logprobs=(0.,) * 4, segment_ids=(0,) * 4, doc_starts=(0,))
    losses = []
    for _ in range(3):
        stats = learner.forward_backward("a", batch)
        assert math.isfinite(stats.loss) and stats.grad_norm > 0, stats
        losses.append(stats.loss)
        learner.optim_step("a")
    after_a, after_b = learner.emit("a"), learner.emit("b")
    assert before_a.adapters != after_a.adapters, "optimizer did not update tenant a"
    assert before_b.adapters == after_b.adapters, "tenant a changed tenant b"
    assert before_a.optim != after_a.optim, "optimizer state did not persist"
    # A new client connection must see the same resident state.
    reconnected = RemoteLearner(transport_for(endpoint), admitted=True)
    assert reconnected.emit("a").adapters == after_a.adapters
    result = {"cuda_learner_steps": 3, "losses": losses, "tenant_isolation": True,
              "reconnected_state": True, "optimizer_state": True}
    if base:
        pool = RemotePool(transport_for(endpoint), base=base)
        pool.add_bundle(Bundle("http-probe-base", {}))
        messages = (Message(Role.USER, "The capital of France is"),)

        async def infer():
            tokens = await asyncio.to_thread(pool.tokenize, " Paris")
            first, second = await asyncio.gather(
                pool.score_tokens(messages, tokens, "http-probe-base"),
                pool.score_tokens(messages, tokens, "http-probe-base"))
            assert first == second and len(first) == len(tokens) and all(math.isfinite(x) for x in first)
            events = [event async for event in pool.sample_tokens(
                messages, SamplingSpec(max_tokens=8), (), "http-probe-base", seed=7)]
            return {"scored_tokens": len(first), "sample_events": len(events), "repeat_scores": True}
        result["vllm"] = asyncio.run(infer())
    for tenant in ("a", "b"):
        learner.uninstall(tenant)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "client"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18000")
    parser.add_argument("--base", help="Optional vLLM model, e.g. Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    if args.mode == "serve":
        asyncio.run(serve(args.port, args.base))
    else:
        print(json.dumps(client(args.endpoint, args.base)), flush=True)
