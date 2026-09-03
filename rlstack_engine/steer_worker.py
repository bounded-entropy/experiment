"""SteerWorker: vLLM's V1 GPU worker with the residual hook installed at
model load — the seam the steer's rollout lowering demands by string
(`worker_cls`, ADR 0004 Q5).

Everything vLLM-specific about the RESIDUAL mechanism is in this file and in
BatchView.from_vllm; SteerPlugin (steer.py) never sees the engine. At
load_model the worker PROBES (I7): every seam the plugin stands on is checked
on the live objects and a missing one is a ProbeError at boot, never a
silent no-steer mid-run. Then the seam is claimed: a pre-hook on the model
builds this forward's routing once, and a hook on every residual boundary
— each decoder layer's output and the final norm's — adds each token's
vector inside its window.

vLLM's decoder layers return the FUSED pair (hidden, residual) whose sum is
the stream and whose next norm sums them; the add lands on `hidden`
(ADR 0004, Q1). The final norm returns the normed stream first. Both are
added to in place.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import vllm
from vllm.v1.worker.gpu_worker import Worker

from rlstack.policy.adapters.steer import STEER_FILE

from rlstack_engine.batch_view import BatchView
from rlstack_engine.plugin import EngineBuild, EnginePlugin, Seam
from rlstack_engine.steer import SteerPlugin, SteerRouting

LAYER = re.compile(r"model\.layers\.\d+")     # a decoder layer's output boundary
FINAL_NORM = "model.norm"                      # the final norm's


def live_symbols(worker: Any) -> frozenset[str]:
    """The seams this build actually has, checked on the LIVE objects — the
    probe's inventory, so a refusal names what is really missing."""
    found = set()
    try:
        from vllm.forward_context import get_forward_context  # noqa: F401
        found.add("vllm.forward_context.get_forward_context")
    except ImportError:
        pass
    if callable(getattr(Worker, "load_model", None)):
        found.add("vllm.v1.worker.gpu_worker.Worker.load_model")
    runner = getattr(worker, "model_runner", None)
    if hasattr(runner, "input_batch") and hasattr(runner.input_batch, "req_ids"):
        found.add("vllm.v1.worker.gpu_model_runner.GPUModelRunner.input_batch")
    if isinstance(getattr(runner, "requests", None), dict):
        found.add("vllm.v1.worker.gpu_model_runner.GPUModelRunner.requests")
    if hasattr(vllm.SamplingParams(), "extra_args"):
        found.add("vllm.sampling_params.SamplingParams.extra_args")
    return frozenset(found)


def boundaries_of(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """The residual boundaries by their SITE PATH — vLLM's module names for a
    dense causal LM coincide with the checkpoint's (`model.layers.<n>`,
    `model.norm`), which is what makes the site schema's paths addressable
    here without a translation table."""
    return {name: module for name, module in model.named_modules()
            if LAYER.fullmatch(name) or name == FINAL_NORM}


class ModelSeam(Seam):
    """The hooks, claimed once per worker."""

    def __init__(self, worker: "SteerWorker") -> None:
        self.worker = worker
        self.claimed: EnginePlugin | None = None

    def claim(self, plugin: EnginePlugin) -> None:
        if self.claimed is not None:
            raise RuntimeError(
                f"{type(self.claimed).__name__} already claimed this worker's "
                f"model; one plugin per seam")
        self.claimed = plugin
        model = self.worker.model_runner.model
        model.register_forward_pre_hook(self.worker.route_forward)
        for path, module in boundaries_of(model).items():
            module.register_forward_hook(self.worker.add_at(path))


class SteerWorker(Worker):
    """vLLM's GPU worker, plus the residual hook."""

    def load_model(self, *args, **kwargs) -> None:
        super().load_model(*args, **kwargs)
        self.steer = SteerPlugin(
            max_slots=int(self.vllm_config.scheduler_config.max_num_seqs),
            device=self.device, dtype=self.model_config.dtype)
        self.steer.probe(EngineBuild(
            fingerprint=f"vllm:{vllm.__version__}",
            attention_backend=str(getattr(
                self.vllm_config.attention_config, "backend", "unknown")),
            symbols=live_symbols(self)))
        if not boundaries_of(self.model_runner.model):
            raise RuntimeError(
                f"no residual boundary named model.layers.<n> or model.norm "
                f"in {type(self.model_runner.model).__name__}: the steer hook "
                f"has nowhere to stand on this model family")
        self.routing: SteerRouting | None = None
        self.steer.install(ModelSeam(self))

    # ---- the hooks -----------------------------------------------------------

    def route_forward(self, module, args) -> None:
        """Once per forward, before the first layer: the batch's view and
        this forward's routing. No view (a warm-up run) routes nothing."""
        view = BatchView.from_vllm(self.model_runner, 0, self.steer.slot_of)
        self.routing = None if view is None else self.steer.routing(view)

    def add_at(self, path: str):
        def hook(module, args, output) -> None:
            if self.routing is None or not self.routing.steers:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            self.steer.add(self.routing, path, hidden)
            return None
        return hook
