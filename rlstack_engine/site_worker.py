"""Pinned vLLM 0.28 V1 bridge for declared residual/linear site programs.

Routing is prepared after runner._prepare_inputs, not in a model pre-hook.
Dummy runs execute the real kernels on inactive rows before graph capture.
Native LoRA remains owned by vLLM. TP/speculation/ubatching are refused here
until their state/layout contracts receive a live qualification.
"""

from __future__ import annotations

import importlib
import json
import os
import re

import torch
import vllm
from vllm.v1.worker.gpu_worker import Worker
from vllm.model_executor.layers.linear import LinearBase
from vllm.lora.layers import BaseLayerWithLoRA

from rlstack.policy.adapters.definition import SiteKind, ProgramManifest
from rlstack.registry import ADAPTER_TYPES
from rlstack_engine.site_program import SiteRuntime
from rlstack_engine.site_layout import projection_slices


class SiteWorker(Worker):
    site_runtimes = None

    def determine_available_memory(self):
        # Install before vLLM profiles activations/capture and sizes its KV
        # cache. Otherwise these persistent banks arrive after the GPU budget
        # was already consumed by base weights and cache.
        before = torch.cuda.memory_allocated(self.device)
        self.install_site_programs()
        self.site_bank_bytes = torch.cuda.memory_allocated(self.device) - before
        self.model_runner.model_memory_usage += self.site_bank_bytes
        return super().determine_available_memory()

    def compile_or_warm_up_model(self, *args, **kwargs):
        if self.site_runtimes is None:
            raise RuntimeError("site programs must be installed before memory profiling")
        return super().compile_or_warm_up_model(*args, **kwargs)

    def install_site_programs(self):
        if self.site_runtimes is not None:
            raise RuntimeError("site programs were already installed")
        if not vllm.__version__.startswith("0.28."):
            raise RuntimeError("site worker is pinned to vLLM 0.28 V1")
        if self.vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("custom site worker TP>1 is not qualified")
        if (self.vllm_config.parallel_config.pipeline_parallel_size != 1
                or self.vllm_config.parallel_config.data_parallel_size != 1):
            raise ValueError("custom site worker requires one model-parallel replica")
        if self.vllm_config.speculative_config is not None:
            raise ValueError("custom site worker speculative decoding is not qualified")
        if self.vllm_config.parallel_config.use_ubatching:
            raise ValueError("custom site worker requires one serialized forward lane")
        for module in json.loads(os.environ["RLSTACK_SITE_MODULES"]):
            importlib.import_module(module)
        runner = self.model_runner
        self.site_runtimes = {}
        capacity = int(self.vllm_config.scheduler_config.max_num_batched_tokens)
        for name in json.loads(os.environ["RLSTACK_SITE_TYPES"]):
            definition = ADAPTER_TYPES.get(name).instance.definition
            runtime = SiteRuntime(definition, int(os.environ["RLSTACK_SITE_SLOTS"]), capacity, self.device)
            count = 0
            wrapped = {path for path, module in runner.model.named_modules()
                       if isinstance(module, BaseLayerWithLoRA)}
            for path, module in runner.model.named_modules():
                if definition.site == SiteKind.RESIDUAL:
                    if not re.fullmatch(r"model\.layers\.\d+|model\.norm", path):
                        continue
                    width = int(self.model_config.get_hidden_size())
                    if definition.program.width is not None and definition.program.width != width:
                        raise ValueError(f"{name}: declared residual width differs from model")
                    runtime.add_site(path, width, width)
                    module.register_forward_hook(self.site_hook(runtime, path), with_kwargs=True)
                    count += 1
                elif isinstance(module, (LinearBase, BaseLayerWithLoRA)):
                    if any(path.startswith(parent + ".") for parent in wrapped):
                        continue  # the wrapper's output, never its base twice
                    base = module.base_layer if isinstance(module, BaseLayerWithLoRA) else module
                    if not isinstance(base, LinearBase) or base.skip_bias_add:
                        continue
                    if base.weight.ndim != 2 or base.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                        continue
                    output_width, input_width = base.weight.shape
                    sizes = (tuple(base.output_sizes) if path.endswith((".qkv_proj", ".gate_up_proj"))
                             else ())
                    views = projection_slices(path, output_width, sizes)
                    for view in views:
                        runtime.add_site(view.path, input_width, view.width)
                    module.register_forward_hook(self.linear_hook(runtime, views), with_kwargs=True)
                    count += 1
            if not count:
                raise ValueError(f"{name}: no supported model sites on this build")
            self.site_runtimes[name] = runtime
        original_prepare = runner._prepare_inputs
        original_dummy = runner._dummy_run

        def prepare(scheduler_output, num_scheduled_tokens):
            result = original_prepare(scheduler_output, num_scheduled_tokens)
            n = scheduler_output.total_num_scheduled_tokens
            count = runner.input_batch.num_reqs
            req_ids = runner.input_batch.req_ids[:count]
            indices = runner.req_indices.gpu[:n].long()
            prefix = runner.num_computed_tokens[indices]
            for name, runtime in self.site_runtimes.items():
                bundles = [(runner.requests[rid].sampling_params.extra_args or {}).get(
                    f"rlstack_site:{name}") for rid in req_ids]
                runtime.prepare(bundles, indices, runner.positions[:n], prefix)
            return result

        def dummy(*dummy_args, **dummy_kwargs):
            for runtime in self.site_runtimes.values():
                runtime.inactive()
            return original_dummy(*dummy_args, **dummy_kwargs)

        runner._prepare_inputs = prepare
        runner._dummy_run = dummy

    @staticmethod
    def linear_hook(runtime, views):
        def hook(module, args, kwargs, original):
            output = original[0] if isinstance(original, tuple) else original
            x = args[0] if args else kwargs["input_"]
            result = output
            for view in views:
                part = result[:, view.start:view.start + view.width]
                changed = runtime.apply(view.path, x, part)
                result = torch.cat((result[:, :view.start], changed,
                                    result[:, view.start + view.width:]), dim=-1)
            return (result, *original[1:]) if isinstance(original, tuple) else result
        return hook

    @staticmethod
    def site_hook(runtime, path):
        def hook(module, args, kwargs, original):
            output = original[0] if isinstance(original, tuple) else original
            residual = (original[1] if isinstance(original, tuple) and len(original) > 1
                        and runtime.definition.site == SiteKind.RESIDUAL else None)
            x = args[0] if args else kwargs.get("hidden_states")
            result = runtime.apply(path, x, output, residual)
            return (result, *original[1:]) if isinstance(original, tuple) else result
        return hook

    def load_site_state(self, name, bundle_id, states):
        runtime = self.site_runtimes[name]
        expected = runtime.definition.source_hash()
        if any(ProgramManifest.from_metadata(metadata).program_hash != expected for metadata, _ in states):
            raise ValueError("site worker program identity disagrees with publisher")
        runtime.load(bundle_id, [(metadata, {key: handle.tensor() for key, handle in tensors.items()})
                                 for metadata, tensors in states])
        return {"adapter_type": name, "bundle_id": bundle_id, "ready": True}

    def release_site_state(self, name, bundle_id):
        self.site_runtimes[name].release(bundle_id)
