"""Shared-definition migration, custom gradients, version routing and capture.

CPU cases exercise real tensor math and IPC. CUDA cases are deliberately
skipped without CUDA/Triton rather than treating fakes as graph evidence.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import patch

from rlstack.policy.adapters.base import AdapterType, Mechanism
from rlstack.policy.adapters.definition import (
    AdapterDefinition, Dim, Input, Operation, Parameter, SiteKind, SiteProgram, Symbol,
)
from rlstack.policy.adapters.lowering import compile_replay, compile_rollout
from rlstack.policy.adapters.rollout import ServingBuild
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from rlstack.policy.adapters.replay import ReplayRows, row_plan
    from rlstack.policy.adapters.site_program_torch import (
        SiteTensors, TokenRows, _ProgramAutograd, emit, load,
    )
    from rlstack.policy.adapters.site_program_vllm import TensorHandle
    from rlstack_engine.site_program import SiteRuntime


LINEAR = SiteMeta("block.proj", "block.proj", True, (4, 3), False)
BOUNDARY = SiteMeta("resid_pre.0", "block", False, None, True)


def affine(tensors, parameters, tokens):
    return tensors.output * parameters[0][tokens.slot] + tensors.input[:, :3]


class DefinitionTests(unittest.TestCase):
    def test_every_builtin_uses_both_common_lowering_entry_points(self):
        for name in ADAPTER_TYPES.names():
            adapter = ADAPTER_TYPES.get(name).instance
            if not type(adapter).__module__.startswith("rlstack.policy.adapters."):
                continue
            self.assertIsInstance(adapter.definition, AdapterDefinition, name)
            self.assertIs(type(adapter).params, AdapterType.params, name)
            self.assertIs(type(adapter).install_replay, AdapterType.install_replay, name)
            self.assertIs(type(adapter).rollout_lowering, AdapterType.rollout_lowering, name)

    def test_native_replay_and_rollout_receive_the_same_definition(self):
        adapter = ADAPTER_TYPES.get("lora").instance
        replay = compile_replay(adapter.definition)
        with tempfile.TemporaryDirectory() as root:
            build = ServingBuild("base", SimpleNamespace(), Path(root), 2, 16)
            fake = SimpleNamespace(mechanism=Mechanism.PUNICA)
            with patch.object(Symbol, "load", return_value=lambda build: fake):
                rollout = compile_rollout(adapter.definition, build, "lora")
        self.assertIs(replay.definition, rollout.definition)
        self.assertEqual(rollout.definition.operation, Operation.LOW_RANK)
        self.assertEqual(rollout.adapter_type, "lora")

    def test_client_import_does_not_load_gpu_packages(self):
        subprocess.run([sys.executable, "-c", (
            "import sys; import rlstack; "
            "assert not {'torch','vllm','triton'} & sys.modules.keys()")], check=True)

    def test_invalid_state_and_unimplemented_scaling_are_refused(self):
        with self.assertRaises(ValueError):
            Parameter("bad", (0,))
        definition = ADAPTER_TYPES.get("site_steer").instance.definition
        with self.assertRaises(ValueError):
            replace(definition, parameters=(Parameter("x", (2,)), Parameter("x", (2,))))
        with self.assertRaises(ValueError):
            replace(definition, scale=2)
        with self.assertRaises(ValueError):
            definition.resolve((LINEAR,), {})

    def test_reference_selection_is_part_of_code_identity(self):
        definition = ADAPTER_TYPES.get("site_steer").instance.definition
        alternative = replace(definition, program=replace(definition.program,
            reference=replace(definition.program.reference, name="low_rank_reference")))
        self.assertNotEqual(definition.source_hash(), alternative.source_hash())

    def test_native_and_custom_reachability_coexist_and_keep_the_legacy_wire_shape(self):
        from rlstack.policy.adapters.base import encode_reachability, decode_reachability
        from rlstack.runner.engines.vllm_engine import VllmEngine
        engine = object.__new__(VllmEngine)
        engine._lowerings = {
            "native": SimpleNamespace(mechanism=Mechanism.PUNICA, reaches=lambda site: True),
            "custom": SimpleNamespace(mechanism=Mechanism.SITE, reaches=lambda site: True)}
        reach = engine.reachability((LINEAR,))
        self.assertEqual(reach[LINEAR.name], frozenset((Mechanism.PUNICA, Mechanism.SITE)))
        self.assertEqual(decode_reachability(json.loads(json.dumps(encode_reachability(reach)))), reach)
        self.assertEqual(encode_reachability({LINEAR.name: Mechanism.PUNICA}),
                         {LINEAR.name: "PUNICA"})
        from rlstack import AdapterSpec
        from rlstack.policy.siteschema import SiteSchema
        from rlstack.spec.validate import check_sites_reachable_on
        for name in ("lora", "site_lora16"):
            spec = SimpleNamespace(policy=SimpleNamespace(bank={
                "entry": AdapterSpec(name, LINEAR.name)}))
            self.assertEqual(check_sites_reachable_on(
                spec, SiteSchema("base", (LINEAR,)), "main", reach), [])


@unittest.skipUnless(torch is not None, "torch is a worker dependency")
class ProgramTensorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.adapter = ADAPTER_TYPES.get("site_lora16").instance
        self.state = self.adapter.initial_params((LINEAR,), {"seed": 3})
        with torch.no_grad():
            self.state.values[LINEAR.path][1].normal_()

    def test_mixed_tenants_share_one_base_call_and_gradients_stay_isolated(self):
        model = torch.nn.Module()
        model.block = torch.nn.Module()
        linear = torch.nn.Linear(4, 3, bias=False)
        model.block.proj = linear
        other = self.adapter.initial_params((LINEAR,), {"seed": 5})
        for state in (self.state, other):
            self.adapter.install_replay(model, state, (LINEAR,))
        x = torch.randn(3, 2, 4, requires_grad=True)
        rows = ReplayRows(({LINEAR.path: self.state}, {LINEAR.path: other}, {}),
                          torch.tensor([0, 1, 2]))
        with patch.object(linear, "forward", wraps=linear.forward) as base:
            with row_plan(model).route(rows):
                out = model.block.proj(x)
                expected = linear(x)
                a, b = self.state.values[LINEAR.path]
                expected = expected.clone()
                expected[0] += (x[0] @ a.T) @ b.T
                torch.testing.assert_close(out, expected)
                self.assertEqual(base.call_count, 2)  # one real call plus the reference above
                out[0].sum().backward()
        self.assertGreater(self.state.parameters()[1].grad.abs().sum().item(), 0)
        other_grad = other.parameters()[0].grad
        if other_grad is not None:
            torch.testing.assert_close(other_grad, torch.zeros_like(other_grad))
        self.assertGreater(x.grad[0].abs().sum().item(), 0)
        torch.testing.assert_close(x.grad[1:], torch.zeros_like(x.grad[1:]))
        self.adapter.uninstall_replay(model, self.state, (LINEAR,))
        self.adapter.uninstall_replay(model, other, (LINEAR,))
        self.assertIs(model.block.proj, linear)

    def test_payload_and_shared_tensor_state_round_trip(self):
        payload = self.adapter.emit(self.state)
        other = self.adapter.initial_params((LINEAR,), {"seed": 999})
        self.adapter.load(other, payload)
        self.assertEqual(self.adapter.emit(other), payload)
        from rlstack.policy.state import payload_layout
        exported = self.adapter.export_state(self.state, 9)
        self.assertEqual(exported.layout(), payload_layout(exported.encode()))

    def test_mismatched_checkpoint_is_validated_before_mutation(self):
        wrong = replace(self.adapter.definition, parameters=(Parameter("A", (2,)),))
        state = compile_replay(wrong).build(wrong.resolve((LINEAR,), {}))
        before = emit(state)
        with self.assertRaises(ValueError):
            load(state, emit(self.state))
        self.assertEqual(emit(state), before)

    def test_explicit_autograd_bridge_propagates_input_output_and_parameter_derivatives(self):
        program = SiteProgram(Symbol(__name__, "affine"), Symbol(__name__, "affine"),
                              inputs=(Input.INPUT, Input.OUTPUT))
        x = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
        output = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        bank = torch.randn(1, 3, dtype=torch.double, requires_grad=True)
        rows = TokenRows(torch.arange(2), torch.zeros(2, dtype=torch.long),
                         torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.long),
                         torch.zeros(2, dtype=torch.long))
        self.assertTrue(torch.autograd.gradcheck(
            lambda x, y, p: _ProgramAutograd.apply(program, rows, x, y, p), (x, output, bank)))

    def test_tensor_ipc_handles_preserve_scalar_and_matrix_layouts(self):
        for tensor in (torch.tensor(3.), torch.randn(2, 4), torch.empty(0)):
            handle = TensorHandle.of(tensor)
            received = pickle.loads(pickle.dumps(handle)).tensor()
            torch.testing.assert_close(received, tensor)
            if tensor.numel():
                received.fill_(11)
                torch.testing.assert_close(received, tensor)

    def runtime(self, kind="site_steer", site=BOUNDARY):
        adapter = ADAPTER_TYPES.get(kind).instance
        runtime = SiteRuntime(adapter.definition, 2, 8, "cpu")
        runtime.add_site(site.path, 4, 4 if site is BOUNDARY else 3)
        return adapter, runtime

    def test_runtime_routes_versions_new_tokens_and_base_only_rows(self):
        adapter, runtime = self.runtime()
        for name, amount in (("a", 1.), ("b", 3.)):
            state = adapter.initial_params((BOUNDARY,), {"d": 4})
            with torch.no_grad():
                state.values["block"][0].fill_(amount)
            exported = state.export_state(0)
            runtime.load(name, [(exported.metadata, exported.tensors)])
        addresses = (runtime.routing.data_ptr(), runtime.banks["block"][0].data_ptr())
        runtime.prepare(["a", "b", None], torch.tensor([0, 1, 2]),
                        torch.tensor([100, 70, 30]), torch.tensor([100, 70, 30]))
        out = runtime.apply("block", None, torch.zeros(3, 4))
        torch.testing.assert_close(out, torch.tensor([[1.]*4, [3.]*4, [0.]*4]))
        # Next decode processes just one token; buffers retain their addresses.
        runtime.prepare(["b"], torch.tensor([0]), torch.tensor([71]), torch.tensor([71]))
        torch.testing.assert_close(runtime.apply("block", None, torch.zeros(1, 4)), torch.full((1, 4), 3.))
        self.assertEqual(addresses, (runtime.routing.data_ptr(), runtime.banks["block"][0].data_ptr()))
        runtime.inactive()
        original = torch.randn(4, 4)
        residual = torch.randn(4, 4) * 1e5
        self.assertTrue(torch.equal(runtime.apply("block", None, original, residual), original))

    def test_failed_load_does_not_publish_or_overwrite_existing_version(self):
        adapter, runtime = self.runtime()
        state = adapter.initial_params((BOUNDARY,), {"d": 4}).export_state(0)
        runtime.load("a", [(state.metadata, state.tensors)])
        with self.assertRaises(ValueError):
            runtime.load("b", [(state.metadata, {"block::direction": torch.zeros(3)})])
        self.assertEqual(runtime.slots.resident(), {"a": 0})
        runtime.release("a")
        runtime.load("b", [(state.metadata, state.tensors)])
        self.assertEqual(runtime.slots.resident(), {"b": 0})

    def test_rollout_uses_tensor_installation_and_requires_exact_worker_ack(self):
        adapter = ADAPTER_TYPES.get("site_steer").instance
        with tempfile.TemporaryDirectory() as root:
            build = ServingBuild("base", SimpleNamespace(), Path(root), 2, 16)
            rollout = adapter.rollout_lowering(build)
            state = adapter.initial_params((BOUNDARY,), {"d": 4}).export_state(0)
            with patch.object(type(state), "encode", side_effect=AssertionError("must not serialize")):
                attached = rollout.attach_state("a", {"entry": state})
            class Engine:
                async def collective_rpc(self, method, timeout, args):
                    return [{"adapter_type": "site_steer", "bundle_id": "WRONG", "ready": True}]
            with self.assertRaises(RuntimeError):
                asyncio.run(rollout.prepare(Engine(), attached))
            self.assertNotIn("a", rollout._prepared)

    def test_cancelled_installation_is_retired_before_slot_reuse(self):
        adapter = ADAPTER_TYPES.get("site_steer").instance
        with tempfile.TemporaryDirectory() as root:
            build = ServingBuild("base", SimpleNamespace(), Path(root), 2, 16)
            rollout = adapter.rollout_lowering(build)
            state = adapter.initial_params((BOUNDARY,), {"d": 4}).export_state(0)
            attached = rollout.attach_state("a", {"entry": state})
            # Even without an acknowledgement, the old worker may have loaded it.
            rollout.detach(attached)
            calls = []
            class Engine:
                async def collective_rpc(self, method, timeout, args):
                    calls.append(method)
                    return [{"adapter_type": "site_steer", "bundle_id": "a", "ready": True}]
            asyncio.run(rollout.prepare(Engine(), attached))
            self.assertEqual(calls, ["release_site_state", "load_site_state"])

    def test_two_custom_types_compose_with_one_full_policy_cache_identity(self):
        with tempfile.TemporaryDirectory() as root:
            build = ServingBuild("base", SimpleNamespace(), Path(root), 2, 16,
                                 site_types=("site_steer", "site_lora16"))
            first = ADAPTER_TYPES.get("site_steer").instance.rollout_lowering(build)
            second = self.adapter.rollout_lowering(build)
            state = ADAPTER_TYPES.get("site_steer").instance.initial_params(
                (BOUNDARY,), {"d": 4}).export_state(0)
            one = first.attach_state("policy", {"steer": state})
            two = second.attach_state("policy", {"lora": self.state.export_state(0)})
            levers = first.apply(one, None).merged_with(second.apply(two, None))
            self.assertEqual(levers.cache_salt, "site:policy")
            self.assertEqual(len(levers.extra_args), 2)

    def test_packed_projection_slices_keep_logical_sites_distinct(self):
        from rlstack_engine.site_layout import projection_slices
        sites = projection_slices("model.layers.0.self_attn.qkv_proj", 12, (8, 2, 2))
        self.assertEqual([(s.path.rsplit(".", 1)[1], s.start, s.width) for s in sites],
                         [("qkv_proj", 0, 12), ("q_proj", 0, 8), ("k_proj", 8, 2), ("v_proj", 10, 2)])
        with self.assertRaises(ValueError):
            projection_slices("model.layers.0.self_attn.qkv_proj", 12, (8, 2))

    def test_pinned_runner_bridge_prepares_outside_forward_and_hooks_native_wrapper_once(self):
        import importlib
        import os

        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(12, 4))
                self.skip_bias_add = False
                self.output_sizes = [8, 2, 2]
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                return x @ self.weight.T, None

        class NativeLora(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base_layer = Linear()

            def forward(self, x):
                return self.base_layer(x)

        class Worker:
            def determine_available_memory(self):
                self.model_runner._dummy_run()
                return 123

            def compile_or_warm_up_model(self):
                self.model_runner._dummy_run()

        modules = {}
        for name in ("vllm", "vllm.v1.worker.gpu_worker",
                     "vllm.model_executor.layers.linear", "vllm.lora.layers"):
            modules[name] = ModuleType(name)
        modules["vllm"].__version__ = "0.28.0"
        modules["vllm.v1.worker.gpu_worker"].Worker = Worker
        modules["vllm.model_executor.layers.linear"].LinearBase = Linear
        modules["vllm.lora.layers"].BaseLayerWithLoRA = NativeLora
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        block = torch.nn.Module()
        block.self_attn = torch.nn.Module()
        block.self_attn.qkv_proj = projection = NativeLora()
        model.model.layers = torch.nn.ModuleList([block])
        runner = SimpleNamespace(
            model=model, model_memory_usage=0,
            _prepare_inputs=lambda *args: "prepared",
            _dummy_run=lambda: projection(torch.ones(2, 4)),
            req_indices=SimpleNamespace(gpu=torch.tensor([0, 0])),
            positions=torch.tensor([7, 8]), num_computed_tokens=torch.tensor([7]),
            input_batch=SimpleNamespace(num_reqs=1, req_ids=["reqA"]),
            requests={"reqA": SimpleNamespace(sampling_params=SimpleNamespace(
                extra_args={"rlstack_site:site_lora16": "a"}))})
        environment = {"RLSTACK_SITE_MODULES": "[]", "RLSTACK_SITE_TYPES": '["site_lora16"]',
                       "RLSTACK_SITE_SLOTS": "2"}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, environment):
            sys.modules.pop("rlstack_engine.site_worker", None)
            try:
                module = importlib.import_module("rlstack_engine.site_worker")
                worker = module.SiteWorker()
                worker.device = "cpu"
                worker.model_runner = runner
                worker.vllm_config = SimpleNamespace(
                    parallel_config=SimpleNamespace(tensor_parallel_size=1, use_ubatching=False,
                                                    pipeline_parallel_size=1, data_parallel_size=1),
                    speculative_config=None, scheduler_config=SimpleNamespace(max_num_batched_tokens=8))
                with patch.object(torch.cuda, "memory_allocated", return_value=0):
                    self.assertEqual(worker.determine_available_memory(), 123)
                worker.compile_or_warm_up_model()
                # The native wrapper is hooked, its base is not hooked twice.
                self.assertEqual(len(projection._forward_hooks), 1)
                self.assertEqual(len(projection.base_layer._forward_hooks), 0)
                path = "model.layers.0.self_attn.q_proj"
                site = SiteMeta(path, path, True, (4, 8), False)
                adapter = ADAPTER_TYPES.get("site_lora16").instance
                state = adapter.initial_params((site,), {})
                with torch.no_grad():
                    for parameter in state.parameters():
                        parameter.fill_(1)
                exported = state.export_state(0)
                worker.load_site_state("site_lora16", "a", [(exported.metadata, {
                    key: TensorHandle.of(value) for key, value in exported.tensors.items()})])
                self.assertEqual(runner._prepare_inputs(SimpleNamespace(total_num_scheduled_tokens=2),
                                                       torch.tensor([2])), "prepared")
                before = projection.base_layer.calls
                output, _ = projection(torch.ones(2, 4))
                self.assertEqual(projection.base_layer.calls - before, 1)
                torch.testing.assert_close(output[:, :8], torch.full((2, 8), 64.))
                torch.testing.assert_close(output[:, 8:], torch.zeros(2, 4))
            finally:
                sys.modules.pop("rlstack_engine.site_worker", None)


@unittest.skipUnless(torch is not None and torch.cuda.is_available()
                     and importlib.util.find_spec("triton") is not None,
                     "CUDA graph/Triton qualification requires a CUDA device")
class LiveGraphTests(unittest.TestCase):
    def test_inactive_capture_then_changed_versions_and_routing(self):
        adapter = ADAPTER_TYPES.get("site_steer").instance
        runtime = SiteRuntime(adapter.definition, 2, 8, "cuda")
        runtime.add_site("block", 4, 4)
        for name, amount in (("a", 1.), ("b", 3.)):
            state = adapter.initial_params((BOUNDARY,), {"d": 4})
            with torch.no_grad():
                state.values["block"][0].fill_(amount)
            exported = state.export_state(0)
            runtime.load(name, [(exported.metadata, exported.tensors)])
        x = torch.randn(4, 4, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                runtime.apply("block", None, x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = runtime.apply("block", None, x)
        for name, amount in (("a", 1.), ("b", 3.)):
            ids = torch.zeros(4, dtype=torch.long, device="cuda")
            runtime.prepare([name], ids, ids, ids)
            graph.replay()
            torch.testing.assert_close(result, x + amount)
