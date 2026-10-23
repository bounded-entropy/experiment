"""Shared allocation, autograd, checkpointing and replay for custom site math.

A program sees flat current-token rows. It never routes tenants or loads
weights itself. Replay calls the base once, then replaces only its own rows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from safetensors.torch import load as load_tensors

from rlstack.policy.adapters.definition import Dim, Init, Input, ResolvedAdapter, ProgramManifest
from rlstack.policy.adapters.replay import SiteWrapper, join_site, leave_site
from rlstack.policy.adapters.state_torch import TensorState
from rlstack.policy.state import AdapterState


@dataclass(frozen=True)
class SiteTensors:
    input: torch.Tensor | None
    output: torch.Tensor


@dataclass(frozen=True)
class TokenRows:
    """Current query rows; request is a batch index, not a persistent ID."""
    position: torch.Tensor
    request: torch.Tensor
    active: torch.Tensor
    prefix_length: torch.Tensor
    slot: torch.Tensor


@dataclass
class ProgramState(AdapterState):
    resolved: ResolvedAdapter
    values: dict[str, tuple[torch.nn.Parameter, ...]]

    def parameters(self):
        return [value for values in self.values.values() for value in values
                if value.requires_grad]

    def export_state(self, version: int) -> TensorState:
        definition = self.resolved.definition
        return TensorState({f"{path}::{spec.name}": value.detach()
                            for path, values in self.values.items()
                            for spec, value in zip(definition.parameters, values)},
                           metadata=ProgramManifest(definition.source_hash(),
                               tuple(s.path for s in self.resolved.sites),
                               {key: value for key, value in self.resolved.settings.items()
                                if key != "seed"}).metadata())


def build(resolved: ResolvedAdapter) -> ProgramState:
    definition = resolved.definition
    values = {}
    for site in resolved.sites:
        if site.shape is None:
            width = definition.program.width or resolved.settings.get("d")
            if width is None:
                raise ValueError("a residual program must declare its width")
            dims = {Dim.INPUT: width, Dim.OUTPUT: width}
        else:
            dims = {Dim.INPUT: site.shape[0], Dim.OUTPUT: site.shape[1]}
        tensors = []
        for spec in definition.parameters:
            shape = tuple(dims[d] if isinstance(d, Dim) else d for d in spec.shape)
            dtype = vars(torch)[spec.dtype]
            tensor = torch.empty(shape, dtype=dtype)
            if spec.init == Init.NORMAL:
                seed = hashlib.sha256(
                    f"{resolved.settings.get('seed', 0)}:{site.path}:{spec.name}".encode()).digest()
                generator = torch.Generator().manual_seed(int.from_bytes(seed[:8], "big"))
                tensor.normal_(std=spec.std, generator=generator)
            elif spec.init == Init.ONES:
                tensor.fill_(1)
            else:
                tensor.zero_()
            tensors.append(torch.nn.Parameter(tensor, requires_grad=spec.trainable))
        values[site.path] = tuple(tensors)
    return ProgramState(resolved, values)


def checked_output(result: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    if result.shape != output.shape or result.dtype != output.dtype or result.device != output.device:
        raise ValueError("site program must preserve output shape, dtype and device")
    return result


class _ProgramAutograd(torch.autograd.Function):
    """Execute the GPU forward, recompute its declared local reference for VJP."""

    @staticmethod
    def forward(ctx, program, tokens, x, output, *parameters):
        ctx.program, ctx.tokens = program, tokens
        ctx.has_input = x is not None
        ctx.save_for_backward(output, *(tuple([x]) if x is not None else ()), *parameters)
        return checked_output(program.forward.load()(SiteTensors(x, output), parameters, tokens), output)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        output = saved[0].detach().requires_grad_(True)
        start = 2 if ctx.has_input else 1
        x = saved[1].detach().requires_grad_(True) if ctx.has_input else None
        parameters = tuple(p.detach().requires_grad_(p.is_floating_point()) for p in saved[start:])
        inputs = (output,) + (() if x is None else (x,)) + parameters
        with torch.enable_grad():
            result = ctx.program.reference.load()(SiteTensors(x, output), parameters, ctx.tokens)
            grads = (torch.autograd.grad(result, inputs, grad_output, allow_unused=True)
                     if result.requires_grad else (None,) * len(inputs))
        output_grad = grads[0]
        input_grad = grads[1] if ctx.has_input else None
        return (None, None, input_grad, output_grad, *grads[start:])


def execute(program, tensors: SiteTensors, parameters: tuple, tokens: TokenRows):
    if torch.is_grad_enabled() and any(t.requires_grad for t in (
            tensors.output, *(() if tensors.input is None else (tensors.input,)), *parameters)):
        if program.reference is None:
            raise ValueError("training a custom site program requires a differentiable reference")
        if tensors.output.is_cuda:
            return _ProgramAutograd.apply(program, tokens, tensors.input, tensors.output, *parameters)
    implementation = (program.reference if not tensors.output.is_cuda and program.reference is not None
                      else program.forward)
    return checked_output(implementation.load()(tensors, parameters, tokens), tensors.output)


class ProgramSite(SiteWrapper):
    def forward(self, *args, **kwargs):
        original = self.inner(*args, **kwargs)
        # Learner boundary modules return the logical stream. Engine fused
        # residual pairs are normalized by the engine's site bridge instead.
        output = original[0] if isinstance(original, tuple) else original
        if output.ndim != 3:
            raise ValueError("program replay expects [rows, tokens, width]")
        rows = self.plan.rows
        if output.shape[0] != rows.index.numel():
            raise ValueError("program row plan and activation rows disagree")
        result = output
        for slot_index, slot in enumerate(rows.slots):
            state = slot.get(self.path)
            if not isinstance(state, ProgramState) or not any(state is s for s in self.installed):
                continue
            program = state.resolved.definition.program
            selected = torch.nonzero(rows.index == slot_index, as_tuple=True)[0]
            if selected.numel() == 0:
                continue
            chosen = output.index_select(0, selected)
            width, tokens = chosen.shape[-1], chosen.shape[1]
            x = args[0].index_select(0, selected).flatten(0, 1) if Input.INPUT in program.inputs else None
            metadata = TokenRows(
                torch.arange(tokens, device=output.device).repeat(selected.numel()),
                selected.repeat_interleave(tokens),
                torch.ones(selected.numel() * tokens, device=output.device, dtype=torch.bool),
                torch.zeros(selected.numel() * tokens, device=output.device, dtype=torch.long),
                torch.zeros(selected.numel() * tokens, device=output.device, dtype=torch.long))
            replacement = execute(program, SiteTensors(x, chosen.reshape(-1, width)),
                                  tuple(p.unsqueeze(0) for p in state.values[self.path]), metadata).reshape_as(chosen)
            result = result.index_copy(0, selected, replacement)
        return (result, *original[1:]) if isinstance(original, tuple) else result


def install(model, state: ProgramState) -> None:
    device = next(model.parameters()).device
    for values in state.values.values():
        for parameter in values:
            parameter.data = parameter.data.to(device)
    for site in state.resolved.sites:
        join_site(model, site.path, ProgramSite, state)


def uninstall(model, state: ProgramState) -> None:
    for site in state.resolved.sites:
        leave_site(model, site.path, ProgramSite, state)


def emit(state: ProgramState) -> bytes:
    return state.export_state(0).encode()


def load(state: ProgramState, payload: bytes) -> None:
    tensors = load_tensors(payload)
    specs = state.resolved.definition.parameters
    expected = {f"{path}::{spec.name}": p for path, values in state.values.items()
                for spec, p in zip(specs, values)}
    if tensors.keys() != expected.keys():
        raise ValueError("site checkpoint parameter names disagree")
    for name, value in tensors.items():
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
            raise ValueError(f"site checkpoint layout mismatch for {name}")
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(tensors[name])
