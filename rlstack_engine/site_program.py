"""Persistent parameter/routing banks for a declared custom site program.

No engine internals live here. prepare() is outside graph replay; apply()
always executes the same program, including inactive capture batches.
"""

from __future__ import annotations

import torch

from rlstack.policy.adapters.definition import Dim, Input, ProgramManifest
from rlstack.policy.adapters.site_program_torch import SiteTensors, TokenRows, checked_output
from rlstack_engine.slots import SlotTable


class SiteRuntime:
    def __init__(self, definition, max_slots, max_tokens, device):
        self.definition = definition
        self.slots = SlotTable(max_slots)
        self.max_tokens = max_tokens
        self.device = device
        self.banks = {}
        self.present = {}
        self.routing = torch.full((max_tokens,), -1, dtype=torch.long, device=device)
        self.position = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.request = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.prefix = torch.zeros(max_tokens, dtype=torch.long, device=device)
        self.forward = definition.program.forward.load()
        self.reference = (None if definition.program.reference is None
                          else definition.program.reference.load())

    def add_site(self, path, input_width, output_width):
        dims = {Dim.INPUT: input_width, Dim.OUTPUT: output_width}
        self.banks[path] = tuple(torch.zeros(
            (self.slots.max_slots, *(dims[d] if isinstance(d, Dim) else d for d in spec.shape)),
            dtype=vars(torch)[spec.dtype], device=self.device)
            for spec in self.definition.parameters)
        self.present[path] = torch.zeros(self.slots.max_slots, dtype=torch.bool, device=self.device)

    def load(self, bundle_id, states):
        if bundle_id in self.slots.resident():
            return
        incoming = {}
        for _, tensors in states:
            for key, value in tensors.items():
                if key in incoming:
                    raise ValueError(f"duplicate site tensor {key}")
                incoming[key] = value
        paths = {path for metadata, _ in states for path in ProgramManifest.from_metadata(metadata).sites}
        if not paths or not paths <= self.banks.keys():
            raise ValueError(f"publication names unreachable site(s): {paths - self.banks.keys()}")
        expected = {f"{path}::{spec.name}": bank for path in paths
                    for spec, bank in zip(self.definition.parameters, self.banks[path])}
        if incoming.keys() != expected.keys():
            raise ValueError("publication parameter names disagree with site definition")
        for key, bank in expected.items():
            value = incoming[key]
            if value.shape != bank.shape[1:] or value.dtype != bank.dtype:
                raise ValueError(f"publication layout mismatch for {key}")
        slot = self.slots.acquire(bundle_id)
        try:
            for path in self.banks:
                self.present[path][slot] = False
            for path in paths:
                for spec, bank in zip(self.definition.parameters, self.banks[path]):
                    bank[slot].copy_(incoming[f"{path}::{spec.name}"])
                self.present[path][slot] = True
        except BaseException:
            self.slots.release(bundle_id)
            raise

    def release(self, bundle_id):
        if bundle_id not in self.slots.resident():
            return
        slot = self.slots.release(bundle_id)
        for mask in self.present.values():
            mask[slot] = False

    def inactive(self):
        self.routing.fill_(-1)

    def prepare(self, request_bundles, request_indices, positions, prefix_lengths):
        n = positions.numel()
        if n > self.max_tokens or request_indices.numel() != n or prefix_lengths.numel() != n:
            raise ValueError("site routing exceeds capacity or has misaligned columns")
        slots = [self.slots.slot_of(bundle) if bundle is not None else -1
                 for bundle in request_bundles]
        request_slots = torch.tensor(slots, dtype=torch.long, device=self.device)
        self.routing.fill_(-1)
        self.routing[:n].copy_(request_slots[request_indices])
        self.position[:n].copy_(positions)
        self.request[:n].copy_(request_indices)
        self.prefix[:n].copy_(prefix_lengths)

    def apply(self, path, x, output, residual=None):
        n = output.shape[0]
        if n > self.max_tokens:
            raise ValueError("site token rows exceed the captured capacity")
        slot = self.routing[:n].clamp_min(0)
        active = (self.routing[:n] >= 0) & self.present[path][slot]
        logical = output if residual is None else output + residual
        tokens = TokenRows(self.position[:n], self.request[:n], active,
                          self.prefix[:n], slot)
        inputs = SiteTensors(x if Input.INPUT in self.definition.program.inputs else None, logical)
        implementation = self.forward if output.is_cuda or self.reference is None else self.reference
        replacement = checked_output(implementation(inputs, self.banks[path], tokens), logical)
        if residual is not None:
            return output + torch.where(active.unsqueeze(-1), replacement - logical, 0)
        return torch.where(active.unsqueeze(-1), replacement, output)
