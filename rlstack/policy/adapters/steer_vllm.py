"""The steer's rollout lowering: the RESIDUAL lever, a hook in the engine
image (ADR 0004).

demands the worker class that installs the hook and eager mode (a hook does
not fire inside a captured graph, and a build that will not pay refuses this
adapter type at construction rather than serving it wrong); attach fuses the
bank's steer entries into ONE file under the build's workdir — the file is
the address, exactly as lora_vllm's adapter dir is for punica; apply names
that file and the request's resolved window in SamplingParams.extra_args
(what the hook reads per request), salts the prefix cache with the bundle
AND the window (a block prefilled under one is wrong for the other), and
records the window as the turn fact replay reads; align contributes nothing,
because a vector occupies no prompt positions.

torch is imported at module scope — this file loads only from
Steer.rollout_lowering (STYLE rule 7). The worker class is named by STRING
and never imported: rlstack_engine ships in the engine image and the import
direction is one-way.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from rlstack.policy.adapters import steer_torch
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.rollout import (
    Alignment, BuildDemands, Levers, Request, RolloutLowering,
)
from rlstack.policy.adapters.steer import (
    STEER_END, STEER_FILE, STEER_START, resolve_window, window_record,
)
from rlstack.policy.siteschema import SiteMeta
from rlstack.registry import ADAPTER_TYPES

WORKER = "rlstack_engine.steer_worker.SteerWorker"   # by string: one-way import
STEER_FILENAME = "steer.safetensors"


@dataclass(frozen=True)
class SteerResident:
    """One bundle's steer state as this build holds it: the bundle it belongs
    to (the salt's first half) and the file the hook loads by address."""

    bundle_id: str
    file: Path


class SteerRollout(RolloutLowering):
    """Per-boundary vectors, served by the engine image's residual hook."""

    adapter_type = "steer"
    mechanism = Mechanism.RESIDUAL
    claims = (f"extra_args.{STEER_FILE}", "cache_salt")

    def demands(self) -> BuildDemands:
        """The seam and its price: OUR worker class (probe at boot, hooks on
        the model), eager mode — a hook does not fire inside a captured graph
        — and the V1 model runner, whose request table the hook reads; vllm
        0.28.0 boots the V2 runner by default and exposes the choice as an
        environment variable only. All three stated, so a build that will not
        pay refuses this adapter type at construction instead of steering
        nothing."""
        return BuildDemands(engine_args={"worker_cls": WORKER,
                                         "enforce_eager": True},
                            env={"VLLM_USE_V2_MODEL_RUNNER": "0"})

    def reaches(self, meta: SiteMeta) -> bool:
        """The residual boundaries the hook stands at: every decoder layer's
        output and the final norm's. The logits are another lever's."""
        return meta.is_boundary and (meta.path.startswith("model.layers.")
                                     or meta.path == "model.norm")

    def attach(self, bundle_id: str,
               payloads: Mapping[str, bytes]) -> SteerResident:
        """The bank's steer entries fused into ONE file, {path: vector} in the
        served dtype, width-checked against the base — written to a temporary
        name and renamed, so a crash mid-attach leaves nothing a hook could
        mistake for the bundle."""
        vectors = steer_torch.merge_vectors(payloads)
        width = int(self.build.config.hidden_size)
        for path, vector in vectors.items():
            if int(vector.shape[-1]) != width:
                raise ValueError(
                    f"bundle {bundle_id!r} carries a {int(vector.shape[-1])}-"
                    f"wide steer at {path!r}; {self.build.base} is {width} wide")
        dtype = self._served_dtype()
        alpha = steer_torch.merge_alpha(payloads)
        directory = self.build.workdir / bundle_id.replace(":", "_")
        directory.mkdir(parents=True, exist_ok=True)
        file = directory / STEER_FILENAME
        staging = directory / (STEER_FILENAME + ".tmp")
        # the fraction rides the fused file's metadata (a norm-scaled entry's
        # payload declared it), so the hook scales what the learner scaled
        save_file({path: vector.to(dtype).contiguous()
                   for path, vector in vectors.items()}, str(staging),
                  metadata=None if alpha is None
                  else {steer_torch.ALPHA_KEY: repr(alpha)})
        os.replace(staging, file)
        return SteerResident(bundle_id=bundle_id, file=file)

    def apply(self, attached: SteerResident, request: Request) -> Levers:
        """This request's window, resolved and recorded; the file and the
        window for the hook; the bundle and the window for the cache."""
        directive = ADAPTER_TYPES.get(self.adapter_type).instance.directive_for(
            request)
        start, end = resolve_window(directive, request)
        return Levers(
            extra_args={STEER_FILE: str(attached.file),
                        STEER_START: start, STEER_END: end},
            cache_salt=cache_salt(attached.bundle_id, start, end),
            turn_extras=window_record(start, end))

    def align(self, attached: SteerResident) -> Alignment:
        """A vector adds no positions — the prompt is exactly the tokens."""
        return Alignment(0)

    def detach(self, attached: SteerResident) -> None:
        """Remove the bundle's file; a request in flight cannot still name it
        (residency never evicts a pinned bundle), and the hook's own bank
        drops a path that is gone."""
        shutil.rmtree(attached.file.parent, ignore_errors=True)

    def _served_dtype(self) -> Any:
        """The dtype the served weights carry — the vector is added to a
        residual in it."""
        declared = (getattr(self.build.config, "dtype", None)
                    or getattr(self.build.config, "torch_dtype", None))
        if isinstance(declared, str):
            return getattr(torch, declared)
        return declared or torch.bfloat16


def cache_salt(bundle_id: str, start: int, end: int | None) -> str:
    """Bundle plus window: reuse lives within one, dies across two."""
    return f"{bundle_id}/{start}:{'' if end is None else end}"



class NSteerRollout(SteerRollout):
    """The norm-scaled steer, served by the same hook: the fused file carries
    the fraction in its metadata and the hook scales each token's add by
    alpha times that token's live residual norm. Same demands, same window,
    same record — one family, one lever."""

    adapter_type = "nsteer"
