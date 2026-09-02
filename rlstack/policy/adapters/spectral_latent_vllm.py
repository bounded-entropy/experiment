"""spectral_latent's rollout lowering: punica, serving the materialized
ensemble.

The trainer already did all the math: emit's payload carries every member
(and the posterior mean) as top-k peft pairs under `peft.<tag>.` keys, plus
the noise each member was drawn with. This side writes one adapter directory
per member, registers the requests, and — the recording half of the
contract — stamps each request's turn with the noise that made its member
(`slatent_eps`, `slatent_member`), which is what replay recomposes with the
current posterior. Seedless traffic is score traffic and gets the MEAN,
plora's rule.

vLLM and torch are imported at module scope — this file loads only from
SpectralLatent.rollout_lowering, never from the package root (STYLE rule 7).
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save as st_save
from vllm.lora.request import LoRARequest

from rlstack.policy.adapters import lora_torch, spectral_latent_torch
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.rollout import (
    Alignment, BuildDemands, Levers, Request, RolloutLowering, ServingBuild,
)
from rlstack.policy.adapters.spectral_latent import EPS_RECORD, MEMBER_RECORD
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.seeds import derive

MEAN_MEMBER = -1


@dataclass(frozen=True)
class SlatentEnsemble:
    """One bundle's ensemble, resident: every member as a registered
    LoRARequest, beside the noise that made it."""

    requests: tuple[LoRARequest, ...]
    mean: LoRARequest
    noise: tuple[tuple[float, ...], ...]
    latent: int
    directory: Path


class SlatentRollout(RolloutLowering):
    """A distribution over spectral gains, served as E ordinary adapters."""

    adapter_type = "spectral_latent"
    mechanism = Mechanism.PUNICA
    claims = ("lora_request",)

    def __init__(self, build: ServingBuild) -> None:
        super().__init__(build)
        self._next_int_id = 1

    def demands(self) -> BuildDemands:
        """Counted in MEMBERS, plora's sizing: one resident bundle is
        members + 1 punica adapters."""
        return BuildDemands(engine_args={
            "enable_lora": True,
            "max_loras": self.build.max_bundles * (self.build.max_members + 1),
            "max_lora_rank": self.build.max_rank})

    def reaches(self, meta: SiteMeta) -> bool:
        return meta.has_weight

    def attach(self, bundle_id: str,
               payloads: Mapping[str, bytes]) -> SlatentEnsemble:
        """Write every member's directory off the payload and register it.
        ONE entry per bank, plora's rule: the recorded facts are a flat
        namespace, so two would record over each other."""
        if len(payloads) != 1:
            raise ValueError(
                f"bundle {bundle_id!r} carries {len(payloads)} spectral_latent "
                f"entries ({sorted(payloads)}); a turn records ONE latent, so "
                f"a bank holds at most one")
        payload = next(iter(payloads.values()))
        head, tensors = spectral_latent_torch.unpack(payload)
        home = self.build.workdir / f"slatent-{bundle_id.replace(':', '_')}"
        home.mkdir(parents=True, exist_ok=True)
        members = tuple(
            self._write_member(head, tensors, home, f"m{index}")
            for index in range(int(head["members"])))
        mean = self._write_member(head, tensors, home, "mean")
        noise = tensors["noise"]
        return SlatentEnsemble(
            requests=members, mean=mean,
            noise=tuple(tuple(float(x) for x in row) for row in noise),
            latent=int(head["latent"]), directory=home)

    def _write_member(self, head: dict, tensors: Mapping[str, torch.Tensor],
                      home: Path, tag: str) -> LoRARequest:
        prefix = f"peft.{tag}."
        peft = {key[len(prefix):]: value for key, value in tensors.items()
                if key.startswith(prefix)}
        leaves = [key[: -len(".lora_A.weight")].rsplit(".", 1)[-1]
                  for key in peft if key.endswith(".lora_A.weight")]
        directory = home / tag
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_model.safetensors").write_bytes(st_save(peft))
        (directory / "adapter_config.json").write_text(
            lora_torch.peft_config(self.build.base, int(head["k"]), leaves))
        request = LoRARequest(lora_name=f"{home.name}/{tag}",
                              lora_int_id=self._next_int_id,
                              lora_path=str(directory))
        self._next_int_id += 1
        return request

    def apply(self, attached: SlatentEnsemble, request: Request) -> Levers:
        """Pick this request's member and RECORD the noise that made it;
        seedless (score) traffic gets the posterior mean."""
        if request.seed is None:
            return Levers(kwargs={"lora_request": attached.mean},
                          turn_extras={EPS_RECORD: [0.0] * attached.latent,
                                       MEMBER_RECORD: MEAN_MEMBER})
        member = derive(request.seed, "slatent") % len(attached.requests)
        return Levers(kwargs={"lora_request": attached.requests[member]},
                      turn_extras={EPS_RECORD: list(attached.noise[member]),
                                   MEMBER_RECORD: member})

    def align(self, attached: SlatentEnsemble) -> Alignment:
        return Alignment(0)

    def detach(self, attached: SlatentEnsemble) -> None:
        shutil.rmtree(attached.directory, ignore_errors=True)
