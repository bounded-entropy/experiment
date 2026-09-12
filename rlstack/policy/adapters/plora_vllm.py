"""plora's rollout lowering: punica again, but serving an ENSEMBLE.

A plora policy is a distribution over adapters, and vLLM has no lever for that
— so this lowering collapses the distribution into the lever vLLM does have. At
attach it draws nothing: the payload already carries the `members` noise vectors
the version was sealed with, and each one is MATERIALIZED into an ordinary rank-k
peft adapter (lora_A = A frozen, lora_B = U C) and registered as its own
LoRARequest, plus one more for the posterior MEAN. Requests then pick a member
by seed, which is a keyword like any other, so nothing downstream of `apply`
knows an ensemble was ever involved.

What this build pays for it: `max_loras` counted in MEMBERS rather than bundles,
because a resident plora bundle occupies members + 1 punica slots. What it needs
besides: a way to read a content-addressed object, since the frozen directions
are megabytes per site and travel by ADDRESS rather than in the payload — a
build handed no reader refuses the adapter type at construction, exactly as it
refuses a plugin it does not install.

vLLM and torch are imported at module scope — this file loads only from
Plora.rollout_lowering, never from the package root (STYLE rule 7).
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save as st_save
from vllm.lora.request import LoRARequest

from rlstack.policy.adapters import plora_factors, plora_torch
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.adapters.plora import EPS_RECORD, MEMBER_RECORD
from rlstack.policy.adapters.rollout import (
    Alignment, BuildDemands, Levers, Request, RolloutLowering, ServingBuild,
)
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.seeds import derive

# The posterior mean is not a draw, so it is not a member index. Score traffic
# is served by it: scoring is seedless and deterministic by contract, and the
# mean is the one member that is a property of the policy rather than of a coin.
MEAN_MEMBER = -1


@dataclass(frozen=True)
class PloraEnsemble:
    """One bundle's plora, resident: every member as a registered LoRARequest,
    beside the noise that made it (which each request RECORDS)."""

    requests: tuple[LoRARequest, ...]        # member e -> its adapter
    mean: LoRARequest                        # z = mu, for seedless traffic
    noise: tuple[tuple[float, ...], ...]     # member e -> the eps it used
    latent: int
    directory: Path                          # everything attach wrote


class PloraRollout(RolloutLowering):
    """A probabilistic low-rank delta, served as E ordinary ones."""

    adapter_type = "plora"
    mechanism = Mechanism.PUNICA
    claims = ("lora_request",)               # so lora and plora cannot share a
    #                                          bundle: one request, one adapter

    def __init__(self, build: ServingBuild) -> None:
        super().__init__(build)
        self._factors: dict[str, dict[str, tuple]] = {}   # uri -> per-site (U, A)
        self.check_can_resolve_factors()

    def check_can_resolve_factors(self) -> None:
        """A build that cannot read content-addressed objects cannot serve
        plora, and says so HERE — at construction, before any reachability is
        reported (I7: die at boot, never mid-run).

        plora's frozen half is deliberately not in the payload: it is identical
        at every version and megabytes per site, so a bundle carrying it would
        pay for it once per update forever. The consequence is that the address
        must be resolvable, and this is the one place that can tell.
        """
        if self.build.cas is None:
            raise NotImplementedError(
                f"this engine build was given no way to read a "
                f"content-addressed object, so it cannot resolve plora's frozen "
                f"factors (their cas address rides in the payload, the "
                f"megabytes do not). Build it VllmEngine(..., cas_get=<the "
                f"store's cas_get>)")

    def demands(self) -> BuildDemands:
        """The lever plus its sizing, counted in MEMBERS: one resident plora
        bundle is members + 1 punica adapters (the ensemble and the mean),
        which is why the build's slot budget multiplies — and it is the
        BUILD's number (ServingBuild.slots), the same one lora demands, so
        both may be served by one engine."""
        return BuildDemands(engine_args={
            "enable_lora": True,
            "max_loras": self.build.slots(),
            "max_lora_rank": self.build.max_rank})

    def reaches(self, meta: SiteMeta) -> bool:
        """Punica reaches every weighted matrix of the served base — and a
        weighted matrix is exactly what plora can factor."""
        return meta.has_weight

    def attach(self, bundle_id: str,
               payloads: Mapping[str, bytes]) -> PloraEnsemble:
        """Materialize this bundle's ensemble onto disk and register it.

        The payload is the trained half; the frozen half is fetched by address
        and cached for the whole build, because every version of every tenant on
        this base shares it. Each member is a plain peft directory, so vLLM
        loads them by the same path it loads any adapter — the probability is
        entirely on this side of the seam.
        """
        payload = self.only_entry(bundle_id, payloads)
        meta, _ = plora_factors.unpack_artifact(payload)
        state = plora_torch.resident(
            payload, self.factors_at(str(meta["factors"]), int(meta["k"])))
        noise = plora_torch.served_noise(payload)
        home = self.build.workdir / f"plora-{bundle_id.replace(':', '_')}"
        home.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            members = tuple(
                self._write_member(state, home, f"m{index}",
                                   plora_torch.reparameterized_latent(state, eps))
                for index, eps in enumerate(noise))
            mean = self._write_member(state, home, "mean", state.mu)
        return PloraEnsemble(
            requests=members, mean=mean,
            noise=tuple(tuple(float(v) for v in eps) for eps in noise),
            latent=state.latent, directory=home)

    def only_entry(self, bundle_id: str,
                   payloads: Mapping[str, bytes]) -> bytes:
        """ONE LATENT PER TRAJECTORY, enforced at the bundle.

        The recorded facts are a flat namespace — a turn has one `plora_eps` —
        so two plora entries in one bank would record over each other and leave
        replay unable to say whose latent it held. Refused here, while the
        bundle is still just an id, rather than at the first replay forward.
        """
        if len(payloads) != 1:
            raise ValueError(
                f"bundle {bundle_id!r} carries {len(payloads)} plora entries "
                f"({sorted(payloads)}); a turn records ONE latent, so a bank "
                f"holds at most one plora")
        return next(iter(payloads.values()))

    def factors_at(self, uri: str, k: int) -> Mapping[str, tuple]:
        """The frozen directions behind one address, fetched once per BUILD.

        Cached on disk under the build's workdir and in memory beside it: the
        artifact is content-addressed, so a hit is proof it is the same bytes,
        and every bundle of every tenant sharing this base and this k shares
        this read.
        """
        if uri not in self._factors:
            cached = self.build.workdir / f"factors-{uri.removeprefix('cas://')}"
            if not cached.exists():
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_bytes(self.build.cas(uri))
            self._factors[uri] = plora_factors.read_factors(
                cached.read_bytes(), base=self.build.base, k=k)
        return self._factors[uri]

    def _write_member(self, state, home: Path, name: str,
                      z: torch.Tensor) -> LoRARequest:
        """One member, as an ordinary peft adapter on disk plus its request.

        Int ids are drawn monotonically and never reused, so no member of any
        version can be confused with a stale cached copy of another.
        """
        directory = home / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_model.safetensors").write_bytes(
            st_save(plora_torch.member_tensors(state, z)))
        (directory / "adapter_config.json").write_text(
            plora_torch.peft_config(
                self.build.base, state.k,
                [path.rsplit(".", 1)[-1] for path in state.paths]))
        request = LoRARequest(lora_name=f"{home.name}/{name}",
                              lora_int_id=self.build.next_lora_id(),
                              lora_path=str(directory))
        return request

    def apply(self, attached: PloraEnsemble, request: Request) -> Levers:
        """Pick this request's member, and RECORD what picking it meant.

        The keyword is the whole mechanism, exactly as it is for lora. What
        plora adds is the other half of its contract: the noise that made this
        member is a sampling-time fact replay cannot re-derive, so it leaves
        with the request and seals into Turn.turn_extras (I6).

        Seedless traffic is SCORE traffic, which is deterministic by contract
        and must not depend on a coin — it gets the posterior mean, and records
        having done so.
        """
        if request.seed is None:
            return Levers(kwargs={"lora_request": attached.mean},
                          turn_extras={EPS_RECORD: [0.0] * attached.latent,
                                       MEMBER_RECORD: MEAN_MEMBER})
        member = derive(request.seed, "plora") % len(attached.requests)
        return Levers(kwargs={"lora_request": attached.requests[member]},
                      turn_extras={EPS_RECORD: list(attached.noise[member]),
                                   MEMBER_RECORD: member})

    def align(self, attached: PloraEnsemble) -> Alignment:
        """A weight delta adds no positions — the prompt is exactly the tokens."""
        return Alignment(0)

    def detach(self, attached: PloraEnsemble) -> None:
        """Delete every member this bundle materialized, and forget their ids.

        The frozen factors are NOT released: they belong to the build, not to
        this bundle, and the next version at the same address reuses the same
        bytes. Detaching is never a loss — a committed version recompiles from
        the store, and the content-addressed id proves it is the same one.
        """
        shutil.rmtree(attached.directory, ignore_errors=True)
