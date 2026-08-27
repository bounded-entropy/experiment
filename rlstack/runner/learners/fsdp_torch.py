"""FsdpTorchLearner: the Learner protocol with the frozen base SHARDED.

TorchLearner holds one frozen base and many tenants' small trainable deltas.
That asymmetry is what makes multi-tenancy affordable (#34), and it is also
what decides the shape of this build: the thing that does not fit is the
frozen base, so the frozen base is the thing that gets sharded.

    the base       sharded across `fsdp` ranks by FSDP2's fully_shard (torch
                   2.13 deprecates FSDP1), one parameter group per decoder
                   block plus the root: all-gathered for a forward, freed
                   after. It is frozen, so no gradient is ever reduced.
    the deltas     NOT sharded. A tenant installs after the wrap, so no FSDP
                   group owns its params; they live whole on every rank, and
                   each rank's optimizer steps its own whole copy.
    the chorus     every rank runs every collective verb on IDENTICAL inputs
                   (ranks.py). RANK 0'S COPY IS THE TRUTH — its TrainStats is
                   what the runner reads and its tensors are what emit
                   serializes; the other ranks are there to stand in the
                   all-gathers, and their answers are discarded.

So this build buys MEMORY, not throughput: the ranks recompute the same
microbatch rather than splitting one. Data-parallel width (different
microbatches per rank, gradients reduced across them) is a later upgrade and
would change exactly one thing — the deltas would need a reduction before
optim_step.

THE WIDTH-INDEPENDENCE INVARIANT (#45): emit()/load() bytes must not depend
on `fsdp`. It holds here by construction — the only sharded thing is the
frozen base, which is never emitted — and `attest_emit_is_width_free` proves
it at every emit instead of trusting it. Resume, warm starts (WarmStart) and
bundle compilation all read those bytes; the day they became rank-shaped, a
run trained at fsdp=2 could never be resumed anywhere else.

Out of scope, deliberately: sleep-sharing (an alternation would have to swing
every rank in step, and an FSDP host is dedicated or concurrent); pipeline or
tensor parallelism for training (sharding is the only axis here).

INTEGRATION POINTS with torch_learner.py — everything this subclass leans on:
  * `TorchLearner(device=, dtype=, grad_clip=)`, and `self.fsdp` as the build
    fact the submit gate attests;
  * `self._model` (the frozen base module) and `_ensure_base(base)`, the load
    step this overrides to shard after it;
  * `self._tenant(tenant).params[entry].parameters()`, the trainable tensors
    the emit attestation walks;
  * the five verbs being DETERMINISTIC FUNCTIONS OF THEIR ARGUMENTS — the
    chorus stays in lockstep under any implementation whose collectives
    depend only on (tenant, batch), which every honest forward is. A verb
    that consulted rank-local state the ranks do not share (a clock, a
    random draw, a store read) would desynchronize them.
"""

from __future__ import annotations

import functools
import traceback
from collections.abc import Mapping

import torch

from rlstack.data.flatten import TokenBatch
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.interfaces import Emitted, TrainStats
from rlstack.runner.learners.ranks import STOP, RankCommand, RankGroup
from rlstack.runner.learners.torch_learner import TorchLearner
from rlstack.spec.specs import ExperimentSpec


class FsdpTorchLearner(TorchLearner):
    """A TorchLearner whose base lives across `ranks.width` devices."""

    def __init__(self, ranks: RankGroup, *,
                 dtype: torch.dtype = torch.bfloat16,
                 grad_clip: float = 1.0) -> None:
        super().__init__(device=ranks.device_str, dtype=dtype,
                         grad_clip=grad_clip)
        self.fsdp = ranks.width      # build fact (#43): the attested width
        self.ranks = ranks

    # ---- the verbs, each announced before it runs ---------------------------

    def install(self, tenant: str, spec: ExperimentSpec,
                resolved_sites: Mapping[str, tuple[SiteMeta, ...]]) -> None:
        """Announced: it loads and shards the base (a collective build), and
        every rank needs this tenant's params to run its half of a forward."""
        self.announce("install", (tenant, spec, dict(resolved_sites)))
        super().install(tenant, spec, resolved_sites)

    def forward_backward(self, tenant: str, batch: TokenBatch) -> TrainStats:
        """Announced: the forward all-gathers the base and the backward
        re-gathers it. Rank 0's stats are the run's stats."""
        self.announce("forward_backward", (tenant, batch))
        return super().forward_backward(tenant, batch)

    def optim_step(self, tenant: str) -> None:
        """Announced, though it takes no collective: every rank holds a whole
        copy of the deltas and must step it, or the ranks would drift apart
        and the next forward would run different adapters on each."""
        self.announce("optim_step", (tenant,))
        super().optim_step(tenant)

    def load(self, tenant: str, adapters: Mapping[str, bytes],
             optim: Mapping[str, bytes] | None) -> None:
        """Announced: the payload comes off rank 0's store and every rank
        must end up holding the same deltas. The bytes are the store's own —
        distributing them is what makes a resume width-independent."""
        self.announce("load", (tenant, dict(adapters),
                               dict(optim) if optim else None))
        super().load(tenant, adapters, optim)

    def emit(self, tenant: str) -> Emitted:
        """NOT announced: the trainable state is replicated, so serializing
        it is rank-local work — and that is exactly the invariant, so it is
        attested here rather than assumed."""
        self.attest_emit_is_width_free(tenant)
        return super().emit(tenant)

    # ---- the two rules this build adds, one named method each ---------------

    def shard_the_frozen_base(self) -> None:
        """FSDP2 over the frozen base: one group per decoder block, then the
        root. Bottom-up is fully_shard's documented contract, and per block
        is the grouping that lets one all-gather overlap the previous block's
        compute. Nothing trainable is inside — tenants install after this —
        so no gradient is ever reduce-scattered and every rank keeps whole
        deltas."""
        from torch.distributed.fsdp import fully_shard

        mesh = self.ranks.mesh()
        for block in self.frozen_blocks():
            fully_shard(block, mesh=mesh)
        fully_shard(self._model, mesh=mesh)
        self.attest_base_is_sharded()

    def frozen_blocks(self) -> list[torch.nn.Module]:
        """The units FSDP shards: the base's decoder blocks, named
        `model.layers` in an HF causal LM — the same path the site schema
        resolves against (`model.layers.N.self_attn.q_proj`), so this reads
        the tree in the schema's own vocabulary."""
        return list(self._model.model.layers)

    def base_parameters(self) -> list[torch.nn.Parameter]:
        """The frozen base's own parameters — the module tree MINUS every
        installed tenant's deltas, excluded by identity.

        After an install the tree carries both, and their shapes are the
        whole design: the base sharded, the deltas whole on every rank. Any
        statement about "the base" has to make that cut first, or it counts
        a LoRA matrix as a parameter fully_shard forgot."""
        delta_ids = {id(param) for tenant in self._tenants.values()
                     for params in tenant.params.values()
                     for param in params.parameters()}
        return [p for p in self._model.parameters() if id(p) not in delta_ids]

    def attest_base_is_sharded(self) -> None:
        """Loud proof that the wrap took: above width 1, every base parameter
        is a DTensor. A silent no-op here would surface as an out-of-memory
        on the first large base — or, worse, as a run that quietly held a
        whole copy per rank and called itself sharded."""
        if self.fsdp == 1:
            return
        from torch.distributed.tensor import DTensor

        whole = [p for p in self.base_parameters()
                 if not isinstance(p, DTensor)]
        if whole:
            raise RuntimeError(
                f"fully_shard left {len(whole)} base parameters unsharded at "
                f"fsdp={self.fsdp} — this build is not the width it reports")

    def shard_report(self) -> dict:
        """What this rank actually holds — the measurement that makes the
        sharding visible in a log, and keeps callers out of the module tree.
        `local` is this rank's share of the base; at width w it should be
        about `whole / w`. `deltas` is what stays whole, on purpose."""
        from torch.distributed.tensor import DTensor

        base = self.base_parameters()
        return {
            "parameters": len(base),
            "sharded": sum(isinstance(p, DTensor) for p in base),
            "whole": sum(p.numel() for p in base),
            "local": sum(p.to_local().numel() if isinstance(p, DTensor)
                         else p.numel() for p in base),
            "deltas": sum(p.numel() for tenant in self._tenants.values()
                          for params in tenant.params.values()
                          for p in params.parameters()),
        }

    def attest_emit_is_width_free(self, tenant: str) -> None:
        """THE invariant (#45): what emit writes may not depend on `fsdp`.

        Concretely, no trainable tensor may be a shard — a DTensor would
        serialize this rank's slice and the store would hold bytes only a
        chorus of the same width could read back. Sharding touches the frozen
        base alone, so this holds by construction; it is checked anyway,
        because those bytes are what resume, warm start and bundle compile
        every later run."""
        if self.fsdp == 1:
            return
        from torch.distributed.tensor import DTensor

        state = self._tenant(tenant)
        for entry, params in state.params.items():
            for param in params.parameters():
                if isinstance(param, DTensor):
                    raise RuntimeError(
                        f"tenant {tenant!r} entry {entry!r} holds a SHARDED "
                        f"trainable parameter: emit would write "
                        f"fsdp={self.fsdp}-shaped bytes, and the store's "
                        f"bytes must be width-independent (#45)")

    # ---- the chorus ----------------------------------------------------------

    def announce(self, verb: str, args: tuple) -> None:
        """Rank 0 tells the chorus what it is about to do. A follower reaches
        its verbs through `follow` and never announces — one voice starts a
        collective, or two ranks would each wait for the other's."""
        if self.ranks.rank == 0:
            self.ranks.announce(RankCommand(verb, args))

    def follow(self, command: RankCommand) -> None:
        """One announced verb, run on this rank. The table IS the contract:
        exactly the verbs rank 0 announces appear here, and each runs the
        same TorchLearner code rank 0 runs, on the same arguments."""
        if command.verb == "install":
            super().install(*command.args)
        elif command.verb == "forward_backward":
            super().forward_backward(*command.args)
        elif command.verb == "optim_step":
            super().optim_step(*command.args)
        elif command.verb == "load":
            super().load(*command.args)
        else:
            raise ValueError(f"unknown chorus verb {command.verb!r}")

    def follow_until_stopped(self) -> None:
        """A follower's whole life: hear a verb, run it, wait for the next."""
        while True:
            command = self.ranks.hear()
            if command.verb == STOP:
                return
            self.follow(command)

    def stop(self) -> None:
        """End the chorus. Deploy code calls this in a finally — a run that
        crashed and a run that finished both leave children holding GPUs."""
        self.ranks.stop()

    # ---- the one override that shards ---------------------------------------

    def _ensure_base(self, base: str) -> None:
        """TorchLearner's rule (one learner, one base) plus this build's: the
        base is loaded whole, then SHARDED, before any tenant is installed on
        it. Order matters both ways — sharding after the load because
        fully_shard shards real tensors, and before the first install because
        a delta installed into an unsharded tree would be swept into an FSDP
        parameter group and emitted as a shard."""
        fresh = self._model is None
        super()._ensure_base(base)
        if fresh:
            self.shard_the_frozen_base()


def lead_fsdp_learner(width: int, *, dtype: torch.dtype = torch.bfloat16,
                      grad_clip: float = 1.0) -> FsdpTorchLearner:
    """Rank 0's constructor: start the chorus, then the learner in front of it.

    This is what a deploy hands to a Host — the learner looks exactly like a
    TorchLearner from the outside, reports `fsdp=width` as its build fact, and
    the host's regimes attest that against the partition it was born on."""
    follower = functools.partial(_follow_rank, dtype=dtype, grad_clip=grad_clip)
    ranks = RankGroup.lead(width, follower)
    return FsdpTorchLearner(ranks, dtype=dtype, grad_clip=grad_clip)


def _follow_rank(rank: int, width: int, port: int, *, dtype: torch.dtype,
                 grad_clip: float) -> None:
    """A non-zero rank's whole life, and the entry point of its process.

    Spawned (so this module is imported fresh here): join the group, build
    the same learner over this rank's own device, run what rank 0 announces,
    exit when it says stop. A failure is printed and ends the process — rank
    0 then fails at its next collective rather than waiting out the
    rendezvous timeout in silence."""
    group = RankGroup.join(rank, width, port)
    try:
        FsdpTorchLearner(group, dtype=dtype,
                         grad_clip=grad_clip).follow_until_stopped()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        group.leave()
