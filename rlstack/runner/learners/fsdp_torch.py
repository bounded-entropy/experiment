"""FsdpTorchLearner: the Learner protocol with the frozen base SHARDED.

The thing that does not fit is the frozen base, so the frozen base is the thing
that gets sharded: it goes across `fsdp` ranks, all-gathered for a forward and
freed after, and it is frozen so no gradient is ever reduced. The DELTAS are
not sharded — a tenant installs after the wrap, so they live whole on every
rank and each rank steps its own copy. This build therefore buys MEMORY, not
throughput: the ranks recompute the same microbatch rather than splitting one.

THE LOAD IS SHARDED TOO, and that is what makes the width real: the base
arrives on the CPU and each decoder block is moved to the device only to be
sharded on the next line, so a rank's load peak is bounded by ITS SHARD rather
than by the checkpoint. Loading whole and sharding after — what this did until
#58 — capped a learner at whatever base fits on ONE card no matter what `fsdp`
said, which is a ceiling exactly where the width was supposed to remove one.

THE WIDTH-INDEPENDENCE INVARIANT: emit()/load() bytes must not depend on
`fsdp`. It holds by construction — the only sharded thing is the frozen base,
which is never emitted — and `attest_emit_is_width_free` proves it at every
emit instead of trusting it, because resume, warm starts and bundle compilation
all read those bytes. Rank 0's copy is the truth; the other ranks exist to
stand in the collectives and their answers are discarded, which is safe only
while the five verbs stay deterministic functions of their arguments — a verb
consulting rank-local state the ranks do not share would desynchronize them.

Out of scope, deliberately: sleep-sharing (an alternation would have to swing
every rank in step), so an FSDP host is dedicated or concurrent.
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
        self.fsdp = ranks.width      # build fact: the attested width
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

    # ---- the rules this build adds, one named method each -------------------

    def shard_the_frozen_base(self) -> None:
        """FSDP2 over the frozen base: one group per decoder block, then the
        root. Bottom-up is fully_shard's documented contract, and per block is
        the grouping that lets one all-gather overlap the previous block's
        compute. Nothing trainable is inside — tenants install after this — so
        no gradient is ever reduce-scattered and every rank keeps whole
        deltas.

        TWO RULES, one per phase, and a rank's memory is bounded by both:

        A RANK'S LOAD PEAK IS BOUNDED BY ITS SHARD, NOT BY THE BASE. The base
        arrives on the CPU (see load_the_frozen_base_sharded) and each block is
        moved to the device only to be sharded on the next line, so the largest
        thing a device ever holds whole is ONE decoder block — not the
        checkpoint. Before this, load-then-shard put the entire base on one
        device first, which silently capped a learner at whatever base fits on
        a SINGLE card whatever `fsdp` said: a 14B in bf16 is 27.5 GiB and an L4
        has 22, so the width that existed to make it fit never got the chance
        to.

        A RANK'S PARAMETER FOOTPRINT IN THE FORWARD IS ONE BLOCK'S UNSHARD,
        NEVER THE MODEL. That is what the per-block grouping plus
        reshard_after_forward buys; it is spelled out rather than left to a
        default because it is the difference between ~1.4 GiB of unshard
        buffers and 27.5 GiB. It bounds PARAMETERS only, and #58 measured that
        parameters were never the thing that filled the card: one ~1000-token
        document's saved activations are, and they are bounded by neither of
        these rules nor by microbatch_tokens, which cannot go below one
        document (flatten.pack: "a single document longer than
        microbatch_tokens gets its own oversized batch").
        """
        from torch.distributed.fsdp import fully_shard

        mesh = self.ranks.mesh()
        for block in self.decoder_blocks():
            block.to(self.device)          # one block whole: the peak's bound
            fully_shard(block, mesh=mesh, reshard_after_forward=True)
        self.move_what_no_block_owns()
        fully_shard(self._model, mesh=mesh)
        self.attest_base_is_sharded()
        # Each move left its whole-tensor staging in torch's caching allocator.
        # A co-resident engine reserves from the driver, not from that cache,
        # so the difference between the peak and the shard has to go back or
        # the sampler beside us cannot start.
        torch.cuda.empty_cache()

    def move_what_no_block_owns(self) -> None:
        """Move the tensors no decoder block owns — the input embedding, the
        untied output head, the final norm — onto the device, AND TOUCH
        NOTHING ELSE.

        `self._model.to(device)` would be the obvious call and it reaches too
        far: Module._apply walks EVERY parameter in the tree, including the
        blocks this method's caller has already sharded, and swaps each one's
        `.data` while fully_shard's FSDPParam bookkeeping still aliases the
        storage it made at wrap time. Nothing here depends on what that would
        do — the point is that it is not this method's business. The move is
        by hand, over the leaves still on the CPU; a sharded parameter is
        already on its device and is skipped by that test alone, so this never
        has to know what FSDP did.

        STATED PLAINLY BECAUSE IT WAS FIRST WRITTEN DOWN WRONG (#58): swapping
        this in for `self._model.to(device)` changed the measured forward peak
        by NOTHING — 21.47 GiB of 22.03, to the byte. It is the tidier of two
        working spellings, not a fix for anything. What actually filled that
        card is one document's forward (see the entry), and no arrangement of
        this method moves it.
        """
        for module in self._model.modules():
            for name, param in list(module._parameters.items()):
                if param is not None and param.device.type == "cpu":
                    module._parameters[name] = torch.nn.Parameter(
                        param.data.to(self.device), requires_grad=False)
            for name, buffer in list(module._buffers.items()):
                if buffer is not None and buffer.device.type == "cpu":
                    module._buffers[name] = buffer.to(self.device)

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
        base is loaded onto the CPU, then SHARDED ONTO THE DEVICE, before any
        tenant is installed on it. Order matters both ways — sharding after the
        load because fully_shard shards real tensors, and before the first
        install because a delta installed into an unsharded tree would be swept
        into an FSDP parameter group and emitted as a shard.

        A learner that already holds a base takes the base class's path, which
        is where the one-learner-one-base refusal lives; only the FRESH load
        differs here, because only the fresh load touches device memory.
        """
        if self._model is not None:
            super()._ensure_base(base)      # the one-base rule, unchanged
            return
        self.load_the_frozen_base_sharded(base)

    def load_the_frozen_base_sharded(self, base: str) -> None:
        """The fresh load, arriving on the CPU so the device never holds the
        base whole.

        This is TorchLearner._ensure_base's body with ONE thing removed: the
        `.to(device)`. The unsharded build has to make that move — it has one
        device and the whole base has to fit on it — but for this build it was
        the thing that made `fsdp` a lie at the only moment it mattered, so the
        move belongs to shard_the_frozen_base, which does it a block at a time.
        Everything else is the base class's: frozen, in eval (replay is exact
        recompute, so never dropout), and the base recorded before any tenant.
        """
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=self.dtype)
        model.requires_grad_(False)
        model.eval()
        self._model = model
        self._base = base
        # recompute BEFORE the wrap: checkpointing replaces each block's
        # forward in place, and fully_shard's hooks then wrap the recomputing
        # one, so the backward's re-gather and the recompute compose in the
        # order FSDP expects
        self.checkpoint_the_blocks()
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
