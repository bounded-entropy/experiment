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
while every verb stays a deterministic function of its arguments — a verb
consulting rank-local state the ranks do not share would desynchronize them.

ALTERNATION IS IN SCOPE SINCE #82, and the rule that lets it be is the one
below: never swap a sharded parameter's storage without re-aliasing FSDP's
view onto it. A sleep swings the whole chorus — `sleep`/`wake` are announced
verbs like any other collective-adjacent one — and the offload itself is
`Module.to`, which `FSDPModule._apply` already reshards and re-aliases around.
A torch without that hook reports `sleeps: false` and keeps the co-resident
footprint (`probe_sharded_sleep`, the certified-substrate rule I7).
"""

from __future__ import annotations

import functools
import traceback
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from rlstack.data.flatten import TokenBatch
from rlstack.runner.interfaces import Emitted, Parameterization, TrainStats
from rlstack.runner.learners.ranks import STOP, RankCommand, RankGroup
from rlstack.runner.learners.torch_learner import TorchLearner
from rlstack.runner.residents import cap_memory


@dataclass(frozen=True)
class SleepProbe:
    """Whether this torch can offload a SHARDED base, and why not if it
    cannot. A build fact, decided once at construction and reported through
    the resident's hello — the certified-substrate rule (I7) at the one place
    a missing hook would otherwise surface as a corrupt all-gather."""

    supported: bool
    reason: str


ON_FSDP_MODULE = ("_apply", "reshard", "_get_fsdp_state")
"""What a sharded sleep needs of the class `fully_shard` mixes into a module:
`_apply` (the OVERRIDE — FSDPModule defines it, nn.Module's own is the one
that would leave the aliases stale), `reshard` (free the all-gather outputs
before the move) and `_get_fsdp_state` (the way to this module's FSDPParams,
which is how the move is attested)."""

ON_FSDP_PARAM = ("reset_sharded_param",)
"""And of FSDP's record of one sharded parameter: the relabel itself —
`_sharded_param_data` re-derived from the CURRENT local tensor."""


def probe_sharded_sleep() -> SleepProbe:
    """Does the pinned torch expose the hooks a sharded offload needs?

    Asked at build, not at the first evict: a learner that cannot sleep must
    say so in its hello so no host wires an alternation hook it would fail —
    the substrate is certified, never assumed (I7). The import is the first
    half of the question and the attributes are the second, because an
    FSDP2 that moved house (`_composable/fsdp` before torch 2.6) and one that
    dropped the relabel are the same answer with different reasons.
    """
    try:
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except ImportError as missing:
        return SleepProbe(False, f"this torch has no FSDP2 surface: {missing}")
    return sleep_probe_of(FSDPModule, FSDPParam)


def sleep_probe_of(fsdp_module: type, fsdp_param: type) -> SleepProbe:
    """The rule itself, over the two torch classes that carry it — taken as
    ARGUMENTS so the refusal path is a test and not a hypothesis."""
    missing = [f"FSDPModule.{name}" for name in ON_FSDP_MODULE
               if not hasattr(fsdp_module, name)]
    missing += [f"FSDPParam.{name}" for name in ON_FSDP_PARAM
                if not hasattr(fsdp_param, name)]
    if missing:
        return SleepProbe(
            False, f"this torch's FSDP2 is missing {', '.join(missing)}: a "
                   f"shard cannot be moved without re-aliasing FSDP's view "
                   f"of it, so this build keeps its device")
    return SleepProbe(True, "")


def fsdp_params_of(module: torch.nn.Module) -> list:
    """One wrapped module's FSDPParams — FSDP's own record of each sharded
    parameter, including the private flat view an all-gather reads.

    Reached exactly the way `FSDPModule.reshard` reaches its group, because
    there is no public way to it and the build already refused to sleep if
    that path is absent (`probe_sharded_sleep`). A module whose group is
    empty — everything it owns belongs to a nested wrap — has none."""
    group = module._get_fsdp_state()._fsdp_param_group
    return list(group.fsdp_params) if group is not None else []


class FsdpTorchLearner(TorchLearner):
    """A TorchLearner whose base lives across `ranks.width` devices."""

    def __init__(self, ranks: RankGroup, *,
                 dtype: torch.dtype = torch.bfloat16,
                 grad_clip: float = 1.0,
                 checkpoint_activations: bool = True) -> None:
        super().__init__(device=ranks.device_str, dtype=dtype,
                         grad_clip=grad_clip,
                         checkpoint_activations=checkpoint_activations)
        self.fsdp = ranks.width      # build fact: the attested width
        self.ranks = ranks
        # a sharded base offloads like any other (#82) — what it needs is the
        # substrate's relabel, so the build fact is PROBED rather than derived
        # from the width. Width 1 asks the same question as width 8: this
        # build's base is `fully_shard`-wrapped at every width, and the wrap is
        # what the move has to be careful with.
        probe = probe_sharded_sleep()
        self.sleeps = probe.supported
        self.sleep_refusal = probe.reason

    # ---- the verbs, each announced before it runs ---------------------------

    def install(self, tenant: str, parameterization: Parameterization) -> None:
        """Announced: it loads and shards the base (a collective build), and
        every rank needs this tenant's params to run its half of a forward.
        What crosses to the followers is the Parameterization and nothing
        more — the chorus narrows with the protocol (ADR 0002, Q2)."""
        self.announce("install", (tenant, parameterization))
        super().install(tenant, parameterization)

    def uninstall(self, tenant: str) -> None:
        """Announced: install was, and every rank wired this tenant's params
        into its own copy of the tree — a rank left holding them would route
        rows the others no longer can."""
        self.announce("uninstall", (tenant,))
        super().uninstall(tenant)

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

    async def sleep(self) -> None:
        """Announced: the base is ONE object across the chorus, so it leaves
        the devices only if every rank puts its own shard down — a rank that
        stayed awake would hold its shard through the engine's whole
        generation, which is the entire memory the alternation exists to buy.

        THE ASYNC VERB AND THE SYNC CHORUS MEET IN A SYNC BODY. This is a door
        verb, so it is awaited (the arbiter's evict hook is async); a follower
        hears verbs on a blocking broadcast, where no loop turns. Both call
        `TorchLearner.hand_the_device_back`, which is why the async signature
        never reaches the wire: what crosses is the verb name and an empty
        argument tuple.

        The announce is itself a collective and briefly blocks rank 0's loop,
        exactly as `install` and `forward_backward` do — and for the same
        reason, that a chorus verb is entered by every rank in the same order.
        """
        self.announce("sleep", ())
        self.hand_the_device_back()

    async def wake(self) -> None:
        """Announced, and the mirror of sleep. The arbiter switches only at
        zero in-flight work, so the first forward after this finds every rank
        awake — which it must: FSDP materializes its comm state on the first
        forward (`FSDPParamGroup.lazy_init`) and an all-gather reads whatever
        device the shard is on."""
        self.announce("wake", ())
        self.take_the_device_back()

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
        blocks this method's caller has already sharded. Nothing here depends
        on what that would do — the point is that it is not this method's
        business. The move is by hand, over the leaves still on the CPU; a
        sharded parameter is already on its device and is skipped by that test
        alone, so this never has to know what FSDP did.

        WHAT THIS PARAGRAPH USED TO SAY, AND WHY IT WAS WRONG (#82): that the
        wide call would swap a sharded parameter's `.data` while FSDP's
        bookkeeping still aliased the wrap-time storage. It would not — each
        wrapped block's own `FSDPModule._apply` reshards and re-aliases as the
        recursion passes through it (see `move_the_base`). The rule is about
        re-aliasing, not about the reach; this method survives on the second
        reason, which is the one that was always load-bearing.

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

    def move_the_base(self, device: str) -> None:
        """The sharded base across the bus — and FSDP's own view of it moved
        with it. THE RULE: NEVER SWAP A SHARDED PARAMETER'S STORAGE WITHOUT
        RE-ALIASING FSDP'S VIEW ONTO IT.

        `fully_shard` turns each parameter into a DTensor whose local tensor is
        this rank's chunk, and keeps a PRIVATE FLAT VIEW of that same storage
        (`FSDPParam._sharded_param_data`) — the thing an all-gather actually
        reads. Move the parameter and leave that view behind and nothing
        raises: the old storage stays alive, so no device memory comes back at
        all, and the next all-gather reads a device the parameter has left.
        That is the resharding confusion the math campaign hit, and until #82
        it was written down here as "never `Module.to()` a tree fully_shard
        owns" — which named the symptom rather than the rule, and cost the
        build an alternation it could have had.

        THE RE-ALIASING IS THE SUBSTRATE'S, NOT OURS. `FSDPModule._apply`
        reshards the module (freeing any all-gather output still allocated),
        runs `nn.Module._apply`, and then calls `reset_sharded_param()` on
        every FSDPParam of its own group — which re-derives the flat view from
        the CURRENT local tensor. `nn.Module._apply` recurses into children
        first, so ONE `.to()` at the root walks every decoder block's wrap on
        the way down and the root's own group on the way out. `Module.to` is
        the operation `reset_sharded_param` exists for, and the reason the move
        is spelled as a plain `.to()` rather than by hand.

        Two things this rests on, and both are checked rather than trusted:
        the hook exists in the pinned torch (`probe_sharded_sleep`, at build,
        or this build reports it cannot sleep), and it took
        (`attest_the_shards_are_realiased`, at every move).

        Before the first forward the wrap has not lazily initialized yet and
        would re-alias everything itself (`FSDPParamGroup.lazy_init`); the
        common alternation case IS that one — a tenant installs and the engine
        samples wave 1 before any backward — so the two paths are deliberately
        the same code, and doing it twice is idempotent.
        """
        self._model.to(device)
        self.attest_the_shards_are_realiased()

    def sharded_modules(self) -> list[torch.nn.Module]:
        """Every module `fully_shard` owns in this tree: each decoder block,
        and the root the second wrap covers (where the embedding, the untied
        head and the final norm live). Found BY TYPE — `fully_shard` mixes
        FSDPModule into the class it wrapped — rather than by remembering what
        `shard_the_frozen_base` did, so a tree that was wrapped some other way
        is still described correctly."""
        from torch.distributed.fsdp import FSDPModule

        return [module for module in self._model.modules()
                if isinstance(module, FSDPModule)]

    def attest_the_shards_are_realiased(self) -> None:
        """Loud proof that a move took: FSDP's private flat view of each shard
        aliases the parameter's CURRENT local tensor, on the current device.

        This is the one failure mode an offloaded shard has and it is SILENT —
        no exception, the memory simply never comes back and the next
        all-gather reads bytes from the device the base just left. A run would
        discover it as a wrong number or an out-of-memory some phases later,
        which is exactly the kind of thing this repo attests at the seam
        instead. Pointer equality holds even for a padded shard: the local
        tensor is a narrow of the padded flat view starting at zero."""
        for module in self.sharded_modules():
            for fsdp_param in fsdp_params_of(module):
                local = fsdp_param.sharded_param._local_tensor
                view = fsdp_param._sharded_param_data
                if view.device != local.device or (
                        local.numel() and view.data_ptr() != local.data_ptr()):
                    raise RuntimeError(
                        f"a sharded parameter moved and FSDP's view of it did "
                        f"not: the shard is on {local.device} and the view "
                        f"an all-gather reads is on {view.device} — this "
                        f"torch's FSDPModule._apply did not re-alias (#82)")

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
        same TorchLearner code rank 0 runs, on the same arguments.

        `sleep` and `wake` enter through the sync body their async door verb
        also calls, which is what makes an awaitable verb and a broadcast
        command the same work (#82)."""
        if command.verb == "install":
            super().install(*command.args)
        elif command.verb == "uninstall":
            super().uninstall(*command.args)
        elif command.verb == "forward_backward":
            super().forward_backward(*command.args)
        elif command.verb == "optim_step":
            super().optim_step(*command.args)
        elif command.verb == "load":
            super().load(*command.args)
        elif command.verb == "sleep":
            self.hand_the_device_back()
        elif command.verb == "wake":
            self.take_the_device_back()
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

    def shutdown(self) -> None:
        """A resident's last verb: end the chorus (the ladder inside the
        resident's own ladder — ADR 0002, Q9)."""
        self.stop()

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
                      grad_clip: float = 1.0,
                      checkpoint_activations: bool = True,
                      memory_fraction: float | None = None) -> FsdpTorchLearner:
    """Rank 0's constructor: start the chorus, then the learner in front of it.

    What a learner resident builds when its regime is sharded (residents.py,
    build_learner) — the learner looks exactly like a TorchLearner from the
    outside, reports `fsdp=width` as its build fact, and the host's regimes
    attest that against the partition it was born on. `memory_fraction` is
    the partition's: rank 0's own cap is the resident's to set before this
    runs, and every follower caps ITS device to the same number here."""
    follower = functools.partial(_follow_rank, dtype=dtype, grad_clip=grad_clip,
                                 checkpoint_activations=checkpoint_activations,
                                 memory_fraction=memory_fraction)
    ranks = RankGroup.lead(width, follower)
    return FsdpTorchLearner(ranks, dtype=dtype, grad_clip=grad_clip,
                            checkpoint_activations=checkpoint_activations)


def _follow_rank(rank: int, width: int, port: int, *, dtype: torch.dtype,
                 grad_clip: float, checkpoint_activations: bool = True,
                 memory_fraction: float | None = None) -> None:
    """A non-zero rank's whole life, and the entry point of its process.

    Spawned (so this module is imported fresh here): join the group, cap this
    rank's device at the partition's fraction, build the same learner over
    it, run what rank 0 announces, exit when it says stop. A failure is
    printed and ends the process — rank 0 then fails at its next collective
    rather than waiting out the rendezvous timeout in silence."""
    group = RankGroup.join(rank, width, port)
    if memory_fraction is not None:
        cap_memory(memory_fraction, devices=(rank,))
    try:
        FsdpTorchLearner(group, dtype=dtype, grad_clip=grad_clip,
                         checkpoint_activations=checkpoint_activations
                         ).follow_until_stopped()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        group.leave()
