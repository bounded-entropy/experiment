"""The rank chorus: the extra processes a sharded build needs, and no more.

The blackboard stays ONE async process. Rank 0 runs the runner — the daemons,
the store, the five Learner verbs — and ranks 1..width-1 exist only to stand
in the collectives that a sharded base forward requires. They are not a
second scheduler, they never touch the store, and they decide nothing: rank 0
ANNOUNCES the verb it is about to run, every rank runs it on identical
inputs, and rank 0 alone answers.

One rule shapes everything here:

    A COLLECTIVE VERB MUST BE ENTERED BY EVERY RANK, IN THE SAME ORDER.

So a verb that touches a collective is broadcast before it runs locally, and
a verb that touches none (emit, over replicated deltas) is not broadcast at
all — a broadcast that buys nothing is a deadlock waiting for the day someone
calls it from one rank.

A second rule shapes the end of it:

    A CHORUS ENDS ONLY WHEN EVERY RANK IS ACTUALLY GONE.

A follower spends its life blocked inside a collective, where no signal
handler gets a turn, so the polite word is tried first and SIGKILL is what
the shutdown actually promises. Anything less leaves a daemonic child that
the interpreter's own exit will then join forever (#53).

The group is deliberately small-minded: one node, one rank per visible
device, rank r on cuda:r. A host is a partition of some GPUs (#43), the
container is that partition, and its device indices are its own.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

STOP = "stop"
"""The one command that is not a Learner verb: leave the loop, exit."""

GRACE_S = 10.0
"""How long a child gets to hear STOP and leave on its own, and how long rank
0 waits for its own farewell to go out. Generous for a broadcast, short
because the whole teardown has to fit inside the venue's shutdown grace —
Modal's is 30 seconds, and what does not finish inside it is killed with its
stdout unflushed (#53)."""

SIGNAL_GRACE_S = 5.0
"""How long a signalled child gets to die before the next rung of the ladder.
A process that will die at all dies immediately here; the wait is for the
kernel, not for the process."""


@dataclass(frozen=True)
class RankCommand:
    """One verb the whole chorus must run, with the arguments it runs on.

    Picklable by construction, because `args` only ever carries what the
    Learner protocol already passes: the tenant id, an ExperimentSpec, site
    metadata, a TokenBatch, payload bytes. Nothing here holds a live handle —
    a command crosses processes, so it may not name a model or a store.
    """

    verb: str
    args: tuple = ()


def free_port() -> int:
    """A port the rendezvous can own. Asked of the OS rather than fixed, so
    two chorusses in one container (a resume after a kill) never collide."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@dataclass(frozen=True)
class Teardown:
    """What ending the chorus actually cost: one field per rung of the ladder.

    Returned by `RankGroup.stop` and, when it is not `graceful`, printed as
    one line. A rank that had to be killed is a fact about the run, and the
    silence that hid it is exactly what made the first wedge (#53) look like
    a container that died for no reason.
    """

    heard_the_farewell: bool = True
    deaf: tuple[int, ...] = ()      # still there after STOP → SIGTERM
    wedged: tuple[int, ...] = ()    # still there after SIGTERM → SIGKILL
    lost: tuple[int, ...] = ()      # still there after SIGKILL

    @property
    def graceful(self) -> bool:
        """Every rank heard STOP and left by itself — the only ending that
        leaves the process group intact enough to be destroyed."""
        return (self.heard_the_farewell
                and not (self.deaf or self.wedged or self.lost))

    def line(self) -> str:
        """The one honest line, in escalation order."""
        parts = []
        if not self.heard_the_farewell:
            parts.append("STOP never went out (the chorus was past hearing)")
        if self.deaf:
            parts.append(f"ranks {list(self.deaf)} ignored STOP → SIGTERM")
        if self.wedged:
            parts.append(f"ranks {list(self.wedged)} survived SIGTERM → SIGKILL")
        if self.lost:
            parts.append(f"ranks {list(self.lost)} SURVIVED SIGKILL")
        return "[chorus] teardown: " + "; ".join(parts)


@dataclass
class RankGroup:
    """One process group: this rank's place in it, and how it hears verbs.

    Built by exactly two named entries — `lead` on rank 0, which starts the
    others, and `join` in each child — so a process's role is a fact of how
    it was constructed, never a flag it consults later.
    """

    rank: int
    width: int
    port: int
    children: tuple[torch.multiprocessing.Process, ...] = ()
    _mesh: Any = field(default=None, repr=False)
    _stopped: bool = False

    # ---- bring-up, one named entry per role ---------------------------------

    @classmethod
    def lead(cls, width: int, follower: Callable[..., None], *,
             timeout_s: float = 1800.0) -> RankGroup:
        """Rank 0: spawn ranks 1..width-1, then take rank 0's own place.

        Spawn, never fork: a child builds its own CUDA context on its own
        device, and a forked one would inherit this process's. width == 1 is
        a legal degenerate chorus — no children, no process group, so an
        unsharded build costs nothing to express."""
        if width == 1:
            return cls(rank=0, width=1, port=0)
        port = free_port()
        context = torch.multiprocessing.get_context("spawn")
        children = tuple(
            context.Process(target=follower, args=(rank, width, port),
                            daemon=True)
            for rank in range(1, width))
        for child in children:
            child.start()
        return cls._take_place(0, width, port, children=children,
                               timeout_s=timeout_s)

    @classmethod
    def join(cls, rank: int, width: int, port: int, *,
             timeout_s: float = 1800.0) -> RankGroup:
        """Every other rank: take the place rank 0 spawned this process for."""
        return cls._take_place(rank, width, port, timeout_s=timeout_s)

    @classmethod
    def _take_place(cls, rank: int, width: int, port: int, *,
                    children: tuple[torch.multiprocessing.Process, ...] = (),
                    timeout_s: float) -> RankGroup:
        """The rendezvous itself: this rank's device, then the group.

        The timeout is generous because the first collective after `install`
        waits out a model load (a large base downloads and loads before any
        rank reaches an all-gather) — a rendezvous that expired there would
        report a network fault for what is a slow disk."""
        torch.cuda.set_device(rank)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://127.0.0.1:{port}",
            world_size=width, rank=rank,
            timeout=timedelta(seconds=timeout_s))
        return cls(rank=rank, width=width, port=port, children=children)

    # ---- what a rank IS ------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    @property
    def device_str(self) -> str:
        """What TorchLearner takes as its `device` — spelled with an index,
        because "cuda" means "the current device" and several ranks in one
        container must never agree about which that is."""
        return f"cuda:{self.rank}"

    def mesh(self):
        """The one-dimensional device mesh FSDP shards over. Built once and
        kept: a mesh is the group's shape, and the group's shape is a birth
        fact (#43)."""
        if self._mesh is None:
            from torch.distributed.device_mesh import init_device_mesh
            self._mesh = init_device_mesh("cuda", (self.width,))
        return self._mesh

    # ---- the wire between ranks ---------------------------------------------

    def announce(self, command: RankCommand) -> None:
        """Rank 0 tells every rank which verb is about to run. A width-1
        chorus has nobody to tell, and says nothing."""
        if self.width == 1:
            return
        dist.broadcast_object_list([command], src=0, device=self.device)

    def hear(self) -> RankCommand:
        """A follower's blocking wait for the next verb.

        STOP is the sentinel on this channel and the ONLY way out of it that
        the follower itself can take: the wait is a collective inside the NCCL
        driver, where no signal handler runs and no other process can close
        anything under it — a follower that misses its STOP waits out the
        group's whole timeout. `stop` is written knowing that."""
        box: list[RankCommand | None] = [None]
        dist.broadcast_object_list(box, src=0, device=self.device)
        command = box[0]
        if command is None:
            raise RuntimeError("the chorus received an empty command frame")
        return command

    # ---- teardown ------------------------------------------------------------

    def stop(self, *, grace_s: float = GRACE_S,
             signal_grace_s: float = SIGNAL_GRACE_S) -> Teardown:
        """Rank 0 ends the chorus, and does not leave without the children.

        Idempotent, because a run that failed and a run that finished both
        arrive here — and BOUNDED, because a child that outlives this call is
        not merely metal left held. The children are daemonic, so
        multiprocessing's exit handler joins them with NO timeout when the
        interpreter finally goes down: one rank still sitting in `hear` hangs
        the process's own exit until the venue kills the container and takes
        the unflushed stdout with it. That is #53's lost report, and the
        reason every rung of this ladder joins what it signalled.

        The budget is a parameter because it is a promise to whoever will kill
        this container: worst case grace_s + 2 x signal_grace_s, then done."""
        if self._stopped or self.rank != 0:
            return Teardown()
        self._stopped = True
        if self.width == 1:
            return Teardown()
        teardown = self.end_the_children(grace_s=grace_s,
                                         signal_grace_s=signal_grace_s)
        if teardown.graceful:
            # every rank left of its own accord, so there is still a group to
            # agree with; after a kill there is not, and the survivors' NCCL
            # communicators died with them.
            dist.destroy_process_group()
        else:
            print(teardown.line(), flush=True)
        return teardown

    def farewell(self, timeout_s: float) -> bool:
        """Say STOP without betting the shutdown on its being heard.

        The announce is itself a collective: against a dead or mis-sequenced
        follower it never matches and blocks until the process group's own
        timeout (1800s by default) — so the farewell that exists to END the
        chorus would be the thing that wedges rank 0. It goes out on a daemon
        thread with a deadline instead; not returning in time — or raising on
        a group that is already broken — means the chorus is past hearing, and
        the ladder below is what ends it."""
        spoken: list[bool] = []

        def say_it() -> None:
            self.announce(RankCommand(STOP))
            spoken.append(True)

        thread = threading.Thread(target=say_it, daemon=True)
        thread.start()
        thread.join(timeout_s)
        return bool(spoken)

    def end_the_children(self, *, grace_s: float,
                         signal_grace_s: float) -> Teardown:
        """The polite half of a teardown and the forceful half, in that order.

        The farewell and the wait for it share ONE graceful deadline, so the
        polite half costs grace_s whether the STOP went out in a millisecond
        or never went out at all — and the whole teardown is then bounded by
        grace_s + 2 x signal_grace_s."""
        graceful_until = time.monotonic() + grace_s
        heard = self.farewell(grace_s)
        teardown = self.escalate(
            grace_s=max(0.0, graceful_until - time.monotonic()),
            signal_grace_s=signal_grace_s)
        return replace(teardown, heard_the_farewell=heard)

    def escalate(self, *, grace_s: float, signal_grace_s: float) -> Teardown:
        """The ladder, in the order a shutdown should try it: left on its own,
        SIGTERM, SIGKILL — and a record of which rung each rank needed.

        A rank blocked inside a collective is not a rank that will notice a
        polite signal: NCCL is down in a driver call, the interpreter runs no
        bytecode until it returns, and SIGTERM waits for a handler that never
        gets a turn. So SIGTERM is the request and SIGKILL is the answer —
        each rung JOINS what it signalled, because a teardown that signals and
        walks away is precisely what left #53's child alive — and a rank that
        survives even the kill is reported rather than pretended away."""
        deaf = self.join_survivors(grace_s)
        for _, child in deaf:
            child.terminate()
        wedged = self.join_survivors(signal_grace_s) if deaf else ()
        for _, child in wedged:
            child.kill()
        lost = self.join_survivors(signal_grace_s) if wedged else ()
        return Teardown(deaf=tuple(rank for rank, _ in deaf),
                        wedged=tuple(rank for rank, _ in wedged),
                        lost=tuple(rank for rank, _ in lost))

    def living(self) -> tuple[tuple[int, torch.multiprocessing.Process], ...]:
        """The children still running, each with its rank. Children are
        spawned in rank order, so position IS rank."""
        return tuple((rank, child)
                     for rank, child in enumerate(self.children, start=1)
                     if child.is_alive())

    def join_survivors(self, timeout_s: float) -> tuple[
            tuple[int, torch.multiprocessing.Process], ...]:
        """Wait out ONE shared deadline for every child, then say who is left.

        Shared, not per-child: the teardown budget is a wall-clock promise,
        and a width-8 chorus must not multiply it by eight."""
        deadline = time.monotonic() + timeout_s
        for _, child in self.living():
            child.join(timeout=max(0.0, deadline - time.monotonic()))
        return self.living()

    def leave(self) -> None:
        """A follower's own teardown, after it hears STOP."""
        if self.width > 1:
            dist.destroy_process_group()
