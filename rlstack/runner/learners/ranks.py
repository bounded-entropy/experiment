"""The rank chorus: the extra processes a sharded build needs, and no more.

The blackboard stays ONE async process. Rank 0 runs the runner and answers;
ranks 1..width-1 exist only to stand in the collectives a sharded forward
requires — they never touch the store and decide nothing. Two rules shape
everything here:

    A COLLECTIVE VERB MUST BE ENTERED BY EVERY RANK, IN THE SAME ORDER.

so a verb touching a collective is announced before it runs locally, and a verb
touching none is not announced at all — a broadcast that buys nothing is a
deadlock waiting for its first caller.

    A CHORUS ENDS ONLY WHEN EVERY RANK IS ACTUALLY GONE.

A follower spends its life blocked inside a collective, where no signal handler
gets a turn, so the polite word cannot be trusted and SIGKILL is what shutdown
actually promises. Anything less leaves a daemonic child the interpreter's own
exit will then join forever.

The group is deliberately small-minded: one node, one rank per visible device,
rank r on cuda:r — a host is a partition, the container is that partition, and
its device indices are its own.
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

from rlstack.runner.residents import (  # noqa: F401 — Teardown is re-exported
    GRACE_S, SIGNAL_GRACE_S, Teardown, escalate, join_survivors, living,
)

STOP = "stop"
"""The one command that is not a Learner verb: leave the loop, exit."""


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


@dataclass
class RankGroup:
    """One process group: this rank's place in it, and how it hears verbs.

    Built by exactly two named entries — `lead` on rank 0, which starts the
    others, and `join` in each child — so a process's place is a fact of how
    it was constructed, never a flag it consults later.
    """

    rank: int
    width: int
    port: int
    children: tuple[torch.multiprocessing.Process, ...] = ()
    _mesh: Any = field(default=None, repr=False)
    _stopped: bool = False

    # ---- bring-up, one named entry per place --------------------------------

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
        """The ladder — left on its own, SIGTERM, SIGKILL, each rung joining
        what it signalled — is every GPU-holding child's, so it lives in
        runner/residents.py (ADR 0002, Q9) and the chorus applies it to its
        ranks. A rank blocked inside a collective is not a rank that will
        notice a polite signal: NCCL is down in a driver call, so SIGTERM is
        the request and SIGKILL is the answer. Position IS rank: children
        are spawned in rank order."""
        return escalate(self.children, grace_s=grace_s,
                        signal_grace_s=signal_grace_s)

    def living(self) -> tuple[tuple[int, torch.multiprocessing.Process], ...]:
        return living(self.children)

    def join_survivors(self, timeout_s: float) -> tuple[
            tuple[int, torch.multiprocessing.Process], ...]:
        return join_survivors(self.children, timeout_s)

    def leave(self) -> None:
        """A follower's own teardown, after it hears STOP."""
        if self.width > 1:
            dist.destroy_process_group()
