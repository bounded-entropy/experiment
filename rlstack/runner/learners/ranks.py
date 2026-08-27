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

The group is deliberately small-minded: one node, one rank per visible
device, rank r on cuda:r. A host is a partition of some GPUs (#43), the
container is that partition, and its device indices are its own.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable

import torch
import torch.distributed as dist

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
    others, and `join` in each child — so a process's role is a fact of how
    it was constructed, never a flag it consults later.
    """

    rank: int
    width: int
    port: int
    children: tuple = ()
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
                    children: tuple = (), timeout_s: float) -> RankGroup:
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
        """A follower's blocking wait for the next verb."""
        box: list[RankCommand | None] = [None]
        dist.broadcast_object_list(box, src=0, device=self.device)
        command = box[0]
        if command is None:
            raise RuntimeError("the chorus received an empty command frame")
        return command

    # ---- teardown ------------------------------------------------------------

    def stop(self) -> None:
        """Rank 0 ends the chorus: announce STOP, wait for the children, tear
        the group down. Idempotent, because a run that failed and a run that
        finished both arrive here, and a stuck child would hold the metal."""
        if self._stopped or self.rank != 0:
            return
        self._stopped = True
        if self.width == 1:
            return
        self.announce(RankCommand(STOP))
        for child in self.children:
            child.join(timeout=120.0)
            if child.is_alive():
                child.terminate()
        dist.destroy_process_group()

    def leave(self) -> None:
        """A follower's own teardown, after it hears STOP."""
        if self.width > 1:
            dist.destroy_process_group()
