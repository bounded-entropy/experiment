"""RunPod slots and addresses; pods are started and stopped by hand (ADR 0021).

rlstack holds no RunPod credential and calls no RunPod API. A person starts
each pod from the prepared image with the shared network volume and global
networking; its worker registers with the desk at the endpoint the config
names for its slot. Every pod — the desk's CPU pod and each GPU pod — mounts
the volume at the same path, so the store is one LocalStore tree there.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re

from rlstack.runner.residents import Builds
from rlstack.runner.venues.serving import check_origin

NOT_ON_RUNPOD = ("{verb} is not implemented on RunPod: rlstack does not provision "
                 "or stop pods. {by_hand}")


class RunPodProvider:
    """The AllocationProvider contract, refused: nothing here buys or ends a pod.

    The desk is built with no bootable metal, so the shared Desk refuses and
    journals a knock before it reaches `boot`; `terminate` is reached from a
    release, whose best-effort path prints this note and records the pod as
    not terminated. A released or idle pod bills until a person stops it."""

    def boot(self, name: str) -> None:
        raise NotImplementedError(NOT_ON_RUNPOD.format(
            verb=f"booting metal {name!r}",
            by_hand="Start its pod by hand; the worker registers itself."))

    async def terminate(self, allocation_id: str) -> bool:
        raise NotImplementedError(NOT_ON_RUNPOD.format(
            verb=f"stopping pod {allocation_id!r}",
            by_hand="Stop it by hand in the RunPod console; it bills until then."))


def desk_token(config: "DeskConfig") -> str:
    """The desk's bearer, for an operator or a worker: `RLSTACK_HTTP_TOKEN`
    when the environment has it, else the desk's own token file where its
    state dir is mounted (every pod on the volume). Nothing is minted here: a
    fresh token is one the desk would never accept."""
    told = os.environ.get("RLSTACK_HTTP_TOKEN", "")
    if told:
        return told
    path = config.state_dir / "http-token"
    if not path.exists():
        raise RuntimeError(f"the desk's token is not here: start the desk first, or export "
                           f"RLSTACK_HTTP_TOKEN from {path} on a pod that mounts the volume")
    return path.read_text().strip()


@dataclass(frozen=True)
class MetalConfig:
    """A hand-started pod's slot: the name it registers under and the origin
    every caller dials it at. The card is measured, never declared."""
    name: str
    endpoint: str
    idle_s: float = 300.0
    builds: Builds | None = None
    environment: dict[str, str] | None = None
    port: int | None = None
    """The port this slot's worker listens on; None is the desk config's
    `worker_port`. Two slots on ONE machine (one worker per GPU, each pinned
    by its environment's CUDA_VISIBLE_DEVICES) need two ports."""

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", self.name):
            raise ValueError("metal name must contain only letters, digits, '_' and '-'")
        check_origin(self.endpoint, f"metal {self.name!r} endpoint")
        if self.port is not None and not 1 <= self.port <= 65535:
            raise ValueError(f"metal {self.name!r} port must be a valid port")
        if not math.isfinite(self.idle_s) or self.idle_s <= 0:
            raise ValueError("automatic idle release must be finite and positive")
        if self.environment is not None and not all(
                isinstance(key, str) and isinstance(value, str) for key, value in self.environment.items()):
            raise ValueError("worker environment names and values must be strings")

    @property
    def address(self) -> str:
        """The metal's plane: its own endpoint, with no relay in between."""
        return self.endpoint.rstrip("/")

    def host_address(self, host: str, epoch: str) -> str:
        return f"{self.address}#{host}@{epoch}"


@dataclass(frozen=True)
class DeskConfig:
    """The desk's pod, the shared volume's store, and the slots it may attach."""
    store: str
    desk_endpoint: str
    metals: tuple[MetalConfig, ...]
    state_dir: Path
    desk_port: int = 18760
    observer_port: int = 8321
    worker_port: int = 8000
    idle_s: float = 300.0
    host_idle_s: float | None = None
    """How long a carved host may sit idle before the desk decarves it (ADR 0014,
    Q9); None is the metal's own limit. A hand-started pod keeps its worker
    for the metal's limit, so a short host limit is what hands an idle
    engine's GB to the next run's learner."""
    operator_endpoint: str = ""
    """Where the operator's verbs reach the desk from outside RunPod's private
    network: its HTTPS proxy origin. Empty means the operator is on that
    network (on the desk pod itself) and uses `desk_endpoint`."""

    def __post_init__(self) -> None:
        if not Path(self.store).is_absolute():
            raise ValueError("store must be an absolute path on the network volume, "
                             "the same path on every pod")
        check_origin(self.desk_endpoint, "desk_endpoint")
        if self.operator_endpoint:
            check_origin(self.operator_endpoint, "operator_endpoint")
        ports = (self.desk_port, self.observer_port)
        if len(set(ports)) != len(ports) or not all(1 <= port <= 65535 for port in
                                                     (*ports, self.worker_port)):
            raise ValueError("desk and observer need distinct valid ports, and workers a valid one")
        if len({metal.name for metal in self.metals}) != len(self.metals):
            raise ValueError("metal names must be unique")
        if not math.isfinite(self.idle_s) or self.idle_s <= 0:
            raise ValueError("automatic idle release must remain finite")
        if self.host_idle_s is not None and (not math.isfinite(self.host_idle_s)
                                             or self.host_idle_s <= 0):
            raise ValueError("host_idle_s must be finite and positive")

    @property
    def operator_door(self) -> str:
        return self.operator_endpoint or self.desk_endpoint

    def metal(self, name: str) -> MetalConfig:
        for metal in self.metals:
            if metal.name == name:
                return metal
        raise ValueError(f"metal {name!r} is not a configured slot: "
                         f"{', '.join(m.name for m in self.metals) or 'none'}")

    @classmethod
    def read(cls, path: str | Path) -> "DeskConfig":
        row = json.loads(Path(path).read_text())
        metals = []
        for raw in row.pop("metals"):
            raw = dict(raw)
            if raw.get("builds") is not None:
                raw["builds"] = Builds.from_row(raw["builds"])
            metals.append(MetalConfig(**raw))
        row["state_dir"] = Path(row["state_dir"])
        return cls(metals=tuple(metals), **row)
