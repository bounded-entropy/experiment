"""The plan: which trajectories form which group of which wave, as DATA.

A wave is the data of exactly one update; a group is the scope one
postprocessor sees. Neither is derived from task identity, so the shape of a
run is not something the runner computes — it is something the run was handed,
content-addressed like its tasks and pinned into identity.

    RunPlan   the waves, in order: line u is update u's wave
    WavePlan  one wave's groups — or a WaveRef to another wave, whole
    GroupPlan one group's leaves, under a key the postprocessor sees
    Leaf      one trajectory-to-be: SAMPLE it, or REPLAY one already sealed

Sample invokes an experiment-owned function on a declared Task. Continue
invokes one on an already sealed trajectory; the function owns both deriving
its next Task and interacting with the model. Replay takes existing sealed
rows. Legacy environment/maker leaves remain readable as historical records
and have no executor.

Plans are written as jsonl — one line per wave — so a plan reads and diffs
like the waves it describes, and a long run's plan streams rather than loads.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

TRAIN = "train"          # the default role: the row trains the policy as sealed
EVAL = "eval"            # the row is evaluation traffic the loss masks out


def route_of_role(role: str) -> str | None:
    """THE ROLE→ROUTE RULE (ADR 0018): a leaf role that is neither `train`
    nor `eval` names the SET the row trains under — `dreamer`, `memory:03` —
    and `realize` stamps it into the row's turn facts as `route`, beside the
    role itself. The two default roles stamp no route: `train` rows keep
    whatever route the record carries, `eval` rows likewise (they were
    sampled under a route and the loss masks them)."""
    return None if role in (TRAIN, EVAL) else role


class PlanError(ValueError):
    """A plan that cannot be realized as written."""


@dataclass(frozen=True)
class Sample:
    """Invoke a declared async function on a task from the run's task sets."""

    task_id: str
    function: str
    role: str = TRAIN
    dependencies: tuple[str, ...] = ()

    def row(self) -> dict:
        return {"leaf": "sample", "task": self.task_id,
                "function": self.function, "role": self.role,
                **({"dependencies": list(self.dependencies)} if self.dependencies else {})}


@dataclass(frozen=True)
class HistoricalSample:
    """An old environment leaf, preserved for reading and never executed."""

    task_id: str
    env: str
    role: str = TRAIN

    def row(self) -> dict:
        return {"leaf": "sample", "task": self.task_id,
                "env": self.env, "role": self.role}


@dataclass(frozen=True)
class Replay:
    """Take this trajectory: one already sealed, named by ref.

    Four ref shapes, one resolver: `self://rollouts/<r>#<i>` (this run's own
    generated wave), `store://<run_id>/waves/<u>#<i>` (another run's sealed
    wave), `store://<run_id>/rollouts/<r>#<i>` (another run's sealed ROLLOUT —
    what a generation-only run leaves), `cas://<sha>#<i>` (a fixed trajectory
    file). Only the first can answer "not yet" — the others exist before the
    run starts, which is why the submit gate can check them.
    """

    ref: str
    role: str = TRAIN

    def row(self) -> dict:
        return {"leaf": "replay", "ref": self.ref, "role": self.role}


@dataclass(frozen=True)
class Continue:
    """Invoke a declared async function on one sealed source trajectory.

    A self source names an earlier rollout wave. The function owns deriving
    any new Task and returns a Rollout for the framework to seal.
    """

    source: str
    function: str
    role: str = TRAIN
    dependencies: tuple[str, ...] = ()

    def row(self) -> dict:
        return {"leaf": "continue", "source": self.source,
                "function": self.function, "role": self.role,
                **({"dependencies": list(self.dependencies)} if self.dependencies else {})}


@dataclass(frozen=True)
class HistoricalDerive:
    """An old maker/environment leaf, preserved for reading only."""

    source: str
    maker: str
    env: str
    role: str = TRAIN

    def row(self) -> dict:
        return {"leaf": "derive", "source": self.source, "maker": self.maker,
                "env": self.env, "role": self.role}


Leaf = Sample | Continue | Replay | HistoricalSample | HistoricalDerive


@dataclass(frozen=True)
class GroupPlan:
    """One group: the leaves whose trajectories a postprocessor sees together.

    `key` is the group's name in the wave and in every postdata column — it is
    ASSIGNED here, never derived, so groups of one task, groups spanning tasks,
    and many groups over one task are all just different plans.
    """

    key: str
    leaves: tuple[Leaf, ...]

    def row(self) -> dict:
        return {"key": self.key, "leaves": [leaf.row() for leaf in self.leaves]}


@dataclass(frozen=True)
class WavePlan:
    """One update's groups."""

    groups: tuple[GroupPlan, ...]

    def row(self) -> dict:
        return {"wave": [group.row() for group in self.groups]}

    def leaves(self) -> Iterator[Leaf]:
        for group in self.groups:
            yield from group.leaves


@dataclass(frozen=True)
class WaveRef:
    """This wave IS that wave: the same groups, the same order, resolved.

    The common case (train on exactly what was just sampled) written once
    instead of restated leaf by leaf. It expands at realization into the
    referenced wave's groups, so nothing downstream knows the difference.
    """

    ref: str

    def row(self) -> dict:
        return {"wave_ref": self.ref}


Waves = WavePlan | WaveRef


@dataclass(frozen=True)
class RunPlan:
    """The waves, in order: `waves[u - 1]` is update u's.

    Its LENGTH is the run's length — a run ends when its plan is exhausted,
    which is why n_updates is not a schedule knob.
    """

    waves: tuple[Waves, ...]

    def __len__(self) -> int:
        return len(self.waves)

    def wave(self, update: int) -> Waves:
        """Update u's wave, 1-based like the ledger."""
        if not 1 <= update <= len(self.waves):
            raise PlanError(
                f"update {update} is outside this plan's {len(self.waves)} waves")
        return self.waves[update - 1]


# ---------------------------------------------------------------------------
# the wire: one line per wave
# ---------------------------------------------------------------------------

def encode(plan: RunPlan) -> bytes:
    """jsonl, one wave per line, keys sorted — the bytes are the identity."""
    return "".join(json.dumps(wave.row(), sort_keys=True) + "\n"
                   for wave in plan.waves).encode("utf-8")


def wave_count(data: bytes) -> int:
    """How many waves a plan's bytes describe — one line per wave, so a run's
    LENGTH is readable without decoding a single leaf. What an observer needs
    to say how far along a run is, now that no schedule states n_updates."""
    return sum(1 for line in data.decode("utf-8").splitlines() if line)


def decode(data: bytes) -> RunPlan:
    return RunPlan(tuple(_wave(json.loads(line))
                         for line in data.decode("utf-8").splitlines() if line))


def _wave(row: dict) -> Waves:
    if "wave_ref" in row:
        return WaveRef(row["wave_ref"])
    return WavePlan(tuple(
        GroupPlan(group["key"], tuple(_leaf(leaf) for leaf in group["leaves"]))
        for group in row["wave"]))


def _leaf(row: dict) -> Leaf:
    if row["leaf"] == "sample":
        if "function" in row:
            return Sample(row["task"], row["function"], row.get("role", TRAIN),
                          tuple(row.get("dependencies", ())))
        return HistoricalSample(row["task"], row["env"], row.get("role", TRAIN))
    if row["leaf"] == "replay":
        return Replay(row["ref"], row.get("role", TRAIN))
    if row["leaf"] == "continue":
        return Continue(row["source"], row["function"], row.get("role", TRAIN),
                        tuple(row.get("dependencies", ())))
    if row["leaf"] == "derive":
        return HistoricalDerive(row["source"], row["maker"], row["env"],
                      row.get("role", TRAIN))
    raise PlanError(f"unknown leaf kind {row['leaf']!r}; plans hold "
                    f"'sample' (make it), 'replay' (take it) and "
                    f"'continue' (interact from sealed source)")
