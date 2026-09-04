"""The plan: which trajectories form which group of which wave, as DATA.

A wave is the data of exactly one update; a group is the scope one
postprocessor sees. Neither is derived from task identity, so the shape of a
run is not something the runner computes — it is something the run was handed,
content-addressed like its tasks and pinned into identity.

    RunPlan   the waves, in order: line u is update u's wave
    WavePlan  one wave's groups — or a WaveRef to another wave, whole
    GroupPlan one group's leaves, under a key the postprocessor sees
    Leaf      one trajectory-to-be: SAMPLE it, or REPLAY one already sealed

THE LEAF IS THE WHOLE TAXONOMY. `Sample` names a task to run under an
environment, and until someone runs it the wave does not exist yet — that
"not yet" is the only thing daemons ever await. `Replay` names a trajectory
that is already sealed somewhere: this run's own rollouts, another run's
waves, a content-addressed file. `Derive` is mint-then-make: a registered
task MAKER turns an already-sealed trajectory into a NEW task (its critique
prompt, its retry request), and the environment runs that — the reflect
loop's leaf, and the reason a plan can pin an iterative run's whole SHAPE
while its content is only derivable at runtime. Live / replay / static /
iterative stop being four feeds and become three leaf constructors, which is
what lets one wave hold any mix and one run go SFT then RL then reflect.

Plans are written as jsonl — one line per wave — so a plan reads and diffs
like the waves it describes, and a long run's plan streams rather than loads.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

TRAIN = "train"          # the only role the gradient path understands (v0)


class PlanError(ValueError):
    """A plan that cannot be realized as written."""


@dataclass(frozen=True)
class Sample:
    """Make this trajectory: run `env` on `task_id`, under this run's sampling.

    The task's CONTENT (prompt, answer, hints) lives in the run's task sets;
    a leaf carries the id alone, so one task set serves many plans.
    """

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
class Derive:
    """Mint it, then make it: a registered task maker turns the sealed
    trajectory at `source` into a NEW task, and `env` runs an episode on it.

    `source` speaks Replay's ref grammar (`self://rollouts/<r>#<i>` and the
    rest); within a rollout plan a self source must name an EARLIER wave —
    the generator makes waves in order, so an earlier wave is always sealed
    by the time this leaf runs, and a later one is a plan error, not an
    await. The maker is registered code (@task_maker): its source hashes into
    run identity through the spec's `gen.makers` declaration, and it is a
    PURE function of the sealed bytes, so a resumed run re-mints the same
    task to the byte.
    """

    source: str
    maker: str
    env: str
    role: str = TRAIN

    def row(self) -> dict:
        return {"leaf": "derive", "source": self.source, "maker": self.maker,
                "env": self.env, "role": self.role}


Leaf = Sample | Replay | Derive


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
        return Sample(row["task"], row["env"], row.get("role", TRAIN))
    if row["leaf"] == "replay":
        return Replay(row["ref"], row.get("role", TRAIN))
    if row["leaf"] == "derive":
        return Derive(row["source"], row["maker"], row["env"],
                      row.get("role", TRAIN))
    raise PlanError(f"unknown leaf kind {row['leaf']!r}; plans hold "
                    f"'sample' (make it), 'replay' (take it) and "
                    f"'derive' (mint it, then make it)")
