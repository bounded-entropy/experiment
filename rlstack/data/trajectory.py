"""The sealed data objects: Task, Trajectory, Turn, Group, Wave, and their rows.

A Trajectory is one sealed episode — frozen training data, what a Rollout
becomes at the seal. The membrane (I1) is the type system: training code takes
Trajectory and cannot receive a half-finished rollout by construction.
Recording is loss-independent and happens at the seal; generated token ids are
the engine's own and are NEVER re-tokenized.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Sequence


class DataError(ValueError):
    """A data-object invariant was violated."""


FINISH_REASONS = frozenset({"stop", "eos", "length"})


class Role(enum.StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """One chat message."""

    role: Role
    content: str


@dataclass(frozen=True)
class Task:
    """One problem: an id, a prompt, and the metadata a verifier or hint reads."""

    id: str
    prompt: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Turn:
    """One request inside a trajectory, and everything recorded at the seal (I6).

    A Turn is one REQUEST: one pinned bundle, one seed, one contiguous KV. The
    bank's kinds record their own facts at whatever granularity they need —
    `token_extras` per generated token (the adapter index drawn at each token),
    `turn_extras` per request (the latent drawn for a probabilistic soft
    prompt). Recorded, never re-derived: replay must reproduce them exactly, so
    they are sealed as data.
    """

    message: Message
    token_ids: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    finish: str
    stop_hit: str | None
    bundle_id: str
    policy_version: Mapping[str, int]
    seed: int
    token_extras: Mapping[str, tuple] = field(default_factory=dict)
    turn_extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.behavior_logprobs):
            raise DataError(
                f"token_ids/behavior_logprobs length mismatch: "
                f"{len(self.token_ids)} != {len(self.behavior_logprobs)}"
            )
        if self.finish not in FINISH_REASONS:
            raise DataError(f"finish={self.finish!r} not in {sorted(FINISH_REASONS)}")
        for name, column in self.token_extras.items():
            if len(column) != len(self.token_ids):
                raise DataError(
                    f"token_extras[{name!r}] length {len(column)} != "
                    f"len(token_ids)={len(self.token_ids)}"
                )


@dataclass(frozen=True)
class Trajectory:
    """One sealed episode — training data, immutable (I1).

    Constructed by Rollout.seal() on the live path, or by trajectory_from_row
    when reading the store. The record holds ONLY what the policy did: rewards,
    advantages and judge scores are computed ABOUT it afterwards by the
    postprocessor pipeline and live in postdata — never in here.
    """

    task: Task
    messages: tuple[Message, ...]
    turns: tuple[Turn, ...]
    env_extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Concatenated content of all messages."""
        return "".join(m.content for m in self.messages)


# ---------------------------------------------------------------------------
# the training-side units of scope: Trajectory ⊂ Group ⊂ Wave
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False)
class Group:
    """One partial contribution to the loss, and the scope a postprocessor sees.

    In GRPO the group is what the advantage baseline is computed over; in a
    preference loss it is the compared pair. `key` is ASSIGNED at wave assembly
    — usually the task id, but membership is never derived from task identity,
    so many groups of one task (test-time training) or groups spanning tasks
    are equally expressible. Sealed trajectories only (I1).
    """

    key: str
    trajectories: tuple[Trajectory, ...]

    def __init__(self, key: str, trajectories: Sequence[Trajectory]) -> None:
        items = tuple(trajectories)
        if not items:
            raise DataError(f"group {key!r} is empty")
        unsealed = [i for i, t in enumerate(items)
                    if not isinstance(t, Trajectory)]
        if unsealed:  # e.g. a Rollout that was never sealed
            raise DataError(
                f"group {key!r} contains unsealed items at indices {unsealed}; "
                f"training consumes Trajectory only (I1)")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "trajectories", items)

    def __len__(self) -> int:
        return len(self.trajectories)


@dataclass(frozen=True, init=False)
class Wave:
    """The data of exactly one update: every group needed for one gradient step.

    Trajectory order is the concatenation of groups, and everything downstream
    (postdata columns, flats, packing) aligns to that order.
    """

    groups: tuple[Group, ...]

    def __init__(self, groups: Sequence[Group]) -> None:
        items = tuple(groups)
        keys = [g.key for g in items]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise DataError(f"duplicate group keys in wave: {duplicates}")
        object.__setattr__(self, "groups", items)

    @property
    def trajectories(self) -> tuple[Trajectory, ...]:
        return tuple(t for g in self.groups for t in g.trajectories)

    def __len__(self) -> int:
        return sum(len(g) for g in self.groups)


# ---------------------------------------------------------------------------
# row serialization: sealed data <-> plain JSON dicts (the store's unit)
# ---------------------------------------------------------------------------

def wave_to_rows(wave: Wave) -> list[dict[str, Any]]:
    """A wave as flat trajectory rows; each row carries its group key, so group
    structure survives the store (offline advantage re-runs need it)."""
    return [dict(trajectory_to_row(traj), group=group.key)
            for group in wave.groups for traj in group.trajectories]


def wave_from_rows(rows: Sequence[Mapping[str, Any]]) -> Wave:
    """Rebuild a wave from trajectory rows, grouping by key in first-seen order."""
    members: dict[str, list[Trajectory]] = {}
    for row in rows:
        members.setdefault(row["group"], []).append(trajectory_from_row(row))
    return Wave([Group(key, trajs) for key, trajs in members.items()])

def trajectory_to_row(traj: Trajectory) -> dict[str, Any]:
    """A trajectory as one plain-JSON row. Lossless roundtrip."""
    if not isinstance(traj, Trajectory):
        raise DataError(
            f"only sealed Trajectory reaches the store (I1), got "
            f"{type(traj).__name__} — call .seal() first")
    return {
        "task": {"id": traj.task.id, "prompt": traj.task.prompt,
                 "meta": dict(traj.task.meta)},
        "messages": [{"role": m.role.value, "content": m.content}
                     for m in traj.messages],
        # Which messages are turns' messages, by position. Matched by OBJECT
        # identity (an injected message that merely equals a turn's message must
        # not be claimed), so identity survives the JSON roundtrip.
        "turn_message_index": [
            next(i for i, m in enumerate(traj.messages) if m is t.message)
            for t in traj.turns
        ],
        "turns": [{
            "content": t.message.content,
            "role": t.message.role.value,
            "token_ids": list(t.token_ids),
            "behavior_logprobs": list(t.behavior_logprobs),
            "finish": t.finish,
            "stop_hit": t.stop_hit,
            "bundle_id": t.bundle_id,
            "policy_version": dict(t.policy_version),
            "seed": t.seed,
            "token_extras": {k: list(v) for k, v in t.token_extras.items()},
            "turn_extras": dict(t.turn_extras),
        } for t in traj.turns],
        "env_extras": dict(traj.env_extras),
    }


def trajectory_from_row(row: Mapping[str, Any]) -> Trajectory:
    """Rebuild a sealed trajectory from its row, message identity intact."""
    task = Task(id=row["task"]["id"], prompt=row["task"]["prompt"],
                meta=dict(row["task"]["meta"]))
    messages = [Message(Role(m["role"]), m["content"]) for m in row["messages"]]
    turns = []
    for spec, index in zip(row["turns"], row["turn_message_index"]):
        turns.append(Turn(
            message=messages[index],  # the SAME object as in messages (flatten keys on id())
            token_ids=tuple(spec["token_ids"]),
            behavior_logprobs=tuple(spec["behavior_logprobs"]),
            finish=spec["finish"],
            stop_hit=spec["stop_hit"],
            bundle_id=spec["bundle_id"],
            policy_version=dict(spec["policy_version"]),
            seed=spec["seed"],
            token_extras={k: tuple(v) for k, v in spec["token_extras"].items()},
            turn_extras=dict(spec["turn_extras"]),
        ))
    return Trajectory(
        task=task, messages=tuple(messages), turns=tuple(turns),
        env_extras=MappingProxyType(dict(row["env_extras"])),
    )
