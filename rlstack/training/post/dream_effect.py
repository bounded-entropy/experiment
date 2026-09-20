"""dream_effect: the prequential probe and the contrast that rewards dreams
(ADR 0018; research-log post 0007).

The pooled half of the dreams pipeline. For every TEXT row of the group (an
arrival, replayed once per memory under its `route` fact) it scores the
text's own tokens under that route — the memory's surprise at the arrival
before this update touches it — and emits it as `prequential_nll`. When the
group also holds DREAMER rows (the first half's dreams, replayed for the
dreamer's step) and the arrival's task carries a `contrast` — the partition
of the memories into one group per dream — it rewards dream k with

    reward_k = mean nll of the memories NOT in group k − mean nll of group k

a dream is worth how much less surprising the second half was to the
memories that trained on it. Rows that are neither get zeros;
`dream_advantage` subtracts the dreamer rows' mean."""
from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.plan import EVAL
from rlstack.data.trajectory import Group, Message, Trajectory, hint_for
from rlstack.policy.adapters.dream_bank import (
    BASE, DREAMER, ROUTE_RECORD, Route, is_library, parts_of,
)
from rlstack.training.post.base import PostProcessor, postprocessor

RIDGE = 1e-2


def role_of(traj: Trajectory) -> str:
    """The row's role, as realize stamped it; `train` when it stamped none."""
    for turn in traj.turns:
        if "role" in turn.turn_extras:
            return str(turn.turn_extras["role"])
    return "train"


def trains_the_dreamer(role: str) -> bool:
    """A row is the dreamer's own sample when the TRAINABLE part of its role
    is the dreamer: `dreamer` alone, or stacked on a frozen library memory
    (`lib:<name>+dreamer`, ADR 0019) — a library part is detached, so the
    row's gradient is the dreamer's either way. `dream_stream` holds the same
    rule for the loss."""
    return [part for part in parts_of(role) if not is_library(part)] == [DREAMER]


def route_of(traj: Trajectory) -> str:
    """The set the row was sampled or is replayed under."""
    for turn in traj.turns:
        if ROUTE_RECORD in turn.turn_extras:
            return str(turn.turn_extras[ROUTE_RECORD])
    return DREAMER


def kind_of(traj: Trajectory) -> str:
    return str(traj.task.meta.get("kind", ""))


async def route_scores(traj: Trajectory, client: PoolClient, route: str,
                       hinted: bool = False) -> list[float]:
    """Every generated turn's tokens scored under `route`, walking the sealed
    message stream in order (teacher_scores' walk, with a directive). With
    `hinted`, the task's hint stands in front of the stream — the conditioned
    context a dream was sampled under."""
    turn_of = {id(t.message): t for t in traj.turns}
    context: list[Message] = [hint_for(traj.task)] if hinted else []
    scores: list[float] = []
    for message in traj.messages:
        turn = turn_of.get(id(message))
        if turn is not None:
            scores.extend(await client.score(context, turn.token_ids,
                                             directives=(Route(route),)))
        context.append(message)
    return scores


def mean_nll(scores: Sequence[float]) -> float:
    return -sum(scores) / len(scores) if scores else 0.0


def contrast(nll_by_memory: Mapping[str, float], plan: Mapping) -> list[float]:
    """The group difference per dream, in dream order: the memories outside
    the dream's group minus the memories in it, so a helpful dream reads
    positive. `plan["assignment"]` maps dream index (as a string) to the
    memory routes that trained on it. A group with no scored memory, or no
    memory outside it, is worth nothing."""
    dreams = sorted(plan["assignment"], key=int)
    rewards = []
    for k in dreams:
        inside = [nll_by_memory[m] for m in plan["assignment"][k] if m in nll_by_memory]
        outside = [v for m, v in nll_by_memory.items() if m not in plan["assignment"][k]]
        if not inside or not outside:
            rewards.append(0.0)
            continue
        rewards.append(sum(outside) / len(outside) - sum(inside) / len(inside))
    return rewards


@postprocessor("dream_effect")
class DreamEffect(PostProcessor):
    """Per text row: the memory's surprise at the arriving text before this
    update (`prequential_nll`) and, as the reference the drift is read
    against, the UNTOUCHED base's surprise at the same text
    (`prequential_base`, scored once per distinct text and repeated on its
    rows); per dreamer row, the contrast reward."""

    produces = ("prequential_nll", "prequential_base", "dream_reward")
    pools = ("main",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        main = client.pool("main")
        trajectories = group.trajectories
        texts = [i for i, t in enumerate(trajectories)
                 if kind_of(t) == "text" and role_of(t) != EVAL]
        scored = await asyncio.gather(
            *(route_scores(trajectories[i], main, route_of(trajectories[i])) for i in texts))
        nll = [0.0] * len(trajectories)
        by_memory: dict[str, float] = {}
        for i, scores in zip(texts, scored):
            nll[i] = mean_nll(scores)
            by_memory[route_of(trajectories[i])] = nll[i]
        # the base's surprise, once per distinct text
        first_of: dict[str, int] = {}
        for i in texts:
            first_of.setdefault(trajectories[i].task.id, i)
        base_scored = await asyncio.gather(
            *(route_scores(trajectories[i], main, BASE) for i in first_of.values()))
        base_by_task = {task_id: mean_nll(scores)
                        for task_id, scores in zip(first_of, base_scored)}
        base = [0.0] * len(trajectories)
        for i in texts:
            base[i] = base_by_task[trajectories[i].task.id]
        reward = [0.0] * len(trajectories)
        dreamers = [i for i, t in enumerate(trajectories) if role_of(t) == DREAMER]
        plan = next((t.task.meta.get("contrast") for t in trajectories
                     if kind_of(t) == "text" and t.task.meta.get("contrast")), None)
        if dreamers and plan and by_memory:
            rewards = contrast(by_memory, plan)
            for k, i in enumerate(dreamers):
                if k < len(rewards):
                    reward[i] = rewards[k]
        return {"prequential_nll": nll, "prequential_base": base, "dream_reward": reward}
