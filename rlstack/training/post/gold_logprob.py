"""gold_logprob: the gold answer's mean token log-probability under the set
the question's task names (ADR 0018).

A continuous companion to exact match with no distractor: the task carries
the gold answer's token ids (the builder tokenized it once, with the base's
tokenizer) and this scores them after the question under `Route(route)`.
Not a calibrated probability of being right — a memory can put its mass on a
paraphrase — but it moves with knowledge and costs one prefill."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group, Message, Role, Trajectory
from rlstack.policy.adapters.dream_bank import Route
from rlstack.training.post.base import PostProcessor, postprocessor
from rlstack.training.post.dream_effect import kind_of


async def gold_score(traj: Trajectory, client: PoolClient) -> float:
    token_ids = tuple(int(t) for t in traj.task.meta["answer_token_ids"])
    if not token_ids:
        return 0.0
    scores = await client.score([Message(Role.USER, traj.task.prompt)], token_ids,
                                directives=(Route(str(traj.task.meta["route"])),))
    return sum(scores) / len(scores)


@postprocessor("gold_logprob")
class GoldLogprob(PostProcessor):
    produces = ("gold_lp",)
    pools = ("main",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        main = client.pool("main")
        indices = [i for i, t in enumerate(group.trajectories) if kind_of(t) == "qa"]
        scored = await asyncio.gather(*(gold_score(group.trajectories[i], main) for i in indices))
        column = [0.0] * len(group)
        for i, value in zip(indices, scored):
            column[i] = value
        return {"gold_lp": column}
