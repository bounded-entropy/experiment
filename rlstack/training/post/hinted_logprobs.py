"""Hinted self-scoring: the policy's own logprobs under privileged conditioning.

For each sealed trajectory, prepend a hint the sampler never saw, re-score the
trajectory's OWN generated tokens through the policy pool (score traffic, one
prefill pass per turn), and emit the per-token result as a token_level column;
opsd requires it and distills the policy toward what it believes when it knows
the answer — one model, no second pool. The hint is task metadata:
meta["hint"] verbatim when present, else "The answer is {meta['answer']}. ",
the demo teacher for verifier-style tasks. Scoring is seedless, so adding this
processor never shifts the run's sampling seeds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group, Message, Role, Trajectory
from rlstack.training.post.base import PostProcessor, postprocessor


def hint_for(traj: Trajectory) -> Message:
    meta = traj.task.meta
    text = meta.get("hint") or f"The answer is {meta['answer']}. "
    return Message(Role.USER, text)


async def hinted_scores(traj: Trajectory, client: PoolClient) -> list[float]:
    """Walk the sealed message stream in flatten order: each generated turn's
    token_ids scored against hint + everything before it."""
    turn_of = {id(t.message): t for t in traj.turns}
    context: list[Message] = [hint_for(traj)]
    scores: list[float] = []
    for message in traj.messages:
        turn = turn_of.get(id(message))
        if turn is not None:
            scores.extend(await client.score(context, turn.token_ids))
        context.append(message)
    return scores


@postprocessor("hinted_logprobs")
class HintedLogprobs(PostProcessor):
    produces = ("hinted_logprobs",)
    token_level = ("hinted_logprobs",)
    pools = ("main",)                 # scores the POLICY pool: self-distillation

    async def process(self, group: Group, data, client: PoolClient
                      ) -> Mapping[str, Sequence]:
        main = client.pool("main")
        return {"hinted_logprobs": [await hinted_scores(traj, main)
                                    for traj in group.trajectories]}
