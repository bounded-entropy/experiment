"""The group's preference verdict as a column: +1 its best, -1 its worst.

A Group is "the compared pair" in a preference loss (trajectory.py says so),
and this processor is where the comparison happens: the group's highest-reward
trajectory is marked chosen (+1), its lowest rejected (-1), everything else 0.
Ties break by sealed order — first highest, last lowest — so the verdict is
deterministic under resume. A group whose rewards are ALL EQUAL marks nothing:
there is no preference in it, and inventing one would train on noise.

The verdict is a COLUMN, not loss-side logic (I9): sdpo reads it per token and
never needs to know what a group was. No pool traffic and no randomness.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor


@postprocessor("dpo_pair")
class DpoPair(PostProcessor):
    consumes = ("reward",)
    produces = ("pair",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        rewards = list(data["reward"])
        pair = [0.0] * len(rewards)
        if max(rewards) > min(rewards):
            chosen = rewards.index(max(rewards))            # first highest
            rejected = (len(rewards) - 1
                        - rewards[::-1].index(min(rewards)))  # last lowest
            pair[chosen] = 1.0
            pair[rejected] = -1.0
        return {"pair": pair}
