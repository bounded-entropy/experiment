"""The group's accuracy as a column: the mean reward, one copy per row.

A group is the advantage's baseline scope, and it is also the natural scope of
"solved": when every completion in a group is right, the z-scored advantage is
identically zero and the group has nothing left to teach the policy. This
column is how a loss can SEE that fact — `grpo_latent_kl_gated` reads it to
decide when the prior's pull is earned. No pool traffic and no randomness:
an average of a recorded column costs nothing (I9).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor


@postprocessor("group_accuracy")
class GroupAccuracy(PostProcessor):
    consumes = ("reward",)
    produces = ("accuracy",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        rewards = data["reward"]
        accuracy = math.fsum(rewards) / len(rewards)
        return {"accuracy": [accuracy] * len(rewards)}
