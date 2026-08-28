"""Reward 1.0 for every trajectory: the degenerate reward column, for tests and
for pipelines whose signal comes from somewhere else."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor


@postprocessor("constant")
class Constant(PostProcessor):
    produces = ("reward",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        return {"reward": [1.0] * len(group)}
