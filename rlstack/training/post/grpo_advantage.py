"""The GRPO estimator as a postprocessor: normalize each trajectory's reward
within its group. The group IS the baseline scope — no parallel id arrays."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rlstack.client import SampleClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor


def zscore(values: Sequence[float]) -> list[float]:
    """(x − mean) / population std; an all-equal list normalizes to zeros."""
    n = len(values)
    mean = math.fsum(values) / n
    std = math.sqrt(math.fsum((x - mean) ** 2 for x in values) / n)
    return [0.0 if std == 0.0 else (x - mean) / std for x in values]


@postprocessor("grpo_advantage")
class GrpoAdvantage(PostProcessor):
    consumes = ("reward",)
    produces = ("advantage",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      llm: SampleClient) -> Mapping[str, Sequence[float]]:
        return {"advantage": zscore(data["reward"])}
