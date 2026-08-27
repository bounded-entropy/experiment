"""LLM-as-a-judge, reference-free: the judge pool solves each trajectory's
task independently (greedy, its own sampling budget), and the reward is
agreement — last number of the judge's answer == last number of the policy's.

The exemplar of the pool treaty: `pools` declares the traffic (Phase 0 checks
a "judge" engine pool is declared), `llm.pool("judge")` carries it, and
`sampling` is the judge's own — not the policy's — hashing into identity
through this class's source like every other declaration."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rlstack.client import SampleClient
from rlstack.data.trajectory import Group, Message, Role
from rlstack.spec.specs import SamplingSpec
from rlstack.training.post.base import PostProcessor, postprocessor

_NUMBER = re.compile(r"-?\d+")


def _last_number(text: str) -> str | None:
    numbers = _NUMBER.findall(text)
    return numbers[-1] if numbers else None


@postprocessor("llm_judge")
class LlmJudge(PostProcessor):
    produces = ("reward",)
    pools = ("judge",)
    sampling = SamplingSpec(temperature=0.0, top_p=1.0, max_tokens=16)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      llm: SampleClient) -> Mapping[str, Sequence[float]]:
        judge = llm.pool("judge")
        rewards = []
        for traj in group.trajectories:
            verdict = await judge.sample((Message(Role.USER, traj.task.prompt),))
            judged = _last_number(verdict.message.content)
            answered = _last_number(traj.turns[-1].message.content)
            rewards.append(float(judged is not None and judged == answered))
        return {"reward": rewards}
