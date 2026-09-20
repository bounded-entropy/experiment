"""dream_advantage: the dreamer's group baseline (ADR 0018), and its global
scale and cap (research-log post 0008).

The group holds the dreamer's samples among other rows — in a stream run an
arrival's texts, dreams and answers; in a dreamer run one memory's forks —
and only the DREAMER rows are the dreamer's samples: a row whose role's last
part is `dreamer`, alone or stacked on a library memory. Each dream's
advantage is its reward minus the mean reward of the group's dreams:
REINFORCE with the group mean as the baseline, no scaling by the group's
spread (Samarth, 2026-09-17: the plain thing). A dreamer row's task may carry
`adv_scale`, ONE global number the advantage is divided by, and `adv_cap`,
the magnitude it is clipped to; a task carrying neither gets the plain
difference. Every other row's advantage is exactly zero, which is what
`dream_stream` multiplies their (masked) tokens by anyway. Inline: no pool,
arithmetic over a column an earlier processor produced."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor
from rlstack.training.post.dream_effect import role_of, trains_the_dreamer


def scaled_and_capped(value: float, meta: Mapping) -> float:
    """value / adv_scale, clipped to ±adv_cap; each only when the task says so."""
    value = value / float(meta.get("adv_scale", 1.0))
    cap = meta.get("adv_cap")
    return value if cap is None else max(-float(cap), min(float(cap), value))


@postprocessor("dream_advantage")
class DreamAdvantage(PostProcessor):
    consumes = ("dream_reward",)
    produces = ("advantage",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        dreamers = [i for i, t in enumerate(group.trajectories) if trains_the_dreamer(role_of(t))]
        advantage = [0.0] * len(group)
        if dreamers:
            rewards = [data["dream_reward"][i] for i in dreamers]
            baseline = sum(rewards) / len(rewards)
            for i, value in zip(dreamers, rewards):
                advantage[i] = scaled_and_capped(value - baseline, group.trajectories[i].task.meta)
        return {"advantage": advantage}
