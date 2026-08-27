"""Built-in rewards (inference world).

Rewards run per-episode BEFORE the seal and may sample (LLM judges are rewards,
never losses). Each declares the component names it writes; advantages declare
what they consume, and Phase 0 joins the two (I4). A reward receives the
finished trajectory and a SampleClient, and returns {component: value}.
"""

from __future__ import annotations

import re
from typing import Any

from rlstack.data.trajectory import Trajectory
from rlstack.registry import reward

_NUMBER = re.compile(r"-?\d+")


@reward("verifier", components=("reward",))
async def verifier(traj: Trajectory, llm: Any) -> dict[str, float]:
    """Pure check, no sampling: the last number in the final turn's text
    against task.meta["answer"]."""
    numbers = _NUMBER.findall(traj.turns[-1].message.content)
    correct = bool(numbers) and numbers[-1] == str(traj.task.meta["answer"])
    return {"reward": float(correct)}


@reward("constant", components=("reward",))
async def constant(traj: Any, llm: Any) -> dict[str, float]:
    """Writes a single flat "reward" component (useful as a baseline/stub)."""
    return {"reward": 1.0}
