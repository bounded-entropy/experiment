"""The reward for a stated answer: read the marker the PROMPT asked for.

`verifier` takes the last integer in the completion. That is right for a terse
few-shot scaffold and wrong wherever a model writes prose before answering: the
last integer of a worked solution is whatever it last did arithmetic on, so a
correct answer scores 0 and a wrong one sometimes scores 1.

This reads the answer a task's own prompt asked the model to state, in the two
forms competition-math prompts actually use — DAPO-Math-17k's `Answer: N` last
line (#60 pins that instruction verbatim in the task set) and the `\\boxed{N}`
convention. The LAST occurrence wins in both, because a model that reconsiders
states its final answer last.

NO BARE-NUMBER FALLBACK, deliberately. A completion that never states an answer
scores 0 — it did not answer — and the alternative is worse than strict: a
truncated attempt ends on some intermediate number, which would match the
target by coincidence often enough to pay for length instead of correctness.
The MATH campaign's measured 21-of-32 truncation rate (#58) is why that risk is
worth refusing rather than hedging.

No pool traffic and no randomness: a checked reward costs nothing (I9).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor

# the two markers, each capturing what follows it on that line / in that box
_MARKERS = (re.compile(r"[Aa]nswer:\s*\$?([^\n$]*)"),
            re.compile(r"\\boxed\s*\{([^{}]*)\}"))


def stated_answer(text: str) -> str | None:
    """What the completion CLAIMS, normalized — or None if it claims nothing.

    Integers normalize through int(), so "042", "+42" and "42" are one answer
    and formatting never decides a reward; anything else is compared as the
    stripped string it is.
    """
    for marker in _MARKERS:
        found = marker.findall(text)
        if found:
            claim = found[-1].replace(",", "").replace("$", "").strip()
            try:
                return str(int(claim))
            except ValueError:
                return claim
    return None


@postprocessor("final_answer")
class FinalAnswer(PostProcessor):
    produces = ("reward",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        rewards = []
        for traj in group.trajectories:
            claimed = stated_answer(traj.turns[-1].message.content)
            expected = str(traj.task.meta["answer"]).strip()
            rewards.append(float(claimed is not None and claimed == expected))
        return {"reward": rewards}
