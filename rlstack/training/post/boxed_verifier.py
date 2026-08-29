"""The reward for a reasoning model: read the BOXED answer, not the last number.

`verifier` takes the last integer in the completion, which is right for a terse
few-shot scaffold and wrong for a model that thinks out loud: the last integer
of a think block is whatever the model happened to arithmetic on last, so a
correct answer scores 0 and a wrong one sometimes scores 1. Competition-math
data (DAPO-Math-17k) states its answer in \\boxed{}, so that is what is read
here, and the LAST box wins because a model that reconsiders states its final
answer last.

The fallback to a trailing integer is deliberate and narrow: a completion with
no box at all is graded the old way rather than auto-failed, so a prompt that
asks for a different form still scores instead of silently zeroing the whole
run. A truncated completion usually has neither, and scores 0 — which is
correct, not a bug: it never produced an answer.

No pool traffic, no randomness — a checked reward costs nothing (I9).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor

_BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_INTEGER = re.compile(r"-?\d+")


def stated_answer(text: str) -> str | None:
    """What the completion claims, normalized — or None if it claims nothing.

    Integers are compared as integers (via str(int)), so "042", "+42" and
    "42" are one answer and formatting never decides a reward.
    """
    boxes = _BOXED.findall(text)
    claim = boxes[-1].strip() if boxes else None
    if claim is None:
        trailing = _INTEGER.findall(text)
        claim = trailing[-1] if trailing else None
    if claim is None:
        return None
    stripped = claim.replace(",", "").replace("$", "").strip()
    try:
        return str(int(stripped))
    except ValueError:
        return stripped


@postprocessor("boxed_verifier")
class BoxedVerifier(PostProcessor):
    produces = ("reward",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        rewards = []
        for traj in group.trajectories:
            claimed = stated_answer(traj.turns[-1].message.content)
            expected = str(traj.task.meta["answer"]).strip()
            rewards.append(float(claimed is not None and claimed == expected))
        return {"reward": rewards}
