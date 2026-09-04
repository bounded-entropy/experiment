"""Reverse KL against the teacher column, as a MEASURED number (ADR 0005).

Per trajectory, the mean over generated tokens of (recorded behavior logprob
− teacher logprob): the one-sample reverse-KL estimate, which is exactly what
`opd`'s ledger `loss` already reports for a run that trains on it. Here it is
reported for a run that does NOT — a measurement's pipeline
(`conditioned_teacher_logprobs` → `reverse_kl`) turns "did the distillation
work" into one float per measured version, beside the run rather than in it
(#70).

POOL-LESS, so the split rule (`split_pipeline`) makes it TRAINER-INLINE: it
sends no traffic, it only reduces a column the pooled half already wrote.
That is the legal direction — a pooled processor may not consume an inline
one's column, and this is its mirror.

The behavior logprobs are the RECORD's: for a measurement they are the
student's own draws under the version being measured, so the number is the
distance from what the student sampled to what the conditioned teacher would
have said about those very tokens. A trajectory that generated nothing scores
0.0 rather than dividing by zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group, Trajectory
from rlstack.training.post.base import PostProcessor, postprocessor


def behavior_logprobs(traj: Trajectory) -> list[float]:
    """The sealed per-token logprobs, in the order a token_level column is:
    every generated turn's, concatenated in flatten order."""
    return [lp for turn in traj.turns for lp in turn.behavior_logprobs]


def reverse_kl_of(traj: Trajectory, teacher: Sequence[float]) -> float:
    """One trajectory's estimate: mean over generated tokens of (behavior −
    teacher). The two vectors are the same walk, so a length mismatch is a
    broken alignment and says so rather than truncating."""
    behavior = behavior_logprobs(traj)
    if len(behavior) != len(teacher):
        raise ValueError(
            f"reverse_kl: {len(teacher)} teacher logprobs for a trajectory "
            f"with {len(behavior)} generated tokens — the teacher column and "
            f"the record must walk the same turns")
    if not behavior:
        return 0.0
    return math.fsum(b - float(t) for b, t in zip(behavior, teacher)) / len(behavior)


@postprocessor("reverse_kl")
class ReverseKL(PostProcessor):
    produces = ("reverse_kl",)
    consumes = ("teacher_logprobs",)
    pools = ()                        # pure reduction: no traffic, so inline

    async def process(self, group: Group, data, client: PoolClient
                      ) -> Mapping[str, Sequence[float]]:
        teacher = data["teacher_logprobs"]
        return {"reverse_kl": [reverse_kl_of(traj, vector)
                               for traj, vector
                               in zip(group.trajectories, teacher)]}
