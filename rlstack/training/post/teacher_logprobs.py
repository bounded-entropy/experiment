"""The teacher channel: a FROZEN OTHER MODEL scores the student's own tokens.

On-policy distillation's one non-student ingredient (#47). For each sealed
trajectory this processor replays the message stream and asks the "teacher"
pool for score traffic on the tokens the student actually chose — one prefill
pass per generated turn — and the per-token result is the token_level column
opd requires. The teacher is a POOL, so switching teachers means pointing that
pool at other metal.

Two preconditions, both real:
  - the teacher shares the student's TOKENIZER. Token ids cross unchanged
    (they are the student's draws), so a teacher with another vocabulary would
    score different text than the student wrote. Same-family pairs
    (Qwen3-8B ← Qwen3-32B) satisfy it; the check belongs at the deploy, which
    is what hands the two pools their metal.
  - the teacher is FROZEN. A non-policy pool serves its bare base — its bundle
    is payload-free, so no delta of this run is ever applied — and scoring is
    seedless, so adding this processor never shifts the run's sampling seeds.

The walk mirrors hinted_logprobs minus the hint, so the column's floats land
token-for-token on the positions the loss mask selects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group, Message, Trajectory
from rlstack.training.post.base import PostProcessor, postprocessor


async def teacher_scores(traj: Trajectory, llm: PoolClient) -> list[float]:
    """Walk the sealed message stream in flatten order: each generated turn's
    token_ids scored against everything before it — the SAME conditioning the
    student had, which is what makes the difference a pure model difference."""
    turn_of = {id(t.message): t for t in traj.turns}
    context: list[Message] = []
    scores: list[float] = []
    for message in traj.messages:
        turn = turn_of.get(id(message))
        if turn is not None:
            scores.extend(await llm.score(context, turn.token_ids))
        context.append(message)
    return scores


@postprocessor("teacher_logprobs")
class TeacherLogprobs(PostProcessor):
    produces = ("teacher_logprobs",)
    token_level = ("teacher_logprobs",)
    pools = ("teacher",)              # another model's metal, declared at submit

    async def process(self, group: Group, data, llm: PoolClient
                      ) -> Mapping[str, Sequence]:
        teacher = llm.pool("teacher")
        return {"teacher_logprobs": [await teacher_scores(traj, teacher)
                                     for traj in group.trajectories]}
