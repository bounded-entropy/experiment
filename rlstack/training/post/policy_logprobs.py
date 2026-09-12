"""Score sealed targets under the policy, without adding conditioning.

The Scorer pins the wave's recorded bundle. These per-token columns preserve
individual answer likelihoods when many candidates share one evaluation wave.
The same walk used for teacher scoring handles injected and generated turns.
"""

import asyncio
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor
from rlstack.training.post.teacher_logprobs import teacher_scores


@postprocessor("policy_logprobs")
class PolicyLogprobs(PostProcessor):
    produces = ("policy_logprobs",)
    token_level = ("policy_logprobs",)
    pools = ("main",)

    async def process(self, group: Group, data, client: PoolClient
                      ) -> Mapping[str, Sequence]:
        main = client.pool("main")
        return {"policy_logprobs": list(await asyncio.gather(
            *(teacher_scores(traj, main) for traj in group.trajectories)))}
