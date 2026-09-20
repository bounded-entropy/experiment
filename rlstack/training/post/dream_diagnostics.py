"""dream_diagnostics: what a dream looks like to the base (ADR 0018).

For every DREAM row of the group — a dream sampled with the arrival in
context, replayed here for a memory to train on — the frozen base's mean NLL
of the dream given that context (`dream_base_nll`: a dreamer drifting off
the base's manifold shows here first), and the dream's length in tokens
(`dream_tokens`). Scored under `base`: the bare model, no set at all."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.policy.adapters.dream_bank import BASE
from rlstack.training.post.base import PostProcessor, postprocessor
from rlstack.training.post.dream_effect import kind_of, mean_nll, route_scores


@postprocessor("dream_diagnostics")
class DreamDiagnostics(PostProcessor):
    produces = ("dream_base_nll", "dream_tokens")
    pools = ("main",)

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        main = client.pool("main")
        indices = [i for i, t in enumerate(group.trajectories) if kind_of(t) == "dream"]
        scored = await asyncio.gather(
            *(route_scores(group.trajectories[i], main, BASE, hinted=True) for i in indices))
        nll, tokens = [0.0] * len(group), [0.0] * len(group)
        for i, scores in zip(indices, scored):
            nll[i] = mean_nll(scores)
            tokens[i] = float(sum(len(t.token_ids) for t in group.trajectories[i].turns))
        return {"dream_base_nll": nll, "dream_tokens": tokens}
