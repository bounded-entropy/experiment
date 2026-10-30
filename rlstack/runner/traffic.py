"""The client side of pool traffic: how an episode talks to pools.

EnginePoolClient drives one pool for one episode — sample assembles a token
stream into Turns, score is one prefill pass — and `pool(name)` hands a
sibling for any other declared pool under one shared seed sequence. Every
call is routed by its AFFINITY KEY among the pool's replicas (ADR 0022): the
episode's id unless the environment names its own.
run_episode is the seal point; collect_wave assembles one wave of episodes
deterministically given (master, update), assigning group keys.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import replace
from collections.abc import Awaitable, Callable, Mapping, Sequence

from rlstack.data.stores.base import Store
from rlstack.data.tasks import load_tasks
from rlstack.data.trajectory import Group, Message, Role, Task, Trajectory, Turn, Wave
from rlstack.policy.adapters.base import Directive
from rlstack.policy.compile import Bundle
from rlstack.runner.interfaces import Engine, FinishEvent
from rlstack.runner.pool import Pool
from rlstack.runner.remote import encode_directives
from rlstack.runner.seeds import derive
from rlstack.spec.specs import SamplingSpec
from rlstack.inference.rollout import Rollout

# One named pool: its replicas (a Pool — or, in a table a test builds by
# hand, one bare Engine) and the bundle its requests pin.
Routes = Mapping[str, tuple[Pool | Engine, Bundle]]


class EnginePoolClient:
    """The concrete PoolClient: one instance per episode per pool, all
    siblings sharing ONE seed sequence, so multi-pool traffic is deterministic
    regardless of which pools an episode touches.

    `calls`, when handed a list, receives one timing row per sample call of
    the episode (all siblings append to the same list): observability only,
    journaled by the generator and never part of the sealed trajectory.

    `episode` is the episode's id — the AFFINITY KEY every call carries
    unless the caller names one (ADR 0022): a seen key returns to the
    replica holding its prefix. Unnamed, it is the episode's seed, which is
    a pure function of the episode's identity all the same."""

    def __init__(self, routes: Routes, sampling: SamplingSpec, episode_seed: int,
                 pool_name: str = "main", _counter: list[int] | None = None,
                 calls: list[dict] | None = None, episode: str | None = None) -> None:
        self._routes = routes
        self._sampling = sampling
        self._episode_seed = episode_seed
        self._pool_name = pool_name
        self._counter = _counter if _counter is not None else [0]
        self._calls = calls
        self._episode = episode if episode is not None else f"seed:{episode_seed}"

    def route(self, name: str) -> tuple[Pool, Bundle]:
        """Resolve only actual traffic; a pool-less processor needs no main.
        A bare engine in a hand-built table is a one-replica pool."""
        if name not in self._routes:
            raise KeyError(
                f"unknown engine pool {name!r}; pools: {sorted(self._routes)}")
        handed, bundle = self._routes[name]
        return Pool.of(name, handed), bundle

    def pool(self, name: str) -> "EnginePoolClient":
        """A sibling client for another pool, sharing this episode's seeds
        and its affinity key."""
        self.route(name)
        return EnginePoolClient(self._routes, self._sampling, self._episode_seed,
                                  name, self._counter, self._calls, self._episode)

    def affinity_of(self, affinity: str | None) -> str:
        """The key a call routes by: the caller's own, else the episode's."""
        return self._episode if affinity is None else affinity

    async def score(self, messages: Sequence[Message],
                    token_ids: Sequence[int], *,
                    directives: Sequence[Directive] = (),
                    affinity: str | None = None) -> tuple[float, ...]:
        """Score given tokens under this pool's pinned bundle. Deterministic:
        consumes NO seed from the episode's sequence (scoring draws nothing),
        so adding a scoring processor never shifts sampling seeds."""
        pool, bundle = self.route(self._pool_name)
        scores, _ = await pool.serve(
            self.affinity_of(affinity),
            lambda engine: engine.score_tokens(
                messages, tuple(token_ids), bundle.bundle_id, tuple(directives)))
        return scores

    async def sample(self, messages: Sequence[Message],
                     stop: tuple[str, ...] = (), *,
                     directives: Sequence[Directive] = (),
                     max_tokens: int | None = None,
                     sampling: SamplingSpec | None = None,
                     affinity: str | None = None) -> Turn:
        pool, bundle = self.route(self._pool_name)
        sampling = ceiled(sampling if sampling is not None else self._sampling, max_tokens)
        seed = derive(self._episode_seed, "call", self._counter[0])
        self._counter[0] += 1

        started = time.time()
        (token_ids, logprobs, text_parts, columns, finish), replica = await pool.serve(
            self.affinity_of(affinity),
            lambda engine: streamed(engine, messages, sampling, stop,
                                    bundle.bundle_id, seed, directives))
        pool.counted(replica, len(token_ids))
        if self._calls is not None:
            self._calls.append(call_row(self._pool_name, directives, started,
                                        len(token_ids), finish.finish, replica,
                                        sampling=sampling, stop=stop, seed=seed))

        return Turn(
            message=Message(Role.ASSISTANT, "".join(text_parts)),
            token_ids=tuple(token_ids),
            behavior_logprobs=tuple(logprobs),
            finish=finish.finish,
            stop_hit=finish.stop_hit,
            bundle_id=bundle.bundle_id,
            policy_version=dict(bundle.policy_version),
            seed=seed,
            token_extras={k: tuple(v) for k, v in columns.items()},
            turn_extras={**dict(finish.turn_extras),
                         "sampling": sampling_row(sampling, stop)},
        )


async def streamed(engine: Engine, messages: Sequence[Message],
                   sampling: SamplingSpec, stop: tuple[str, ...], bundle_id: str,
                   seed: int, directives: Sequence[Directive],
                   ) -> tuple[list[int], list[float], list[str], dict[str, list], FinishEvent]:
    """ONE WHOLE TOKEN STREAM from one engine, assembled: the unit a pool
    retries on a survivor when a replica dies mid-stream, so a retry starts
    over and never duplicates a token."""
    token_ids: list[int] = []
    logprobs: list[float] = []
    text_parts: list[str] = []
    columns: dict[str, list] = {}
    finish: FinishEvent | None = None

    stream = engine.sample_tokens(
        messages, sampling, stop, bundle_id, seed, tuple(directives))
    async for event in stream:
        if isinstance(event, FinishEvent):
            finish = event
            break
        token_ids.append(event.token_id)
        logprobs.append(event.logprob)
        text_parts.append(event.text_delta)
        for name, value in event.extras.items():
            # a column appearing mid-stream backfills earlier positions
            columns.setdefault(name, [None] * (len(token_ids) - 1)).append(value)
        for name in columns:
            if len(columns[name]) < len(token_ids):
                columns[name].append(None)
    if finish is None:
        raise RuntimeError("engine stream ended without a FinishEvent")
    return token_ids, logprobs, text_parts, columns, finish


def ceiled(sampling: SamplingSpec, max_tokens: int | None) -> SamplingSpec:
    """The run's sampling with a per-call ceiling: the smaller budget wins,
    and no ceiling is the run's own."""
    if max_tokens is None:
        return sampling
    if max_tokens < 1:
        raise ValueError(f"a call's max_tokens ceiling must be positive, got {max_tokens}")
    return replace(sampling, max_tokens=min(sampling.max_tokens, int(max_tokens)))


def call_row(pool: str, directives: Sequence[Directive], started: float,
             tokens: int, finish: str, replica: int = 0, *,
             sampling: SamplingSpec, stop: tuple[str, ...], seed: int) -> dict:
    """One sample call's timing: where it went (pool and replica), when it
    started, how long it took, how many tokens it made and why it stopped."""
    return {"pool": pool, "replica": replica,
            "directives": encode_directives(directives), "t": started,
            "seconds": time.time() - started, "tokens": tokens, "finish": finish,
            "sampling": sampling_row(sampling, stop), "seed": seed}


def sampling_row(sampling: SamplingSpec, stop: tuple[str, ...]) -> dict:
    """The effective request configuration, persisted beside its sampled tokens."""
    row = {"temperature": sampling.temperature, "top_p": sampling.top_p,
           "max_tokens": sampling.max_tokens, "stop": list(stop)}
    if sampling.min_tokens_after is not None:
        row["min_tokens_after"] = {
            "sequence": list(sampling.min_tokens_after.sequence),
            "min_tokens": sampling.min_tokens_after.min_tokens}
    return row


def load_task_sets(store: Store, uris: Sequence[str]) -> dict[str, Task]:
    """Every declared task set, keyed by task id — what a plan's leaves name.

    Ids are global across a run's sets: a leaf carries an id alone, so two sets
    sharing one id would make a leaf ambiguous, and the collision is refused
    here rather than silently resolved by set order.
    """
    tasks: dict[str, Task] = {}
    for uri in uris:
        for task in load_tasks(store, uri):
            if task.id in tasks:
                raise ValueError(
                    f"task id {task.id!r} appears in two declared task sets; "
                    f"a plan's leaf names an id alone, so ids must be unique")
            tasks[task.id] = task
    return tasks


async def run_episode(function: Callable[..., Awaitable[Rollout]],
                      source: Task | Trajectory,
                      client: EnginePoolClient) -> Trajectory:
    """Invoke the experiment's plain function and seal its Rollout (I1)."""
    rollout = await function(client, source)
    if not isinstance(rollout, Rollout):
        raise TypeError("an interaction function must return a Rollout for the runner to seal")
    return rollout.seal()
