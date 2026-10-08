<!-- ADR formatting: keep each prose paragraph and list item on one source line. Do not hard-wrap prose at a fixed column width. Preserve intentional line breaks in fenced code, tables, and diagrams. Apply this convention to all future ADRs. -->

# ADR 0022 — A run spans metals: pools own their replicas, the learner composes dp × fsdp, carves are measured

| | |
|---|---|
| **Date** | 2026-10-08 |
| **Status** | Implemented (2026-10-08; CONTEXT #97) — accepted with Q1–Q5 answered in chat 2026-10-08 |
| **Author** | Claude (agent session, post 0011 / B300 node) |
| **Touches** | `spec/specs.py` (pool replicas, learner dp), `inference/` sample call (affinity key), `runner/traffic.py` (the pool's router), `runner/pool.py` (new: replica health and revival), `runner/roles/generator.py` (deliver/hand to every replica), `runner/arbiter.py` (admit a pool), `runner/desk.py` + `runner/host.py` (carve N hosts for one pool), `runner/learners/` (dp × fsdp), `runner/residents.py` (measured booking) |
| **Invariants** | I3/I5: replicas, routing and learner width are resources, not science; they stay out of the run id, and resume-equivalence is unchanged |
| **CONTEXT** | Extends ADR 0001 (partitions in GB), 0006 (a run is daemons with resources), 0011 (training commits, inference loads), 0017 (shared desk/metal runtimes); #45 (TP/FSDP on metal) |

## Original prompt

> the goal is to make these decode engines and learners work multi-gpu. then, the actual experiment-processes for the thing that submits to the engine (i forget what this is called) can have a multi-gpu setting where it just automatically load-balances sending requests (or is this sth that vllm natively supports, this sort of load balancing having multiple replicas of a gpu). for the training, i think it's pretty clear that if one thing is the same FSDP process or DDP training process or whatever it should be relatively fine, right? also, we'll have a shit ton of extra space on these GPUs for other stuff, so we need to ensure that carving works

> yeah ok. so there should be a setting where a pool maps to several engines. does vLLM automatically do routing between this sort of stuff? ok let's create an ADR that addresses all the points above. keep the ADR well formatted and brief so I can review it properly

> im happy doing our own routing, but how do we ensure that kv caches get preserved across requests (we route to the same GPUs)? we probably need to add some sort of metadata to the request in the environment (like some sort of uuid) that helps us route to the correct GPU (this is also what OAI and whatever does for their apis)
>
> we should route per request, but the request should contain some metadata like i said before.
>
> the pool should be the unit responsible for replicas dying, and should revive them itself and potentially serve to other gpus in the meanwhile
>
> so, dp and fsdp are complementary right, we should be able to compose both of these together
>
> i mean the fork training etting its own replicated pool is a question not for this ADR, since the ADR applies to general codebase stuff. also, what do you define as a sleeping engine here?

## Context / problem

On one H100, a post 0011 GRPO update took about 40 minutes: about 33 minutes of rollouts and about 8 minutes of forks plus the learner step, time-sliced on one GPU. On the 8×B300 node one run should use several GPUs at once. Today it can't:

1. **One engine per pool.** `Topology` allows each pool name once (`spec/specs.py:224`). A route maps a name to exactly one `(Engine, Bundle)` (`runner/traffic.py:30`), so all of a run's generation goes to one engine.
2. **The multi-GPU learner adds memory, not throughput.** `FsdpTorchLearner` shards the frozen base, but every rank recomputes the same microbatch (`runner/learners/fsdp_torch.py:1-8`). Its NCCL rendezvous is `tcp://127.0.0.1` (`runner/learners/ranks.py:127-131`).
3. **Carving is declared, not measured, and not steerable.** Partitions are booked in GB (ADR 0001), and nothing checks a resident against its booking (`runner/arbiter.py:108-120`). The carve rung tries metals in name order. On the B300 main desk a dreamer would land on the bank's GPU 4 whenever GPU 4 had room.

**vLLM's own routing.** vLLM's data-parallel mode (`data_parallel_size`) balances N engine cores behind one front end by queue length. It has no session affinity, it lives in one process group on one metal, and sleep/wake and adapter loads hit all cores at once. It is rejected (Q1).

## Decision

**A pool owns its replicas.** A pool may declare `replicas: N`; the desk carves N hosts for it. The pool, not the run, is responsible for them:
- **Routing:** every request is routed by its affinity key (below).
- **Health:** the pool notices a dead replica, re-carves it through the desk, and serves the dead replica's keys from the survivors until it returns.
- **Delivery:** policy and memory adapters reach every live replica before the ledger commit, and a revived replica receives the committed bundle before it is admitted.

**Requests carry an affinity key.** This is the same idea as OpenAI's `prompt_cache_key` and session-affinity headers.
- The environment's sample call carries `affinity: str`. It defaults to the episode id, and an environment may set its own, e.g. per conversation branch.
- The router routes per request. A key it has seen goes to the replica that holds it, so that replica's prefix cache is reused. A new key goes to the least-loaded replica.
- The key moves only when its replica dies; the cost is one cache miss.

**The learner composes dp × fsdp.** Width = `dp` × `fsdp` ranks on one metal:
- the frozen base is sharded across each fsdp group;
- each dp group takes its own slice of the microbatches;
- the LoRA gradients are all-reduced across dp groups in a fixed rank order, then one optimizer step.

`dp=1` is today's FSDP learner, and `fsdp=1` is plain data parallelism.

**Carves are measured.** Each host's measured peak memory is journaled next to its declared booking, and an overrun is a warning in `observe/`. Idle hosts are still released by `host_idle_s`, as today.

**What "sleeping engine" meant.** It was a vLLM engine put to sleep on an alternating host (engine and learner taking turns on one partition): weights offloaded, KV cache freed, process alive. Refunding its memory was a mistake. On an alternating host the learner uses that space while the engine sleeps, and a wholly idle host is already released by `host_idle_s`. The question is dropped.

### Touched / untouched

- **Touched:**
  - `spec/specs.py`: `PoolMember.replicas` (default 1), `LearnerMember.dp` (default 1, composing with the existing `fsdp`), and feasibility validation at submit.
  - The sample call in `inference/`: an optional `affinity` field.
  - `runner/traffic.py`: `Routes` maps a pool name to a `Pool`, and `EnginePoolClient` forwards requests with their key.
  - `runner/pool.py` (new): the `Pool` record holds the replicas, the key table, health checks and revival.
  - `runner/roles/generator.py`: delivery and library handling go through the pool.
  - `runner/arbiter.py`: admission of a pool.
  - `runner/desk.py`, `runner/host.py`: carve N hosts for one pool. No allow-list or spread rule (Q5).
  - `runner/learners/`: a dp axis on the torch learner.
  - `runner/residents.py`: measured booking.
- **Untouched:**
  - Environments that don't set a key. The default key is the episode id, so existing environments work unchanged.
  - Postprocessors and losses. A scored wave doesn't depend on which replica produced it.
  - The store layout and run identity.
  - Venues. They already list several metals per desk.

### Promises / non-promises

- **Promises:**
  - With the deterministic fake engines and learners, a run's sealed waves and committed adapters are byte-identical for `replicas` ∈ {1, N}, any routing, and `dp` ∈ {1, K}, at a fixed seed, in float64 on CPU.
  - Resume reads sealed waves as today, so resume-equivalence is unchanged.
  - A key is never served by a replica holding an older bundle than the committed one.
  - A replica death doesn't fail the wave: in-flight requests are retried elsewhere, and the replica is revived.
- **Non-promises:**
  - Real vLLM generation is not batch-invariant, so the same request on two replicas may sample different tokens. That was already true of one engine under different batching, and sealed bytes are what training and resume use.
  - Throughput scaling is unmeasured until the B300 runs.
  - No cross-metal NCCL: all of a learner's ranks live on one metal.
  - No MIG/MPS isolation; neighbours on a card share compute.

### Interfaces

- **Spec:** `PoolMember(name="main", replicas=3, vram_gb=…)` and `LearnerMember(dp=2, fsdp=2, …)`.
- **Environment:** `client.sample(prompt, affinity=key)`.
- **Desk:** a pool with `replicas: N` is N host placements under one pool name. Each joins or carves like a single host does today.
- **Observe:** per replica: in-flight requests, tokens/s, cache-hit rate and revivals. The key table's size and moves. Learner step time per rank. Declared vs measured GB per carve.

### Sketches

```python
@dataclass(frozen=True)
class PoolMember:
    name: str
    replicas: int = 1                  # resources, not science: out of the run id

class Pool:
    """The replicas of one pool name: routes by affinity, revives the dead, delivers bundles."""
    def route(self, affinity: str) -> Replica: ...        # sticky; new keys to least-loaded
    async def deliver(self, bundle: Bundle) -> None: ...  # every live replica, before commit
    async def revive(self, replica: Replica) -> None: ... # re-carve via the desk, deliver, readmit

class EnginePoolClient:
    async def sample(self, prompt, *, affinity: str | None = None): ...  # default: the episode id

@dataclass(frozen=True)
class LearnerMember:
    fsdp: int = 1                      # shards the frozen base within a group
    dp: int = 1                        # splits microbatches across groups; width = dp * fsdp
```

## Questions

**Q1. Our own routing, or vLLM's data-parallel mode?**
Recommendation: our own, in the pool. It can be sticky per key, works across metals, and does delivery and sleep per replica.

> **Samarth:** agree — "im happy doing our own routing".

**Q2. Route per request with an affinity key, or pin per episode?**
Decision: per request, keyed by an `affinity` field the environment sets (default: the episode id). A seen key always returns to its replica; it moves only when that replica dies. No load-based rebalancing of keys.

> **Samarth:** "we should route per request, but the request should contain some metadata like i said before." On a load-based move cap: disagree — "i think it makes things too complicated".

**Q3. Who handles a replica that dies?**
Decision: the pool. It notices the death and retries in-flight requests on survivors. It re-carves the replica through the desk, possibly on another metal, delivers the committed bundle to it, and readmits it. Meanwhile the dead replica's keys are served by the survivors.

> **Samarth:** "the pool should be the unit responsible for replicas dying, and should revive them itself and potentially serve to other gpus in the meanwhile".

**Q4. dp and fsdp?**
Decision: they compose, with width = dp × fsdp on one metal. fsdp shards the frozen base within a group; dp splits the microbatches across groups and all-reduces the LoRA gradients in a fixed order.

> **Samarth:** "dp and fsdp are complementary right, we should be able to compose both of these together".

**Q5. Can a submission say which metals its hosts may land on?**
Today the carve rung takes the first metal by name with room, and nothing pins a submission. Measured on the B300 main desk, 2026-10-08: a dreamer would land on the bank's GPU 4, and N replicas could stack on one card.
Recommendation: an optional metal allow-list on the SUBMISSION, outside the spec so run identity is unchanged, plus a spread rule: replicas of one pool prefer distinct metals.
If no: operators size demands so hosts can't fit side by side, which wastes the leftover memory this ADR is meant to use.

> **Samarth:** disagree — "i feel like it's fine i dont really see a point". No allow-list or spread rule; placement stays first-fit by metal name. Consequence: replicas of one pool can share a card, and operators size demands when they need them apart.

## Outcome

Implemented on `claude/dreams` (267ca65 … e5eba7f); the record after the code is CONTEXT #97.

- **What landed, as decided.**
  - `PoolMember.replicas` and `LearnerMember.dp` are `identity: resource`: out of `run_id` and the manifest, carried on the wire by `wire_json`.
  - `runner/pool.py` holds the `Pool`: affinity routing (a seen key is sticky; a key moves only when its replica dies), health on contact, revival through the desk (`place-replica`), delivery to every live replica before the commit, and bring-up before readmission.
  - `client.sample/score(affinity=…)`, defaulting to the episode id.
  - The desk places one unit per replica.
  - `Learner.accumulate` runs on a dp × fsdp chorus that all-gathers per-microbatch gradients and sums them in microbatch order.
  - Measured booking: `booking` rows with an `overrun` warning on the host page.
- **What the answers changed.**
  - Q2: no rebalancing, so the key table only grows, and moves count deaths alone.
  - Q3: revival starts the moment a death is found, from a request or from a delivery.
  - Q4: Samarth chose exact identity over an all-reduce. dp = K steps dp = 1's gradient bit for bit, at the price of one gather per microbatch, to be measured on metal.
  - Q5: no allow-list, so replicas may share a card. They never share a LISTING, because one host is one engine; that is not a spread rule.
- **Consequences stated in the code.**
  - A replicated pool is its HostSpec's only member (`check_replicated_pools_stand_alone`). It never alternates, so a run that has one puts its learner on a host of its own.
  - `max_inflight` is now per replica.
  - A lost replica host no longer strands its run.
- **Metal found five bugs, all fixed:**
  - a closed resident door was not a death;
  - the in-flight bound was per run;
  - HTTP frames rode a 32-thread default executor;
  - the server's listen backlog was 5, which reset a healthy replica under a wave's first burst;
  - a revival gave up before the dead replica's GB was decarved.
- **Tests.** 2053 in the fakes suite, the known test_strangeloop_hosted the one failure. tests/test_replicas.py has 29; the torch-gated float64 CPU tests/test_dp_torch.py passed on the B300. test_resume.py is untouched.
- **Measured, B300 (results kept privately).** Two-turn GRPO on Qwen3.5-4B, 512 episodes per wave.
  - Sampling: 4.2k tokens/s at replicas = 1 against 5.2–6.7k at 3. The waves are tail-latency bound at that size.
  - Prefix-cache hit rate per replica 0.30–0.34, against 0.31 for one engine.
  - Train time per update unchanged.
  - An unplanned replica death was survived with the run completing.
- **Unproven.**
  - A clean revival on metal: the drill was running when the node went down at about 21:38 UTC.
  - dp = 2 on GPUs and the gather's share of step time.
  - Throughput scaling at a wave size that is not tail-bound.
