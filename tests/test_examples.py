"""SPEC.md §3, executable: the six worked examples, as far as Phase A reaches.

Phase A has no engines, so "run" here means what a submit does — construct the
spec, validate it jointly against a site schema (I4), and compute its identity
(I3) — plus, for the environment example, walking a trajectory through the
membrane by hand: fill → seal → advantage → flatten → pack.
"""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace
from typing import Any

from rlstack import (
    ADAPTER_TYPES,
    AdapterSpec,
    AlgoSpec,
    ExperimentSpec,
    GenSpec,
    GpuConfig,
    GpuGroup,
    Group,
    GroupPlan,
    Message,
    OptimSpec,
    PolicySpec,
    Role,
    Plans,
    RunPlan,
    Sample,
    Schedule,
    PostProcessor,
    Rollout,
    SamplingSpec,
    Seeds,
    Task,
    Turn,
    Wave,
    WavePlan,
    WaveRef,
    attn_bias,
    broadcast,
    canonical_json,
    code_hashes,
    encode,
    pool,
    environment,
    flatten,
    gpus,
    learner,
    lora,
    loss,
    fake_qwen_schema,
    pack,
    postprocessor,
    run_id,
    run_pipeline,
    soft_prompt,
    validate,
    validate_or_raise,
)
from rlstack import Bundle, FakeEngine  # noqa: E402

SCHEMA = fake_qwen_schema(32, base="Qwen/Qwen3-8B")
SCHEMA_35B = fake_qwen_schema(32, base="Qwen/Qwen3.5-35B-A3B")
SCHEMA_17B = fake_qwen_schema(32, base="Qwen/Qwen3-1.7B")
TASKS = "cas://3fa9c2/math_train.jsonl"
HELD_OUT = "cas://8c31f0/gsm_heldout.jsonl"
HELD_OUT_IDS = tuple(f"gsm-{i:04d}" for i in range(4))   # ids inside HELD_OUT
UPDATES = 30


# --- Example 4's registrations. math_single_turn and verifier are REAL
# builtins (inference/environments/, inference/rewards/) — importing rlstack
# registered them; only the still-stubbed ones are declared here.

from rlstack import Environment  # noqa: E402


@environment("tool_use")
class ToolUse(Environment):
    """Sample / execute-tool loop; tool-result tokens get loss_mask=0."""

    async def run(self, client: Any, task: Any) -> Any:
        raise NotImplementedError


# A judge is just a postprocessor that SAMPLES — scoring is post-seal
# analysis, never part of the episode and never inside the loss. The real
# one is a builtin now: training/post/llm_judge.py (produces "reward",
# declares pools=("judge",), carries its own SamplingSpec).


# --- Example 1 — a basic LoRA experiment, end to end -------------------------
#
# §3 wrote this run's shape as schedule knobs: group_size=4,
# trajectories_per_wave=16, n_updates=30, eval n_samples=2. Those knobs are
# gone (#59) — a run's shape IS its plan, so the same numbers are written
# below as data and the spec names each plan by the sha of its bytes.

def cas(plan: RunPlan) -> str:
    """The uri a store returns for a plan's bytes. Content addressing is what
    lets a spec name a plan before anyone writes it — and what makes a changed
    plan a different experiment without anyone saying so (I3)."""
    return f"cas://{hashlib.sha256(encode(plan)).hexdigest()}"


def group_of_four(task: str, env: str) -> GroupPlan:
    """One group: four samples of ONE task under one environment — GRPO's
    baseline scope, assigned here rather than derived from task identity."""
    return GroupPlan(task, tuple(Sample(task, env) for _ in range(4)))


def rollout_plan(env: str) -> RunPlan:
    """What the Generator MAKES: thirty waves of four groups of four — §3's
    "wave of 16", written out. WHICH task runs under WHICH environment is the
    plan's business; gen only declares which are nameable."""
    return RunPlan(tuple(
        WavePlan(tuple(group_of_four(f"math-{4 * u + g:04d}", env)
                       for g in range(4)))
        for u in range(UPDATES)))


def train_plan() -> RunPlan:
    """What the Trainer TAKES: update u trains on rollout u, whole — the
    on-policy pairing, one WaveRef instead of sixteen restated leaves. One
    wave is one gradient update, so this plan's LENGTH is the run's length."""
    return RunPlan(tuple(WaveRef(f"self://rollouts/{u}")
                         for u in range(1, UPDATES + 1)))


def plans_for(env: str) -> Plans:
    """The two plans of a run that makes and takes. Measurement has no plan
    here (#70): observing held-out tasks is a Measurement's own manifest,
    outside the run."""
    return Plans(train=cas(train_plan()), rollout=cas(rollout_plan(env)))


def example_1() -> ExperimentSpec:
    return ExperimentSpec(
        policy=PolicySpec(
            base="Qwen/Qwen3-8B",
            bank={
                "attn": lora(site="layers.0-15.self_attn.*", r=16),
                "mlp": lora(site="layers.16-31.mlp.*", r=32),
                "head": lora(site="layers.28-31.self_attn.o_proj", r=8),
            },
        ),
        # both task sets are DECLARED here; the firewall between them is the
        # plans', which is why eval no longer carries a task file of its own
        gen=GenSpec(envs=("math_single_turn",), tasks=(TASKS, HELD_OUT)),
        plans=plans_for("math_single_turn"),
        algo=AlgoSpec(
            loss="grpo",
            post=("verifier", "grpo_advantage"),
            optim=OptimSpec("adamw", lr=1e-5, betas=(0.9, 0.95),
                            overrides={"head": {"lr": 3e-6}}),
            schedule=Schedule(),
        ),
        gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(n=6), (pool("main", tp=2, n=3), learner(fsdp=2))),
            GpuGroup(gpus(n=1), (pool("eval"),)),
        )),
        seeds=Seeds(master=17),
    )


class TestExample1BasicLora(unittest.TestCase):
    def test_validates_clean(self) -> None:
        validate_or_raise(example_1(), SCHEMA)

    def test_the_plan_states_the_shape_the_schedule_used_to(self) -> None:
        """§3's narrative — "a wave of 16 (4 groups of 4)", thirty updates —
        read off the plan, which is where those numbers live now."""
        wave = rollout_plan("math_single_turn").wave(1)
        self.assertEqual([len(group.leaves) for group in wave.groups],
                         [4, 4, 4, 4])
        self.assertEqual(len(train_plan()), UPDATES)   # the run's length IS this
        self.assertEqual(train_plan().wave(1), WaveRef("self://rollouts/1"))

    def test_run_id_is_computed_never_typed(self) -> None:
        spec = example_1()
        rid = run_id(spec, code_hashes(spec), data_fingerprint="sha256:feed")
        self.assertEqual(rid, run_id(spec, code_hashes(spec), "sha256:feed"))
        self.assertEqual(len(rid), 12)

    def test_editing_a_registered_body_would_change_the_run(self) -> None:
        spec = example_1()
        hashes = code_hashes(spec)
        self.assertIn("loss:grpo", hashes)
        tampered = dict(hashes, **{"loss:grpo": "0" * 64})
        self.assertNotEqual(run_id(spec, hashes, "d"), run_id(spec, tampered, "d"))


# --- Example 2 — adding PPO: what new code exists ----------------------------

@postprocessor("gae")
class Gae(PostProcessor):
    """GAE as a postprocessor: per group, consumes rewards, emits advantages."""

    consumes = ("reward",)
    produces = ("gae_advantage",)

    async def process(self, group: Any, data: Any, client: Any) -> Any:
        raise NotImplementedError  # detached, CPU, group-scope


@loss("ppo_critic", requires=("values", "gae_advantage"))
def ppo_critic(out: Any, batch: Any) -> Any:
    raise NotImplementedError  # grads flow to the critic head


class TestExample2PPO(unittest.TestCase):
    def test_the_entire_diff_is_a_bank_entry_and_an_algo(self) -> None:
        exp = example_1()
        bank = dict(exp.policy.bank)
        bank["critic"] = AdapterSpec(adapter_type="value_head", site="final_hidden",
                                     init={"hidden": 4096})
        exp2 = replace(
            exp,
            policy=replace(exp.policy, bank=bank),
            algo=replace(exp.algo, loss="ppo_critic", post=("verifier", "gae")),
        )
        validate_or_raise(exp2, SCHEMA)  # value_head provides "values"; ppo_critic needs it

    def test_ppo_without_a_critic_is_caught_at_submit(self) -> None:
        exp = replace(example_1(), algo=replace(example_1().algo, loss="ppo_critic"))
        self.assertEqual({i.code for i in validate(exp, SCHEMA)},
                         {"unsatisfied-requires"})


# --- Example 3 — soft prompt + learned attention bias ------------------------

class TestExample3SoftPromptAttnBias(unittest.TestCase):
    def test_validates_clean(self) -> None:
        """The bias site is not in any schema: the soft prompt EXPORTS it."""
        exp = replace(example_1(), policy=PolicySpec(
            base="Qwen/Qwen3.5-35B-A3B",
            bank={
                "latent": soft_prompt("prompt[:8]", n=8, d=2048),
                "readout": attn_bias("queries -> prompt[:8]",
                                     param="bounded_sigmoid", cap="ln(64)"),
            },
        ))
        validate_or_raise(exp, SCHEMA_35B)

    def test_only_attn_bias_touches_the_engine(self) -> None:
        self.assertEqual(ADAPTER_TYPES.get("attn_bias").instance.engine_plugin,
                         "rlstack_engine.side_attention")
        for name in ("lora", "soft_prompt", "value_head"):
            self.assertIsNone(ADAPTER_TYPES.get(name).instance.engine_plugin)


# --- Example 4 — environments: the data path through the membrane ------------

def char_tokenize(text: str) -> tuple[int, ...]:
    return tuple(ord(c) for c in text)


def one_rollout(task_id: str, answer: str, correct: float):
    """What math_single_turn + verifier produce, built by hand: a Rollout is
    filled in the inference world, then sealed into a Trajectory."""
    turn = Turn(
        message=Message(Role.ASSISTANT, answer),
        token_ids=char_tokenize(answer),          # the ENGINE's ids, verbatim (I6)
        behavior_logprobs=tuple(-0.5 for _ in answer),
        finish="eos", stop_hit=None,
        bundle_id="bundle:abc", policy_version={"attn": 3}, seed=7,
    )
    rollout = Rollout(task=Task(task_id, "2+40=?", {"answer": 42}),
                      messages=[Message(Role.USER, "2+40=?"), turn.message],
                      turns=[turn])
    return rollout.seal()  # the membrane closes here; scoring comes AFTER


class TestExample4DataPath(unittest.TestCase):
    def test_seal_post_flatten_pack(self) -> None:
        import asyncio

        # One group of 4 trajectories of one task — the group is the scope a
        # partial loss contribution (here, the GRPO baseline) is computed over.
        wave = Wave([Group("t0", [
            one_rollout("t0", "42", 1.0), one_rollout("t0", "41", 0.0),
            one_rollout("t0", "42", 1.0), one_rollout("t0", "40", 0.0)])])

        # Postprocessing: the declared pipeline runs per group, after the seal.
        bundle = Bundle("bundle:ex4", {})
        engine = FakeEngine()
        engine.add_bundle(bundle)
        columns = asyncio.run(run_pipeline(
            ("verifier", "grpo_advantage"), wave, {"main": (engine, bundle)},
            SamplingSpec(), master=17, update=1))
        self.assertEqual(columns["reward"], [1.0, 0.0, 1.0, 0.0])
        adv = columns["advantage"]
        self.assertAlmostEqual(sum(adv), 0.0)     # group-centered
        self.assertGreater(adv[0], 0)             # correct above the mean
        self.assertLess(adv[1], 0)

        # Flatten: one pass captures ids, mask, and recorded logprobs together —
        # prompt tokens injected (mask 0, logprob 0), generated verbatim (I6).
        flats = [flatten(t, char_tokenize) for t in wave.trajectories]
        first = flats[0]
        prompt_len = len("2+40=?")
        self.assertEqual(first.loss_mask[:prompt_len], (0,) * prompt_len)
        self.assertEqual(first.token_ids[prompt_len:], char_tokenize("42"))
        self.assertEqual(first.behavior_logprobs[prompt_len:], (-0.5, -0.5))

        # Pack: postdata rides as named per-token columns next to the logprobs.
        docs = list(zip(flats, broadcast(columns, flats)))
        batches = pack(docs, microbatch_tokens=16)
        self.assertEqual(sum(len(b.doc_starts) for b in batches), 4)
        for batch in batches:
            self.assertEqual(len(batch.token_ids), len(batch.postdata["advantage"]))


# --- Example 5 — multi-GPU with sharding: 35B on three nodes -----------------

class TestExample5MultiNode(unittest.TestCase):
    def test_topology_is_pure_demand(self) -> None:
        exp = replace(
            example_1(),
            policy=PolicySpec(base="Qwen/Qwen3.5-35B-A3B",
                              bank={"pi": lora("layers.0-31.self_attn.*", r=32)}),
            # a leaf names its environment, so a run under a different env is a
            # different plan — the declaration and the plan move together
            gen=replace(example_1().gen, envs=("tool_use",)),
            plans=plans_for("tool_use"),
            algo=replace(example_1().algo,
                         schedule=Schedule(max_policy_lag=1)),
            gpu_config=GpuConfig(groups=(
                GpuGroup(gpus(n=16, nodes=2), (pool("main", tp=2, n=8),)),
                GpuGroup(gpus(n=8), (learner(fsdp=8),)),
                GpuGroup(gpus(n=1), (pool("eval"),)),
            )),
        )
        validate_or_raise(exp, SCHEMA_35B)
        # Semantics-neutral (I5): no provider name anywhere in the spec.
        for provider in ("modal", "skypilot", "aws", "H100"):
            self.assertNotIn(provider, canonical_json(exp))


# --- Example 6 — one GPU, 1B model, many replicates --------------------------

class TestExample6Replicates(unittest.TestCase):
    def base(self) -> ExperimentSpec:
        return replace(
            example_1(),
            policy=PolicySpec(base="Qwen/Qwen3-1.7B",
                              bank={"pi": lora("layers.0-31.mlp.*", r=16)}),
            plans=example_1().plans,
            gpu_config=GpuConfig(groups=(
                GpuGroup(gpus(ids=("0",)),
                      (pool("main", n=2, fraction=0.30), learner(fraction=0.25))),
            )),
        )

    def test_seed_replicates_are_five_distinct_runs(self) -> None:
        base = self.base()
        validate_or_raise(base, SCHEMA_17B)
        hashes = code_hashes(base)
        ids = {run_id(replace(base, seeds=Seeds(master=s)), hashes, "d")
               for s in range(5)}
        self.assertEqual(len(ids), 5)  # five run_ids, one resident engine

    def test_the_memory_treaty_is_checked_at_submit(self) -> None:
        overfull = replace(self.base(), gpu_config=GpuConfig(groups=(
            GpuGroup(gpus(ids=("0",)),
                  (pool("main", n=2, fraction=0.80), learner(fraction=0.25))),)))
        self.assertEqual({i.code for i in validate(overfull, SCHEMA_17B)},
                         {"fraction-overflow"})


if __name__ == "__main__":
    unittest.main()
