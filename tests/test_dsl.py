"""The DSL campaign's pieces: two invented tool languages, their graders, and
the loop and the two spectral adapters wired end to end on fakes.

The claims under test: the graders pay the milestone ladders exactly (dense by
design — the DAPO all-or-nothing lesson); the task builders are deterministic
content with the stated generalization holes (mol never trains, ring
directions eval wider than they train); validate refuses the new loss when
its provider is missing and passes the campaign's actual arm
shapes; and each arm shape RUNS on fake metal — seal, post, train, ledger —
with the reverse head's provided name reaching the ledger's train block.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from common import cas_uri, make_turn
from rlstack import (
    AlgoSpec, ExperimentSpec, FakeEngine, FakeLearner, GenSpec, GpuConfig,
    HostSpec, GroupPlan, Host, LocalStore, Message, OptimSpec, Plans,
    PolicySpec, Role, Rollout, RunPlan, Sample, Schedule, Seeds, Task,
    WavePlan, WaveRef, encode, fake_qwen_schema, learner, lora, pool,
    validate, write_tasks,
)
from rlstack.data.plan import Derive, decode
from rlstack.data.tasks.glyph_exchange import glyph_eval_tasks, glyph_train_tasks
from rlstack.data.tasks.stamp_office import stamp_eval_tasks, stamp_train_tasks
from rlstack.inference.makers.reflect import Reflect
from rlstack.policy.adapters.spectral import spectral
from rlstack.policy.adapters.spectral_latent import spectral_latent
from rlstack.policy.adapters.value_head import value_head
from rlstack.registry import ADAPTER_TYPES, ENVS, LOSSES, MAKERS, POST
from rlstack.training.post.glyph_grade import GlyphGrade
from rlstack.training.post.stamp_grade import StampGrade


BASE = "Qwen/Qwen3-0.6B"
SCHEMA = fake_qwen_schema(4, base=BASE)

GOLD_STAMP = "grab(47)\nfold(47)\nink(47, vex)\nseal(47)\nfile(47, D3)"


def go(coro):
    return asyncio.run(coro)


class StampGradeTest(unittest.TestCase):
    def grade(self, completion: str, doc: str = "47", color: str = "vex"):
        return StampGrade().grade(completion, doc, color)

    def test_the_gold_path_pays_in_full(self) -> None:
        self.assertEqual(self.grade(GOLD_STAMP), (1.0, 1.0))

    def test_the_ladder_pays_progress(self) -> None:
        """Each legal prefix of the gold path earns its rung — the dense
        reward a 0.6B needs (the DAPO all-or-nothing lesson)."""
        self.assertEqual(self.grade("grab(47)"), (0.15, 0.0))
        self.assertEqual(self.grade("grab(47)\nfold(47)"), (0.30, 0.0))
        self.assertEqual(self.grade("grab(47)\nfold(47)\nink(47, vex)"),
                         (0.45, 0.0))
        self.assertEqual(
            self.grade("grab(47)\nfold(47)\nink(47, vex)\nseal(47)"),
            (0.60, 0.0))

    def test_a_rung_counts_only_on_the_rungs_below(self) -> None:
        """Sealed over the WRONG ink is a document that left the gold path at
        the ink: the ladder stops at folded."""
        reward, exact = self.grade(
            "grab(47)\nfold(47)\nink(47, mol)\nseal(47)\nfile(47, D2)")
        self.assertEqual((reward, exact), (0.30, 0.0))

    def test_illegal_calls_cost_and_floor_at_zero(self) -> None:
        reward, exact = self.grade("seal(47)\n" + GOLD_STAMP)   # seal first: illegal
        self.assertEqual((round(reward, 6), exact), (0.95, 0.0))
        reward, _ = self.grade("what is a stamp office?")
        self.assertEqual(reward, 0.0)

    def test_the_tray_holds_only_the_named_document(self) -> None:
        reward, _ = self.grade("grab(99)\ngrab(47)")
        self.assertEqual(round(reward, 6), 0.10)    # grabbed, one illegal

    def test_folding_twice_ruins_the_document(self) -> None:
        reward, _ = self.grade("grab(47)\nfold(47)\nfold(47)\nink(47, vex)")
        # second fold illegal (skipped), so ink lands on a once-folded doc
        self.assertEqual(round(reward, 6), 0.40)

    def test_the_drawer_is_fixed_by_the_ink(self) -> None:
        reward, exact = self.grade(
            "grab(47)\nfold(47)\nink(47, vex)\nseal(47)\nfile(47, D1)")
        self.assertEqual((round(reward, 6), exact), (0.55, 0.0))


class GlyphGradeTest(unittest.TestCase):
    def grade(self, completion: str, purse: str = "P7", source: str = "wex",
              target: str = "sarn"):
        return GlyphGrade().grade(completion, purse, source, target)

    def test_the_gold_route_pays_in_full(self) -> None:
        self.assertEqual(
            self.grade("give(to_sarn(to_polk(to_drin(load(P7)))))"),
            (1.0, 1.0))

    def test_prose_around_the_expression_is_tolerated(self) -> None:
        self.assertEqual(
            self.grade("Sure! give(to_sarn(to_polk(to_drin(load(P7))))) done"),
            (1.0, 1.0))

    def test_the_route_pays_its_correct_prefix(self) -> None:
        reward, exact = self.grade("give(to_polk(to_drin(load(P7))))")
        self.assertEqual((round(reward, 6), exact), (0.6, 0.0))  # 2 of 3 steps

    def test_a_wrong_step_ends_the_prefix_and_costs(self) -> None:
        reward, exact = self.grade("give(to_wex(to_drin(load(P7))))")
        # one correct step (0.2), then a wrong one: spurious, -0.05
        self.assertEqual((round(reward, 6), exact), (0.35, 0.0))

    def test_the_wrong_purse_scores_nothing(self) -> None:
        self.assertEqual(self.grade("give(to_drin(load(P9)))",
                                    source="wex", target="drin"), (0.0, 0.0))

    def test_the_ring_wraps(self) -> None:
        self.assertEqual(
            self.grade("give(to_drin(to_wex(load(P2))))", purse="P2",
                       source="sarn", target="drin"), (1.0, 1.0))

    def test_garbage_scores_zero(self) -> None:
        self.assertEqual(self.grade("the answer is 42"), (0.0, 0.0))


class TaskBuilderTest(unittest.TestCase):
    def test_the_sets_are_deterministic_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStore(tmp)
            first = write_tasks(store, stamp_train_tasks())
            second = write_tasks(store, stamp_train_tasks())
            self.assertEqual(first, second)      # same bytes, same identity

    def test_the_training_hole_is_real(self) -> None:
        """mol never appears in a stamp train task; every eval color does —
        memorizing the train answers has nothing to say about mol."""
        train_colors = {t.meta["color"] for t in stamp_train_tasks()}
        eval_colors = {t.meta["color"] for t in stamp_eval_tasks()}
        self.assertNotIn("mol", train_colors)
        self.assertEqual(eval_colors, {"rju", "vex", "mol"})
        self.assertEqual(len(stamp_train_tasks()), 2)
        self.assertEqual(len(stamp_eval_tasks()), 72)

    def test_the_glyph_eval_sweeps_directions_the_train_never_shows(self) -> None:
        train_pairs = {(t.meta["source"], t.meta["target"])
                       for t in glyph_train_tasks()}
        eval_pairs = {(t.meta["source"], t.meta["target"])
                      for t in glyph_eval_tasks()}
        self.assertEqual(len(glyph_train_tasks()), 2)
        self.assertEqual(len(eval_pairs), 12)     # every ordered pair
        self.assertIn(("sarn", "wex"), eval_pairs - train_pairs)

    def test_ids_are_unique_across_both_families(self) -> None:
        everything = (stamp_train_tasks() + stamp_eval_tasks()
                      + glyph_train_tasks() + glyph_eval_tasks())
        self.assertEqual(len({t.id for t in everything}), len(everything))


# ---------------------------------------------------------------------------
# the campaign's arm shapes: registration, validation, and fake metal
# ---------------------------------------------------------------------------

def stamp_plan(task_ids: list[str], updates: int, size: int = 2) -> RunPlan:
    return RunPlan(tuple(
        WavePlan(tuple(
            GroupPlan(task, tuple(Sample(task, "stamp_office")
                                  for _ in range(size)))
            for task in task_ids))
        for _ in range(updates)))


def arm_spec(store, *, loss: str, post: tuple[str, ...],
             bank=None, updates: int = 2) -> ExperimentSpec:
    tasks = stamp_train_tasks()
    uri = write_tasks(store, tasks)
    rollout = encode(stamp_plan([t.id for t in tasks], updates))
    train = encode(RunPlan(tuple(WaveRef(f"self://rollouts/{u}")
                                 for u in range(1, updates + 1))))
    store.cas_put(rollout)
    store.cas_put(train)
    return ExperimentSpec(
        policy=PolicySpec(base=BASE, bank=bank if bank is not None
                          else {"pi": lora("layers.0-3.self_attn.*", r=8)}),
        gen=GenSpec(envs=("stamp_office",), tasks=(uri,)),
        plans=Plans(train=cas_uri(train), rollout=cas_uri(rollout)),
        algo=AlgoSpec(loss=loss, post=post,
                      optim=OptimSpec("adamw", lr=1e-5),
                      schedule=Schedule(microbatch_tokens=2048)),
        gpu_config=GpuConfig(hosts=(HostSpec((pool("main"),)),
                                    HostSpec((learner(),)))),
        seeds=Seeds(master=23))


class WiringTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = LocalStore(tmp.name)

    def test_everything_is_registered(self) -> None:
        for name in ("stamp_grade", "glyph_grade"):
            self.assertIsNotNone(POST.get(name))
        for name in ("stamp_office", "glyph_exchange"):
            self.assertIsNotNone(ENVS.get(name))
        for name in ("value_head", "spectral", "spectral_latent"):
            self.assertIsNotNone(ADAPTER_TYPES.get(name))
        self.assertIsNotNone(MAKERS.get("reflect"))
        self.assertIsNotNone(LOSSES.get("sdpo"))
        self.assertIsNotNone(ENVS.get("reflect_retry"))

    def test_the_arm_shapes_validate_clean(self) -> None:
        arms = (
            arm_spec(self.store, loss="grpo",
                     post=("stamp_grade", "grpo_advantage")),
            arm_spec(self.store, loss="grpo",
                     post=("stamp_grade", "grpo_advantage"),
                     bank={"pi": spectral("layers.0-3.self_attn.*", k=8)}),
            arm_spec(self.store, loss="grpo_latent_kl_gated",
                     post=("stamp_grade", "grpo_advantage", "group_accuracy"),
                     bank={"pi": spectral_latent("layers.0-3.self_attn.*",
                                                 k=8, latent=8, members=2)}),
        )
        for spec in arms:
            self.assertEqual(validate(spec, SCHEMA), [],
                             spec.policy.bank["pi"].adapter_type)

    def test_the_gated_loss_without_a_latent_is_refused(self) -> None:
        """grpo_latent_kl_gated requires the latent_kl provide: a plain
        spectral bank (no latent, no KL) cannot carry it."""
        spec = arm_spec(self.store, loss="grpo_latent_kl_gated",
                        post=("stamp_grade", "grpo_advantage",
                              "group_accuracy"),
                        bank={"pi": spectral("layers.0-3.self_attn.*", k=8)})
        codes = {issue.code for issue in validate(spec, SCHEMA)}
        self.assertIn("unsatisfied-requires", codes)

    def test_the_head_refuses_a_weighted_site(self) -> None:
        bank = {"pi": lora("layers.0-3.self_attn.*", r=8),
                "vh": value_head("layers.0.self_attn.q_proj", d_model=64)}
        spec = arm_spec(self.store, loss="grpo",
                        post=("stamp_grade", "grpo_advantage"), bank=bank)
        codes = {issue.code for issue in validate(spec, SCHEMA)}
        self.assertTrue(codes)      # site_ok says no somewhere in the report


class ArmRunTest(unittest.TestCase):
    """Each arm shape runs whole on fake metal: seal, post, train, ledger."""

    def run_arm(self, loss: str, post: tuple[str, ...], bank=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = LocalStore(tmp.name)
        spec = arm_spec(store, loss=loss, post=post, bank=bank)
        host = Host("dsl-fake", engines=(FakeEngine(),), learner=FakeLearner(),
                    store=store)
        report = go(host.submit(spec, SCHEMA, store=store))
        entries = store.peek_ledger(report.run_id)
        self.assertEqual(len(entries), 2, entries)
        return entries

    def test_grpo_on_the_stamp_office(self) -> None:
        self.run_arm("grpo", ("stamp_grade", "grpo_advantage"))

    def test_spectral_grpo_on_the_stamp_office(self) -> None:
        entries = self.run_arm(
            "grpo", ("stamp_grade", "grpo_advantage"),
            bank={"pi": spectral("layers.0-3.self_attn.*", k=8)})
        self.assertIn("spectral_gain_span", json.dumps(entries[-1]))

    def test_spectral_latent_gated_on_the_stamp_office(self) -> None:
        entries = self.run_arm(
            "grpo_latent_kl_gated",
            ("stamp_grade", "grpo_advantage", "group_accuracy"),
            bank={"pi": spectral_latent("layers.0-3.self_attn.*",
                                        k=8, latent=8, members=2)})
        self.assertIn("latent_kl", json.dumps(entries[-1]))


def sealed_attempt(task: Task, content: str):
    """One sealed single-turn episode on `task` — the reflect maker's food."""
    turn = make_turn(content, tuple(ord(c) for c in content))
    return Rollout(task=task,
                   messages=[Message(Role.USER, task.prompt), turn.message],
                   turns=[turn]).seal()


class ReflectLoopTest(unittest.TestCase):
    def test_derive_rides_the_plan_wire(self) -> None:
        plan = RunPlan((WavePlan((GroupPlan("g", (
            Derive("self://rollouts/1#0", "reflect", "reflect_retry"),)),)),))
        again = decode(encode(plan))
        self.assertEqual(
            again.waves[0].groups[0].leaves[0],
            Derive("self://rollouts/1#0", "reflect", "reflect_retry"))

    def test_the_reflect_maker_extends_the_transcript(self) -> None:
        """The derived prompt IS the conversation so far plus the ask; the
        metadata passes through (the graders judge retries), the reflection
        counter climbs, and pointing the maker at its own output iterates."""
        task = stamp_train_tasks()[0]
        first = Reflect().make(sealed_attempt(task, "grab(47)"))
        self.assertTrue(first.prompt.startswith(task.prompt))
        self.assertIn("grab(47)", first.prompt)
        self.assertIn("wrong", first.prompt)
        self.assertEqual(first.meta["color"], "vex")
        self.assertEqual(first.meta["reflection"], 1)
        self.assertEqual(first.id, f"{task.id}~r1")
        second = Reflect().make(sealed_attempt(first, "fold(47)"))
        self.assertEqual(second.meta["reflection"], 2)
        self.assertEqual(second.id, f"{task.id}~r2")
        self.assertIn("fold(47)", second.prompt)

    def test_the_reflect_env_criticizes_then_retries(self) -> None:
        class StubClient:
            def __init__(self) -> None:
                self.calls = 0

            async def sample(self, messages, stop=()):
                self.calls += 1
                return make_turn(f"turn-{self.calls}", (self.calls,))

            def pool(self, name):
                return self

        from rlstack.registry import ENVS
        task = stamp_train_tasks()[0]
        rollout = go(ENVS.get("reflect_retry").instance.run(StubClient(), task))
        self.assertEqual(len(rollout.turns), 2)
        self.assertEqual(len(rollout.messages), 4)
        self.assertIn("corrected", rollout.messages[2].content)  # the injected ask
        self.assertEqual(rollout.turns[-1].message.content, "turn-2")

    def test_the_loop_arm_runs_on_fake_metal(self) -> None:
        """The whole iterative-SDPO shape, fakes end to end: Sample wave, two
        Derive waves minted from it in order, training only on the loop-final
        waves, the sdpo loss cloning final turns — two loops, two updates."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = LocalStore(tmp.name)
        tasks = stamp_train_tasks()
        uri = write_tasks(store, tasks)
        ids = [t.id for t in tasks]

        def loop_waves(base: int) -> list[WavePlan]:
            attempt = WavePlan(tuple(
                GroupPlan(t, (Sample(t, "stamp_office"),)) for t in ids))
            reflect = lambda source_wave: WavePlan(tuple(
                GroupPlan(t, (Derive(f"self://rollouts/{source_wave}#{i}",
                                     "reflect", "reflect_retry"),))
                for i, t in enumerate(ids)))
            return [attempt, reflect(base + 1), reflect(base + 2)]

        rollout = encode(RunPlan(tuple(loop_waves(0) + loop_waves(3))))
        train = encode(RunPlan((WaveRef("self://rollouts/3"),
                                WaveRef("self://rollouts/6"))))
        store.cas_put(rollout)
        store.cas_put(train)
        spec = ExperimentSpec(
            policy=PolicySpec(base=BASE,
                              bank={"pi": lora("layers.0-3.self_attn.*", r=8)}),
            gen=GenSpec(envs=("stamp_office", "reflect_retry"), tasks=(uri,),
                        makers=("reflect",)),
            plans=Plans(train=cas_uri(train), rollout=cas_uri(rollout)),
            algo=AlgoSpec(loss="sdpo", post=("stamp_grade",),
                          optim=OptimSpec("adamw", lr=1e-5),
                          schedule=Schedule(microbatch_tokens=4096)),
            gpu_config=GpuConfig(hosts=(HostSpec((pool("main"),)),
                                        HostSpec((learner(),)))),
            seeds=Seeds(master=29))
        self.assertEqual(validate(spec, SCHEMA), [])
        host = Host("loop-fake", engines=(FakeEngine(),), learner=FakeLearner(),
                    store=store)
        report = go(host.submit(spec, SCHEMA, store=store))
        entries = store.peek_ledger(report.run_id)
        self.assertEqual(len(entries), 2, entries)

    def test_an_unknown_maker_is_a_validate_finding(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store = LocalStore(tmp.name)
        spec = arm_spec(store, loss="grpo",
                        post=("stamp_grade", "grpo_advantage"))
        import dataclasses
        spec = dataclasses.replace(
            spec, gen=dataclasses.replace(spec.gen, makers=("nonesuch",)))
        codes = {issue.code for issue in validate(spec, SCHEMA)}
        self.assertIn("unknown-maker", codes)


# ---------------------------------------------------------------------------
# the numerical halves — torch ships in the deploy image, so these SKIP
# locally and RUN in the image (test_plora's rule)
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:                                  # the client environment
    torch = None

needs_torch = unittest.skipUnless(
    torch is not None, "torch is trainer metal: this suite runs in the image")

if torch is not None:
    from rlstack.data.flatten import TokenBatch
    from rlstack.policy.adapters import (
        spectral_latent_torch, spectral_torch, value_head_torch,
    )
    from rlstack.policy.siteschema import SiteMeta
    from rlstack.training.losses import PolicyOutputs
    from rlstack.training.losses.sdpo import sdpo as sdpo_loss


@needs_torch
class SpectralTorchTest(unittest.TestCase):
    def one_site(self, k: int = 2, n: int = 6):
        site = SiteMeta(name="lin", path="lin", has_weight=True,
                        shape=(n, n), is_boundary=False)
        state = spectral_torch.build((site,), {"k": k})
        model = torch.nn.Module()
        model.lin = torch.nn.Linear(n, n, bias=False)
        with torch.no_grad():
            model.lin.weight.copy_(torch.randn(
                n, n, generator=torch.Generator().manual_seed(5)))
        spectral_torch.install(model, state)
        return state, model

    def test_the_value_is_topk_and_the_gradient_is_dense(self) -> None:
        """The straight-through contract: at most k directions move the
        forward VALUE, every direction receives GRADIENT."""
        state, _ = self.one_site(k=2, n=6)
        with torch.no_grad():
            state.delta["lin"].copy_(torch.tensor([.1, .2, .3, .4, .5, .6]))
        eff = spectral_torch.effective_gains(state, "lin")
        self.assertEqual(int((eff.detach() != 0).sum()), 2)
        eff.sum().backward()
        self.assertEqual(int((state.delta["lin"].grad != 0).sum()), 6)

    def test_the_replay_delta_equals_the_served_adapter(self) -> None:
        """Parity by construction: the forward's delta and the emitted peft
        pair compute the same matrix."""
        state, model = self.one_site(k=2, n=6)
        with torch.no_grad():
            state.delta["lin"].uniform_(-0.5, 0.5,
                                        generator=torch.Generator().manual_seed(7))
        x = torch.randn(3, 6, generator=torch.Generator().manual_seed(9))
        replayed = spectral_torch._whole_batch_delta(x, state, "lin")
        _, tensors = spectral_torch.unpack(spectral_torch.emit(state))
        a = tensors["peft.base_model.model.lin.lora_A.weight"].to(torch.float32)
        b = tensors["peft.base_model.model.lin.lora_B.weight"].to(torch.float32)
        served = (x @ a.T) @ b.T
        self.assertTrue(torch.allclose(replayed, served, atol=2e-2),
                        (replayed - served).abs().max())

    def test_emit_load_roundtrips_the_dense_gains(self) -> None:
        state, model = self.one_site()
        with torch.no_grad():
            state.delta["lin"].uniform_(-1, 1,
                                        generator=torch.Generator().manual_seed(3))
        payload = spectral_torch.emit(state)
        fresh, _ = self.one_site()
        spectral_torch.load(fresh, payload)
        self.assertTrue(torch.equal(fresh.delta["lin"], state.delta["lin"]))

    def test_k_past_the_spectrum_is_refused(self) -> None:
        site = SiteMeta(name="lin", path="lin", has_weight=True,
                        shape=(4, 8), is_boundary=False)
        with self.assertRaises(ValueError):
            spectral_torch.build((site,), {"k": 5})


@needs_torch
class ValueHeadTorchTest(unittest.TestCase):
    def head_on_stub(self):
        site = SiteMeta(name="final_hidden", path="norm", has_weight=False,
                        shape=None, is_boundary=True)
        state = value_head_torch.build((site,), {"d_model": 8, "hidden": 4})

        class Stub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = torch.nn.Identity()
                self.anchor = torch.nn.Parameter(torch.zeros(1))

            def forward(self, h, attention_mask=None):
                return self.norm(h)

        model = Stub()
        value_head_torch.install(model, state)
        return state, model

    def test_values_are_zero_at_version_zero_and_padding_is_zeroed(self) -> None:
        state, model = self.head_on_stub()
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        model(torch.randn(1, 3, 8), attention_mask=mask)
        v = value_head_torch.values(state)
        self.assertTrue(torch.equal(v, torch.zeros(1, 3)))   # zero-init head
        with torch.no_grad():
            state.w_out.uniform_(-1, 1)
        model(torch.randn(1, 3, 8), attention_mask=mask)
        v = value_head_torch.values(state)
        self.assertNotEqual(float(v[0, 0]), 0.0)
        self.assertEqual(float(v[0, 2]), 0.0)                # padding zeroed

    def test_the_capture_is_detached(self) -> None:
        state, model = self.head_on_stub()
        h = torch.randn(1, 2, 8, requires_grad=True)
        model(h, attention_mask=torch.ones(1, 2))
        self.assertFalse(state.captured.requires_grad)


def _hand_batch(reward: float = 1.0):
    """One document: 1 injected token then 4 generated, two turns (1 + 3)."""
    return TokenBatch(
        token_ids=(9, 1, 2, 3, 4), loss_mask=(0, 1, 1, 1, 1),
        behavior_logprobs=(0.0, -0.5, -0.5, -0.5, -0.5),
        segment_ids=(-1, 0, 1, 1, 1), doc_starts=(0,),
        postdata={"reward": (0.0, reward, reward, reward, reward)})


@needs_torch
class LossMathTest(unittest.TestCase):
    def test_sdpo_clones_the_final_turn_only(self) -> None:
        batch = _hand_batch()
        lp = torch.tensor([0.0, -1.0, -2.0, -4.0, -6.0], requires_grad=True)
        result = sdpo_loss(PolicyOutputs(logprobs=lp), batch)
        self.assertAlmostEqual(float(result.loss), (2.0 + 4.0 + 6.0) / 3, places=6)
        result.loss.backward()
        self.assertEqual(float(lp.grad[1]), 0.0)     # turn 0: conditioning only


@needs_torch
class SpectralLatentTorchTest(unittest.TestCase):
    def one_site(self, k: int = 2, n: int = 6, latent: int = 4):
        site = SiteMeta(name="lin", path="lin", has_weight=True,
                        shape=(n, n), is_boundary=False)
        state = spectral_latent_torch.build(
            (site,), {"k": k, "latent": latent, "members": 2,
                      "prior_std": 0.05, "hidden": 8})
        model = torch.nn.Module()
        model.lin = torch.nn.Linear(n, n, bias=False)
        with torch.no_grad():
            model.lin.weight.copy_(torch.randn(
                n, n, generator=torch.Generator().manual_seed(5)))
        spectral_latent_torch.install(model, state)
        return state, model

    def test_version_zero_is_the_base_for_every_latent(self) -> None:
        """Zero heads make every gain zero whatever z says, and the KL starts
        at exactly zero — the identity element, spectral_latent's copy."""
        state, _ = self.one_site()
        z = torch.randn(4, generator=torch.Generator().manual_seed(1))
        eff = spectral_latent_torch.effective_gains(state, "lin", z)
        self.assertTrue(torch.equal(eff.detach(), torch.zeros(6)))
        provided = spectral_latent_torch.provide(state)
        self.assertAlmostEqual(float(provided["latent_kl"]), 0.0, places=6)

    def test_the_member_payload_matches_the_replay_delta(self) -> None:
        """Parity: the served member's peft pair and the replay expression
        under that member's noise compute the same matrix."""
        state, _ = self.one_site()
        with torch.no_grad():
            for head in state.heads.values():
                head.uniform_(-0.5, 0.5,
                              generator=torch.Generator().manual_seed(11))
        payload = spectral_latent_torch.emit(state)
        head, tensors = spectral_latent_torch.unpack(payload)
        eps = tensors["noise"][0]
        x = torch.randn(3, 6, generator=torch.Generator().manual_seed(9))
        eff = spectral_latent_torch.effective_gains(
            state, "lin", spectral_latent_torch.reparameterized_latent(
                state, eps))
        v = state.v["lin"].to(torch.float32)
        u = state.u["lin"].to(torch.float32)
        replayed = ((x @ v) * eff.detach()) @ u.T
        a = tensors["peft.m0.base_model.model.lin.lora_A.weight"].to(torch.float32)
        b = tensors["peft.m0.base_model.model.lin.lora_B.weight"].to(torch.float32)
        self.assertTrue(torch.allclose(replayed, (x @ a.T) @ b.T, atol=2e-2))

    def test_emit_load_roundtrips_counter_included(self) -> None:
        state, _ = self.one_site()
        with torch.no_grad():
            state.mu.uniform_(-1, 1, generator=torch.Generator().manual_seed(2))
        first = spectral_latent_torch.emit(state)      # version 0, then 1
        self.assertEqual(state.version, 1)
        fresh, _ = self.one_site()
        spectral_latent_torch.load(fresh, first)
        self.assertEqual(fresh.version, 0)             # restored, pre-advance
        self.assertTrue(torch.equal(fresh.mu.data, state.mu.data))


@needs_torch
class DevicePlacementTest(unittest.TestCase):
    """The GPU-only device-mismatch class the fakes suite cannot see: seeded
    Generators draw on CPU, weights live on cuda, and every recomposition
    must meet on the weight's device (both bugs observed live on dsl-b)."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("cuda-only: the mismatch needs two devices")

    def test_random_factors_follow_the_weight_device(self) -> None:
        from rlstack.policy.adapters import plora_factors

        weight = torch.randn(8, 6, device="cuda")
        u, a = plora_factors.random_factors(weight, 3, 7, "lin")
        self.assertEqual(u.device.type, "cuda")
        self.assertEqual(a.device.type, "cuda")

    def test_slatent_emits_from_a_cuda_state(self) -> None:
        site = SiteMeta(name="lin", path="lin", has_weight=True,
                        shape=(6, 6), is_boundary=False)
        state = spectral_latent_torch.build(
            (site,), {"k": 2, "latent": 4, "members": 2,
                      "prior_std": 0.05, "hidden": 8})
        model = torch.nn.Module()
        model.lin = torch.nn.Linear(6, 6, bias=False).cuda()
        spectral_latent_torch.install(model, state)
        payload = spectral_latent_torch.emit(state)     # must not device-mix
        head, _ = spectral_latent_torch.unpack(payload)
        self.assertEqual(int(head["members"]), 2)


if __name__ == "__main__":
    unittest.main()
