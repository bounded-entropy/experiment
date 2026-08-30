"""plora (rlstack.policy.adapters.plora*): a probabilistic low-rank delta, and
the seams it grew.

The claims under test, in the order the adapter type's own story runs:

  THE DECLARATION     the sugar builds it, the registry holds it, the flow
                      graph names both its records and both its provides —
                      including one nothing requires, which still reaches the
                      ledger and the run's own dictionary.
  THE FROZEN HALF     top_svd_factors reconstructs and is sign-canonical, and
                      the artifact roundtrips to exactly what the trainer
                      recomputes (which is what binds the two lowerings).
  THE IDENTITY        version 0 is the base for EVERY latent, and KL(q||p)
                      starts at exactly 0.
  THE MEMBRANE        the noise drawn at rollout seals into turn_extras,
                      flattens per turn, packs per document, and arrives at the
                      replay forward as that row's fact.
  THE MATH            one slot and one latent is the plain expression; rows on
                      different latents match a per-row loop.
  THE PAYLOAD         emit/load roundtrip, counter included, so a resumed run
                      re-emits the ensemble it crashed holding.

torch ships in the deploy image, not in the client environment, so the
numerical half SKIPS locally and RUNS in the image; everything declarative,
every seam, and the whole emission path are stdlib and always run.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest

from common import arith_spec, arith_store, char_tokenize, make_turn, sealed
from rlstack import (
    AdapterSpec, AlgoSpec, FakeEngine, FakeLearner, OptimSpec, PolicySpec,
    Rollout, Message, Role, Task, Schedule, fake_qwen_schema, flatten, pack,
    plora, run_experiment, validate,
)
from rlstack.policy.adapters.plora import (
    EPS_RECORD, KL_PROVIDED, MEMBER_RECORD, SIGMA_PROVIDED,
)
from rlstack.policy.adapters.rollout import Levers, Request, ServingBuild
from rlstack.registry import ADAPTER_TYPES
from rlstack.runner.daemons.trainer import _train_summary
from rlstack.runner.interfaces import TrainStats
from rlstack.spec.flow import flow_graph

try:
    import torch
except ImportError:                                  # the client environment
    torch = None

if torch is not None:
    from rlstack.policy.adapters import plora_factors, plora_torch
    from rlstack.policy.adapters.replay import ReplayRows, row_plan
    from rlstack.training.losses.grpo_latent_kl import BETA, grpo_latent_kl
    from rlstack.training.losses.grpo_latent_kl_gated import grpo_latent_kl_gated
    from rlstack.training.losses import PolicyOutputs

needs_torch = unittest.skipUnless(
    torch is not None, "torch is trainer metal: this suite runs in the image")

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
FACTORS = "cas://" + "0" * 64
SITE = "layers.0-3.self_attn.*"


def plora_spec(train_uri: str, heldout_uri: str | None = None, **init):
    """An arith spec whose bank is one plora entry and whose loss prices its
    latent — the smallest thing that exercises the whole declaration."""
    settings = dict(k=4, latent=8, members=2, factors=FACTORS)
    settings.update(init)
    entry = plora(SITE, **settings)
    return arith_spec(
        train_uri, heldout_uri,
        policy=PolicySpec(base="Qwen/Qwen3-0.6B", bank={"pi": entry}),
        algo=AlgoSpec(loss="grpo_latent_kl",
                      post=("verifier", "grpo_advantage"),
                      optim=OptimSpec("adamw", lr=1e-5),
                      schedule=Schedule(microbatch_tokens=64)))


def gated_plora_spec(train_uri: str,
                     post=("verifier", "group_accuracy", "grpo_advantage")):
    """plora_spec with the gate: the loss that earns the prior's pull, and the
    pipeline that writes the column it reads."""
    entry = plora(SITE, k=4, latent=8, members=2, factors=FACTORS)
    return arith_spec(
        train_uri,
        policy=PolicySpec(base="Qwen/Qwen3-0.6B", bank={"pi": entry}),
        algo=AlgoSpec(loss="grpo_latent_kl_gated", post=post,
                      optim=OptimSpec("adamw", lr=1e-5),
                      schedule=Schedule(microbatch_tokens=64)))


# ---------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------

class DeclarationTest(unittest.TestCase):
    def test_the_sugar_is_a_plain_adapter_spec(self) -> None:
        entry = plora("layers.*.self_attn.*", k=16, factors=FACTORS)
        self.assertIsInstance(entry, AdapterSpec)
        self.assertEqual(entry.adapter_type, "plora")
        self.assertEqual(entry.init, {
            "k": 16, "latent": 32, "members": 8, "prior_std": 0.05,
            "hidden": 128, "factors": FACTORS})

    def test_the_registry_holds_the_declaration_half(self) -> None:
        """Everything Phase 0 needs to reason about plora is reachable without
        importing a line of torch."""
        instance = ADAPTER_TYPES.get("plora").instance
        self.assertEqual(instance.serving, "punica")
        self.assertEqual(instance.records, (EPS_RECORD, MEMBER_RECORD))
        self.assertEqual(instance.provides, {KL_PROVIDED, SIGMA_PROVIDED})

    def test_it_lives_only_at_a_weighted_site(self) -> None:
        """The frozen half IS the matrix's own singular directions, so a
        boundary has nothing to factor."""
        instance = ADAPTER_TYPES.get("plora").instance
        weighted = [m for m in SCHEMA.sites if m.has_weight][0]
        boundary = [m for m in SCHEMA.sites if not m.has_weight][0]
        self.assertTrue(instance.site_ok(weighted))
        self.assertFalse(instance.site_ok(boundary))

    def test_the_default_provide_and_param_groups_are_the_old_behavior(self) -> None:
        """Every adapter type that predates these verbs keeps working: no
        provided tensors, and one optimizer group holding the whole entry."""
        class Params:
            def parameters(self):
                return ["p"]

        lora_type = ADAPTER_TYPES.get("lora").instance
        params = Params()
        self.assertEqual(lora_type.provide(params), {})
        self.assertEqual(lora_type.param_groups(params), {"": ["p"]})


# ---------------------------------------------------------------------------
# the rollout seam
# ---------------------------------------------------------------------------

class RolloutSeamTest(unittest.TestCase):
    def test_a_request_carries_its_seed_and_score_traffic_carries_none(self) -> None:
        self.assertIsNone(Request(token_ids=(1, 2)).seed)
        self.assertEqual(Request(token_ids=(1, 2), seed=9).seed, 9)

    def test_merged_levers_union_their_recorded_facts(self) -> None:
        """Two adapter types' facts join, exactly as their keywords do — and
        unambiguously, because claims already refused a collision."""
        merged = (Levers(kwargs={"a": 1}, turn_extras={"x": 1})
                  .merged_with(Levers(kwargs={"b": 2}, turn_extras={"y": 2})))
        self.assertEqual(merged.kwargs, {"a": 1, "b": 2})
        self.assertEqual(merged.turn_extras, {"x": 1, "y": 2})

    def test_a_build_states_its_ensemble_budget_and_its_cas_reader(self) -> None:
        build = ServingBuild(base="b", config=None, workdir=None,
                             max_bundles=2, max_rank=8)
        self.assertEqual(build.max_members, 0)
        self.assertIsNone(build.cas)


# ---------------------------------------------------------------------------
# the flow graph, and the provide nothing requires
# ---------------------------------------------------------------------------

class FlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = flow_graph(plora_spec("cas://x/t.jsonl"))

    def nodes(self, name: str) -> list:
        return [n for n in self.graph.nodes if n.name == name]

    def test_the_records_are_wave_records(self) -> None:
        for record in (EPS_RECORD, MEMBER_RECORD):
            node, = self.nodes(record)
            self.assertEqual((node.kind, node.phase), ("record", "wave"))
            self.assertEqual(node.producer, "adapter:plora")

    def test_a_required_provide_has_a_forward_node_and_a_stat_twin(self) -> None:
        forward, stat = sorted(self.nodes(KL_PROVIDED),
                               key=lambda n: n.phase)   # forward, train
        self.assertEqual((forward.kind, forward.phase), ("provided", "forward"))
        self.assertTrue(forward.feeds_loss)
        self.assertEqual(forward.consumers, ("loss:grpo_latent_kl",))
        self.assertFalse(forward.stored)
        self.assertEqual((stat.kind, stat.phase), ("stat", "train"))
        self.assertEqual(stat.granularity, "update")
        self.assertTrue(stat.stored)

    def test_a_provide_nothing_requires_is_still_emitted(self) -> None:
        """THE observability ruling, pinned: plora_sigma_mean feeds no loss and
        has no consumer, and it STILL gets both nodes — so it reaches the
        ledger per update and the run's dictionary describes it. An adapter
        type declares what a reader should watch, not only what a loss eats."""
        forward, stat = sorted(self.nodes(SIGMA_PROVIDED),
                               key=lambda n: n.phase)
        for node in (forward, stat):
            self.assertFalse(node.feeds_loss)
            self.assertEqual(node.consumers, ())
        self.assertEqual(forward.phase, "forward")
        self.assertEqual((stat.phase, stat.kind, stat.granularity),
                         ("train", "stat", "update"))

    def test_the_forward_node_is_what_the_loss_resolves_against(self) -> None:
        """A stat twin is a report, not a column a loss may require: only the
        forward node is available to the loss, so the twin cannot accidentally
        satisfy a requires."""
        self.assertIn(KL_PROVIDED, self.graph.available_to_loss())
        self.assertEqual(self.graph.unsatisfied_requires(), ())


# ---------------------------------------------------------------------------
# the submit gate
# ---------------------------------------------------------------------------

class ValidateTest(unittest.TestCase):
    def codes(self, **init) -> list[str]:
        spec = plora_spec("cas://x/t.jsonl", **init)
        return [issue.code for issue in validate(spec, SCHEMA)]

    def test_a_complete_entry_passes(self) -> None:
        self.assertEqual(self.codes(), [])

    def test_the_factors_address_is_not_optional(self) -> None:
        """Without it nothing servable exists, and identity would cover two
        different factorizations under one run_id."""
        self.assertIn("plora-factors-missing", self.codes(factors=None))
        self.assertIn("plora-factors-missing",
                      self.codes(factors="/tmp/factors.st"))

    def test_counts_must_be_counts(self) -> None:
        for field in ("k", "latent", "members", "hidden"):
            with self.subTest(field=field):
                self.assertIn("plora-bad-shape", self.codes(**{field: 0}))

    def test_the_prior_must_have_a_scale(self) -> None:
        self.assertIn("plora-bad-shape", self.codes(prior_std=0.0))

    def test_the_gated_loss_validates_with_its_column(self) -> None:
        issues = validate(gated_plora_spec("cas://x/t.jsonl"), SCHEMA)
        self.assertEqual(issues, [])

    def test_the_gate_without_its_producer_is_refused(self) -> None:
        """"accuracy" is a data column like any other (I9): drop
        group_accuracy from the pipeline and Phase 0 refuses the plan,
        never a KeyError mid-update."""
        spec = gated_plora_spec("cas://x/t.jsonl",
                                post=("verifier", "grpo_advantage"))
        codes = {issue.code for issue in validate(spec, SCHEMA)}
        self.assertIn("unsatisfied-requires", codes)


# ---------------------------------------------------------------------------
# the membrane: a per-request draw, from the engine to the replay row
# ---------------------------------------------------------------------------

class TurnExtrasTest(unittest.TestCase):
    def a_trajectory(self, *draws):
        """One document whose turns carry the given per-request facts."""
        turns = [make_turn(f"t{i}", char_tokenize(f"t{i}"),
                           turn_extras=dict(draw)) for i, draw in enumerate(draws)]
        rollout = Rollout(task=Task("t", "What is 2+2?", {}),
                          messages=[Message(Role.USER, "q"),
                                    *(t.message for t in turns)],
                          turns=list(turns))
        return rollout.seal()

    def test_flatten_keeps_one_mapping_per_turn(self) -> None:
        """Per-request facts are NOT token-aligned: they ride beside the tokens,
        one per turn, and injected messages contribute none."""
        flat = flatten(self.a_trajectory({"e": [1.0]}, {"e": [2.0]}),
                       char_tokenize)
        self.assertEqual(flat.turn_extras, ({"e": [1.0]}, {"e": [2.0]}))
        self.assertEqual(len(flat.token_ids), flat.doc_len)

    def test_pack_addresses_them_by_document(self) -> None:
        docs = [(flatten(self.a_trajectory({"e": [float(i)]}), char_tokenize), {})
                for i in range(3)]
        batch, = pack(docs, microbatch_tokens=1000)
        self.assertEqual(len(batch.doc_turn_extras), len(batch.doc_starts))
        self.assertEqual([turns[0]["e"] for turns in batch.doc_turn_extras],
                         [[0.0], [1.0], [2.0]])

    def test_pack_stamps_the_updates_microbatch_count_on_every_batch(self) -> None:
        """A loss adding a per-update term has to know how many times it is
        about to be asked, and the split is only known once it is done."""
        docs = [(flatten(sealed("t", "abcd"), char_tokenize), {})
                for _ in range(4)]
        budget = 2 * docs[0][0].doc_len                 # exactly two per batch
        batches = pack(docs, microbatch_tokens=budget)
        self.assertEqual(len(batches), 2)
        self.assertTrue(all(b.microbatches_in_update == 2 for b in batches))
        one, = pack(docs[:1], microbatch_tokens=1000)
        self.assertEqual(one.microbatches_in_update, 1)

    def test_a_partial_per_document_tuple_is_refused(self) -> None:
        from rlstack.data.trajectory import DataError
        from rlstack import TokenBatch

        with self.assertRaises(DataError):
            TokenBatch(token_ids=(1, 2), loss_mask=(1, 1),
                       behavior_logprobs=(0.0, 0.0), segment_ids=(0, 1),
                       doc_starts=(0, 1), doc_turn_extras=({"e": 1},))

    def test_the_fake_engine_records_per_request_facts(self) -> None:
        """The FinishEvent mirror of record_draws: a stdlib run that exercises
        the whole per-request channel with no GPU."""
        import asyncio

        from rlstack import EnginePoolClient
        from rlstack.policy.compile import Bundle

        engine = FakeEngine(record_latent=True)
        bundle = Bundle.pin("bundle:x", {"pi": 0})
        engine.add_bundle(bundle)
        client = EnginePoolClient({"main": (engine, bundle)},
                                  arith_spec("cas://x").gen.sampling, 7)
        turn = asyncio.run(client.sample([Message(Role.USER, "What is 2+2?")]))
        self.assertEqual(sorted(turn.turn_extras), ["latent_draw", "member_draw"])
        self.assertEqual(len(turn.turn_extras["latent_draw"]), 4)

    def test_switching_the_recording_on_shifts_no_other_byte(self) -> None:
        """The draw comes off the SEED TREE, not the token stream's generator,
        so a run that starts recording produces the same tokens it did."""
        import asyncio

        from rlstack import EnginePoolClient
        from rlstack.policy.compile import Bundle

        def turn_from(engine):
            bundle = Bundle.pin("bundle:x", {"pi": 0})
            engine.add_bundle(bundle)
            client = EnginePoolClient({"main": (engine, bundle)},
                                      arith_spec("cas://x").gen.sampling, 7)
            return asyncio.run(client.sample([Message(Role.USER, "What is 2+2?")]))

        plain = turn_from(FakeEngine())
        recording = turn_from(FakeEngine(record_latent=True))
        self.assertEqual(plain.token_ids, recording.token_ids)
        self.assertEqual(plain.behavior_logprobs, recording.behavior_logprobs)


# ---------------------------------------------------------------------------
# the emission plane: a declared provide reaches the ledger
# ---------------------------------------------------------------------------

class EmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)

    def test_the_summary_means_each_provided_name_across_microbatches(self) -> None:
        stats = [TrainStats(1.0, 1.0, 0.0, 0.0, 4, provided={"plora_kl": 2.0}),
                 TrainStats(3.0, 1.0, 0.0, 0.0, 4, provided={"plora_kl": 4.0})]
        summary = _train_summary(stats)
        self.assertEqual(summary["plora_kl"], 3.0)
        self.assertEqual(summary["loss"], 2.0)
        self.assertEqual(summary["microbatches"], 2)

    def test_the_standing_rails_own_their_names(self) -> None:
        """A provided name colliding with a rail never shadows it: the rails
        are written over the folded-in provides."""
        stats = [TrainStats(1.0, 1.0, 0.0, 0.0, 4, provided={"loss": 99.0})]
        self.assertEqual(_train_summary(stats)["loss"], 1.0)

    def test_a_run_lands_both_provides_in_its_ledger(self) -> None:
        """End to end on fakes: a bank declaring two provided tensors puts both
        in every update's train block — including the one no loss requires."""
        report = run_experiment(plora_spec(self.train), SCHEMA, self.store,
                                FakeEngine(), FakeLearner())
        run = self.store.open_run(report.run_id)
        for entry in run.read_ledger():
            self.assertIn(KL_PROVIDED, entry["train"])
            self.assertIn(SIGMA_PROVIDED, entry["train"])

    def test_the_run_describes_both_provides_to_a_reader(self) -> None:
        """dictionary.json is how a UI renders a run without a registry (I11),
        so the stat twin has to be in it."""
        report = run_experiment(plora_spec(self.train), SCHEMA, self.store,
                                FakeEngine(), FakeLearner())
        dictionary = json.loads(self.store.path_of(
            f"runs/{report.run_id}/dictionary.json").read_text())
        twins = [c for c in dictionary["columns"]
                 if c["name"] == SIGMA_PROVIDED and c["phase"] == "train"]
        self.assertEqual(len(twins), 1)
        self.assertEqual(twins[0]["granularity"], "update")
        self.assertFalse(twins[0]["feeds_loss"])
        self.assertEqual(twins[0]["consumers"], [])

    def test_a_bank_that_provides_nothing_adds_nothing(self) -> None:
        """The old ledger shape is untouched where no adapter type declares a
        provide — the emission is declaration-driven, not unconditional."""
        report = run_experiment(arith_spec(self.train), SCHEMA, self.store,
                                FakeEngine(), FakeLearner())
        run = self.store.open_run(report.run_id)
        entry = run.read_ledger()[0]
        self.assertEqual(sorted(entry["train"]), sorted(
            ["microbatches", "tokens", "loss", "mean_ratio", "logprob_gap",
             "grad_norm"]))


# ---------------------------------------------------------------------------
# the frozen half
# ---------------------------------------------------------------------------

@needs_torch
class FactorsTest(unittest.TestCase):
    def a_weight(self, out: int = 6, inn: int = 8, seed: int = 3):
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(out, inn, generator=generator)

    def test_the_factors_reconstruct_the_top_k_subspace(self) -> None:
        """U (Sigma V^T) is the rank-k truncation, so it must match torch's own
        SVD truncation to numerical precision."""
        weight = self.a_weight()
        u, a = plora_factors.top_svd_factors(weight, 3)
        left, singular, right = torch.linalg.svd(weight.float(),
                                                 full_matrices=False)
        expected = left[:, :3] @ torch.diag(singular[:3]) @ right[:3]
        self.assertTrue(torch.allclose(u @ a, expected, atol=1e-4))

    def test_full_rank_factors_reconstruct_the_whole_matrix(self) -> None:
        weight = self.a_weight()
        u, a = plora_factors.top_svd_factors(weight, 6)
        self.assertTrue(torch.allclose(u @ a, weight.float(), atol=1e-4))

    def test_the_signs_are_canonical_and_therefore_reproducible(self) -> None:
        """Each right vector's largest-magnitude entry is positive — the one
        freedom an SVD leaves, pinned, so both lowerings share a coordinate
        system."""
        weight = self.a_weight()
        u, a = plora_factors.top_svd_factors(weight, 3)
        pivots = a.gather(1, a.abs().argmax(dim=1, keepdim=True)).squeeze(1)
        self.assertTrue(bool((pivots > 0).all()))
        again = plora_factors.top_svd_factors(weight, 3)
        self.assertTrue(torch.equal(u, again[0]))
        self.assertTrue(torch.equal(a, again[1]))

    def test_a_rank_above_the_matrix_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            plora_factors.top_svd_factors(self.a_weight(), 7)

    def test_the_artifact_roundtrips_to_what_the_trainer_recomputes(self) -> None:
        """THE BINDING between the two lowerings: the engine reads these bytes,
        the trainer recomputes the same function from the base it holds, and
        this is the test that says they agree."""
        from rlstack.policy.siteschema import SiteMeta

        weights = {"block.proj": self.a_weight(6, 8),
                   "block.up": self.a_weight(10, 4, seed=5)}
        metas = tuple(SiteMeta(name=path, path=path, has_weight=True,
                               shape=(w.shape[1], w.shape[0]), is_boundary=False)
                      for path, w in weights.items())
        payload = plora_factors.build_factors("Qwen/Qwen3-0.6B", metas, 3,
                                              weights.__getitem__)
        read = plora_factors.read_factors(payload, base="Qwen/Qwen3-0.6B", k=3)
        for path, weight in weights.items():
            direct = plora_factors.top_svd_factors(weight, 3)
            for stored, computed in zip(read[path], direct):
                self.assertTrue(torch.allclose(stored.float(), computed,
                                               atol=1e-2))

    def test_the_artifact_is_a_pure_function_of_its_inputs(self) -> None:
        from rlstack.policy.siteschema import SiteMeta

        weight = self.a_weight()
        meta = (SiteMeta("block.proj", "block.proj", True, (8, 6), False),)
        first = plora_factors.build_factors("b", meta, 3, lambda _: weight)
        second = plora_factors.build_factors("b", meta, 3, lambda _: weight)
        self.assertEqual(first, second)

    def test_the_wrong_artifact_is_refused_rather_than_interpreted(self) -> None:
        from rlstack.policy.siteschema import SiteMeta

        meta = (SiteMeta("block.proj", "block.proj", True, (8, 6), False),)
        payload = plora_factors.build_factors("base-a", meta, 3,
                                              lambda _: self.a_weight())
        with self.assertRaises(ValueError):
            plora_factors.read_factors(payload, base="base-b")
        with self.assertRaises(ValueError):
            plora_factors.read_factors(payload, k=4)
        head, tensors = plora_factors.unpack_artifact(payload)
        forged = plora_factors.pack_artifact({**head, "algo": "other-v9"},
                                             tensors)
        with self.assertRaises(ValueError):
            plora_factors.read_factors(forged)


# ---------------------------------------------------------------------------
# the trainable half
# ---------------------------------------------------------------------------

PATH = "block.proj"


def a_model():
    """A one-site stand-in for the base: `block.proj` is the matched Linear."""
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.proj = torch.nn.Linear(8, 6, bias=False)
    return model


def a_site_meta():
    from rlstack.policy.siteschema import SiteMeta
    return (SiteMeta(name=PATH, path=PATH, has_weight=True, shape=(8, 6),
                     is_boundary=False),)


def a_state(seed: int = 1, k: int = 3, latent: int = 4, hidden: int = 8,
            members: int = 2, prior_std: float = 0.05):
    return plora_torch.build(a_site_meta(), {
        "k": k, "latent": latent, "hidden": hidden, "members": members,
        "prior_std": prior_std, "factors": FACTORS, "seed": seed})


def a_live_state(seed: int = 1, **kwargs):
    """A state whose latent path is LIVE: build() zeroes the heads (so every
    core is zero and version 0 is the base) and puts q exactly on p (so the KL
    is zero) — both gradients would be structurally zero, which is the identity
    element, not a working policy."""
    state = a_state(seed, **kwargs)
    generator = torch.Generator().manual_seed(seed)
    state.heads[PATH].data = torch.randn(
        *state.heads[PATH].shape, generator=generator) * 0.1
    state.mu.data = torch.randn(state.latent, generator=generator) * 0.1
    state.log_std.data = state.log_std.data + 0.25
    return state


def routed(model, rows):
    return row_plan(model).route(rows)


def one_slot(state, noise, device="cpu"):
    """The row plan a single-tenant microbatch gets: one slot, and one recorded
    latent per row."""
    return ReplayRows(
        slots=({PATH: state},),
        index=torch.zeros(len(noise), dtype=torch.long),
        facts=tuple(({EPS_RECORD: list(eps)},) for eps in noise))


@needs_torch
class IdentityTest(unittest.TestCase):
    def test_version_zero_is_the_base_for_every_latent(self) -> None:
        """ControlNet-style zero heads: every core is zero, so the delta is
        EXACTLY zero whatever noise a request drew."""
        model, state = a_model(), a_state()
        plora_torch.install(model, state)
        x = torch.randn(2, 3, 8)
        generator = torch.Generator().manual_seed(4)
        noise = torch.randn(2, state.latent, generator=generator)
        with routed(model, one_slot(state, noise)):
            self.assertTrue(torch.equal(model.block.proj(x),
                                        model.block.proj.inner(x)))

    def test_the_kl_starts_at_exactly_zero(self) -> None:
        state = a_state()
        self.assertEqual(
            float(plora_torch.analytic_kl(state.mu, state.log_std,
                                          state.prior_std)), 0.0)

    def test_the_kl_matches_the_closed_form(self) -> None:
        mu = torch.tensor([0.5, -0.25])
        log_std = torch.tensor([math.log(0.2), math.log(0.1)])
        prior = 0.05
        expected = sum(
            math.log(prior) - float(ls)
            + (math.exp(2 * float(ls)) + float(m) ** 2) / (2 * prior ** 2) - 0.5
            for m, ls in zip(mu, log_std))
        self.assertAlmostEqual(
            float(plora_torch.analytic_kl(mu, log_std, prior)), expected,
            places=5)

    def test_provide_returns_the_declared_tensors(self) -> None:
        state = a_live_state()
        provided = ADAPTER_TYPES.get("plora").instance.provide(state)
        self.assertEqual(sorted(provided), sorted([KL_PROVIDED, SIGMA_PROVIDED]))
        self.assertAlmostEqual(
            float(provided[SIGMA_PROVIDED].detach()),
            float(torch.exp(state.log_std).mean().detach()), places=6)
        self.assertGreater(float(provided[KL_PROVIDED].detach()), 0.0)

    def test_param_groups_split_the_mapper_from_the_posterior(self) -> None:
        state = a_state()
        groups = ADAPTER_TYPES.get("plora").instance.param_groups(state)
        self.assertEqual(sorted(groups), ["mapper", "posterior"])
        self.assertEqual([id(p) for p in groups["posterior"]],
                         [id(state.mu), id(state.log_std)])
        self.assertEqual(
            sum(p.numel() for group in groups.values() for p in group),
            sum(p.numel() for p in state.parameters()))

    def test_the_hypernet_is_bias_free(self) -> None:
        """A bias would let a nonzero core survive z = 0 — a mean delta the KL
        never prices, since the KL sees only mu."""
        state = a_state()
        zero = torch.zeros(state.latent)
        cores = plora_torch.materialize_core(state, zero)
        state.heads[PATH].data.normal_(generator=torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(cores[PATH], torch.zeros_like(cores[PATH])))
        again = plora_torch.materialize_core(state, zero)
        self.assertTrue(torch.equal(again[PATH], torch.zeros_like(again[PATH])))


@needs_torch
class ReplayMathTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = a_model()
        self.state = a_live_state()
        plora_torch.install(self.model, self.state)

    def reference(self, x, state, eps):
        """The delta, written the slow honest way: reparameterize, generate the
        core, apply U C A."""
        z = state.mu + torch.exp(state.log_std) * eps
        core = plora_torch.materialize_core(state, z)[PATH]
        return ((x @ state.a[PATH].T) @ core.T) @ state.u[PATH].T

    def test_one_latent_over_the_whole_batch_matches_the_reference(self) -> None:
        x = torch.randn(2, 3, 8)
        eps = torch.randn(self.state.latent,
                          generator=torch.Generator().manual_seed(6))
        with routed(self.model, one_slot(self.state, eps.expand(2, -1))):
            got = self.model.block.proj(x)
        expected = self.model.block.proj.inner(x) + self.reference(
            x, self.state, eps)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_rows_on_different_latents_match_a_per_row_loop(self) -> None:
        """THE per-row claim: the batched expression equals applying each row's
        own generated core on its own."""
        x = torch.randn(3, 4, 8)
        noise = torch.randn(3, self.state.latent,
                            generator=torch.Generator().manual_seed(7))
        with routed(self.model, one_slot(self.state, noise)):
            got = self.model.block.proj(x)
        expected = torch.stack([
            self.model.block.proj.inner(x[row])
            + self.reference(x[row], self.state, noise[row])
            for row in range(3)])
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_two_tenants_route_to_their_own_hypernets(self) -> None:
        other = a_live_state(seed=42)
        plora_torch.install(self.model, other)
        x = torch.randn(2, 3, 8)
        noise = torch.randn(2, self.state.latent,
                            generator=torch.Generator().manual_seed(8))
        rows = ReplayRows(
            slots=({PATH: self.state}, {PATH: other}),
            index=torch.tensor([0, 1]),
            facts=tuple(({EPS_RECORD: list(eps)},) for eps in noise))
        with routed(self.model, rows):
            got = self.model.block.proj(x)
        expected = torch.stack([
            self.model.block.proj.inner(x[0]) + self.reference(x[0], self.state,
                                                               noise[0]),
            self.model.block.proj.inner(x[1]) + self.reference(x[1], other,
                                                               noise[1])])
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_a_slot_without_a_plora_here_sees_the_base(self) -> None:
        """The transparent case: the site is wrapped because SOME tenant has a
        delta at it, and a tenant whose bank never mentions it must see the
        module it would have seen alone."""
        x = torch.randn(2, 3, 8)
        rows = ReplayRows(slots=({},), index=torch.zeros(2, dtype=torch.long))
        with routed(self.model, rows):
            self.assertTrue(torch.equal(self.model.block.proj(x),
                                        self.model.block.proj.inner(x)))

    def test_a_batch_that_recorded_no_latent_is_refused(self) -> None:
        """The latent was SAMPLED; it is not re-derivable from the parameters,
        so a forward without the recording is a wiring bug."""
        rows = ReplayRows(slots=({PATH: self.state},),
                          index=torch.zeros(2, dtype=torch.long))
        with routed(self.model, rows):
            with self.assertRaises(ValueError) as raised:
                self.model.block.proj(torch.randn(2, 3, 8))
        self.assertIn(EPS_RECORD, str(raised.exception))

    def test_a_row_whose_turns_disagree_is_refused(self) -> None:
        """ONE LATENT PER TRAJECTORY: averaging two would train a policy that
        never sampled anything."""
        rows = ReplayRows(
            slots=({PATH: self.state},), index=torch.zeros(1, dtype=torch.long),
            facts=(({EPS_RECORD: [0.0] * self.state.latent},
                    {EPS_RECORD: [1.0] * self.state.latent}),))
        with routed(self.model, rows):
            with self.assertRaises(ValueError) as raised:
                self.model.block.proj(torch.randn(1, 3, 8))
        self.assertIn("one trajectory is one draw", str(raised.exception))

    def test_a_wrong_width_latent_is_refused(self) -> None:
        rows = ReplayRows(slots=({PATH: self.state},),
                          index=torch.zeros(1, dtype=torch.long),
                          facts=(({EPS_RECORD: [0.0, 1.0]},),))
        with routed(self.model, rows):
            with self.assertRaises(ValueError):
                self.model.block.proj(torch.randn(1, 3, 8))

    def test_uninstall_restores_the_linear(self) -> None:
        plora_torch.uninstall(self.model, self.state)
        self.assertIsInstance(self.model.block.proj, torch.nn.Linear)

    def test_the_frozen_half_is_the_sites_own_singular_directions(self) -> None:
        """install resolves U and A from the base's own weight, under the same
        pinned recipe the artifact was built with."""
        weight = self.model.block.proj.inner.weight
        u, a = plora_factors.top_svd_factors(weight, self.state.k)
        self.assertTrue(torch.equal(u, self.state.u[PATH]))
        self.assertTrue(torch.equal(a, self.state.a[PATH]))


@needs_torch
class PayloadTest(unittest.TestCase):
    def test_emit_load_roundtrips_every_trained_tensor(self) -> None:
        source, target = a_live_state(seed=3), a_state(seed=99)
        payload = plora_torch.emit(source)
        plora_torch.load(target, payload)
        self.assertTrue(torch.equal(source.mu.data, target.mu.data))
        self.assertTrue(torch.equal(source.log_std.data, target.log_std.data))
        self.assertTrue(torch.equal(source.heads[PATH].data,
                                    target.heads[PATH].data))
        for left, right in zip(source.trunk.parameters(),
                               target.trunk.parameters()):
            self.assertTrue(torch.equal(left.data, right.data))

    def test_the_counter_is_used_then_advanced_so_a_resume_re_emits(self) -> None:
        """THE resume rule. emit records the counter it drew with and only then
        advances; load restores it — so re-emitting after a crash reproduces the
        bytes the crashed process had, and the next emit continues the
        sequence.

        The resumed state is rebuilt at the SAME seed because a seed is an init
        fact, not a payload fact: `_init_seed(master, entry)` is a pure function
        of the spec (I3), so Phase 1 hands the same one on every attach. Putting
        it in the payload would let a warm start inherit its parent's noise
        stream, which is precisely the confusion the seed tree exists to
        prevent."""
        state = a_live_state()
        first = plora_torch.emit(state)                 # version 0
        self.assertEqual(state.version, 1)
        second = plora_torch.emit(state)                # version 1
        self.assertEqual(state.version, 2)
        self.assertNotEqual(first, second)

        resumed = a_state()                             # same seed, fresh state
        plora_torch.load(resumed, second)               # restores version 1
        self.assertEqual(resumed.version, 1)
        self.assertEqual(plora_torch.emit(resumed), second)
        self.assertEqual(resumed.version, 2)

    def test_the_seed_never_rides_in_the_payload(self) -> None:
        """Seeds come from the spec, never from bytes on a volume."""
        head, _ = plora_factors.unpack_artifact(plora_torch.emit(a_state()))
        self.assertNotIn("seed", head)

    def test_the_ensemble_is_a_pure_function_of_seed_and_version(self) -> None:
        noise = plora_torch.noise_for(11, 3, 4, 8)
        self.assertEqual(tuple(noise.shape), (4, 8))
        self.assertTrue(torch.equal(noise, plora_torch.noise_for(11, 3, 4, 8)))
        self.assertFalse(torch.equal(noise, plora_torch.noise_for(11, 4, 4, 8)))
        self.assertFalse(torch.equal(noise[0], noise[1]))

    def test_the_payload_carries_the_address_and_not_the_factors(self) -> None:
        """Kilobytes, not megabytes: the frozen half is identical at every
        version, so the payload names it instead of shipping it."""
        state = a_live_state()
        head, tensors = plora_factors.unpack_artifact(plora_torch.emit(state))
        self.assertEqual(head["factors"], FACTORS)
        self.assertNotIn(f"{PATH}.U", tensors)
        self.assertEqual(sorted(k for k in tensors if k.startswith("posterior")),
                         ["posterior.log_std", "posterior.mu"])

    def test_a_serving_side_rebuilds_the_same_arithmetic(self) -> None:
        """The engine materializes members with the SAME functions the trainer
        differentiates through, so B = U C is the replay expression folded."""
        state = a_live_state()
        plora_torch.install(a_model(), state)
        payload = plora_torch.emit(state)
        factors = {PATH: (state.u[PATH], state.a[PATH])}
        resident = plora_torch.resident(payload, factors)
        z = torch.randn(state.latent,
                        generator=torch.Generator().manual_seed(12))
        mine = plora_torch.materialize_core(state, z)[PATH]
        theirs = plora_torch.materialize_core(resident, z)[PATH]
        self.assertTrue(torch.allclose(mine, theirs, atol=1e-6))
        b = plora_torch.materialize_b(resident, {PATH: theirs})[PATH]
        self.assertTrue(torch.allclose(b, state.u[PATH] @ theirs, atol=1e-5))

    def test_peft_scaling_is_exactly_one(self) -> None:
        """alpha == r == k, so the engine computes x A^T B^T verbatim."""
        config = json.loads(plora_torch.peft_config("base", 7, ["q_proj"]))
        self.assertEqual(config["r"], 7)
        self.assertEqual(config["lora_alpha"], 7)


# ---------------------------------------------------------------------------
# the loss
# ---------------------------------------------------------------------------

@needs_torch
class LatentKlLossTest(unittest.TestCase):
    def a_batch(self, microbatches: int = 1):
        from rlstack import TokenBatch
        return TokenBatch(
            token_ids=(1, 2, 3, 4), loss_mask=(1, 1, 1, 1),
            behavior_logprobs=(-0.5,) * 4, segment_ids=(0, 0, 0, 0),
            doc_starts=(0,), postdata={"advantage": (1.0, 1.0, -1.0, -1.0)},
            microbatches_in_update=microbatches)

    def outputs(self, state, logprobs):
        return PolicyOutputs(
            logprobs=logprobs,
            provided=ADAPTER_TYPES.get("plora").instance.provide(state))

    def test_the_kl_is_counted_once_per_update(self) -> None:
        """A wave is one gradient update, and the KL is a function of the
        parameters alone — adding it whole to each microbatch would make the
        effective beta depend on microbatch_tokens."""
        state = a_live_state()
        logprobs = torch.full((4,), -0.5)
        whole = grpo_latent_kl(self.outputs(state, logprobs), self.a_batch(1))
        split = grpo_latent_kl(self.outputs(state, logprobs), self.a_batch(4))
        kl = float(plora_torch.analytic_kl(state.mu, state.log_std,
                                           state.prior_std))
        self.assertAlmostEqual(float(whole.loss) - float(split.loss),
                               BETA * kl * (1 - 1 / 4), places=6)

    def test_the_surrogate_is_grpos_verbatim(self) -> None:
        from rlstack.training.losses.grpo import grpo

        state = a_state()                       # KL is exactly 0 at init
        logprobs = torch.full((4,), -0.5)
        plain = grpo(PolicyOutputs(logprobs=logprobs), self.a_batch())
        both = grpo_latent_kl(self.outputs(state, logprobs), self.a_batch())
        self.assertAlmostEqual(float(plain.loss), float(both.loss), places=7)
        self.assertEqual(plain.mean_ratio, both.mean_ratio)
        self.assertEqual(plain.logprob_gap, both.logprob_gap)

    def test_the_gradient_reaches_the_posterior_and_the_mapper(self) -> None:
        """The whole point: a draw taken before these parameters had their
        present values still puts them on the gradient path — the surrogate
        reaches the mapper through the delta, the KL reaches the posterior
        directly, and the reparameterization reaches it through the delta too.
        """
        model, state = a_model(), a_live_state()
        plora_torch.install(model, state)
        x = torch.randn(2, 3, 8)
        noise = torch.randn(2, state.latent,
                            generator=torch.Generator().manual_seed(13))
        with routed(model, one_slot(state, noise)):
            out = model.block.proj(x)
            logprobs = -out.reshape(-1)[:4].float()
            result = grpo_latent_kl(self.outputs(state, logprobs),
                                    self.a_batch())
            result.loss.backward()
        for name, parameter in (("mu", state.mu), ("log_std", state.log_std),
                                ("head", state.heads[PATH]),
                                ("trunk", state.trunk.enter)):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)


@needs_torch
class GatedLatentKlLossTest(unittest.TestCase):
    """accuracy first, then the KL: the gate is the solved share of the batch."""

    def a_batch(self, accuracy, microbatches: int = 1):
        from rlstack import TokenBatch
        return TokenBatch(
            token_ids=(1, 2, 3, 4), loss_mask=(1, 1, 1, 1),
            behavior_logprobs=(-0.5,) * 4, segment_ids=(0, 0, 0, 0),
            doc_starts=(0,),
            postdata={"advantage": (1.0, 1.0, -1.0, -1.0),
                      "accuracy": tuple(accuracy)},
            microbatches_in_update=microbatches)

    def outputs(self, state, logprobs):
        return PolicyOutputs(
            logprobs=logprobs,
            provided=ADAPTER_TYPES.get("plora").instance.provide(state))

    def kl(self, state) -> float:
        return float(plora_torch.analytic_kl(state.mu, state.log_std,
                                             state.prior_std))

    def test_an_unsolved_batch_pays_no_kl(self) -> None:
        """The whole point: while nothing is solved the prior is SILENT, even
        against a live posterior whose KL is far from zero."""
        from rlstack.training.losses.grpo import grpo

        state = a_live_state()
        logprobs = torch.full((4,), -0.5)
        batch = self.a_batch((0.5,) * 4)
        plain = grpo(PolicyOutputs(logprobs=logprobs), batch)
        gated = grpo_latent_kl_gated(self.outputs(state, logprobs), batch)
        self.assertGreater(self.kl(state), 0.0)
        self.assertAlmostEqual(float(plain.loss), float(gated.loss), places=7)

    def test_a_solved_batch_pays_the_whole_beta(self) -> None:
        """Fully solved coincides with grpo_latent_kl: the gate is 1, so the
        two objectives share one KL price and can never drift apart."""
        state = a_live_state()
        logprobs = torch.full((4,), -0.5)
        gated = grpo_latent_kl_gated(self.outputs(state, logprobs),
                                     self.a_batch((1.0,) * 4))
        ungated = grpo_latent_kl(self.outputs(state, logprobs),
                                 self.a_batch((1.0,) * 4))
        self.assertAlmostEqual(float(gated.loss), float(ungated.loss),
                               places=7)

    def test_a_half_solved_batch_pays_half(self) -> None:
        """Two of four masked tokens sit in a solved group, so the effective
        beta is BETA / 2 — the KL enters in proportion to the solved share."""
        from rlstack.training.losses.grpo import grpo

        state = a_live_state()
        logprobs = torch.full((4,), -0.5)
        batch = self.a_batch((1.0, 1.0, 0.0, 0.0))
        plain = grpo(PolicyOutputs(logprobs=logprobs), batch)
        gated = grpo_latent_kl_gated(self.outputs(state, logprobs), batch)
        self.assertAlmostEqual(float(gated.loss) - float(plain.loss),
                               0.5 * BETA * self.kl(state), places=6)


@needs_torch
class LearnerSeamTest(unittest.TestCase):
    def test_the_learner_routes_a_batchs_facts_to_its_rows(self) -> None:
        """Adapter-blind: the rows of a padded forward ARE the documents, so
        row r's facts are document r's, copied across without a key being
        read."""
        from rlstack.runner.learners.torch_learner import TorchLearner, _Tenant
        from rlstack import TokenBatch

        batch = TokenBatch(
            token_ids=(1, 2), loss_mask=(1, 1), behavior_logprobs=(0.0, 0.0),
            segment_ids=(0, 0), doc_starts=(0, 1),
            doc_turn_extras=(({EPS_RECORD: [1.0]},), ({EPS_RECORD: [2.0]},)))
        tenant = _Tenant(loss_fn=None, trainable=[], entries=[])
        tenant.slot = {PATH: "state"}
        rows = TorchLearner(device="cpu")._rows_of(tenant, batch, 2)
        self.assertEqual(rows.facts, batch.doc_turn_extras)

    def test_facts_must_be_addressed_by_row(self) -> None:
        with self.assertRaises(ValueError):
            ReplayRows(slots=({},), index=torch.zeros(2, dtype=torch.long),
                       facts=(({"e": 1},),))

    def test_a_provided_tensor_summarizes_to_one_float(self) -> None:
        """A 0-dim tensor is its own value; anything else is its mean — the one
        rule that makes every provide journalable."""
        from rlstack.runner.learners.torch_learner import _summarize

        summary = _summarize({"scalar": torch.tensor(2.5),
                              "vector": torch.tensor([1.0, 3.0])})
        self.assertEqual(summary, {"scalar": 2.5, "vector": 2.0})

    def test_the_dotted_override_wins_over_the_entry_wide_one(self) -> None:
        """`pi` reaches every group, `pi.mapper` reaches one, and the specific
        one wins where both apply — the only reading under which writing both
        is not a contradiction."""
        from rlstack.runner.learners.torch_learner import TorchLearner

        optim = OptimSpec("adamw", lr=1e-4, weight_decay=0.0, overrides={
            "pi": {"lr": 5e-4}, "pi.mapper": {"weight_decay": 1e-2}})
        settings = TorchLearner._group_settings(optim, "pi", "mapper")
        self.assertEqual(settings, {"lr": 5e-4, "weight_decay": 1e-2})
        self.assertEqual(TorchLearner._group_settings(optim, "pi", "posterior"),
                         {"lr": 5e-4, "weight_decay": 0.0})
        self.assertEqual(TorchLearner._group_settings(optim, "other", ""),
                         {"lr": 1e-4, "weight_decay": 0.0})


if __name__ == "__main__":
    unittest.main()
