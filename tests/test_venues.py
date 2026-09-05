"""The venues on the chassis: they import, and their science did not move.

ADR 0007 rewrote three venue files onto `deploy/modal_venue.py` — the
transports, the desk container, the metal container's bring-up and the doors'
take-down ceremony all left them. PROMISE 7 is that nothing semantics-bearing
went with it: every spec value is unchanged, so every run identity is
unchanged, so a run placed before the rewrite resumes after it (I3).

The claim is checked the only way it can be: `tests/venue_spec_rows.json`
holds the canonical row of every spec the three venues build, captured from
the PRE-REWRITE files at commit 90bbc8e, and this file builds the same specs
from the rewritten ones and compares byte for byte. A cas uri in those rows is
itself a hash of encoded plan bytes, so a plan that changed shows up as a
changed uri — the fixture pins the plans too.

The venue files name their app and their images at module scope, so importing
one needs the Modal SDK, which the suite does not have and must not want.
`tests/venue_stub.py` stands it in.

THE FOURTH VENUE. `deploy/gsm_a100.py` came back onto main after the rewrite,
ported from `gsm-campaign-legacy` onto the chassis. Its five rows in the
fixture were captured from THAT branch's file (the pre-chassis venue, with the
elbo arm) over the same seeded store this file builds — ten families of twelve
instances, family 5 the train family — and the same stand-in chat formatter,
because the real one needs the tokenizer. What is pinned is that the port
moved no value; what the stand-in leaves unpinned is only the prompt text
itself, which is content the tokenizer writes, not a spec value.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

from venue_stub import modal_stubbed

REPO = pathlib.Path(__file__).resolve().parent.parent
DEPLOY = REPO / "deploy"
ROWS = json.loads((REPO / "tests" / "venue_spec_rows.json").read_text())

DAPO_SETS = (
    "09499d32b51e5e1b2a644b1c65e01b44aa42ff1a5bfac78ead41f98f89f09c93",
    "82ae4626dbb59a2c50e2b13cbe7250c5f1ddd02dfb81edc7495efb77759d420b",
)
"""The two task-set digests the L4 venues PIN in their specs (#60). A spec
names them by uri and never reads them, except for `spec_for`'s one assertion
that the screened task is in the train set — so the test seeds those two keys
with a stand-in set containing it. What is pinned is the uri, and the uri is
what the fixture compares."""

SCREENED = "dapo-math-17k/a6d38312-86c7-4022-b8d2-adcf19fa0c3a"


def stand_in_chat(text: str) -> str:
    """The gsm venue's prompt formatter, stood in: the real one is Qwen3's
    chat template and needs transformers. The fixture's gsm rows were
    captured under exactly this function."""
    return f"<user>{text}</user>"


def seed_gsm(store, rows_key: str, screen_key: str) -> None:
    """The dataset rows and the screen's verdict the gsm venue reads off the
    store: ten families of twelve instances (enough for two train instances,
    ten near and four far per family) and family 5 chosen to train on. The
    fixture's gsm rows were captured over exactly this seeding."""
    rows = [{"id": f, "instance": i, "question": f"Q {f}-{i}?",
             "answer": f"work...\n#### {10 * f + i}"}
            for f in range(10) for i in range(12)]
    store._write(rows_key, json.dumps(rows, sort_keys=True).encode())
    verdict = {"accuracy": {str(f): 0.05 * (f + 1) for f in range(10)},
               "chosen": list(range(10)), "train_family": 5,
               "band": [0.0, 0.40], "samples": 4, "instances": 6}
    store._write(screen_key, json.dumps(verdict, sort_keys=True).encode())


def calls_named(path: pathlib.Path, names: set[str]) -> list[str]:
    """Every call in this file to a function with one of these names — the
    structural way to ask "does this venue release metal?", which a substring
    search over a file full of prose cannot answer."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sorted({node.func.id for node in ast.walk(tree)
                   if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name)
                   and node.func.id in names})


def load_venue(name: str):
    """One `deploy/*.py` imported under the Modal stand-in, by path — never
    by adding `deploy/` to `sys.path` permanently, because two venues define
    the same module names."""
    path = DEPLOY / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"venue_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VenueFixture(unittest.TestCase):
    """The four venues, imported once, over a store their specs can build in."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._stub = modal_stubbed()
        cls._stub.__enter__()
        sys.path.insert(0, str(DEPLOY))          # so `import modal_venue` works
        try:
            cls.concept_steer = load_venue("concept_steer")
            cls.concept_burgers = load_venue("concept_burgers")
            cls.concept_anger = load_venue("concept_anger")
            cls.steer_l4 = load_venue("steer_l4")
            cls.stress_fleet = load_venue("stress_fleet")
            cls.gsm_a100 = load_venue("gsm_a100")
            cls.modal_venue = load_venue("modal_venue")
            cls.desk_venue = load_venue("desk")
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        if str(DEPLOY) in sys.path:
            sys.path.remove(str(DEPLOY))
        for name in list(sys.modules):
            if name.startswith("venue_") or name == "modal_venue":
                sys.modules.pop(name, None)
        cls._stub.__exit__(None, None, None)

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.concept_tasks = self.seeded_store(tmp.name)
        seed_gsm(self.store, self.gsm_a100.ROWS_KEY, self.gsm_a100.SCREEN_KEY)

    @staticmethod
    def seeded_store(root: str):
        """A LocalStore holding the two pinned dapo sets and enough concept
        prompts for the teacher's plan (64 waves of 32, plus the split's
        headroom)."""
        from rlstack import LocalStore
        from rlstack.data.tasks.base import Task, write_tasks

        store = LocalStore(root)
        dapo = [{"id": SCREENED, "prompt": "1+1?", "meta": {}},
                {"id": "dapo-math-17k/other", "prompt": "2+2?", "meta": {}}]
        rows = "".join(json.dumps(row, sort_keys=True) + "\n" for row in dapo)
        for digest in DAPO_SETS:
            store._write(f"cas/{digest}/blob", rows.encode())
        concept = write_tasks(store, [Task(id=f"p{i:05d}", prompt=f"q{i}",
                                           meta={}) for i in range(2304)])
        return store, concept

    def assertRowUnchanged(self, key: str, spec) -> None:
        """This spec canonicalizes to exactly the row the pre-rewrite venue
        produced — the whole of promise 7, one spec at a time."""
        from rlstack.spec.canonical import canonical_json

        self.assertEqual(canonical_json(spec), ROWS[key],
                         f"{key}: a spec value moved in the rewrite, so this "
                         f"run's identity moved with it (ADR 0007, promise 7)")


class ThreeArmsTest(VenueFixture):
    """The concept venue's three families and two algorithms build and
    validate: a free steer, a norm-scaled steer providing alpha, a rank-4 MLP
    LoRA, each as an SFT arm over the teacher's set and as an on-policy arm
    with the teacher as its own host."""

    def test_every_family_builds_a_valid_sft_arm(self) -> None:
        from rlstack.spec.validate import validate
        from rlstack.policy.siteschema import fake_qwen_schema
        schema = fake_qwen_schema(64, base=self.concept_steer.BASE)
        for adapter in self.concept_steer.ADAPTERS:
            spec = self.concept_steer.student_spec(self.store, "teacher-run-0", 10, adapter)
            self.assertEqual(validate(spec, schema), [], adapter)
        entry = self.concept_steer.bank_entry("mlp_lora", 10)
        self.assertEqual((entry.adapter_type, entry.site, entry.init["r"]),
                         ("lora", "layers.10.mlp.*", 4))

    def test_the_on_policy_arm_declares_its_teacher_host(self) -> None:
        from rlstack.spec.validate import validate
        from rlstack.policy.siteschema import fake_qwen_schema
        schema = fake_qwen_schema(64, base=self.concept_steer.BASE)
        spec = self.concept_steer.opd_spec(self.store, self.concept_tasks, 32, "nsteer")
        self.assertEqual(validate(spec, schema), [])
        self.assertEqual(spec.algo.loss, "opd")
        self.assertEqual(spec.algo.post, ("conditioned_teacher_logprobs",))
        pools = [m.name for h in spec.topology.hosts for m in h.members if hasattr(m, "tp")]
        self.assertEqual(pools, ["main", "teacher"])
        self.assertEqual(len(spec.topology.hosts), 2)         # the teacher is its own unit
        self.assertIsNotNone(spec.plans.rollout)
        self.assertIsNotNone(spec.plans.train)


class SpecsAreUnchangedTest(VenueFixture):
    def test_the_teachers_generation_only_spec(self) -> None:
        self.assertRowUnchanged(
            "concept_steer.teacher_spec",
            self.concept_steer.teacher_spec(self.store, self.concept_tasks))

    def test_every_steer_arm(self) -> None:
        """The three anchors differ in the bank and in nothing else, exactly
        as they did — same plan bytes, same optimizer, same topology."""
        for layer in self.concept_steer.ANCHORS:
            self.assertRowUnchanged(
                f"concept_steer.student_spec@{layer}",
                self.concept_steer.student_spec(self.store, "teacher-run-0",
                                                layer))

    def test_the_two_tenants_of_the_steer_check(self) -> None:
        venue = self.steer_l4
        for name, (bank, overrides) in venue.the_two_banks().items():
            self.assertRowUnchanged(
                f"steer_l4.spec_for/{name}",
                venue.spec_for(self.store, bank, overrides, venue.UPDATES, 11))

    def test_the_three_tenants_of_the_stress_check(self) -> None:
        venue = self.stress_fleet
        for name, (bank, overrides) in venue.the_banks().items():
            self.assertRowUnchanged(
                f"stress_fleet.spec_for/{name}",
                venue.spec_for(self.store, bank, overrides, venue.UPDATES, 41))

    def test_the_five_arms_of_the_gsm_campaign(self) -> None:
        """The port onto the chassis moved no value: each arm — the lora
        baseline, SVF, the gated latent, the ELBO with its learned prior, the
        sdpo loop — canonicalizes to the row the pre-chassis file built."""
        specs, _ = self.gsm_a100.campaign_specs(self.store, stand_in_chat)
        self.assertEqual(sorted(specs), ["gsm-elbo", "gsm-grpo", "gsm-sdpo",
                                         "gsm-slatent", "gsm-spectral"])
        for name, spec in specs.items():
            self.assertRowUnchanged(f"gsm_a100.campaign_specs/{name}", spec)

    def test_the_fixture_covers_every_spec_the_venues_build(self) -> None:
        """A row nobody compares is a promise nobody keeps."""
        self.assertEqual(len(ROWS), 19)


class ChassisTest(VenueFixture):
    """ADR 0007's structural promises, read off the files themselves."""

    def test_no_venue_defines_a_transport(self) -> None:
        """Promise 2: exactly ONE Modal transport exists, and it is in
        `rlstack/runner/transports/`. `git grep "class .*Transport" -- deploy`
        is empty, said as a test."""
        for path in sorted(DEPLOY.glob("*.py")):
            self.assertNotIn(
                "Transport:", path.read_text(),
                f"{path.name} defines a Transport — the wire substrates live "
                f"in rlstack/runner/transports/ (ADR 0007, Q1)")

    def test_only_deploy_desk_defines_a_desk_container(self) -> None:
        """Promise 4: one desk app, one journal. A venue that declared its own
        `Desk` class had its own fleet and could not share metal (Q2)."""
        desks = [path.name for path in sorted(DEPLOY.glob("*.py"))
                 if "class Desk" in path.read_text()]
        self.assertEqual(desks, ["desk.py"])

    def test_no_venue_keeps_its_own_fleet_journal(self) -> None:
        """The per-venue `FLEET_LOG`s and their Store subclasses are gone: one
        desk writes one journal (I10)."""
        for path in sorted(DEPLOY.glob("*.py")):
            self.assertNotIn("FLEET_LOG", path.read_text(), path.name)

    def test_every_venue_addresses_the_same_desk(self) -> None:
        self.assertEqual(self.modal_venue.DESK_ADDRESS,
                         "modal://rlstack-desk/Desk")

    def test_the_addresses_the_chassis_mints_parse(self) -> None:
        """A venue's addresses are the grammar's, so `transport_for` reads
        them — the desk reaches metal in an app it never heard of (Q3)."""
        from rlstack.runner.remote import parse_address

        plane = self.modal_venue.metal_address("rlstack-steer-l4")
        host = self.modal_venue.host_address("rlstack-steer-l4", "steer:0.main")
        self.assertEqual(parse_address(plane).app, "rlstack-steer-l4")
        self.assertEqual(parse_address(plane).host, "")
        self.assertEqual(parse_address(host).host, "steer:0.main")
        self.assertEqual(parse_address(host).cls, "MetalS")

    def test_a_host_address_may_name_one_instance(self) -> None:
        """ADR 0008, Q2: a carved host's address carries the EPOCH of the
        container that carved it — carve names recycle when a metal is
        reborn, and the epoch is what tells the corpse from the newborn."""
        from rlstack.runner.remote import parse_address

        address = self.modal_venue.host_address(
            "rlstack-concept-steer", "concept-a100:0.main.c1", epoch="e7")
        parsed = parse_address(address)
        self.assertEqual((parsed.host, parsed.epoch),
                         ("concept-a100:0.main.c1", "e7"))
        # the PLANE address stays epoch-free: it is the container's stable
        # name, and a re-registration must read as one metal turning over
        self.assertEqual(
            parse_address(self.modal_venue.metal_address(
                "rlstack-concept-steer")).epoch, "")

    def test_the_chassis_renews_at_the_desk_s_cadence(self) -> None:
        """F1: the lease constants are the desk's; the venue carries only the
        fallback it uses until the registration reply tells it otherwise."""
        self.assertEqual(self.modal_venue.HEARTBEAT_S, 20.0)
        source = (DEPLOY / "modal_venue.py").read_text()
        self.assertIn("heartbeat_s", source)
        self.assertIn("watch_residents", source)

    def test_both_doors_of_every_container_are_async(self) -> None:
        """ADR 0008, Q4: a cancelled input of a SYNCHRONOUS method on a
        concurrent container has no clean interruption, so Modal shuts the
        container down — which is how the desk died three times on
        2026-09-04. Both doors are async methods now, on the metal and on the
        desk alike."""
        for path in (DEPLOY / "modal_venue.py", DEPLOY / "desk.py"):
            source = path.read_text()
            self.assertIn("async def door(", source, path.name)
            self.assertIn("async def door_ask(", source, path.name)
            self.assertNotIn("\n    def door_ask(", source, path.name)

    def test_no_venue_helper_wraps_a_desk_call_in_a_timeout(self) -> None:
        """Q4's other half: a client NEVER cancels a fleet input — the
        deadline is the desk's own, server-side. The chassis' helpers poll
        patiently (`wait_for_metal`) and drive one call to completion
        (`fleet`); nothing here puts a timeout on a door."""
        source = (DEPLOY / "modal_venue.py").read_text()
        self.assertIn("def fleet()", source)
        for banned in ("wait_for(", "timeout=", ".cancel("):
            for line in source.splitlines():
                if banned in line and "call.get(" not in line:
                    self.assertNotIn(
                        "desk()", line,
                        f"a desk call is wrapped in a client-side bound: "
                        f"{line.strip()}")

    def test_every_venue_smokes_its_own_image(self) -> None:
        """ADR 0008, F5: a build is a declaration until something runs in it,
        so every venue wires a `smoke` its header tells you to run BEFORE
        `modal deploy` — the image, exercised, on no metal."""
        for name in ("concept_steer", "steer_l4", "stress_fleet"):
            source = (DEPLOY / f"{name}.py").read_text()
            self.assertIn("smoke = smoke_function(", source, name)
            self.assertIn(f"deploy/{name}.py::smoke", source,
                          f"{name}'s header does not say to run its smoke")

    def test_the_concept_venue_s_smoke_builds_its_specs(self) -> None:
        """The strongest smoke a venue can run without metal: the client-side
        path a submit takes, against a throwaway store, inside the image."""
        told = self.concept_steer.the_science()
        self.assertEqual(sorted(told),
                         ["student@10", "student@32", "student@54", "teacher"])
        self.assertTrue(all(size > 0 for size in told.values()), told)

    def test_no_door_stands_on_an_on_demand_function(self) -> None:
        """ADR 0008, F6: the canonical row is computed on the CLIENT, the plan
        bytes go through the desk's `put_plan`, progress is an HTTP GET at the
        observer, and the measuring pass runs on the metal. The three CPU
        functions those doors used to stand on — `canonical`, `progress`,
        `measure_once` — are gone, and for an hour on 2026-09-04 Modal
        scheduled none of them."""
        for name in ("concept_steer", "steer_l4", "stress_fleet"):
            source = (DEPLOY / f"{name}.py").read_text()
            for retired in ("def canonical(", "def measure_once(",
                            "progress_function"):
                self.assertNotIn(retired, source,
                                 f"{name} still stands on {retired}")
        chassis = (DEPLOY / "modal_venue.py").read_text()
        self.assertNotIn("def progress_function(", chassis)
        self.assertIn("def canonical_row(", chassis)
        self.assertIn("/api/run/", chassis)

    def test_the_chassis_follows_a_run_through_the_observer(self) -> None:
        """F6: progress is a store read and a read-only service over the
        store is already standing, so the door polls it — and says loudly
        where it is when nobody has told it."""
        with self.assertRaises(SystemExit) as caught:
            self.modal_venue.progress("some-run")
        self.assertIn("RLSTACK_OBSERVER", str(caught.exception))

    def test_a_campaign_door_never_releases(self) -> None:
        """Q6: `concept_steer` and `gsm_a100` are CAMPAIGN venues — arms on
        one booted metal — so no door of either hands metal back. Idle metal
        is the desk's (ADR 0003), and under one desk a door's teardown would
        take whatever else had joined."""
        for name in ("concept_steer", "gsm_a100"):
            self.assertEqual(calls_named(DEPLOY / f"{name}.py",
                                         {"take_down", "guarded_release",
                                          "release", "sweep"}), [],
                             f"a campaign door of {name} released metal")

    def test_a_check_venue_releases_only_its_own_metal(self) -> None:
        """Q2/Q6: the check venues still end in a release, scoped to the
        metals THEY registered and guarded by the desk."""
        for name in ("steer_l4", "stress_fleet"):
            source = (DEPLOY / f"{name}.py").read_text()
            self.assertIn("MINE = ", source)
            self.assertIn("take_down(MINE", source)

    def test_the_retired_venues_are_gone(self) -> None:
        """Q8: the two hand-built venues bypassed the desk and built hosts in
        the driver's container. History keeps them; the tree does not."""
        self.assertFalse((DEPLOY / "dapo_grpo.py").exists())
        self.assertFalse((DEPLOY / "plora_l4.py").exists())

    def test_the_venue_recipes_are_proposals_the_desk_can_read(self) -> None:
        """Q4: a venue may say what its metal is for, and what it says is a
        `Builds` row — the same shape `deploy/desk.py::recipe` writes."""
        from rlstack.runner.residents import Builds

        for venue in (self.concept_steer, self.steer_l4, self.stress_fleet,
                      self.gsm_a100):
            recipe = venue.proposed_recipe()
            self.assertEqual(Builds.from_row(recipe.row()), recipe)


class DeskDoorTest(VenueFixture):
    """`deploy/desk.py::recipe`'s flag grammar — the operator's declaration."""

    def test_a_build_line_reads_as_kwargs(self) -> None:
        self.assertEqual(
            self.desk_venue.parse_build(
                "max_model_len=4096,serves=steer,serves=lora,enforce_eager"),
            {"max_model_len": 4096, "serves": ("steer", "lora"),
             "enforce_eager": True})

    def test_an_empty_line_is_the_build_s_own_defaults(self) -> None:
        self.assertEqual(self.desk_venue.parse_build(""), {})

    def test_the_door_builds_the_recipe_a_carve_would_ride(self) -> None:
        """What the door declares is exactly what a venue proposes: a metal
        told `serves=lora,serves=steer` at the desk carves the same engine
        `steer_l4` would have proposed for itself."""
        self.assertEqual(
            self.desk_venue.declared(
                "max_model_len=2048,max_bundles=8,max_rank=16,serves=lora,"
                "serves=steer,enforce_eager", ""),
            self.steer_l4.proposed_recipe())


if __name__ == "__main__":
    unittest.main()


class BurgersTest(VenueFixture):
    """The 0.6B proof of concept (2026-09-05): the same science off one
    `Campaign` record, a burger for a concept, three anchors at ~15/50/85 %
    of 28 layers, and the pool and the learner as two placement units."""

    def test_the_campaign_pins_its_rows(self) -> None:
        v = self.concept_burgers
        self.assertRowUnchanged("concept_burgers.teacher_spec",
                                v.teacher_spec(self.store, self.concept_tasks))
        for layer in v.ANCHORS:
            self.assertRowUnchanged(
                f"concept_burgers.student_spec@{layer}",
                v.student_spec(self.store, "teacher-run-0", layer, "nsteer"))
        self.assertRowUnchanged("concept_burgers.opd_spec@14",
                                v.opd_spec(self.store, self.concept_tasks, 14, "nsteer"))

    def test_the_anchors_span_the_depth_and_the_placement_is_split(self) -> None:
        v = self.concept_burgers
        self.assertEqual(v.ANCHORS, (4, 14, 24))
        self.assertEqual(v.BASE, "Qwen/Qwen3-0.6B")
        spec = v.student_spec(self.store, "teacher-run-0", 14, "mlp_lora")
        self.assertEqual(len(spec.topology.hosts), 2)        # the pool, the learner
        opd = v.opd_spec(self.store, self.concept_tasks, 24, "nsteer")
        self.assertEqual(len(opd.topology.hosts), 3)         # ... and the teacher
        self.assertEqual(spec.policy.bank["v"].site, "layers.14.mlp.*")

    def test_a_third_of_the_depth_is_one_arm(self) -> None:
        """`layer` may be a range: one direction per boundary of the third,
        one alpha; validates like any anchor arm."""
        from rlstack.spec.validate import validate_or_raise
        from rlstack.policy.siteschema import fake_qwen_schema

        v = self.concept_burgers
        schema = fake_qwen_schema(28, base=v.BASE)
        for third in v.THIRDS:
            spec = v.student_spec(self.store, "teacher-run-0", third, "nsteer")
            self.assertEqual(spec.policy.bank["v"].site, f"resid_pre.{third}")
            validate_or_raise(spec, schema)
        self.assertEqual(v.THIRDS, ("0-8", "9-17", "18-27"))

    def test_a_probe_serves_a_parents_delta_frozen(self) -> None:
        """A probe is generation-only: a frozen bank entry warm-started from
        the parent's sealed version, a few held-out waves, no algo."""
        from rlstack.policy.siteschema import fake_qwen_schema
        from rlstack.spec.validate import validate_or_raise

        v = self.concept_burgers
        spec = v.probe_spec(self.store, self.concept_tasks, "parent-run-0", 64, "0-8")
        self.assertIsNone(spec.algo)
        self.assertFalse(spec.policy.bank["v"].trainable)
        self.assertEqual(spec.init.policy, "store://parent-run-0@64")
        self.assertEqual(len(spec.topology.hosts), 1)
        validate_or_raise(spec, fake_qwen_schema(28, base=v.BASE))
        bare = v.probe_spec(self.store, self.concept_tasks, None, 0, 4)
        self.assertEqual(bare.policy.bank, {})
        self.assertIsNone(bare.init)

    def test_the_happiness_venue_still_speaks_its_own_names(self) -> None:
        """The delegation kept every name the doors and this file use."""
        v = self.concept_steer
        self.assertEqual(v.HAPPINESS.anchors, v.ANCHORS)
        self.assertFalse(v.HAPPINESS.split)
        self.assertEqual(len(v.topology().hosts), 1)


class AngerTest(VenueFixture):
    """The 32B anger campaign (2026-09-05): six nsteer arms under SFT — three
    anchors, three thirds — over the alternating placement."""

    def test_every_arm_validates_and_alternates(self) -> None:
        from rlstack.policy.siteschema import fake_qwen_schema
        from rlstack.spec.validate import validate_or_raise

        v = self.concept_anger
        schema = fake_qwen_schema(64, base=v.BASE)
        self.assertEqual((v.BASE, v.HIDDEN, v.WIDTH), ("Qwen/Qwen3-32B", 5120, 2))
        self.assertEqual(v.THIRDS, ("0-20", "21-42", "43-63"))
        for where in (*v.ANCHORS, *v.THIRDS):
            spec = v.student_spec(self.store, "teacher-run-0", where, "nsteer")
            self.assertEqual(spec.policy.bank["v"].site, f"resid_pre.{where}")
            self.assertEqual(len(spec.topology.hosts), 1)          # alternating
            self.assertEqual(len(spec.topology.hosts[0].members), 2)
            validate_or_raise(spec, schema)
        teacher = v.teacher_spec(self.store, self.concept_tasks)
        self.assertIsNone(teacher.algo)
        validate_or_raise(teacher, schema)

    def test_the_anger_block_is_dense(self) -> None:
        from rlstack.data.tasks.concept_prompts import SYSTEM_PROMPTS

        block = SYSTEM_PROMPTS["anger"].format(concept="anger")
        for word in ("FURIOUS", "as many individual words", "word by word"):
            self.assertIn(word, block)
