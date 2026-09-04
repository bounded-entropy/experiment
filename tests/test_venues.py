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
    """The three venues, imported once, over a store their specs can build in."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._stub = modal_stubbed()
        cls._stub.__enter__()
        sys.path.insert(0, str(DEPLOY))          # so `import modal_venue` works
        try:
            cls.concept_steer = load_venue("concept_steer")
            cls.steer_l4 = load_venue("steer_l4")
            cls.stress_fleet = load_venue("stress_fleet")
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

    def test_the_fixture_covers_every_spec_the_venues_build(self) -> None:
        """A row nobody compares is a promise nobody keeps."""
        self.assertEqual(len(ROWS), 9)


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

    def test_a_campaign_door_never_releases(self) -> None:
        """Q6: `concept_steer` is a CAMPAIGN venue — three arms on one booted
        metal — so no door of it hands metal back. Idle metal is the desk's
        (ADR 0003), and under one desk a door's teardown would take whatever
        else had joined."""
        self.assertEqual(calls_named(DEPLOY / "concept_steer.py",
                                     {"take_down", "guarded_release",
                                      "release", "sweep"}), [],
                         "a campaign door released metal")

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

        for venue in (self.concept_steer, self.steer_l4, self.stress_fleet):
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
        from rlstack.runner.residents import Builds, EngineBuild, LearnerBuild

        declared = Builds(
            engine=EngineBuild(**self.desk_venue.parse_build(
                "max_model_len=2048,max_bundles=8,max_rank=16,serves=lora,"
                "serves=steer,enforce_eager")),
            learner=LearnerBuild(**self.desk_venue.parse_build("")))
        self.assertEqual(declared, self.steer_l4.proposed_recipe())


if __name__ == "__main__":
    unittest.main()
