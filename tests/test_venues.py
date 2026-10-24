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
import inspect
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

from venue_stub import modal_stubbed

REPO = pathlib.Path(__file__).resolve().parent.parent
DEPLOY = REPO / "deploy"
ROWS = json.loads((REPO / "tests" / "venue_spec_rows.json").read_text())

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
    """The framework venues, imported once, over a store their specs can build in."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._stub = modal_stubbed()
        cls._stub.__enter__()
        sys.path.insert(0, str(DEPLOY))          # so `import modal_venue` works
        try:
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
        """Real content-addressed concept prompts for the teacher's plan."""
        from rlstack import LocalStore
        from rlstack.data.tasks.base import Task, write_tasks

        store = LocalStore(root)
        concept = write_tasks(store, [Task(id=f"p{i:05d}", prompt=f"q{i}",
                                           meta={}) for i in range(2304)])
        return store, concept

    def assertScientificSettings(self, key: str, spec) -> None:
        """Keep historical scientific settings while changing code and plan vocabulary.

        The external code declaration and function-plan CAS reference deliberately
        change identity under ADR0030; optimizer, adapter, sampling and placement
        settings still match the historical control.
        """
        from rlstack.spec.canonical import canonical_json

        actual, expected = json.loads(canonical_json(spec)), json.loads(ROWS[key])
        for row in (actual, expected):
            row.pop("code", None)
            if row.get("gen"):
                row["gen"].pop("envs", None)
                row["gen"].pop("makers", None)
                row["gen"].pop("functions", None)
                row["gen"].pop("tasks", None)
            row["plans"].pop("rollout", None)
            if row.get("algo"):
                row["algo"].pop("post", None)
        self.assertEqual(actual, expected,
                         f"{key}: scientific settings changed during the boundary move")




class SpecsAreUnchangedTest(VenueFixture):


    def test_the_two_tenants_of_the_steer_check(self) -> None:
        venue = self.steer_l4
        for name, (bank, overrides) in venue.the_two_banks().items():
            self.assertScientificSettings(
                f"steer_l4.spec_for/{name}",
                venue.spec_for(self.store, bank, overrides, venue.UPDATES, 11))

    def test_the_three_tenants_of_the_stress_check(self) -> None:
        venue = self.stress_fleet
        for name, (bank, overrides) in venue.the_banks().items():
            self.assertScientificSettings(
                f"stress_fleet.spec_for/{name}",
                venue.spec_for(self.store, bank, overrides, venue.UPDATES, 41))


    def test_the_fixture_covers_every_spec_the_venues_build(self) -> None:
        """A row nobody compares is a promise nobody keeps."""
        self.assertEqual(len([key for key in ROWS if not key.startswith("concept_")]), 5)


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
                 if calls_named(path, {"desk_class"})]
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
        from unittest.mock import patch
        from rlstack.runner.desk import Metal, MetalService
        from rlstack.runner.venues.runtime import MetalRuntime

        with patch.object(MetalService, "measure", return_value=Metal("test", "L4", 1, 24)):
            runtime = MetalRuntime.measured(
                "test", self.store, self.modal_venue.desk(), address="modal://test/MetalS",
                host_address=lambda host, epoch: host, container="ta-test", idle_s=90,
                heartbeat_s=self.modal_venue.HEARTBEAT_S)
        self.assertEqual(runtime.heartbeat_s, 20.0)
        runtime.learn_heartbeat({"heartbeat_s": 7.0})
        self.assertEqual(runtime.heartbeat_s, 7.0)

    def test_both_doors_of_every_container_are_async(self) -> None:
        """ADR 0008, Q4: a cancelled input of a SYNCHRONOUS method on a
        concurrent container has no clean interruption, so Modal shuts the
        container down — which is how the desk died three times on
        2026-09-04. Both doors are async methods now, on the metal and on the
        desk alike."""
        worker = self.modal_venue.metal_class(
            self.modal_venue.modal.App("test"), "test", "test", "L4", None,
            module="venue_test")
        for container in (worker, self.desk_venue.Desk):
            self.assertTrue(inspect.iscoroutinefunction(container.door))
            self.assertTrue(inspect.iscoroutinefunction(container.door_ask))

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
        for name in ("steer_l4", "stress_fleet"):
            source = (DEPLOY / f"{name}.py").read_text()
            self.assertIn("smoke = smoke_function(", source, name)
            self.assertIn(f"deploy/{name}.py::smoke", source,
                          f"{name}'s header does not say to run its smoke")


    def test_no_door_stands_on_an_on_demand_function(self) -> None:
        """ADR 0008, F6: the canonical row is computed on the CLIENT, the plan
        bytes go through the desk's `put_plan`, progress is an HTTP GET at the
        observer, and the measuring pass runs on the metal. The three CPU
        functions those doors used to stand on — `canonical`, `progress`,
        `measure_once` — are gone, and for an hour on 2026-09-04 Modal
        scheduled none of them."""
        for name in ("steer_l4", "stress_fleet"):
            source = (DEPLOY / f"{name}.py").read_text()
            for retired in ("def canonical(", "def measure_once(",
                            "progress_function"):
                self.assertNotIn(retired, source,
                                 f"{name} still stands on {retired}")
        chassis = (DEPLOY / "modal_venue.py").read_text()
        self.assertNotIn("def progress_function(", chassis)
        self.assertIn("def canonical_row(", chassis)
        from rlstack.runner.venues.client import VenueClient
        self.assertIn("/api/run/", inspect.getsource(VenueClient.progress))

    def test_the_chassis_follows_a_run_through_the_observer(self) -> None:
        """F6: progress is a store read and a read-only service over the
        store is already standing, so the door polls it — and says loudly
        where it is when nobody has told it."""
        with self.assertRaises(SystemExit) as caught:
            self.modal_venue.progress("some-run")
        self.assertIn("RLSTACK_OBSERVER", str(caught.exception))


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

        for venue in (self.steer_l4, self.stress_fleet):
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
