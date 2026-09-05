"""The venue chassis' shape, on fakes and with no Modal anywhere (ADR 0007, Q9).

WHAT A VENUE ACTUALLY DOES, once `deploy/modal_venue.py` holds the plumbing:
a metal container comes up BARE, registers itself with THE desk at an address
that carries its venue, is told what to build, and thereafter answers carve
and host frames through `transport_for` — the one factory — while a campaign
submits a spec to that same desk and follows the run to its extent.

Every one of those steps is exercised here against a process-free fake metal
(the `FakeEngineBuild` precedent) reached over the `local://` scheme, so what
is under test is the CHASSIS' SHAPE — bare boot, the desk's declaration, the
address grammar, the in-process rule, submit-and-follow — and not Modal, which
this file never imports. What is NOT proven here is the venue: no container is
built, no card is measured, and `deploy/steer_l4.py::check` on real metal is
still what says the rewrite changed no number.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from common import arith_spec, arith_store
from rlstack import (
    FakeEngine, FakeLearner, HostSpec, Metal, Topology, fake_qwen_schema,
    learner, pool, run_progress,
)
from rlstack.runner.campaign import Campaigns
from rlstack.runner.desk import Desk, MetalService
from rlstack.runner.remote import (
    IN_PROCESS, RemoteHost, RemoteMetal, serve_in_process,
    stop_serving_in_process, transport_for,
)
from rlstack.runner.residents import Builds, Resident

BASE = "Qwen/Qwen3-0.6B"


def go(coro):
    return asyncio.run(coro)


class ChassisFixture(unittest.TestCase):
    """One bare metal and one desk, wired the way a venue wires them: every
    resolver is `transport_for` and every address is the grammar's."""

    METAL = "fake-l4"
    PLANE = "local://fake-l4/plane"

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)
        self.addCleanup(stop_serving_in_process, self.PLANE)

    def bare_metal(self) -> MetalService:
        """`metal_class`'s bring-up, minus Modal: measured facts (typed here
        because there is no card), the store, `transport_for` as the router,
        and NO RECIPE — the container does not know what it is for until the
        desk says (ADR 0007, Q4)."""
        service = MetalService(
            Metal(self.METAL, "L4", 2, 24.0), store=self.store,
            address_of=lambda host: f"local://{self.METAL}/{host}",
            schema_for=lambda base: fake_qwen_schema(4, base=base),
            transport_for=transport_for,
            spawn=lambda birth: Resident.in_process(
                birth,
                FakeLearner() if birth.regime.capability == "training"
                else FakeEngine(base=birth.regime.base)))
        serve_in_process(self.PLANE, service)      # the venue's `door`
        return service

    def a_desk(self) -> Desk:
        """`deploy/desk.py`'s bring-up, minus Modal: `host_for` and
        `metal_for` are the ONE factory, which is the whole of Q3."""
        return Desk(self.store,
                    host_for=lambda address: RemoteHost(transport_for(address)),
                    metal_for=lambda address: RemoteMetal(transport_for(address)))

    def a_spec(self):
        """One HostSpec, two members: a serving pool and a learner."""
        return arith_spec(self.train, topology=Topology(hosts=(
            HostSpec((pool("main"), learner())),)))


class ChassisTest(ChassisFixture):
    def test_a_venue_boots_bare_registers_is_told_and_runs(self) -> None:
        """THE WHOLE CHASSIS, END TO END. A bare metal announces itself at its
        plane address; the desk lists it as inventory it may not yet carve on;
        the desk DECLARES what it builds; a spec submitted to that desk places,
        carves, adopts and reaches its extent; and `run_progress` — the one
        predicate a campaign door follows — says done."""
        service = self.bare_metal()
        desk = self.a_desk()

        # announce: `metal_class`'s own registration, proposing nothing
        go(desk.register_metal(service.metal, address=self.PLANE, idle_s=None))
        self.assertIsNone(desk.recipe_for(self.METAL))
        self.assertIsNone(service.builds)

        # a campaign submitted before the declaration finds nowhere to go
        early = go(Campaigns(desk).submit(self.a_spec()))
        self.assertFalse(early["accepted"], early)

        # the desk's own door (deploy/desk.py::recipe): what this metal builds
        desk.recipe(self.METAL, Builds.fakes())

        async def drive():
            reply = await Campaigns(desk).submit(self.a_spec())
            self.assertTrue(reply["accepted"], reply)
            await service.hosts[reply["host"]]._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())

        # the recipe reached the container through the CARVE, never a venue
        # constant, and the run followed to its extent
        self.assertEqual(service.builds, Builds.fakes())
        told = run_progress(self.store, reply["run_id"])
        self.assertTrue(told.done, told)
        self.assertEqual(told.completed, told.planned)

    def test_the_carved_hosts_address_is_reachable_through_the_factory(self) -> None:
        """#77's rule, now the chassis': a host carved inside this process is
        published on the in-process switchboard by `MetalService.route`, so
        `transport_for` on its address answers HERE instead of dialling the
        container it is already in — and a decarve takes the entry with it,
        so a stale address never routes."""
        service = self.bare_metal()
        desk = self.a_desk()
        go(desk.register_metal(service.metal, address=self.PLANE, idle_s=None))
        desk.recipe(self.METAL, Builds.fakes())
        go(Campaigns(desk).submit(self.a_spec()))

        listed = sorted(desk.listings)
        self.assertEqual(len(listed), 1)
        address = desk.listings[listed[0]].address
        self.assertIn(address, IN_PROCESS)
        self.assertTrue(RemoteHost(transport_for(address)).status())

        go(service.decarve(listed[0]))
        self.assertNotIn(address, IN_PROCESS)
        with self.assertRaises(ValueError):
            transport_for(address)

    def test_a_release_takes_the_switchboard_with_it(self) -> None:
        """The shift ends, every host comes down, and nothing this container
        published still answers — the state a reborn metal starts from."""
        service = self.bare_metal()
        desk = self.a_desk()
        go(desk.register_metal(service.metal, address=self.PLANE, idle_s=None))
        desk.recipe(self.METAL, Builds.fakes())
        go(Campaigns(desk).submit(self.a_spec()))
        addresses = [listing.address for listing in desk.listings.values()]

        go(desk.release(self.METAL, reason="the door is done", force=True))
        for address in addresses:
            self.assertNotIn(address, IN_PROCESS)
        self.assertTrue(service.released.is_set())     # the keepalive returns
        self.assertEqual(service.hosts, {})


if __name__ == "__main__":
    unittest.main()


DEPLOY = __import__('pathlib').Path(__file__).resolve().parents[1] / "deploy"


class OneWorkspaceTest(unittest.TestCase):
    """Every deploy and door runs in the yu-masala workspace and no other
    (Samarth, 2026-09-05): the chassis refuses another profile by name and
    stands down where the profile is not a string (the stub)."""

    def test_another_profile_is_refused_by_name(self) -> None:
        import sys
        import types

        from venue_stub import modal_stubbed
        with modal_stubbed():
            sys.path.insert(0, str(DEPLOY))
            try:
                config = types.ModuleType("modal.config")
                config._profile = "samarthmbhargav"
                sys.modules["modal"].config = config
                sys.modules["modal"].is_local = lambda: True      # the client
                sys.modules["modal.config"] = config
                import importlib
                spec = importlib.util.spec_from_file_location("venue_modal_venue_ws", DEPLOY / "modal_venue.py")
                module = importlib.util.module_from_spec(spec)
                with self.assertRaisesRegex(SystemExit, "yu-masala-workspace.*samarthmbhargav"):
                    spec.loader.exec_module(module)
                config._profile = "yu-masala-workspace"
                spec.loader.exec_module(module)          # the right one passes
                module.require_workspace()
                # inside a container the check stands down whatever the profile
                config._profile = "default"
                sys.modules["modal"].is_local = lambda: False
                module.require_workspace()
            finally:
                sys.path.remove(str(DEPLOY))
                sys.modules.pop("modal.config", None)
                sys.modules.pop("venue_modal_venue_ws", None)

