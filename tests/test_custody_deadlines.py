"""A missed roster reply cannot authorize another writer or a teardown."""

import asyncio
import unittest

from test_desk import DeskFixture, go
from rlstack.runner.campaign import demands_of, frame_for
from rlstack.runner.desk import DeskError, demand_rows
from rlstack.runner.remote import Unreachable


class CustodyDeadlineTest(DeskFixture):
    async def exercise(self, action):
        gate = asyncio.Event()
        self.metal_service(devices=2, sample_gate=gate)
        desk = self.desk_with_metal("fake-metal")
        spec = self.split_spec()
        rows, frame = demand_rows(demands_of(spec)), frame_for(spec)
        first = await desk.submit(rows, frame)
        self.assertTrue(first["accepted"])
        owner = desk.listings[first["host"]].host
        original = owner.status

        async def missed(*, deadline_s):
            raise Unreachable("busy owner missed a bounded status read")

        before = len([e for e in self.store.read_fleet_log()
                      if e["event"] == "place" and e.get("delivered")])
        owner.status = missed
        try:
            with self.assertRaisesRegex(DeskError, "custody is unknown"):
                await action(desk, first, rows, frame)
            self.assertIn(first["host"], desk.listings)
            after = len([e for e in self.store.read_fleet_log()
                         if e["event"] == "place" and e.get("delivered")])
            self.assertEqual(after, before, "unknown custody must not redeliver")
        finally:
            owner.status = original
            await desk.stop_anchored(first["run_id"])
            gate.set()

    def test_timeout_does_not_mean_no_running_runs(self):
        async def action(desk, first, rows, frame):
            await desk.running_runs()
        go(self.exercise(action))

    def test_resubmit_cannot_duplicate_an_unreadable_owner(self):
        async def action(desk, first, rows, frame):
            await desk.submit(rows, frame)
        go(self.exercise(action))

    def test_move_requires_confirmed_old_owner_state(self):
        async def action(desk, first, rows, frame):
            await desk.reroute(first["run_id"], avoiding=first["host"], park=True)
        go(self.exercise(action))

    def test_guarded_teardown_cannot_discard_an_unreadable_owner(self):
        async def action(desk, first, rows, frame):
            await desk.decommission(first["host"], force=False)
        go(self.exercise(action))

    def test_all_rosters_are_known_before_stopping_a_visible_owner(self):
        async def exercise():
            desk = self.desk()
            stops = []
            for name in ("a-visible", "z-unreadable"):
                host = self.stand_up(name, f"fleet://{name}",
                                     serves_pool=False, trains=True)
                desk.list_host(name, host.regimes, f"fleet://{name}")

            async def visible(*, deadline_s):
                return {"tenants": {"run": {"status": "running"}}}

            async def missed(*, deadline_s):
                raise Unreachable("later roster timed out")

            async def stop(run_id, *, deadline_s):
                stops.append(run_id)
                return {"stopped": True}

            desk.listings["a-visible"].host.status = visible
            desk.listings["a-visible"].host.stop = stop
            desk.listings["z-unreadable"].host.status = missed
            with self.assertRaisesRegex(DeskError, "custody is unknown"):
                await desk.stop_anchored("run")
            self.assertEqual(stops, [])
        go(exercise())

    def test_duplicate_owners_are_not_resolved_by_first_roster_order(self):
        async def exercise():
            desk = self.desk()
            stops = []

            async def occupied(*, deadline_s):
                return {"tenants": {"run": {"status": "running"}}}

            async def stop(run_id, *, deadline_s):
                stops.append(run_id)
                return {"stopped": True}

            for name in ("owner-a", "owner-b"):
                host = self.stand_up(name, f"fleet://{name}",
                                     serves_pool=False, trains=True)
                desk.list_host(name, host.regimes, f"fleet://{name}")
                desk.listings[name].host.status = occupied
                desk.listings[name].host.stop = stop
            with self.assertRaisesRegex(DeskError, "multiple running owners"):
                await desk.stop_anchored("run")
            self.assertEqual(stops, [])
        go(exercise())


if __name__ == "__main__":
    unittest.main()
