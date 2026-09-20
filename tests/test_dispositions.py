"""Stop is a disposition (ADR 0014, Part C).

A deliberate stop DRAINS (the Trainer checkpoints its last commit first) and
lands on the fleet journal as `stopped` — the row that outranks `parked`:
nothing automatic revives it, not a reap, not a metal registration, not a
knock; only a new submission does, and that delivery supersedes the stop.
A run that dies of its own accord is `failed` and stays; a run that dies of
the WIRE is `parked` and retried from its checkpoint. The observer ranks the
three the same way.
"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from common import arith_spec
from rlstack import FakeEngine
from rlstack.runner.campaign import Campaigns
from rlstack.runner.checkpointing import Checkpointing
from rlstack.runner.roles import Trainer
from rlstack.runner.remote import Unreachable
from rlstack.spec.specs import Seeds
from rlstack.observe.views import note_the_fleet
from test_desk import BASE, DeskFixture, go

EVERY_SEVEN = Checkpointing(7)      # no checkpoint before the extent: a drain must make one


class HoldAfter(FakeEngine):
    """A FakeEngine that holds sampling at a gate once `when()` says so —
    deterministically parks a run in its RUNNING state with some commits
    behind it, so a test can stop it mid-flight."""

    def __init__(self, gate: asyncio.Event, when, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gate, self.when = gate, when

    async def sample_tokens(self, messages, sampling, stop, bundle_id, seed,
                            directives=()):
        if self.when():
            await self.gate.wait()
        async for event in FakeEngine.sample_tokens(
                self, messages, sampling, stop, bundle_id, seed, directives):
            yield event


class StopIsADispositionTest(DeskFixture):
    def committed(self, rid: str) -> int:
        entries = self.store.peek_ledger(rid)
        return int(entries[-1]["update"]) if entries else 0

    def test_a_stop_drains_journals_and_nothing_automatic_revives_it(self) -> None:
        gate = asyncio.Event()
        held = {"rid": None}
        engine = HoldAfter(gate, lambda: held["rid"] is not None and self.committed(held["rid"]) >= 2,
                           base=BASE)
        host = self.stand_up("h", "fleet://h", serves_pool=True, trains=True, engine=engine)
        desk = self.desk()
        desk.list_host("h", host.regimes, "fleet://h")
        campaigns = Campaigns(desk)

        async def drive():
            accepted = await campaigns.submit(arith_spec(self.train), checkpointing=EVERY_SEVEN)
            rid = accepted["run_id"]
            held["rid"] = rid
            while self.committed(rid) < 2:              # two commits, no checkpoint yet
                await asyncio.sleep(0.01)
            stopped = await campaigns.serve("stop", {"run_id": rid, "reason": "scope cut"})
            return accepted, rid, stopped

        accepted, rid, stopped = go(drive())
        self.assertTrue(accepted["accepted"], accepted)
        self.assertTrue(stopped["stopped"], stopped)
        self.assertTrue(stopped["drained"], stopped)
        self.assertEqual(stopped["state"], "stopped")
        # THE DRAIN: the last commit is now a checkpoint, and the run is not done
        sealed = [e["update"] for e in self.store.peek_checkpoints(rid)]
        self.assertEqual(sealed[-1], self.committed(rid))
        self.assertGreaterEqual(sealed[-1], 2)
        self.assertEqual(host.roster[rid].status, "stopped")
        self.assertFalse(desk.finished(rid))
        # THE DISPOSITION, on the record and in the reader
        self.assertIn(rid, desk.stopped())
        self.assertEqual(desk.stopped()[rid]["reason"], "scope cut")
        self.assertNotIn(rid, desk.parked())
        told = desk.answer("dispositions", {})
        self.assertIn(rid, told["stopped"])

        # NOTHING AUTOMATIC REVIVES IT: a park cannot re-queue it, a reap and
        # a metal registration reroute nothing
        desk.park(rid, "a stray park")
        self.assertNotIn(rid, desk.parked())
        with patch.object(desk, "reroute", AsyncMock(return_value={"rerouted": True})) as move:
            go(desk.reap())
            go(desk.serve("metal", {"name": "new-metal", "gpu": "L4", "devices": 1,
                                    "vram_gb": 24, "idle_s": 300}))
            self.assertEqual(go(desk.retry_parked()), {})
            move.assert_not_awaited()
        self.assertIn(rid, desk.stopped())

        # ONLY A NEW SUBMISSION MOVES IT — and that supersedes the stop
        gate.set()
        held["rid"] = None

        async def resume():
            reborn = await campaigns.submit(arith_spec(self.train), checkpointing=EVERY_SEVEN)
            await host._adoptions[reborn["run_id"]]
            return reborn

        reborn = go(resume())
        self.assertTrue(reborn["accepted"], reborn)
        self.assertEqual(reborn["run_id"], rid)
        self.assertEqual(host.roster[rid].status, "done")
        self.assertNotIn(rid, desk.stopped())
        self.assertTrue(desk.finished(rid))
        # the resumed run took the drain's checkpoint as its resume point:
        # the record continues from it rather than from 0
        self.assertEqual([e["update"] for e in self.store.peek_checkpoints(rid)][:2], [0, sealed[-1]])

    def test_stop_subdir_stops_every_unfinished_run_under_it(self) -> None:
        gate = asyncio.Event()
        engine = HoldAfter(gate, lambda: True, base=BASE)
        host = self.stand_up("h", "fleet://h", serves_pool=True, trains=True, engine=engine)
        desk = self.desk()
        desk.list_host("h", host.regimes, "fleet://h")
        campaigns = Campaigns(desk)

        async def drive():
            one = await campaigns.submit(arith_spec(self.train), subdir="family/a",
                                         checkpointing=EVERY_SEVEN)
            other = await campaigns.submit(arith_spec(self.train, seeds=Seeds(master=31)),
                                           subdir="family/b", checkpointing=EVERY_SEVEN)
            told = await campaigns.serve("stop_subdir", {"subdir": "family/a", "reason": "done with a"})
            gate.set()                                  # the other family runs on
            await host._adoptions[other["run_id"]]
            return one, other, told

        one, other, told = go(drive())
        self.assertEqual(sorted(told["stopped"]), [one["run_id"]])
        self.assertIn(one["run_id"], desk.stopped())
        self.assertEqual(desk.stopped()[one["run_id"]]["subdir"], "family/a")
        self.assertEqual(host.roster[one["run_id"]].status, "stopped")
        self.assertEqual(host.roster[other["run_id"]].status, "done")

    def test_an_undrained_stop_cancels_at_once(self) -> None:
        gate = asyncio.Event()
        engine = HoldAfter(gate, lambda: True, base=BASE)
        host = self.stand_up("h", "fleet://h", serves_pool=True, trains=True, engine=engine)
        desk = self.desk()
        desk.list_host("h", host.regimes, "fleet://h")

        async def drive():
            accepted = await Campaigns(desk).submit(arith_spec(self.train), checkpointing=EVERY_SEVEN)
            return accepted, await desk.stop(accepted["run_id"], "now", drain=False)

        accepted, stopped = go(drive())
        self.assertTrue(stopped["stopped"])
        self.assertFalse(stopped["drained"])
        self.assertEqual(host.roster[accepted["run_id"]].status, "stopped")
        detaches = [e for e in self.store.read_host_log("h") if e.get("event") == "detach"]
        self.assertEqual([e["status"] for e in detaches], ["stopped"])


class DeathsTest(DeskFixture):
    """The desk reads the rosters on its tick and journals what died: the
    experiment's own death is `failed` and stays; the wire's is `parked`
    and retried from the checkpoint."""

    def dying_at(self, update: int, death: BaseException):
        original = Trainer.commit

        async def commit_then_die(trainer, u, *args, **kwargs):
            await original(trainer, u, *args, **kwargs)
            if u == update:
                raise death
        return patch.object(Trainer, "commit", commit_then_die)

    def test_the_experiments_own_death_is_failed_and_never_retried(self) -> None:
        host = self.stand_up("h", "fleet://h", serves_pool=True, trains=True)
        desk = self.desk()
        desk.list_host("h", host.regimes, "fleet://h")

        async def drive():
            with self.dying_at(2, RuntimeError("the loss went NaN")):
                accepted = await Campaigns(desk).submit(arith_spec(self.train),
                                                        checkpointing=Checkpointing(2))
                with self.assertRaises(RuntimeError):
                    await host._adoptions[accepted["run_id"]]
            return accepted["run_id"], await desk.reap()

        rid, reaped = go(drive())
        self.assertEqual(reaped["deaths"], {rid: "failed"})
        self.assertEqual(reaped["runs"], {})
        self.assertIn(rid, desk.failed())
        self.assertIn("the loss went NaN", desk.failed()[rid]["error"])
        self.assertNotIn(rid, desk.parked())
        self.assertEqual(host.roster[rid].status, "failed")
        self.assertFalse(host.roster[rid].wire)
        # a second tick journals nothing new and retries nothing
        with patch.object(desk, "reroute", AsyncMock(return_value={"rerouted": True})) as move:
            self.assertEqual(go(desk.reap())["deaths"], {})
            move.assert_not_awaited()
        rows = [e for e in self.store.read_fleet_log() if e.get("event") == "failed"]
        self.assertEqual(len(rows), 1)

    def test_a_wire_death_is_parked_and_resumes_from_the_checkpoint(self) -> None:
        host = self.stand_up("h", "fleet://h", serves_pool=True, trains=True)
        desk = self.desk()
        desk.list_host("h", host.regimes, "fleet://h")

        async def drive():
            with self.dying_at(3, Unreachable("the pool stopped answering")):
                accepted = await Campaigns(desk).submit(arith_spec(self.train),
                                                        checkpointing=Checkpointing(2))
                with self.assertRaises(Unreachable):
                    await host._adoptions[accepted["run_id"]]
            rid = accepted["run_id"]
            self.assertTrue(host.roster[rid].wire)
            reaped = await desk.reap()                  # parks it, then retries it
            await host._adoptions[rid]
            return rid, reaped

        rid, reaped = go(drive())
        self.assertEqual(reaped["deaths"], {rid: "parked"})
        self.assertEqual(reaped["runs"], {rid: "rerouted"})
        self.assertEqual(host.roster[rid].status, "done")
        self.assertTrue(desk.finished(rid))
        self.assertNotIn(rid, desk.parked())
        self.assertNotIn(rid, desk.failed())
        # it resumed from checkpoint 2, not from 0: the record reads 0, 2, 4
        self.assertEqual([e["update"] for e in self.store.peek_checkpoints(rid)], [0, 2, 4])


class TheObserverRanksDispositionsTest(unittest.TestCase):
    def test_stopped_and_failed_outrank_parked_and_a_host_journals_last_word(self) -> None:
        notes = {"parked": {"p": {"reason": "no metal"}, "s": {"reason": "stale"}},
                 "stopped": {"s": {"reason": "scope cut", "drained": True}},
                 "failed": {"f": {"error": "OOM"}},
                 "unreachable": {}}
        rows = [{"run_id": "p", "status": "running"}, {"run_id": "s", "status": "running"},
                {"run_id": "f", "status": "running"}, {"run_id": "d", "status": "done"}]
        note_the_fleet(rows, notes, "run_id")
        self.assertEqual([r["status"] for r in rows], ["parked", "stopped", "failed", "done"])
        self.assertEqual(rows[1]["stopped"]["reason"], "scope cut")
        self.assertEqual(rows[2]["failed"]["error"], "OOM")


if __name__ == "__main__":
    unittest.main()
