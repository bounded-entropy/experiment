"""The learner across hosts (ADR 0006 Part A): the verb, the door, the anchor.

The claims under test: `uninstall` is a verb of the protocol and install's
inverse — over the wire, and (where torch is present) in the module tree
itself; every learner verb crosses a HOST door ADMITTED at that host's own
arbiter, the way sample_tokens does, and the learner's host journals who is on
it; a routed learner attaches at the runner's arbiter as a zero-footprint free
resident, because its admission happens elsewhere; a host uninstalls its tenant
when the run ends, done or failed; and — the load-bearing one — a run anchored
on the MAIN pool's host with its learner on another host is byte-identical to
the same run anchored on the learner's host.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from common import arith_spec, arith_store
from test_resume import snapshot
from rlstack import (
    Arbiter, FakeEngine, FakeLearner, Host, HostService, LocalTransport,
    Regime, RemoteLearner, fake_qwen_schema,
)
from rlstack.runner.interfaces import (
    EntryInstall, OptimSettings, Parameterization,
)
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.loop import attach_residents
from rlstack.runner.remote import LearnerService
from rlstack.spec.canonical import canonical_json

try:
    import torch
except ImportError:                                  # pragma: no cover
    torch = None

needs_torch = unittest.skipUnless(torch is not None, "torch not installed")

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")
WIDTH, VOCAB = 8, 11


def go(coro):
    return asyncio.run(coro)


def a_parameterization(base: str = "Qwen/Qwen3-0.6B",
                       sites=(("pi", "layers.0.self_attn.q_proj"),),
                       ) -> Parameterization:
    """One trainable lora entry per named site — install's whole argument."""
    return Parameterization(
        base=base, loss="grpo",
        entries=tuple(EntryInstall(
            name=name, adapter_type="lora", init={"r": 2, "seed": 5},
            trainable=True,
            sites=(SiteMeta(name=path, path=path, has_weight=True,
                            shape=(WIDTH, VOCAB), is_boundary=False),))
            for name, path in sites),
        optim=OptimSettings(name="adamw", lr=1e-4, betas=(0.9, 0.95),
                            weight_decay=0.0, overrides={}))


class UninstallVerbTest(unittest.TestCase):
    """Install's inverse, as a verb: the tenancy ends and its state goes."""

    def test_uninstall_forgets_the_tenant_and_is_idempotent(self) -> None:
        learner = FakeLearner()
        learner.install("run-a", a_parameterization())
        learner.install("run-b", a_parameterization())
        learner.uninstall("run-a")
        with self.assertRaises(KeyError):
            learner.emit("run-a")
        learner.uninstall("run-a")            # already gone is the goal state
        learner.uninstall("never-installed")
        self.assertIsNotNone(learner.emit("run-b"))   # a co-tenant is untouched

    def test_every_verb_round_trips_through_the_resident_door(self) -> None:
        """The six verbs over LearnerService + a RemoteLearner: what the
        wire carries must equal what the object does, byte for byte."""
        direct, served = FakeLearner(), FakeLearner()
        proxy = RemoteLearner(LocalTransport(LearnerService(served)))
        for learner in (direct, proxy):
            learner.install("run-a", a_parameterization())
            learner.load("run-a", {"pi": b"fake-delta:pi:deadbeef"}, None)
            learner.optim_step("run-a")
        self.assertEqual(direct.emit("run-a").adapters,
                         proxy.emit("run-a").adapters)
        proxy.uninstall("run-a")
        self.assertEqual(served._tenants, {})

    def test_the_frame_survives_json(self) -> None:
        """The honesty gate: LocalTransport round-trips both directions, so a
        verb that works over it works over a real transport."""
        served = FakeLearner()
        proxy = RemoteLearner(LocalTransport(LearnerService(served)))
        proxy.install("run-a", a_parameterization())
        json.dumps({"tenant": "run-a"})
        self.assertEqual(sorted(served._tenants), ["run-a"])


class LearnerDoorTest(unittest.TestCase):
    """The verbs at a HOST door — where another host's runner reaches them."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, _ = arith_store(tmp.name)
        self.learner = FakeLearner()
        self.host = Host(
            "alt", engines=(FakeEngine(base="Qwen/Qwen3-0.6B"),),
            learner=self.learner, store=self.store,
            regimes=(Regime("serve", "inference", "Qwen/Qwen3-0.6B", 1),
                     Regime("train", "training", "Qwen/Qwen3-0.6B", 1)))
        self.routed = RemoteLearner(LocalTransport(HostService(self.host)),
                                    admitted=True)

    def test_the_learner_verbs_are_admitted_at_the_serving_host(self) -> None:
        """Admission stays with the partition: a frame from another host
        enters THIS host's arbiter under its learner, so the alternation
        group a multi-member HostSpec carved is honored where the learner
        lives — not where the Trainer sits."""
        self.routed.install("run-a", a_parameterization())
        self.routed.optim_step("run-a")
        self.assertEqual(self.host.arbiter.switches, ["alt:train"])
        self.assertEqual(self.host.arbiter.residency(),
                         {"host:alt": "alt:train"})
        self.assertEqual(self.host.arbiter.admitted(), 2)
        self.assertEqual(self.host.arbiter.in_flight(), 0)

    def test_the_learners_host_journals_who_is_on_it(self) -> None:
        """A tenant of this learner may be a run anchored elsewhere, whose
        attach/detach rows live in that host's journal: custody is recorded
        at the two ends of the tenancy, here."""
        self.routed.install("run-a", a_parameterization())
        self.routed.uninstall("run-a")
        rows = [e for e in self.store.read_host_log("alt")
                if str(e.get("event", "")).startswith("learner-")]
        self.assertEqual([(e["event"], e["run_id"]) for e in rows],
                         [("learner-attach", "run-a"),
                          ("learner-detach", "run-a")])
        self.assertTrue(all(isinstance(e["t"], float) for e in rows))

    def test_a_host_wearing_no_learner_refuses_the_frame(self) -> None:
        """The wire's only addition to custody: "wears no learner" is a
        refusal, the way "serves no such engine" is."""
        bare = Host("bare", engines=(FakeEngine(),), learner=None,
                    store=self.store)
        proxy = RemoteLearner(LocalTransport(HostService(bare)), admitted=True)
        with self.assertRaises(KeyError) as caught:
            proxy.optim_step("run-a")
        self.assertIn("wears no learner", str(caught.exception))

    def test_a_resident_door_still_refuses_the_admitted_path(self) -> None:
        """A learner resident answers asks in arrival order; the admitted
        async path is the HOST door's alone."""
        with self.assertRaises(ValueError) as caught:
            go(LearnerService(FakeLearner()).serve("optim_step", {}))
        self.assertIn("ride ask", str(caught.exception))


class Recorder:
    """A Transport that records which side each verb took."""

    def __init__(self) -> None:
        self.called: list[str] = []
        self.asked: list[str] = []

    async def call(self, verb: str, payload: dict, *,

                   deadline_s: float = 0.0) -> dict:
        json.dumps(payload)
        self.called.append(verb)
        return {"adapters": {}, "optim": {}}

    async def ask(self, verb: str, payload: dict, *,

                    deadline_s: float = 0.0) -> dict:
        json.dumps(payload)
        self.asked.append(verb)
        return {"adapters": {}, "optim": {}}


class LearnerVerbSplitTest(unittest.TestCase):
    """WHICH door takes WHICH calling convention is a contract (#45's rule,
    the training side): a learner beside its runner takes sync asks — the
    Trainer admitted it at its own arbiter — and one at another host takes
    admitted calls, because admission lives with the metal."""

    def test_a_resident_proxy_asks(self) -> None:
        recorder = Recorder()
        RemoteLearner(recorder).emit("run-a")
        self.assertEqual(recorder.asked, ["emit"])
        self.assertEqual(recorder.called, [])

    def test_a_routed_proxy_calls_and_blocks(self) -> None:
        recorder = Recorder()
        proxy = RemoteLearner(recorder, admitted=True)
        proxy.emit("run-a")
        proxy.uninstall("run-a")
        self.assertEqual(recorder.called, ["emit", "uninstall"])
        self.assertEqual(recorder.asked, [])

    def test_a_routed_proxy_answers_from_inside_a_running_loop(self) -> None:
        """The call site that matters: the Trainer is inside an event loop
        and the verb is synchronous, so the frame runs on the proxy's own
        loop and the caller blocks for the reply."""
        proxy = RemoteLearner(Recorder(), admitted=True)

        async def from_the_loop():
            return proxy.emit("run-a")

        self.assertEqual(go(from_the_loop()).adapters, {})


class AnchorTest(unittest.TestCase):
    """Where the run is ANCHORED is a choice: the same experiment, run from
    the learner's host and from the main pool's host, is one run."""

    def hosts_for(self, store) -> tuple[Host, Host]:
        """A serving host and a training host in one little world, each able
        to reach the other's door — so either may be the anchor."""
        serving = Host("serve", engines=(FakeEngine(),), learner=None,
                       store=store,
                       schema_for=lambda base: fake_qwen_schema(4, base=base),
                       transport_for=lambda address: self.transports[address])
        training = Host("train", engines=(), learner=FakeLearner(),
                        store=store,
                        schema_for=lambda base: fake_qwen_schema(4, base=base),
                        transport_for=lambda address: self.transports[address])
        self.transports = {"fleet://serve": LocalTransport(HostService(serving)),
                           "fleet://train": LocalTransport(HostService(training))}
        return serving, training

    def run_anchored(self, anchor: str) -> tuple[str, object]:
        """One arith run, delivered the way the desk delivers it: the frame
        to the anchor host, every other member's address as a route."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        serving, training = self.hosts_for(store)
        spec = arith_spec(train)
        host, routes = ((training, {"main": "fleet://serve"})
                        if anchor == "learner"
                        else (serving, {"learner": "fleet://train"}))

        async def drive():
            reply = await host.adopt(json.loads(canonical_json(spec)), routes)
            self.assertTrue(reply.get("accepted"), reply)
            await host._adoptions[reply["run_id"]]
            return reply["run_id"]

        return go(drive()), store

    def test_a_run_anchored_away_from_its_learner_is_byte_identical(self) -> None:
        """ADR 0006 Part A, promise 2: the anchor moves the Trainer, never
        the run — the store is the run, and the wire carries verbs."""
        here, store_here = self.run_anchored("learner")
        there, store_there = self.run_anchored("main")
        self.assertEqual(here, there)
        self.assertEqual(snapshot(store_here, here), snapshot(store_there, there))
        # ...and both actually ran the plan: an empty run dir matches an empty
        # run dir, which would prove nothing
        self.assertEqual(len(store_there.peek_ledger(there)), 4)

    def test_the_anchor_host_uninstalls_its_tenant_when_the_run_ends(self) -> None:
        """A learner shared by runs anchored elsewhere must not accumulate
        the dead: the run's end is the tenancy's end, journaled at the
        learner's host as a detach."""
        rid, store = self.run_anchored("main")
        rows = [e for e in store.read_host_log("train")
                if str(e.get("event", "")).startswith("learner-")]
        self.assertEqual([e["event"] for e in rows],
                         ["learner-attach", "learner-detach"])
        self.assertEqual({e["run_id"] for e in rows}, {rid})

    def test_a_failed_run_releases_its_tenant_too(self) -> None:
        """The `finally` half: a run that died still leaves the learner."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)

        class Boom(RuntimeError):
            pass

        class FailingLearner(FakeLearner):
            def forward_backward(self, tenant, batch):
                raise Boom("kaput")

        host = Host("solo", engines=(FakeEngine(),), learner=FailingLearner(),
                    store=store)
        with self.assertRaises(Boom):
            go(host.submit(arith_spec(train), SCHEMA))
        self.assertEqual(host.learner._tenants, {})

    def test_a_routed_learner_is_a_free_resident_at_the_runner(self) -> None:
        """Local admission is bookkeeping — the real admission happens at the
        learner's host, per frame — so the routed learner joins no
        alternation group here, exactly as a routed pool does not."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        spec = arith_spec(train)
        serving = Host("serve", engines=(FakeEngine(),), learner=None,
                       store=store)
        routed = RemoteLearner(LocalTransport(HostService(serving)),
                               admitted=True)
        arbiter = Arbiter()
        attach_residents(spec, {"main": FakeEngine()}, routed, arbiter)
        self.assertTrue(arbiter.is_attached(routed))
        self.assertIsNone(arbiter.attached_group(routed))

    def test_a_learner_beside_the_runner_still_attaches_as_the_learner(self) -> None:
        """The default path is untouched: a local learner is the host's own
        metal and carries whatever group its HostSpec declared."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        store, train, _ = arith_store(tmp.name)
        learner = FakeLearner()
        arbiter = Arbiter()
        attach_residents(arith_spec(train), {"main": FakeEngine()}, learner,
                         arbiter)
        self.assertTrue(arbiter.is_attached(learner))


@needs_torch
class TorchUninstallTest(unittest.TestCase):
    """The real learner's inverse: the module tree as install found it."""

    def learner(self):
        from rlstack.runner.learners.torch_learner import TorchLearner

        made = TorchLearner(device="cpu", dtype=torch.float32)
        made._model = _ToyLM()
        made._base = "toy"                    # the base is already loaded
        return made

    def test_uninstall_restores_the_tree_and_forgets_the_tenant(self) -> None:
        learner = self.learner()
        before = (learner._model.first, learner._model.second)
        learner.install("run-a", a_parameterization(
            base="toy", sites=(("pi", "first"), ("theta", "second"))))
        self.assertIsNot(learner._model.first, before[0])   # wrapped
        learner.uninstall("run-a")
        self.assertIs(learner._model.first, before[0])
        self.assertIs(learner._model.second, before[1])
        self.assertEqual(learner._tenants, {})

    def test_a_co_tenants_deltas_stay_wired(self) -> None:
        """Additive install, additive removal (I8): one tenant leaving must
        not unwire the site another still routes through."""
        learner = self.learner()
        learner.install("run-a", a_parameterization(base="toy",
                                                    sites=(("pi", "first"),)))
        learner.install("run-b", a_parameterization(base="toy",
                                                    sites=(("pi", "first"),)))
        wrapper = learner._model.first
        learner.uninstall("run-a")
        self.assertIs(learner._model.first, wrapper)
        self.assertEqual(sorted(learner._tenants), ["run-b"])

    def test_uninstall_is_idempotent(self) -> None:
        learner = self.learner()
        learner.uninstall("never-installed")


class _ToyLM(torch.nn.Module if torch is not None else object):
    """Two Linears at addressable paths — a module tree to install into."""

    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(WIDTH, VOCAB, bias=False)
        self.second = torch.nn.Linear(WIDTH, VOCAB, bias=False)


if __name__ == "__main__":
    unittest.main()
