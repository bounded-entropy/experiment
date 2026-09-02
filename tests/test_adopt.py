"""Adoption: the submission door (Host.adopt, spec_from_json, transport_for).

The claims under test: canonical JSON decodes back to the SAME spec value (one
identity on both ends of the wire); an adopted run is byte-identical to a
submitted one (the door adds transport, never semantics); re-adoption is
resume; the wire path (HostService + LocalTransport + RemoteHost) carries all
of it JSON-safely; and every refusal — no schema_for, no transport_for, an unbindable
pool — comes back in the reply instead of detonating in a background task.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from common import arith_spec, arith_store
from rlstack import (
    FakeEngine, FakeLearner, GpuConfig, HostSpec, Host, fake_qwen_schema,
    learner, plora, pool,
)
from rlstack.runner.remote import (
    HostService, LocalTransport, RemoteHost, spec_from_json,
)
from rlstack.spec.canonical import canonical_json
from rlstack.spec.specs import Seeds, WarmStart

SCHEMA = fake_qwen_schema(4, base="Qwen/Qwen3-0.6B")


def go(coro):
    return asyncio.run(coro)


def row_of(spec) -> dict:
    """The wire form: what an adopt frame carries."""
    return json.loads(canonical_json(spec))


class SpecRoundtripTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)

    def test_canonical_json_roundtrips_to_the_same_spec(self) -> None:
        """The decode law: one spec value on both ends of the wire, and
        therefore one identity."""
        spec = arith_spec(self.train, self.heldout)
        self.assertEqual(spec_from_json(row_of(spec)), spec)

    def test_roundtrip_covers_the_whole_field_zoo(self) -> None:
        """Tuples re-tuple, optionals survive, mixed-type init mappings pass
        through, a warm start rides along — the maximal spec is the test."""
        spec = arith_spec(
            self.train, self.heldout,
            policy=arith_spec(self.train).policy.__class__(
                base="Qwen/Qwen3-0.6B",
                bank={"pi": plora("layers.0-3.self_attn.*", k=4, latent=8,
                                  members=2, factors="cas://feedbeef")}),
            init=WarmStart(policy="cas://cafe", optim="fresh",
                           map={"theirs": "pi"}),
            seeds=Seeds(master=23))
        decoded = spec_from_json(row_of(spec))
        self.assertEqual(decoded, spec)
        self.assertIsInstance(decoded.gen.envs, tuple)
        self.assertIsInstance(decoded.gpu_config.hosts, tuple)
        self.assertIsInstance(decoded.gpu_config.hosts[0].members, tuple)

    def test_an_unknown_tag_is_refused(self) -> None:
        with self.assertRaises(TypeError):
            spec_from_json({"__type__": "NotASpec"})


class AdoptTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store, self.train, self.heldout = arith_store(tmp.name)

    def host(self, **kwargs) -> Host:
        defaults = dict(engines=(FakeEngine(),), learner=FakeLearner(),
                        store=self.store,
                        schema_for=lambda base: fake_qwen_schema(4, base=base))
        defaults.update(kwargs)
        return Host("door-host", **defaults)

    def adopt_and_finish(self, host: Host, spec, routes=None) -> dict:
        """Adopt, then await the background tenancy — one loop for both, the
        way a standing host's own loop holds both."""
        async def drive():
            reply = await host.adopt(row_of(spec), routes)
            if reply.get("run_id") in host._adoptions:
                await host._adoptions[reply["run_id"]]
            return reply
        return go(drive())

    def test_an_adopted_run_is_byte_identical_to_a_submitted_one(self) -> None:
        """The door adds transport, never semantics."""
        spec = arith_spec(self.train)
        reply = self.adopt_and_finish(self.host(), spec)
        self.assertTrue(reply["accepted"])
        self.assertEqual(reply["state"], "adopted")

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        other_store, other_train, _ = arith_store(tmp.name)
        other = Host("plain-host", engines=(FakeEngine(),),
                     learner=FakeLearner(), store=other_store)
        report = go(other.submit(arith_spec(other_train), SCHEMA))
        self.assertEqual(reply["run_id"], report.run_id)
        self.assertEqual(
            self.store.path_of(f"runs/{reply['run_id']}/ledger.jsonl").read_bytes(),
            other_store.path_of(f"runs/{report.run_id}/ledger.jsonl").read_bytes())

    def test_readoption_is_resume(self) -> None:
        """The same spec at the door twice: the second pass attaches, finds
        every update committed, and the ledger bytes never change."""
        host = self.host()
        spec = arith_spec(self.train)
        first = self.adopt_and_finish(host, spec)
        ledger = self.store.path_of(
            f"runs/{first['run_id']}/ledger.jsonl").read_bytes()
        second = self.adopt_and_finish(host, spec)
        self.assertEqual(second["run_id"], first["run_id"])
        self.assertEqual(second["state"], "adopted")
        self.assertEqual(
            self.store.path_of(
                f"runs/{first['run_id']}/ledger.jsonl").read_bytes(), ledger)
        self.assertEqual(host.status()["tenants"][first["run_id"]]["status"],
                         "done")

    def test_adoption_over_the_wire(self) -> None:
        """HostService + LocalTransport + RemoteHost: the frame survives json
        both ways, and the client watches its tenancy through status alone."""
        host = self.host()
        remote = RemoteHost(LocalTransport(HostService(host)))
        spec = arith_spec(self.train)

        async def drive():
            reply = await remote.adopt(spec)
            await host._adoptions[reply["run_id"]]
            return reply
        reply = go(drive())
        self.assertTrue(reply["accepted"])
        status = remote.status()
        self.assertEqual(status["tenants"][reply["run_id"]]["status"], "done")

    def test_routes_are_resolved_into_engines(self) -> None:
        """A demanded pool this host does not serve arrives as an ADDRESS and
        leaves as a live engine — resolved, once, at the door."""
        resolved: list[str] = []
        aux = Host("aux-host", engines=(FakeEngine(),), learner=None,
                   store=self.store)

        def transport_for(address: str) -> LocalTransport:
            resolved.append(address)
            return LocalTransport(HostService(aux))

        host = self.host(transport_for=transport_for)
        spec = arith_spec(self.train, gpu_config=GpuConfig(hosts=(
            HostSpec((pool("main"),)), HostSpec((learner(),)),
            HostSpec((pool("aux"),)))))
        reply = self.adopt_and_finish(host, spec, routes={"aux": "fleet://aux"})
        self.assertTrue(reply["accepted"], reply)
        self.assertEqual(resolved, ["fleet://aux"])
        self.assertEqual(host.status()["tenants"][reply["run_id"]]["status"],
                         "done")

    def test_refusals_come_back_in_the_reply(self) -> None:
        """No schema_for, no transport_for, an unbindable pool: each is a refusal at
        the door, never a background detonation."""
        spec = arith_spec(self.train)

        deaf = self.host(schema_for=None)
        reply = go(deaf.adopt(row_of(spec)))
        self.assertFalse(reply["accepted"])
        self.assertIn("schema_for", reply["error"])

        unresolved = self.host()
        reply = go(unresolved.adopt(row_of(spec), {"aux": "fleet://aux"}))
        self.assertFalse(reply["accepted"])
        self.assertIn("transport_for", reply["error"])

        wide = arith_spec(self.train, gpu_config=GpuConfig(hosts=(
            HostSpec((pool("main", tp=2),)), HostSpec((learner(),)))))
        reply = go(self.host().adopt(row_of(wide)))
        self.assertFalse(reply["accepted"])
        self.assertIn("tp=2", reply["error"])


if __name__ == "__main__":
    unittest.main()
