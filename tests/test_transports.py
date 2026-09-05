"""The address grammar and the one transport factory (ADR 0007, Q1 / Q3).

The claims under test: an address carries its venue and parses to a typed
record; the factory reaches an in-process service without a wire at all; a
`local://` nobody serves and an unknown scheme are each refused BY NAME; and
the Modal branch is imported INSIDE the branch — so this file, and the suite
it belongs to, never needs the Modal SDK installed.
"""

from __future__ import annotations

import sys
import types
import unittest

from rlstack.runner.remote import (
    EPOCH_KEY, IN_PROCESS, Address, LocalTransport, WrongEpoch, check_epoch,
    parse_address, serve_in_process, stop_serving_in_process, transport_for,
    with_epoch,
)


class Echo:
    """A service that answers by repeating the frame — the smallest thing a
    transport can carry frames to."""

    async def serve(self, verb: str, payload: dict) -> dict:
        return {"served": verb, **payload}

    def answer(self, verb: str, payload: dict) -> dict:
        return {"answered": verb, **payload}


class GrammarTest(unittest.TestCase):
    def test_a_modal_address_carries_app_class_and_host(self) -> None:
        """Q3: the venue is IN the address, which is what lets one desk
        command metal deployed in several apps."""
        self.assertEqual(
            parse_address("modal://rlstack-steer-l4/MetalS#steer-l4:0.main.c1"),
            Address(scheme="modal", app="rlstack-steer-l4", cls="MetalS",
                    host="steer-l4:0.main.c1"))

    def test_a_class_without_a_host_is_that_containers_own_plane(self) -> None:
        self.assertEqual(parse_address("modal://rlstack-desk/Desk"),
                         Address(scheme="modal", app="rlstack-desk",
                                 cls="Desk", host=""))

    def test_a_local_address_is_a_name_in_this_process(self) -> None:
        self.assertEqual(parse_address("local://carved-1"),
                         Address(scheme="local", host="carved-1"))

    def test_an_address_may_name_one_instance(self) -> None:
        """ADR 0008, Q2: the epoch is one optional segment on the grammar the
        desk already keeps, so a journaled address says WHICH LIFE of a
        container it was written about."""
        self.assertEqual(
            parse_address("modal://rlstack-concept-steer/MetalS#c:0.main.c1@e7"),
            Address(scheme="modal", app="rlstack-concept-steer", cls="MetalS",
                    host="c:0.main.c1", epoch="e7"))
        self.assertEqual(parse_address("local://carved-1@e7"),
                         Address(scheme="local", host="carved-1", epoch="e7"))
        self.assertEqual(parse_address("local://carved-1").epoch, "")

    def test_an_epoch_is_composed_onto_an_address_and_never_into_it(self) -> None:
        """A metal's PLANE address stays epoch-free on the row — a
        re-registration must read as one metal turning over, not two deploys
        colliding on a name — so the desk composes the two where it builds
        the remote."""
        self.assertEqual(with_epoch("modal://app/MetalS", "e7"),
                         "modal://app/MetalS@e7")
        self.assertEqual(with_epoch("modal://app/MetalS", ""),
                         "modal://app/MetalS")

    def test_a_half_address_is_refused_by_name(self) -> None:
        """A modal address that names no class is a venue bug, and the frame
        that would ride it is a lost hour — so it never gets built."""
        with self.assertRaises(ValueError) as caught:
            parse_address("modal://rlstack-desk")
        self.assertIn("modal://<app>/<cls>", str(caught.exception))
        with self.assertRaises(ValueError):
            parse_address("just-a-name")


class FactoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(stop_serving_in_process, "local://echo")

    def test_an_address_served_here_never_touches_a_wire(self) -> None:
        """#77's rule, said once in the factory: what answers in THIS process
        is answered in-process — a Modal self-call would wait on the loop it
        is itself blocking."""
        serve_in_process("local://echo", Echo())
        transport = transport_for("local://echo")
        self.assertIsInstance(transport, LocalTransport)
        self.assertEqual(transport.ask("who", {"x": 1}),
                         {"answered": "who", "x": 1})

    def test_a_modal_address_served_here_is_still_local(self) -> None:
        """A carved host's address is its venue address, and inside its own
        container it is answered locally — the rule is about the PROCESS, not
        about the scheme."""
        address = "modal://rlstack-steer-l4/MetalS#carved-1"
        serve_in_process(address, Echo())
        self.addCleanup(stop_serving_in_process, address)
        self.assertIsInstance(transport_for(address), LocalTransport)

    def test_an_unserved_local_address_is_refused_by_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            transport_for("local://nobody-home")
        self.assertIn("nothing answers", str(caught.exception))

    def test_an_unknown_scheme_is_refused_by_name(self) -> None:
        with self.assertRaises(ValueError) as caught:
            transport_for("aws://some-lambda")
        self.assertIn("aws", str(caught.exception))
        self.assertIn("transports", str(caught.exception))

    def test_the_modal_branch_imports_modal_only_when_taken(self) -> None:
        """Rule 7 from the caller's side: the Modal SDK is reached only by
        the branch that names it. The suite is stdlib-only, so the SDK is
        stood in for here — and the transport built carries the address's app,
        class and host, which is the whole of what it needs to know."""
        self.assertNotIn("rlstack.runner.transports.modal_cls", sys.modules)
        stub = types.ModuleType("modal")
        stub.Cls = None                    # never resolved: nothing is called
        sys.modules["modal"] = stub
        self.addCleanup(sys.modules.pop, "modal", None)
        self.addCleanup(sys.modules.pop,
                        "rlstack.runner.transports.modal_cls", None)

        transport = transport_for("modal://rlstack-desk/Desk#a-host")
        self.assertEqual(
            (transport.app, transport.cls, transport.host),
            ("rlstack-desk", "Desk", "a-host"))

    def test_a_transport_built_from_an_epoch_stamps_every_frame(self) -> None:
        """F2's mechanism, said once: the epoch travels IN the payload, under
        a reserved key no verb's own argument may be, because every rlstack
        door has the same three-argument shape."""
        serve_in_process("local://echo", Echo())
        transport = transport_for("local://echo")
        self.assertNotIn(EPOCH_KEY, transport.ask("who", {}))

        address = "local://echo@e7"
        serve_in_process(address, Echo())
        self.addCleanup(stop_serving_in_process, address)
        self.assertEqual(transport_for(address).ask("who", {})[EPOCH_KEY], "e7")

    def test_the_modal_branch_carries_the_epoch_too(self) -> None:
        stub = types.ModuleType("modal")
        stub.Cls = None
        sys.modules["modal"] = stub
        self.addCleanup(sys.modules.pop, "modal", None)
        self.addCleanup(sys.modules.pop,
                        "rlstack.runner.transports.modal_cls", None)
        transport = transport_for("modal://rlstack-desk/Desk#a-host@e7")
        self.assertEqual((transport.host, transport.epoch), ("a-host", "e7"))

    def test_the_switchboard_is_empty_between_tests(self) -> None:
        """A service published here is unpublished when it stops standing —
        a decarve, a release — so a stale address never routes."""
        serve_in_process("local://echo", Echo())
        stop_serving_in_process("local://echo")
        self.assertNotIn("local://echo", IN_PROCESS)


if __name__ == "__main__":
    unittest.main()


class EpochRefusalTest(unittest.TestCase):
    """ADR 0008, F2 — the refusal itself, as a rule with two deliberate
    silences: a frame naming no epoch is served (a registration is exactly
    that frame), and a receiver wearing no epoch serves anything (a
    hand-built host in a test process is one instance forever)."""

    def test_a_mismatch_is_refused_and_names_both_lives(self) -> None:
        with self.assertRaises(WrongEpoch) as caught:
            check_epoch({EPOCH_KEY: "old"}, "new", "metal 'concept-a100'")
        said = str(caught.exception)
        self.assertIn("concept-a100", said)
        self.assertIn("'old'", said)
        self.assertIn("'new'", said)

    def test_an_unaddressed_frame_and_an_epochless_receiver_are_served(self) -> None:
        check_epoch({}, "new", "metal 'x'")
        check_epoch({EPOCH_KEY: "old"}, "", "a hand-built host")
        check_epoch({EPOCH_KEY: "same"}, "same", "metal 'x'")
