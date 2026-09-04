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
    IN_PROCESS, Address, LocalTransport, parse_address, serve_in_process,
    stop_serving_in_process, transport_for,
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

    def test_the_switchboard_is_empty_between_tests(self) -> None:
        """A service published here is unpublished when it stops standing —
        a decarve, a release — so a stale address never routes."""
        serve_in_process("local://echo", Echo())
        stop_serving_in_process("local://echo")
        self.assertNotIn("local://echo", IN_PROCESS)


if __name__ == "__main__":
    unittest.main()
