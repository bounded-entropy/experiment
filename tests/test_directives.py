"""The per-request directive (ADR 0004, Q2): a typed instruction from the
caller of `sample` / `score` to one adapter type, carried to the engine,
recorded at the seal, and readable at replay — with no GPU in the loop.

Every seam the directive crosses is pinned here: the adapter type picks its
own by type and refuses two; the wire encodes by adapter type name and the
registered class decodes its own record; the pool client hands it to the
engine; the engine records what the adapter type says it did (the real
adapter type's rule, run by the fake bus) into the FinishEvent, and the Turn
seals it. The two new levers ride the same fold: extra args JOIN, and one
request carries ONE cache salt.
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass

from rlstack import (
    Bundle, EnginePoolClient, FakeEngine, Message, Role, SamplingSpec,
)
from rlstack.policy.adapters.base import (
    AdapterType, Directive, Mechanism, adapter_type,
)
from rlstack.policy.adapters.rollout import Levers, Request
from rlstack.runner.remote import (
    EngineService, LocalTransport, RemotePool, decode_directives,
    encode_directives, json_roundtrip,
)


@dataclass(frozen=True)
class _Nudge(Directive):
    """A window, the shape the steer's directive has."""

    adapter_type = "val_directed"
    start: int = 0
    end: int | None = None


@adapter_type("val_directed")
class _Directed(AdapterType):
    """An adapter type that accepts a directive and RECORDS the window it
    resolved — default included, offset by what sits in front."""

    serving = Mechanism.LOGITS
    records = ("nudge_window",)
    directive = _Nudge

    def site_ok(self, meta) -> bool:
        return not meta.has_weight

    def record_directive(self, directive, request):
        window = _Nudge() if directive is None else directive
        end = None if window.end is None else window.end + request.occupied
        return {"nudge_window": [window.start + request.occupied, end]}


def go(coro):
    return asyncio.run(coro)


BUNDLE = Bundle("bundle:d", {"pi": 0}, payloads={"pi": b"\x00"},
                adapter_types={"pi": "val_directed"})
PROMPT = (Message(Role.USER, "2+2"),)


class PickTest(unittest.TestCase):
    def test_an_adapter_type_picks_its_own_directive(self) -> None:
        request = Request((1, 2), directives=(_Nudge(start=1),))
        self.assertEqual(_Directed().directive_for(request), _Nudge(start=1))

    def test_none_when_the_request_carries_none_for_it(self) -> None:
        self.assertIsNone(_Directed().directive_for(Request((1,))))

    def test_two_for_one_adapter_type_are_refused(self) -> None:
        request = Request((1,), directives=(_Nudge(), _Nudge(start=3)))
        with self.assertRaises(ValueError):
            _Directed().directive_for(request)

    def test_an_adapter_type_without_a_directive_sees_none(self) -> None:
        class _Plain(AdapterType):
            pass
        self.assertIsNone(_Plain().directive_for(
            Request((1,), directives=(_Nudge(),))))


class WireTest(unittest.TestCase):
    def test_directives_cross_by_adapter_type_name(self) -> None:
        rows = json_roundtrip({"d": encode_directives((_Nudge(2, 5),))})["d"]
        self.assertEqual(rows, [{"adapter_type": "val_directed",
                                 "start": 2, "end": 5}])
        self.assertEqual(decode_directives(rows), (_Nudge(2, 5),))

    def test_an_adapter_type_accepting_none_refuses_one(self) -> None:
        with self.assertRaises(ValueError):
            decode_directives([{"adapter_type": "lora", "start": 0}])

    def test_an_unregistered_adapter_type_is_refused(self) -> None:
        with self.assertRaises(KeyError):
            decode_directives([{"adapter_type": "never", "start": 0}])


class RoundTripTest(unittest.TestCase):
    """directive -> engine -> FinishEvent -> Turn.turn_extras."""

    def client(self, engine) -> EnginePoolClient:
        engine.add_bundle(BUNDLE)
        return EnginePoolClient({"main": (engine, BUNDLE)}, SamplingSpec(),
                                episode_seed=3)

    def test_the_engine_sees_it_and_the_turn_seals_its_record(self) -> None:
        engine = FakeEngine()
        turn = go(self.client(engine).sample(PROMPT,
                                             directives=(_Nudge(start=2),)))
        self.assertEqual(engine.directives_seen, [(_Nudge(start=2),)])
        self.assertEqual(turn.turn_extras["nudge_window"], [2, None])

    def test_no_directive_records_the_adapter_types_default(self) -> None:
        turn = go(self.client(FakeEngine()).sample(PROMPT))
        self.assertEqual(turn.turn_extras["nudge_window"], [0, None])

    def test_score_carries_one_too_and_seals_nothing(self) -> None:
        engine = FakeEngine()
        scores = go(self.client(engine).score(PROMPT, (52,),
                                              directives=(_Nudge(1, 2),)))
        self.assertEqual(len(scores), 1)
        self.assertEqual(engine.directives_seen, [(_Nudge(1, 2),)])

    def test_a_bundle_carrying_no_such_adapter_type_records_nothing(self) -> None:
        engine = FakeEngine()
        plain = Bundle("bundle:p", {"pi": 0}, payloads={"pi": b"\x00"},
                       adapter_types={"pi": "lora"})
        engine.add_bundle(plain)
        client = EnginePoolClient({"main": (engine, plain)}, SamplingSpec(),
                                  episode_seed=3)
        turn = go(client.sample(PROMPT, directives=(_Nudge(start=2),)))
        self.assertNotIn("nudge_window", turn.turn_extras)

    def test_the_same_turn_seals_over_the_wire(self) -> None:
        local, served = FakeEngine(), FakeEngine()
        served.add_bundle(BUNDLE)
        remote = RemotePool(LocalTransport(EngineService(served)))
        remote.add_bundle(BUNDLE)
        here = go(self.client(local).sample(PROMPT, directives=(_Nudge(1, 4),)))
        there = go(EnginePoolClient({"main": (remote, BUNDLE)}, SamplingSpec(),
                                    episode_seed=3)
                   .sample(PROMPT, directives=(_Nudge(1, 4),)))
        self.assertEqual(there, here)
        self.assertEqual(there.turn_extras["nudge_window"], [1, 4])


class LeverFoldTest(unittest.TestCase):
    def test_extra_args_join_like_keywords(self) -> None:
        merged = (Levers(extra_args={"a": 1})
                  .merged_with(Levers(extra_args={"b": 2})))
        self.assertEqual(merged.extra_args, {"a": 1, "b": 2})

    def test_one_salt_survives_and_none_is_transparent(self) -> None:
        salted = Levers().merged_with(Levers(cache_salt="bundle:a/0:"))
        self.assertEqual(salted.cache_salt, "bundle:a/0:")
        self.assertEqual(salted.merged_with(Levers()).cache_salt, "bundle:a/0:")
        self.assertEqual(
            salted.merged_with(Levers(cache_salt="bundle:a/0:")).cache_salt,
            "bundle:a/0:")

    def test_two_different_salts_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            Levers(cache_salt="x").merged_with(Levers(cache_salt="y"))

    def test_a_request_defaults_to_nothing_in_front_and_no_directives(self) -> None:
        request = Request((1, 2, 3))
        self.assertEqual((request.occupied, request.directives), (0, ()))


if __name__ == "__main__":
    unittest.main()
