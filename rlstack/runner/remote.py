"""The wire: pools served by ANOTHER host, behind the same Engine protocol.

HostService is the host-side end — it executes pool verbs on its own metal
under its own arbiter, because admission stays with the partition, and engines
are addressed by CAPABILITY (base, tp), never by pool name. A Transport carries
JSON-safe dict frames; RemotePool implements the whole Engine protocol over it,
so the runner cannot tell remote from local. The verb split is the contract
every transport implements against:

    call (async)   sample_tokens / score_tokens — they occupy the metal, so
                   the service wraps each in the owning host's arbiter.admit.
    ask  (sync)    add_bundle / reachability / tokenize — additive
                   registration and build facts, which by the tenancy
                   invariant never disturb traffic, so they need no admission
                   and may run from sync call sites.

Costs, stated: sample replies are non-streamed (one reply carries the whole
event list), add_bundle ships payload bytes as base64, and the learner is never
remote. LocalTransport round-trips every frame through json in both directions,
so anything that works over it works over a real transport.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

from rlstack.data.trajectory import Message, Role
from rlstack.policy.adapters.base import Mechanism
from rlstack.policy.compile import Bundle
from rlstack.policy.siteschema import SiteMeta
from rlstack.runner.interfaces import Engine, FinishEvent, TokenEvent
from rlstack.runner.meters import TrafficMeter
from rlstack.spec.specs import SamplingSpec

if TYPE_CHECKING:
    from rlstack.runner.host import Host


# ---------------------------------------------------------------------------
# codecs — every record that crosses the wire, encoded to JSON-safe dicts
# ---------------------------------------------------------------------------

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def encode_messages(messages: Sequence[Message]) -> list[dict]:
    return [{"role": m.role.value, "content": m.content} for m in messages]


def decode_messages(rows: Sequence[dict]) -> tuple[Message, ...]:
    return tuple(Message(Role(r["role"]), r["content"]) for r in rows)


def encode_sampling(sampling: SamplingSpec) -> dict:
    return {"temperature": sampling.temperature, "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens}


def decode_sampling(row: dict) -> SamplingSpec:
    return SamplingSpec(**row)


def encode_events(events: Sequence[TokenEvent | FinishEvent]) -> list[dict]:
    """Extras/turn_extras must already be JSON-safe — the same discipline the
    membrane demands of them at seal time, so the wire adds no new rule."""
    out: list[dict] = []
    for event in events:
        if isinstance(event, TokenEvent):
            out.append({"kind": "token", "token_id": event.token_id,
                        "logprob": event.logprob,
                        "text_delta": event.text_delta,
                        "extras": dict(event.extras)})
        else:
            out.append({"kind": "finish", "finish": event.finish,
                        "stop_hit": event.stop_hit,
                        "turn_extras": dict(event.turn_extras)})
    return out


def decode_events(rows: Sequence[dict]) -> tuple[TokenEvent | FinishEvent, ...]:
    return tuple(
        TokenEvent(token_id=r["token_id"], logprob=r["logprob"],
                   text_delta=r["text_delta"], extras=r["extras"])
        if r["kind"] == "token"
        else FinishEvent(finish=r["finish"], stop_hit=r["stop_hit"],
                         turn_extras=r["turn_extras"])
        for r in rows)


def encode_bundle(bundle: Bundle) -> dict:
    return {"bundle_id": bundle.bundle_id,
            "policy_version": dict(bundle.policy_version),
            "payloads": {name: _b64(data)
                         for name, data in bundle.payloads.items()},
            "adapter_types": dict(bundle.adapter_types)}


def decode_bundle(row: dict) -> Bundle:
    return Bundle(bundle_id=row["bundle_id"],
                  policy_version=row["policy_version"],
                  payloads={name: _unb64(text)
                            for name, text in row["payloads"].items()},
                  adapter_types=row["adapter_types"])


def encode_sites(sites: Sequence[SiteMeta]) -> list[dict]:
    return [{"name": s.name, "path": s.path, "has_weight": s.has_weight,
             "shape": list(s.shape) if s.shape is not None else None,
             "is_boundary": s.is_boundary} for s in sites]


def decode_sites(rows: Sequence[dict]) -> tuple[SiteMeta, ...]:
    return tuple(
        SiteMeta(name=r["name"], path=r["path"], has_weight=r["has_weight"],
                 shape=tuple(r["shape"]) if r["shape"] is not None else None,
                 is_boundary=r["is_boundary"]) for r in rows)


# ---------------------------------------------------------------------------
# the transport
# ---------------------------------------------------------------------------

class Transport(Protocol):
    """Carries dict frames to one host's service. Frames are JSON-safe by
    contract; LocalTransport enforces it, real transports inherit it free."""

    async def call(self, verb: str, payload: dict) -> dict:
        """An admitted verb: the serving host wraps it in its arbiter."""
        ...

    def ask(self, verb: str, payload: dict) -> dict:
        """An admission-free verb: registration and build facts."""
        ...


class HostService:
    """The host-side end of the wire.

    Executes pool verbs on this host's engines under this host's arbiter — the
    physical resource owns admission, so a remote experiment is just one more
    source of admitted work and never gets a vote. Engines are addressed by
    CAPABILITY (base, tp), the fleet's demand vocabulary, never by pool name —
    pool names are an experiment's private routing."""

    def __init__(self, host: Host) -> None:
        self.host = host

    def describe(self) -> dict:
        """What this host serves — the advertisement the fleet matches on."""
        return {"host": self.host.name,
                "serves": [{"base": engine.base, "tp": engine.tp}
                           for engine in self.host.engines]}

    def _engine(self, base: str | None, tp: int) -> Engine:
        """Resolution delegates to the host's own shape-matched lookup
        (Host.engine_for — the one home of the rule); the wire's only
        addition is turning "not served" into a refusal."""
        engine = self.host.engine_for(base, tp)
        if engine is None:
            raise KeyError(
                f"host {self.host.name!r} serves no ({base!r}, tp={tp}); it "
                f"serves {[(e.base, e.tp) for e in self.host.engines]}")
        return engine

    async def serve(self, verb: str, payload: dict) -> dict:
        """One admitted verb, admission included: enter the owning host's
        arbiter, run, leave. A regime-host's engines are attached at birth; a
        bare host's attach here on first remote use, at zero footprint — a
        joiner never re-counts a fraction the partition already owns."""
        engine = self._engine(payload["base"], payload["tp"])
        if not self.host.arbiter.is_attached(engine):
            self.host.arbiter.attach(
                engine, label=f"{self.host.name}:{payload['base'] or '*'}")
        if verb not in ("sample_tokens", "score_tokens"):
            raise ValueError(f"unknown admitted verb {verb!r}")
        async with self.host.arbiter.admit(engine):
            if verb == "sample_tokens":
                events = [event async for event in engine.sample_tokens(
                    decode_messages(payload["messages"]),
                    decode_sampling(payload["sampling"]),
                    tuple(payload["stop"]), payload["bundle_id"],
                    payload["seed"])]
                return {"events": encode_events(events)}
            scores = await engine.score_tokens(
                decode_messages(payload["messages"]),
                tuple(payload["token_ids"]), payload["bundle_id"])
            return {"logprobs": list(scores)}

    def answer(self, verb: str, payload: dict) -> dict:
        """One admission-free verb: additive registration (add_bundle never
        disturbs traffic — the multi-tenancy invariant) and build facts
        (reachability, tokenize), all callable from sync call sites."""
        engine = self._engine(payload["base"], payload["tp"])
        if verb == "add_bundle":
            engine.add_bundle(decode_bundle(payload["bundle"]))
            return {}
        if verb == "reachability":
            reach = engine.reachability(decode_sites(payload["sites"]))
            return {"mechanisms": {name: mech.name
                                   for name, mech in reach.items()}}
        if verb == "tokenize":
            return {"token_ids": list(engine.tokenize(payload["text"]))}
        raise ValueError(f"unknown admission-free verb {verb!r}")


def _json_roundtrip(frame: dict) -> dict:
    """The honesty gate: a frame that survives this survives any real wire."""
    return json.loads(json.dumps(frame))


class LocalTransport:
    """Same-process transport that still crosses the serialization boundary
    (json round-trip both ways), so a fleet whose hosts share one process is
    indistinguishable, from above, from one whose hosts do not."""

    def __init__(self, service: HostService) -> None:
        self.service = service

    async def call(self, verb: str, payload: dict) -> dict:
        return _json_roundtrip(
            await self.service.serve(verb, _json_roundtrip(payload)))

    def ask(self, verb: str, payload: dict) -> dict:
        return _json_roundtrip(
            self.service.answer(verb, _json_roundtrip(payload)))


# ---------------------------------------------------------------------------
# the client end
# ---------------------------------------------------------------------------

class RemotePool:
    """An Engine served by another host, reached through a Transport.

    Implements the full Engine protocol; base and tp are this pool's
    CAPABILITY ADDRESS on the serving host (the fleet's demand vocabulary),
    and double as the build facts the submit gate attests — the serving host
    refuses an address it does not serve, so the attestation is real.

    TRAFFIC IS COUNTED WHERE IT IS SERVED, never here: every verb below lands
    in HostService, which admits it through the SERVING host's arbiter and
    runs it on that host's engine — both already wired to that host's meter.
    A client-side count would attribute another partition's load to this one,
    so this meter exists only to satisfy the protocol and stays at zero."""

    def __init__(self, transport: Transport, *, base: str | None = None,
                 tp: int = 1) -> None:
        self.base = base
        self.tp = tp
        self.meter = TrafficMeter()
        self._transport = transport

    def _address(self) -> dict:
        return {"base": self.base, "tp": self.tp}

    async def sample_tokens(self, messages, sampling, stop, bundle_id,
                            seed) -> AsyncIterator[TokenEvent | FinishEvent]:
        reply = await self._transport.call("sample_tokens", {
            **self._address(), "messages": encode_messages(messages),
            "sampling": encode_sampling(sampling), "stop": list(stop),
            "bundle_id": bundle_id, "seed": seed})
        for event in decode_events(reply["events"]):
            yield event

    async def score_tokens(self, messages, token_ids,
                           bundle_id) -> tuple[float, ...]:
        reply = await self._transport.call("score_tokens", {
            **self._address(), "messages": encode_messages(messages),
            "token_ids": list(token_ids), "bundle_id": bundle_id})
        return tuple(reply["logprobs"])

    def add_bundle(self, bundle: Bundle) -> None:
        self._transport.ask("add_bundle", {
            **self._address(), "bundle": encode_bundle(bundle)})

    def reachability(self, sites: Sequence[SiteMeta]) -> Mapping[str, Mechanism]:
        reply = self._transport.ask("reachability", {
            **self._address(), "sites": encode_sites(sites)})
        return {name: Mechanism[mech]
                for name, mech in reply["mechanisms"].items()}

    def tokenize(self, text: str) -> tuple[int, ...]:
        reply = self._transport.ask("tokenize", {
            **self._address(), "text": text})
        return tuple(reply["token_ids"])
