"""Parity certificates: a mechanism is never trusted on assumption (I7).

An engine's inventory says a build CLAIMS a lever; a certificate says the
lever, on this build, numerically agrees with the kind's replay lowering. The
key is the point — it includes the build fingerprint, so a version bump, a
different attention backend or a quantization flip misses the cache and parity
re-runs before any wave is sampled.

DESIGNED AND UNWIRED: nothing calls these. The parity mechanism actually
running is the per-update logprob_gap rail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rlstack import Mechanism


@dataclass(frozen=True)
class CertificateKey:
    """What a parity result is valid FOR — change any field, re-certify."""

    build_fingerprint: str
    base: str
    kind: str                     # the registered kind AdapterSpec.kind names
    mechanism: Mechanism


@dataclass(frozen=True)
class Certificate:
    key: CertificateKey
    passed: bool
    detail: str                   # once wired: max gap, tolerances, seeds


class CertificateCache(ABC):
    @abstractmethod
    def get(self, key: CertificateKey) -> Certificate | None: ...

    @abstractmethod
    def put(self, certificate: Certificate) -> None: ...


class InMemoryCertificates(CertificateCache):
    def __init__(self) -> None:
        self._held: dict[CertificateKey, Certificate] = {}

    def get(self, key: CertificateKey) -> Certificate | None:
        return self._held.get(key)

    def put(self, certificate: Certificate) -> None:
        self._held[certificate.key] = certificate
