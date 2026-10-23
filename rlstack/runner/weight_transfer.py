"""Publication policy, separate from checkpoint cadence and adapter type."""

from dataclasses import asdict, dataclass
import math
from typing import Literal


@dataclass(frozen=True)
class WeightTransfer:
    backend: Literal["base64", "nccl"] = "base64"
    timeout_s: float = 60.0
    # Address of the inference resident's TCP rendezvous, reachable from the
    # learner. The default is for same-node residents; remote builds set it.
    rendezvous_host: str = "127.0.0.1"

    def __post_init__(self):
        if self.backend not in ("base64", "nccl"):
            raise ValueError(f"unknown weight-transfer backend {self.backend!r}")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("weight-transfer timeout must be finite and positive")
        if not self.rendezvous_host:
            raise ValueError("NCCL rendezvous needs an advertised host")

    def row(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row) -> "WeightTransfer":
        return cls(**row) if row is not None else cls()
