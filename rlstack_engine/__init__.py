"""rlstack_engine — code that ships in the ENGINE image, not in the client.

The package an adapter type's rollout lowering names by string when its
mechanism is a
PLUGIN: the plugin contract, the typed view a plugin sees of a batch, slot
bookkeeping, and the certificate vocabulary. Its charter is narrow on purpose —
only code that patches engine internals belongs here. The import direction is
one-way and enforced (tests/test_architecture.py): this package may import
rlstack types; rlstack refers to plugins by string and never imports back.
"""

from rlstack_engine.plugin import EngineBuild, EnginePlugin, ProbeError, Seam
from rlstack_engine.batch_view import BatchView
from rlstack_engine.slots import SlotTable, SlotsFull
from rlstack_engine.side_attention import SideAttention
from rlstack_engine.certificates import (
    Certificate, CertificateCache, CertificateKey, InMemoryCertificates,
)
from rlstack_engine.fakes import FakeSeam, fake_build
