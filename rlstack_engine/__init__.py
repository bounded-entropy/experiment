"""rlstack_engine — code that ships in the ENGINE image, not in the client.

This is the package `Adapter.engine_plugin` strings name. It holds the plugin
contract (one serving mechanism the stock engine lacks = one EnginePlugin),
the typed view plugins see of a batch, slot bookkeeping, and the certificate
vocabulary. Import direction is one-way: this package may import rlstack
types; rlstack refers to plugins by string only and never imports back
(tests/test_architecture.py enforces it).

Phase B1 ships the contract, the bookkeeping, and fakes; B3 lands the vLLM
seams (BatchView.from_vllm, SideAttention numerics) behind probe().
"""

from rlstack_engine.plugin import EngineBuild, EnginePlugin, ProbeError, Seam
from rlstack_engine.batch_view import BatchView
from rlstack_engine.slots import SlotTable, SlotsFull
from rlstack_engine.side_attention import SideAttention
from rlstack_engine.certificates import (
    Certificate, CertificateCache, CertificateKey, InMemoryCertificates,
)
from rlstack_engine.fakes import FakeSeam, fake_build
