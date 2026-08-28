"""Fakes for the plugin lifecycle: the contract is executable and testable before
any engine exists — the same methodology as runner/fakes.py.
"""

from __future__ import annotations

from rlstack_engine.plugin import EngineBuild, EnginePlugin, Seam


class FakeSeam(Seam):
    """Records claims; refuses a double claim like the real seam would."""

    def __init__(self) -> None:
        self.claimed: list[EnginePlugin] = []

    def claim(self, plugin: EnginePlugin) -> None:
        if any(type(p) is type(plugin) for p in self.claimed):
            raise RuntimeError(f"{type(plugin).__name__} already claimed this seam")
        self.claimed.append(plugin)


def fake_build(*symbols: str, backend: str = "FLASH_ATTN",
               fingerprint: str = "build:fake0001") -> EngineBuild:
    """An EngineBuild whose seam inventory is exactly `symbols`."""
    return EngineBuild(fingerprint=fingerprint, attention_backend=backend,
                       symbols=frozenset(symbols))
