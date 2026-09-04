"""A stand-in for the Modal SDK, so a venue file can be IMPORTED by the suite.

`deploy/*.py` name their app, their images and their containers at module
scope, which means importing one needs `modal` — and the fakes suite is
stdlib-only by rule (STYLE rule 7). So the suite stands the SDK in: every
attribute answers, every call returns something that answers, and every
decorator hands the decorated object back unchanged.

WHAT THIS BUYS. The venue files become testable as Python: their constants,
their topologies, their plans and their specs are ordinary functions, and
`tests/test_venues.py` can compare a rewritten venue's canonical spec rows
against the ones the pre-rewrite file produced (ADR 0007, promise 7 — no run
identity may change). It buys nothing about Modal itself: no container is
built, nothing is deployed, and a venue's behaviour ON metal is exactly as
unproven as it was.
"""

from __future__ import annotations

import contextlib
import sys
import types


class Anything:
    """An object that answers every attribute, call and decoration with
    another one of itself — the whole of the SDK's shape, for import's
    purposes. It records the name it was reached by, which is enough for a
    test to assert an image or an app was named at all."""

    def __init__(self, name: str = "modal") -> None:
        self._name = name

    def __getattr__(self, attr: str) -> "Anything":
        if attr.startswith("__"):
            raise AttributeError(attr)
        return Anything(f"{self._name}.{attr}")

    def __call__(self, *args, **kwargs):
        """A CALL IS EITHER A DECORATION OR A CONSTRUCTION. One positional
        argument that is a class or a function is a decorator being applied
        (`app.function(...)(f)`, `modal.concurrent(64)(C)`): hand it straight
        back, so the venue's own object survives with its own name. Anything
        else builds another stand-in."""
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return Anything(f"{self._name}()")

    def __repr__(self) -> str:
        return f"<stub {self._name}>"


def modal_module() -> types.ModuleType:
    """The `modal` package as one stand-in, plus the `modal.experimental`
    submodule a metal container imports inside its keepalive."""
    module = types.ModuleType("modal")
    module.__getattr__ = lambda attr: Anything(f"modal.{attr}")
    experimental = types.ModuleType("modal.experimental")
    experimental.stop_fetching_inputs = lambda: None
    module.experimental = experimental
    return module


@contextlib.contextmanager
def modal_stubbed():
    """`modal` stood in for the duration, and every module that imported it
    forgotten afterwards — so a later test importing the real SDK, or this
    one importing a second venue, starts clean."""
    before = {name: sys.modules.get(name)
              for name in ("modal", "modal.experimental")}
    module = modal_module()
    sys.modules["modal"] = module
    sys.modules["modal.experimental"] = module.experimental
    try:
        yield module
    finally:
        for name, was in before.items():
            if was is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = was
