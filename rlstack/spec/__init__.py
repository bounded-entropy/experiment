"""The contract: declarative values, identity, the flow graph, the submit gate.

Sits above both worlds — nothing here samples, allocates or takes gradients.
Everything in this package is a derivation over spec values alone.
"""

from rlstack.spec import canonical, specs, validate  # noqa: F401  (import order matters)
