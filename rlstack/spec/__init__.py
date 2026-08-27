"""The contract: declarative spec values, computed identity, the submit gate.

Sits above both worlds — nothing here samples or takes gradients.
"""

from rlstack.spec import canonical, specs, validate  # noqa: F401  (import order matters)
