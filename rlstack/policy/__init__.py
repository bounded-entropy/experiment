"""The bridge (I2): the policy is the ONLY primitive living in both worlds.

Sites name where deltas attach; adapters say how each half lowers on each side.
"""

from rlstack.policy import siteschema, adapters, compile  # noqa: F401  (import order matters)
