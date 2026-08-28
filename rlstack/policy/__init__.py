"""The bridge (I2): a policy — a base plus its bank — is the only primitive
living in both worlds, and so the only one carrying a parity obligation.

siteschema names the sites a delta may attach to; adapters/ holds one registered
kind per file plus both of its lowerings; compile lowers a bank into the bundle,
the only data channel back from the training world.
"""

from rlstack.policy import siteschema, adapters, compile  # noqa: F401  (import order matters)
