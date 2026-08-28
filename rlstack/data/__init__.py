"""The membrane: the data objects that cross between the worlds, and the store.

trajectory.py holds the sealed records (Task, Trajectory, Turn, Group, Wave),
flatten.py their packed forms (Flat, TokenBatch), stores/ the run store.
Dumb, loss-independent, estimator-free; imports no other rlstack package
(enforced by tests/test_architecture.py).
"""

from rlstack.data import trajectory, flatten, stores  # noqa: F401
