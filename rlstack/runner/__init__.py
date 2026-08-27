"""The runner: the substrate that drives both worlds as daemons on a blackboard.

Engine (inference metal) and Learner (training metal) are the only seams.
Daemons (daemons/) synchronize through the store alone (signals.py) and share
metal through leases (lease.py); the loop plans them from the spec and runs
them. This is the one package allowed to import both worlds.
"""

from rlstack.runner import (  # noqa: F401
    interfaces, seeds, client, waves, signals, lease, sources, post, daemons,
    loop, fakes,
)
