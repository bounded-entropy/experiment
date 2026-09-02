"""Task makers: one class per file, all inheriting TaskMaker (base.py).

The contract is `make(source: Trajectory) -> Task` — pure, synchronous, the
content half of a Derive leaf. Importing this package registers the builtins.
"""

from rlstack.inference.makers.base import (  # noqa: F401
    MakerDef, TaskMaker, task_maker,
)
from rlstack.inference.makers import (  # noqa: F401  (registers builtins)
    reflect,
)
