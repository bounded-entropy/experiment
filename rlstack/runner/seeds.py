"""The seed tree: every random draw in a run derives from Seeds.master.

No module anywhere in the runner touches global RNG state. A seed is
h(master, *path) where the path names what the seed is for, so any part of a
run can be regenerated in isolation — resume rebuilds an unsealed wave
bit-for-bit without replaying anything else.
"""

from __future__ import annotations

import hashlib
import json


def derive(master: int, *path: str | int) -> int:
    """A 63-bit seed for one named draw site. Same path → same seed, forever."""
    material = json.dumps([master, *path], separators=(",", ":"))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1
