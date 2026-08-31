"""The Glyph Exchange's reward: parse the one expression, walk the ring.

The parser and the ring live INSIDE this class for stamp_grade's reason: the
rules that decide a reward must sit where code_hashes reads (I3).

DENSE BY DESIGN, like the Stamp Office: the demanded route is the unique
directed path source -> target along the ring (1 to 3 edges), and the ladder
pays for the correct PREFIX of the gold expression

    load(the named purse)               0.2
    each correct conversion, in order   0.6 * (correct prefix / needed)
    give as the outermost call          0.2 (only once the route is complete)

A malformed expression scores 0. A well-formed one whose innermost call is
not load-of-the-named-purse stays at 0. Wrong or extra conversions end the
prefix where they happen and cost 0.05 each; the floor is 0.0. `exact` is a
complete correct route, given, with nothing spurious.

No pool traffic and no randomness: a checked reward costs nothing (I9).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor

TOKEN = re.compile(r"\s*(load|to_drin|to_polk|to_sarn|to_wex|give)\s*\(\s*")
UNITS = ("wex", "drin", "polk", "sarn")     # the ring, in edge order
STEPS = {"to_drin": "wex", "to_polk": "drin",
         "to_sarn": "polk", "to_wex": "sarn"}   # tool -> the unit it accepts
AFTER = {"to_drin": "drin", "to_polk": "polk",
         "to_sarn": "sarn", "to_wex": "wex"}    # tool -> the unit it yields
ILLEGAL_COST = 0.05
LOAD_WORTH = 0.2
ROUTE_WORTH = 0.6
GIVE_WORTH = 0.2


@postprocessor("glyph_grade")
class GlyphGrade(PostProcessor):
    produces = ("reward", "exact")

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        rewards, exacts = [], []
        for traj in group.trajectories:
            reward, exact = self.grade(traj.turns[-1].message.content,
                                       str(traj.task.meta["purse"]),
                                       str(traj.task.meta["source"]),
                                       str(traj.task.meta["target"]))
            rewards.append(reward)
            exacts.append(exact)
        return {"reward": rewards, "exact": exacts}

    def grade(self, completion: str, purse: str, source: str,
              target: str) -> tuple[float, float]:
        chain = self.parse(completion)
        if chain is None:
            return 0.0, 0.0
        return self.walk(chain, purse, source, target)

    def parse(self, completion: str) -> list[tuple[str, str | None]] | None:
        """The first nested expression in the completion, innermost-first:
        [("load", "P7"), ("to_drin", None), ..., ("give", None)] — or None
        when nothing parses. Only the nesting grammar is judged here; whether
        the calls make SENSE is walk()'s business."""
        opened = re.search(r"(?:give|load)\s*\(", completion)
        if opened is None:
            return None
        text = completion[opened.start():]
        outer: list[str] = []
        position = 0
        while True:
            matched = TOKEN.match(text, position)
            if matched is None:
                break
            outer.append(matched.group(1))
            position = matched.end()
        if not outer:
            return None
        inner = re.match(r"\s*([A-Za-z0-9_]*)\s*", text[position:])
        argument = inner.group(1) or None
        chain = [(outer[-1], argument)]
        chain += [(verb, None) for verb in reversed(outer[:-1])]
        return chain

    def walk(self, chain: list[tuple[str, str | None]], purse: str,
             source: str, target: str) -> tuple[float, float]:
        """Pay the ladder: load, then the gold route's prefix, then give."""
        route = self.gold_route(source, target)
        verb, argument = chain[0]
        if verb != "load" or argument != purse:
            return 0.0, 0.0
        score = LOAD_WORTH
        taken = [v for v, _ in chain[1:] if v in STEPS]
        prefix = 0
        while prefix < min(len(taken), len(route)) and taken[prefix] == route[prefix]:
            prefix += 1
        score += ROUTE_WORTH * (prefix / len(route))
        spurious = len(taken) - prefix
        gave = chain[-1][0] == "give" and sum(v == "give" for v, _ in chain) == 1
        complete = prefix == len(route) and spurious == 0
        if complete and gave:
            score += GIVE_WORTH
        reward = max(0.0, score - ILLEGAL_COST * spurious)
        exact = float(complete and gave)
        return reward, exact

    def gold_route(self, source: str, target: str) -> list[str]:
        """The unique directed path source -> target along the ring, as the
        tool names that walk it."""
        tools = ("to_drin", "to_polk", "to_sarn", "to_wex")
        route: list[str] = []
        at = UNITS.index(source)
        while UNITS[at] != target:
            route.append(tools[at])
            at = (at + 1) % len(UNITS)
        return route
