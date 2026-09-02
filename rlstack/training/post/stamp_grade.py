"""The Stamp Office's reward: run the calls, pay the milestone ladder.

THE WHOLE STATE MACHINE IS INSIDE THIS CLASS ON PURPOSE: code_hashes covers a
registered name's class source and nothing it imports, so the rules that
decide a reward must live where the hash reads — an edit to any method here is
a different experiment, exactly as it should be (I3).

DENSE BY DESIGN — the DAPO lesson at 0.6B: an all-or-nothing reward makes
every group score identically and the advantage carries no signal, so the
ladder pays for PROGRESS along the one gold path:

    grabbed 0.15 -> folded 0.30 -> demanded ink 0.45 -> sealed 0.60
    -> filed in the demanded drawer 1.00

A rung counts only on top of the rungs below it (sealed over the wrong ink is
still 0.30: the gold path was left at the ink), each ILLEGAL call — parse
failure, precondition violation, a document the tray does not hold — costs
0.05, and the floor is 0.0. `exact` is the strict bit an eval curve wants:
the document filed in the demanded drawer with not one illegal call.

No pool traffic and no randomness: a checked reward costs nothing (I9).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rlstack.client import PoolClient
from rlstack.data.trajectory import Group
from rlstack.training.post.base import PostProcessor, postprocessor

CALL = re.compile(r"^\s*(grab|fold|ink|seal|file)\(\s*([^,()\s]+)\s*"
                  r"(?:,\s*([^,()\s]+)\s*)?\)\s*$")
ILLEGAL_COST = 0.05
LADDER = (0.15, 0.30, 0.45, 0.60, 1.00)
DRAWERS = {"rju": "D1", "vex": "D3", "mol": "D2"}


@dataclass
class _Doc:
    """One document's office state as the calls mutate it."""

    grabbed: bool = False
    folds: int = 0
    inks: list[str] = field(default_factory=list)
    sealed: bool = False
    filed: str | None = None


@postprocessor("stamp_grade")
class StampGrade(PostProcessor):
    produces = ("reward", "exact")

    async def process(self, group: Group, data: Mapping[str, Sequence[float]],
                      client: PoolClient) -> Mapping[str, Sequence[float]]:
        rewards, exacts = [], []
        for traj in group.trajectories:
            reward, exact = self.grade(traj.turns[-1].message.content,
                                       str(traj.task.meta["doc"]),
                                       str(traj.task.meta["color"]))
            rewards.append(reward)
            exacts.append(exact)
        return {"reward": rewards, "exact": exacts}

    def grade(self, completion: str, doc: str, color: str) -> tuple[float, float]:
        """Execute the completion's calls against a fresh office and pay it."""
        state, illegal = self.execute(completion, doc)
        ladder = self.milestones(state, color)
        reward = max(0.0, ladder - ILLEGAL_COST * illegal)
        exact = float(ladder == LADDER[-1] and illegal == 0)
        return reward, exact

    def execute(self, completion: str, doc: str) -> tuple[_Doc, int]:
        """Walk the lines: a legal call mutates the state, an illegal one is
        skipped and counted — the office refuses, it does not crash."""
        state = _Doc()
        illegal = 0
        for line in completion.splitlines():
            if not line.strip():
                continue
            parsed = CALL.match(line)
            if parsed is None:
                illegal += 1
                continue
            verb, subject, argument = parsed.group(1), parsed.group(2), parsed.group(3)
            if not self.apply(state, verb, subject, argument, doc):
                illegal += 1
        return state, illegal

    def apply(self, state: _Doc, verb: str, subject: str,
              argument: str | None, doc: str) -> bool:
        """One call against the rulebook: True if legal (and applied)."""
        if subject != doc:
            return False                       # the tray holds only `doc`
        if state.filed is not None:
            return False                       # filing ends the document
        if verb == "grab":
            if state.grabbed or argument is not None:
                return False
            state.grabbed = True
            return True
        if not state.grabbed:
            return False                       # grab must be the first action
        if verb == "fold":
            if state.folds >= 1 or state.sealed or argument is not None:
                return False                   # folding twice ruins it
            state.folds += 1
            return True
        if verb == "ink":
            if (state.folds != 1 or state.sealed or state.inks
                    or argument not in DRAWERS):
                return False                   # folded first, one ink only
            state.inks.append(argument)
            return True
        if verb == "seal":
            if state.sealed or len(state.inks) != 1 or argument is not None:
                return False                   # exactly one ink under a seal
            state.sealed = True
            return True
        if verb == "file":
            if not state.sealed or argument != DRAWERS[state.inks[0]]:
                return False                   # the ink fixes the drawer
            state.filed = argument
            return True
        return False

    def milestones(self, state: _Doc, color: str) -> float:
        """The highest rung of the gold path the document reached — each rung
        counts only on top of the rungs below it."""
        rungs = (state.grabbed,
                 state.folds == 1,
                 state.inks == [color],
                 state.sealed,
                 state.filed == DRAWERS[color])
        score = 0.0
        for reached, worth in zip(rungs, LADDER):
            if not reached:
                break
            score = worth
        return score
