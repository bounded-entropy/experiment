"""Pure token accounting shared by serving enforcement and recorded evidence."""

from collections.abc import Sequence

CONTINUATION_KEY = "rlstack_min_tokens_after_v1"


class ContinuationCounter:
    """Incremental first-sequence matching over generated IDs only.

    KMP keeps overlapping markers correct without rescanning a long thought
    on every decode step. Neither prompt tokens nor terminal IDs count.
    """

    def __init__(self, sequence: tuple[int, ...], terminal_ids: tuple[int, ...]) -> None:
        self.sequence = sequence
        self.terminals = frozenset(terminal_ids)
        self.failure = [0] * len(sequence)
        j = 0
        for i in range(1, len(sequence)):
            while j and sequence[i] != sequence[j]:
                j = self.failure[j - 1]
            if sequence[i] == sequence[j]:
                j += 1
            self.failure[i] = j
        self.matched = 0
        self.output_tokens = 0
        self.boundary_end: int | None = None
        self.nonterminal_tokens = 0
        self.terminal_tokens = 0

    def extend(self, tokens: Sequence[int]) -> None:
        for token in tokens:
            self.output_tokens += 1
            if self.boundary_end is not None:
                if token in self.terminals:
                    self.terminal_tokens += 1
                else:
                    self.nonterminal_tokens += 1
                continue
            while self.matched and token != self.sequence[self.matched]:
                self.matched = self.failure[self.matched - 1]
            if token == self.sequence[self.matched]:
                self.matched += 1
            if self.matched == len(self.sequence):
                self.boundary_end = self.output_tokens

    def suppresses_terminals(self, minimum: int) -> bool:
        return self.boundary_end is not None and self.nonterminal_tokens < minimum

    def evidence(self, minimum: int) -> dict:
        return {"activated": self.boundary_end is not None,
                "boundary_end_output_index": self.boundary_end,
                "output_tokens": self.output_tokens,
                "nonterminal_tokens_after_boundary": self.nonterminal_tokens,
                "terminal_tokens_after_boundary": self.terminal_tokens,
                "floor_met": self.boundary_end is not None and
                             self.nonterminal_tokens >= minimum}
