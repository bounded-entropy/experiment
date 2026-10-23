"""Bounded, opt-in memoization at one unchanged loss-gradient measurement point.

Only the reference vector stays resident. Candidate entries contain scalar
comparisons and document statistics. Nothing here is durable or optimizer state.
"""
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json

from rlstack.client import GradientComparison


class GradientCache:
    def __init__(self, max_bytes: int, max_comparisons: int = 1024):
        if max_bytes < 0:
            raise ValueError('gradient cache bytes must be nonnegative')
        self.max_bytes = max_bytes
        self.max_comparisons = max_comparisons
        self.context = None
        self.reference_key = None
        self.saved_reference = None
        self.comparisons: OrderedDict[tuple, GradientComparison] = OrderedDict()

    def clear(self):
        """Release cached GPU storage before sleep, uninstall or a state change."""
        self.context = None
        self.reference_key = None
        self.saved_reference = None
        self.comparisons.clear()

    def invalidate(self, state_id: int) -> None:
        """Payload loads can copy through .data without changing tensor versions."""
        if self.context is not None and self.context[0] == state_id:
            self.clear()

    def select(self, context):
        if context != self.context:
            self.clear()
            self.context = context

    def key(self, batches):
        """Exact packed IDs, masks, routes, reductions and replay facts."""
        if not self.max_bytes:
            return None
        body = json.dumps([asdict(b) for b in batches], sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(body.encode()).hexdigest()

    def reference(self, key):
        return self.saved_reference if key is not None and key == self.reference_key else None

    def save_reference(self, key, vector, documents, loss_reason, norm, reason):
        if key is not None and sum(v.numel()*v.element_size() for v in vector) <= self.max_bytes:
            self.reference_key = key
            self.saved_reference = (vector, documents, loss_reason, norm, reason)

    def comparison(self, key):
        result = self.comparisons.get(key) if self.max_bytes else None
        if result is not None:
            self.comparisons.move_to_end(key)
        return result

    def save_comparison(self, key, comparison):
        if self.max_bytes:
            self.comparisons[key] = comparison
            self.comparisons.move_to_end(key)
            while len(self.comparisons) > self.max_comparisons:
                self.comparisons.popitem(last=False)
