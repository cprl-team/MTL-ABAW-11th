from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of(path) -> str:
    """Full SHA-256 of a file (read in chunks; checkpoints can be large)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BestTracker:
    """Track the best validation score (higher-is-better by default).

    Usage:
        tracker = BestTracker()
        if tracker.update(val_score):
            save_checkpoint(...)   # new best -> persist
    """

    def __init__(self, higher_is_better: bool = True):
        self.higher_is_better = higher_is_better
        self.best = None
        self.best_step = -1
        self._step = -1

    def update(self, score: float) -> bool:
        self._step += 1
        improved = (self.best is None
                    or (score > self.best if self.higher_is_better else score < self.best))
        if improved:
            self.best = score
            self.best_step = self._step
        return improved
