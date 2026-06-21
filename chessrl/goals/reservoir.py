"""Fixed-size reservoir sampler over delta vectors (Plan 2, Task 2).

Classic Algorithm R: the first ``capacity`` items are kept; each later item
(index i, 0-based, i >= capacity) replaces a uniformly random slot with
probability ``capacity / (i + 1)``. This yields a uniform random sample of the
whole stream at any time — so the k-means fit sees old and new deltas, not just
the most recent window."""
from __future__ import annotations

import numpy as np


class Reservoir:
    def __init__(self, capacity: int, dim: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.rng = rng
        self._buf = np.zeros((self.capacity, self.dim), dtype=np.float32)
        self._n = 0      # filled slots
        self.seen = 0    # total ever added

    def add(self, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(self.dim)
        if self._n < self.capacity:
            self._buf[self._n] = v
            self._n += 1
        else:
            j = int(self.rng.integers(self.seen + 1))
            if j < self.capacity:
                self._buf[j] = v
        self.seen += 1

    def array(self) -> np.ndarray:
        return self._buf[: self._n].copy()

    def __len__(self) -> int:
        return self._n
