"""NumPy k-means for goal discovery (Plan 2, Task 3).

Lloyd's algorithm with empty-cluster reseeding: any cluster that loses all its
members is moved to the single point currently farthest from its assigned
centroid. This guarantees exactly ``k`` non-empty centroids every fit (no
collapse), so the discovered-goal vocabulary size stays constant. No
scikit-learn dependency — refits are periodic, not per-step, so plain Lloyd's
is fast enough."""
from __future__ import annotations

import numpy as np


def assign_nearest(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # squared euclidean distance to each centroid, argmin
    d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).astype(np.int64)


def kmeans_fit(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    d = x.shape[1]
    if n == 0:
        return np.zeros((k, d), dtype=np.float32)
    if n < k:
        # pad by duplicating random points to length k
        idx = rng.integers(n, size=k)
        return x[idx].copy()
    # k-means++-lite init: random distinct starting points
    start = rng.choice(n, size=k, replace=False)
    centroids = x[start].copy()
    for _ in range(iters):
        labels = assign_nearest(x, centroids)
        new = centroids.copy()
        for j in range(k):
            members = x[labels == j]
            if len(members) == 0:
                # reseed to the farthest point from its own centroid
                d2 = ((x - centroids[labels]) ** 2).sum(axis=1)
                new[j] = x[int(d2.argmax())]
            else:
                new[j] = members.mean(axis=0)
        if np.allclose(new, centroids):
            centroids = new
            break
        centroids = new
    return centroids.astype(np.float32)


def median_radius(x: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    if x.shape[0] == 0:
        return 0.0
    dists = np.linalg.norm(x - centroids[labels], axis=1)
    return float(np.median(dists))
