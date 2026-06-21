# chessrl/goals/goalspace.py
"""Discovered-goal space: clusters of state-delta embeddings (Plan 2, Task 4).

A goal is a cluster id; its code is the centroid in the FROZEN encoder's
embedding space. GoalSpace accumulates deltas in a reservoir, fits k-means to
get the goal vocabulary, and answers assign()/achieved() by nearest centroid
within the median-radius threshold tau. The embedder is injected (Plan 1's
VectorGoalNetEvaluator in production; a fake in tests) and is treated as frozen
within an epoch; maybe_refresh() swaps in a fresh snapshot and re-fits."""
from __future__ import annotations

import chess
import numpy as np

from chessrl.config.config import GoalConfig
from chessrl.goals.reservoir import Reservoir
from chessrl.goals.clustering import kmeans_fit, assign_nearest, median_radius


class GoalSpace:
    def __init__(self, cfg: GoalConfig, embedder, rng: np.random.Generator):
        self.cfg = cfg
        self.embedder = embedder
        self.rng = rng
        self.d = int(embedder.embed_boards([chess.Board()]).shape[1])
        self.reservoir = Reservoir(cfg.reservoir_size, self.d, rng)
        self.centroids: np.ndarray | None = None
        self.tau: float = 0.0
        self._last_refresh_epoch = -1

    # --- embedding / observation ----------------------------------------
    def delta(self, start_board: chess.Board, end_board: chess.Board) -> np.ndarray:
        e = self.embedder.embed_boards([start_board, end_board])
        return (e[1] - e[0]).astype(np.float32)

    def observe(self, start_board: chess.Board, end_board: chess.Board) -> None:
        self.observe_delta(self.delta(start_board, end_board))

    def observe_delta(self, delta: np.ndarray) -> None:
        self.reservoir.add(np.asarray(delta, dtype=np.float32))

    # --- fitting --------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.centroids is not None and len(self.reservoir) >= self.cfg.min_reservoir

    @property
    def n_clusters(self) -> int:
        return 0 if self.centroids is None else self.centroids.shape[0]

    def fit(self) -> None:
        x = self.reservoir.array()
        cents = kmeans_fit(x, self.cfg.cluster_k, self.rng)
        labels = assign_nearest(x, cents)
        self.centroids = cents
        self.tau = median_radius(x, cents, labels)

    def maybe_refresh(self, games_seen: int, embedder=None) -> bool:
        epoch = games_seen // self.cfg.refresh_every
        if epoch <= self._last_refresh_epoch:
            return False
        if len(self.reservoir) < self.cfg.min_reservoir:
            return False
        if embedder is not None:
            self.embedder = embedder
        self.fit()
        self._last_refresh_epoch = epoch
        return True

    # --- queries --------------------------------------------------------
    def assign(self, delta: np.ndarray) -> int:
        assert self.centroids is not None, "GoalSpace not fit yet"
        d = np.asarray(delta, dtype=np.float32).reshape(1, self.d)
        return int(assign_nearest(d, self.centroids)[0])

    def centroid(self, goal_id: int) -> np.ndarray:
        assert self.centroids is not None, "GoalSpace not fit yet"
        return self.centroids[goal_id].copy()

    def achieved(self, delta: np.ndarray, goal_id: int) -> bool:
        if self.centroids is None:
            return False
        d = np.asarray(delta, dtype=np.float32).reshape(self.d)
        if self.assign(d) != goal_id:
            return False
        return bool(np.linalg.norm(d - self.centroids[goal_id]) <= self.tau)
