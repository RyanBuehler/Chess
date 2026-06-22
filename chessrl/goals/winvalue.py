"""Interventional per-cluster win-value + win-valued cluster curriculum (Plan 4d).

win_value(g) = E[P(win | do(assign g))] - base_winrate, estimated from a
Beta-Bernoulli posterior fed ONLY by epsilon-explore games (uniform-random
assignment) so the estimate is de-confounded. The curriculum samples sub-goals
biased toward win-relevant clusters: w(g) = beta*novelty(g) + gamma*max(0,win_value(g))."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class WinValueEstimator:
    def __init__(self, prior_a: float = 1.0, prior_b: float = 1.0):
        self.prior_a = prior_a
        self.prior_b = prior_b
        self._wins: dict[int, int] = {}
        self._att: dict[int, int] = {}

    def update(self, cluster: int, won: bool) -> None:
        c = int(cluster)
        self._att[c] = self._att.get(c, 0) + 1
        self._wins[c] = self._wins.get(c, 0) + (1 if won else 0)

    @property
    def base_winrate(self) -> float:
        tot = sum(self._att.values())
        if tot == 0:
            return 0.0
        return sum(self._wins.values()) / tot

    def attempts(self, cluster: int) -> int:
        return self._att.get(int(cluster), 0)

    def win_value(self, cluster: int) -> float:
        c = int(cluster)
        if self._att.get(c, 0) == 0:
            return 0.0
        a = self.prior_a + self._wins.get(c, 0)
        b = self.prior_b + (self._att[c] - self._wins.get(c, 0))
        return a / (a + b) - self.base_winrate

    def to_dict(self) -> dict:
        return {"prior_a": self.prior_a, "prior_b": self.prior_b,
                "wins": self._wins, "att": self._att}

    @classmethod
    def from_dict(cls, d: dict) -> "WinValueEstimator":
        e = cls(d.get("prior_a", 1.0), d.get("prior_b", 1.0))
        e._wins = {int(k): int(v) for k, v in d.get("wins", {}).items()}
        e._att = {int(k): int(v) for k, v in d.get("att", {}).items()}
        return e

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path) -> "WinValueEstimator":
        return cls.from_dict(json.loads(Path(path).read_text()))


class ClusterCurriculum:
    def __init__(self, estimator: WinValueEstimator, n_clusters: int,
                 novelty_beta: float = 1.0, gamma_winvalue: float = 1.0, win_floor: float = 0.2):
        self.est = estimator
        self.n_clusters = int(n_clusters)
        self.novelty_beta = novelty_beta
        self.gamma_winvalue = gamma_winvalue
        self.win_floor = win_floor

    def _novelty(self, c: int) -> float:
        return 1.0 / np.sqrt(1.0 + self.est.attempts(c))

    def _weight(self, c: int) -> float:
        return self.novelty_beta * self._novelty(c) + self.gamma_winvalue * max(0.0, self.est.win_value(c))

    def sample(self, rng: np.random.Generator) -> int:
        if self.n_clusters <= 0 or rng.random() < self.win_floor:
            return -1
        w = np.array([self._weight(c) for c in range(self.n_clusters)], dtype=np.float64)
        tot = w.sum()
        probs = (w / tot) if (tot > 0 and np.isfinite(tot)) else np.full(self.n_clusters, 1.0 / self.n_clusters)
        return int(rng.choice(self.n_clusters, p=probs))

    def record_attempt(self, cluster: int) -> None:
        pass   # attempts tracked by the estimator's update(); placeholder for symmetry
