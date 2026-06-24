import numpy as np
import chess

from chessrl.selfplay.meansend_chained import select_next_goal
from chessrl.config.config import GoalConfig
from chessrl.goals.winvalue import WinValueEstimator, ClusterCurriculum


class _Eval:
    """v_goal high for cluster `fav` (achievable), v_win fixed; one row per goal_vec."""
    def __init__(self, fav=1):
        self.fav = fav

    def evaluate_planes(self, planes, goal_vecs, deadlines):
        n = len(goal_vecs)
        vg = np.array([0.9 if i == self.fav else 0.1 for i in range(n)], np.float32)
        return np.zeros((n, 1), np.float32), np.full(n, 0.0, np.float32), vg


class _GS:
    centroids = np.eye(3, 4, dtype=np.float32)   # 3 clusters, dim 4
    n_clusters = 3
    ready = True


def test_select_prefers_winvalued_achievable_cluster():
    est = WinValueEstimator()
    for _ in range(10):
        est.update(1, True)                      # cluster 1 = win-valued
    cur = ClusterCurriculum(est, 3)
    c, vec = select_next_goal(
        chess.Board(), _GS(), cur, _Eval(fav=1),
        GoalConfig(goal_mode="emergent_chained", goal_select_temp=0.01, epsilon=0.0),
        np.random.default_rng(0))
    assert c == 1 and vec.shape == (4,)


def test_curriculum_weight_is_public_and_nonzero_via_novelty():
    cur = ClusterCurriculum(WinValueEstimator(), 3)
    assert cur.weight(0) > 0.0                    # novelty floor => never zero
