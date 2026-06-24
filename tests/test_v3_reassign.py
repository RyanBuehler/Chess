import numpy as np
import chess

from chessrl.selfplay.meansend_chained import _ChainSideGoal, maybe_reassign_goal
from chessrl.config.config import GoalConfig
from chessrl.goals.winvalue import WinValueEstimator, ClusterCurriculum


class _Eval:
    def evaluate_planes(self, planes, goal_vecs, deadlines):
        n = len(goal_vecs)
        return np.zeros((n, 1), np.float32), np.zeros(n, np.float32), np.full(n, 0.1, np.float32)

    def embed_boards(self, boards):
        return np.zeros((len(boards), 4), np.float32)


class _GS:
    centroids = np.eye(3, 4, dtype=np.float32)
    n_clusters = 3
    tau = 10.0
    ready = True

    def achieved(self, delta, c):
        return False   # never achieved -> only expiry triggers


def _cur():
    return ClusterCurriculum(WinValueEstimator(), 3)


def test_expiry_triggers_reassignment():
    cfg = GoalConfig(goal_mode="emergent_chained", goal_window=8, epsilon=0.0)
    side = _ChainSideGoal(0, np.zeros(4, np.float32), start_ply=0,
                          start_emb=np.zeros(4, np.float32), explore=False)
    same = maybe_reassign_goal(side, chess.Board(), 4, _GS(), _cur(), _Eval(), cfg,
                               np.random.default_rng(0))
    assert same is side                       # within window, not expired
    nxt = maybe_reassign_goal(side, chess.Board(), 8, _GS(), _cur(), _Eval(), cfg,
                              np.random.default_rng(0))
    assert nxt is not side and nxt.start_ply == 8   # expired at goal_window -> reassigned
