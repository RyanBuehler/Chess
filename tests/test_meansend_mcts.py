import pytest
import numpy as np
import chess
from chessrl.config.config import MCTSConfig
from chessrl.mcts.batched import BatchedMCTS


def test_meansend_alpha_default():
    assert MCTSConfig().meansend_alpha == 0.0


def test_meansend_alpha_settable():
    assert MCTSConfig(meansend_alpha=0.5).meansend_alpha == 0.5


def test_meansend_alpha_rejects_out_of_range():
    with pytest.raises(ValueError):
        MCTSConfig(meansend_alpha=1.5)


class FakeDualEval:
    """Dual-head batched evaluator: uniform policy, fixed v_win, fixed v_goal.
    evaluate_planes(planes (N,21,8,8), goal_vecs (N,d), deadlines (N,)) ->
    (policies (N,4672), v_win (N,), v_goal (N,))."""
    def __init__(self, v_win=0.4, v_goal=0.9, n_actions=4672):
        self.v_win = v_win; self.v_goal = v_goal; self.n = n_actions
    def evaluate_planes(self, planes, goal_vecs, deadlines):
        n = planes.shape[0]
        pol = np.full((n, self.n), 1.0 / self.n, dtype=np.float32)
        return pol, np.full(n, self.v_win, np.float32), np.full(n, self.v_goal, np.float32)


def _cfg(alpha):
    return MCTSConfig(simulations=16, leaves_per_tree=1, meansend_alpha=alpha)


def test_meansend_runs_and_produces_visits():
    mcts = BatchedMCTS(FakeDualEval(), _cfg(0.25), np.random.default_rng(0), meansend=True)
    gv = np.zeros(8, np.float32)
    tree = mcts.init_tree_for_meansend(chess.Board(), gv, deadline=20, add_noise=False)
    mcts.run(tree)
    visits = mcts.visit_counts(tree)
    assert sum(visits.values()) > 0


def test_meansend_leaf_blend_alpha0_is_win_only():
    # alpha=0 -> root q reflects v_win only (negamax over v_win); v_goal ignored.
    mcts0 = BatchedMCTS(FakeDualEval(v_win=0.4, v_goal=0.9), _cfg(0.0),
                        np.random.default_rng(1), meansend=True)
    t0 = mcts0.init_tree_for_meansend(chess.Board(), np.zeros(8, np.float32), 20)
    mcts0.run(t0)
    # With uniform policy + constant leaf value, root q ~ the leaf value (sign aside);
    # alpha=0 must equal a run where v_goal is different but alpha=0 (goal ignored).
    mcts0b = BatchedMCTS(FakeDualEval(v_win=0.4, v_goal=0.1), _cfg(0.0),
                         np.random.default_rng(1), meansend=True)
    t0b = mcts0b.init_tree_for_meansend(chess.Board(), np.zeros(8, np.float32), 20)
    mcts0b.run(t0b)
    assert abs(mcts0.root_q(t0) - mcts0b.root_q(t0b)) < 1e-6  # v_goal had no effect


def test_meansend_alpha_uses_v_goal():
    # alpha=1 -> leaf value = 2*v_goal-1; changing v_goal changes root q.
    a = BatchedMCTS(FakeDualEval(v_win=0.4, v_goal=0.9), _cfg(1.0),
                    np.random.default_rng(2), meansend=True)
    ta = a.init_tree_for_meansend(chess.Board(), np.zeros(8, np.float32), 20); a.run(ta)
    b = BatchedMCTS(FakeDualEval(v_win=0.4, v_goal=0.1), _cfg(1.0),
                    np.random.default_rng(2), meansend=True)
    tb = b.init_tree_for_meansend(chess.Board(), np.zeros(8, np.float32), 20); b.run(tb)
    assert abs(a.root_q(ta) - b.root_q(tb)) > 1e-3   # v_goal now matters


def test_vanilla_path_untouched():
    # A vanilla BatchedMCTS (no meansend, no goal) still works.
    class FakeVanilla:
        def evaluate_many(self, boards):
            n = len(boards)
            return np.full((n, 4672), 1.0/4672, np.float32), np.zeros(n, np.float32)
        def evaluate_planes(self, planes):
            n = planes.shape[0]
            return np.full((n, 4672), 1.0/4672, np.float32), np.zeros(n, np.float32)
    mcts = BatchedMCTS(FakeVanilla(), MCTSConfig(simulations=8, leaves_per_tree=1),
                       np.random.default_rng(0))
    tree = mcts.init_tree(chess.Board())
    mcts.run(tree)
    assert sum(mcts.visit_counts(tree).values()) > 0
