import numpy as np
import chess
from chessrl.config.config import GoalConfig, MCTSConfig, SelfPlayConfig
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.selfplay.concurrent import (
    _ClusterSideGoal, assign_cluster_goal, maybe_switch_cluster_to_terminal,
    play_meansend_games_concurrent,
)


class ReadyGoalSpace:
    centroids = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    ready = True
    tau = 1.0
    def centroid(self, c):
        return self.centroids[c].copy()
    @property
    def n_clusters(self):
        return 3
    def achieved(self, delta, c):   # v3 chained path; False => chaining driven by expiry
        return False


class UnfitGoalSpace:
    centroids = None
    ready = False
    n_clusters = 0


WIN_VEC = np.full(4, -1.0, np.float32)


def test_assign_terminal_when_unfit():
    g = assign_cluster_goal(UnfitGoalSpace(), WIN_VEC, GoalConfig(goal_mode="emergent", win_floor=0.0),
                            np.random.default_rng(0))
    assert g.is_terminal()
    assert np.allclose(g.active_vec, WIN_VEC)


def test_assign_subgoal_when_ready():
    # win_floor=0 forces a sub-goal when ready
    g = assign_cluster_goal(ReadyGoalSpace(), WIN_VEC, GoalConfig(goal_mode="emergent", win_floor=0.0),
                            np.random.default_rng(1))
    assert not g.is_terminal()
    assert 0 <= g.active_cluster < 3
    assert np.allclose(g.active_vec, ReadyGoalSpace().centroid(g.active_cluster))


def test_win_floor_forces_terminal():
    g = assign_cluster_goal(ReadyGoalSpace(), WIN_VEC, GoalConfig(goal_mode="emergent", win_floor=1.0),
                            np.random.default_rng(2))
    assert g.is_terminal()


def test_deadline_switch_to_terminal():
    g = _ClusterSideGoal(assigned_cluster=1, assigned_vec=np.zeros(4, np.float32), deadline=3,
                         start_ply=0, active_cluster=1, active_vec=np.zeros(4, np.float32))
    maybe_switch_cluster_to_terminal(g, ply=2, win_vector=WIN_VEC, deadline_max=60)
    assert not g.is_terminal()          # 2 < 3, no switch yet
    maybe_switch_cluster_to_terminal(g, ply=3, win_vector=WIN_VEC, deadline_max=60)
    assert g.is_terminal()              # 3 >= 3, switched
    assert np.allclose(g.active_vec, WIN_VEC)


class FakeVectorEval:
    """Dual-head batched vector evaluator. evaluate_planes(planes, goal_vecs,
    deadlines) -> (policies uniform, v_win, v_goal)."""
    def evaluate_planes(self, planes, goal_vecs, deadlines):
        n = planes.shape[0]
        pol = np.full((n, NUM_ACTIONS), 1.0 / NUM_ACTIONS, np.float32)
        return pol, np.zeros(n, np.float32), np.full(n, 0.5, np.float32)

    def embed_boards(self, boards):   # v3 chained path
        out = [[float(b.fullmove_number), float(len(b.piece_map())), 0.0, 0.0] for b in boards]
        return np.asarray(out, np.float32)

    def win_value(self, planes, deadlines):   # v3 chained path: neutral -> alpha stays high
        return np.zeros(len(planes), np.float32)


def test_meansend_selfplay_produces_cluster_records():
    gs = ReadyGoalSpace()
    recs = play_meansend_games_concurrent(
        FakeVectorEval(), MCTSConfig(simulations=8, leaves_per_tree=1),
        SelfPlayConfig(ply_cap=6, resign_playout_fraction=1.0),
        GoalConfig(goal_mode="emergent", win_floor=0.0, goal_window=2, deadline_max=60),
        gs, np.full(4, -1.0, np.float32), np.random.default_rng(0), num_games=2,
    )
    assert len(recs) == 2
    for rec, board, z, meta in recs:
        assert rec.has_cluster_goals()
        assert rec.active_vec.shape[1] == 4
        assert "win_ply_fraction" in meta
        # deadline switch (goal_window=2) means later plies are terminal (-1)
        assert (rec.active_cluster == -1).any()


def test_meansend_selfplay_unfit_is_all_terminal():
    recs = play_meansend_games_concurrent(
        FakeVectorEval(), MCTSConfig(simulations=8, leaves_per_tree=1),
        SelfPlayConfig(ply_cap=4, resign_playout_fraction=1.0),
        GoalConfig(goal_mode="emergent", win_floor=0.0, goal_window=2, deadline_max=60),
        UnfitGoalSpace(), np.full(4, -1.0, np.float32), np.random.default_rng(0), num_games=1,
    )
    rec = recs[0][0]
    assert rec.has_cluster_goals()
    assert (rec.active_cluster == -1).all()   # unfit -> always terminal


def test_epsilon_explore_marks_explore_and_uniform():
    rng = np.random.default_rng(0)
    seen_explore = 0
    for _ in range(200):
        g = assign_cluster_goal(ReadyGoalSpace(), WIN_VEC,
                                GoalConfig(goal_mode="emergent", win_floor=0.0, epsilon=1.0),
                                rng)
        if g.explore: seen_explore += 1
    assert seen_explore == 200   # epsilon=1 -> always explore


def test_curriculum_used_when_not_exploring():
    class Cur:
        def sample(self, rng): return 2
    g = assign_cluster_goal(ReadyGoalSpace(), WIN_VEC,
                            GoalConfig(goal_mode="emergent", win_floor=0.0, epsilon=0.0),
                            np.random.default_rng(0), curriculum=Cur())
    assert g.active_cluster == 2 and g.explore is False
