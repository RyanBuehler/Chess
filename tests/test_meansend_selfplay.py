import numpy as np
from chessrl.config.config import GoalConfig
from chessrl.selfplay.concurrent import (
    _ClusterSideGoal, assign_cluster_goal, maybe_switch_cluster_to_terminal,
)


class ReadyGoalSpace:
    centroids = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    ready = True
    def centroid(self, c):
        return self.centroids[c].copy()
    @property
    def n_clusters(self):
        return 3


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
