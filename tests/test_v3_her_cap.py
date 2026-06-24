import numpy as np

from chessrl.training.cluster_her import cluster_goal_samples
from chessrl.training.her import reconstruct_states
from tests.test_cluster_her import _game, FakeEmbedder, FakeGoalSpace


def test_lookahead_cap_limits_achievement_window():
    rec = _game(n=6)
    states = reconstruct_states(rec)
    capped = cluster_goal_samples(rec, states, FakeEmbedder(), FakeGoalSpace(),
                                  np.random.default_rng(0), lookahead_cap=1)
    uncapped = cluster_goal_samples(rec, states, FakeEmbedder(), FakeGoalSpace(),
                                    np.random.default_rng(0))
    cap_pos = sum(1 for s in capped if s.v_goal == 1.0 and s.v_win_mask == 0.0)
    unc_pos = sum(1 for s in uncapped if s.v_goal == 1.0 and s.v_win_mask == 0.0)
    assert cap_pos <= unc_pos


def test_cap_none_is_v2_behavior():
    rec = _game(n=6)
    states = reconstruct_states(rec)
    a = cluster_goal_samples(rec, states, FakeEmbedder(), FakeGoalSpace(),
                             np.random.default_rng(0))
    b = cluster_goal_samples(rec, states, FakeEmbedder(), FakeGoalSpace(),
                             np.random.default_rng(0), lookahead_cap=None)
    assert len(a) == len(b)   # default None == unchanged v2 behavior
