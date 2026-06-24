import numpy as np

from chessrl.config.config import GoalConfig, MCTSConfig, SelfPlayConfig
from chessrl.selfplay.meansend_chained import play_meansend_chained_games_concurrent
from chessrl.goals.winvalue import WinValueEstimator, ClusterCurriculum
from tests.test_meansend_selfplay import FakeVectorEval, ReadyGoalSpace


def test_chained_game_produces_a_chain():
    gs = ReadyGoalSpace()
    cur = ClusterCurriculum(WinValueEstimator(), gs.n_clusters)
    res = play_meansend_chained_games_concurrent(
        FakeVectorEval(),
        MCTSConfig(simulations=4, leaves_per_tree=1, meansend_alpha=0.25),
        SelfPlayConfig(ply_cap=24, resign_playout_fraction=1.0),
        GoalConfig(goal_mode="emergent_chained", goal_window=2, win_floor=0.0,
                   deadline_max=20, epsilon=0.0),
        gs, np.full(4, -1.0, np.float32), np.random.default_rng(0),
        num_games=2, curriculum=cur, game_id_prefix="v3_")
    assert len(res) == 2
    rec = res[0][0]
    ac = np.asarray(rec.active_cluster)
    # goal_window=2 over a multi-ply game => the active cluster changes at least once
    assert len(set(int(x) for x in ac)) >= 2
    # whole game is goal-directed (ready goalspace => no terminal bootstrap): no -1 plies
    assert (ac >= 0).mean() > 0.5


def test_unfit_goalspace_falls_back_to_terminal():
    from chessrl.selfplay.meansend_chained import _terminal_side
    s = _terminal_side(np.full(4, -1.0, np.float32), 0)
    assert s.is_terminal() and s.active_cluster == -1
