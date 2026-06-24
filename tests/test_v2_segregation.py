"""SEGREGATION GATE: v2-halcyon (goal_mode 'emergent') must be byte-reproducible —
v3 additions (per-tree alpha unset, lookahead_cap None, the separate chained driver)
must not alter the v2 path. Same evaluator + seed => identical v2 means-end records."""
import numpy as np

from chessrl.config.config import GoalConfig, MCTSConfig, SelfPlayConfig
from chessrl.selfplay.concurrent import play_meansend_games_concurrent
from tests.test_meansend_selfplay import FakeVectorEval, ReadyGoalSpace


def _run():
    return play_meansend_games_concurrent(
        FakeVectorEval(), MCTSConfig(simulations=4, leaves_per_tree=1, meansend_alpha=0.25),
        SelfPlayConfig(ply_cap=16, resign_playout_fraction=1.0),
        GoalConfig(goal_mode="emergent", goal_window=3, win_floor=0.0, deadline_max=20),
        ReadyGoalSpace(), np.full(4, -1.0, np.float32), np.random.default_rng(7),
        num_games=2, game_id_prefix="v2_")


def test_v2_emergent_path_reproducible():
    a = _run()
    b = _run()
    assert len(a) == len(b) == 2
    for (ra, *_), (rb, *_) in zip(a, b):
        assert np.array_equal(np.asarray(ra.active_cluster), np.asarray(rb.active_cluster))
        assert np.array_equal(np.asarray(ra.played), np.asarray(rb.played))
