import numpy as np
from chessrl.goals.winvalue import WinValueEstimator, ClusterCurriculum


def test_winvalue_lift_and_base():
    e = WinValueEstimator()
    for _ in range(40): e.update(0, True)    # cluster 0 wins a lot
    for _ in range(40): e.update(1, False)   # cluster 1 loses a lot
    assert e.win_value(0) > e.win_value(1)
    assert 0.0 <= e.base_winrate <= 1.0
    assert e.win_value(2) == 0.0             # no data -> neutral
    assert e.attempts(0) == 40


def test_curriculum_biases_toward_high_winvalue():
    e = WinValueEstimator()
    for _ in range(60): e.update(3, True)    # cluster 3 high win-value
    for _ in range(60): e.update(5, False)
    cur = ClusterCurriculum(e, n_clusters=8, novelty_beta=0.1, gamma_winvalue=5.0, win_floor=0.0)
    rng = np.random.default_rng(0)
    picks = [cur.sample(rng) for _ in range(400)]
    assert picks.count(3) > picks.count(5)   # biased toward the win-relevant cluster


def test_curriculum_win_floor_returns_terminal():
    cur = ClusterCurriculum(WinValueEstimator(), n_clusters=4, novelty_beta=1.0,
                            gamma_winvalue=1.0, win_floor=1.0)
    assert cur.sample(np.random.default_rng(0)) == -1


def test_save_load_roundtrip(tmp_path):
    e = WinValueEstimator()
    for _ in range(10): e.update(2, True)
    e.save(tmp_path / "wv.json")
    e2 = WinValueEstimator.load(tmp_path / "wv.json")
    assert e2.win_value(2) == e.win_value(2) and e2.attempts(2) == 10
