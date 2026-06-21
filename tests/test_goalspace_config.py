import pytest
from chessrl.config.config import GoalConfig


def test_goalspace_defaults():
    c = GoalConfig()
    assert c.cluster_k == 48
    assert c.refresh_every == 2000
    assert c.reservoir_size == 20000
    assert c.min_reservoir == 5000
    assert c.goal_window == 8


def test_emergent_mode_allowed():
    assert GoalConfig(goal_mode="emergent").goal_mode == "emergent"


def test_bad_mode_still_rejected():
    with pytest.raises(ValueError):
        GoalConfig(goal_mode="nonsense")
