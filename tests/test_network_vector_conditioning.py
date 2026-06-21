import pytest
from chessrl.config.config import NetworkConfig


def test_goal_cond_defaults_to_planes():
    assert NetworkConfig().goal_cond == "planes"


def test_goal_cond_accepts_vector():
    assert NetworkConfig(goal_cond="vector").goal_cond == "vector"


def test_goal_cond_rejects_unknown():
    with pytest.raises(ValueError):
        NetworkConfig(goal_cond="bogus")
