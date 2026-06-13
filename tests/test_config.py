import json

import pytest

from chessrl.config.config import RunConfig


def test_defaults():
    cfg = RunConfig()
    assert cfg.network.blocks == 6
    assert cfg.network.filters == 64
    assert cfg.mcts.simulations == 200
    assert cfg.mcts.dirichlet_alpha == 0.3
    assert cfg.selfplay.ply_cap == 512
    assert cfg.selfplay.resign_threshold == -0.95
    assert cfg.training.samples_per_position == 2.0


def test_yaml_partial_override(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text("run_name: tiny\nnetwork:\n  blocks: 2\nmcts:\n  simulations: 8\n")
    cfg = RunConfig.from_yaml(p)
    assert cfg.run_name == "tiny"
    assert cfg.network.blocks == 2
    assert cfg.network.filters == 64      # untouched default survives
    assert cfg.mcts.simulations == 8
    assert cfg.mcts.c_puct == 1.5


def test_json_round_trip(tmp_path):
    cfg = RunConfig(run_name="rt")
    p = tmp_path / "config.json"
    p.write_text(cfg.to_json())
    cfg2 = RunConfig.from_json(p)
    assert cfg2 == cfg
    assert json.loads(cfg.to_json())["run_name"] == "rt"


def test_goal_config_defaults_and_modes():
    from chessrl.config.config import GoalConfig
    g = GoalConfig()
    assert g.goal_mode == "none"                 # vanilla default
    assert g.win_floor == 0.2
    assert g.lp_window == 200
    assert g.novelty_beta > 0
    assert GoalConfig(goal_mode="lp").goal_mode == "lp"
    with pytest.raises(ValueError):
        GoalConfig(goal_mode="bogus")            # validated in __post_init__


def test_goal_config_wired_into_runconfig():
    cfg = RunConfig()
    assert cfg.goal.goal_mode == "none"
    assert cfg.goal.win_floor == 0.2


def test_goal_config_yaml_override(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text("goal:\n  goal_mode: lp\n  win_floor: 0.3\n")
    cfg = RunConfig.from_yaml(p)
    assert cfg.goal.goal_mode == "lp"
    assert cfg.goal.win_floor == 0.3
    assert cfg.goal.lp_window == 200              # untouched default survives


def test_goal_config_json_round_trip(tmp_path):
    from chessrl.config.config import GoalConfig
    cfg = RunConfig(run_name="g", goal=GoalConfig(goal_mode="random", win_floor=0.25))
    p = tmp_path / "config.json"
    p.write_text(cfg.to_json())
    cfg2 = RunConfig.from_json(p)
    assert cfg2 == cfg
    assert cfg2.goal.goal_mode == "random"
