from chessrl.config.config import GoalConfig, RunConfig


def test_emergent_chained_mode_and_defaults():
    g = GoalConfig(goal_mode="emergent_chained")
    assert g.goal_select_temp == 0.5
    assert g.win_ramp == 0.6
    assert g.alpha_resign_gate == 0.05
    assert g.endgame_margin == 20


def test_v3_yaml_loads():
    c = RunConfig.from_yaml("experiments/v3-zenith-w8.yaml")
    assert c.goal.goal_mode == "emergent_chained"
    assert c.goal.goal_window == 8
    assert c.goal.min_reservoir == 1500 and c.goal.delta_samples_per_game == 16
