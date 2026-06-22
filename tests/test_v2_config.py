from pathlib import Path
from chessrl.config.config import RunConfig


def test_v2_yaml_loads_emergent_vector():
    cfg = RunConfig.from_yaml(Path("experiments/v2-stage1.yaml"))
    assert cfg.goal.goal_mode == "emergent"
    assert cfg.network.goal_cond == "vector"
    assert cfg.goal.cluster_k > 0
    assert cfg.goal.goal_window > 0
