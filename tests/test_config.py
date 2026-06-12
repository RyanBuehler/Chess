import json
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
