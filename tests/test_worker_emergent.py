import numpy as np
from chessrl.config.config import RunConfig
from chessrl.selfplay import worker as W


def test_build_evaluator_emergent_returns_vector(tmp_path):
    from pathlib import Path
    cfg = RunConfig.from_yaml(Path("experiments/v2-stage1.yaml"))
    ev = W._build_evaluator(tmp_path, cfg, "cpu", seed=0)
    from chessrl.model.network import VectorGoalNetEvaluator
    assert isinstance(ev, VectorGoalNetEvaluator)


def test_load_goalspace_none_when_absent(tmp_path):
    from pathlib import Path
    cfg = RunConfig.from_yaml(Path("experiments/v2-stage1.yaml"))
    assert W._load_goalspace(tmp_path, cfg, "cpu") is None
