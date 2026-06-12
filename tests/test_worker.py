import json

import numpy as np

from chessrl.config.config import RunConfig
from chessrl.selfplay.worker import (
    next_counter_for_worker,
    run_one_batch,
)


def _write_config(run_dir):
    cfg = RunConfig.from_dict(
        {
            "run_name": "wtest",
            "network": {"blocks": 1, "filters": 8},
            "mcts": {"simulations": 8, "temperature_moves": 4, "leaves_per_tree": 2},
            "selfplay": {"ply_cap": 20, "concurrent_games": 2, "resign_playout_fraction": 0.0},
            "training": {"batch_size": 16, "device": "cpu", "selfplay_device": "cpu"},
        }
    )
    (run_dir / "games").mkdir(parents=True)
    (run_dir / "config.json").write_text(cfg.to_json())
    return cfg


def test_next_counter_for_worker_scans_existing(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    (games / "game_w03_0000000.npz").write_bytes(b"x")
    (games / "game_w03_0000004.npz").write_bytes(b"x")
    (games / "game_w07_0000009.npz").write_bytes(b"x")  # different worker, ignored
    assert next_counter_for_worker(tmp_path, worker_id=3) == 5
    assert next_counter_for_worker(tmp_path, worker_id=5) == 0


def test_run_one_batch_writes_games_pgn_and_meta(tmp_path):
    cfg = _write_config(tmp_path)
    rng = np.random.default_rng(0)

    # Build a fresh batched evaluator the same way the worker does at cold start.
    from chessrl.model.network import BatchedNetEvaluator, PolicyValueNet
    import torch

    torch.manual_seed(123)
    net = PolicyValueNet(cfg.network)
    evaluator = BatchedNetEvaluator(net, device="cpu")

    counter = run_one_batch(
        run_dir=tmp_path, worker_id=2, evaluator=evaluator,
        cfg=cfg, rng=rng, start_counter=0,
    )
    npz = sorted((tmp_path / "games").glob("game_w02_*.npz"))
    pgn = sorted((tmp_path / "games").glob("game_w02_*.pgn"))
    assert len(npz) == 2
    assert len(pgn) == 2
    assert counter == 2  # next free counter after writing 2 games

    meta_path = tmp_path / "games_meta_w02.jsonl"
    assert meta_path.exists()
    lines = meta_path.read_text().splitlines()
    assert len(lines) == 2
    m = json.loads(lines[0])
    for key in ("plies", "z", "resigned", "playout", "would_resign", "fp", "game"):
        assert key in m
