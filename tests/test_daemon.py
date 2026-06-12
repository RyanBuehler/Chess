import json
from pathlib import Path

import numpy as np
import torch

from chessrl.config.config import EvalConfig, NetworkConfig, TrainingConfig
from chessrl.evaluation.daemon import (
    evaluate_checkpoint,
    eligible_checkpoints,
    run_once,
)
from chessrl.evaluation.store import LadderStore
from chessrl.model.network import PolicyValueNet
from chessrl.training.trainer import Trainer


def _make_run_with_checkpoint(tmp_path, run_name="r1", step_target=1):
    """Create runs/<run> with config.json and one tiny checkpoint."""
    run_dir = tmp_path / "runs" / run_name
    (run_dir / "checkpoints").mkdir(parents=True)
    net_cfg = NetworkConfig(blocks=2, filters=8)
    from chessrl.config.config import RunConfig
    cfg = RunConfig(network=net_cfg)
    (run_dir / "config.json").write_text(cfg.to_json())
    torch.manual_seed(0)
    net = PolicyValueNet(net_cfg)
    trainer = Trainer(net, TrainingConfig(batch_size=4, device="cpu"), run_dir)
    ckpt = trainer.save_checkpoint()       # ckpt_00000000.pt
    return run_dir, ckpt, net_cfg


def test_evaluate_checkpoint_records_results_and_elo(tmp_path):
    run_dir, ckpt, net_cfg = _make_run_with_checkpoint(tmp_path)
    store = LadderStore(tmp_path / "runs" / "ladder.sqlite")
    cfg = EvalConfig(games_per_rung=2, agent_simulations=8, max_plies=40, stockfish_path="")

    elo = evaluate_checkpoint(run_dir, ckpt, cfg, store, openings_offset=0)

    # Floors-only ladder: random, greedy, minimax2 -> 3 rungs * 2 games = 6 results.
    assert len(store.all_results()) == 6
    # the agent player and the three floor players are registered
    players = store.all_players()
    assert any("@" in name for name in players)            # agent {run_id}@{step}
    assert {"random", "greedy", "minimax2"} <= set(players)
    # elo.jsonl appended with this checkpoint
    elo_lines = (run_dir / "elo.jsonl").read_text().splitlines()
    assert len(elo_lines) == 1
    entry = json.loads(elo_lines[0])
    for key in ("ts", "step", "ckpt", "elo", "nu"):
        assert key in entry
    assert entry["step"] == 0
    assert np.isfinite(entry["elo"])
    # PGNs saved
    pgns = list((run_dir / "eval_games").glob("*.pgn"))
    assert len(pgns) == 6
    # checkpoint marked evaluated
    assert store.is_evaluated(str(ckpt))


def test_eligible_checkpoints_respects_every_n(tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "checkpoints").mkdir(parents=True)
    for step in (0, 1000, 2000, 3000, 4000, 5000):
        (run_dir / "checkpoints" / f"ckpt_{step:08d}.pt").write_bytes(b"x")
    store = LadderStore(tmp_path / "runs" / "ladder.sqlite")
    cfg = EvalConfig(every_n_checkpoints=2)
    # every 2nd checkpoint by index: indices 0,2,4 -> steps 0,2000,4000
    elig = eligible_checkpoints(run_dir, cfg, store)
    steps = [int(Path(c).stem.split("_")[1]) for c in elig]
    assert steps == [0, 2000, 4000]
    # after marking one evaluated it is skipped
    store.mark_evaluated(str(run_dir / "checkpoints" / "ckpt_00000000.pt"))
    elig2 = eligible_checkpoints(run_dir, cfg, store)
    steps2 = [int(Path(c).stem.split("_")[1]) for c in elig2]
    assert steps2 == [2000, 4000]


def test_run_once_evaluates_latest_eligible_then_returns(tmp_path):
    run_dir, ckpt, net_cfg = _make_run_with_checkpoint(tmp_path)
    runs_root = tmp_path / "runs"
    cfg = EvalConfig(every_n_checkpoints=1, games_per_rung=2, agent_simulations=8,
                     max_plies=40, stockfish_path="")
    n_eval = run_once(runs_root, cfg, run_filter=None)
    assert n_eval == 1
    store = LadderStore(runs_root / "ladder.sqlite")
    assert store.is_evaluated(str(ckpt))
    assert (run_dir / "elo.jsonl").exists()
    # a second --once pass finds nothing new
    n_eval2 = run_once(runs_root, cfg, run_filter=None)
    assert n_eval2 == 0
