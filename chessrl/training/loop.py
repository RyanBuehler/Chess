"""Single-process train loop (M4): generate -> pace -> train -> checkpoint.
This module IS the smoke pipeline; scripts/train.py is a thin wrapper."""
import argparse
import json
import random
import time
from pathlib import Path

import chess.pgn
import numpy as np
import torch

from chessrl.config.config import RunConfig
from chessrl.model.network import NetEvaluator, PolicyValueNet
from chessrl.selfplay.pgn_io import save_pgn
from chessrl.selfplay.play import play_game
from chessrl.training.buffer import ReplayBuffer
from chessrl.training.provenance import build_provenance
from chessrl.training.trainer import Trainer


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config path (new run)")
    ap.add_argument("--resume", help="run directory name under runs-root")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args(argv)

    if args.resume:
        run_dir = Path(args.runs_root) / args.resume
        cfg = RunConfig.from_json(run_dir / "config.json")
        state = json.loads((run_dir / "state.json").read_text())
        game_no, total_positions = state["games"], state["positions"]
    else:
        cfg = RunConfig.from_yaml(args.config) if args.config else RunConfig()
        run_dir = Path(args.runs_root) / f"{cfg.run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
        (run_dir / "games").mkdir(parents=True)
        (run_dir / "config.json").write_text(cfg.to_json())
        (run_dir / "provenance.json").write_text(json.dumps(build_provenance(cfg), indent=2))
        game_no = total_positions = 0

    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + game_no)  # don't replay identical games on resume

    net = PolicyValueNet(cfg.network)
    trainer = Trainer(net, cfg.training, run_dir)
    buffer = ReplayBuffer(cfg.training.buffer_size)
    if args.resume:
        ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
        if ckpts:
            trainer.load_checkpoint(ckpts[-1])
        buffer = ReplayBuffer.from_run_dir(run_dir, cfg.training.buffer_size)
    evaluator = NetEvaluator(net, trainer.device)

    metrics_path = run_dir / "metrics.jsonl"
    for it in range(args.iterations):
        for _ in range(cfg.selfplay.games_per_iteration):
            rec, final_board, z = play_game(evaluator, cfg.mcts, cfg.selfplay, rng)
            buffer.add_game(rec)
            rec.save(run_dir / "games" / f"game_{game_no:07d}.npz")
            save_pgn(final_board, z, run_dir / "games" / f"game_{game_no:07d}.pgn")
            total_positions += len(rec)
            game_no += 1
        metrics = {"iteration": it, "games": game_no, "positions": total_positions}
        n = trainer.allowed_steps(total_positions)
        if n > 0 and len(buffer) >= cfg.training.batch_size:
            metrics.update(trainer.train_steps(buffer, n, rng))
        with metrics_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        trainer.save_checkpoint()
        (run_dir / "state.json").write_text(
            json.dumps({"games": game_no, "positions": total_positions})
        )
        print(metrics)
    return run_dir
