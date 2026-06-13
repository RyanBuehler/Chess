"""Single-process train loop (M4): generate -> pace -> train -> checkpoint.
This module IS the smoke pipeline; scripts/train.py is a thin wrapper.

Also hosts the **wishful-thinking thermometer** (spec sec 11/16, plan Task 3.4):
per goal kind, the self-play achievement rate and (when held-out vs-Stockfish
data is present) the self-play-minus-Stockfish achievement gap — a pre-registered
optimism diagnostic.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

from chessrl.config.config import RunConfig
from chessrl.goals.verifier import achieved_by_deadline
from chessrl.model.network import NetEvaluator, PolicyValueNet
from chessrl.selfplay.pgn_io import save_pgn
from chessrl.selfplay.play import play_game
from chessrl.selfplay.records import GameRecord, deserialize_goal
from chessrl.training.buffer import ReplayBuffer
from chessrl.training.her import reconstruct_states
from chessrl.training.provenance import build_provenance
from chessrl.training.trainer import Trainer


def goal_achievement_rates(records) -> dict:
    """Per-goal-kind self-play achievement rate over a set of goal records.

    For each side of each game, take the side's *assigned* goal and ask the
    verifier whether it was achieved by its deadline (start_ply == 0, the side's
    pursuit origin). Aggregate (achieved, attempts) per goal kind. Vanilla (no
    goals) records are skipped. Returns ``{kind: rate}`` over kinds attempted.
    """
    achieved = defaultdict(int)
    attempts = defaultdict(int)
    for rec in records:
        if not rec.has_goals():
            continue
        states = reconstruct_states(rec)
        # The assigned goal is immutable per side; read it off each side's first
        # ply (White at ply 0, Black at ply 1). Fall back gracefully on short games.
        seen = {}
        for t in range(len(rec)):
            proto = chess.WHITE if rec.protagonist[t] == 1 else chess.BLACK
            if proto in seen:
                continue
            goal = deserialize_goal(str(rec.assigned_blob[t]))
            seen[proto] = (t, goal)
        for proto, (start_ply, goal) in seen.items():
            ok, _ = achieved_by_deadline(states, goal, proto, start_ply)
            attempts[goal.kind] += 1
            achieved[goal.kind] += 1 if ok else 0
    return {k: achieved[k] / attempts[k] for k in attempts if attempts[k] > 0}


def wishful_thinking_thermometer(self_play_rates: dict, stockfish_rates: dict | None = None) -> dict:
    """Assemble the thermometer metric (spec sec 11/16, plan Task 3.4).

    Returns ``{kind: {"self_play": rate, "vs_stockfish": rate|None,
    "gap": rate|None}}``. The gap (self-play minus held-out vs-Stockfish
    achievement rate) flags optimism; it is only populated where vs-Stockfish
    data is present for that kind.
    """
    out = {}
    sf = stockfish_rates or {}
    for kind, sp in self_play_rates.items():
        vs = sf.get(kind)
        out[kind] = {
            "self_play": sp,
            "vs_stockfish": vs,
            "gap": (sp - vs) if vs is not None else None,
        }
    return out


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
