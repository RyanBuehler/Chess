"""Parallel training loop (M5): self-play worker processes generate games while
the main process ingests them, paces training, and checkpoints.

Spawn start method everywhere. worker_main lives in chessrl.selfplay.worker so
spawn can re-import it; main() must not run at import time. A run_dir/STOP
sentinel file signals workers to stop.
"""
import argparse
import json
import multiprocessing as mp
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from chessrl.config.config import RunConfig
from chessrl.model.network import PolicyValueNet
from chessrl.selfplay.worker import worker_main
from chessrl.training.buffer import GoalReplayBuffer, ReplayBuffer
from chessrl.training.loop import goal_achievement_rates, wishful_thinking_thermometer
from chessrl.training.provenance import build_provenance
from chessrl.training.trainer import Trainer


def make_run_dir(cfg: RunConfig, runs_root) -> Path:
    run_dir = Path(runs_root) / f"{cfg.run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "games").mkdir(parents=True)
    (run_dir / "config.json").write_text(cfg.to_json())
    (run_dir / "provenance.json").write_text(json.dumps(build_provenance(cfg), indent=2))
    return run_dir


def ingest_new_games(run_dir, buffer, ingested: set, recent_records: deque | None = None) -> tuple:
    """Add any .npz games not yet ingested. Returns (games_added, positions_added).
    Files that fail to load (half-written) are skipped and retried next pass.

    When ``recent_records`` is given (goal runs), the loaded records are also
    appended there for the wishful-thinking thermometer (a bounded window)."""
    from chessrl.selfplay.records import GameRecord

    games_dir = Path(run_dir) / "games"
    new_files = sorted(
        (p for p in games_dir.glob("*.npz") if p.name not in ingested),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    games_added = positions_added = 0
    for f in new_files:
        try:
            rec = GameRecord.load(f)
        except Exception:
            continue  # half-written; leave un-ingested for a later pass
        buffer.add_game(rec)
        ingested.add(f.name)
        games_added += 1
        positions_added += len(rec)
        if recent_records is not None and rec.has_goals():
            recent_records.append(rec)
    return games_added, positions_added


def load_stockfish_achievement_rates(run_dir) -> dict | None:
    """Read held-out vs-Stockfish per-goal achievement rates if the evaluator has
    written them (``goal_eval.json`` -> ``{"stockfish_rates": {kind: rate}}``).
    Returns None when absent so the thermometer omits the gap (spec sec 11/16)."""
    path = Path(run_dir) / "goal_eval.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    rates = data.get("stockfish_rates")
    return rates if isinstance(rates, dict) and rates else None


def aggregate_resign_fp(run_dir) -> dict:
    """Aggregate resignation false-positive stats from all worker meta files."""
    playout = fp = 0
    for meta_file in Path(run_dir).glob("games_meta_w*.jsonl"):
        for line in meta_file.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            if m.get("playout"):
                playout += 1
                if m.get("fp"):
                    fp += 1
    rate = (fp / playout) if playout else 0.0
    return {"playout_games": playout, "false_positives": fp, "resign_fp_rate": rate}


def _spawn_worker(ctx, worker_id, run_dir, stop_path, device):
    p = ctx.Process(
        target=worker_main,
        args=(worker_id, str(run_dir), str(stop_path), device),
        daemon=False,
    )
    p.start()
    return p


def main(argv=None) -> Path:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set (e.g. called from a test that already set it)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config path (new run)")
    ap.add_argument("--resume", help="run directory name under runs-root")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--games", type=int, default=200, help="total NEW games this invocation")
    args = ap.parse_args(argv)

    if args.resume:
        run_dir = Path(args.runs_root) / args.resume
        cfg = RunConfig.from_json(run_dir / "config.json")
        state = json.loads((run_dir / "state.json").read_text())
        total_positions = state["positions"]
        baseline_games = state["games"]
    else:
        cfg = RunConfig.from_yaml(args.config) if args.config else RunConfig()
        run_dir = make_run_dir(cfg, args.runs_root)
        total_positions = 0
        baseline_games = 0

    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + baseline_games)

    goal_mode = cfg.goal.goal_mode != "none"
    net = PolicyValueNet(cfg.network, goal_conditioned=goal_mode)
    trainer = Trainer(net, cfg.training, run_dir)
    # Goal runs use the HER goal buffer + BCE training; vanilla is unchanged.
    if goal_mode:
        buffer = GoalReplayBuffer(cfg.training.buffer_size, deadline_max=cfg.goal.deadline_max)
    else:
        buffer = ReplayBuffer(cfg.training.buffer_size)
    # Bounded window of recent goal records for the wishful-thinking thermometer.
    recent_records: deque | None = deque(maxlen=512) if goal_mode else None
    ingested: set = set()
    if args.resume:
        ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
        if ckpts:
            trainer.load_checkpoint(ckpts[-1])
        if goal_mode:
            buffer = GoalReplayBuffer.from_run_dir(
                run_dir, cfg.training.buffer_size, deadline_max=cfg.goal.deadline_max
            )
        else:
            buffer = ReplayBuffer.from_run_dir(run_dir, cfg.training.buffer_size)
        for f in (run_dir / "games").glob("*.npz"):
            ingested.add(f.name)

    stop_path = run_dir / "STOP"
    if stop_path.exists():
        stop_path.unlink()

    ctx = mp.get_context("spawn")
    procs = [
        _spawn_worker(ctx, wid, run_dir, stop_path, cfg.training.selfplay_device)
        for wid in range(cfg.selfplay.workers)
    ]

    metrics_path = run_dir / "metrics.jsonl"
    games_seen = 0
    restarts = 0
    last_ckpt_bucket = trainer.step // cfg.training.checkpoint_every_steps
    start = time.time()

    try:
        while games_seen < args.games:
            added, positions = ingest_new_games(run_dir, buffer, ingested, recent_records)
            games_seen += added
            total_positions += positions

            steps_done = 0
            n = trainer.allowed_steps(total_positions)
            if n > 0 and len(buffer) >= cfg.training.batch_size:
                m = (
                    trainer.train_steps_goal(buffer, n, rng)
                    if goal_mode
                    else trainer.train_steps(buffer, n, rng)
                )
                steps_done = n
                bucket = trainer.step // cfg.training.checkpoint_every_steps
                if bucket > last_ckpt_bucket:
                    trainer.save_checkpoint()
                    last_ckpt_bucket = bucket
            else:
                m = {"policy_loss": None, "value_loss": None, "step": trainer.step}

            # restart any dead worker
            for i, p in enumerate(procs):
                if not p.is_alive():
                    procs[i] = _spawn_worker(
                        ctx, i, run_dir, stop_path, cfg.training.selfplay_device
                    )
                    restarts += 1

            elapsed = max(time.time() - start, 1e-9)
            fp_stats = aggregate_resign_fp(run_dir)
            metrics = {
                "games": baseline_games + games_seen,
                "new_games": games_seen,
                "positions": total_positions,
                "step": trainer.step,
                "steps_this_cycle": steps_done,
                "policy_loss": m.get("policy_loss"),
                "value_loss": m.get("value_loss"),
                "games_per_hour": games_seen / elapsed * 3600.0,
                "worker_restarts": restarts,
                "resign_fp_rate": fp_stats["resign_fp_rate"],
            }
            if goal_mode and recent_records:
                sp_rates = goal_achievement_rates(recent_records)
                sf_rates = load_stockfish_achievement_rates(run_dir)
                metrics["goal_achievement_rate"] = sp_rates
                metrics["wishful_thinking"] = wishful_thinking_thermometer(sp_rates, sf_rates)
            with metrics_path.open("a") as f:
                f.write(json.dumps(metrics) + "\n")
            # Flush counters every cycle so a hard crash (which skips finally)
            # still resumes with honest pacing; the buffer rebuilds from disk.
            (run_dir / "state.json").write_text(
                json.dumps({"games": baseline_games + games_seen, "positions": total_positions})
            )

            if added == 0 and steps_done == 0:
                time.sleep(1.0)
    finally:
        stop_path.write_text("stop")
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        # final drain of any games written during shutdown
        added, positions = ingest_new_games(run_dir, buffer, ingested)
        games_seen += added
        total_positions += positions
        trainer.save_checkpoint()
        (run_dir / "state.json").write_text(
            json.dumps({"games": baseline_games + games_seen, "positions": total_positions})
        )
        if stop_path.exists():
            stop_path.unlink()

    return run_dir
