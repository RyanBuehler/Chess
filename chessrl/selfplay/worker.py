"""Self-play worker process: spawn-safe, sentinel-file controlled.

worker_main is a top-level function so the spawn start method can re-import and
call it. It polls run_dir/config.json's run config, loads the newest checkpoint
when one appears, plays batches of concurrent games, and writes sparse records,
PGNs, and per-game meta lines. A run_dir/STOP sentinel file (not a
multiprocessing.Event) signals shutdown -- simpler across spawn and debuggable.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from chessrl.config.config import RunConfig
from chessrl.goals.assignment import make_assigner
from chessrl.goals.curriculum import Curriculum
from chessrl.goals.repertoire import Repertoire
from chessrl.model.network import (
    BatchedGoalNetEvaluator,
    BatchedNetEvaluator,
    PolicyValueNet,
)
from chessrl.selfplay.concurrent import (
    play_games_concurrent,
    play_goal_games_concurrent,
)
from chessrl.selfplay.pgn_io import save_pgn


def next_counter_for_worker(run_dir, worker_id: int) -> int:
    """Highest existing counter for this worker id + 1 (0 if none). Makes the
    counter collision-proof across process restarts."""
    prefix = f"game_w{worker_id:02d}_"
    best = -1
    for f in (Path(run_dir) / "games").glob(f"{prefix}*.npz"):
        stem = f.stem  # game_wWW_CCCCCCC
        try:
            best = max(best, int(stem[len(prefix):]))
        except ValueError:
            continue
    return best + 1


def _resolve_device(requested: str) -> str:
    return requested if (requested == "cpu" or torch.cuda.is_available()) else "cpu"


def _newest_checkpoint(run_dir) -> Path | None:
    ckpts = sorted((Path(run_dir) / "checkpoints").glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


REPERTOIRE_FILE = "repertoire.json"


def _load_curriculum(run_dir, cfg: RunConfig):
    """Build an LP ``Curriculum`` from the run's persisted repertoire snapshot.

    Returns ``None`` for non-lp modes (the assigner falls back to random) and
    when no snapshot exists yet (early in a fresh lp run). Workers reload this on
    the same cadence as the net checkpoint (spec sec 14 / plan Task 4.3)."""
    if cfg.goal.goal_mode != "lp":
        return None
    path = Path(run_dir) / REPERTOIRE_FILE
    if not path.exists():
        return None
    rep = Repertoire.load_or_new(
        path, lp_window=cfg.goal.lp_window, deadline_max=cfg.goal.deadline_max
    )
    return Curriculum(
        rep,
        novelty_beta=cfg.goal.novelty_beta,
        min_attempts_for_lp=cfg.goal.min_attempts_for_lp,
        win_floor=cfg.goal.win_floor,
    )


def _build_evaluator(run_dir, cfg: RunConfig, device: str, seed: int):
    """Newest checkpoint if present, else a fresh net seeded identically across
    workers so a cold-start run begins from the same weights everywhere.

    Returns a BatchedNetEvaluator for vanilla (goal_mode=none), or a
    BatchedGoalNetEvaluator (batched goal-conditioned net) for goal modes — the
    batched evaluator drives the concurrent goal self-play driver."""
    goal_mode = cfg.goal.goal_mode != "none"
    ckpt = _newest_checkpoint(run_dir)
    if goal_mode:
        if ckpt is not None:
            return BatchedGoalNetEvaluator.from_checkpoint(ckpt, cfg.network, device=device)
        torch.manual_seed(seed)
        net = PolicyValueNet(cfg.network, goal_conditioned=True)
        return BatchedGoalNetEvaluator(net, device=device)
    if ckpt is not None:
        return BatchedNetEvaluator.from_checkpoint(ckpt, cfg.network, device=device)
    torch.manual_seed(seed)
    net = PolicyValueNet(cfg.network)
    return BatchedNetEvaluator(net, device=device)


def _run_goal_batch(
    evaluator, cfg: RunConfig, rng: np.random.Generator, curriculum=None,
    publisher=None, game_id_prefix: str = "",
) -> list:
    """Play one batch of goal-conditioned games CONCURRENTLY, batching leaf
    evaluations across all in-flight games/sides into one BatchedGoalNetEvaluator
    call per search round (spec sec 7/10). Returns list[(GameRecord, final_board,
    z, meta)] in the same shape as play_games_concurrent. Meta adds goal
    diagnostics: win_ply_fraction (the per-game control variable, spec sec 7/16).
    ``curriculum`` (lp mode) is the LP sampler built from the reloaded repertoire
    snapshot. Each game reproduces play_goal_game's semantics exactly. ``publisher``
    and ``game_id_prefix`` thread the live feed through, mirroring the vanilla path."""
    assigner = make_assigner(cfg.goal, rng, curriculum=curriculum)
    return play_goal_games_concurrent(
        evaluator, cfg.mcts, cfg.selfplay, cfg.goal, rng,
        num_games=cfg.selfplay.concurrent_games,
        assigner=assigner,
        publisher=publisher,
        game_id_prefix=game_id_prefix,
    )


def run_one_batch(
    run_dir, worker_id: int, evaluator, cfg: RunConfig,
    rng: np.random.Generator, start_counter: int, publisher=None, batch_index: int = 0,
    curriculum=None,
) -> int:
    """Play one batch of concurrent_games games, persist them, append meta.
    Returns the next free counter. Both goal and vanilla modes play the batch
    CONCURRENTLY (many games advanced in lockstep through one batched MCTS): goal
    modes via play_goal_games_concurrent, vanilla via play_games_concurrent. Both
    thread the live-feed publisher and per-game id prefix through."""
    if cfg.goal.goal_mode != "none":
        results = _run_goal_batch(
            evaluator, cfg, rng, curriculum=curriculum,
            publisher=publisher,
            game_id_prefix=f"w{worker_id:02d}_b{batch_index}_",
        )
    else:
        results = play_games_concurrent(
            evaluator, cfg.mcts, cfg.selfplay, rng,
            num_games=cfg.selfplay.concurrent_games,
            publisher=publisher,
            game_id_prefix=f"w{worker_id:02d}_b{batch_index}_",
        )
    games_dir = Path(run_dir) / "games"
    meta_path = Path(run_dir) / f"games_meta_w{worker_id:02d}.jsonl"
    counter = start_counter
    with meta_path.open("a") as mf:
        for rec, final_board, z, meta in results:
            name = f"game_w{worker_id:02d}_{counter:07d}"
            rec.save(games_dir / f"{name}.npz")
            save_pgn(final_board, z, games_dir / f"{name}.pgn")
            line = dict(meta)
            line["game"] = name
            line["worker"] = worker_id
            mf.write(json.dumps(line) + "\n")
            counter += 1
    return counter


def worker_main(worker_id: int, run_dir: str, stop_path: str, device: str) -> None:
    from chessrl.selfplay.feed import FeedPublisher, NullPublisher

    run_dir = Path(run_dir)
    stop_path = Path(stop_path)
    cfg = RunConfig.from_json(run_dir / "config.json")
    resolved_device = _resolve_device(device)
    rng = np.random.default_rng(cfg.training.seed + 1000 * worker_id)

    counter = next_counter_for_worker(run_dir, worker_id)
    evaluator = _build_evaluator(run_dir, cfg, resolved_device, cfg.training.seed)
    loaded_ckpt: Path | None = _newest_checkpoint(run_dir)
    # LP curriculum snapshot (lp mode only); reloaded on the checkpoint cadence.
    curriculum = _load_curriculum(run_dir, cfg)
    rep_path = run_dir / REPERTOIRE_FILE
    rep_mtime = rep_path.stat().st_mtime if rep_path.exists() else None

    publisher = NullPublisher()
    if cfg.selfplay.feed_port > 0:
        try:
            publisher = FeedPublisher(cfg.selfplay.feed_port + worker_id)
        except Exception:
            publisher = NullPublisher()             # feed is best-effort, never fatal

    batch_index = 0
    try:
        while not stop_path.exists():
            newest = _newest_checkpoint(run_dir)
            if newest is not None and newest != loaded_ckpt:
                try:
                    if cfg.goal.goal_mode != "none":
                        evaluator = BatchedGoalNetEvaluator.from_checkpoint(
                            newest, cfg.network, device=resolved_device
                        )
                    else:
                        evaluator = BatchedNetEvaluator.from_checkpoint(
                            newest, cfg.network, device=resolved_device
                        )
                    loaded_ckpt = newest
                except Exception:
                    # Half-written or vanishing checkpoint: keep playing with the
                    # current net and retry on the next loop. Never crash a worker
                    # over this - spawn restarts cost 10-20s each.
                    pass
            # Reload the repertoire snapshot when the trainer has rewritten it
            # (lp mode). Same best-effort discipline as the checkpoint reload.
            if cfg.goal.goal_mode == "lp" and rep_path.exists():
                try:
                    m = rep_path.stat().st_mtime
                    if m != rep_mtime:
                        curriculum = _load_curriculum(run_dir, cfg)
                        rep_mtime = m
                except Exception:
                    pass
            counter = run_one_batch(
                run_dir, worker_id, evaluator, cfg, rng, counter,
                publisher=publisher, batch_index=batch_index, curriculum=curriculum,
            )
            batch_index += 1
            # tight loop is fine; the sentinel check between batches paces shutdown.
            time.sleep(0.01)
    finally:
        publisher.close()
