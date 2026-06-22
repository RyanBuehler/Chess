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
from chessrl.goals.goalspace import GoalSpace
from chessrl.goals.repertoire import Repertoire
from chessrl.goals.winvalue import WinValueEstimator
from chessrl.model.network import PolicyValueNet, VectorGoalNetEvaluator
from chessrl.selfplay.worker import worker_main
from chessrl.training.buffer import GoalReplayBuffer, ReplayBuffer
from chessrl.training.her import reconstruct_states
from chessrl.training.loop import (
    goal_achievement_rates,
    repertoire_learning_progress,
    win_ply_fraction,
    wishful_thinking_thermometer,
)
from chessrl.training.provenance import build_provenance
from chessrl.training.trainer import Trainer
from chessrl.training.vector_buffer import VectorGoalReplayBuffer


GOALSPACE_DIR = "goalspace"
FROZEN_ENCODER = "frozen_encoder.pt"
WINVALUE_FILE = "winvalue.json"


def update_winvalue_from_record(estimator, rec) -> None:
    """Update the interventional win-value estimator from one EXPLORE game: for
    each side whose game-assigned goal was epsilon-explore (and a real cluster),
    credit a win/loss by that side's outcome. z_white = the White-frame result.

    Side-to-move is inferred from ply parity (ply 0 = White, ply 1 = Black, ...)
    when the protagonist field is absent (cluster-only records from emergent mode);
    falls back to rec.protagonist when present (v1 goal records)."""
    if not rec.has_cluster_goals():
        return
    import chess

    def _is_white(i: int) -> bool:
        if rec.protagonist is not None:
            return int(rec.protagonist[i]) == 1
        # Cluster records omit protagonist; derive from ply parity (White=0,2,4,…).
        return (i % 2) == 0

    # White-frame z: outcomes[i] is from side-to-move; flip for Black.
    z_white = None
    for i in range(len(rec)):
        z_white = int(rec.outcomes[i]) if _is_white(i) else -int(rec.outcomes[i])
        break
    if z_white is None:
        return
    # Per side: find its assigned cluster + explore flag from a ply where it moved.
    for white_side, won in ((True, z_white > 0), (False, z_white < 0)):
        for i in range(len(rec)):
            if _is_white(i) == white_side:
                if bool(rec.explore[i]) and int(rec.assigned_cluster[i]) >= 0:
                    estimator.update(int(rec.assigned_cluster[i]), won)
                break


def snapshot_frozen_encoder(net, run_dir, network_cfg, device) -> "VectorGoalNetEvaluator":
    """Save net.state_dict() to run_dir/frozen_encoder.pt (atomic) and return a
    VectorGoalNetEvaluator loaded from it."""
    import os
    path = Path(run_dir) / FROZEN_ENCODER
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"model": net.state_dict()}, tmp)
    os.replace(tmp, path)
    return VectorGoalNetEvaluator.from_checkpoint(path, network_cfg, device=device)


def observe_game_deltas(goalspace, rec, embedder, max_samples: int, rng) -> None:
    """Sample up to max_samples plies of a game, compute frozen-encoder window
    deltas e(s_{i+w}) - e(s_i), and add them to the GoalSpace reservoir."""
    states = reconstruct_states(rec)
    T = len(states) - 1
    w = goalspace.cfg.goal_window
    starts = [i for i in range(T) if i + w <= T]
    if not starts:
        return
    if len(starts) > max_samples:
        starts = [int(s) for s in rng.choice(starts, size=max_samples, replace=False)]
    emb = embedder.embed_boards([states[i] for i in starts] + [states[i + w] for i in starts])
    half = len(starts)
    for k in range(half):
        goalspace.observe_delta(emb[half + k] - emb[k])


def make_run_dir(cfg: RunConfig, runs_root, run_dir_name: str | None = None) -> Path:
    """Create a fresh run dir. By default ``runs_root/<run_name>-<timestamp>``.
    When ``run_dir_name`` is given the dir is named EXACTLY that (no timestamp),
    so an orchestrator can address an arm by its bare name; an existing dir is a
    hard error (we must never silently resume on top of a fresh launch)."""
    if run_dir_name is not None:
        run_dir = Path(runs_root) / run_dir_name
        if run_dir.exists():
            raise SystemExit(
                f"--run-dir-name {run_dir_name!r}: {run_dir} already exists; "
                f"refusing to overwrite. Use --resume {run_dir_name} to continue it."
            )
    else:
        run_dir = Path(runs_root) / f"{cfg.run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "games").mkdir(parents=True)
    (run_dir / "config.json").write_text(cfg.to_json())
    (run_dir / "provenance.json").write_text(json.dumps(build_provenance(cfg), indent=2))
    return run_dir


REPERTOIRE_FILE = "repertoire.json"


def ingest_new_games(
    run_dir, buffer, ingested: set, recent_records: deque | None = None,
    repertoire=None, on_record=None,
) -> tuple:
    """Add any .npz games not yet ingested. Returns (games_added, positions_added).
    Files that fail to load (half-written) are skipped and retried next pass.

    When ``recent_records`` is given (goal runs), the loaded records are also
    appended there for the wishful-thinking thermometer (a bounded window).

    When ``repertoire`` is given (lp mode, plan Task 4.3) each new record drives
    the repertoire feedback loop: mint first-seen deltas, update assigned-goal
    stats, spawn plateaued children. The caller persists the snapshot.

    When ``on_record`` is given (emergent mode), it is called with each freshly
    loaded GameRecord immediately after buffer ingestion. v1 callers pass None
    and are unaffected."""
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
        if repertoire is not None and rec.has_goals():
            repertoire.update_and_refine_from_record(rec)
        if on_record is not None:
            on_record(rec)
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


def worker_last_progress(run_dir, worker_id: int) -> float | None:
    """Most recent moment worker ``worker_id`` produced a game, as a unix
    timestamp, or None if it has produced nothing yet (Task 5.2).

    A worker appends one line to ``games_meta_w{WW}.jsonl`` and writes a
    ``game_w{WW}_*.npz`` per game; the max mtime across both is the worker's
    heartbeat. The meta file is the primary signal (always appended, even for
    zero-position games); the newest game .npz is a fallback before the first
    meta flush."""
    run_dir = Path(run_dir)
    mtimes = []
    meta = run_dir / f"games_meta_w{worker_id:02d}.jsonl"
    if meta.exists():
        mtimes.append(meta.stat().st_mtime)
    prefix = f"game_w{worker_id:02d}_"
    for f in (run_dir / "games").glob(f"{prefix}*.npz"):
        try:
            mtimes.append(f.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else None


def hung_workers(
    run_dir, worker_ids, last_seen: dict, now: float, heartbeat_seconds: float
) -> list[int]:
    """Return the ids of workers that are *hung*: alive but with no new game in
    the heartbeat window (Task 5.2).

    ``last_seen`` maps worker_id -> (progress_ts_or_None, observed_at_ts) and is
    mutated in place: when a worker's progress timestamp advances we refresh its
    observation; a worker is hung only when its progress has NOT advanced for
    longer than ``heartbeat_seconds``. Workers with no output yet are timed from
    when we first observed them (so a worker that never starts producing is also
    caught). ``heartbeat_seconds <= 0`` disables detection (returns [])."""
    if heartbeat_seconds <= 0:
        return []
    hung = []
    for wid in worker_ids:
        progress = worker_last_progress(run_dir, wid)
        prev = last_seen.get(wid)
        if prev is None or prev[0] != progress:
            # First observation or progress advanced: (re)start the clock.
            last_seen[wid] = (progress, now)
            continue
        # No advance since we last refreshed the clock at prev[1]; measure the
        # stall from THERE, not from the absolute last-game mtime (which for a
        # healthy worker mid-long-batch is far in the past and would falsely flag
        # it, and -- because it never moves -- would re-flag every loop iteration
        # even after a restart, causing a restart storm). prev[1] is reset to
        # `now` on first observation, on progress advance, and on restart, so the
        # fresh process always gets a full window before it can be judged hung.
        ref = prev[1]
        if now - ref >= heartbeat_seconds:
            hung.append(wid)
    return hung


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
    ap.add_argument("--run-dir-name", help="for a fresh --config run, name the run dir EXACTLY this (no timestamp); errors if it exists")
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
        run_dir = make_run_dir(cfg, args.runs_root, run_dir_name=args.run_dir_name)
        total_positions = 0
        baseline_games = 0

    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + baseline_games)

    goal_mode = cfg.goal.goal_mode != "none"
    emergent_mode = cfg.goal.goal_mode == "emergent"
    net = PolicyValueNet(cfg.network, goal_conditioned=goal_mode)
    trainer = Trainer(net, cfg.training, run_dir)

    # Emergent mode: vector FiLM net + frozen encoder + GoalSpace + VectorGoalReplayBuffer.
    # v1 goal modes and vanilla are unchanged.
    frozen_encoder = None
    goalspace = None
    winvalue = None
    if emergent_mode:
        frozen_encoder = snapshot_frozen_encoder(net, run_dir, cfg.network, trainer.device)
        goalspace = GoalSpace(cfg.goal, frozen_encoder, rng)
        buffer = VectorGoalReplayBuffer(
            cfg.training.buffer_size, frozen_encoder, goalspace,
            deadline_max=cfg.goal.deadline_max,
        )
        goalspace.save(run_dir / GOALSPACE_DIR)
        winvalue = WinValueEstimator()
    elif goal_mode:
        # Goal runs use the HER goal buffer + BCE training; vanilla is unchanged.
        buffer = GoalReplayBuffer(cfg.training.buffer_size, deadline_max=cfg.goal.deadline_max)
    else:
        buffer = ReplayBuffer(cfg.training.buffer_size)

    # Bounded window of recent goal records for the wishful-thinking thermometer
    # (v1 goal modes only; emergent does not use the repertoire/thermometer path).
    recent_records: deque | None = (deque(maxlen=512) if goal_mode and not emergent_mode else None)
    # LP curriculum repertoire (lp mode only): the trainer owns the canonical
    # snapshot, mints/updates it from new games, and persists it so workers can
    # reload it (plan Task 4.3). Resume reconstructs it from the persisted file.
    lp_mode = cfg.goal.goal_mode == "lp"
    repertoire = None
    if lp_mode:
        repertoire = Repertoire.load_or_new(
            run_dir / REPERTOIRE_FILE,
            lp_window=cfg.goal.lp_window,
            deadline_max=cfg.goal.deadline_max,
        )
        repertoire.save(run_dir / REPERTOIRE_FILE)  # seed the snapshot for workers
    ingested: set = set()
    if args.resume:
        ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
        if ckpts:
            trainer.load_checkpoint(ckpts[-1])
        if emergent_mode:
            # Reload or rebuild the frozen encoder and GoalSpace; then rebuild buffer.
            # CRITICAL: on resume, LOAD the persisted frozen encoder (the snapshot
            # whose embedding space the persisted GoalSpace centroids live in) —
            # do NOT re-snapshot the current net, or the centroids and the encoder
            # would be in different metric spaces and assign()/achieved() would be
            # silently wrong. Only snapshot fresh if no frozen encoder exists yet.
            gs_path = run_dir / GOALSPACE_DIR
            frozen_enc_path = run_dir / FROZEN_ENCODER
            if frozen_enc_path.exists():
                frozen_encoder = VectorGoalNetEvaluator.from_checkpoint(
                    frozen_enc_path, cfg.network, device=trainer.device
                )
            else:
                frozen_encoder = snapshot_frozen_encoder(net, run_dir, cfg.network, trainer.device)
            if gs_path.exists():
                goalspace = GoalSpace.load(gs_path, cfg.goal, frozen_encoder, rng)
            else:
                goalspace = GoalSpace(cfg.goal, frozen_encoder, rng)
            wv_path = run_dir / WINVALUE_FILE
            winvalue = WinValueEstimator.load(wv_path) if wv_path.exists() else WinValueEstimator()
            buffer = VectorGoalReplayBuffer.from_run_dir(
                run_dir, cfg.training.buffer_size, frozen_encoder, goalspace,
                deadline_max=cfg.goal.deadline_max,
            )
        elif goal_mode:
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
    hung_restarts = 0
    # Per-worker heartbeat state for the hung-worker watchdog (Task 5.2).
    heartbeat_seconds = getattr(cfg.selfplay, "worker_heartbeat_seconds", 0.0)
    last_seen: dict = {}
    last_ckpt_bucket = trainer.step // cfg.training.checkpoint_every_steps
    start = time.time()

    try:
        while games_seen < args.games:
            if emergent_mode:
                # Emergent branch: observe GoalSpace deltas per new record,
                # maybe_refresh (re-snapshot encoder + rebuild buffer), then train
                # with the vector dual-head trainer.
                added, positions = ingest_new_games(
                    run_dir, buffer, ingested,
                    on_record=lambda rec: (
                        observe_game_deltas(goalspace, rec, frozen_encoder, max_samples=8, rng=rng),
                        update_winvalue_from_record(winvalue, rec),
                    ),
                )
                games_seen += added
                total_positions += positions
                if added and winvalue is not None:
                    winvalue.save(run_dir / WINVALUE_FILE)
                # Only snapshot the frozen encoder when a refresh will actually fire
                # (epoch turned + reservoir ready): the snapshot is an expensive
                # state_dict copy + disk write, and snapshotting every cycle also opens
                # a crash-recovery hole where frozen_encoder.pt is ahead of the saved
                # centroids (adversarial review Bug A/E).
                if added and goalspace.should_refresh(baseline_games + games_seen):
                    new_enc = snapshot_frozen_encoder(net, run_dir, cfg.network, trainer.device)
                    goalspace.maybe_refresh(baseline_games + games_seen, embedder=new_enc)
                    frozen_encoder = new_enc
                    buffer = VectorGoalReplayBuffer.from_run_dir(
                        run_dir, cfg.training.buffer_size, frozen_encoder, goalspace,
                        deadline_max=cfg.goal.deadline_max,
                    )
                    goalspace.save(run_dir / GOALSPACE_DIR)
            else:
                added, positions = ingest_new_games(
                    run_dir, buffer, ingested, recent_records, repertoire=repertoire
                )
                games_seen += added
                total_positions += positions
                # Persist the updated repertoire snapshot so workers reload it on
                # their checkpoint cadence (plan Task 4.3). Atomic write.
                if repertoire is not None and added:
                    repertoire.save(run_dir / REPERTOIRE_FILE)

            steps_done = 0
            n = trainer.allowed_steps(total_positions)
            if n > 0 and len(buffer) >= cfg.training.batch_size:
                if emergent_mode:
                    m = trainer.train_steps_vector(buffer, n, rng)
                elif goal_mode:
                    m = trainer.train_steps_goal(buffer, n, rng)
                else:
                    m = trainer.train_steps(buffer, n, rng)
                steps_done = n
                bucket = trainer.step // cfg.training.checkpoint_every_steps
                if bucket > last_ckpt_bucket:
                    trainer.save_checkpoint()
                    last_ckpt_bucket = bucket
            else:
                m = {"policy_loss": None, "value_loss": None, "step": trainer.step}

            # restart any dead OR hung worker. A hung worker is alive but has
            # produced no new game within the heartbeat window; left running it
            # silently halves an arm's throughput (spec sec 14, Task 5.2).
            now = time.time()
            hung = set(hung_workers(
                run_dir, range(len(procs)), last_seen, now, heartbeat_seconds
            ))
            for i, p in enumerate(procs):
                dead = not p.is_alive()
                if not dead and i not in hung:
                    continue
                if not dead:
                    # Terminate the stuck process before respawning so we don't
                    # leak it; it may be wedged in a non-returning call.
                    p.terminate()
                    p.join(timeout=10)
                    hung_restarts += 1
                procs[i] = _spawn_worker(
                    ctx, i, run_dir, stop_path, cfg.training.selfplay_device
                )
                restarts += 1
                # Reset this worker's heartbeat clock so the fresh process gets a
                # full window before it can be judged hung again.
                last_seen[i] = (worker_last_progress(run_dir, i), now)

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
                "worker_hung_restarts": hung_restarts,
                "resign_fp_rate": fp_stats["resign_fp_rate"],
            }
            if goal_mode and recent_records:
                sp_rates = goal_achievement_rates(recent_records)
                sf_rates = load_stockfish_achievement_rates(run_dir)
                metrics["goal_achievement_rate"] = sp_rates
                metrics["wishful_thinking"] = wishful_thinking_thermometer(sp_rates, sf_rates)
                # Observability time series (spec sec 16, Task 5.3): fraction of
                # plies under g=win (control variable) and, when a repertoire is
                # present (lp mode), its size + per-kind learning-progress.
                wpf = win_ply_fraction(recent_records)
                if wpf is not None:
                    metrics["win_ply_fraction"] = wpf
                if repertoire is not None:
                    metrics["repertoire_size"] = len(repertoire.templates())
                    metrics["learning_progress"] = repertoire_learning_progress(repertoire)
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
        if emergent_mode:
            added, positions = ingest_new_games(
                run_dir, buffer, ingested,
                on_record=lambda rec: (
                    observe_game_deltas(goalspace, rec, frozen_encoder, max_samples=8, rng=rng),
                    update_winvalue_from_record(winvalue, rec),
                ),
            )
            if goalspace is not None:
                goalspace.save(run_dir / GOALSPACE_DIR)
            if winvalue is not None:
                winvalue.save(run_dir / WINVALUE_FILE)
        else:
            added, positions = ingest_new_games(
                run_dir, buffer, ingested, repertoire=repertoire
            )
            if repertoire is not None:
                repertoire.save(run_dir / REPERTOIRE_FILE)
        games_seen += added
        total_positions += positions
        trainer.save_checkpoint()
        (run_dir / "state.json").write_text(
            json.dumps({"games": baseline_games + games_seen, "positions": total_positions})
        )
        if stop_path.exists():
            stop_path.unlink()

    return run_dir
