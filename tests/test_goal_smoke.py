"""End-to-end goal-pipeline smoke (plan Task 3.5). SLOW: spawns real workers on
CPU. Marked slow so it is deselected by default; run with `-m slow`.

Runs a tiny budget (a handful of games, low sims) per arm and asserts: runs
start, checkpoints write, no worker crashes (no restarts), and the always-win
arm runs end-to-end through the goal pipeline (goal records on disk with goal
columns + the wishful-thinking thermometer emitted)."""
import json

import pytest

from chessrl.selfplay.records import GameRecord
from chessrl.training.parallel_loop import main

# Tiny budget: 1 block / 8 filters, 6 sims, short games, CPU, 1 worker.
_COMMON = """\
network: {blocks: 1, filters: 8}
mcts: {simulations: 6, temperature_moves: 2, leaves_per_tree: 1}
selfplay: {ply_cap: 24, workers: 1, concurrent_games: 2, resign_playout_fraction: 0.0, feed_port: 0}
training: {batch_size: 8, buffer_size: 1000, samples_per_position: 2.0, checkpoint_every_steps: 1, device: cpu, selfplay_device: cpu}
"""


def _yaml(run_name: str, goal_mode: str) -> str:
    return f"run_name: {run_name}\n{_COMMON}goal: {{goal_mode: {goal_mode}, win_floor: 0.2}}\n"


def _run_arm(tmp_path, run_name, goal_mode, games):
    cfg = tmp_path / f"{run_name}.yaml"
    cfg.write_text(_yaml(run_name, goal_mode))
    run_dir = main(
        ["--config", str(cfg), "--runs-root", str(tmp_path / "runs"), "--games", str(games)]
    )
    npz = list((run_dir / "games").glob("*.npz"))
    ckpts = list((run_dir / "checkpoints").glob("ckpt_*.pt"))
    assert npz, f"{run_name}: no game records written"
    assert ckpts, f"{run_name}: no checkpoint written"

    metrics = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert metrics, f"{run_name}: no metrics emitted"
    # No worker crashed during the slice.
    assert metrics[-1]["worker_restarts"] == 0, f"{run_name}: a worker crashed"
    return run_dir, npz, metrics


@pytest.mark.slow
def test_vanilla_arm_smoke(tmp_path):
    run_dir, npz, _ = _run_arm(tmp_path, "gp-vanilla", "none", games=2)
    # Vanilla records carry NO goal columns.
    rec = GameRecord.load(npz[0])
    assert not rec.has_goals()


@pytest.mark.slow
def test_always_win_arm_runs_through_goal_pipeline(tmp_path):
    run_dir, npz, metrics = _run_arm(tmp_path, "gp-always-win", "always_win", games=2)
    # Goal records present, with goal columns -> the goal pipeline ran end-to-end.
    rec = GameRecord.load(npz[0])
    assert rec.has_goals()
    assert rec.win_ply_fraction() == 1.0          # always-win: every ply under WIN
    # The wishful-thinking thermometer was emitted for a goal run.
    therm = [m for m in metrics if m.get("wishful_thinking")]
    assert therm, "always-win arm did not emit the thermometer"


@pytest.mark.slow
def test_random_goal_arm_smoke(tmp_path):
    run_dir, npz, metrics = _run_arm(tmp_path, "gp-random-goal", "random", games=2)
    rec = GameRecord.load(npz[0])
    assert rec.has_goals()
    # Goal achievement-rate diagnostic emitted at least once.
    assert any(m.get("goal_achievement_rate") for m in metrics)
