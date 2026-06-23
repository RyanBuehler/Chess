"""Regression: resuming an emergent run must NOT clobber the persisted frozen
encoder + GoalSpace.

The emergent init block in parallel_loop.main re-snapshotted the (freshly
constructed, pre-checkpoint) encoder and saved an EMPTY GoalSpace on every start
— including resume — *before* the resume branch reloaded them. That silently
reset the goal space (reservoir -> 0, centroids -> empty) and overwrote
frozen_encoder.pt with a random-net snapshot, so `ready` went False and every
game fell to terminal "win" pursuit (goal-conditioning silently disabled).
"""
import json

import numpy as np
import pytest

from chessrl.training.parallel_loop import main

_TINY_CFG = """
run_name: resume-test
network: {blocks: 1, filters: 8, goal_cond: vector}
mcts: {simulations: 6, temperature_moves: 4, leaves_per_tree: 1, meansend_alpha: 0.25}
selfplay: {workers: 1, concurrent_games: 4, ply_cap: 16, feed_port: 0}
training:
  batch_size: 8
  buffer_size: 2000
  samples_per_position: 1.0
  checkpoint_every_steps: 5
  device: cpu
  selfplay_device: cpu
goal:
  goal_mode: emergent
  cluster_k: 3
  refresh_every: 1
  reservoir_size: 500
  min_reservoir: 20
  goal_window: 2
  deadline_max: 8
  delta_samples_per_game: 8
"""


@pytest.mark.slow
def test_resume_preserves_emergent_goalspace_and_encoder(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    cfg_yaml = tmp_path / "tiny.yaml"
    cfg_yaml.write_text(_TINY_CFG)

    # Fresh run: enough games to fill the reservoir and fit clusters.
    rd = main(["--config", str(cfg_yaml), "--runs-root", str(runs_root),
               "--run-dir-name", "r", "--games", "30"])
    cents_before = np.load(rd / "goalspace" / "centroids.npy")
    seen_before = json.loads((rd / "goalspace" / "meta.json").read_text())["seen"]
    enc_before = (rd / "frozen_encoder.pt").read_bytes()
    assert cents_before.shape[0] > 0, "fresh run should have fit clusters"
    assert seen_before >= 20, "fresh run should have filled the reservoir"

    # Resume with no new games: the persisted goalspace + encoder must survive.
    main(["--resume", "r", "--runs-root", str(runs_root), "--games", "0"])

    cents_after = np.load(rd / "goalspace" / "centroids.npy")
    seen_after = json.loads((rd / "goalspace" / "meta.json").read_text())["seen"]
    assert cents_after.shape[0] == cents_before.shape[0], \
        "resume clobbered the fit centroids (reset to empty)"
    assert seen_after >= seen_before, "resume reset the reservoir (seen dropped)"
    assert (rd / "frozen_encoder.pt").read_bytes() == enc_before, \
        "resume overwrote the persisted frozen encoder with a fresh snapshot"
