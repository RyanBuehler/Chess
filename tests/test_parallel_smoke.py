"""M5 gate: the full parallel pipeline runs end to end with a real spawned
worker. Marked slow (Windows spawn + CPU); run with `-m slow`."""
import json

import pytest

from chessrl.training.parallel_loop import main

SMOKE_YAML = """\
run_name: psmoke
network: {blocks: 1, filters: 8}
mcts: {simulations: 8, temperature_moves: 4, leaves_per_tree: 2}
selfplay: {ply_cap: 30, workers: 1, concurrent_games: 2, resign_playout_fraction: 0.0}
training: {batch_size: 16, buffer_size: 1000, samples_per_position: 2.0, checkpoint_every_steps: 1, device: cpu, selfplay_device: cpu}
"""


@pytest.mark.slow
def test_parallel_smoke(tmp_path):
    cfg = tmp_path / "psmoke.yaml"
    cfg.write_text(SMOKE_YAML)
    run_dir = main(
        ["--config", str(cfg), "--runs-root", str(tmp_path / "runs"), "--games", "2"]
    )

    npz = list((run_dir / "games").glob("*.npz"))
    pgn = list((run_dir / "games").glob("*.pgn"))
    ckpts = list((run_dir / "checkpoints").glob("ckpt_*.pt"))
    assert len(npz) >= 2
    assert len(pgn) >= 2
    assert len(ckpts) >= 1

    metrics_lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(metrics_lines) >= 1
    last = json.loads(metrics_lines[-1])
    assert "games_per_hour" in last
    assert "resign_fp_rate" in last
    assert "worker_restarts" in last

    # STOP sentinel cleaned up; state.json written.
    assert not (run_dir / "STOP").exists()
    state = json.loads((run_dir / "state.json").read_text())
    assert state["games"] >= 2

    # at least one worker meta file with valid json lines
    metas = list(run_dir.glob("games_meta_w*.jsonl"))
    assert metas
    first_meta = metas[0].read_text().splitlines()[0]
    assert "plies" in json.loads(first_meta)
