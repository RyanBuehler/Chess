"""M4 gate: the whole pipeline runs end to end in a tiny configuration.
Spec: single-process path only, well under 2 minutes."""
import json

from chessrl.training.loop import main

SMOKE_YAML = """\
run_name: smoke
network: {blocks: 2, filters: 16}
mcts: {simulations: 8, temperature_moves: 4}
selfplay: {ply_cap: 30, games_per_iteration: 2}
training: {batch_size: 16, buffer_size: 1000, samples_per_position: 1.0, device: cpu}
"""


def _launch(tmp_path, extra):
    cfg = tmp_path / "smoke.yaml"
    cfg.write_text(SMOKE_YAML)
    return main(["--config", str(cfg), "--iterations", "1",
                 "--runs-root", str(tmp_path / "runs"), *extra])


def test_smoke_pipeline(tmp_path):
    run_dir = _launch(tmp_path, [])
    assert (run_dir / "config.json").exists()
    prov = json.loads((run_dir / "provenance.json").read_text())
    assert prov["torch_version"]  # git_commit may be None outside a repo
    assert len(list((run_dir / "games").glob("*.npz"))) == 2
    assert len(list((run_dir / "games").glob("*.pgn"))) == 2
    assert len(list((run_dir / "checkpoints").glob("ckpt_*.pt"))) == 1
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 1
    m = json.loads(lines[0])
    assert m["games"] == 2 and m["positions"] > 0


def test_smoke_resume(tmp_path):
    run_dir = _launch(tmp_path, [])
    out = main(["--resume", run_dir.name, "--iterations", "1",
                "--runs-root", str(run_dir.parent)])
    assert out == run_dir
    assert len(list((run_dir / "games").glob("*.npz"))) == 4
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 2
    state = json.loads((run_dir / "state.json").read_text())
    assert state["games"] == 4
