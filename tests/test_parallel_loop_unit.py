import json

import numpy as np

from chessrl.config.config import RunConfig
from chessrl.supervised.pgn_import import record_from_pgn
from chessrl.training.buffer import ReplayBuffer
from chessrl.training.parallel_loop import (
    aggregate_resign_fp,
    ingest_new_games,
    make_run_dir,
)

FOOLS_MATE = '[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n'


def test_ingest_new_games_adds_only_unseen(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    rec = record_from_pgn(FOOLS_MATE)  # 4 positions
    rec.save(games / "game_w00_0000000.npz")
    buf = ReplayBuffer(1000)
    ingested = set()

    added, positions = ingest_new_games(tmp_path, buf, ingested)
    assert added == 1
    assert positions == 4
    assert len(buf) == 4

    # Second call: nothing new.
    added2, positions2 = ingest_new_games(tmp_path, buf, ingested)
    assert added2 == 0 and positions2 == 0

    # A new file is picked up.
    rec.save(games / "game_w00_0000001.npz")
    added3, positions3 = ingest_new_games(tmp_path, buf, ingested)
    assert added3 == 1 and positions3 == 4
    assert len(buf) == 8


def test_ingest_skips_partial_npz(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    (games / "game_w00_0000000.npz").write_bytes(b"not a real npz yet")
    buf = ReplayBuffer(1000)
    ingested = set()
    added, positions = ingest_new_games(tmp_path, buf, ingested)
    # A half-written file fails to load and is left for a later pass.
    assert added == 0 and positions == 0
    assert len(ingested) == 0


def test_aggregate_resign_fp(tmp_path):
    (tmp_path / "games_meta_w00.jsonl").write_text(
        json.dumps({"playout": True, "would_resign": True, "fp": True}) + "\n"
        + json.dumps({"playout": True, "would_resign": True, "fp": False}) + "\n"
        + json.dumps({"playout": False, "would_resign": False, "fp": False}) + "\n"
    )
    (tmp_path / "games_meta_w01.jsonl").write_text(
        json.dumps({"playout": True, "would_resign": False, "fp": False}) + "\n"
    )
    stats = aggregate_resign_fp(tmp_path)
    # 3 playout games total; 1 false positive -> rate 1/3.
    assert stats["playout_games"] == 3
    assert stats["false_positives"] == 1
    assert abs(stats["resign_fp_rate"] - 1.0 / 3.0) < 1e-9


def test_make_run_dir_writes_config_and_provenance(tmp_path):
    cfg = RunConfig.from_dict({"run_name": "pll"})
    run_dir = make_run_dir(cfg, runs_root=tmp_path / "runs")
    assert (run_dir / "config.json").exists()
    assert (run_dir / "provenance.json").exists()
    assert (run_dir / "games").is_dir()
    loaded = RunConfig.from_json(run_dir / "config.json")
    assert loaded.run_name == "pll"
