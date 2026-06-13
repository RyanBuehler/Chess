import json

import numpy as np

from chessrl.config.config import RunConfig
from chessrl.supervised.pgn_import import record_from_pgn
from chessrl.training.buffer import ReplayBuffer
from chessrl.training.parallel_loop import (
    aggregate_resign_fp,
    hung_workers,
    ingest_new_games,
    make_run_dir,
    worker_last_progress,
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


def _touch_worker_game(run_dir, worker_id, counter, mtime):
    """Write a fake per-worker game file with a controlled mtime (Task 5.2)."""
    games = run_dir / "games"
    games.mkdir(exist_ok=True)
    f = games / f"game_w{worker_id:02d}_{counter:07d}.npz"
    f.write_bytes(b"x")
    import os
    os.utime(f, (mtime, mtime))
    return f


def test_worker_last_progress_tracks_newest_game(tmp_path):
    assert worker_last_progress(tmp_path, 0) is None
    _touch_worker_game(tmp_path, 0, 0, mtime=100.0)
    _touch_worker_game(tmp_path, 0, 1, mtime=250.0)
    _touch_worker_game(tmp_path, 1, 0, mtime=300.0)
    assert worker_last_progress(tmp_path, 0) == 250.0   # newest of worker 0
    assert worker_last_progress(tmp_path, 1) == 300.0


def test_hung_worker_detected_after_window_no_real_sleep(tmp_path):
    (tmp_path / "games").mkdir()
    last_seen: dict = {}
    window = 60.0
    # t=1000: worker 0 produced a game at t=1000; first observation starts clock.
    _touch_worker_game(tmp_path, 0, 0, mtime=1000.0)
    assert hung_workers(tmp_path, [0], last_seen, now=1000.0, heartbeat_seconds=window) == []
    # t=1030: still inside window, no new game -> not yet hung.
    assert hung_workers(tmp_path, [0], last_seen, now=1030.0, heartbeat_seconds=window) == []
    # t=1061: > window since last progress (1000) and no new game -> hung.
    assert hung_workers(tmp_path, [0], last_seen, now=1061.0, heartbeat_seconds=window) == [0]


def test_progress_resets_the_heartbeat_clock(tmp_path):
    (tmp_path / "games").mkdir()
    last_seen: dict = {}
    window = 60.0
    _touch_worker_game(tmp_path, 0, 0, mtime=1000.0)
    hung_workers(tmp_path, [0], last_seen, now=1000.0, heartbeat_seconds=window)
    # A new game lands at t=1050 -> progress advanced -> clock resets.
    _touch_worker_game(tmp_path, 0, 1, mtime=1050.0)
    assert hung_workers(tmp_path, [0], last_seen, now=1050.0, heartbeat_seconds=window) == []
    # t=1100: only 50s since the 1050 game -> still healthy.
    assert hung_workers(tmp_path, [0], last_seen, now=1100.0, heartbeat_seconds=window) == []
    # t=1111: > window since 1050, no newer game -> now hung.
    assert hung_workers(tmp_path, [0], last_seen, now=1111.0, heartbeat_seconds=window) == [0]


def test_worker_with_no_output_is_caught_from_first_observation(tmp_path):
    (tmp_path / "games").mkdir()
    last_seen: dict = {}
    window = 60.0
    # Worker never produces anything; clock starts at first observation (t=500).
    assert hung_workers(tmp_path, [0], last_seen, now=500.0, heartbeat_seconds=window) == []
    assert hung_workers(tmp_path, [0], last_seen, now=561.0, heartbeat_seconds=window) == [0]


def test_heartbeat_disabled_never_flags(tmp_path):
    (tmp_path / "games").mkdir()
    last_seen: dict = {}
    # window <= 0 disables detection entirely.
    assert hung_workers(tmp_path, [0, 1], last_seen, now=1e9, heartbeat_seconds=0.0) == []


def test_make_run_dir_writes_config_and_provenance(tmp_path):
    cfg = RunConfig.from_dict({"run_name": "pll"})
    run_dir = make_run_dir(cfg, runs_root=tmp_path / "runs")
    assert (run_dir / "config.json").exists()
    assert (run_dir / "provenance.json").exists()
    assert (run_dir / "games").is_dir()
    loaded = RunConfig.from_json(run_dir / "config.json")
    assert loaded.run_name == "pll"
