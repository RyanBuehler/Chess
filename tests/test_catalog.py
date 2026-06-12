"""Read-only REST catalog over a fabricated tmp run dir. Uses FastAPI's
synchronous TestClient (httpx under the hood). No real training artifacts needed;
we hand-build the run-dir layout the server reads."""
import json

import chess
import chess.pgn
from fastapi.testclient import TestClient

from chessrl.config.config import RunConfig
from server.app import create_app


def _make_run(runs_root, run_id="r1"):
    run = runs_root / run_id
    (run / "checkpoints").mkdir(parents=True)
    (run / "games").mkdir(parents=True)
    cfg = RunConfig(run_name=run_id)
    (run / "config.json").write_text(cfg.to_json())
    (run / "state.json").write_text(json.dumps({"step": 1234, "games": 50}))
    (run / "metrics.jsonl").write_text(
        json.dumps({"step": 100, "loss": 2.0, "games_per_hour": 90.0}) + "\n"
        + json.dumps({"step": 200, "loss": 1.5, "games_per_hour": 95.0}) + "\n"
    )
    (run / "elo.jsonl").write_text(
        json.dumps({"ts": 1.0, "step": 100, "ckpt": "c", "elo": 500.0, "nu": 1.2}) + "\n"
        + json.dumps({"ts": 2.0, "step": 200, "ckpt": "c", "elo": 620.0, "nu": 1.3}) + "\n"
    )
    (run / "checkpoints" / "ckpt_00000100.pt").write_bytes(b"x")
    (run / "checkpoints" / "ckpt_00000200.pt").write_bytes(b"x")
    # a tiny real PGN
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push(chess.Move.from_uci(uci))
    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = "1/2-1/2"
    (run / "games" / "game_0000000.pgn").write_text(str(game))
    return run


def _client(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    _make_run(runs_root, "r1")
    _make_run(runs_root, "r2")
    return TestClient(create_app(runs_root))


def test_list_runs(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/runs")
    assert r.status_code == 200
    runs = r.json()
    ids = {x["run_id"] for x in runs}
    assert ids == {"r1", "r2"}
    one = next(x for x in runs if x["run_id"] == "r1")
    assert one["state"]["step"] == 1234
    assert one["config"]["run_name"] == "r1"


def test_metrics_parsed(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/runs/r1/metrics")
    assert r.status_code == 200
    rows = r.json()
    assert [row["step"] for row in rows] == [100, 200]
    assert rows[1]["loss"] == 1.5


def test_elo_curve(tmp_path):
    c = _client(tmp_path)
    rows = c.get("/api/runs/r1/elo").json()
    assert [row["elo"] for row in rows] == [500.0, 620.0]


def test_checkpoints_listed(tmp_path):
    c = _client(tmp_path)
    cks = c.get("/api/runs/r1/checkpoints").json()
    steps = [x["step"] for x in cks]
    assert steps == [100, 200]
    assert cks[0]["name"] == "ckpt_00000100.pt"


def test_games_list_and_pgn_and_moves(tmp_path):
    c = _client(tmp_path)
    games = c.get("/api/runs/r1/games").json()
    assert "game_0000000.pgn" in [g["name"] for g in games]
    pgn = c.get("/api/runs/r1/games/game_0000000.pgn/pgn").text
    assert "1. e4 e5 2. Nf3" in pgn
    moves = c.get("/api/runs/r1/games/game_0000000.pgn/moves").json()
    assert moves["moves"] == ["e2e4", "e7e5", "g1f3"]
    assert moves["result"] == "1/2-1/2"


def test_unknown_run_is_404(tmp_path):
    c = _client(tmp_path)
    assert c.get("/api/runs/nope/metrics").status_code == 404


def test_path_traversal_rejected(tmp_path):
    c = _client(tmp_path)
    # A run id / game name escaping runs/ must never resolve to a real file.
    assert c.get("/api/runs/..%2f..%2fetc/metrics").status_code in (400, 404)
    assert c.get("/api/runs/r1/games/..%2f..%2fconfig.json/pgn").status_code in (400, 404)


def test_missing_optional_files_are_empty_not_500(tmp_path):
    # A run with no metrics/elo yet returns [] rather than erroring.
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    bare = runs_root / "bare"
    (bare / "checkpoints").mkdir(parents=True)
    bare.joinpath("config.json").write_text(RunConfig(run_name="bare").to_json())
    c = TestClient(create_app(runs_root))
    assert c.get("/api/runs/bare/metrics").json() == []
    assert c.get("/api/runs/bare/elo").json() == []
    assert c.get("/api/runs/bare/games").json() == []
