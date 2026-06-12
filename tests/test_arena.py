"""Arena websocket: random-vs-greedy at delay 0 completes and drops a valid
inbox JSON. We avoid checkpoint/stockfish players here (builtin floors are fast
and dependency-free); the agent path is covered by the Play tests."""
import json

from fastapi.testclient import TestClient

from server.app import create_app


def _client(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    return TestClient(create_app(runs_root)), runs_root


def test_arena_random_vs_greedy_completes_and_writes_inbox(tmp_path):
    client, runs_root = _client(tmp_path)
    with client.websocket_connect("/ws/arena") as ws:
        ws.send_json({
            "type": "start",
            "white": {"kind": "random"},
            "black": {"kind": "greedy"},
            "delay_ms": 0,
            "opening_idx": 0,
            "max_plies": 40,
        })
        last = None
        while True:
            msg = ws.receive_json()
            if msg["type"] == "state":
                assert "fen" in msg and "ply" in msg
                last = msg
            elif msg["type"] == "gameover":
                assert msg["z"] in (-1, 0, 1)
                break
        assert last is not None
    # inbox file written with the ingest_inbox schema
    inbox = runs_root / "ladder_inbox"
    files = list(inbox.glob("*.json"))
    assert len(files) == 1
    d = json.loads(files[0].read_text())
    assert set(d) >= {"white", "black", "z", "opening", "conditions"}
    assert d["white"] == "random"
    assert d["black"] == "greedy"
    assert d["z"] in (-1, 0, 1)
    assert d["opening"] == 0


def test_arena_inbox_is_ingestible_by_store(tmp_path):
    # The dropped file must be consumable by M6's LadderStore.ingest_inbox.
    from chessrl.evaluation.store import LadderStore

    client, runs_root = _client(tmp_path)
    with client.websocket_connect("/ws/arena") as ws:
        ws.send_json({"type": "start", "white": {"kind": "random"},
                      "black": {"kind": "random"}, "delay_ms": 0,
                      "opening_idx": 3, "max_plies": 30})
        while ws.receive_json()["type"] != "gameover":
            pass
    store = LadderStore(runs_root / "ladder.sqlite")
    n = store.ingest_inbox(runs_root / "ladder_inbox")
    assert n == 1
    assert len(store.all_results()) == 1


def test_arena_pause_step_resume(tmp_path):
    client, runs_root = _client(tmp_path)
    with client.websocket_connect("/ws/arena") as ws:
        ws.send_json({"type": "start", "white": {"kind": "random"},
                      "black": {"kind": "random"}, "delay_ms": 100000,  # huge -> effectively paused between moves
                      "opening_idx": 0, "max_plies": 20})
        first = ws.receive_json()
        assert first["type"] == "state"
        # With a huge delay, stepping advances exactly one move promptly.
        ws.send_json({"type": "step"})
        stepped = ws.receive_json()
        assert stepped["type"] == "state"
        assert stepped["ply"] == first["ply"] + 1
        ws.send_json({"type": "stop"})
        # stop ends the game; a gameover (or final state then gameover) arrives.
        msg = ws.receive_json()
        assert msg["type"] in ("state", "gameover")


def test_arena_stockfish_spec_rejected_when_unconfigured(tmp_path):
    client, runs_root = _client(tmp_path)   # no stockfish path in cfg
    with client.websocket_connect("/ws/arena") as ws:
        ws.send_json({"type": "start", "white": {"kind": "stockfish", "elo": 1320},
                      "black": {"kind": "random"}, "delay_ms": 0, "opening_idx": 0})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "stockfish" in msg["message"].lower()
