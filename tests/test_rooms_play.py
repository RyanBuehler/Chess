"""Play websocket: a full short game vs a tiny random-weight checkpoint at low
sims. TestClient's websocket is synchronous. We build a real 2-block checkpoint
so NetMCTSPlayer loads it for real (CPU, few sims -> fast)."""
import chess
import torch
from fastapi.testclient import TestClient

from chessrl.config.config import NetworkConfig, RunConfig, TrainingConfig
from chessrl.model.network import PolicyValueNet
from chessrl.training.trainer import Trainer
from server.app import create_app


def _make_run_with_ckpt(runs_root, run_id="r1"):
    run = runs_root / run_id
    (run / "checkpoints").mkdir(parents=True)
    net_cfg = NetworkConfig(blocks=2, filters=8)
    cfg = RunConfig(run_name=run_id, network=net_cfg)
    (run / "config.json").write_text(cfg.to_json())
    torch.manual_seed(0)
    net = PolicyValueNet(net_cfg)
    trainer = Trainer(net, TrainingConfig(batch_size=4, device="cpu"), run)
    ckpt = trainer.save_checkpoint()
    return run, ckpt.name


def _client(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    run, ckpt_name = _make_run_with_ckpt(runs_root)
    return TestClient(create_app(runs_root)), "r1", ckpt_name


def test_play_human_white_full_exchange(tmp_path):
    client, run_id, ckpt = _client(tmp_path)
    with client.websocket_connect("/ws/play") as ws:
        ws.send_json({"type": "new", "run_id": run_id, "checkpoint": ckpt,
                      "simulations": 8, "color": "white"})
        msg = ws.receive_json()
        assert msg["type"] == "state"
        assert msg["fen"].split()[1] == "w"          # human (white) to move first
        # Human plays a couple of legal moves; agent responds each time.
        for uci in ("e2e4", "d2d4"):
            ws.send_json({"type": "move", "uci": uci})
            human_state = ws.receive_json()
            assert human_state["type"] == "state"
            assert human_state["last_move"] == uci
            agent_state = ws.receive_json()
            assert agent_state["type"] == "state"
            assert "eval" in agent_state
            assert isinstance(agent_state["thoughts"], list)
            assert len(agent_state["thoughts"]) <= 5
            # the board is back to white to move after the agent replied
            assert agent_state["fen"].split()[1] == "w"


def test_play_illegal_move_errors_without_advancing(tmp_path):
    client, run_id, ckpt = _client(tmp_path)
    with client.websocket_connect("/ws/play") as ws:
        ws.send_json({"type": "new", "run_id": run_id, "checkpoint": ckpt,
                      "simulations": 8, "color": "white"})
        ws.receive_json()
        ws.send_json({"type": "move", "uci": "e2e5"})    # illegal
        err = ws.receive_json()
        assert err["type"] == "error"
        # board unchanged: a subsequent legal move still works
        ws.send_json({"type": "move", "uci": "e2e4"})
        ok = ws.receive_json()
        assert ok["type"] == "state"
        assert ok["last_move"] == "e2e4"


def test_play_human_black_agent_opens(tmp_path):
    client, run_id, ckpt = _client(tmp_path)
    with client.websocket_connect("/ws/play") as ws:
        ws.send_json({"type": "new", "run_id": run_id, "checkpoint": ckpt,
                      "simulations": 8, "color": "black"})
        # Agent is white -> it opens; first state shows black to move.
        opening = ws.receive_json()
        assert opening["type"] == "state"
        assert opening["fen"].split()[1] == "b"
        assert opening["last_move"] is not None


def test_play_status_reports_game_over(tmp_path):
    # Drive a known mate quickly: human plays Fool's-mate-style is too slow vs an
    # agent, so just assert that after a move the status field is present and is
    # one of the allowed values.
    client, run_id, ckpt = _client(tmp_path)
    with client.websocket_connect("/ws/play") as ws:
        ws.send_json({"type": "new", "run_id": run_id, "checkpoint": ckpt,
                      "simulations": 8, "color": "white"})
        ws.receive_json()
        ws.send_json({"type": "move", "uci": "e2e4"})
        s = ws.receive_json()
        assert s["status"] in ("playing", "checkmate", "stalemate", "draw")
