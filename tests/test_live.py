"""Live backend: a real zmq PUB (the 'worker') -> the server SUB -> a /ws/live
browser receives updates. With no feed configured, the endpoint still opens and
emits an empty roster."""
import json
import time

from fastapi.testclient import TestClient

from server.app import create_app


def _free_port():
    import socket
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def test_live_no_feed_emits_empty_roster(tmp_path):
    runs_root = tmp_path / "runs"; runs_root.mkdir()
    app = create_app(runs_root)            # no feed_ports -> live disabled
    client = TestClient(app)
    with client.websocket_connect("/ws/live") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "roster"
        assert msg["games"] == []


def test_live_receives_published_game(tmp_path):
    import zmq

    runs_root = tmp_path / "runs"; runs_root.mkdir()
    port = _free_port()
    # Configure the app to subscribe to [port].
    app = create_app(runs_root)
    app.state.feed_ports = [port]

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(f"tcp://127.0.0.1:{port}")

    client = TestClient(app)
    try:
        with client.websocket_connect("/ws/live") as ws:
            first = ws.receive_json()
            assert first["type"] == "roster"
            time.sleep(0.3)                 # slow-joiner settle for the server SUB
            payload = {"game_id": "w00_b0_0", "fen": "startpos", "ply": 3,
                       "root_q": 0.1, "top_moves": [["e2e4", 0.5]], "done": False,
                       "z": None}
            for _ in range(10):
                pub.send_multipart([b"w00_b0_0", json.dumps(payload).encode()])
                time.sleep(0.03)
            # Expect an update (or roster+update) carrying our game_id.
            got = None
            for _ in range(20):
                msg = ws.receive_json()
                if msg["type"] == "update" and msg["game"]["game_id"] == "w00_b0_0":
                    got = msg
                    break
                if msg["type"] == "roster" and "w00_b0_0" in msg["games"]:
                    got = msg
                    break
            assert got is not None
    finally:
        pub.close()
