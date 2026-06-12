"""Live-feed publisher: drop-on-full never blocks; a real PUB/SUB round-trip
delivers a published payload. zmq's slow-joiner means we subscribe, sleep to let
the subscription propagate, THEN publish."""
import json
import time

import numpy as np
import pytest

from chessrl.config.config import MCTSConfig, SelfPlayConfig
from chessrl.selfplay.feed import FeedPublisher, NullPublisher


def _free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_null_publisher_is_silent_noop():
    pub = NullPublisher()
    # Never raises, returns nothing useful, has the same surface as FeedPublisher.
    pub.publish("g0", {"any": "payload"})
    pub.close()


def test_publisher_drop_on_full_never_blocks():
    # Tiny HWM, NO subscriber -> the queue fills and every further send is dropped
    # via zmq.Again. 1000 publishes must complete near-instantly (bounded time),
    # proving publish() never blocks a worker.
    port = _free_port()
    pub = FeedPublisher(port, sndhwm=8)
    try:
        start = time.perf_counter()
        for i in range(1000):
            pub.publish("g0", {"i": i})
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"publish blocked: {elapsed:.2f}s for 1000 msgs"
    finally:
        pub.close()


def test_publisher_subscriber_round_trip():
    import zmq

    port = _free_port()
    pub = FeedPublisher(port, sndhwm=100)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    try:
        sub.connect(f"tcp://127.0.0.1:{port}")
        sub.setsockopt(zmq.SUBSCRIBE, b"g42")
        time.sleep(0.3)                      # slow-joiner: let the subscription register
        sent = {"game_id": "g42", "fen": "startpos", "ply": 1}
        # Publish a few times; PUB/SUB may drop the first frame during join.
        for _ in range(5):
            pub.publish("g42", sent)
            time.sleep(0.02)
        sub.RCVTIMEO = 1000
        topic, body = sub.recv_multipart()
        assert topic == b"g42"
        assert json.loads(body.decode())["fen"] == "startpos"
    finally:
        sub.close()
        pub.close()


def test_subscriber_topic_filtering_isolates_games():
    import zmq

    port = _free_port()
    pub = FeedPublisher(port, sndhwm=100)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    try:
        sub.connect(f"tcp://127.0.0.1:{port}")
        sub.setsockopt(zmq.SUBSCRIBE, b"keep")     # only the "keep" topic
        time.sleep(0.3)
        for _ in range(5):
            pub.publish("drop", {"game_id": "drop"})
            pub.publish("keep", {"game_id": "keep"})
            time.sleep(0.02)
        sub.RCVTIMEO = 1000
        topic, body = sub.recv_multipart()
        assert topic == b"keep"                    # the "drop" topic is filtered out
    finally:
        sub.close()
        pub.close()


def test_concurrent_self_play_publishes_moves():
    """play_games_concurrent with a real publisher emits per-move payloads with
    the normative keys, and a terminal done=True frame, to per-game topics."""
    import zmq

    from chessrl.selfplay.concurrent import play_games_concurrent

    class _Stub:
        """Deterministic tiny evaluator: uniform policy, value 0."""
        def evaluate_planes(self, planes_batch):
            n = planes_batch.shape[0]
            pol = np.full((n, 4672), 1.0 / 4672, dtype=np.float32)
            val = np.zeros((n,), dtype=np.float32)
            return pol, val

        def evaluate_many(self, boards):
            n = len(boards)
            pol = np.full((n, 4672), 1.0 / 4672, dtype=np.float32)
            val = np.zeros((n,), dtype=np.float32)
            return pol, val

    port = _free_port()
    pub = FeedPublisher(port, sndhwm=1000)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{port}")
    sub.setsockopt(zmq.SUBSCRIBE, b"")             # all topics
    time.sleep(0.3)
    try:
        mcts_cfg = MCTSConfig(simulations=2)
        sp_cfg = SelfPlayConfig(ply_cap=6, concurrent_games=1, resign_playout_fraction=1.0)
        rng = np.random.default_rng(0)
        play_games_concurrent(
            _Stub(), mcts_cfg, sp_cfg, rng, num_games=1,
            publisher=pub, game_id_prefix="w00_",
        )
        time.sleep(0.2)
        # Drain whatever arrived; with a tiny ply cap we expect at least one frame.
        sub.RCVTIMEO = 1000
        seen = []
        while True:
            try:
                topic, body = sub.recv_multipart()
            except zmq.Again:
                break
            seen.append((topic.decode(), json.loads(body.decode())))
        assert seen, "no live-feed frames were published"
        for topic, payload in seen:
            assert topic.startswith("w00_")
            for key in ("game_id", "fen", "ply", "root_q", "top_moves", "done"):
                assert key in payload, f"missing {key} in {payload}"
            assert isinstance(payload["top_moves"], list)
            assert len(payload["top_moves"]) <= 5
        assert any(p["done"] for _, p in seen), "no terminal done=True frame"
    finally:
        sub.close()
        pub.close()


def test_selfplay_config_has_feed_port_default_zero():
    assert SelfPlayConfig().feed_port == 0          # disabled by default (no zmq import)
