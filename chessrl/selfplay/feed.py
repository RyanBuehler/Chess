"""Live-feed publisher for in-progress self-play (M7).

ZeroMQ PUB/SUB over tcp://127.0.0.1, ONE TOPIC PER GAME, bounded send-HWM with
DROP-ON-FULL. publish() never blocks a worker: it sends with zmq.DONTWAIT and
silently drops the frame when the queue is full (no subscriber, or a slow one).
A stalled/absent subscriber loses frames, never training time (spec: Process
model / Live feed).

`import zmq` lives ONLY in this module, so a training run with the feed disabled
(SelfPlayConfig.feed_port == 0) never imports zmq. NullPublisher is the no-op
twin used everywhere the feed is off.
"""
import json


class NullPublisher:
    """No-op publisher (default). Same surface as FeedPublisher; does nothing."""

    def publish(self, game_id: str, payload: dict) -> None:
        return

    def close(self) -> None:
        return


class FeedPublisher:
    """zmq PUB bound to tcp://127.0.0.1:{port}. Small SNDHWM bounds memory;
    LINGER=0 so close() never hangs on undelivered frames. Drop-on-full."""

    def __init__(self, port: int, sndhwm: int = 100):
        import zmq                                   # lazy: only when the feed is ON

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.SNDHWM, sndhwm)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://127.0.0.1:{port}")
        self.port = port
        self._zmq = zmq

    def publish(self, game_id: str, payload: dict) -> None:
        try:
            self._sock.send_multipart(
                [game_id.encode(), json.dumps(payload).encode()],
                flags=self._zmq.DONTWAIT,
            )
        except self._zmq.Again:
            pass                                      # queue full -> drop this frame
        except Exception:
            pass                                      # never let the feed crash a worker

    def close(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:
            pass
