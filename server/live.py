"""Live-training backend: a single zmq SUB connected to the configured feed
ports, fanned out to every /ws/live browser. The SUB recv runs in a background
thread feeding an asyncio.Queue, so the event loop never blocks on recv. The
server keeps the latest payload per game_id, caps the active roster, and drops a
finished game after a short grace period. No feed configured -> empty roster.

Sampling/capping (spec: live view shows a sampled subset, default 12 boards):
the browser picks which <=12 to render; the server keeps up to MAX_ACTIVE recent
games and streams every update + a roster message on changes.
"""
import asyncio
import json
import threading
import time

MAX_ACTIVE = 24           # keep this many most-recent active games
DROP_FINISHED_AFTER = 30  # seconds to keep a finished game before dropping


class LiveHub:
    """Owns the SUB socket + the active-game table. One per app."""

    def __init__(self, feed_ports):
        self.feed_ports = list(feed_ports or [])
        self.games: dict[str, dict] = {}          # game_id -> latest payload
        self._finished_at: dict[str, float] = {}
        self._subscribers: set[asyncio.Queue] = set()
        self._loop = None
        self._thread = None
        self._stop = threading.Event()

    @property
    def enabled(self) -> bool:
        return bool(self.feed_ports)

    def start(self, loop):
        if not self.enabled or self._thread is not None:
            return
        self._loop = loop
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import zmq

        ctx = zmq.Context.instance()
        sub = ctx.socket(zmq.SUB)
        for port in self.feed_ports:
            sub.connect(f"tcp://127.0.0.1:{port}")
        sub.setsockopt(zmq.SUBSCRIBE, b"")        # all game topics
        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=200))
            if sub in socks:
                try:
                    _topic, body = sub.recv_multipart(flags=zmq.NOBLOCK)
                except Exception:
                    continue
                try:
                    payload = json.loads(body.decode())
                except json.JSONDecodeError:
                    continue
                self._ingest(payload)
        sub.close(linger=0)

    def _ingest(self, payload: dict):
        gid = payload.get("game_id")
        if gid is None:
            return
        roster_changed = gid not in self.games
        self.games[gid] = payload
        if payload.get("done"):
            self._finished_at[gid] = time.time()
        self._evict()
        # hand off to the event loop thread-safely
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._broadcast, payload, roster_changed)

    def _evict(self):
        now = time.time()
        for gid, t in list(self._finished_at.items()):
            if now - t > DROP_FINISHED_AFTER:
                self.games.pop(gid, None)
                self._finished_at.pop(gid, None)
        if len(self.games) > MAX_ACTIVE:
            # drop oldest by insertion order (dict preserves it); keep recent
            for gid in list(self.games)[:-MAX_ACTIVE]:
                self.games.pop(gid, None)
                self._finished_at.pop(gid, None)

    def _broadcast(self, payload: dict, roster_changed: bool):
        for q in list(self._subscribers):
            q.put_nowait({"type": "update", "game": payload})
            if roster_changed:
                q.put_nowait({"type": "roster", "games": list(self.games)})

    def add_subscriber(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def remove_subscriber(self, q):
        self._subscribers.discard(q)

    def roster(self) -> list:
        return list(self.games)

    def stop(self):
        self._stop.set()


def register_live_ws(app):
    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/ws/live")
    async def live_ws(ws: WebSocket):
        await ws.accept()
        # Lazily create the hub from app.state.feed_ports (set by serve.py/tests).
        hub = getattr(app.state, "_live_hub", None)
        if hub is None:
            hub = LiveHub(getattr(app.state, "feed_ports", []))
            app.state._live_hub = hub
            hub.start(asyncio.get_running_loop())

        await ws.send_json({"type": "roster", "games": hub.roster()})
        if not hub.enabled:
            # No feed: keep the socket open but idle (UI shows a hint).
            try:
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                return

        q = hub.add_subscriber()
        try:
            while True:
                msg = await q.get()
                await ws.send_json(msg)
        except WebSocketDisconnect:
            return
        finally:
            hub.remove_subscriber(q)
