# Chess RL Web Server + Four UI Views + Live Feed (M7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the training system watchable, playable, and pittable from a browser. Build the FastAPI server (REST catalog over read-only run dirs + websockets for live content), the ZeroMQ live-feed publisher wired into concurrent self-play (bounded HWM, drop-on-full, never blocks a worker), the GameRoom abstraction backing human-vs-agent play and arena matches, the arena's `ladder_inbox/` result submission, and the four static UI views (Dashboard, Live training, Play, Arena) with no build step. This delivers spec Success Criterion #3 ("the user can at any time: watch live self-play, play any checkpoint, or pit any two players against each other with a settable move delay").

**Architecture:** Extends the M1–M6 core (`docs/superpowers/specs/2026-06-11-chess-rl-design.md`, "Web Server & UI" + "Process model"). The server is a **separate process** that mirrors AlphaZero's actor/learner/evaluator separation: it **reads run dirs READ-ONLY** (it can never corrupt training state) and submits arena results as JSON to `runs/ladder_inbox/` for the M6 evaluator to ingest — it **never writes `ladder.sqlite` directly**. Server-side inference runs on **CPU by default** (a small net at interactive sim counts is fine on CPU; GPU is opt-in) so playing the agent doesn't contend with training on the single laptop GPU. The **live feed** is ZeroMQ PUB/SUB over `tcp://127.0.0.1`, one **topic per game**, bounded send-HWM with **drop-on-full** — publishing in a worker uses `zmq.DONTWAIT` and silently drops on a full queue, so a stalled or absent subscriber loses frames, not training time. Each worker binds its own PUB port (`feed_port + worker_id`); the server's SUB socket connects to the whole port range (PUB/SUB lets one SUB connect to many PUBs). `pyzmq` is imported **lazily** inside the feed module and the worker only constructs a real publisher when `cfg.selfplay.feed_port > 0`, so training without the server stack never imports zmq and never breaks. The core reuses M6 building blocks: `NetMCTSPlayer` for Play/Arena agent moves, `chess.pgn` for replay, the `store.ingest_inbox` JSON schema (`{white, black, z, opening, conditions}`) for arena submission, and the existing run-dir layout (`config.json`, `state.json`, `metrics.jsonl`, `elo.jsonl`, `games/`, `checkpoints/`, `eval_games/`).

**Tech Stack:** Python 3.11+, FastAPI + uvicorn[standard] (REST + websockets, now CORE deps), pyzmq (live feed, CORE dep but imported lazily so the worker path is import-clean when the feed is off), python-chess (1.11.x, board state, legality, PGN, `chess.pgn.Game`), PyTorch (CPU by default for server inference), NumPy, httpx (dev/test extra; FastAPI's `TestClient` uses it). Frontend: static HTML/CSS/vanilla-JS served by FastAPI with **no build toolchain** — chessground (Lichess board lib) via jsDelivr CDN, uPlot via CDN for charts. All websocket and board-catalog tests use FastAPI's synchronous `TestClient` (fine on Windows); zmq round-trip tests use a real PUB/SUB pair on localhost with the slow-joiner settle sleep.

**Scope:** Milestone M7 of the spec: the live-feed publisher + worker wiring (Task 1); the FastAPI app factory + read-only REST catalog (Task 2); the GameRoom + human-vs-agent Play websocket (Task 3); the Arena websocket with delay/pause/step/stop + `ladder_inbox` submission (Task 4); the Live-training SUB fan-out backend (Task 5); the static UI — Dashboard + game browser (Task 6), Live + Play + Arena views (Task 7); and `scripts/serve.py` + the integration gate against the real `runs/` (Task 8).

**Out of scope for M7 (stated so workers don't gold-plate):**
- The evaluator daemon, rating fit, Stockfish provisioning, and `ladder.sqlite` **writes** — all M6, done. M7 only *reads* `elo.jsonl`/PGNs the evaluator produced and *submits* to `ladder_inbox/` (which M6's `ingest_inbox` consumes).
- Auth / public deployment — the server is **LAN-only, trusted network** (spec "Out of Scope"). No login, no HTTPS, bind `127.0.0.1` by default.
- TensorBoard mirroring, W&B, checkpoint retention/pruning policy (config-only, deferred per spec).
- Automated browser/UI tests (Selenium/Playwright). The plan's gate is **endpoint-level** (REST + websocket via `TestClient`) plus a **manual verification checklist** in Task 8 — the four HTML views are exercised by hand against the live server. (Stated explicitly so a worker doesn't try to stand up a headless-browser harness.)
- GPU server inference by default (opt-in `--device cuda` flag exists but defaults to CPU per the spec's "server-side inference runs on CPU by default").
- The full live-firehose: the Live view subscribes per-game-topic and is **sampled + capped at 12 boards** (spec) — never the full 256-game concurrency.

**Conventions used throughout (normative, from the spec and M1–M6 code — do not drift):**
- **Read-only run dirs.** No server handler writes anything under `runs/<run-id>/`. The *only* path the server writes is `runs/ladder_inbox/*.json` (arena results, Task 4). A `runs/` arg is a directory; the server resolves run ids as its immediate subdirectories containing `config.json`. Path traversal is rejected (a `run_id`/`name` containing `/`, `\`, or `..` → 400/404), so a malicious URL can never escape `runs/`.
- **Result sign `z` (White's perspective):** `+1` White win, `0` draw, `-1` Black win — identical to M6. Arena `ladder_inbox` JSON uses exactly the `store.ingest_inbox` schema `{white, black, z, opening, conditions}`.
- **Live-feed payload (normative):** each published move is `{game_id, fen, last_move_uci, ply, root_q, top_moves: [[uci, visit_frac], ...up to 5], done, z}`. `root_q` is from the side-to-move perspective (the search's `root_q`); `top_moves` are the root's children by visit count, normalized to fractions of total visits, decoded to UCI. `done` true marks the final frame (with `z`).
- **Lazy zmq.** `import zmq` lives only inside `chessrl/selfplay/feed.py` (and the server's live module). `NullPublisher` is the default everywhere so `play_games_concurrent` and the worker run with zero zmq involvement when `feed_port == 0`. Publishing is **always non-blocking** (`zmq.DONTWAIT`); a full queue drops the frame.
- **CPU-default server inference.** Play/Arena agent players are built with `device="cpu"` unless an explicit override is passed; the server never assumes CUDA.
- **No eval-time noise/temperature in Play/Arena agent moves.** Reuse `NetMCTSPlayer` (which already searches `add_noise=False`, argmax visits) — the agent in the UI plays the same deterministic search M6 rates. (Live self-play *does* use noise/temperature; that is the trainer's existing behavior, untouched.)
- **Determinism / portability.** `pathlib.Path` only. Run all commands from `C:\Chess` with the venv: `.venv\Scripts\python -m pytest ...`. Tests force CPU. No OS-specific IPC beyond `tcp://127.0.0.1` (works identically on Windows and the Spark — CPython has no `AF_UNIX` on Windows, which is exactly why the spec chose TCP).
- **Async hygiene.** Agent move computation (synchronous, CPU, ~<2 s at 100–200 sims) runs in a thread executor (`asyncio.to_thread` / `run_in_executor`) so the event loop and other websockets are never blocked. The zmq SUB poll loop runs in a background thread feeding an `asyncio.Queue`, never blocking the loop on `recv`.

**Definition of done for M7:** the new test files pass on CPU (`test_feed.py`, `test_catalog.py`, `test_rooms_play.py`, `test_arena.py`, `test_live.py`), all existing M1–M6 tests still pass unchanged (the spec's 123-test suite stays green — `feed_port` defaults to 0 so nothing existing imports zmq or changes behavior), `scripts/serve.py` starts and serves the four views against the real `runs/` even with the feed and Stockfish absent, the REST endpoints return valid JSON for the real dev-tiny runs, a live `TestClient` play game and arena game complete end-to-end, and the manual checklist (Task 8) is walked. The static UI has no build step and loads its libraries from CDNs.

---

### Task 1: Live-feed publisher + concurrent self-play wiring

**Files:**
- Create: `chessrl/selfplay/feed.py`
- Modify: `chessrl/config/config.py` (one additive `feed_port` field on `SelfPlayConfig`)
- Modify: `chessrl/selfplay/concurrent.py` (publish after every applied move; additive params)
- Modify: `chessrl/selfplay/worker.py` (construct a real publisher iff `feed_port > 0`)
- Test: `tests/test_feed.py`

`FeedPublisher(port)` binds a zmq PUB to `tcp://127.0.0.1:{port}` with a small `SNDHWM` and `LINGER=0`, and `publish(game_id, payload)` sends a 2-frame multipart message `[game_id_bytes, json_bytes]` with `zmq.DONTWAIT`, dropping on `zmq.Again` (queue full / no subscriber). `NullPublisher` is the no-op twin (default). `SelfPlayConfig` gains `feed_port: int = 0` (0 = disabled). `play_games_concurrent` gains `publisher=None` and `game_id_prefix=""` and publishes the normative payload after each applied move (and a terminal `done` frame). The worker binds `feed_port + worker_id` so each of N workers owns a distinct port; the server later connects its SUB to the whole range. **`import zmq` happens only inside `feed.py`**, so a training run with `feed_port == 0` never touches zmq.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feed.py
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
        def evaluate_many(self, boards):
            import chess
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_feed.py -v`
Expected: FAIL — `ImportError` on `chessrl.selfplay.feed`, and `AttributeError` on `SelfPlayConfig().feed_port`.

- [ ] **Step 3: Implement the publisher (`chessrl/selfplay/feed.py`)**

```python
# chessrl/selfplay/feed.py
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
```

- [ ] **Step 4: Add the config field (`chessrl/config/config.py`)**

Add `feed_port` to `SelfPlayConfig` (after `concurrent_games`):

```python
    concurrent_games: int = 32           # M5: concurrent game trees per worker batch
    feed_port: int = 0                   # M7: live-feed base PUB port (0 = disabled, no zmq). Worker w binds feed_port + w.
```

`from_dict`/`asdict`/round-trip pick this up automatically (the existing `build(SelfPlayConfig, "selfplay")` path). No other config change.

- [ ] **Step 5: Wire the publisher into concurrent self-play (`chessrl/selfplay/concurrent.py`)**

Add a lazy default import and two parameters; publish the normative payload after each applied move and on terminal. Edit the signature and the move loop:

```python
# at top of file, add:
from chessrl.chess_env.moves import index_to_move
from chessrl.selfplay.feed import NullPublisher
```

Change `play_games_concurrent`'s signature and per-game id assignment:

```python
def play_games_concurrent(
    evaluator_many,
    mcts_cfg: MCTSConfig,
    sp_cfg: SelfPlayConfig,
    rng: np.random.Generator,
    num_games: int,
    publisher=None,
    game_id_prefix: str = "",
) -> list:
    """Returns list[(GameRecord, final_board, z, meta)] of length num_games,
    in slot order. z is from White's perspective (+1/0/-1). If `publisher` is
    given, every applied move is published to the live feed under the per-game
    topic f"{game_id_prefix}{slot}"; a terminal done=True frame is published when
    a game ends."""
    publisher = publisher or NullPublisher()
    mcts = BatchedMCTS(evaluator_many, mcts_cfg, rng)

    games: list[_Game] = []
    for slot in range(num_games):
        board = chess.Board()
        allow_resign = rng.random() >= sp_cfg.resign_playout_fraction
        tree = mcts.init_tree(board, add_noise=True)
        g = _Game(tree, board, allow_resign)
        g.game_id = f"{game_id_prefix}{slot}"          # NEW: stable per-game topic
        games.append(g)
```

Add `"game_id"` to `_Game.__slots__` (so the assignment above is legal):

```python
    __slots__ = (
        "tree", "builder", "board", "allow_resign", "resign_streak",
        "ply", "done", "z", "resigned", "would_resign_side", "game_id",
    )
```

In `_play_one_move`, after `mcts.advance(...)` / `g.board = g.tree.board` / `g.ply += 1` / termination check, publish the frame. Pass `publisher` through. Change `_play_one_move`'s signature to accept `publisher` and add the publish at the end:

```python
        for g in active:
            _play_one_move(g, mcts, mcts_cfg, sp_cfg, rng, publisher)
```

```python
def _play_one_move(
    g: _Game, mcts: BatchedMCTS, mcts_cfg: MCTSConfig, sp_cfg: SelfPlayConfig,
    rng: np.random.Generator, publisher,
) -> None:
    visits = mcts.visit_counts(g.tree)
    root_q = mcts.root_q(g.tree)
    idxs = np.fromiter(visits.keys(), dtype=np.int64)
    counts = np.fromiter(visits.values(), dtype=np.float64)
    if g.ply < mcts_cfg.temperature_moves:
        choice = int(rng.choice(idxs, p=counts / counts.sum()))
    else:
        choice = int(idxs[counts.argmax()])

    g.builder.add(g.board, idxs.astype(np.int32), counts.astype(np.int32), choice)

    # Decode the chosen move BEFORE committing (need the pre-move board context),
    # and build top-5 (uci, visit_frac) from the root visit distribution.
    flip = g.board.turn == chess.BLACK
    chosen_move = index_to_move(choice, flip, g.board)
    total = float(counts.sum())
    order = np.argsort(counts)[::-1][:5]
    top_moves = [
        [index_to_move(int(idxs[k]), flip, g.board).uci(), float(counts[k] / total)]
        for k in order
    ]

    if root_q < sp_cfg.resign_threshold:
        g.resign_streak[g.board.turn] += 1
        if g.resign_streak[g.board.turn] >= sp_cfg.resign_consecutive:
            if g.would_resign_side is None:
                g.would_resign_side = g.board.turn
            if g.allow_resign:
                g.z = -1 if g.board.turn == chess.WHITE else 1
                g.resigned = True
                g.done = True
                _publish_move(publisher, g, chosen_move, root_q, top_moves)
                return
    else:
        g.resign_streak[g.board.turn] = 0

    mcts.advance(g.tree, choice)
    g.board = g.tree.board
    g.ply += 1
    mcts.add_root_noise(g.tree)
    _check_pre_move_termination(g, sp_cfg)
    _publish_move(publisher, g, chosen_move, root_q, top_moves)


def _publish_move(publisher, g: _Game, chosen_move, root_q: float, top_moves: list) -> None:
    publisher.publish(g.game_id, {
        "game_id": g.game_id,
        "fen": g.board.fen(),
        "last_move_uci": chosen_move.uci(),
        "ply": g.ply,
        "root_q": float(root_q),
        "top_moves": top_moves,
        "done": bool(g.done),
        "z": int(g.z) if g.done else None,
    })
```

(The publish call is added in both the resign-return branch and the normal path so a resigned game still emits its terminal `done=True` frame. `NullPublisher` makes this free when the feed is off.)

- [ ] **Step 6: Wire a real publisher into the worker (`chessrl/selfplay/worker.py`)**

The worker constructs a `FeedPublisher` on `feed_port + worker_id` iff `feed_port > 0`, threads a `game_id_prefix` of `f"w{worker_id:02d}_b{batch}_"` so topics are unique across batches, and closes the publisher on shutdown. Add to `run_one_batch` and `worker_main`:

```python
# in run_one_batch signature, add publisher + batch_index, and pass them through:
def run_one_batch(
    run_dir, worker_id: int, evaluator: BatchedNetEvaluator, cfg: RunConfig,
    rng: np.random.Generator, start_counter: int, publisher=None, batch_index: int = 0,
) -> int:
    results = play_games_concurrent(
        evaluator, cfg.mcts, cfg.selfplay, rng,
        num_games=cfg.selfplay.concurrent_games,
        publisher=publisher,
        game_id_prefix=f"w{worker_id:02d}_b{batch_index}_",
    )
    ...
```

In `worker_main`, after building the evaluator, construct the publisher and a batch counter, and close in a `finally`:

```python
    from chessrl.selfplay.feed import FeedPublisher, NullPublisher

    publisher = NullPublisher()
    if cfg.selfplay.feed_port > 0:
        try:
            publisher = FeedPublisher(cfg.selfplay.feed_port + worker_id)
        except Exception:
            publisher = NullPublisher()             # feed is best-effort, never fatal

    batch_index = 0
    try:
        while not stop_path.exists():
            ... (checkpoint reload unchanged) ...
            counter = run_one_batch(
                run_dir, worker_id, evaluator, cfg, rng, counter,
                publisher=publisher, batch_index=batch_index,
            )
            batch_index += 1
            time.sleep(0.01)
    finally:
        publisher.close()
```

- [ ] **Step 7: Run the feed tests + the existing self-play tests (no regressions)**

Run: `.venv\Scripts\python -m pytest tests/test_feed.py tests/test_concurrent.py tests/test_worker.py -v`
Expected: feed tests pass (the two zmq round-trip tests rely on the 0.3 s slow-joiner sleep — do not remove it); existing concurrent/worker tests still pass unchanged (publisher defaults to `NullPublisher`, `feed_port` defaults to 0). If `test_publisher_subscriber_round_trip` is flaky, the settle sleep is too short or the publish loop sends only once — keep the 5×publish + 0.3 s settle.

> If `tests/test_concurrent.py` / `tests/test_worker.py` have different names, run `.venv\Scripts\python -m pytest tests/ -k "concurrent or worker" -v` to find and run the existing self-play suite.

- [ ] **Step 8: Commit**

```powershell
git add chessrl/selfplay/feed.py chessrl/config/config.py chessrl/selfplay/concurrent.py chessrl/selfplay/worker.py tests/test_feed.py
git commit -m "feat: M7 live-feed publisher (zmq PUB, drop-on-full, lazy import) + concurrent self-play + worker wiring"
```

---

### Task 2: Server scaffolding + read-only REST catalog

**Files:**
- Create: `server/__init__.py` (empty package marker)
- Create: `server/catalog.py` (run-dir reading, path-safe)
- Create: `server/app.py` (FastAPI app factory + REST routes + static mount)
- Modify: `pyproject.toml` (add fastapi, uvicorn[standard], pyzmq to core deps; httpx to dev)
- Create: `web/.gitkeep` (so the static mount dir exists for tests; replaced by real files in Tasks 6–7)
- Test: `tests/test_catalog.py`

`create_app(runs_root: Path, cfg=None, device="cpu")` returns a FastAPI app. REST: `GET /api/runs`, `GET /api/runs/{run_id}/metrics`, `/elo`, `/checkpoints`, `/games`, `/games/{name}/pgn`, `/games/{name}/moves`. All read-only; all path-validated. Static `web/` mounted at `/`. The catalog functions live in `server/catalog.py` (pure, testable without HTTP); `app.py` is thin routing over them.

- [ ] **Step 1: Add server deps to `pyproject.toml`**

```toml
dependencies = [
    "chess>=1.10",
    "numpy>=1.26",
    "PyYAML>=6.0",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "pyzmq>=25.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27"]
```

Install into the venv:

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
```

Expected: fastapi, starlette, uvicorn, pyzmq, httpx resolve and install. Sanity:

```powershell
.venv\Scripts\python -c "import fastapi, uvicorn, zmq, httpx; print('server deps OK')"
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_catalog.py
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_catalog.py -v`
Expected: FAIL — `ModuleNotFoundError: server.app`.

- [ ] **Step 4: Implement the catalog (`server/catalog.py`)**

```python
# server/catalog.py
"""Read-only reading of the run-dir layout (config.json, state.json,
metrics.jsonl, elo.jsonl, checkpoints/, games/). Pure functions over a runs_root
Path; NOTHING here writes inside a run dir. All run-id / file-name inputs are
validated so a request can never escape runs_root (path-traversal safe)."""
import json
from pathlib import Path

import chess
import chess.pgn

_SAFE_REJECT = ("/", "\\", "..")


def _safe_name(name: str) -> bool:
    return bool(name) and not any(tok in name for tok in _SAFE_REJECT)


def run_dir(runs_root: Path, run_id: str) -> Path | None:
    """Resolved run dir if run_id is a safe, existing run (has config.json),
    else None."""
    if not _safe_name(run_id):
        return None
    d = Path(runs_root) / run_id
    if d.is_dir() and (d / "config.json").exists():
        return d
    return None


def list_runs(runs_root: Path) -> list[dict]:
    out = []
    root = Path(runs_root)
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "config.json").exists()):
        out.append({
            "run_id": d.name,
            "config": _read_json(d / "config.json", default={}),
            "state": _read_json(d / "state.json", default={}),
        })
    return out


def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def list_checkpoints(rdir: Path) -> list[dict]:
    out = []
    for c in sorted((Path(rdir) / "checkpoints").glob("ckpt_*.pt")):
        try:
            step = int(c.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        out.append({"name": c.name, "step": step})
    return out


def list_games(rdir: Path) -> list[dict]:
    gdir = Path(rdir) / "games"
    if not gdir.exists():
        return []
    return [{"name": p.name} for p in sorted(gdir.glob("*.pgn"))]


def game_pgn_path(rdir: Path, name: str) -> Path | None:
    if not _safe_name(name) or not name.endswith(".pgn"):
        return None
    p = Path(rdir) / "games" / name
    return p if p.exists() else None


def pgn_to_moves(pgn_text: str) -> dict:
    """Parse a PGN into a UCI move list + result (client replays known-legal
    moves, no client-side legality engine needed)."""
    game = chess.pgn.read_game(_StringIO(pgn_text))
    if game is None:
        return {"moves": [], "result": "*"}
    board = game.board()
    moves = []
    for mv in game.mainline_moves():
        moves.append(mv.uci())
        board.push(mv)
    return {"moves": moves, "result": game.headers.get("Result", "*")}


def _read_json(path: Path, default):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return default


# Local import alias keeps the module dependency-light (io is stdlib).
from io import StringIO as _StringIO
```

- [ ] **Step 5: Implement the app factory (`server/app.py`)**

```python
# server/app.py
"""FastAPI app factory: read-only REST catalog + websocket routes (wired in
later tasks) + static web/ mount. create_app(runs_root, cfg, device) is the
single entry point used by scripts/serve.py and every test."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from server import catalog

_WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def create_app(runs_root, cfg=None, device: str = "cpu") -> FastAPI:
    runs_root = Path(runs_root)
    app = FastAPI(title="chessrl server")
    app.state.runs_root = runs_root
    app.state.cfg = cfg
    app.state.device = device

    def _require_run(run_id: str) -> Path:
        rdir = catalog.run_dir(runs_root, run_id)
        if rdir is None:
            raise HTTPException(status_code=404, detail="run not found")
        return rdir

    @app.get("/api/runs")
    def get_runs():
        return catalog.list_runs(runs_root)

    @app.get("/api/runs/{run_id}/metrics")
    def get_metrics(run_id: str):
        return catalog.read_jsonl(_require_run(run_id) / "metrics.jsonl")

    @app.get("/api/runs/{run_id}/elo")
    def get_elo(run_id: str):
        return catalog.read_jsonl(_require_run(run_id) / "elo.jsonl")

    @app.get("/api/runs/{run_id}/checkpoints")
    def get_checkpoints(run_id: str):
        return catalog.list_checkpoints(_require_run(run_id))

    @app.get("/api/runs/{run_id}/games")
    def get_games(run_id: str):
        return catalog.list_games(_require_run(run_id))

    @app.get("/api/runs/{run_id}/games/{name}/pgn", response_class=PlainTextResponse)
    def get_game_pgn(run_id: str, name: str):
        p = catalog.game_pgn_path(_require_run(run_id), name)
        if p is None:
            raise HTTPException(status_code=404, detail="game not found")
        return p.read_text()

    @app.get("/api/runs/{run_id}/games/{name}/moves")
    def get_game_moves(run_id: str, name: str):
        p = catalog.game_pgn_path(_require_run(run_id), name)
        if p is None:
            raise HTTPException(status_code=404, detail="game not found")
        return catalog.pgn_to_moves(p.read_text())

    # Websocket routes are attached by later tasks (rooms/arena/live) via
    # register_* functions to keep this factory cohesive.
    from server.rooms import register_play_ws
    from server.arena import register_arena_ws
    from server.live import register_live_ws

    register_play_ws(app)
    register_arena_ws(app)
    register_live_ws(app)

    # Static UI last so /api and /ws take precedence.
    if _WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
    return app
```

> Note: `app.py` imports `register_play_ws`/`register_arena_ws`/`register_live_ws` from Tasks 3–5. To keep Task 2 independently runnable, create **stub** modules first (each `register_*` a no-op), then flesh them out in their tasks:

```python
# server/rooms.py  (stub for Task 2; replaced in Task 3)
def register_play_ws(app):
    pass
```
```python
# server/arena.py  (stub for Task 2; replaced in Task 4)
def register_arena_ws(app):
    pass
```
```python
# server/live.py   (stub for Task 2; replaced in Task 5)
def register_live_ws(app):
    pass
```

Also create `server/__init__.py` (empty) and `web/.gitkeep` (empty placeholder so the static mount dir exists).

- [ ] **Step 6: Run the catalog tests**

Run: `.venv\Scripts\python -m pytest tests/test_catalog.py -v`
Expected: 9 passed. If `test_games_list_and_pgn_and_moves` fails on the SAN string, the PGN exporter formatting differs — assert on `moves["moves"]` (UCI list) as the load-bearing check; the SAN substring is a convenience. If `test_path_traversal_rejected` fails, the `_safe_name` reject set must cover `/`, `\`, and `..` and run BEFORE any filesystem touch.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml server/__init__.py server/catalog.py server/app.py server/rooms.py server/arena.py server/live.py web/.gitkeep tests/test_catalog.py
git commit -m "feat: M7 FastAPI app factory + read-only REST catalog (path-safe) + server deps"
```

---

### Task 3: GameRoom + human-vs-agent Play websocket

**Files:**
- Replace: `server/rooms.py` (real implementation)
- Test: `tests/test_rooms_play.py`

`GameRoom` holds a `chess.Board`, an optional agent player, and the side the agent plays. The Play websocket `/ws/play` speaks JSON messages: client `{type:"new", checkpoint, simulations, color}` loads a `NetMCTSPlayer` (CPU default), resets the board, and if the agent is to move first replies with the agent's move; client `{type:"move", uci}` validates (illegal → `{type:"error"}`), applies, replies `{type:"state", ...}`, then the agent replies with its move (computed in a thread executor). Each `state` carries `fen, last_move, eval (root_q), thoughts (top-5 visit %), status`. Game-over uses `board.outcome(claim_draw=True)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rooms_play.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_rooms_play.py -v`
Expected: FAIL — `/ws/play` is the Task-2 no-op stub, so `websocket_connect` rejects / no `state` arrives.

- [ ] **Step 3: Implement (`server/rooms.py`)**

```python
# server/rooms.py
"""GameRoom + the human-vs-agent Play websocket.

A GameRoom owns one chess.Board and an optional agent (NetMCTSPlayer, CPU by
default). The Play protocol is JSON over /ws/play:
  client -> {type:"new", run_id, checkpoint, simulations, color}
  client -> {type:"move", uci}
  server -> {type:"state", fen, last_move, eval, thoughts, status, turn}
  server -> {type:"error", message}
The agent reuses M6's NetMCTSPlayer (add_noise=False, argmax visits) so UI play
is the same deterministic search the evaluator rates. Agent move computation runs
in a thread executor so it never blocks the event loop.
"""
import asyncio
from pathlib import Path

import chess

from chessrl.config.config import RunConfig
from server import catalog


def _status(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.outcome(claim_draw=True) is not None:
        return "draw"
    return "playing"


class GameRoom:
    """One human-vs-agent board. agent_color is the color the AGENT plays."""

    def __init__(self):
        self.board = chess.Board()
        self.agent = None
        self.agent_color = chess.BLACK
        self.last_move = None

    def load_agent(self, runs_root: Path, run_id: str, checkpoint: str,
                   simulations: int, human_color: str, device: str) -> None:
        from chessrl.evaluation.players import NetMCTSPlayer

        rdir = catalog.run_dir(runs_root, run_id)
        if rdir is None:
            raise ValueError("unknown run")
        if not catalog._safe_name(checkpoint) or not checkpoint.endswith(".pt"):
            raise ValueError("bad checkpoint")
        ckpt_path = rdir / "checkpoints" / checkpoint
        if not ckpt_path.exists():
            raise ValueError("checkpoint not found")
        run_cfg = RunConfig.from_json(rdir / "config.json")
        self.board = chess.Board()
        self.last_move = None
        self.agent = NetMCTSPlayer(
            "agent", ckpt_path, run_cfg.network, int(simulations), device=device,
        )
        self.agent_color = chess.BLACK if human_color == "white" else chess.WHITE

    def agent_to_move(self) -> bool:
        return (self.agent is not None
                and self.board.outcome(claim_draw=True) is None
                and self.board.turn == self.agent_color)

    def apply_human(self, uci: str) -> bool:
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            return False
        if mv not in self.board.legal_moves:
            return False
        self.board.push(mv)
        self.last_move = uci
        return True

    def agent_move(self) -> str:
        """Synchronous agent move (run in a thread executor by the caller).
        Returns the UCI played; sets last_move and thoughts."""
        mv = self.agent.play(self.board)
        self.board.push(mv)
        self.last_move = mv.uci()
        return self.last_move

    def thoughts(self) -> list:
        """Top-5 (uci, visit_frac) from the agent's last search root, or []."""
        mcts = getattr(self.agent, "_mcts", None)
        return getattr(self.agent, "_last_thoughts", []) if mcts is not None else []

    def root_q(self) -> float:
        return float(getattr(self.agent, "_last_root_q", 0.0))

    def state_msg(self) -> dict:
        return {
            "type": "state",
            "fen": self.board.fen(),
            "last_move": self.last_move,
            "eval": self.root_q(),
            "thoughts": self.thoughts(),
            "status": _status(self.board),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
        }


def register_play_ws(app):
    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/ws/play")
    async def play_ws(ws: WebSocket):
        await ws.accept()
        room = GameRoom()
        runs_root = app.state.runs_root
        device = app.state.device
        try:
            while True:
                msg = await ws.receive_json()
                mtype = msg.get("type")
                if mtype == "new":
                    try:
                        await asyncio.to_thread(
                            room.load_agent, runs_root, msg.get("run_id"),
                            msg.get("checkpoint"), msg.get("simulations", 100),
                            msg.get("color", "white"), device,
                        )
                    except Exception as e:
                        await ws.send_json({"type": "error", "message": str(e)})
                        continue
                    if room.agent_to_move():
                        await asyncio.to_thread(room.agent_move)
                    await ws.send_json(room.state_msg())
                elif mtype == "move":
                    if room.agent is None:
                        await ws.send_json({"type": "error", "message": "no game"})
                        continue
                    if not room.apply_human(msg.get("uci", "")):
                        await ws.send_json({"type": "error", "message": "illegal move"})
                        continue
                    await ws.send_json(room.state_msg())
                    if room.agent_to_move():
                        await asyncio.to_thread(room.agent_move)
                        await ws.send_json(room.state_msg())
                else:
                    await ws.send_json({"type": "error", "message": "unknown type"})
        except WebSocketDisconnect:
            return
        finally:
            if room.agent is not None:
                getattr(room.agent, "close", lambda: None)()
```

`NetMCTSPlayer.play` must expose its last search for the thoughts/eval. Add a tiny capture to `NetMCTSPlayer.play` in `chessrl/evaluation/players.py` (additive, does not change its return or the M6 tests):

```python
    def play(self, board: chess.Board) -> chess.Move:
        from chessrl.chess_env.moves import index_to_move

        visits, root_q = self._mcts.search(board, add_noise=False)
        best_idx = max(visits, key=visits.get)
        # M7: stash top-5 (uci, visit_frac) + root_q for the Play view's eval bar
        # and agent "thoughts". Harmless to evaluation (read only by the server).
        flip = board.turn == chess.BLACK
        total = float(sum(visits.values())) or 1.0
        top = sorted(visits.items(), key=lambda kv: kv[1], reverse=True)[:5]
        self._last_thoughts = [
            [index_to_move(idx, flip, board).uci(), c / total] for idx, c in top
        ]
        self._last_root_q = float(root_q)
        return index_to_move(best_idx, flip, board)
```

Then simplify `GameRoom.thoughts` to read `self.agent._last_thoughts` (guard for None):

```python
    def thoughts(self) -> list:
        return list(getattr(self.agent, "_last_thoughts", [])) if self.agent else []
```

- [ ] **Step 4: Run the play tests**

Run: `.venv\Scripts\python -m pytest tests/test_rooms_play.py tests/test_players.py -v`
Expected: `test_rooms_play.py` 4 passed; `test_players.py` still passes (the additive `_last_thoughts` capture doesn't change `play`'s contract). If a websocket test hangs, the agent move is being computed on the event loop — confirm `room.agent_move` and `room.load_agent` go through `asyncio.to_thread`.

- [ ] **Step 5: Commit**

```powershell
git add server/rooms.py chessrl/evaluation/players.py tests/test_rooms_play.py
git commit -m "feat: M7 GameRoom + human-vs-agent /ws/play (CPU agent in executor, eval + thoughts)"
```

---

### Task 4: Arena websocket + ladder_inbox submission

**Files:**
- Replace: `server/arena.py` (real implementation)
- Test: `tests/test_arena.py`

`/ws/arena` plays any two players move-by-move, pushing `{type:"state", ...}` after each, sleeping `delay_ms` between moves (`asyncio.sleep`), honoring `set_delay`/`pause`/`resume`/`step`/`stop` mid-game. Player spec: `{kind:"checkpoint", run_id, checkpoint, sims}` | `{kind:"random"|"greedy"|"minimax"}` | `{kind:"stockfish", elo|nodes}` (Stockfish only if a path is configured). At game end, write a result JSON to `runs/ladder_inbox/` matching the `store.ingest_inbox` schema `{white, black, z, opening, conditions}` — **the server's only write outside its own process**, and it goes to the inbox, never `ladder.sqlite`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_arena.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_arena.py -v`
Expected: FAIL — `/ws/arena` is the Task-2 stub.

- [ ] **Step 3: Implement (`server/arena.py`)**

```python
# server/arena.py
"""Arena websocket: pit any two players, live-adjustable delay, pause/step/stop,
and submit the result to runs/ladder_inbox/ (the M6 evaluator ingests it). The
server NEVER writes ladder.sqlite directly (spec: arena results go through the
inbox as JSON).

Protocol over /ws/arena:
  client -> {type:"start", white: spec, black: spec, delay_ms, opening_idx, max_plies}
  client -> {type:"set_delay", delay_ms} | {type:"pause"} | {type:"resume"}
            | {type:"step"} | {type:"stop"}
  server -> {type:"state", fen, last_move, ply, turn}
  server -> {type:"gameover", z, result, inbox}
  server -> {type:"error", message}
player spec: {kind:"random"|"greedy"|"minimax"} | {kind:"checkpoint", run_id,
  checkpoint, sims} | {kind:"stockfish", elo|nodes}  (stockfish only if configured)
"""
import asyncio
import json
import time
import uuid
from pathlib import Path

import chess

from chessrl.config.config import RunConfig
from chessrl.evaluation.match import _RESULT_STR
from chessrl.evaluation.openings import opening_board
from server import catalog


def _build_player(spec: dict, runs_root: Path, cfg, device: str):
    kind = spec.get("kind")
    if kind == "random":
        from chessrl.evaluation.players import RandomPlayer
        return RandomPlayer(seed=int(spec.get("seed", 0)))
    if kind == "greedy":
        from chessrl.evaluation.players import GreedyMaterialPlayer
        return GreedyMaterialPlayer(seed=int(spec.get("seed", 0)))
    if kind == "minimax":
        from chessrl.evaluation.players import MinimaxPlayer
        return MinimaxPlayer(depth=int(spec.get("depth", 2)), seed=int(spec.get("seed", 0)))
    if kind == "checkpoint":
        from chessrl.evaluation.players import NetMCTSPlayer
        rdir = catalog.run_dir(runs_root, spec.get("run_id"))
        if rdir is None:
            raise ValueError("unknown run for checkpoint player")
        name = spec.get("checkpoint", "")
        if not catalog._safe_name(name) or not name.endswith(".pt"):
            raise ValueError("bad checkpoint")
        ckpt = rdir / "checkpoints" / name
        if not ckpt.exists():
            raise ValueError("checkpoint not found")
        run_cfg = RunConfig.from_json(rdir / "config.json")
        label = f"{rdir.name}@{name}"
        return NetMCTSPlayer(label, ckpt, run_cfg.network, int(spec.get("sims", 100)), device=device)
    if kind == "stockfish":
        sf_path = getattr(cfg, "stockfish_path", "") if cfg is not None else ""
        if not sf_path:
            raise ValueError("stockfish not configured on this server")
        from chessrl.evaluation.players import StockfishPlayer
        if spec.get("elo") is not None:
            return StockfishPlayer(sf_path, elo=int(spec["elo"]))
        if spec.get("nodes") is not None:
            return StockfishPlayer(sf_path, nodes=int(spec["nodes"]))
        return StockfishPlayer(sf_path, movetime_ms=int(spec.get("movetime_ms", 100)))
    raise ValueError(f"unknown player kind: {kind}")


def _write_inbox(runs_root: Path, white: str, black: str, z: int, opening: int,
                 conditions: dict) -> str:
    inbox = Path(runs_root) / "ladder_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    name = f"arena_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.json"
    payload = {"white": white, "black": black, "z": int(z),
               "opening": int(opening), "conditions": conditions}
    (inbox / name).write_text(json.dumps(payload))
    return name


def register_arena_ws(app):
    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/ws/arena")
    async def arena_ws(ws: WebSocket):
        await ws.accept()
        runs_root = app.state.runs_root
        cfg = app.state.cfg
        device = app.state.device
        white = black = None
        try:
            start = await ws.receive_json()
            if start.get("type") != "start":
                await ws.send_json({"type": "error", "message": "expected start"})
                return
            try:
                white = _build_player(start["white"], runs_root, cfg, device)
                black = _build_player(start["black"], runs_root, cfg, device)
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
                return

            opening_idx = int(start.get("opening_idx", 0))
            board = opening_board(opening_idx)
            max_plies = int(start.get("max_plies", 200))
            delay_ms = int(start.get("delay_ms", 500))
            paused = False
            stopped = False

            async def drain_controls(block: bool):
                """Apply any queued control messages. If block, wait for one."""
                nonlocal delay_ms, paused, stopped
                while True:
                    try:
                        if block:
                            ctrl = await ws.receive_json()
                            block = False
                        else:
                            ctrl = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                    except (asyncio.TimeoutError,):
                        return
                    t = ctrl.get("type")
                    if t == "set_delay":
                        delay_ms = int(ctrl.get("delay_ms", delay_ms))
                    elif t == "pause":
                        paused = True
                    elif t == "resume":
                        paused = False
                    elif t == "stop":
                        stopped = True
                        return
                    elif t == "step":
                        paused = True
                        return                # caller plays exactly one move

            await ws.send_json({
                "type": "state", "fen": board.fen(), "last_move": None,
                "ply": len(board.move_stack),
                "turn": "white" if board.turn == chess.WHITE else "black",
            })

            while board.outcome(claim_draw=True) is None and len(board.move_stack) < max_plies:
                await drain_controls(block=False)
                if stopped:
                    break
                if paused:
                    await drain_controls(block=True)  # wait for resume/step/stop
                    if stopped:
                        break
                player = white if board.turn == chess.WHITE else black
                mv = await asyncio.to_thread(player.play, board)
                board.push(mv)
                await ws.send_json({
                    "type": "state", "fen": board.fen(), "last_move": mv.uci(),
                    "ply": len(board.move_stack),
                    "turn": "white" if board.turn == chess.WHITE else "black",
                })
                if delay_ms > 0 and not paused:
                    # sleep in small slices so set_delay/pause/stop stay responsive
                    slept = 0
                    while slept < delay_ms and not stopped:
                        await drain_controls(block=False)
                        if stopped or paused:
                            break
                        step = min(50, delay_ms - slept)
                        await asyncio.sleep(step / 1000.0)
                        slept += step

            z = _result_z(board)
            cond_w = white.conditions() if hasattr(white, "conditions") else {}
            cond = {"source": "arena", "white": getattr(white, "name", "white"),
                    "black": getattr(black, "name", "black"), **cond_w}
            inbox_name = _write_inbox(runs_root, getattr(white, "name", "white"),
                                      getattr(black, "name", "black"), z,
                                      opening_idx, cond)
            await ws.send_json({"type": "gameover", "z": z,
                                "result": _RESULT_STR[z], "inbox": inbox_name})
        except WebSocketDisconnect:
            return
        finally:
            for p in (white, black):
                if p is not None:
                    getattr(p, "close", lambda: None)()


def _result_z(board: chess.Board) -> int:
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0                       # ongoing-at-cap or draw -> adjudicated draw
    return 1 if outcome.winner == chess.WHITE else -1
```

- [ ] **Step 4: Run the arena tests**

Run: `.venv\Scripts\python -m pytest tests/test_arena.py -v`
Expected: 4 passed. If `test_arena_pause_step_resume` is flaky, the `step` control must set `paused=True` and play exactly one move; the small-slice delay loop must re-check `stopped`/`paused` each slice. If `_RESULT_STR` import fails, confirm it is module-level in `chessrl/evaluation/match.py` (it is, per M6 Task 4).

- [ ] **Step 5: Commit**

```powershell
git add server/arena.py tests/test_arena.py
git commit -m "feat: M7 /ws/arena (delay/pause/step/stop) + ladder_inbox JSON submission"
```

---

### Task 5: Live-training view backend (zmq SUB fan-out)

**Files:**
- Replace: `server/live.py` (real implementation)
- Test: `tests/test_live.py`

A background zmq SUB connects to the configured feed ports, keeps the latest payload per `game_id` (cap to the most-recent active games; drop finished games after a grace period), and `/ws/live` streams updates to browsers plus a `{type:"roster"}` message when the active set changes. The SUB `recv` runs in a thread feeding an `asyncio.Queue` so the event loop never blocks. With no feed configured, `/ws/live` sends an empty roster (the UI shows a hint).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_live.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_live.py -v`
Expected: FAIL — `/ws/live` is the Task-2 stub (no roster emitted).

- [ ] **Step 3: Implement (`server/live.py`)**

```python
# server/live.py
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
```

- [ ] **Step 4: Run the live tests**

Run: `.venv\Scripts\python -m pytest tests/test_live.py -v`
Expected: 2 passed. The publish test relies on the 0.3 s slow-joiner settle for the server's SUB and a burst of 10 publishes — keep both. If `test_live_receives_published_game` times out, confirm the hub thread is started with the running loop and `call_soon_threadsafe` delivers to the per-subscriber queue.

- [ ] **Step 5: Commit**

```powershell
git add server/live.py tests/test_live.py
git commit -m "feat: M7 live-training backend (zmq SUB hub, asyncio fan-out, /ws/live)"
```

---

### Task 6: Web UI — Dashboard + game browser (static, no build)

**Files:**
- Create: `web/index.html`, `web/common.js`, `web/dashboard.js`, `web/style.css`
- (No automated test — static assets; exercised by the Task 8 manual checklist + the catalog endpoints they call, which are already tested.)

The Dashboard fetches `/api/runs`, renders the run list, draws loss + games/hr charts from `/api/runs/{id}/metrics` and the Elo curve from `/api/runs/{id}/elo` (uPlot via CDN), and a game browser lists `/api/runs/{id}/games`, fetches `/games/{name}/moves` on click, and replays with a chessground board (moves are known-legal — a trivial index replayer suffices, no client legality engine). `common.js` holds shared helpers (a `boardFactory`, a `ws` helper, fetch wrappers) reused by Tasks 6–7. No build step: all libraries load from jsDelivr.

- [ ] **Step 1: Shared helpers (`web/common.js`)**

```javascript
// web/common.js — tiny shared helpers (no framework, no build step).
export async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}
export async function getText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.text();
}
export function openWS(path, onMessage) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}${path}`);
  ws.addEventListener("message", (e) => onMessage(JSON.parse(e.data), ws));
  return ws;
}
// chessground factory (loaded from CDN as window.Chessground).
export function makeBoard(el, opts = {}) {
  return window.Chessground(el, Object.assign({
    coordinates: false,
    viewOnly: true,
    fen: "start",
  }, opts));
}
// Build chessground "lastMove" highlight from a uci string.
export function lastMovePair(uci) {
  if (!uci || uci.length < 4) return undefined;
  return [uci.slice(0, 2), uci.slice(2, 4)];
}
```

- [ ] **Step 2: Dashboard page (`web/index.html`)**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>chessrl — dashboard</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.base.css" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.brown.css" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.cburnett.css" />
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.min.css" />
  <link rel="stylesheet" href="/style.css" />
</head>
<body>
  <nav class="topnav">
    <a href="/index.html" class="active">Dashboard</a>
    <a href="/live.html">Live</a>
    <a href="/play.html">Play</a>
    <a href="/arena.html">Arena</a>
  </nav>
  <main>
    <section id="runs"><h2>Runs</h2><ul id="run-list"></ul></section>
    <section id="charts">
      <h2>Metrics <small id="sel-run"></small></h2>
      <div id="loss-chart" class="chart"></div>
      <div id="rate-chart" class="chart"></div>
      <div id="elo-chart" class="chart"></div>
    </section>
    <section id="browser">
      <h2>Game browser</h2>
      <ul id="game-list"></ul>
      <div id="replay">
        <div id="replay-board" class="board-md"></div>
        <div class="controls">
          <button id="rp-prev">◀</button>
          <button id="rp-play">▶ play</button>
          <button id="rp-next">▶▶</button>
          <span id="rp-status"></span>
        </div>
      </div>
    </section>
  </main>
  <script src="https://cdn.jsdelivr.net/npm/chessground@9/dist/chessground.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/uplot@1.6.31/dist/uPlot.iife.min.js"></script>
  <script type="module" src="/dashboard.js"></script>
</body>
</html>
```

- [ ] **Step 3: Dashboard logic (`web/dashboard.js`)**

```javascript
// web/dashboard.js
import { getJSON, makeBoard, lastMovePair } from "/common.js";

let selectedRun = null;
let board = null;
let replay = { moves: [], idx: 0, timer: null };

async function init() {
  const runs = await getJSON("/api/runs");
  const ul = document.getElementById("run-list");
  ul.innerHTML = "";
  for (const r of runs) {
    const li = document.createElement("li");
    const step = (r.state && r.state.step) ?? "—";
    li.textContent = `${r.run_id}  (step ${step})`;
    li.onclick = () => selectRun(r.run_id);
    ul.appendChild(li);
  }
  board = makeBoard(document.getElementById("replay-board"));
  if (runs.length) selectRun(runs[0].run_id);
  wireReplayControls();
}

function lineChart(elId, xs, ys, label) {
  const el = document.getElementById(elId);
  el.innerHTML = "";
  if (!xs.length) { el.textContent = `(no ${label} data)`; return; }
  new uPlot({
    width: el.clientWidth || 360, height: 180, title: label,
    scales: { x: { time: false } },
    series: [{ label: "step" }, { label, stroke: "#3b6ea5" }],
  }, [xs, ys], el);
}

async function selectRun(runId) {
  selectedRun = runId;
  document.getElementById("sel-run").textContent = runId;
  const metrics = await getJSON(`/api/runs/${runId}/metrics`);
  const elo = await getJSON(`/api/runs/${runId}/elo`);
  const steps = metrics.map((m) => m.step);
  lineChart("loss-chart", steps, metrics.map((m) => m.loss ?? null), "loss");
  lineChart("rate-chart", steps, metrics.map((m) => m.games_per_hour ?? null), "games/hr");
  lineChart("elo-chart", elo.map((e) => e.step), elo.map((e) => e.elo), "Elo");
  await loadGames(runId);
}

async function loadGames(runId) {
  const games = await getJSON(`/api/runs/${runId}/games`);
  const ul = document.getElementById("game-list");
  ul.innerHTML = "";
  for (const g of games.slice(0, 200)) {
    const li = document.createElement("li");
    li.textContent = g.name;
    li.onclick = () => loadReplay(runId, g.name);
    ul.appendChild(li);
  }
}

async function loadReplay(runId, name) {
  const data = await getJSON(`/api/runs/${runId}/games/${name}/moves`);
  stopPlay();
  replay = { moves: data.moves, idx: 0, timer: null, result: data.result };
  renderReplay();
}

import { /* placeholder to keep tree-shake happy */ } from "/common.js";

function fenAt(idx) {
  // Replay known-legal moves on a scratch board via chess rules in the browser?
  // We avoid a JS chess engine: chessground renders FENs, and the server already
  // gave us UCI moves. We reconstruct FEN by applying moves with a minimal
  // make-move on a piece map. Simpler: ask chessground to animate by setting
  // lastMove + moving pieces. For a robust no-engine replay we store FEN per ply
  // from the server moves endpoint is not available, so we apply moves with a
  // tiny built-in mover below.
  return null;
}

// Minimal UCI mover over a chessground-style piece map keyed by square.
function startMap() {
  const map = new Map();
  const back = ["r","n","b","q","k","b","n","r"];
  for (let f = 0; f < 8; f++) {
    map.set(sq(f, 0), { role: roleOf(back[f]), color: "white" });
    map.set(sq(f, 1), { role: "pawn", color: "white" });
    map.set(sq(f, 6), { role: "pawn", color: "black" });
    map.set(sq(f, 7), { role: roleOf(back[f]), color: "black" });
  }
  return map;
}
function sq(file, rank) { return "abcdefgh"[file] + (rank + 1); }
function roleOf(c) {
  return { r: "rook", n: "knight", b: "bishop", q: "queen", k: "king", p: "pawn" }[c];
}
function applyUci(map, uci) {
  const from = uci.slice(0, 2), to = uci.slice(2, 4), promo = uci[4];
  const piece = map.get(from);
  if (!piece) return;
  map.delete(from);
  // en-passant: pawn moves diagonally to an empty square -> capture passed pawn
  if (piece.role === "pawn" && from[0] !== to[0] && !map.get(to)) {
    map.delete(to[0] + from[1]);
  }
  // castling: king two squares -> move the rook too
  if (piece.role === "king" && Math.abs(from.charCodeAt(0) - to.charCodeAt(0)) === 2) {
    const rank = from[1];
    if (to[0] === "g") { map.set("f" + rank, map.get("h" + rank)); map.delete("h" + rank); }
    if (to[0] === "c") { map.set("d" + rank, map.get("a" + rank)); map.delete("a" + rank); }
  }
  map.set(to, promo ? { role: roleOf(promo), color: piece.color } : piece);
}
function mapToCg(map) {
  const pieces = new Map();
  for (const [s, p] of map) pieces.set(s, { role: p.role, color: p.color });
  return pieces;
}

function renderReplay() {
  const map = startMap();
  for (let i = 0; i < replay.idx; i++) applyUci(map, replay.moves[i]);
  const last = replay.idx > 0 ? lastMovePair(replay.moves[replay.idx - 1]) : undefined;
  board.set({ fen: undefined, lastMove: last });
  board.setPieces(mapToCg(map));
  document.getElementById("rp-status").textContent =
    `${replay.idx}/${replay.moves.length}  ${replay.result || ""}`;
}

function wireReplayControls() {
  document.getElementById("rp-prev").onclick = () => { if (replay.idx > 0) { replay.idx--; renderReplay(); } };
  document.getElementById("rp-next").onclick = () => { if (replay.idx < replay.moves.length) { replay.idx++; renderReplay(); } };
  document.getElementById("rp-play").onclick = togglePlay;
}
function togglePlay() {
  if (replay.timer) { stopPlay(); return; }
  replay.timer = setInterval(() => {
    if (replay.idx >= replay.moves.length) { stopPlay(); return; }
    replay.idx++; renderReplay();
  }, 600);
}
function stopPlay() { if (replay.timer) { clearInterval(replay.timer); replay.timer = null; } }

init();
```

> Implementer note: the duplicate `import { } from "/common.js"` placeholder line and the unused `fenAt` stub are an artifact — delete them; `setPieces` + the minimal `applyUci` mover is the chosen no-engine replay path (chessground renders the piece map directly, so no client legality engine is needed; moves are server-validated known-legal). Keep `applyUci`'s en-passant + castling + promotion handling (the three special cases a naive mover would render wrong).

- [ ] **Step 4: Styles (`web/style.css`)**

```css
/* web/style.css — minimal, no framework. */
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; color: #222; }
.topnav { display: flex; gap: 1rem; background: #2b2b2b; padding: .6rem 1rem; }
.topnav a { color: #ccc; text-decoration: none; }
.topnav a.active, .topnav a:hover { color: #fff; }
main { display: grid; grid-template-columns: 240px 1fr; gap: 1rem; padding: 1rem; }
#charts { grid-column: 2; }
#runs, #browser { grid-column: 1; }
.chart { width: 100%; max-width: 480px; margin: .5rem 0; }
ul { list-style: none; padding: 0; margin: 0; }
#run-list li, #game-list li { padding: .25rem .4rem; cursor: pointer; border-radius: 4px; }
#run-list li:hover, #game-list li:hover { background: #eef; }
.board-sm { width: 180px; height: 180px; }
.board-md { width: 320px; height: 320px; }
.board-lg { width: 480px; height: 480px; }
.controls { margin-top: .5rem; display: flex; gap: .5rem; align-items: center; }
#live-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .5rem; }
.eval-bar { width: 24px; height: 480px; background: #ddd; position: relative; }
.eval-fill { position: absolute; bottom: 0; width: 100%; background: #4a4; }
.thoughts li { font-family: monospace; }
.hint { color: #888; font-style: italic; }
```

- [ ] **Step 5: Manual smoke (deferred to Task 8 integration gate)**

The Dashboard cannot be unit-tested headlessly here; its data endpoints are covered by `test_catalog.py`. Verification is the Task 8 checklist (open the page, confirm runs list, charts draw, a PGN replays). Commit the static assets now.

- [ ] **Step 6: Commit**

```powershell
git add web/index.html web/common.js web/dashboard.js web/style.css
git commit -m "feat: M7 dashboard + game browser UI (chessground + uPlot via CDN, no build step)"
```

---

### Task 7: Web UI — Live + Play + Arena views

**Files:**
- Create: `web/live.html`, `web/live.js`
- Create: `web/play.html`, `web/play.js`
- Create: `web/arena.html`, `web/arena.js`
- (No automated test — static assets over the already-tested websockets; exercised by the Task 8 manual checklist.)

Three views over the websockets from Tasks 3–5. **Live:** a grid of up to 12 small chessground boards updating from `/ws/live` (render the roster, subscribe, update by `game_id`; if the roster is empty show the no-feed hint). **Play:** a full board with drag-drop (chessground in legal-ish move mode; the server validates and the client reverts on `{type:"error"}`), an eval bar div, and a thoughts list, over `/ws/play`. **Arena:** two player pickers populated from `/api/runs` checkpoints + builtin kinds, a delay slider wired to `set_delay`, pause/step/stop buttons, over `/ws/arena`.

- [ ] **Step 1: Live view (`web/live.html` + `web/live.js`)**

```html
<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>chessrl — live</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.base.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.brown.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.cburnett.css"/>
<link rel="stylesheet" href="/style.css"/>
</head><body>
<nav class="topnav"><a href="/index.html">Dashboard</a><a href="/live.html" class="active">Live</a><a href="/play.html">Play</a><a href="/arena.html">Arena</a></nav>
<main style="grid-template-columns:1fr;">
  <h2>Live self-play <small id="live-hint" class="hint"></small></h2>
  <div id="live-grid"></div>
</main>
<script src="https://cdn.jsdelivr.net/npm/chessground@9/dist/chessground.min.js"></script>
<script type="module" src="/live.js"></script>
</body></html>
```

```javascript
// web/live.js
import { openWS, makeBoard, lastMovePair } from "/common.js";

const MAX_BOARDS = 12;
const cells = new Map();   // game_id -> { board, el }

function ensureCell(gameId) {
  if (cells.has(gameId)) return cells.get(gameId);
  if (cells.size >= MAX_BOARDS) return null;   // sampled + capped at 12
  const wrap = document.createElement("div");
  wrap.className = "live-cell";
  const title = document.createElement("div");
  title.textContent = gameId;
  title.style.font = "11px monospace";
  const boardEl = document.createElement("div");
  boardEl.className = "board-sm";
  wrap.append(title, boardEl);
  document.getElementById("live-grid").appendChild(wrap);
  const board = makeBoard(boardEl, { coordinates: false });
  const cell = { board, el: wrap };
  cells.set(gameId, cell);
  return cell;
}

function onMsg(msg) {
  const hint = document.getElementById("live-hint");
  if (msg.type === "roster") {
    if (!msg.games.length) hint.textContent = "(no live feed — start a run with selfplay.feed_port set)";
    else hint.textContent = "";
    return;
  }
  if (msg.type === "update") {
    const g = msg.game;
    const cell = ensureCell(g.game_id);
    if (!cell) return;
    cell.board.set({ fen: g.fen.split(" ")[0], lastMove: lastMovePair(g.last_move_uci) });
    if (g.done) cell.el.style.opacity = 0.5;
  }
}

openWS("/ws/live", onMsg);
```

- [ ] **Step 2: Play view (`web/play.html` + `web/play.js`)**

```html
<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>chessrl — play</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.base.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.brown.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.cburnett.css"/>
<link rel="stylesheet" href="/style.css"/>
</head><body>
<nav class="topnav"><a href="/index.html">Dashboard</a><a href="/live.html">Live</a><a href="/play.html" class="active">Play</a><a href="/arena.html">Arena</a></nav>
<main style="grid-template-columns:auto auto 1fr;">
  <div>
    <div class="controls">
      <select id="run"></select>
      <select id="ckpt"></select>
      <label>sims <input id="sims" type="number" value="100" min="1" style="width:5rem"/></label>
      <select id="color"><option value="white">White</option><option value="black">Black</option></select>
      <button id="newgame">New game</button>
    </div>
    <div id="play-board" class="board-lg"></div>
    <div id="status" class="hint"></div>
  </div>
  <div class="eval-bar"><div id="eval-fill" class="eval-fill"></div></div>
  <div><h3>Agent thoughts</h3><ul id="thoughts" class="thoughts"></ul></div>
</main>
<script src="https://cdn.jsdelivr.net/npm/chessground@9/dist/chessground.min.js"></script>
<script type="module" src="/play.js"></script>
</body></html>
```

```javascript
// web/play.js
import { getJSON, openWS, makeBoard, lastMovePair } from "/common.js";

let ws = null, board = null, myColor = "white", lastFen = "start";

async function populateRuns() {
  const runs = await getJSON("/api/runs");
  const runSel = document.getElementById("run");
  runSel.innerHTML = "";
  for (const r of runs) {
    const o = document.createElement("option");
    o.value = r.run_id; o.textContent = r.run_id; runSel.appendChild(o);
  }
  runSel.onchange = populateCkpts;
  if (runs.length) await populateCkpts();
}
async function populateCkpts() {
  const runId = document.getElementById("run").value;
  const cks = await getJSON(`/api/runs/${runId}/checkpoints`);
  const sel = document.getElementById("ckpt");
  sel.innerHTML = "";
  for (const c of cks) {
    const o = document.createElement("option");
    o.value = c.name; o.textContent = `step ${c.step}`; sel.appendChild(o);
  }
}

function setupBoard() {
  board = makeBoard(document.getElementById("play-board"), {
    viewOnly: false,
    movable: { free: false, color: myColor, events: { after: onUserMove } },
  });
}
function onUserMove(from, to) {
  const uci = from + to;   // promotion: server accepts q-promo; UI keeps it simple
  ws.send(JSON.stringify({ type: "move", uci }));
}

function applyState(msg) {
  lastFen = msg.fen;
  board.set({ fen: msg.fen.split(" ")[0], lastMove: lastMovePair(msg.last_move),
    turn: msg.turn, movable: { color: msg.turn === myColor ? myColor : undefined } });
  // eval bar: root_q in [-1,1] from side-to-move -> normalize to white's view
  const q = msg.turn === "white" ? msg.eval : -msg.eval;
  const pct = Math.round((q + 1) / 2 * 100);
  document.getElementById("eval-fill").style.height = pct + "%";
  const t = document.getElementById("thoughts");
  t.innerHTML = "";
  for (const [uci, frac] of (msg.thoughts || [])) {
    const li = document.createElement("li");
    li.textContent = `${uci}  ${(frac * 100).toFixed(0)}%`;
    t.appendChild(li);
  }
  document.getElementById("status").textContent = msg.status;
}

function newGame() {
  myColor = document.getElementById("color").value;
  if (ws) ws.close();
  setupBoard();
  ws = openWS("/ws/play", (msg) => {
    if (msg.type === "error") { board.set({ fen: lastFen.split(" ")[0] }); document.getElementById("status").textContent = msg.message; return; }
    if (msg.type === "state") applyState(msg);
  });
  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({
      type: "new", run_id: document.getElementById("run").value,
      checkpoint: document.getElementById("ckpt").value,
      simulations: Number(document.getElementById("sims").value),
      color: myColor,
    }));
  });
}

document.getElementById("newgame").onclick = newGame;
populateRuns();
```

- [ ] **Step 3: Arena view (`web/arena.html` + `web/arena.js`)**

```html
<!doctype html><html lang="en"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>chessrl — arena</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.base.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.brown.css"/>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/chessground@9/assets/chessground.cburnett.css"/>
<link rel="stylesheet" href="/style.css"/>
</head><body>
<nav class="topnav"><a href="/index.html">Dashboard</a><a href="/live.html">Live</a><a href="/play.html">Play</a><a href="/arena.html" class="active">Arena</a></nav>
<main style="grid-template-columns:auto 1fr;">
  <div>
    <div class="controls">
      <label>White <select id="white-kind"></select></label>
      <select id="white-ckpt" class="ckpt-pick" hidden></select>
    </div>
    <div class="controls">
      <label>Black <select id="black-kind"></select></label>
      <select id="black-ckpt" class="ckpt-pick" hidden></select>
    </div>
    <div class="controls">
      <label>delay <input id="delay" type="range" min="0" max="2000" value="500"/></label>
      <span id="delay-val">500ms</span>
    </div>
    <div class="controls">
      <button id="start">Start</button>
      <button id="pause">Pause</button>
      <button id="resume">Resume</button>
      <button id="step">Step</button>
      <button id="stop">Stop</button>
    </div>
    <div id="arena-board" class="board-lg"></div>
    <div id="arena-status" class="hint"></div>
  </div>
</main>
<script src="https://cdn.jsdelivr.net/npm/chessground@9/dist/chessground.min.js"></script>
<script type="module" src="/arena.js"></script>
</body></html>
```

```javascript
// web/arena.js
import { getJSON, openWS, makeBoard, lastMovePair } from "/common.js";

let ws = null, board = null, runs = [];
const KINDS = ["random", "greedy", "minimax", "checkpoint", "stockfish"];

async function init() {
  runs = await getJSON("/api/runs");
  for (const side of ["white", "black"]) {
    const sel = document.getElementById(`${side}-kind`);
    for (const k of KINDS) { const o = document.createElement("option"); o.value = k; o.textContent = k; sel.appendChild(o); }
    sel.onchange = () => toggleCkpt(side);
    await fillCkpts(side);
  }
  board = makeBoard(document.getElementById("arena-board"));
  wireControls();
}
async function fillCkpts(side) {
  const sel = document.getElementById(`${side}-ckpt`);
  sel.innerHTML = "";
  for (const r of runs) {
    const cks = await getJSON(`/api/runs/${r.run_id}/checkpoints`);
    for (const c of cks) {
      const o = document.createElement("option");
      o.value = JSON.stringify({ run_id: r.run_id, checkpoint: c.name });
      o.textContent = `${r.run_id} step ${c.step}`;
      sel.appendChild(o);
    }
  }
}
function toggleCkpt(side) {
  const isCkpt = document.getElementById(`${side}-kind`).value === "checkpoint";
  document.getElementById(`${side}-ckpt`).hidden = !isCkpt;
}
function spec(side) {
  const kind = document.getElementById(`${side}-kind`).value;
  if (kind === "checkpoint") {
    const v = JSON.parse(document.getElementById(`${side}-ckpt`).value || "{}");
    return { kind, run_id: v.run_id, checkpoint: v.checkpoint, sims: 100 };
  }
  if (kind === "stockfish") return { kind, elo: 1320 };
  return { kind };
}

function wireControls() {
  const delay = document.getElementById("delay");
  delay.oninput = () => {
    document.getElementById("delay-val").textContent = delay.value + "ms";
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "set_delay", delay_ms: Number(delay.value) }));
  };
  document.getElementById("start").onclick = start;
  for (const t of ["pause", "resume", "step", "stop"]) {
    document.getElementById(t).onclick = () => ws && ws.send(JSON.stringify({ type: t }));
  }
}
function start() {
  if (ws) ws.close();
  ws = openWS("/ws/arena", (msg) => {
    if (msg.type === "error") { document.getElementById("arena-status").textContent = "error: " + msg.message; return; }
    if (msg.type === "state") board.set({ fen: msg.fen.split(" ")[0], lastMove: lastMovePair(msg.last_move) });
    if (msg.type === "gameover") document.getElementById("arena-status").textContent = `result ${msg.result} (z=${msg.z}) → ${msg.inbox}`;
  });
  ws.addEventListener("open", () => ws.send(JSON.stringify({
    type: "start", white: spec("white"), black: spec("black"),
    delay_ms: Number(document.getElementById("delay").value), opening_idx: 0, max_plies: 200,
  })));
}

init();
```

- [ ] **Step 4: Verify the static files are served (endpoint-level smoke)**

After Task 2's static mount, a `TestClient` GET of each page returns 200. Add this quick check to confirm the mount + files line up (run it ad-hoc, not a committed test unless desired):

```powershell
.venv\Scripts\python -c "from fastapi.testclient import TestClient; from server.app import create_app; from pathlib import Path; c=TestClient(create_app(Path('runs'))); print({p: c.get(p).status_code for p in ['/index.html','/live.html','/play.html','/arena.html','/common.js','/style.css']})"
```

Expected: every path returns `200`.

- [ ] **Step 5: Commit**

```powershell
git add web/live.html web/live.js web/play.html web/play.js web/arena.html web/arena.js
git commit -m "feat: M7 live + play + arena UI views (vanilla JS over the M7 websockets)"
```

---

### Task 8: serve.py + integration gate

**Files:**
- Create: `scripts/serve.py`
- (No new test file — this task launches the server against the real `runs/`, curls the REST endpoints, runs a live websocket play game, and prints a manual-verification checklist. Findings go in the milestone report.)

`scripts/serve.py` parses `--runs-root`, `--host` (default `127.0.0.1`), `--port` (default `8000`), `--feed-ports` (comma list, default empty → live disabled), `--device` (default `cpu`), `--stockfish` (optional path), builds the app via `create_app`, attaches `feed_ports`/cfg, and runs uvicorn. The server must **start and serve even when the feed and Stockfish are absent**.

- [ ] **Step 1: Implement `scripts/serve.py`**

```python
# scripts/serve.py
"""Run the chessrl web server (REST catalog + websockets + static UI).

  python scripts/serve.py --runs-root runs
  python scripts/serve.py --runs-root runs --feed-ports 5550,5551,5552,5553
  python scripts/serve.py --runs-root runs --device cuda          # opt-in GPU
  python scripts/serve.py --runs-root runs --stockfish tools/stockfish/stockfish.exe

The server reads run dirs READ-ONLY and submits arena results to
runs/ladder_inbox/ (never ladder.sqlite). Bind 127.0.0.1 by default (LAN-only,
trusted network — no auth, per the spec).
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import uvicorn

from server.app import create_app


@dataclass
class ServerConfig:
    stockfish_path: str = ""


def build(argv=None):
    ap = argparse.ArgumentParser(description="chessrl web server")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--feed-ports", default="", help="comma-separated PUB ports to subscribe (live view)")
    ap.add_argument("--device", default="cpu", help="server inference device (cpu default; cuda opt-in)")
    ap.add_argument("--stockfish", default="", help="path to a Stockfish binary (enables stockfish arena players)")
    args = ap.parse_args(argv)

    feed_ports = [int(p) for p in args.feed_ports.split(",") if p.strip()]
    cfg = ServerConfig(stockfish_path=args.stockfish)
    app = create_app(Path(args.runs_root), cfg=cfg, device=args.device)
    app.state.feed_ports = feed_ports
    return app, args


def main(argv=None) -> int:
    app, args = build(argv)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: REST smoke against the real runs/ (server not even running — TestClient)**

```powershell
.venv\Scripts\python -c @"
from fastapi.testclient import TestClient
from pathlib import Path
from scripts.serve import build
app, _ = build(['--runs-root','runs'])
c = TestClient(app)
runs = c.get('/api/runs').json()
print('runs:', [r['run_id'] for r in runs])
rid = runs[0]['run_id']
print('metrics rows:', len(c.get(f'/api/runs/{rid}/metrics').json()))
print('elo rows:', len(c.get(f'/api/runs/{rid}/elo').json()))
print('checkpoints:', len(c.get(f'/api/runs/{rid}/checkpoints').json()))
games = c.get(f'/api/runs/{rid}/games').json()
print('games:', len(games))
if games:
    mv = c.get(f'/api/runs/{rid}/games/{games[0][\"name\"]}/moves').json()
    print('first game plies:', len(mv['moves']), 'result', mv['result'])
"@
```

Expected: prints the real dev-tiny run ids (e.g. `dev-tiny-20260611-214204`), non-zero metrics/elo/checkpoints/games counts, and a first-game ply count + result. This proves the REST catalog reads the real run-dir layout produced by M4–M6 with zero writes.

- [ ] **Step 3: Live websocket play game against a real checkpoint (TestClient)**

```powershell
.venv\Scripts\python -c @"
from fastapi.testclient import TestClient
from pathlib import Path
from scripts.serve import build
app, _ = build(['--runs-root','runs'])
c = TestClient(app)
runs = c.get('/api/runs').json()
rid = runs[0]['run_id']
ckpt = c.get(f'/api/runs/{rid}/checkpoints').json()[0]['name']
with c.websocket_connect('/ws/play') as ws:
    ws.send_json({'type':'new','run_id':rid,'checkpoint':ckpt,'simulations':16,'color':'white'})
    print('opening state:', ws.receive_json()['fen'].split()[1], 'to move')
    ws.send_json({'type':'move','uci':'e2e4'})
    print('human:', ws.receive_json()['last_move'])
    agent = ws.receive_json()
    print('agent:', agent['last_move'], 'eval', round(agent['eval'],3), 'thoughts', len(agent['thoughts']))
print('play websocket OK')
"@
```

Expected: prints `opening state: w to move`, `human: e2e4`, an agent reply move with a finite eval and up to 5 thoughts, and `play websocket OK`. This runs the real `NetMCTSPlayer` (CPU, 16 sims) over a real M6 checkpoint end-to-end through the websocket.

- [ ] **Step 4: Arena game + inbox submission (TestClient), then verify M6 can ingest it**

```powershell
.venv\Scripts\python -c @"
from fastapi.testclient import TestClient
from pathlib import Path
from scripts.serve import build
app, _ = build(['--runs-root','runs'])
c = TestClient(app)
before = len(list((Path('runs')/'ladder_inbox').glob('*.json'))) if (Path('runs')/'ladder_inbox').exists() else 0
with c.websocket_connect('/ws/arena') as ws:
    ws.send_json({'type':'start','white':{'kind':'random'},'black':{'kind':'greedy'},
                  'delay_ms':0,'opening_idx':0,'max_plies':60})
    while True:
        m = ws.receive_json()
        if m['type'] == 'gameover':
            print('arena gameover z=', m['z'], 'inbox', m['inbox'])
            break
after = len(list((Path('runs')/'ladder_inbox').glob('*.json')))
print('inbox files before/after:', before, after)
"@
```

Expected: prints `arena gameover z=<-1|0|1> inbox arena_...json` and `inbox files before/after: <n> <n+1>`. Leave the inbox file in place — the M6 evaluator's `ingest_inbox` consumes it on its next pass (or delete it manually to keep `runs/` clean: `Remove-Item runs\ladder_inbox\arena_*.json`). This proves the server's *only* write path is the inbox and it matches the `store.ingest_inbox` schema.

- [ ] **Step 5: Run the full fast suite (regression gate — all existing tests stay green)**

Run: `.venv\Scripts\python -m pytest --durations=15`
Expected: all fast tests pass — the M1–M6 suite unchanged (the spec's 123 tests; `feed_port` defaults to 0 so nothing existing imports zmq or changes behavior) plus the new M7 tests (`test_feed`, `test_catalog`, `test_rooms_play`, `test_arena`, `test_live`). The zmq round-trip tests rely on the localhost slow-joiner settle sleep — they are fast (sub-second) and stay in the default suite. Any `slow`-marked tests remain deselected.

- [ ] **Step 6: Launch the real server and walk the manual checklist**

```powershell
.venv\Scripts\python scripts\serve.py --runs-root runs
```

Expected: uvicorn prints `Uvicorn running on http://127.0.0.1:8000`. Open `http://127.0.0.1:8000` in a browser and walk this checklist (record pass/fail in the milestone report):

- [ ] **Dashboard** (`/index.html`): the run list shows the real `dev-tiny-*` runs; clicking one draws the loss, games/hr, and Elo charts (the Elo curve has the M6 points); the game browser lists PGNs; clicking a game replays it with prev/next/play and shows the result.
- [ ] **Live** (`/live.html`): with no `--feed-ports`, shows the "no live feed" hint. (Optional: start a worker run with `selfplay.feed_port` set + relaunch with `--feed-ports`, and confirm boards appear and animate — this is the only step needing a running trainer; note it as optional in the report.)
- [ ] **Play** (`/play.html`): pick a run + checkpoint + sims + color, click New game; drag a legal move — the agent replies, the eval bar moves, and the thoughts list shows top moves with visit %; an illegal drag reverts.
- [ ] **Arena** (`/arena.html`): pick two players (e.g. greedy vs minimax, or a checkpoint vs random), set the delay slider, Start; moves play with the delay; pause/step/resume/stop work; at game end the status shows the result and the inbox filename.

The server must serve all four views with the feed and Stockfish absent (the default launch). Stop with Ctrl-C.

- [ ] **Step 7: Commit**

```powershell
git add scripts/serve.py
git commit -m "feat: M7 scripts/serve.py + integration gate (REST/ws smoke against real runs, manual checklist)"
```

---

## Self-Review Notes

- **Spec coverage (M7):** the FastAPI server with REST catalog (runs, checkpoints, ratings via `elo.jsonl`, stored games) + websockets for live content (Task 2–5); the **GameRoom** abstraction backing human-vs-agent Play and Arena (Task 3, 4); the four views — **Dashboard** (run list, loss/Elo/throughput curves), **Live training** (sampled, capped at 12 rotating boards, per-game-topic subscription, never the firehose), **Play** (checkpoint/color/sims, drag-drop, eval bar + agent thoughts via visit %), **Arena** (any two players, live delay slider, pause/step/stop, results to `ladder_inbox/`) (Task 6, 7); the **ZeroMQ live feed** (PUB/SUB over `tcp://127.0.0.1`, one topic per game, bounded HWM drop-on-full, publishing never blocks) (Task 1); the read-only run-dir contract and inbox-only writes (Task 2, 4); `scripts/serve.py` (Task 8). This satisfies Success Criterion #3 (watch live self-play, play any checkpoint, pit any two players with a settable delay). Out-of-scope items (evaluator/rating-fit/Stockfish provisioning → M6 done; auth/HTTPS; TensorBoard; retention; automated browser tests; default-GPU inference; the full live firehose) are stated in the header and not built.
- **Design refinements made (and why):**
  - *Per-move publish hook found in `_play_one_move`.* I verified `play_games_concurrent` (read `chessrl/selfplay/concurrent.py`): the natural hook is the end of `_play_one_move`, right after `mcts.advance` commits the move and `_check_pre_move_termination` sets `g.done`/`g.z` — every field the normative payload needs (`fen`, `last_move_uci`, `ply`, `root_q`, `top_moves`, `done`, `z`) is in scope there. I add `_publish_move` in both the resign-return branch and the normal path so a resigned game still emits its terminal frame. `top_moves` reuses the already-computed `visits`/`counts` and decodes via `index_to_move(idx, flip, board)` (verified signature in `chessrl/chess_env/moves.py`). A `game_id` slot was added to `_Game.__slots__` (it has `__slots__`, so the field must be declared — a real gotcha caught by reading the class).
  - *Lazy zmq, NullPublisher default.* `import zmq` lives only inside `feed.py` and `live.py`; `play_games_concurrent`/`worker_main` default to `NullPublisher`, and the worker constructs a `FeedPublisher` only when `cfg.selfplay.feed_port > 0`. So the existing 123-test suite and any feed-off training run never import zmq — `test_selfplay_config_has_feed_port_default_zero` and the unchanged existing concurrent/worker tests lock this. Publishing is `zmq.DONTWAIT` with `zmq.Again` → drop, and the drop-on-full test proves 1000 sends to a full queue with no subscriber complete in bounded time (never block a worker), per the spec's "publishing never blocks."
  - *Per-worker port, range-subscribe.* Each worker binds `feed_port + worker_id`; the server's SUB connects to the whole range (PUB/SUB allows one SUB → many PUBs). This avoids a shared bind (two PUBs can't bind the same port) and is documented in `feed.py` + the worker wiring. The live tests use a real PUB on a free port and assert end-to-end delivery through the server SUB hub.
  - *Read-only + path-safety, enforced before any FS touch.* `catalog._safe_name` rejects `/`, `\`, `..` and runs before resolving any path; `run_dir`/`game_pgn_path` return `None` for unsafe or missing inputs → 404. The traversal test (`%2f`-encoded `..`) locks it. The ONLY write path in the whole server is `arena._write_inbox` → `runs/ladder_inbox/*.json`; nothing writes inside a run dir or touches `ladder.sqlite`. The arena inbox test + a `LadderStore.ingest_inbox` round-trip test prove the JSON matches the M6 schema `{white, black, z, opening, conditions}` exactly (verified against `chessrl/evaluation/store.py`).
  - *Reuse `NetMCTSPlayer` for Play/Arena agents.* Verified its constructor `(name, checkpoint_path, network_cfg, simulations, device, seed)` and that `play` searches `add_noise=False` + argmax (`chessrl/evaluation/players.py`). The server builds it `device="cpu"` by default (spec: server inference CPU-default). I added an *additive* capture in `play` (`_last_thoughts`, `_last_root_q`) so the Play view's eval bar + thoughts read the real search without changing `play`'s return — `test_players.py` (M6) still passes, and `test_rooms_play.py` asserts the thoughts/eval surface.
  - *CPU agent move in a thread executor.* `load_agent` and `agent_move` (Play) and `player.play` (Arena) go through `asyncio.to_thread`, so a ~<2 s CPU search never blocks the event loop or other websockets. The websocket tests would hang if this regressed (noted as the failure signature).
  - *Arena control responsiveness.* Delay is slept in ≤50 ms slices, re-checking `set_delay`/`pause`/`stop` each slice, and `step` plays exactly one move then re-pauses — so the slider/pause/step/stop stay live mid-game. `test_arena_pause_step_resume` (huge delay → step advances exactly one ply) locks it.
  - *No-engine PGN replay + live render.* I chose the **server-side** moves endpoint (`/games/{name}/moves` → `{moves:[uci...], result}` parsed by python-chess) over client PGN parsing — simpler and no JS chess dependency. The browser replays known-legal UCI on a chessground piece map via a minimal `applyUci` mover that handles the three special cases (en-passant, castling, promotion) a naive mover renders wrong; chessground draws the map directly, so no client legality engine is needed.
  - *Static, no build step.* chessground 9 + uPlot load from jsDelivr; `common.js` is a shared ES-module of helpers; the four pages are plain HTML + `<script type="module">`. The static mount is added last in `create_app` so `/api` and `/ws` take precedence; an ad-hoc TestClient GET of each page (Task 7 Step 4) confirms they serve 200.
- **Type/name consistency check against the real M1–M6 signatures I read:**
  - `play_games_concurrent(evaluator_many, mcts_cfg, sp_cfg, rng, num_games)` and `_play_one_move(g, mcts, mcts_cfg, sp_cfg, rng)` — extended with `publisher`/`game_id_prefix` additively; `_Game.__slots__`, `mcts.visit_counts`/`root_q`/`advance`, `index_to_move(idx, flip, board)` all verified in `concurrent.py`/`batched.py`/`moves.py`. ✓
  - `SelfPlayConfig` is a frozen dataclass with `from_dict`'s `build(SelfPlayConfig, "selfplay")`; adding `feed_port: int = 0` round-trips via `asdict`/`to_json` unchanged (same pattern M6 used for `EvalConfig`). ✓
  - `NetMCTSPlayer(name, ckpt, network_cfg, simulations, device="cpu")`, `.play(board)->Move`, optional `.close()`; `BatchedNetEvaluator.from_checkpoint(path, network_cfg, device)` underneath — matches `players.py`/`network.py`. ✓
  - `RandomPlayer`/`GreedyMaterialPlayer`/`MinimaxPlayer`/`StockfishPlayer` constructors + `.name`/`.play`/`.conditions`/`.close` — matches `players.py`; arena `_build_player` uses exactly these. ✓
  - `opening_board(idx)` and module-level `_RESULT_STR` in `chessrl/evaluation/match.py`; `LadderStore.ingest_inbox` schema `{white, black, z, opening, conditions}` in `store.py` — arena submission matches field-for-field. ✓
  - `Trainer(net, TrainingConfig, run_dir).save_checkpoint() -> Path` writing `checkpoints/ckpt_{step:08d}.pt` — the Play test builds a real checkpoint this way (same as M6's daemon test). ✓
  - Run-dir layout (`config.json`, `state.json`, `metrics.jsonl`, `elo.jsonl`, `checkpoints/`, `games/*.pgn`, `eval_games/`) — verified against the real `runs/dev-tiny-20260611-214204`; the catalog reads exactly these, and the real `elo.jsonl` rows have the `{ts, step, ckpt, elo, nu}` keys the Dashboard plots. ✓
  - `chess.Board.outcome(claim_draw=True)`, `chess.pgn.Game.from_board`/`read_game`, `index_to_move`/`move_to_index` — python-chess APIs already used across M1–M6. ✓
- **No placeholders:** every task ships complete, runnable test + implementation code. The one deliberate artifact — the stray duplicate `import` line and unused `fenAt` stub in `dashboard.js` — is explicitly flagged with a "delete these" implementer note and the chosen `setPieces` + `applyUci` replay path is spelled out in full, so the implementer ends with concrete, working code and no TODO. The Task-2 `register_*` stubs are intentional and each is replaced by its own task (3/4/5) with the real implementation.
- **Known intentional simplifications:** the Live view caps at 12 boards client-side and the server keeps ≤24 recent games (the spec's "sampled, capped" — the browser samples which to render); promotion in the Play view defaults to queen (the server's `apply_human` accepts the 4-char UCI and python-chess fills the implicit queen flag — under-promotion UI is post-M7 polish); the eval bar maps `root_q∈[-1,1]` linearly to a 0–100% fill (a centipawn/logistic mapping is cosmetic); arena `conditions` records the source + player names + (for Stockfish white) its pinned options — enough for the evaluator to attribute the result, without duplicating M6's full per-engine condition recording; the server is single-process with one shared `LiveHub` lazily created on the first `/ws/live` connect (fine for a LAN-only, single-user research tool — no connection storm to guard against).