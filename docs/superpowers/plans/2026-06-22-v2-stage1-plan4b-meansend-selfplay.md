# v2 Stage 1 — Plan 4b: Means-End Concurrent Self-Play

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a concurrent self-play driver that plays cluster-goal games with the Plan 4a means-end MCTS: assign a discovered cluster goal per side (or terminal), pursue it under the vector-conditioned blended search, switch to terminal pursuit when the goal window elapses, and write the Plan 3 cluster records (`active_vec`/`cluster`/`explore`). Also green the pre-existing live-feed aux test.

**Architecture:** Mirrors `play_goal_games_concurrent` but: (1) goals are cluster centroids from a `GoalSpace`, not `GoalTemplate`s; (2) the per-ply tree is built with `init_tree_for_meansend(board, goal_vec, deadline)` where `goal_vec` is the active centroid (sub-goal) or the net's reserved `win_vector` (terminal pursuit); (3) the switch to terminal is **deadline-based** (after the goal window) — no live achievement check; (4) records carry the cluster columns. Resignation fires only during terminal pursuit (same root-Q gate as vanilla). The v1 goal self-play (`play_goal_games_concurrent`) is left intact.

**Tech Stack:** Python, NumPy, python-chess, pytest. File: `chessrl/selfplay/concurrent.py`; uses `chessrl/goals/goalspace.py`, `chessrl/mcts/batched.py`, `chessrl/selfplay/records.py`.

## Global Constraints

- Goal vector fed to the search: active centroid `goalspace.centroid(c)` for a sub-goal, or `win_vector` (passed in — the net's reserved terminal-pursuit code) for terminal pursuit. Deadline = remaining plies to the goal window (sub-goal) or `ply_cap - ply` (terminal).
- Switch to terminal pursuit when `ply - start_ply >= goal_window` (deadline-based; no `GoalSpace.achieved` call in self-play). Achievement is recomputed at HER train time (Plan 3).
- Records (per ply): `cluster_active` (id or -1=terminal), `cluster_assigned` (game-assigned id or -1), `active_vec` (the goal vector used this ply), `explore` (per-game ε flag). Do NOT set the v1 `assigned_goal`/`active_goal` (GoalTemplate) columns — these are cluster records (`has_cluster_goals()` True, `has_goals()` False).
- Resignation: only while active is terminal; `root_q < sp_cfg.resign_threshold` with the consecutive-streak rule (mirror v1).
- The v1 `play_goal_games_concurrent` and `_GoalGame` stay UNCHANGED.
- Windows venv tests, unpiped/foreground. Stage only named files; never `git add -A`.

---

### Task 1: Green the pre-existing live-feed aux test

**Files:**
- Modify: `tests/test_batched_goal_concurrent.py` (the `test_concurrent_goal_emits_live_feed_frames` assertion)
- (Commit the in-progress `chessrl/selfplay/concurrent.py` both-sides-aux change it depends on)

**Context:** `concurrent.py` already emits `aux` as a both-sides table dict `{cols, to_move, rows}` (the live-feed work in the working tree), but `test_concurrent_goal_emits_live_feed_frames` still asserts the OLD list-of-`[label,value]` format, so it fails. Reconcile the test to the table format and commit the aux change.

- [ ] **Step 1: Run the failing test to see the current assertion fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_batched_goal_concurrent.py::test_concurrent_goal_emits_live_feed_frames -v`
Expected: FAIL (asserts the old aux shape).

- [ ] **Step 2: Read the test and update its aux assertions to the table schema**

Open `tests/test_batched_goal_concurrent.py`, find `test_concurrent_goal_emits_live_feed_frames`. The published frame's `aux` is now a dict `{"cols": ["White","Black"], "to_move": 0|1, "rows": [["goal", w, b], ["phase", w, b], ["P(achieve)", w, b]]}` (see `_goal_aux` in `concurrent.py`). Replace the old-format assertions with assertions on this structure: that `aux` is a dict with keys `cols`/`to_move`/`rows`, `cols == ["White","Black"]`, `to_move in (0,1)`, and that `rows` contains a `"goal"` row and a `"P(achieve)"` row. Keep the rest of the test (frame count, fen/last_move/ply presence) unchanged.

- [ ] **Step 3: Run the test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_batched_goal_concurrent.py -v`
Expected: PASS (all, incl. the reconciled live-feed test).

- [ ] **Step 4: Commit (includes the in-progress concurrent.py aux change)**

```bash
git add chessrl/selfplay/concurrent.py tests/test_batched_goal_concurrent.py
git commit -m "fix(v2): reconcile live-feed aux test to both-sides table schema"
```

---

### Task 2: Cluster side-goal + assigner + deadline switch

**Files:**
- Modify: `chessrl/selfplay/concurrent.py` (add the means-end goal helpers, near the existing goal section)
- Test: `tests/test_meansend_selfplay.py` (new)

**Interfaces:**
- `@dataclass class _ClusterSideGoal` with: `assigned_cluster:int` (-1=terminal), `assigned_vec:np.ndarray`, `deadline:int`, `start_ply:int`, `active_cluster:int`, `active_vec:np.ndarray`. Method `is_terminal()` → `self.active_cluster < 0`.
- `assign_cluster_goal(goalspace, win_vector, goal_cfg, rng) -> _ClusterSideGoal`: with prob `goal_cfg.win_floor` → terminal (cluster -1, vec=`win_vector`, deadline=`deadline_max`); else a uniform-random ready cluster (vec=centroid, deadline=`goal_window`). If `goalspace` is not ready (centroids None / not ready), always terminal.
- `maybe_switch_cluster_to_terminal(side, ply, win_vector, deadline_max)`: if not terminal and `ply - side.start_ply >= side.deadline`, set `active_cluster=-1`, `active_vec=win_vector`, `deadline=deadline_max` (start counting terminal deadline from here is unnecessary — terminal deadline is the ply cap). Idempotent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_meansend_selfplay.py
import numpy as np
from chessrl.config.config import GoalConfig
from chessrl.selfplay.concurrent import (
    _ClusterSideGoal, assign_cluster_goal, maybe_switch_cluster_to_terminal,
)


class ReadyGoalSpace:
    centroids = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    ready = True
    def centroid(self, c):
        return self.centroids[c].copy()
    @property
    def n_clusters(self):
        return 3


class UnfitGoalSpace:
    centroids = None
    ready = False
    n_clusters = 0


WIN_VEC = np.full(4, -1.0, np.float32)


def test_assign_terminal_when_unfit():
    g = assign_cluster_goal(UnfitGoalSpace(), WIN_VEC, GoalConfig(goal_mode="emergent", win_floor=0.0),
                            np.random.default_rng(0))
    assert g.is_terminal()
    assert np.allclose(g.active_vec, WIN_VEC)


def test_assign_subgoal_when_ready():
    # win_floor=0 forces a sub-goal when ready
    g = assign_cluster_goal(ReadyGoalSpace(), WIN_VEC, GoalConfig(goal_mode="emergent", win_floor=0.0),
                            np.random.default_rng(1))
    assert not g.is_terminal()
    assert 0 <= g.active_cluster < 3
    assert np.allclose(g.active_vec, ReadyGoalSpace().centroid(g.active_cluster))


def test_win_floor_forces_terminal():
    g = assign_cluster_goal(ReadyGoalSpace(), WIN_VEC, GoalConfig(goal_mode="emergent", win_floor=1.0),
                            np.random.default_rng(2))
    assert g.is_terminal()


def test_deadline_switch_to_terminal():
    g = _ClusterSideGoal(assigned_cluster=1, assigned_vec=np.zeros(4, np.float32), deadline=3,
                         start_ply=0, active_cluster=1, active_vec=np.zeros(4, np.float32))
    maybe_switch_cluster_to_terminal(g, ply=2, win_vector=WIN_VEC, deadline_max=60)
    assert not g.is_terminal()          # 2 < 3, no switch yet
    maybe_switch_cluster_to_terminal(g, ply=3, win_vector=WIN_VEC, deadline_max=60)
    assert g.is_terminal()              # 3 >= 3, switched
    assert np.allclose(g.active_vec, WIN_VEC)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_meansend_selfplay.py -v`
Expected: FAIL — names not importable.

- [ ] **Step 3: Implement** (add to `chessrl/selfplay/concurrent.py`, after the imports; import `dataclass`)

```python
from dataclasses import dataclass


@dataclass
class _ClusterSideGoal:
    """One side's cluster goal for a means-end game. ``active_cluster < 0`` means
    the side is pursuing the terminal/extrinsic objective (vec = the net's
    win_vector)."""
    assigned_cluster: int
    assigned_vec: np.ndarray
    deadline: int
    start_ply: int
    active_cluster: int
    active_vec: np.ndarray

    def is_terminal(self) -> bool:
        return self.active_cluster < 0


def assign_cluster_goal(goalspace, win_vector, goal_cfg, rng) -> _ClusterSideGoal:
    """Pick one side's goal: with prob win_floor (or always, if the goal space
    isn't ready) the terminal objective; else a uniform-random discovered
    cluster. Stage 4c replaces the uniform draw with the LP+win-value curriculum
    and the epsilon-explore branch."""
    win_vector = np.asarray(win_vector, np.float32)
    ready = getattr(goalspace, "ready", False) and getattr(goalspace, "centroids", None) is not None
    if not ready or rng.random() < goal_cfg.win_floor:
        return _ClusterSideGoal(-1, win_vector, goal_cfg.deadline_max, 0, -1, win_vector)
    c = int(rng.integers(goalspace.n_clusters))
    vec = np.asarray(goalspace.centroid(c), np.float32)
    return _ClusterSideGoal(c, vec, goal_cfg.goal_window, 0, c, vec)


def maybe_switch_cluster_to_terminal(side: _ClusterSideGoal, ply: int, win_vector, deadline_max: int) -> None:
    """Deadline-based switch: once the goal window elapses, pursue the terminal
    objective for the rest of the game (idempotent)."""
    if side.is_terminal():
        return
    if ply - side.start_ply >= side.deadline:
        side.active_cluster = -1
        side.active_vec = np.asarray(win_vector, np.float32)
        side.deadline = deadline_max
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_meansend_selfplay.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add chessrl/selfplay/concurrent.py tests/test_meansend_selfplay.py
git commit -m "feat(v2): cluster side-goal + assigner + deadline switch"
```

---

### Task 3: `play_meansend_games_concurrent` driver

**Files:**
- Modify: `chessrl/selfplay/concurrent.py` (add the driver + `_MeansEndGame`)
- Test: `tests/test_meansend_selfplay.py` (extend)

**Interfaces:**
- `play_meansend_games_concurrent(evaluator_vector, mcts_cfg, sp_cfg, goal_cfg, goalspace, win_vector, rng, num_games, explore=False, publisher=None, game_id_prefix="") -> list[(GameRecord, final_board, z, meta)]`.
  - One `BatchedMCTS(evaluator_vector, mcts_cfg, rng, meansend=True)`.
  - Per game: a `_ClusterSideGoal` per side via `assign_cluster_goal`. Per ply (protagonist = side to move): `tree = mcts.init_tree_for_meansend(board, side.active_vec, remaining, add_noise=True)` where `remaining = side.deadline - (ply - side.start_ply)` for a sub-goal, or `sp_cfg.ply_cap - ply` for terminal (min 1). Run to sims, pick (temp/argmax), record with cluster columns, resign-gate only if terminal, push, `maybe_switch_cluster_to_terminal`, publish.
  - `meta` includes `win_ply_fraction` (fraction of plies with `cluster_active == -1`).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_meansend_selfplay.py
import chess
from chessrl.config.config import MCTSConfig, SelfPlayConfig
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.selfplay.concurrent import play_meansend_games_concurrent


class FakeVectorEval:
    """Dual-head batched vector evaluator. evaluate_planes(planes, goal_vecs,
    deadlines) -> (policies uniform, v_win, v_goal)."""
    def evaluate_planes(self, planes, goal_vecs, deadlines):
        n = planes.shape[0]
        pol = np.full((n, NUM_ACTIONS), 1.0 / NUM_ACTIONS, np.float32)
        return pol, np.zeros(n, np.float32), np.full(n, 0.5, np.float32)


def test_meansend_selfplay_produces_cluster_records():
    gs = ReadyGoalSpace()
    recs = play_meansend_games_concurrent(
        FakeVectorEval(), MCTSConfig(simulations=8, leaves_per_tree=1),
        SelfPlayConfig(ply_cap=6, resign_playout_fraction=1.0),
        GoalConfig(goal_mode="emergent", win_floor=0.0, goal_window=2, deadline_max=60),
        gs, np.full(4, -1.0, np.float32), np.random.default_rng(0), num_games=2,
    )
    assert len(recs) == 2
    for rec, board, z, meta in recs:
        assert rec.has_cluster_goals()
        assert rec.active_vec.shape[1] == 4
        assert "win_ply_fraction" in meta
        # deadline switch (goal_window=2) means later plies are terminal (-1)
        assert (rec.active_cluster == -1).any()


def test_meansend_selfplay_unfit_is_all_terminal():
    recs = play_meansend_games_concurrent(
        FakeVectorEval(), MCTSConfig(simulations=8, leaves_per_tree=1),
        SelfPlayConfig(ply_cap=4, resign_playout_fraction=1.0),
        GoalConfig(goal_mode="emergent", win_floor=0.0, goal_window=2, deadline_max=60),
        UnfitGoalSpace(), np.full(4, -1.0, np.float32), np.random.default_rng(0), num_games=1,
    )
    rec = recs[0][0]
    assert rec.has_cluster_goals()
    assert (rec.active_cluster == -1).all()   # unfit -> always terminal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_meansend_selfplay.py -v`
Expected: FAIL — `play_meansend_games_concurrent` missing.

- [ ] **Step 3: Implement** (add to `chessrl/selfplay/concurrent.py`)

```python
class _MeansEndGame:
    __slots__ = ("builder", "board", "sides", "explore", "allow_resign",
                 "resign_streak", "ply", "done", "z", "tree", "game_id", "last_pv")

    def __init__(self, board, sides, explore, allow_resign):
        self.builder = RecordBuilder()
        self.board = board
        self.sides = sides
        self.explore = explore
        self.allow_resign = allow_resign
        self.resign_streak = {chess.WHITE: 0, chess.BLACK: 0}
        self.ply = 0
        self.done = False
        self.z = 0
        self.tree = None
        self.game_id = ""
        self.last_pv = {chess.WHITE: None, chess.BLACK: None}


def play_meansend_games_concurrent(
    evaluator_vector, mcts_cfg, sp_cfg, goal_cfg, goalspace, win_vector, rng,
    num_games, explore: bool = False, publisher=None, game_id_prefix: str = "",
) -> list:
    """Means-end concurrent self-play (v2). Each side pursues a discovered cluster
    goal (or the terminal objective) under the Plan 4a means-end MCTS; the switch
    to terminal pursuit is deadline-based. Writes Plan 3 cluster records. Returns
    list[(GameRecord, final_board, z, meta)] in slot order."""
    publisher = publisher or NullPublisher()
    win_vector = np.asarray(win_vector, np.float32)
    mcts = BatchedMCTS(evaluator_vector, mcts_cfg, rng, meansend=True)

    games: list[_MeansEndGame] = []
    for slot in range(num_games):
        board = chess.Board()
        allow_resign = rng.random() >= sp_cfg.resign_playout_fraction
        sides = {
            chess.WHITE: assign_cluster_goal(goalspace, win_vector, goal_cfg, rng),
            chess.BLACK: assign_cluster_goal(goalspace, win_vector, goal_cfg, rng),
        }
        g = _MeansEndGame(board, sides, explore, allow_resign)
        g.game_id = f"{game_id_prefix}{slot}"
        games.append(g)

    for g in games:
        _goal_check_pre_move_termination(g, sp_cfg)

    while any(not g.done for g in games):
        active = [g for g in games if not g.done]
        for g in active:
            side = g.sides[g.board.turn]
            if side.is_terminal():
                remaining = max(1, sp_cfg.ply_cap - g.ply)
            else:
                remaining = max(1, side.deadline - (g.ply - side.start_ply))
            g.tree = mcts.init_tree_for_meansend(g.board, side.active_vec, remaining, add_noise=True)
        trees = [g.tree for g in active]
        while any(t.root.visit_count < mcts_cfg.simulations + 1 for t in trees):
            mcts.step_round(trees)
        for g in active:
            _play_one_meansend_move(g, mcts, mcts_cfg, sp_cfg, goal_cfg, win_vector, rng, publisher)

    results = []
    for g in games:
        rec = g.builder.finalize(g.z)
        wpf = float((rec.active_cluster == -1).mean()) if rec.has_cluster_goals() and len(rec) else 0.0
        meta = {"plies": len(rec), "z": g.z, "resigned": False,
                "playout": not g.allow_resign, "would_resign": False, "fp": False,
                "win_ply_fraction": wpf}
        results.append((rec, g.board, g.z, meta))
    return results


def _play_one_meansend_move(g, mcts, mcts_cfg, sp_cfg, goal_cfg, win_vector, rng, publisher) -> None:
    protagonist = g.board.turn
    side = g.sides[protagonist]
    visits = mcts.visit_counts(g.tree)
    root_q = mcts.root_q(g.tree)
    g.last_pv[protagonist] = float(root_q)
    idxs = np.fromiter(visits.keys(), dtype=np.int64)
    counts = np.fromiter(visits.values(), dtype=np.float64)
    if g.ply < mcts_cfg.temperature_moves:
        choice = int(rng.choice(idxs, p=counts / counts.sum()))
    else:
        choice = int(idxs[counts.argmax()])

    g.builder.add(
        g.board, idxs.astype(np.int32), counts.astype(np.int32), choice,
        protagonist=protagonist,
        cluster_active=side.active_cluster, cluster_assigned=side.assigned_cluster,
        active_vec=side.active_vec, explore=g.explore,
    )

    flip = g.board.turn == chess.BLACK
    chosen_move = index_to_move(choice, flip, g.board)
    total = float(counts.sum())
    order = np.argsort(counts)[::-1][:5]
    top_moves = [[index_to_move(int(idxs[k]), flip, g.board).uci(), float(counts[k] / total)] for k in order]

    if side.is_terminal():
        if root_q < sp_cfg.resign_threshold:
            g.resign_streak[protagonist] += 1
            if g.allow_resign and g.resign_streak[protagonist] >= sp_cfg.resign_consecutive:
                g.z = -1 if protagonist == chess.WHITE else 1
                g.done = True
                _publish_move(publisher, g, chosen_move, root_q, top_moves)
                return
        else:
            g.resign_streak[protagonist] = 0
    else:
        g.resign_streak[protagonist] = 0

    g.board.push(index_to_move(choice, protagonist == chess.BLACK, g.board))
    g.ply += 1
    maybe_switch_cluster_to_terminal(side, g.ply, win_vector, goal_cfg.deadline_max)
    _goal_check_pre_move_termination(g, sp_cfg)
    _publish_move(publisher, g, chosen_move, root_q, top_moves)
```

(`_goal_check_pre_move_termination` and `_publish_move` already exist and work on any object with the right attributes — `_MeansEndGame` provides `board`/`ply`/`z`/`done`/`game_id`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_meansend_selfplay.py -v`
Expected: PASS (6 passed). Regression: `.venv\Scripts\python.exe -m pytest tests/test_batched_goal_concurrent.py -v` — PASS.

- [ ] **Step 5: Commit**

```bash
git add chessrl/selfplay/concurrent.py tests/test_meansend_selfplay.py
git commit -m "feat(v2): play_meansend_games_concurrent driver (cluster goals, means-end MCTS)"
```

---

## Plan 4b deliverable

A concurrent means-end self-play driver that generates cluster-goal games using the Plan 4a search, writes Plan 3 cluster records, and publishes live frames — plus a green test suite. v1 goal self-play untouched.

## Out of scope (4c)

The training loop assembly: frozen-encoder snapshot + GoalSpace refresh + `VectorGoalReplayBuffer` + `winvalue.py` (interventional win-value) + curriculum `γ·win_value` + ε-explore in `assign_cluster_goal` + `train_steps_vector` dual-head loss + `parallel_loop` `goal_mode=="emergent"` branch + the I1 buffer-on-refit handling + `experiments/v2-stage1.yaml`. The worker (`worker.py`) must call `play_meansend_games_concurrent` for emergent mode.
