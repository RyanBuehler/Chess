# v2 Stage 1 — Plan 4d: Win-Value Directed Curriculum

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make goal selection **win-valued** instead of uniform: estimate each cluster's interventional win-value `P(win | do(assign g)) − base` from ε-explore games, and sample sub-goals biased toward win-relevant clusters (`w(g) = β·novelty(g) + γ·win_value(g)`). This is the directed means-end layer (roadmap item D).

**Architecture:** Assignment becomes: with prob `ε` assign a **uniform-random** cluster (the `do(assign g)` intervention, `explore=True`) — these de-confounded games feed the `WinValueEstimator`; otherwise sample from the win-valued `ClusterCurriculum` (`explore=False`). The trainer updates the estimator from each ingested **explore** game (per side: its assigned cluster + whether that side won) and persists it next to the GoalSpace; workers reload it and rebuild the curriculum. LP (learning-progress) is deferred — `novelty + win_value` is the Stage-1 directed curriculum.

**Tech Stack:** Python, NumPy, pytest. Files: `chessrl/goals/winvalue.py` (new), `chessrl/config/config.py`, `chessrl/selfplay/concurrent.py`, `chessrl/selfplay/worker.py`, `chessrl/training/parallel_loop.py`, `experiments/v2-stage1.yaml`.

## Global Constraints

- ε-explore assignment is the ONLY de-confounded source for win-value (curriculum games are confounded). `win_value(g) = E[Beta_g] − base_winrate`, clamped ≥0 in the sampler.
- `_ClusterSideGoal` gains an `explore: bool`; records store the per-side explore flag (the `explore` column already exists — set it to the side-to-move's goal's explore flag).
- Curriculum sampler: `w(g) = novelty_beta·novelty(g) + gamma_winvalue·max(0, win_value(g))`, `novelty(g)=1/sqrt(1+attempts(g))`; win-floor (prob `win_floor` → terminal) retained; falls back to uniform when no win-value data yet.
- Win-value update per ingested explore game, per side: `won = (z>0 if side==White else z<0)`; draws count as not-won. Only sides whose goal was `explore` update the estimator.
- WinValueEstimator persisted to `run_dir/winvalue.json`; workers reload on mtime change (mirror goalspace reload). v1/vanilla unaffected.
- Windows venv tests, unpiped/foreground. Stage only named files; never `git add -A`.

---

### Task 1: `WinValueEstimator` + `ClusterCurriculum` + config

**Files:**
- Create: `chessrl/goals/winvalue.py`
- Modify: `chessrl/config/config.py` (GoalConfig); `experiments/v2-stage1.yaml`
- Test: `tests/test_winvalue.py` (new)

**Interfaces:**
- `WinValueEstimator(prior_a=1.0, prior_b=1.0)`: `update(cluster:int, won:bool)`; `win_value(cluster:int)->float` (=posterior mean − base_winrate, 0.0 if no data); `base_winrate->float`; `attempts(cluster:int)->int`; `to_dict()/from_dict()`; `save(path)/load(path)`.
- `ClusterCurriculum(estimator, n_clusters, novelty_beta, gamma_winvalue, win_floor)`: `sample(rng)->int` (-1 = terminal via win_floor, else a cluster); `record_attempt(cluster)`.
- `GoalConfig`: `epsilon: float = 0.15`, `gamma_winvalue: float = 1.0`, `novelty_beta: float = 1.0` (the last two may already exist — reuse; add only what's missing). Validate `0<=epsilon<=1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_winvalue.py
import numpy as np
from chessrl.goals.winvalue import WinValueEstimator, ClusterCurriculum


def test_winvalue_lift_and_base():
    e = WinValueEstimator()
    for _ in range(40): e.update(0, True)    # cluster 0 wins a lot
    for _ in range(40): e.update(1, False)   # cluster 1 loses a lot
    assert e.win_value(0) > e.win_value(1)
    assert 0.0 <= e.base_winrate <= 1.0
    assert e.win_value(2) == 0.0             # no data -> neutral
    assert e.attempts(0) == 40


def test_curriculum_biases_toward_high_winvalue():
    e = WinValueEstimator()
    for _ in range(60): e.update(3, True)    # cluster 3 high win-value
    for _ in range(60): e.update(5, False)
    cur = ClusterCurriculum(e, n_clusters=8, novelty_beta=0.1, gamma_winvalue=5.0, win_floor=0.0)
    rng = np.random.default_rng(0)
    picks = [cur.sample(rng) for _ in range(400)]
    assert picks.count(3) > picks.count(5)   # biased toward the win-relevant cluster


def test_curriculum_win_floor_returns_terminal():
    cur = ClusterCurriculum(WinValueEstimator(), n_clusters=4, novelty_beta=1.0,
                            gamma_winvalue=1.0, win_floor=1.0)
    assert cur.sample(np.random.default_rng(0)) == -1


def test_save_load_roundtrip(tmp_path):
    e = WinValueEstimator()
    for _ in range(10): e.update(2, True)
    e.save(tmp_path / "wv.json")
    e2 = WinValueEstimator.load(tmp_path / "wv.json")
    assert e2.win_value(2) == e.win_value(2) and e2.attempts(2) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_winvalue.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# chessrl/goals/winvalue.py
"""Interventional per-cluster win-value + win-valued cluster curriculum (Plan 4d).

win_value(g) = E[P(win | do(assign g))] - base_winrate, estimated from a
Beta-Bernoulli posterior fed ONLY by epsilon-explore games (uniform-random
assignment) so the estimate is de-confounded. The curriculum samples sub-goals
biased toward win-relevant clusters: w(g) = beta*novelty(g) + gamma*max(0,win_value(g))."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class WinValueEstimator:
    def __init__(self, prior_a: float = 1.0, prior_b: float = 1.0):
        self.prior_a = prior_a
        self.prior_b = prior_b
        self._wins: dict[int, int] = {}
        self._att: dict[int, int] = {}

    def update(self, cluster: int, won: bool) -> None:
        c = int(cluster)
        self._att[c] = self._att.get(c, 0) + 1
        self._wins[c] = self._wins.get(c, 0) + (1 if won else 0)

    @property
    def base_winrate(self) -> float:
        tot = sum(self._att.values())
        if tot == 0:
            return 0.0
        return sum(self._wins.values()) / tot

    def attempts(self, cluster: int) -> int:
        return self._att.get(int(cluster), 0)

    def win_value(self, cluster: int) -> float:
        c = int(cluster)
        if self._att.get(c, 0) == 0:
            return 0.0
        a = self.prior_a + self._wins.get(c, 0)
        b = self.prior_b + (self._att[c] - self._wins.get(c, 0))
        return a / (a + b) - self.base_winrate

    def to_dict(self) -> dict:
        return {"prior_a": self.prior_a, "prior_b": self.prior_b,
                "wins": self._wins, "att": self._att}

    @classmethod
    def from_dict(cls, d: dict) -> "WinValueEstimator":
        e = cls(d.get("prior_a", 1.0), d.get("prior_b", 1.0))
        e._wins = {int(k): int(v) for k, v in d.get("wins", {}).items()}
        e._att = {int(k): int(v) for k, v in d.get("att", {}).items()}
        return e

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path) -> "WinValueEstimator":
        return cls.from_dict(json.loads(Path(path).read_text()))


class ClusterCurriculum:
    def __init__(self, estimator: WinValueEstimator, n_clusters: int,
                 novelty_beta: float = 1.0, gamma_winvalue: float = 1.0, win_floor: float = 0.2):
        self.est = estimator
        self.n_clusters = int(n_clusters)
        self.novelty_beta = novelty_beta
        self.gamma_winvalue = gamma_winvalue
        self.win_floor = win_floor

    def _novelty(self, c: int) -> float:
        return 1.0 / np.sqrt(1.0 + self.est.attempts(c))

    def _weight(self, c: int) -> float:
        return self.novelty_beta * self._novelty(c) + self.gamma_winvalue * max(0.0, self.est.win_value(c))

    def sample(self, rng: np.random.Generator) -> int:
        if self.n_clusters <= 0 or rng.random() < self.win_floor:
            return -1
        w = np.array([self._weight(c) for c in range(self.n_clusters)], dtype=np.float64)
        tot = w.sum()
        probs = (w / tot) if (tot > 0 and np.isfinite(tot)) else np.full(self.n_clusters, 1.0 / self.n_clusters)
        return int(rng.choice(self.n_clusters, p=probs))

    def record_attempt(self, cluster: int) -> None:
        pass   # attempts tracked by the estimator's update(); placeholder for symmetry
```

In `chessrl/config/config.py` `GoalConfig`, add any missing fields:

```python
    epsilon: float = 0.15           # fraction of assignments that are uniform-random (interventional)
    gamma_winvalue: float = 1.0     # curriculum weight on win_value(g)
    # novelty_beta may already exist (v1 lp); if not, add: novelty_beta: float = 1.0
```

Add `0.0 <= epsilon <= 1.0` validation to its `__post_init__`. In `experiments/v2-stage1.yaml` under `goal:` add `epsilon: 0.15`, `gamma_winvalue: 1.0`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_winvalue.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add chessrl/goals/winvalue.py chessrl/config/config.py experiments/v2-stage1.yaml tests/test_winvalue.py
git commit -m "feat(v2): WinValueEstimator + ClusterCurriculum + config (win-valued goals)"
```

---

### Task 2: ε-explore + curriculum in cluster assignment

**Files:**
- Modify: `chessrl/selfplay/concurrent.py` (`_ClusterSideGoal`, `assign_cluster_goal`, driver threading)
- Test: `tests/test_meansend_selfplay.py` (extend)

**Interfaces:**
- `_ClusterSideGoal` gains `explore: bool = False`.
- `assign_cluster_goal(goalspace, win_vector, goal_cfg, rng, curriculum=None)`: if not ready → terminal. Else: with prob `goal_cfg.epsilon` → **uniform-random** cluster, `explore=True`; elif `curriculum` is not None → `curriculum.sample(rng)` (-1→terminal, else cluster), `explore=False`; else (no curriculum) → uniform cluster, `explore=False` (current behavior). win-floor handled inside the curriculum; when no curriculum, apply `win_floor` as before.
- `play_meansend_games_concurrent(..., curriculum=None)`: thread `curriculum` into `assign_cluster_goal`; the per-ply record's `explore` = the side-to-move's goal `explore` flag.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_meansend_selfplay.py
def test_epsilon_explore_marks_explore_and_uniform():
    rng = np.random.default_rng(0)
    seen_explore = 0
    for _ in range(200):
        g = assign_cluster_goal(ReadyGoalSpace(), WIN_VEC,
                                GoalConfig(goal_mode="emergent", win_floor=0.0, epsilon=1.0),
                                rng)
        if g.explore: seen_explore += 1
    assert seen_explore == 200   # epsilon=1 -> always explore


def test_curriculum_used_when_not_exploring():
    class Cur:
        def sample(self, rng): return 2
    g = assign_cluster_goal(ReadyGoalSpace(), WIN_VEC,
                            GoalConfig(goal_mode="emergent", win_floor=0.0, epsilon=0.0),
                            np.random.default_rng(0), curriculum=Cur())
    assert g.active_cluster == 2 and g.explore is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_meansend_selfplay.py -v`
Expected: FAIL — `assign_cluster_goal` has no `curriculum`/`epsilon`/`explore`.

- [ ] **Step 3: Implement** — update `_ClusterSideGoal` (add `explore: bool = False` field), rewrite `assign_cluster_goal`:

```python
def assign_cluster_goal(goalspace, win_vector, goal_cfg, rng, curriculum=None) -> _ClusterSideGoal:
    win_vector = np.asarray(win_vector, np.float32)
    ready = getattr(goalspace, "ready", False) and getattr(goalspace, "centroids", None) is not None
    if not ready:
        return _ClusterSideGoal(-1, win_vector, goal_cfg.deadline_max, 0, -1, win_vector, explore=False)

    def terminal():
        return _ClusterSideGoal(-1, win_vector, goal_cfg.deadline_max, 0, -1, win_vector, explore=False)

    def subgoal(c, explore):
        vec = np.asarray(goalspace.centroid(c), np.float32)
        return _ClusterSideGoal(c, vec, goal_cfg.goal_window, 0, c, vec, explore=explore)

    eps = getattr(goal_cfg, "epsilon", 0.0)
    if rng.random() < eps:                                  # interventional: uniform-random cluster
        return subgoal(int(rng.integers(goalspace.n_clusters)), explore=True)
    if curriculum is not None:                              # win-valued curriculum
        c = curriculum.sample(rng)
        return terminal() if c < 0 else subgoal(c, explore=False)
    # no curriculum (e.g. early / tests): win-floor then uniform
    if rng.random() < goal_cfg.win_floor:
        return terminal()
    return subgoal(int(rng.integers(goalspace.n_clusters)), explore=False)
```

In `play_meansend_games_concurrent`, add a `curriculum=None` param and pass it to both `assign_cluster_goal(...)` calls. In `_play_one_meansend_move`, set the record's `explore` from the side: change the `builder.add(..., explore=g.explore)` to `explore=side.explore` (the side-to-move's goal explore flag).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_meansend_selfplay.py -v`
Expected: PASS. Confirm existing means-end self-play tests still pass.

- [ ] **Step 5: Commit**

```bash
git add chessrl/selfplay/concurrent.py tests/test_meansend_selfplay.py
git commit -m "feat(v2): epsilon-explore + win-valued curriculum in cluster assignment"
```

---

### Task 3: Win-value update + persistence wiring (loop + worker)

**Files:**
- Modify: `chessrl/training/parallel_loop.py` (update estimator from explore games; persist); `chessrl/selfplay/worker.py` (load estimator → ClusterCurriculum → pass to driver)
- Test: `tests/test_winvalue_wiring.py` (new — a unit test of the per-game update helper)

**Interfaces:**
- `parallel_loop.update_winvalue_from_record(estimator, rec)`: for each side (White=even plies, Black=odd plies), if that side's goal was `explore` and assigned a cluster ≥0, `estimator.update(assigned_cluster, won)` where `won = (z>0)` for White / `(z<0)` for Black (z = White-frame outcome = `rec.outcomes[0]` mapped... use the record's stored outcome: White-frame z is `outcomes[i]` when side-to-move is White at ply i, so derive z_white once).
- Emergent loop: build `WinValueEstimator` (load `run_dir/winvalue.json` on resume), update it per ingested explore game, save it each cycle that added games, and build a `ClusterCurriculum(estimator, goalspace.n_clusters, ...)` passed to nothing in the trainer (workers use it) — the trainer just maintains+persists the estimator.
- `worker.py`: load `winvalue.json` (mtime-reload like goalspace) → `ClusterCurriculum(est, goalspace.n_clusters, cfg.goal.novelty_beta, cfg.goal.gamma_winvalue, cfg.goal.win_floor)`; pass `curriculum=` to `play_meansend_games_concurrent`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_winvalue_wiring.py
import numpy as np, chess
from chessrl.selfplay.records import RecordBuilder
from chessrl.chess_env.moves import move_to_index
from chessrl.goals.winvalue import WinValueEstimator
from chessrl.training.parallel_loop import update_winvalue_from_record


def _explore_game(white_cluster=2, black_cluster=5, z_white=1, n=4):
    b = RecordBuilder(); board = chess.Board()
    wv = np.zeros(4, np.float32)
    for ply in range(n):
        mv = list(board.legal_moves)[0]; idx = move_to_index(mv, board.turn == chess.BLACK)
        c = white_cluster if board.turn == chess.WHITE else black_cluster
        b.add(board, [idx], [1], idx, protagonist=board.turn,
              cluster_active=c, cluster_assigned=c, active_vec=wv, explore=True)
        board.push(mv)
    return b.finalize(z_white=z_white)


def test_update_winvalue_credits_winner_side():
    est = WinValueEstimator()
    update_winvalue_from_record(est, _explore_game(white_cluster=2, black_cluster=5, z_white=1))
    # White won (z=1): White's cluster 2 gets a win; Black's cluster 5 gets a loss
    assert est.attempts(2) == 1 and est.attempts(5) == 1
    assert est.win_value(2) > est.win_value(5)


def test_update_skips_non_explore():
    est = WinValueEstimator()
    b = RecordBuilder(); board = chess.Board()
    mv = list(board.legal_moves)[0]; idx = move_to_index(mv, False)
    b.add(board, [idx], [1], idx, protagonist=chess.WHITE, cluster_active=1, cluster_assigned=1,
          active_vec=np.zeros(4, np.float32), explore=False)
    update_winvalue_from_record(est, b.finalize(z_white=1))
    assert est.attempts(1) == 0   # non-explore -> no update
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_winvalue_wiring.py -v`
Expected: FAIL — `update_winvalue_from_record` missing.

- [ ] **Step 3: Implement** — add to `chessrl/training/parallel_loop.py`:

```python
WINVALUE_FILE = "winvalue.json"


def update_winvalue_from_record(estimator, rec) -> None:
    """Update the interventional win-value estimator from one EXPLORE game: for
    each side whose game-assigned goal was epsilon-explore (and a real cluster),
    credit a win/loss by that side's outcome. z_white = the White-frame result."""
    if not rec.has_cluster_goals():
        return
    import chess
    # White-frame z: outcomes[i] is from side-to-move at ply i; convert.
    z_white = None
    for i in range(len(rec)):
        white_to_move = (rec.protagonist[i] == 1)
        z_white = int(rec.outcomes[i]) if white_to_move else -int(rec.outcomes[i])
        break
    if z_white is None:
        return
    # Per side: find its assigned cluster + explore flag from a ply where it moved.
    for color, won in ((chess.WHITE, z_white > 0), (chess.BLACK, z_white < 0)):
        want = 1 if color == chess.WHITE else 0
        for i in range(len(rec)):
            if int(rec.protagonist[i]) == want:
                if bool(rec.explore[i]) and int(rec.assigned_cluster[i]) >= 0:
                    estimator.update(int(rec.assigned_cluster[i]), won)
                break
```

In `main`'s emergent setup: `winvalue = WinValueEstimator.load(run_dir/WINVALUE_FILE) if (run_dir/WINVALUE_FILE).exists() else WinValueEstimator()`. In the ingest `on_record` callback, ALSO call `update_winvalue_from_record(winvalue, rec)`. After a cycle that added games, `winvalue.save(run_dir/WINVALUE_FILE)`. (The trainer maintains + persists; workers consume.)

In `chessrl/selfplay/worker.py`: add a loader that, when a goalspace is present, also loads `winvalue.json` and builds `ClusterCurriculum(WinValueEstimator.load(...), goalspace.n_clusters, cfg.goal.novelty_beta, cfg.goal.gamma_winvalue, cfg.goal.win_floor)`; pass `curriculum=` into `play_meansend_games_concurrent`. Reload on `winvalue.json` mtime change (mirror the goalspace reload). When absent, pass `curriculum=None` (uniform fallback).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_winvalue_wiring.py -v` then regression `.venv\Scripts\python.exe -m pytest tests/test_worker_emergent.py tests/test_meansend_selfplay.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add chessrl/training/parallel_loop.py chessrl/selfplay/worker.py tests/test_winvalue_wiring.py
git commit -m "feat(v2): wire interventional win-value updates + curriculum into loop/worker"
```

---

## Plan 4d deliverable

Goals are now **win-valued**: ε of assignments are interventional (uniform, de-confounding), the estimator learns each cluster's `P(win|do g)` online, and the curriculum biases sub-goal selection toward win-relevant clusters. The first real run trains the full Stage-1 design.

## Out of scope

- LP (learning-progress) curriculum term (novelty+win_value is the Stage-1 directed curriculum).
- Live cluster display (separate follow-on — next).
- α-sweep harness.
