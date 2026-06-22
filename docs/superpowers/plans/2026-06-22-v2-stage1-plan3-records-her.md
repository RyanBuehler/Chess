# v2 Stage 1 — Plan 3: Cluster-Goal Records Contract + HER

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Extend `GameRecord` to store v2 cluster-goal data (assigned/active cluster id + the goal centroid vector + explore flag, per ply), then build the cluster-goal HER + replay buffer that generates dual-head training samples for the vector net (`v_win` outcome target + `v_goal` achievement target). Tested in isolation with synthetic records + a fake embedder.

**Architecture:** v1's HER relabels goals via board predicates over a fixed vocabulary. v2 relabels via the **frozen-encoder embedding delta → nearest cluster** (`GoalSpace`). Each ply yields: one **active** sample (the searched cluster goal — policy target + outcome target + achievement label), plus HER **future positives** (clusters actually reached later) and **negatives** (clusters not reached). Because the net now has a **dual head**, every sample carries both a `v_win` target (the game outcome `z`, used only on the active sample) and a `v_goal` target (achievement ∈ {0,1}). The goal-conditioning input is the stored **centroid vector** (records store ids + vectors, so vectors are exact even after a re-fit).

**Tech Stack:** Python, NumPy, python-chess, pytest. Files in `chessrl/selfplay/records.py`, `chessrl/training/`.

## Global Constraints

- `d` (centroid/goal-vector dim) = `NetworkConfig.filters`. Records store `active_vec (T,d) float32`.
- v2 cluster-goal columns are OPTIONAL and additive: vanilla and v1 goal records are byte-for-byte unchanged; `has_cluster_goals()` reports presence. Do not touch `_FIELDS` or `_GOAL_FIELDS` (v1).
- Win goal sentinel: `active_cluster = -1` and `active_vec = win_vector` denote WIN-pursuit (not a cluster).
- Dual-head targets: `v_win` = game outcome `z` in {-1,0,1} (side-to-move frame), present only on the active sample (mask elsewhere). `v_goal` = achievement label in {0,1} for the sample's cluster goal.
- Achievement (cluster `c`, from ply `i`, horizon `rem`): True iff ∃ `t ∈ (i, min(i+rem, T)]` with `GoalSpace.assign(e(s_t) − e(s_i)) == c` AND `‖·−centroid_c‖ ≤ τ`. Uses the FROZEN embedder.
- Determinism: HER sampling uses an explicit `rng`.
- Windows venv tests, unpiped/foreground: `.venv\Scripts\python.exe -m pytest <path> -v`. Stage only named files; never `git add -A`.

---

### Task 1: Records — v2 cluster-goal columns

**Files:**
- Modify: `chessrl/selfplay/records.py` (GameRecord fields + save/load; RecordBuilder.add/finalize)
- Test: `tests/test_records_cluster_goals.py` (new)

**Interfaces:**
- `GameRecord` new optional columns: `assigned_cluster (T,) int32`, `active_cluster (T,) int32`, `active_vec (T,d) float32`, `explore (T,) int8`. New `has_cluster_goals() -> bool` (True iff `active_vec is not None`).
- `RecordBuilder.add(..., cluster_active=None, cluster_assigned=None, active_vec=None, explore=None)`; `finalize` emits the columns when any cluster data was added.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_records_cluster_goals.py
import numpy as np
import chess
from chessrl.selfplay.records import GameRecord, RecordBuilder


def _tiny_game(d=4):
    b = RecordBuilder()
    board = chess.Board()
    for ply in range(3):
        b.add(board, [0, 1], [3, 1], played_index=0,
              protagonist=board.turn,
              cluster_active=(ply % 2), cluster_assigned=1,
              active_vec=np.full(d, float(ply), np.float32), explore=(ply == 0))
        board.push(list(board.legal_moves)[0])
    return b.finalize(z_white=1)


def test_cluster_columns_present_and_shaped():
    rec = _tiny_game(d=4)
    assert rec.has_cluster_goals()
    assert rec.active_cluster.tolist() == [0, 1, 0]
    assert rec.assigned_cluster.tolist() == [1, 1, 1]
    assert rec.active_vec.shape == (3, 4)
    assert rec.explore.tolist() == [1, 0, 0]


def test_save_load_roundtrip(tmp_path):
    rec = _tiny_game(d=4)
    p = tmp_path / "g.npz"
    rec.save(p)
    rl = GameRecord.load(p)
    assert rl.has_cluster_goals()
    assert np.array_equal(rl.active_cluster, rec.active_cluster)
    assert np.allclose(rl.active_vec, rec.active_vec)
    assert np.array_equal(rl.explore, rec.explore)


def test_vanilla_record_unaffected():
    b = RecordBuilder()
    board = chess.Board()
    b.add(board, [0], [1], played_index=0)
    rec = b.finalize(z_white=0)
    assert rec.has_cluster_goals() is False
    assert rec.has_goals() is False
    assert rec.active_vec is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_records_cluster_goals.py -v`
Expected: FAIL — `add()` rejects the new kwargs / `has_cluster_goals` missing.

- [ ] **Step 3: Implement**

In `chessrl/selfplay/records.py`:

Add after the existing `_GOAL_FIELDS` line:

```python
_CLUSTER_FIELDS = ("assigned_cluster", "active_cluster", "active_vec", "explore")
```

Add fields to the `GameRecord` dataclass (after the v1 goal columns):

```python
    assigned_cluster: np.ndarray | None = None  # (T,) int32 cluster id (-1 = win)
    active_cluster: np.ndarray | None = None     # (T,) int32 cluster id (-1 = win)
    active_vec: np.ndarray | None = None         # (T, d) float32 goal centroid
    explore: np.ndarray | None = None            # (T,) int8 (1 = epsilon-explore game)
```

Add the predicate (next to `has_goals`):

```python
    def has_cluster_goals(self) -> bool:
        return self.active_vec is not None
```

Extend `save` (after the v1 goal block):

```python
        if self.has_cluster_goals():
            for f in _CLUSTER_FIELDS:
                out[f] = getattr(self, f)
```

Extend `load` (after the v1 goal block, still inside the `with`):

```python
            if "active_vec" in z.files:
                for f in _CLUSTER_FIELDS:
                    kw[f] = z[f]
```

In `RecordBuilder.__init__` add parallel lists:

```python
        self._cl_assigned: list[int] = []
        self._cl_active: list[int] = []
        self._cl_vec: list[np.ndarray] = []
        self._cl_explore: list[int] = []
        self._has_clusters = False
```

In `RecordBuilder.add`, add keyword params `cluster_active=None, cluster_assigned=None, active_vec=None, explore=None` to the signature, and after the existing goal block:

```python
        if active_vec is not None:
            self._has_clusters = True
            self._cl_assigned.append(int(cluster_assigned) if cluster_assigned is not None else -1)
            self._cl_active.append(int(cluster_active) if cluster_active is not None else -1)
            self._cl_vec.append(np.asarray(active_vec, dtype=np.float32))
            self._cl_explore.append(1 if explore else 0)
```

In `RecordBuilder.finalize`, before `return rec`, add:

```python
        if self._has_clusters:
            rec.assigned_cluster = np.array(self._cl_assigned, dtype=np.int32)
            rec.active_cluster = np.array(self._cl_active, dtype=np.int32)
            rec.active_vec = np.stack(self._cl_vec).astype(np.float32)
            rec.explore = np.array(self._cl_explore, dtype=np.int8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_records_cluster_goals.py -v`
Expected: PASS (3 passed). Regression: `.venv\Scripts\python.exe -m pytest tests/test_selfplay_goals.py -v` — PASS.

- [ ] **Step 5: Commit**

```bash
git add chessrl/selfplay/records.py tests/test_records_cluster_goals.py
git commit -m "feat(v2): cluster-goal record columns (ids + centroid vectors + explore)"
```

---

### Task 2: Cluster-goal HER sample generation

**Files:**
- Create: `chessrl/training/cluster_her.py`
- Test: `tests/test_cluster_her.py` (new)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) ClusterGoalSample` with: `ply:int`, `goal_vec:np.ndarray`, `cluster:int`, `remaining:int`, `v_win:float`, `v_win_mask:float`, `v_goal:float`, `v_goal_weight:float`.
  - `cluster_goal_samples(rec, states, embedder, goalspace, rng, weights=None, deadline_max=60) -> list[ClusterGoalSample]`. `states` = `reconstruct_states(rec)` (reused from `her.py`). For each ply: an **active** sample (goal_vec = `rec.active_vec[i]`, cluster = `rec.active_cluster[i]`, `v_win = outcome`, `v_win_mask = 1`, `v_goal` = achievement of the active cluster, weight = `search_laundered`); plus **future positives** (clusters reached within the window, `v_goal=1`, mask 0) and **negatives** (clusters not reached, `v_goal=0`, mask 0). Reuse `HERWeights` from `her.py`.
  - Helper `_delta(embedder, states, i, t) -> np.ndarray`: `embedder.embed_boards([states[i], states[t]])` → `e[1]-e[0]`.
  - Helper `_achieved_cluster(embedder, goalspace, states, i, cluster, rem) -> bool`: per the achievement constraint.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cluster_her.py
import numpy as np
import chess
from chessrl.selfplay.records import RecordBuilder
from chessrl.training.her import reconstruct_states
from chessrl.training.cluster_her import cluster_goal_samples


class FakeEmbedder:
    """e(board) = [fullmove, n_pieces, 0, 0]; deltas grow monotonically."""
    def embed_boards(self, boards):
        out = [[float(b.fullmove_number), float(sum(1 for _ in b.piece_map())), 0.0, 0.0] for b in boards]
        return np.asarray(out, dtype=np.float32)


class FakeGoalSpace:
    """3 clusters along axis 0; assign by rounding delta[0] to {0,1,2}; tau large."""
    tau = 100.0
    def assign(self, delta):
        return int(min(2, max(0, round(float(delta[0])))))
    centroids = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]], np.float32)
    def achieved(self, delta, cluster):
        return self.assign(delta) == cluster and float(np.linalg.norm(delta - self.centroids[cluster])) <= self.tau


def _game(d=4, n=4):
    b = RecordBuilder(); board = chess.Board()
    for ply in range(n):
        b.add(board, [0, 1], [3, 1], played_index=0, protagonist=board.turn,
              cluster_active=1, cluster_assigned=1,
              active_vec=np.array([1, 0, 0, 0], np.float32), explore=False)
        board.push(list(board.legal_moves)[0])
    return b.finalize(z_white=1)


def test_active_sample_per_ply_with_win_target():
    rec = _game()
    states = reconstruct_states(rec)
    samples = cluster_goal_samples(rec, states, FakeEmbedder(), FakeGoalSpace(), np.random.default_rng(0))
    actives = [s for s in samples if s.v_win_mask == 1.0]
    assert len(actives) == len(rec)             # one active per ply
    for s in actives:
        assert s.v_win in (-1.0, 0.0, 1.0)
        assert s.goal_vec.shape == (4,)


def test_her_samples_have_goal_targets_no_win_mask():
    rec = _game()
    states = reconstruct_states(rec)
    samples = cluster_goal_samples(rec, states, FakeEmbedder(), FakeGoalSpace(), np.random.default_rng(0))
    her = [s for s in samples if s.v_win_mask == 0.0]
    assert her, "expected HER future/negative samples"
    for s in her:
        assert s.v_goal in (0.0, 1.0)


def test_vanilla_record_yields_nothing():
    b = RecordBuilder(); b.add(chess.Board(), [0], [1], 0)
    rec = b.finalize(z_white=0)
    out = cluster_goal_samples(rec, reconstruct_states(rec), FakeEmbedder(), FakeGoalSpace(), np.random.default_rng(0))
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cluster_her.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# chessrl/training/cluster_her.py
"""Cluster-goal HER sample generation for the v2 dual-head vector net (Plan 3).

Relabels goals via the frozen-encoder embedding delta -> nearest cluster
(GoalSpace), instead of v1's board predicates. Each ply yields an *active*
sample (searched cluster goal: outcome target for the tanh win head + achievement
target for the sigmoid goal head + policy target) plus HER future-positives and
negatives (goal-head achievement targets only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chessrl.training.her import HERWeights


@dataclass(frozen=True)
class ClusterGoalSample:
    ply: int
    goal_vec: np.ndarray
    cluster: int
    remaining: int
    v_win: float       # game outcome z (side-to-move), used when v_win_mask==1
    v_win_mask: float  # 1.0 on the active sample, else 0.0
    v_goal: float      # achievement label in {0,1}
    v_goal_weight: float


def _delta(embedder, states, i, t):
    e = embedder.embed_boards([states[i], states[t]])
    return (e[1] - e[0]).astype(np.float32)


def _achieved_cluster(embedder, goalspace, states, i, cluster, rem) -> bool:
    T = len(states) - 1
    end = min(i + rem, T)
    for t in range(i + 1, end + 1):
        if goalspace.achieved(_delta(embedder, states, i, t), cluster):
            return True
    return False


def cluster_goal_samples(rec, states, embedder, goalspace, rng,
                         weights: HERWeights | None = None, deadline_max: int = 60):
    if not rec.has_cluster_goals():
        return []
    w = weights or HERWeights()
    out: list[ClusterGoalSample] = []
    T_ = len(rec)
    k_clusters = goalspace.centroids.shape[0]
    for i in range(T_):
        rem = min(deadline_max, T_ - i)
        active_cluster = int(rec.active_cluster[i])
        active_vec = np.asarray(rec.active_vec[i], np.float32)
        # active sample: outcome target (tanh head) + active-goal achievement (goal head)
        ach = active_cluster >= 0 and _achieved_cluster(embedder, goalspace, states, i, active_cluster, rem)
        out.append(ClusterGoalSample(
            ply=i, goal_vec=active_vec, cluster=active_cluster, remaining=rem,
            v_win=float(rec.outcomes[i]), v_win_mask=1.0,
            v_goal=1.0 if ach else 0.0, v_goal_weight=w.search_laundered))
        if rem <= 0:
            continue
        # future positives: clusters actually reached within the window
        reached = set()
        for t in range(i + 1, min(i + rem, T_) + 1):
            reached.add(goalspace.assign(_delta(embedder, states, i, t)))
        reached.discard(-1)
        pos = sorted(reached - {active_cluster})
        if pos and w.future_samples > 0:
            for j in rng.choice(len(pos), size=min(w.future_samples, len(pos)), replace=False):
                c = int(pos[int(j)])
                out.append(ClusterGoalSample(
                    ply=i, goal_vec=goalspace.centroids[c].astype(np.float32), cluster=c,
                    remaining=rem, v_win=0.0, v_win_mask=0.0,
                    v_goal=1.0, v_goal_weight=w.her_positive))
        # negatives: clusters not reached
        neg = [c for c in range(k_clusters) if c not in reached]
        if neg and w.negative_samples > 0:
            for j in rng.choice(len(neg), size=min(w.negative_samples, len(neg)), replace=False):
                c = int(neg[int(j)])
                out.append(ClusterGoalSample(
                    ply=i, goal_vec=goalspace.centroids[c].astype(np.float32), cluster=c,
                    remaining=rem, v_win=0.0, v_win_mask=0.0,
                    v_goal=0.0, v_goal_weight=w.negative))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cluster_her.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add chessrl/training/cluster_her.py tests/test_cluster_her.py
git commit -m "feat(v2): cluster-goal HER sample generation (dual-head targets)"
```

---

### Task 3: Vector goal replay buffer

**Files:**
- Create: `chessrl/training/vector_buffer.py`
- Test: `tests/test_vector_buffer.py` (new)

**Interfaces:**
- Produces `VectorGoalReplayBuffer(capacity, embedder, goalspace, weights=None, deadline_max=60)`:
  - `add_game(rec, rng=None)`: skip if `not rec.has_cluster_goals()`; else generate `cluster_goal_samples` and append flat tuples `(board_planes int8, goal_vec, remaining, v_win, v_win_mask, v_goal, v_goal_weight, p_idxs, p_cnts)`. Policy target (`p_idxs/p_cnts` from the record) attaches ONLY to the active (`v_win_mask==1`) sample at each ply.
  - `sample(batch, rng) -> (x (B,21,8,8) f32, goal_vec (B,d) f32, deadline (B,) f32, p (B,NUM_ACTIONS) f32, p_mask (B,) f32, v_win (B,) f32, v_win_mask (B,) f32, v_goal (B,) f32, v_goal_weight (B,) f32)`.
  - `__len__`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vector_buffer.py
import numpy as np
import chess
from chessrl.selfplay.records import RecordBuilder
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.training.vector_buffer import VectorGoalReplayBuffer
from tests.test_cluster_her import FakeEmbedder, FakeGoalSpace


def _game(n=4):
    b = RecordBuilder(); board = chess.Board()
    for _ in range(n):
        legal = list(board.legal_moves)
        b.add(board, [0, 1], [3, 1], played_index=0, protagonist=board.turn,
              cluster_active=1, cluster_assigned=1,
              active_vec=np.array([1, 0, 0, 0], np.float32), explore=False)
        board.push(legal[0])
    return b.finalize(z_white=1)


def test_buffer_sample_shapes():
    buf = VectorGoalReplayBuffer(1000, FakeEmbedder(), FakeGoalSpace())
    buf.add_game(_game(), rng=np.random.default_rng(0))
    assert len(buf) > 0
    x, gv, dl, p, pm, vw, vwm, vg, vgw = buf.sample(8, np.random.default_rng(1))
    assert x.shape == (8, 21, 8, 8)
    assert gv.shape == (8, 4)
    assert dl.shape == (8,) and vw.shape == (8,) and vg.shape == (8,)
    assert p.shape == (8, NUM_ACTIONS)
    assert set(np.unique(pm)).issubset({0.0, 1.0})
    assert set(np.unique(vwm)).issubset({0.0, 1.0})


def test_policy_only_on_active_samples():
    buf = VectorGoalReplayBuffer(1000, FakeEmbedder(), FakeGoalSpace())
    buf.add_game(_game(), rng=np.random.default_rng(0))
    x, gv, dl, p, pm, vw, vwm, vg, vgw = buf.sample(64, np.random.default_rng(2))
    # policy mask is set exactly where the win mask is (the active sample)
    assert np.array_equal(pm, vwm)


def test_skips_vanilla():
    b = RecordBuilder(); b.add(chess.Board(), [0], [1], 0)
    buf = VectorGoalReplayBuffer(10, FakeEmbedder(), FakeGoalSpace())
    buf.add_game(b.finalize(z_white=0))
    assert len(buf) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_vector_buffer.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# chessrl/training/vector_buffer.py
"""Replay buffer for the v2 dual-head vector goal net (Plan 3).

Stores flat per-sample descriptors generated from cluster-goal GameRecords via
train-time cluster HER. Yields the vector net's inputs (board planes + goal
centroid + deadline) and dual targets (tanh win outcome + sigmoid goal
achievement) with per-sample masks/weights."""
from collections import deque
from pathlib import Path

import numpy as np

from chessrl.chess_env.encoding import to_model_input
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.selfplay.records import GameRecord
from chessrl.training.her import HERWeights, reconstruct_states
from chessrl.training.cluster_her import cluster_goal_samples


class VectorGoalReplayBuffer:
    def __init__(self, capacity, embedder, goalspace, weights: HERWeights | None = None, deadline_max: int = 60):
        self._data = deque(maxlen=capacity)
        self.embedder = embedder
        self.goalspace = goalspace
        self.weights = weights or HERWeights()
        self.deadline_max = deadline_max

    def __len__(self):
        return len(self._data)

    def add_game(self, rec: GameRecord, rng=None) -> None:
        if not rec.has_cluster_goals():
            return
        if rng is None:
            seed = (len(rec) * 1000003 + int(rec.played[0] if len(rec.played) else 0)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
        states = reconstruct_states(rec)
        samples = cluster_goal_samples(rec, states, self.embedder, self.goalspace, rng,
                                       self.weights, self.deadline_max)
        for s in samples:
            board_planes = rec.planes[s.ply]
            p_idxs = p_cnts = None
            if s.v_win_mask == 1.0:   # active sample carries the policy target
                a, b = rec.policy_offsets[s.ply], rec.policy_offsets[s.ply + 1]
                p_idxs = rec.policy_indices[a:b].astype(np.int64)
                p_cnts = rec.policy_counts[a:b].astype(np.float64)
            self._data.append((board_planes, s.goal_vec, s.remaining, s.v_win,
                               s.v_win_mask, s.v_goal, s.v_goal_weight, p_idxs, p_cnts))

    def sample(self, batch_size, rng):
        if not self._data:
            raise ValueError("cannot sample from an empty vector goal buffer")
        d = self._data[0][1].shape[0]
        picks = rng.integers(0, len(self._data), size=batch_size)
        x = np.empty((batch_size, 21, 8, 8), dtype=np.float32)
        gv = np.empty((batch_size, d), dtype=np.float32)
        deadline = np.empty(batch_size, dtype=np.float32)
        p = np.zeros((batch_size, NUM_ACTIONS), dtype=np.float32)
        p_mask = np.zeros(batch_size, dtype=np.float32)
        v_win = np.empty(batch_size, dtype=np.float32)
        v_win_mask = np.empty(batch_size, dtype=np.float32)
        v_goal = np.empty(batch_size, dtype=np.float32)
        v_goal_w = np.empty(batch_size, dtype=np.float32)
        for row, i in enumerate(picks):
            bp, goal_vec, rem, vw, vwm, vg, vgw, p_idxs, p_cnts = self._data[i]
            x[row] = to_model_input(bp)
            gv[row] = goal_vec
            deadline[row] = float(rem)
            v_win[row] = vw
            v_win_mask[row] = vwm
            v_goal[row] = vg
            v_goal_w[row] = vgw
            if p_idxs is not None and len(p_idxs) and p_cnts.sum() > 0:
                p[row, p_idxs] = (p_cnts / p_cnts.sum()).astype(np.float32)
                p_mask[row] = 1.0
        return x, gv, deadline, p, p_mask, v_win, v_win_mask, v_goal, v_goal_w

    @classmethod
    def from_run_dir(cls, run_dir, capacity, embedder, goalspace,
                     weights: HERWeights | None = None, deadline_max: int = 60):
        buf = cls(capacity, embedder, goalspace, weights=weights, deadline_max=deadline_max)
        files = sorted((Path(run_dir) / "games").glob("*.npz"),
                       key=lambda p: (p.stat().st_mtime, p.name))
        selected, total = [], 0
        for f in reversed(files):
            rec = GameRecord.load(f)
            if not rec.has_cluster_goals():
                continue
            selected.append(rec)
            total += len(rec)
            if total >= capacity:
                break
        for rec in reversed(selected):
            buf.add_game(rec)
        return buf
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_vector_buffer.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add chessrl/training/vector_buffer.py tests/test_vector_buffer.py
git commit -m "feat(v2): VectorGoalReplayBuffer (dual-head HER samples)"
```

---

## Plan 3 deliverable

Records carry cluster-goal ids + centroid vectors + explore flags; cluster HER relabels via the frozen embedder → nearest cluster and emits dual-head samples; `VectorGoalReplayBuffer` yields `(x, goal_vec, deadline, p, p_mask, v_win, v_win_mask, v_goal, v_goal_weight)` for the Plan 1b net. All tested with synthetic records + a fake embedder — no self-play yet.

## Out of scope (Plan 4)

Self-play driving `GoalSpace` (assign clusters, ε-explore, play under win-value + PBS shaping, write the new records), frozen-encoder snapshotting in the training loop, the dual-head loss (MSE on `v_win` masked + weighted BCE on `v_goal`), interventional win-value, curriculum `γ·win_value`, and `experiments/v2-stage1.yaml`.
