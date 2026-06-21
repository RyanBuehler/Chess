# v2 Stage 1 — Plan 2: GoalSpace (Frozen-Encoder Discovered Goals)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `GoalSpace` — the module that discovers goals as clusters of state-delta embeddings from a **frozen encoder snapshot**: it accumulates deltas in a reservoir, fits k-means to get centroid "goal codes", assigns/【checks-achievement-of】 goals by nearest-centroid distance, refreshes periodically, and persists/reloads.

**Architecture:** Plan 1 gave the net a vector-conditioning interface and a `VectorGoalNetEvaluator.embed_boards(boards) -> (N, d)` that returns the trunk embedding `e(s)`. Plan 2 builds three small, independently-testable units — a fixed-size `Reservoir`, a numpy `clustering` util (k-means with empty-cluster reseeding), and the `GoalSpace` orchestrator that ties them to an injected embedder — plus persistence. No self-play / HER / training wiring (those are Plans 3–4); `GoalSpace` is tested directly with a fake embedder.

**Tech Stack:** Python 3, NumPy (no scikit-learn dependency — k-means is implemented in numpy), pytest, python-chess. New code in `chessrl/goals/`.

## Global Constraints

- `d` (embedding / centroid dim) = `NetworkConfig.filters`. `GoalSpace` reads it from the injected embedder's output, never hardcodes it.
- A goal is a **cluster id** (int in `[0, n_clusters)`); its code is the centroid `c_k ∈ R^d`. The WIN goal is NOT a cluster (handled elsewhere).
- Delta over a window: `Δ = e(s_end) − e(s_start)` (embedding difference), `e` from the FROZEN encoder snapshot, not the live net.
- Achievement: goal `k` achieved at state `s` iff `argmin_j ‖Δ(s) − c_j‖ == k` AND `‖Δ(s) − c_k‖ ≤ τ`, where `τ` = median intra-cluster member distance at the last fit.
- Frozen-encoder snapshot is stationary within an epoch; refresh (re-snapshot + re-fit) happens every `refresh_every` games. Persisted to the run dir, reloaded on resume.
- Cold start: before the reservoir holds `min_reservoir` deltas, `GoalSpace.ready` is `False` and callers must fall back to WIN-only (no clusters exist yet).
- Determinism: all randomness takes an explicit `numpy.random.Generator`; same seed + same data ⇒ same centroids.
- Run tests on Windows with the venv, unpiped and foreground: `.venv\Scripts\python.exe -m pytest <path> -v`.
- Do NOT `git add -A`; stage only the files each task names.

---

### Task 1: GoalConfig fields for the goal space

**Files:**
- Modify: `chessrl/config/config.py` (GoalConfig dataclass, ~lines 66–76)
- Test: `tests/test_goalspace_config.py` (new)

**Interfaces:**
- Produces on `GoalConfig`: `cluster_k: int = 48`, `refresh_every: int = 2000`, `reservoir_size: int = 20000`, `min_reservoir: int = 5000`, `goal_window: int = 8`. Also extend the `goal_mode` validation to accept `"emergent"` (the v2 mode) in addition to the existing `none|always_win|random|lp`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goalspace_config.py
import pytest
from chessrl.config.config import GoalConfig


def test_goalspace_defaults():
    c = GoalConfig()
    assert c.cluster_k == 48
    assert c.refresh_every == 2000
    assert c.reservoir_size == 20000
    assert c.min_reservoir == 5000
    assert c.goal_window == 8


def test_emergent_mode_allowed():
    assert GoalConfig(goal_mode="emergent").goal_mode == "emergent"


def test_bad_mode_still_rejected():
    with pytest.raises(ValueError):
        GoalConfig(goal_mode="nonsense")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_goalspace_config.py -v`
Expected: FAIL — `AttributeError`/`TypeError` on the missing fields; `test_emergent_mode_allowed` raises `ValueError`.

- [ ] **Step 3: Add the fields + extend validation**

In `chessrl/config/config.py`, inside `GoalConfig` add (keep existing fields like `goal_mode`, `win_floor`, `deadline_max`):

```python
    cluster_k: int = 48          # k-means cluster count = discovered goal vocabulary size
    refresh_every: int = 2000    # games between frozen-encoder re-snapshot + re-fit
    reservoir_size: int = 20000  # capacity of the delta reservoir
    min_reservoir: int = 5000    # deltas required before the goal space is "ready"
    goal_window: int = 8         # plies over which a state-delta is measured
```

Extend the existing `goal_mode` check (it currently lists `none|always_win|random|lp`):

```python
        if self.goal_mode not in ("none", "always_win", "random", "lp", "emergent"):
            raise ValueError(f"bad goal_mode {self.goal_mode}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_goalspace_config.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add chessrl/config/config.py tests/test_goalspace_config.py
git commit -m "feat(v2): GoalConfig fields for goal space + emergent goal_mode"
```

---

### Task 2: `Reservoir` — fixed-size reservoir sampler for delta vectors

**Files:**
- Create: `chessrl/goals/reservoir.py`
- Test: `tests/test_goal_reservoir.py` (new)

**Interfaces:**
- Produces: `Reservoir(capacity: int, dim: int, rng: np.random.Generator)`; `add(vec: np.ndarray) -> None`; `array() -> np.ndarray (n, dim)` (n = min(seen, capacity)); `__len__`; `seen: int` (total added). Uniform reservoir sampling once full.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goal_reservoir.py
import numpy as np
from chessrl.goals.reservoir import Reservoir


def test_fills_then_caps():
    r = Reservoir(capacity=100, dim=4, rng=np.random.default_rng(0))
    for i in range(50):
        r.add(np.full(4, float(i)))
    assert len(r) == 50 and r.seen == 50
    assert r.array().shape == (50, 4)
    for i in range(200):
        r.add(np.full(4, 1000.0 + i))
    assert len(r) == 100 and r.seen == 250
    assert r.array().shape == (100, 4)


def test_uniform_sampling_keeps_a_mix():
    # After overflow, the reservoir should contain some early and some late items
    # (not exclusively the last `capacity`). With a fixed seed this is deterministic.
    r = Reservoir(capacity=10, dim=1, rng=np.random.default_rng(42))
    for i in range(1000):
        r.add(np.array([float(i)]))
    vals = r.array().ravel()
    assert len(vals) == 10
    assert vals.min() < 990  # at least one item from before the final window survived


def test_dtype_is_float32():
    r = Reservoir(capacity=5, dim=3, rng=np.random.default_rng(1))
    r.add(np.ones(3, dtype=np.float64))
    assert r.array().dtype == np.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_goal_reservoir.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessrl.goals.reservoir'`.

- [ ] **Step 3: Implement**

```python
# chessrl/goals/reservoir.py
"""Fixed-size reservoir sampler over delta vectors (Plan 2, Task 2).

Classic Algorithm R: the first ``capacity`` items are kept; each later item
(index i, 0-based, i >= capacity) replaces a uniformly random slot with
probability ``capacity / (i + 1)``. This yields a uniform random sample of the
whole stream at any time — so the k-means fit sees old and new deltas, not just
the most recent window."""
from __future__ import annotations

import numpy as np


class Reservoir:
    def __init__(self, capacity: int, dim: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.rng = rng
        self._buf = np.zeros((self.capacity, self.dim), dtype=np.float32)
        self._n = 0      # filled slots
        self.seen = 0    # total ever added

    def add(self, vec: np.ndarray) -> None:
        v = np.asarray(vec, dtype=np.float32).reshape(self.dim)
        if self._n < self.capacity:
            self._buf[self._n] = v
            self._n += 1
        else:
            j = int(self.rng.integers(self.seen + 1))
            if j < self.capacity:
                self._buf[j] = v
        self.seen += 1

    def array(self) -> np.ndarray:
        return self._buf[: self._n].copy()

    def __len__(self) -> int:
        return self._n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_goal_reservoir.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add chessrl/goals/reservoir.py tests/test_goal_reservoir.py
git commit -m "feat(v2): fixed-size delta Reservoir (Algorithm R)"
```

---

### Task 3: `clustering` — numpy k-means with empty-cluster reseeding

**Files:**
- Create: `chessrl/goals/clustering.py`
- Test: `tests/test_goal_clustering.py` (new)

**Interfaces:**
- Produces:
  - `kmeans_fit(x (n,d), k, rng, iters=25) -> centroids (k,d)` — Lloyd's; an empty cluster is reseeded to the point farthest from its centroid, so exactly `k` centroids are always returned (no collapse). If `n < k`, returns the `n` unique points padded by duplicating random points to length `k`.
  - `assign_nearest(x (m,d), centroids (k,d)) -> labels (m,)` int64.
  - `median_radius(x (n,d), centroids, labels) -> float` — median over points of `‖x_i − c_{label_i}‖` (the achievement threshold τ). Returns 0.0 for empty input.

This empty-cluster-reseed approach **supersedes** the spec's drop-empties / split-largest / K_min language: reseeding keeps a constant `k` and prevents degenerate (empty) clusters inline, achieving the same "no degenerate clusters" goal more simply.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goal_clustering.py
import numpy as np
from chessrl.goals.clustering import kmeans_fit, assign_nearest, median_radius


def _three_blobs(rng):
    a = rng.normal([0, 0], 0.05, size=(100, 2))
    b = rng.normal([5, 5], 0.05, size=(100, 2))
    c = rng.normal([0, 5], 0.05, size=(100, 2))
    return np.vstack([a, b, c]).astype(np.float32)


def test_recovers_blob_centers():
    rng = np.random.default_rng(0)
    x = _three_blobs(rng)
    cents = kmeans_fit(x, k=3, rng=rng)
    assert cents.shape == (3, 2)
    # every true center has a near centroid
    for tc in ([0, 0], [5, 5], [0, 5]):
        d = np.linalg.norm(cents - np.array(tc), axis=1).min()
        assert d < 0.5, (tc, cents)


def test_assign_nearest_labels():
    cents = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
    x = np.array([[0.1, 0.0], [9.9, 10.1]], dtype=np.float32)
    labels = assign_nearest(x, cents)
    assert labels.tolist() == [0, 1]


def test_always_returns_k_centroids_even_with_few_points():
    rng = np.random.default_rng(1)
    x = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)  # n=2 < k=5
    cents = kmeans_fit(x, k=5, rng=rng)
    assert cents.shape == (5, 2)


def test_no_empty_clusters_after_fit():
    rng = np.random.default_rng(2)
    x = _three_blobs(rng)
    cents = kmeans_fit(x, k=5, rng=rng)  # more clusters than blobs
    labels = assign_nearest(x, cents)
    # every centroid index appears (reseeding prevents empties)
    assert set(labels.tolist()) == set(range(5))


def test_median_radius_nonneg():
    rng = np.random.default_rng(3)
    x = _three_blobs(rng)
    cents = kmeans_fit(x, k=3, rng=rng)
    labels = assign_nearest(x, cents)
    tau = median_radius(x, cents, labels)
    assert tau >= 0.0 and np.isfinite(tau)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_goal_clustering.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessrl.goals.clustering'`.

- [ ] **Step 3: Implement**

```python
# chessrl/goals/clustering.py
"""NumPy k-means for goal discovery (Plan 2, Task 3).

Lloyd's algorithm with empty-cluster reseeding: any cluster that loses all its
members is moved to the single point currently farthest from its assigned
centroid. This guarantees exactly ``k`` non-empty centroids every fit (no
collapse), so the discovered-goal vocabulary size stays constant. No
scikit-learn dependency — refits are periodic, not per-step, so plain Lloyd's
is fast enough."""
from __future__ import annotations

import numpy as np


def assign_nearest(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # squared euclidean distance to each centroid, argmin
    d2 = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).astype(np.int64)


def kmeans_fit(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    d = x.shape[1]
    if n == 0:
        return np.zeros((k, d), dtype=np.float32)
    if n < k:
        # pad by duplicating random points to length k
        idx = rng.integers(n, size=k)
        return x[idx].copy()
    # k-means++-lite init: random distinct starting points
    start = rng.choice(n, size=k, replace=False)
    centroids = x[start].copy()
    for _ in range(iters):
        labels = assign_nearest(x, centroids)
        new = centroids.copy()
        for j in range(k):
            members = x[labels == j]
            if len(members) == 0:
                # reseed to the farthest point from its own centroid
                d2 = ((x - centroids[labels]) ** 2).sum(axis=1)
                new[j] = x[int(d2.argmax())]
            else:
                new[j] = members.mean(axis=0)
        if np.allclose(new, centroids):
            centroids = new
            break
        centroids = new
    return centroids.astype(np.float32)


def median_radius(x: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    if x.shape[0] == 0:
        return 0.0
    dists = np.linalg.norm(x - centroids[labels], axis=1)
    return float(np.median(dists))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_goal_clustering.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add chessrl/goals/clustering.py tests/test_goal_clustering.py
git commit -m "feat(v2): numpy k-means with empty-cluster reseeding + median radius"
```

---

### Task 4: `GoalSpace` — orchestrator (delta, observe, fit, assign, achieved, ready)

**Files:**
- Create: `chessrl/goals/goalspace.py`
- Test: `tests/test_goalspace.py` (new)

**Interfaces:**
- Consumes: `Reservoir` (Task 2); `kmeans_fit`/`assign_nearest`/`median_radius` (Task 3); `GoalConfig` (Task 1); an **embedder** with `embed_boards(boards: list[chess.Board]) -> np.ndarray (N, d)` (Plan 1's `VectorGoalNetEvaluator`, or a fake in tests).
- Produces:
  - `GoalSpace(cfg: GoalConfig, embedder, rng)` — `d` inferred from a one-board probe of the embedder.
  - `delta(start_board, end_board) -> np.ndarray (d,)`.
  - `observe(start_board, end_board) -> None` (adds the delta to the reservoir).
  - `ready -> bool` (`len(reservoir) >= cfg.min_reservoir` AND a fit exists).
  - `fit() -> None` (k-means over the reservoir; sets `centroids (k,d)`, `tau`, `n_clusters`).
  - `assign(delta (d,)) -> int`.
  - `achieved(delta (d,), goal_id: int) -> bool` (`assign(delta)==goal_id and ‖delta−centroid‖<=tau`).
  - `centroid(goal_id) -> np.ndarray (d,)`.
  - `maybe_refresh(games_seen: int, embedder=None) -> bool` (re-fit when `games_seen` crosses a multiple of `cfg.refresh_every` and ready; swaps in a new frozen embedder if provided; returns True if it refit).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goalspace.py
import numpy as np
import chess
from chessrl.config.config import GoalConfig
from chessrl.goals.goalspace import GoalSpace


class FakeEmbedder:
    """Deterministic embedder: e(board) = [material_count, n_pieces, fullmove, 0].
    d=4. Distinct boards map to distinct, meaningful vectors."""
    def embed_boards(self, boards):
        out = []
        for b in boards:
            mat = sum(len(b.pieces(pt, c)) * v for pt, v in
                      [(chess.PAWN, 1), (chess.KNIGHT, 3), (chess.BISHOP, 3),
                       (chess.ROOK, 5), (chess.QUEEN, 9)] for c in (chess.WHITE, chess.BLACK))
            n = sum(1 for _ in b.piece_map())
            out.append([float(mat), float(n), float(b.fullmove_number), 0.0])
        return np.asarray(out, dtype=np.float32)


def _cfg(**kw):
    return GoalConfig(goal_mode="emergent", cluster_k=3, min_reservoir=20,
                      reservoir_size=200, refresh_every=50, **kw)


def test_delta_dim_and_value():
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    b0 = chess.Board()
    b1 = chess.Board()
    b1.push_san("e4")
    d = gs.delta(b0, b1)
    assert d.shape == (4,)


def test_not_ready_before_min_reservoir():
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    assert gs.ready is False
    b = chess.Board()
    for _ in range(5):
        gs.observe(b, b)
    assert gs.ready is False  # below min_reservoir and not fit


def test_fit_makes_ready_and_assigns():
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    rng = np.random.default_rng(7)
    # synthesize 60 deltas in 3 separable groups by adding directly
    for grp in ([10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0]):
        for _ in range(20):
            gs.observe_delta(np.array(grp, np.float32) + rng.normal(0, 0.1, 4).astype(np.float32))
    gs.fit()
    assert gs.ready is True
    assert gs.n_clusters == 3
    gid = gs.assign(np.array([10, 0, 0, 0], np.float32))
    assert 0 <= gid < 3
    # a delta near that centroid is achieved for that goal
    assert gs.achieved(gs.centroid(gid), gid) is True
    # a far delta is not achieved for that goal
    assert gs.achieved(np.array([0, 0, 10, 0], np.float32), gid) is False


def test_maybe_refresh_refits_on_boundary():
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    for _ in range(30):
        gs.observe_delta(np.random.default_rng(0).normal(0, 1, 4).astype(np.float32))
    gs.fit()
    first = gs.centroids.copy()
    # add more, cross a refresh_every boundary
    for _ in range(30):
        gs.observe_delta(np.array([100, 100, 100, 0], np.float32))
    did = gs.maybe_refresh(games_seen=50)
    assert did is True
    assert not np.allclose(first, gs.centroids)  # refit moved centroids


def test_observe_uses_embedder_delta():
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    b0 = chess.Board()
    b1 = chess.Board(); b1.push_san("e4")
    gs.observe(b0, b1)
    assert len(gs.reservoir) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_goalspace.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chessrl.goals.goalspace'`.

- [ ] **Step 3: Implement**

```python
# chessrl/goals/goalspace.py
"""Discovered-goal space: clusters of state-delta embeddings (Plan 2, Task 4).

A goal is a cluster id; its code is the centroid in the FROZEN encoder's
embedding space. GoalSpace accumulates deltas in a reservoir, fits k-means to
get the goal vocabulary, and answers assign()/achieved() by nearest centroid
within the median-radius threshold tau. The embedder is injected (Plan 1's
VectorGoalNetEvaluator in production; a fake in tests) and is treated as frozen
within an epoch; maybe_refresh() swaps in a fresh snapshot and re-fits."""
from __future__ import annotations

import chess
import numpy as np

from chessrl.config.config import GoalConfig
from chessrl.goals.reservoir import Reservoir
from chessrl.goals.clustering import kmeans_fit, assign_nearest, median_radius


class GoalSpace:
    def __init__(self, cfg: GoalConfig, embedder, rng: np.random.Generator):
        self.cfg = cfg
        self.embedder = embedder
        self.rng = rng
        self.d = int(embedder.embed_boards([chess.Board()]).shape[1])
        self.reservoir = Reservoir(cfg.reservoir_size, self.d, rng)
        self.centroids: np.ndarray | None = None
        self.tau: float = 0.0
        self._last_refresh_epoch = -1

    # --- embedding / observation ----------------------------------------
    def delta(self, start_board: chess.Board, end_board: chess.Board) -> np.ndarray:
        e = self.embedder.embed_boards([start_board, end_board])
        return (e[1] - e[0]).astype(np.float32)

    def observe(self, start_board: chess.Board, end_board: chess.Board) -> None:
        self.observe_delta(self.delta(start_board, end_board))

    def observe_delta(self, delta: np.ndarray) -> None:
        self.reservoir.add(np.asarray(delta, dtype=np.float32))

    # --- fitting --------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.centroids is not None and len(self.reservoir) >= self.cfg.min_reservoir

    @property
    def n_clusters(self) -> int:
        return 0 if self.centroids is None else self.centroids.shape[0]

    def fit(self) -> None:
        x = self.reservoir.array()
        cents = kmeans_fit(x, self.cfg.cluster_k, self.rng)
        labels = assign_nearest(x, cents)
        self.centroids = cents
        self.tau = median_radius(x, cents, labels)

    def maybe_refresh(self, games_seen: int, embedder=None) -> bool:
        epoch = games_seen // self.cfg.refresh_every
        if epoch <= self._last_refresh_epoch:
            return False
        if len(self.reservoir) < self.cfg.min_reservoir:
            return False
        if embedder is not None:
            self.embedder = embedder
        self.fit()
        self._last_refresh_epoch = epoch
        return True

    # --- queries --------------------------------------------------------
    def assign(self, delta: np.ndarray) -> int:
        assert self.centroids is not None, "GoalSpace not fit yet"
        d = np.asarray(delta, dtype=np.float32).reshape(1, self.d)
        return int(assign_nearest(d, self.centroids)[0])

    def centroid(self, goal_id: int) -> np.ndarray:
        assert self.centroids is not None, "GoalSpace not fit yet"
        return self.centroids[goal_id].copy()

    def achieved(self, delta: np.ndarray, goal_id: int) -> bool:
        if self.centroids is None:
            return False
        d = np.asarray(delta, dtype=np.float32).reshape(self.d)
        if self.assign(d) != goal_id:
            return False
        return bool(np.linalg.norm(d - self.centroids[goal_id]) <= self.tau)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_goalspace.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add chessrl/goals/goalspace.py tests/test_goalspace.py
git commit -m "feat(v2): GoalSpace orchestrator (observe/fit/assign/achieved/refresh)"
```

---

### Task 5: GoalSpace persistence (save/load to run dir)

**Files:**
- Modify: `chessrl/goals/goalspace.py` (add `save`/`load`)
- Test: `tests/test_goalspace.py` (extend)

**Interfaces:**
- Produces:
  - `GoalSpace.save(path: str | Path) -> None` — writes `centroids`, the reservoir buffer, `tau`, `d`, `cluster_k`, and `_last_refresh_epoch` under `path` (a directory).
  - `GoalSpace.load(path, cfg, embedder, rng) -> GoalSpace` (classmethod) — reconstructs a fitted, ready GoalSpace whose `assign`/`achieved` match the saved one.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_goalspace.py
from pathlib import Path


def test_save_load_roundtrip(tmp_path):
    gs = GoalSpace(_cfg(), FakeEmbedder(), np.random.default_rng(0))
    rng = np.random.default_rng(7)
    for grp in ([10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0]):
        for _ in range(20):
            gs.observe_delta(np.array(grp, np.float32) + rng.normal(0, 0.1, 4).astype(np.float32))
    gs.fit()
    probe = np.array([10, 0, 0, 0], np.float32)
    gid_before = gs.assign(probe)
    tau_before = gs.tau

    gs.save(tmp_path / "goalspace")
    gs2 = GoalSpace.load(tmp_path / "goalspace", _cfg(), FakeEmbedder(), np.random.default_rng(99))

    assert gs2.ready is True
    assert gs2.n_clusters == gs.n_clusters
    assert np.allclose(gs2.centroids, gs.centroids)
    assert gs2.tau == tau_before
    assert gs2.assign(probe) == gid_before
    assert len(gs2.reservoir) == len(gs.reservoir)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_goalspace.py::test_save_load_roundtrip -v`
Expected: FAIL — `AttributeError: 'GoalSpace' object has no attribute 'save'`.

- [ ] **Step 3: Implement**

Add the imports at the top of `chessrl/goals/goalspace.py` (with the existing imports):

```python
import json
from pathlib import Path
```

Add these methods to `GoalSpace`:

```python
    def save(self, path) -> None:
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "centroids.npy", self.centroids if self.centroids is not None
                else np.zeros((0, self.d), np.float32))
        np.save(d / "reservoir.npy", self.reservoir.array())
        meta = {"d": self.d, "tau": self.tau, "cluster_k": self.cfg.cluster_k,
                "last_refresh_epoch": self._last_refresh_epoch, "seen": self.reservoir.seen}
        (d / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path, cfg: GoalConfig, embedder, rng: np.random.Generator) -> "GoalSpace":
        d = Path(path)
        meta = json.loads((d / "meta.json").read_text())
        gs = cls(cfg, embedder, rng)
        res_arr = np.load(d / "reservoir.npy")
        for row in res_arr:
            gs.reservoir.add(row)
        gs.reservoir.seen = int(meta["seen"])
        cents = np.load(d / "centroids.npy")
        gs.centroids = cents if cents.shape[0] > 0 else None
        gs.tau = float(meta["tau"])
        gs._last_refresh_epoch = int(meta["last_refresh_epoch"])
        return gs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_goalspace.py -v`
Expected: PASS (all, including roundtrip).

- [ ] **Step 5: Commit**

```bash
git add chessrl/goals/goalspace.py tests/test_goalspace.py
git commit -m "feat(v2): GoalSpace save/load persistence for resume"
```

---

## Plan 2 deliverable

A `GoalSpace` that discovers goals as k-means clusters of frozen-encoder state-delta embeddings: reservoir accumulation, periodic re-fit, nearest-centroid `assign`/`achieved` within τ, `ready` cold-start gating, and save/load — all green, tested with a fake embedder, no self-play/training coupling. This is the goal vocabulary Plan 3 (HER/buffer onto cluster goals) and Plan 4 (means-end objective + win-value + curriculum + ε-explore assignment) build on.

## Out of scope (later plans)

- Plan 3: HER/buffer rewrite onto cluster goals (`her.py`, `buffer.py`) — relabel achieved deltas to cluster ids via `GoalSpace`, emit `(board_planes, goal_vector, deadline, BCE target)` samples for the vector net.
- Plan 4: means-end objective (`V(s,win)=net(s,c_win)` target + PBS shaping `Φ=net(s,c_g)` + α); interventional ε-explore assignment + per-cluster win-value; curriculum `γ·win_value` term; frozen-encoder snapshotting wired into the training loop; `experiments/v2-stage1.yaml` (`goal_mode: emergent`).
- Plan 5: eval (curve backfill, α-sweep) + live UI (cluster-id + win-value).
