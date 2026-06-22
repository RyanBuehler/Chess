# v2 Stage 1 — Plan 4a: Vector-Conditioned Means-End MCTS

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a **means-end mode** to `BatchedMCTS`: vanilla-style negamax search over the expected **terminal reward** (tanh `v_win`), conditioned on a goal **vector** (centroid) via the dual-head `VectorGoalNetEvaluator`, with a **blended leaf value** `(1−α)·v_win + α·(2·v_goal−1)` realizing the intrinsic goal-shaping. Preserves v1 planes goal-mode and vanilla unchanged.

**Architecture:** Reading `batched.py` showed means-end search is *vanilla negamax* (game terminals, sign-flip backup, vanilla `_select`) — NOT v1's protagonist-frame achievement minimax. So means-end keeps `goal_mode=False` (reusing all negamax machinery untouched) and adds a separate `meansend` flag. The ONLY new behavior is at leaf evaluation: park `(board_planes, goal_vec, deadline)`, call `evaluator.evaluate_planes(planes, goal_vecs, deadlines) -> (policies, v_win, v_goal)`, and back up the **blended** value. This MCTS backs up values (no per-edge reward), so the spec's potential-based shaping is realized as this α-blended leaf value — the MCTS-native intrinsic-bonus form; at α=0 the value is pure terminal reward (vanilla-strength floor) while the goal-conditioned *policy priors* still shape exploration.

**Tech Stack:** Python, NumPy, python-chess, pytest. Files: `chessrl/config/config.py`, `chessrl/mcts/batched.py`.

## Global Constraints

- Means-end mode uses NEGAMAX (sign-flip backup), GAME terminals (`terminal_value`), and the VANILLA `_select` — i.e. `self.goal_mode` stays `False`; a new `self.meansend` flag gates only the leaf evaluation. v1 planes goal-mode (`goal_mode=True`) and vanilla are BYTE-FOR-BYTE unchanged.
- Leaf value (means-end) = `(1−α)·v_win + α·(2·v_goal−1)`, all in [-1,1]. `v_win` = tanh terminal-reward head; `v_goal` = sigmoid achievement head, remapped `2·v_goal−1` to [-1,1].
- Conditioning is a goal **vector** (centroid, dim `d=filters`) + a **raw** deadline scalar (the net scales internally — the C1 fix). Means-end leaf planes are **board-only** (`NUM_PLANES`), NOT board⊕goal-planes.
- Evaluator contract (from Plan 1b): `VectorGoalNetEvaluator.evaluate_planes(planes (N,21,8,8) f32, goal_vecs (N,d) f32, deadlines (N,) f32) -> (policies (N,4672) f32, v_win (N,) f32, v_goal (N,) f32)`.
- `α = MCTSConfig.meansend_alpha` (default 0.0).
- Windows venv tests, unpiped/foreground. Stage only named files; never `git add -A`.

---

### Task 1: `MCTSConfig.meansend_alpha`

**Files:**
- Modify: `chessrl/config/config.py` (MCTSConfig)
- Test: `tests/test_meansend_mcts.py` (new)

**Interfaces:** `MCTSConfig.meansend_alpha: float = 0.0` (validated `0.0 <= alpha <= 1.0`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_meansend_mcts.py
import pytest
from chessrl.config.config import MCTSConfig


def test_meansend_alpha_default():
    assert MCTSConfig().meansend_alpha == 0.0


def test_meansend_alpha_settable():
    assert MCTSConfig(meansend_alpha=0.5).meansend_alpha == 0.5


def test_meansend_alpha_rejects_out_of_range():
    with pytest.raises(ValueError):
        MCTSConfig(meansend_alpha=1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_meansend_mcts.py -v`
Expected: FAIL — unknown kwarg / missing attribute.

- [ ] **Step 3: Implement**

In `chessrl/config/config.py`, add to `MCTSConfig`:

```python
    meansend_alpha: float = 0.0   # v2 means-end leaf blend: (1-a)*v_win + a*(2*v_goal-1)
```

Add/extend its `__post_init__`:

```python
    def __post_init__(self):
        if not (0.0 <= self.meansend_alpha <= 1.0):
            raise ValueError(f"meansend_alpha must be in [0,1], got {self.meansend_alpha}")
```

(If `MCTSConfig` already has a `__post_init__`, append the check.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_meansend_mcts.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add chessrl/config/config.py tests/test_meansend_mcts.py
git commit -m "feat(v2): MCTSConfig.meansend_alpha (means-end leaf blend)"
```

---

### Task 2: `BatchedMCTS` means-end mode

**Files:**
- Modify: `chessrl/mcts/batched.py` (SearchTree slots; `__init__`; `init_tree_for_meansend`; `step_round` leaf-collect + batched-eval; `_expand_leaf`)
- Test: `tests/test_meansend_mcts.py` (extend)

**Interfaces:**
- `SearchTree` gains `goal_vec` (np.ndarray | None) and `deadline_origin` (int | None) slots.
- `BatchedMCTS(evaluator, cfg, rng, meansend=False)`: when `meansend=True`, `self.meansend=True` and `self.goal_mode=False`.
- `init_tree_for_meansend(board, goal_vec, deadline, add_noise=False) -> SearchTree`.
- Leaf value in means-end mode = `(1-α)·v_win + α·(2·v_goal-1)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_meansend_mcts.py`:

```python
import numpy as np
import chess
from chessrl.config.config import MCTSConfig
from chessrl.mcts.batched import BatchedMCTS


class FakeDualEval:
    """Dual-head batched evaluator: uniform policy, fixed v_win, fixed v_goal.
    evaluate_planes(planes (N,21,8,8), goal_vecs (N,d), deadlines (N,)) ->
    (policies (N,4672), v_win (N,), v_goal (N,))."""
    def __init__(self, v_win=0.4, v_goal=0.9, n_actions=4672):
        self.v_win = v_win; self.v_goal = v_goal; self.n = n_actions
    def evaluate_planes(self, planes, goal_vecs, deadlines):
        n = planes.shape[0]
        pol = np.full((n, self.n), 1.0 / self.n, dtype=np.float32)
        return pol, np.full(n, self.v_win, np.float32), np.full(n, self.v_goal, np.float32)


def _cfg(alpha):
    return MCTSConfig(simulations=16, leaves_per_tree=1, meansend_alpha=alpha)


def test_meansend_runs_and_produces_visits():
    mcts = BatchedMCTS(FakeDualEval(), _cfg(0.25), np.random.default_rng(0), meansend=True)
    gv = np.zeros(8, np.float32)
    tree = mcts.init_tree_for_meansend(chess.Board(), gv, deadline=20, add_noise=False)
    mcts.run(tree)
    visits = mcts.visit_counts(tree)
    assert sum(visits.values()) > 0


def test_meansend_leaf_blend_alpha0_is_win_only():
    # alpha=0 -> root q reflects v_win only (negamax over v_win); v_goal ignored.
    mcts0 = BatchedMCTS(FakeDualEval(v_win=0.4, v_goal=0.9), _cfg(0.0),
                        np.random.default_rng(1), meansend=True)
    t0 = mcts0.init_tree_for_meansend(chess.Board(), np.zeros(8, np.float32), 20)
    mcts0.run(t0)
    # With uniform policy + constant leaf value, root q ~ the leaf value (sign aside);
    # alpha=0 must equal a run where v_goal is different but alpha=0 (goal ignored).
    mcts0b = BatchedMCTS(FakeDualEval(v_win=0.4, v_goal=0.1), _cfg(0.0),
                         np.random.default_rng(1), meansend=True)
    t0b = mcts0b.init_tree_for_meansend(chess.Board(), np.zeros(8, np.float32), 20)
    mcts0b.run(t0b)
    assert abs(mcts0.root_q(t0) - mcts0b.root_q(t0b)) < 1e-6  # v_goal had no effect


def test_meansend_alpha_uses_v_goal():
    # alpha=1 -> leaf value = 2*v_goal-1; changing v_goal changes root q.
    a = BatchedMCTS(FakeDualEval(v_win=0.4, v_goal=0.9), _cfg(1.0),
                    np.random.default_rng(2), meansend=True)
    ta = a.init_tree_for_meansend(chess.Board(), np.zeros(8, np.float32), 20); a.run(ta)
    b = BatchedMCTS(FakeDualEval(v_win=0.4, v_goal=0.1), _cfg(1.0),
                    np.random.default_rng(2), meansend=True)
    tb = b.init_tree_for_meansend(chess.Board(), np.zeros(8, np.float32), 20); b.run(tb)
    assert abs(a.root_q(ta) - b.root_q(tb)) > 1e-3   # v_goal now matters


def test_vanilla_path_untouched():
    # A vanilla BatchedMCTS (no meansend, no goal) still works.
    class FakeVanilla:
        def evaluate_many(self, boards):
            n = len(boards)
            return np.full((n, 4672), 1.0/4672, np.float32), np.zeros(n, np.float32)
        def evaluate_planes(self, planes):
            n = planes.shape[0]
            return np.full((n, 4672), 1.0/4672, np.float32), np.zeros(n, np.float32)
    mcts = BatchedMCTS(FakeVanilla(), MCTSConfig(simulations=8, leaves_per_tree=1),
                       np.random.default_rng(0))
    tree = mcts.init_tree(chess.Board())
    mcts.run(tree)
    assert sum(mcts.visit_counts(tree).values()) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_meansend_mcts.py -v`
Expected: FAIL — `meansend` kwarg / `init_tree_for_meansend` missing.

- [ ] **Step 3: Implement**

In `chessrl/mcts/batched.py`:

Add two slots to `SearchTree.__slots__` and `__init__`:

```python
    __slots__ = ("root", "board", "sims_done", "baseline", "goal", "protagonist",
                 "goal_vec", "deadline_origin")

    def __init__(self, board, baseline=None, goal=None, protagonist=None,
                 goal_vec=None, deadline_origin=None):
        self.root = Node(0.0)
        self.board = board
        self.sims_done = 0
        self.baseline = baseline
        self.goal = goal
        self.protagonist = protagonist
        self.goal_vec = goal_vec
        self.deadline_origin = deadline_origin
```

In `BatchedMCTS.__init__`, add a `meansend=False` parameter and set the flag (keep `goal_mode` False in means-end):

```python
    def __init__(self, evaluator_many, cfg, rng=None, goal=None,
                 protagonist=None, goal_mode=None, meansend=False):
        ...
        self.meansend = meansend
        # (existing goal_mode logic unchanged; meansend does NOT set goal_mode)
```

Add the builder:

```python
    def init_tree_for_meansend(self, board, goal_vec, deadline, add_noise=False):
        """Means-end tree: vanilla negamax mechanics, but leaves are evaluated by
        the dual-head vector evaluator and the value is the alpha-blend. Carries
        its own goal centroid + deadline origin. goal_mode stays False."""
        if not self.meansend:
            raise ValueError("init_tree_for_meansend requires meansend=True")
        b = board.copy()
        tree = SearchTree(b, goal_vec=np.asarray(goal_vec, np.float32),
                          deadline_origin=int(deadline))
        value = self._expand_leaf(tree, tree.root, plies_from_root=0)
        tree.root.visit_count += 1
        tree.root.value_sum += value
        if add_noise and tree.root.children:
            self._add_dirichlet(tree.root)
        return tree
```

In `step_round`, the leaf-collection block currently branches `if self.goal_mode: ... else: ...` to build `planes`/`deadline`. Add a means-end case FIRST (board-only planes + goal_vec + raw deadline), and extend the parked tuple with a `goal_vec` element. Replace the parked-append region with:

```python
                    flip = tree.board.turn == chess.BLACK
                    legal_idxs = [move_to_index(m, flip) for m in tree.board.legal_moves]
                    goal_vec = None
                    if self.meansend:
                        planes = to_model_input(encode_board(tree.board))
                        deadline = tree.deadline_origin - plies
                        goal_vec = tree.goal_vec
                    elif self.goal_mode:
                        goal, protagonist = self._ctx(tree)
                        remaining = goal.deadline - plies
                        goal_planes, _ = encode_goal(goal, remaining, protagonist)
                        board_planes = to_model_input(encode_board(tree.board))
                        planes = np.concatenate([board_planes, goal_planes.astype(np.float32)], axis=0)
                        deadline = remaining
                    else:
                        planes = to_model_input(encode_board(tree.board))
                        deadline = None
                    self._pop_to_root(tree, path)
                    parked.append((tree, path, planes, legal_idxs, deadline, goal_vec))
```

Update the batched-eval block to handle means-end (3-return + blend) and to unpack the 6-tuple:

```python
        if parked:
            planes_batch = np.stack([p[2] for p in parked])
            if self.meansend:
                goal_vecs = np.stack([p[5] for p in parked])
                deadlines = np.asarray([p[4] for p in parked], dtype=np.float32)
                policies, v_win, v_goal = self.evaluator.evaluate_planes(
                    planes_batch, goal_vecs, deadlines)
                a = self.cfg.meansend_alpha
                values = (1.0 - a) * v_win + a * (2.0 * v_goal - 1.0)
            elif self.goal_mode:
                deadlines = np.asarray([p[4] for p in parked], dtype=np.float32)
                policies, values = self.evaluator.evaluate_planes(planes_batch, deadlines)
            else:
                policies, values = self.evaluator.evaluate_planes(planes_batch)
            for (tree, path, _planes, legal_idxs, _deadline, _gv), policy, value in zip(
                parked, policies, values
            ):
                self._expand_from_idxs(path[-1], legal_idxs, policy, float(value))
                self._backup(path, float(value))
```

In `_expand_leaf`, add a means-end branch (batch-of-1 dual-head eval + blend) before the existing `if self.goal_mode:`:

```python
        if self.meansend:
            remaining = tree.deadline_origin - plies_from_root
            bp = to_model_input(encode_board(board))[None]
            policies, v_win, v_goal = self.evaluator.evaluate_planes(
                bp, tree.goal_vec[None], np.asarray([remaining], np.float32))
            a = self.cfg.meansend_alpha
            policy = policies[0]
            value = float((1.0 - a) * v_win[0] + a * (2.0 * v_goal[0] - 1.0))
            self._expand_from_idxs(node, idxs, policy, value)
            return value
        if self.goal_mode:
            ...
```

(`_select`, `_backup`, `_terminal_value` are UNCHANGED: with `self.goal_mode=False` they take the vanilla negamax / game-terminal paths automatically — which is exactly means-end semantics.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_meansend_mcts.py -v`
Expected: PASS (all). Then regression:
`.venv\Scripts\python.exe -m pytest tests/test_batched_goal_equivalence.py tests/test_mcts_goal_regression.py tests/test_batched_goal_concurrent.py -v`
Expected: PASS (v1 planes goal-mode + vanilla equivalence unchanged).

- [ ] **Step 5: Commit**

```bash
git add chessrl/mcts/batched.py tests/test_meansend_mcts.py
git commit -m "feat(v2): BatchedMCTS means-end mode (vector-conditioned, blended leaf)"
```

---

## Plan 4a deliverable

`BatchedMCTS(meansend=True)` searches vanilla-negamax over expected terminal reward, conditioned on a goal centroid vector via the dual-head evaluator, with an α-blended leaf value. v1 planes goal-mode and vanilla are unchanged (regression suites green). This is the search Plan 4b's self-play driver uses to generate means-end games.

## Out of scope (4b / 4c)

- 4b: the concurrent means-end self-play driver (GoalSpace cluster assignment + ε-explore, cluster `_SideGoal`/switch-to-win, write cluster records, publish live frames).
- 4c: frozen-encoder snapshot + GoalSpace refresh + `VectorGoalReplayBuffer` + `winvalue.py` + curriculum `γ·win_value` + `train_steps_vector` + I1/M5 carry-forwards + `experiments/v2-stage1.yaml`.
