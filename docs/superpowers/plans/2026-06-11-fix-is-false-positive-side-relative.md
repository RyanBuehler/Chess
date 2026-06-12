# Fix `_is_false_positive` Side-Relative Rule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `_is_false_positive` in `chessrl/selfplay/concurrent.py` so it correctly identifies false positives for both White and Black would-be resigners, using TDD.

**Architecture:** Track `would_resign_side: chess.Color | None` in `_Game` (replacing the boolean `would_resign` tracking internally, while keeping `meta["would_resign"]` as a bool for schema compatibility). Update `_is_false_positive` to check `z >= 0` for White and `z <= 0` for Black. Write the failing test first, then fix.

**Tech Stack:** Python 3.13, python-chess, numpy, pytest

---

### Task 1: Write the failing test

**Files:**
- Modify: `tests/test_concurrent_selfplay.py`

- [ ] **Step 1: Add `BlackIsLostBatchedEvaluator` and failing test**

Add this to `tests/test_concurrent_selfplay.py` after the `WhiteIsLostBatchedEvaluator` class and before the first test function:

```python
class BlackIsLostBatchedEvaluator(UniformBatchedEvaluator):
    """Black-to-move positions evaluate as lost, White's as won.
    Mirrors WhiteIsLostBatchedEvaluator with colours swapped."""

    def evaluate_many(self, boards):
        policies, values = super().evaluate_many(boards)
        values = np.array(
            [1.0 if b.turn == chess.BLACK else -1.0 for b in boards], dtype=np.float32
        )
        return policies, values
```

And add this test at the end of the file:

```python
def test_false_positive_detection_is_side_relative():
    """Regression: fp must be computed relative to the would-be resigner's side.

    With BlackIsLostBatchedEvaluator:
    - root_q is from the side-to-move perspective
    - Black's resign streak fires (Black thinks it's losing)
    - But with resign_playout_fraction=1.0 the game plays out
    - White typically wins (z=+1 from White's view) -> Black genuinely lost
      -> NOT a false positive (fp must be False)
    - If z<=0, Black did not lose -> IS a false positive (fp must be True)

    The bug: old code used `g.z >= 0` regardless of side, so z=+1 with
    Black-would-resign was incorrectly labelled fp=True.
    """
    mcts_cfg = MCTSConfig(simulations=4, temperature_moves=0)
    sp_cfg = SelfPlayConfig(
        ply_cap=100,
        resign_threshold=-0.5,
        resign_consecutive=2,
        resign_playout_fraction=1.0,   # all games are playout games
    )
    results = play_games_concurrent(
        BlackIsLostBatchedEvaluator(), mcts_cfg, sp_cfg,
        np.random.default_rng(42), num_games=4,
    )

    games_with_would_resign = [(z, meta) for _rec, _b, z, meta in results if meta["would_resign"]]
    # There should be at least one game where the resign criterion fired.
    assert games_with_would_resign, "Expected at least one game to hit the resign threshold"

    for z, meta in games_with_would_resign:
        if z == 1:
            # Black genuinely lost (z=+1 means White won) -> NOT a false positive
            assert meta["fp"] is False, (
                f"z=+1 with Black-would-resign must be fp=False (Black lost), got fp=True. "
                f"meta={meta}"
            )
        if z <= 0:
            # Black did not lose (draw or Black win) -> IS a false positive
            assert meta["fp"] is True, (
                f"z={z} with Black-would-resign must be fp=True (Black didn't lose), got fp=False. "
                f"meta={meta}"
            )
```

- [ ] **Step 2: Run the test to confirm it FAILS**

Run: `cd C:\Chess && .\.venv\Scripts\python -m pytest tests/test_concurrent_selfplay.py::test_false_positive_detection_is_side_relative -v`

Expected: FAIL — the assertion `meta["fp"] is False` for `z==1` will fail because the current code returns `g.z >= 0` (True for z=1) regardless of side.

---

### Task 2: Apply the fix to `concurrent.py`

**Files:**
- Modify: `chessrl/selfplay/concurrent.py:22-37` (`_Game.__slots__` and `__init__`)
- Modify: `chessrl/selfplay/concurrent.py:77-84` (meta dict construction)
- Modify: `chessrl/selfplay/concurrent.py:115-122` (`_play_one_move` resign tracking)
- Modify: `chessrl/selfplay/concurrent.py:136-150` (`_is_false_positive`)

- [ ] **Step 3: Update `_Game.__slots__` and `__init__`**

Replace the `__slots__` tuple and `__init__` in `_Game`:

Old `__slots__`:
```python
    __slots__ = (
        "tree", "builder", "board", "allow_resign", "resign_streak",
        "ply", "done", "z", "resigned", "would_resign",
    )
```

New `__slots__` (replace `would_resign` bool with `would_resign_side`):
```python
    __slots__ = (
        "tree", "builder", "board", "allow_resign", "resign_streak",
        "ply", "done", "z", "resigned", "would_resign_side",
    )
```

Old `__init__` last two lines:
```python
        self.resigned = False
        self.would_resign = False
```

New `__init__` last two lines:
```python
        self.resigned = False
        self.would_resign_side = None   # chess.Color of side that would resign, or None
```

- [ ] **Step 4: Update `_play_one_move` to set `would_resign_side`**

Old resign tracking code in `_play_one_move`:
```python
            if g.resign_streak[g.board.turn] >= sp_cfg.resign_consecutive:
                g.would_resign = True
                if g.allow_resign:
```

New code (capture the side, set only on first trigger):
```python
            if g.resign_streak[g.board.turn] >= sp_cfg.resign_consecutive:
                if g.would_resign_side is None:
                    g.would_resign_side = g.board.turn
                if g.allow_resign:
```

- [ ] **Step 5: Update meta dict in `play_games_concurrent`**

Old meta dict:
```python
            "would_resign": g.would_resign,
```

New meta dict (expose as bool for schema compatibility):
```python
            "would_resign": g.would_resign_side is not None,
```

- [ ] **Step 6: Update `_is_false_positive`**

Old function body (lines 145-150):
```python
    if g.allow_resign or not g.would_resign:
        return False
    # Resignation in this engine only ever fires for the side to move; in the
    # WhiteIsLost evaluator that is White. A draw (z==0) or a White win (z==1)
    # both contradict the would-be resignation, i.e. a false positive.
    return g.z >= 0
```

New function body:
```python
    if g.allow_resign or g.would_resign_side is None:
        return False
    # z is always from White's perspective.
    # A false positive is: the would-be resigner did NOT actually lose.
    # White would-resigner: fp when z >= 0 (draw or White win -> White didn't lose).
    # Black would-resigner: fp when z <= 0 (draw or Black win -> Black didn't lose).
    if g.would_resign_side == chess.WHITE:
        return g.z >= 0
    else:
        return g.z <= 0
```

---

### Task 3: Verify all tests pass and commit

**Files:** (no new files)

- [ ] **Step 7: Run the new test to confirm it now PASSES**

Run: `cd C:\Chess && .\.venv\Scripts\python -m pytest tests/test_concurrent_selfplay.py::test_false_positive_detection_is_side_relative -v`

Expected: PASS

- [ ] **Step 8: Run the full test suite**

Run: `cd C:\Chess && .\.venv\Scripts\python -m pytest -v`

Expected: 80 tests collected, all PASS.

- [ ] **Step 9: Commit**

```bash
git add tests/test_concurrent_selfplay.py chessrl/selfplay/concurrent.py
git commit -m "fix: resignation false-positive rule is side-relative"
```
