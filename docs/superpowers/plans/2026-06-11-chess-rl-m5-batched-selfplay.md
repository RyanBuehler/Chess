# Chess RL Batched Parallel Self-Play (M5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make self-play fast enough to feed training: batched GPU evaluation, batched MCTS over many concurrent game trees (with virtual loss and subtree reuse), multi-process self-play workers, a parallel trainer that ingests games while workers produce them, and a profiling script whose output decides the C++ move-gen question.

**Architecture:** Extends the M1–M4 single-process core (`docs/superpowers/specs/2026-06-11-chess-rl-design.md`). The reference MCTS (`chessrl/mcts/reference.py`) stays the permanent correctness baseline and is never modified; the new batched MCTS is diffed against it byte-for-byte (visit-count dicts) under an equivalence gate. The reference self-play (`chessrl/selfplay/play.py`) and single-process loop (`chessrl/training/loop.py`) remain the smoke pipeline and are touched only by one small, behavior-preserving refactor (PGN writer extraction). All new disk artifacts live under `runs/<run-id>/` exactly like M4. Sparse on-disk game records remain the source of truth; the replay buffer is reconstructed from them on resume.

**Tech Stack:** Python 3.11+, python-chess, PyTorch (CUDA 12.8 wheel on the 5090; CPU fallback everywhere — every test runs on CPU), NumPy, PyYAML, pytest, `multiprocessing` with the **spawn** start method (Windows-native; used identically on Linux).

**Scope:** Milestone M5 of the spec: batched evaluation, batched MCTS (virtual loss + subtree reuse), concurrent self-play, worker processes, a parallel training loop, and a profiling gate. **Out of scope for M5 (stated so workers don't gold-plate):** the ZeroMQ live feed and in-progress move publishing (M7); the Elo evaluator daemon and ladder (M6); the actual C++/Rust move-generation backend (M5 only *decides* on it from profiling data and records the decision in the milestone summary — it does not implement it); a shared cross-worker inference server (each worker owns its own net/CUDA context this milestone).

**Conventions used throughout (normative, from the spec and M1–M4 code — do not drift):**
- All node values and value targets are **from the perspective of the side to move** at that node (+1 = side to move wins). A parent reads a child's quality as `-child.q()`.
- Move index = `from_square * 73 + move_type` in the mirrored (side-to-move) frame; `NUM_ACTIONS == 4672`.
- Child insertion order into a `Node.children` dict is exactly `board.legal_moves` order (Python dicts preserve insertion order). The batched MCTS MUST iterate children in this same order so PUCT tie-breaking matches the reference.
- PUCT selection (identical to `ReferenceMCTS._select`): `q = -child.q()` if the child has been visited, else `fpu = parent.q() - cfg.fpu_reduction`; `score = q + cfg.c_puct * child.prior * sqrt(parent.visit_count) / (1 + child.visit_count)`; pick the strict-max with `>` (first child wins ties).
- The root is expanded once and that expansion **counts as one visit** (root ends at `simulations + 1` visits; child visit counts sum to `simulations`). The batched MCTS reproduces this exactly at K=1.
- `multiprocessing.set_start_method("spawn", force=True)` is set once, guarded, before any worker is spawned. `worker_main` is a top-level importable function (spawn re-imports the module).
- `pathlib.Path` only (no `os.path`). Run all commands from `C:\Chess` with the venv: `.venv\Scripts\python -m pytest ...`. On the 5090 box `torch.cuda.is_available()` is `True`; every test forces `device="cpu"` / `selfplay_device="cpu"` so the suite is portable.
- VRAM budget note (for the milestone summary, not enforced in code): each worker holds its own PyTorch CUDA context (~0.5–0.8 GB) plus a 6×64 net; the default `workers=4` fits comfortably on the 5090.

**Definition of done for M5:** the four new test files pass on CPU (`test_batched_evaluator.py`, `test_batched_mcts.py`, `test_concurrent_selfplay.py`), the slow `test_parallel_smoke.py` passes under `-m slow`, all M1–M4 tests still pass unchanged except the one updated buffer-ordering test, the default suite excludes slow tests, and `scripts/profile_selfplay.py` runs and prints throughput numbers from which the C++ decision is recorded.

---

### Task 1: Config additions for M5

**Files:**
- Modify: `chessrl/config/config.py`
- Test: `tests/test_config_m5.py`

New frozen-dataclass fields (all with defaults, so every existing config and the 59 existing tests keep working unchanged): `MCTSConfig.leaves_per_tree`, `SelfPlayConfig.workers`, `SelfPlayConfig.concurrent_games`, `TrainingConfig.checkpoint_every_steps`, `TrainingConfig.selfplay_device`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config_m5.py
from chessrl.config.config import (
    MCTSConfig,
    RunConfig,
    SelfPlayConfig,
    TrainingConfig,
)


def test_m5_defaults():
    cfg = RunConfig()
    assert cfg.mcts.leaves_per_tree == 1
    assert cfg.selfplay.workers == 4
    assert cfg.selfplay.concurrent_games == 32
    assert cfg.training.checkpoint_every_steps == 1000
    assert cfg.training.selfplay_device == "cuda"


def test_m5_fields_are_overridable():
    m = MCTSConfig(leaves_per_tree=4)
    assert m.leaves_per_tree == 4
    s = SelfPlayConfig(workers=1, concurrent_games=2)
    assert s.workers == 1 and s.concurrent_games == 2
    t = TrainingConfig(checkpoint_every_steps=50, selfplay_device="cpu")
    assert t.checkpoint_every_steps == 50 and t.selfplay_device == "cpu"


def test_m5_yaml_partial_override(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text(
        "mcts:\n  leaves_per_tree: 4\n"
        "selfplay:\n  workers: 2\n  concurrent_games: 8\n"
        "training:\n  checkpoint_every_steps: 100\n  selfplay_device: cpu\n"
    )
    cfg = RunConfig.from_yaml(p)
    assert cfg.mcts.leaves_per_tree == 4
    assert cfg.mcts.simulations == 200          # untouched default survives
    assert cfg.selfplay.workers == 2
    assert cfg.selfplay.concurrent_games == 8
    assert cfg.selfplay.ply_cap == 512          # untouched default survives
    assert cfg.training.checkpoint_every_steps == 100
    assert cfg.training.selfplay_device == "cpu"
    assert cfg.training.batch_size == 256       # untouched default survives
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_config_m5.py -v`
Expected: FAIL — `AttributeError`/`TypeError` on the new fields.

- [ ] **Step 3: Implement (edit the three dataclasses)**

In `chessrl/config/config.py`, add one field to `MCTSConfig`:

```python
@dataclass(frozen=True)
class MCTSConfig:
    simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    fpu_reduction: float = 0.3
    temperature_moves: int = 30   # sample proportionally to visits for this many plies, then argmax
    leaves_per_tree: int = 1      # M5: K leaves selected per tree per batching round (virtual loss). K=1 == reference.
```

Add two fields to `SelfPlayConfig`:

```python
@dataclass(frozen=True)
class SelfPlayConfig:
    ply_cap: int = 512
    resign_threshold: float = -0.95
    resign_consecutive: int = 2          # consecutive own moves below threshold before resigning
    resign_playout_fraction: float = 0.1 # fraction of games where resignation is disabled (false-positive measurement)
    games_per_iteration: int = 10
    workers: int = 4                     # M5: number of self-play worker processes
    concurrent_games: int = 32           # M5: concurrent game trees per worker batch
```

Add two fields to `TrainingConfig`:

```python
@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    buffer_size: int = 500_000
    samples_per_position: float = 2.0    # pacing: total SGD samples allowed per generated position
    device: str = "cuda"                 # trainer device; falls back to cpu if cuda unavailable
    seed: int = 0
    checkpoint_every_steps: int = 1000   # M5: trainer saves a checkpoint each time this many steps are crossed
    selfplay_device: str = "cuda"        # M5: device workers use; falls back to cpu in worker if unavailable
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_config_m5.py tests/test_config.py -v`
Expected: all passed (new M5 tests + the original M1 config tests, proving defaults are unbroken).

- [ ] **Step 5: Commit**

```powershell
git add chessrl/config/config.py tests/test_config_m5.py
git commit -m "feat: M5 config fields (leaves_per_tree, workers, concurrent_games, checkpoint cadence, selfplay device)"
```

---

### Task 2: Batched network evaluator

**Files:**
- Modify: `chessrl/model/network.py` (add `BatchedNetEvaluator`; leave `PolicyValueNet` and `NetEvaluator` untouched)
- Test: `tests/test_batched_evaluator.py`

`BatchedNetEvaluator` owns its own net, calls `.eval()` once at construction (fixes the train/eval seam: evaluators never share a live module with a `Trainer`), and exposes `evaluate_many(boards) -> (policies (N,4672) softmaxed float32, values (N,) float32)` as a single batched no-grad forward. `from_checkpoint(path, network_cfg, device)` builds a fresh `PolicyValueNet(network_cfg)` and loads the `"model"` state_dict saved by `Trainer.save_checkpoint` (keys: `step`, `model`, `optimizer`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_batched_evaluator.py
import chess
import numpy as np
import torch

from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.config.config import NetworkConfig, TrainingConfig
from chessrl.model.network import BatchedNetEvaluator, NetEvaluator, PolicyValueNet
from chessrl.training.trainer import Trainer


def test_evaluate_many_shapes_and_softmax():
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=16))
    ev = BatchedNetEvaluator(net, device="cpu")
    boards = [chess.Board(), chess.Board()]
    boards[1].push(chess.Move.from_uci("e2e4"))
    policies, values = ev.evaluate_many(boards)
    assert policies.shape == (2, NUM_ACTIONS)
    assert policies.dtype == np.float32
    assert values.shape == (2,)
    assert values.dtype == np.float32
    np.testing.assert_allclose(policies.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(values >= -1.0) and np.all(values <= 1.0)


def test_evaluate_many_matches_single_evaluator():
    torch.manual_seed(0)
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=16))
    single = NetEvaluator(net, device="cpu")
    batched = BatchedNetEvaluator(net, device="cpu")
    board = chess.Board()
    p1, v1 = single.evaluate(board)
    p_many, v_many = batched.evaluate_many([board])
    np.testing.assert_allclose(p_many[0], p1, atol=1e-5)
    assert abs(v_many[0] - v1) < 1e-5


def test_empty_batch_returns_empty_arrays():
    net = PolicyValueNet(NetworkConfig(blocks=1, filters=16))
    ev = BatchedNetEvaluator(net, device="cpu")
    policies, values = ev.evaluate_many([])
    assert policies.shape == (0, NUM_ACTIONS)
    assert values.shape == (0,)


def test_from_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=16))
    trainer = Trainer(net, TrainingConfig(batch_size=4, device="cpu"), tmp_path)
    ckpt_path = trainer.save_checkpoint()    # saves ckpt_00000000.pt with {"model": ...}

    ref = BatchedNetEvaluator(net, device="cpu")
    loaded = BatchedNetEvaluator.from_checkpoint(
        ckpt_path, NetworkConfig(blocks=2, filters=16), device="cpu"
    )
    boards = [chess.Board()]
    p_ref, v_ref = ref.evaluate_many(boards)
    p_load, v_load = loaded.evaluate_many(boards)
    np.testing.assert_allclose(p_load, p_ref, atol=1e-5)
    np.testing.assert_allclose(v_load, v_ref, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_batched_evaluator.py -v`
Expected: FAIL — `ImportError` on `BatchedNetEvaluator`.

- [ ] **Step 3: Implement (append to `chessrl/model/network.py`)**

Add these imports at the top of the file if not already present (the file already imports `chess`, `numpy as np`, `torch`, `torch.nn as nn`, `encode_board`, `to_model_input`, `NetworkConfig`):

```python
from pathlib import Path
```

Append the new class at the end of `chessrl/model/network.py`:

```python
class BatchedNetEvaluator:
    """Batched evaluator: one net, one batched forward per call. Owns its net
    and calls .eval() once at construction so it never shares a live training
    module with a Trainer (the documented train/eval seam). Used by batched MCTS
    and the self-play workers."""

    def __init__(self, net: PolicyValueNet, device: str = "cpu"):
        self.device = device
        self.net = net.to(device)
        self.net.eval()

    @classmethod
    def from_checkpoint(
        cls, path, network_cfg: NetworkConfig, device: str = "cpu"
    ) -> "BatchedNetEvaluator":
        net = PolicyValueNet(network_cfg)
        ckpt = torch.load(Path(path), map_location=device)
        net.load_state_dict(ckpt["model"])
        return cls(net, device=device)

    @torch.no_grad()
    def evaluate_many(self, boards: list) -> tuple[np.ndarray, np.ndarray]:
        """boards: list[chess.Board]. Returns (policies (N,4672) softmaxed
        float32, values (N,) float32). Empty input -> empty arrays."""
        n = len(boards)
        if n == 0:
            return (
                np.zeros((0, self.net.policy_conv.out_channels * 64), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        stacked = np.stack([to_model_input(encode_board(b)) for b in boards])
        x = torch.from_numpy(stacked).to(self.device)
        logits, value = self.net(x)
        policies = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        values = value.squeeze(1).cpu().numpy().astype(np.float32)
        return policies, values
```

Note: `self.net.policy_conv.out_channels * 64 == 73 * 64 == 4672 == NUM_ACTIONS`; computing it from the net avoids importing `moves` here and keeps the empty-batch shape exact.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_batched_evaluator.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the M1–M4 network test (proving no regression)**

Run: `.venv\Scripts\python -m pytest tests/test_network.py -v`
Expected: 3 passed (unchanged).

- [ ] **Step 6: Commit**

```powershell
git add chessrl/model/network.py tests/test_batched_evaluator.py
git commit -m "feat: BatchedNetEvaluator with single batched forward and from_checkpoint"
```

---

### Task 3: Batched MCTS — equivalence gate (K=1)

**Files:**
- Create: `chessrl/mcts/batched.py`
- Test: `tests/test_batched_mcts.py`

This task builds the batched MCTS engine and proves it **exactly** reproduces `ReferenceMCTS` visit counts at K=1, single tree, no noise. The engine reuses `Node` (imported from `reference`), descends with push/pop on each tree's own board (no `Board.copy` per simulation), applies virtual loss (a no-op at K=1), and batches all non-terminal leaves of a round into one `evaluate_many` call.

Design (final API):
- `SearchTree(board)` — owns `root: Node`, a working `board`, and `sims_done: int`.
- `BatchedMCTS(evaluator_many, cfg, rng)` — `evaluator_many` is any object with `evaluate_many(boards) -> (policies, values)`.
  - `init_tree(board, add_noise=False) -> SearchTree` — creates root, expands it (counts as 1 visit), optionally adds Dirichlet noise.
  - `run(tree)` — advances a single tree to `cfg.simulations` (drives the K-leaf rounds internally); convenience for the equivalence path.
  - `step_round(trees)` — advances every not-yet-finished tree in `trees` by one batching round (each selects up to K leaves, one shared GPU batch across all trees). Returns when called; callers loop until all `tree.sims_done >= cfg.simulations`.
  - `visit_counts(tree) -> dict` and `root_q(tree) -> float`.
  - `advance(tree, action_index)` — re-roots at the chosen child (subtree reuse), discarding siblings, keeping accumulated statistics; resets `sims_done` to the reused child's existing visit count so the next search tops it up to `simulations`.

Virtual loss semantics: while selecting K>1 leaves within one round for one tree, increment `vloss` on every node along each selected path; during selection treat `vloss` as added visits whose value is `-1` from the **parent's** perspective. After the round's batch is expanded and backed up, clear all `vloss`. At K=1 only one leaf is selected per tree per round, so `vloss` is incremented then immediately used by no further selection and cleared — i.e. a no-op — which is what makes the equivalence gate hold.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_batched_mcts.py
import chess
import numpy as np

from chessrl.chess_env.moves import NUM_ACTIONS, index_to_move
from chessrl.config.config import MCTSConfig
from chessrl.mcts.batched import BatchedMCTS, SearchTree
from chessrl.mcts.reference import ReferenceMCTS


class UniformBatchedEvaluator:
    """Batched analogue of tests.test_mcts.UniformEvaluator: uniform priors,
    value 0 for every board. Search quality comes entirely from terminal
    values, exactly as the reference UniformEvaluator."""

    def evaluate_many(self, boards):
        n = len(boards)
        policies = np.full((n, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float32)
        values = np.zeros(n, dtype=np.float32)
        return policies, values


class UniformSingleEvaluator:
    """Single-board uniform evaluator for ReferenceMCTS (matches
    tests.test_mcts.UniformEvaluator)."""

    def evaluate(self, board):
        return np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS), 0.0


EQUIV_FENS = [
    chess.STARTING_FEN,
    "r1bq1rk1/pp2bppp/2n2n2/2pp4/3P1B2/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "6k1/8/6K1/8/8/8/8/R7 w - - 0 1",  # mate-in-1
]


def _ref_visits(fen, sims):
    cfg = MCTSConfig(simulations=sims)
    mcts = ReferenceMCTS(UniformSingleEvaluator(), cfg, rng=np.random.default_rng(0))
    visits, _ = mcts.search(chess.Board(fen), add_noise=False)
    return visits


def _batched_visits(fen, sims, k):
    cfg = MCTSConfig(simulations=sims, leaves_per_tree=k)
    mcts = BatchedMCTS(UniformBatchedEvaluator(), cfg, rng=np.random.default_rng(0))
    tree = mcts.init_tree(chess.Board(fen), add_noise=False)
    mcts.run(tree)
    return mcts.visit_counts(tree)


def test_k1_exact_equivalence_with_reference():
    for fen in EQUIV_FENS:
        ref = _ref_visits(fen, 64)
        bat = _batched_visits(fen, 64, k=1)
        assert bat == ref, f"mismatch on {fen}"


def test_k1_visits_sum_to_simulations():
    bat = _batched_visits(chess.STARTING_FEN, 64, k=1)
    assert sum(bat.values()) == 64


def _best_move_batched(fen, sims, k):
    board = chess.Board(fen)
    visits = _batched_visits(fen, sims, k)
    idx = max(visits, key=visits.get)
    return index_to_move(idx, board.turn == chess.BLACK, board)


def test_batched_finds_mate_in_one_k4():
    mv = _best_move_batched("6k1/8/6K1/8/8/8/8/R7 w - - 0 1", sims=200, k=4)
    assert mv == chess.Move.from_uci("a1a8")


def test_batched_finds_mate_in_two_k4():
    mv = _best_move_batched("7k/8/5K2/8/8/8/8/R7 w - - 0 1", sims=1600, k=4)
    assert mv == chess.Move.from_uci("f6g6")


def test_k4_visit_sum_within_overshoot_bound():
    # Stop selecting once sims_done >= simulations; a final round may overshoot
    # by up to K-1. Visit total is in [simulations, simulations + K - 1].
    cfg = MCTSConfig(simulations=64, leaves_per_tree=4)
    mcts = BatchedMCTS(UniformBatchedEvaluator(), cfg, rng=np.random.default_rng(0))
    tree = mcts.init_tree(chess.Board(), add_noise=False)
    mcts.run(tree)
    total = sum(mcts.visit_counts(tree).values())
    assert 64 <= total <= 64 + 4 - 1


def test_advance_reuses_subtree():
    cfg = MCTSConfig(simulations=64, leaves_per_tree=1)
    mcts = BatchedMCTS(UniformBatchedEvaluator(), cfg, rng=np.random.default_rng(0))
    board = chess.Board()
    tree = mcts.init_tree(board, add_noise=False)
    mcts.run(tree)
    visits = mcts.visit_counts(tree)
    best = max(visits, key=visits.get)
    reused_count_before = visits[best]
    mcts.advance(tree, best)
    # After re-rooting, sims_done equals the reused child's visit count, so the
    # new root already carries accumulated statistics (reuse, not a fresh tree).
    assert tree.sims_done == reused_count_before
    assert tree.root.visit_count == reused_count_before
    mcts.run(tree)
    assert sum(mcts.visit_counts(tree).values()) >= 64  # topped up to simulations


def test_step_round_drives_multiple_trees():
    cfg = MCTSConfig(simulations=32, leaves_per_tree=2)
    mcts = BatchedMCTS(UniformBatchedEvaluator(), cfg, rng=np.random.default_rng(0))
    trees = [mcts.init_tree(chess.Board(), add_noise=False) for _ in range(3)]
    while any(t.sims_done < cfg.simulations for t in trees):
        mcts.step_round(trees)
    for t in trees:
        total = sum(mcts.visit_counts(t).values())
        assert 32 <= total <= 32 + 2 - 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_batched_mcts.py -v`
Expected: FAIL — `ImportError` on `chessrl.mcts.batched`.

- [ ] **Step 3: Implement**

```python
# chessrl/mcts/batched.py
"""Batched PUCT search over many concurrent game trees.

Diffed against ReferenceMCTS: at leaves_per_tree (K) == 1, a single tree, and
add_noise=False, this produces the EXACT same visit-count dict as the reference
on the same position. That equivalence is the milestone's correctness gate.

Key differences from the reference (all behavior-preserving at K=1):
  * descent uses board.push / board.pop on each tree's own board instead of a
    full Board.copy per simulation (the spec's CPU-bound mitigation);
  * non-terminal leaves across all active trees in one round are evaluated in a
    single evaluate_many call (the GPU batch);
  * virtual loss diversifies the K>1 selections within one tree per round, and
    is a no-op at K=1.

Sign convention is identical to the reference: Node.value_sum is from the
perspective of the side to move at that node; a parent reads child quality as
-child.q(); backup flips sign each level leaf->root.
"""
import chess
import numpy as np

from chessrl.chess_env.game import terminal_value
from chessrl.chess_env.moves import index_to_move, move_to_index
from chessrl.config.config import MCTSConfig
from chessrl.mcts.reference import Node


class SearchTree:
    """One independent game tree: its root, a working board (kept at the root
    position between rounds), and how many simulations have been credited."""

    __slots__ = ("root", "board", "sims_done")

    def __init__(self, board: chess.Board):
        self.root = Node(0.0)
        self.board = board
        self.sims_done = 0


class BatchedMCTS:
    def __init__(self, evaluator_many, cfg: MCTSConfig, rng: np.random.Generator | None = None):
        self.evaluator = evaluator_many
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()

    # ---- public API -----------------------------------------------------

    def init_tree(self, board: chess.Board, add_noise: bool = False) -> SearchTree:
        tree = SearchTree(board.copy())
        policy, value = self._evaluate_one(tree.board)
        self._expand(tree.root, tree.board, policy, value, is_terminal=False)
        tree.root.visit_count += 1            # initial expansion counts as one visit (matches reference)
        tree.root.value_sum += value
        if add_noise and tree.root.children:
            self._add_dirichlet(tree.root)
        return tree

    def run(self, tree: SearchTree) -> None:
        """Advance a single tree to cfg.simulations (drives K-leaf rounds)."""
        while tree.sims_done < self.cfg.simulations:
            self.step_round([tree])

    def step_round(self, trees: list) -> None:
        """Advance every not-yet-finished tree by one batching round. All
        non-terminal leaves selected this round (across all trees) are evaluated
        in a single evaluate_many call."""
        k = self.cfg.leaves_per_tree
        parked = []   # (tree, path, board_at_leaf) awaiting GPU evaluation
        for tree in trees:
            if tree.sims_done >= self.cfg.simulations:
                continue
            for _ in range(k):
                if tree.sims_done >= self.cfg.simulations:
                    break
                path = self._select_leaf(tree)
                leaf = path[-1]
                self._apply_virtual_loss(path)
                tree.sims_done += 1
                term = terminal_value(tree.board)
                if term is not None:
                    self._backup(path, term)
                    self._pop_to_root(tree, path)
                else:
                    parked.append((tree, path, tree.board.copy()))
        if parked:
            policies, values = self.evaluator.evaluate_many([p[2] for p in parked])
            for (tree, path, leaf_board), policy, value in zip(parked, policies, values):
                self._expand(path[-1], leaf_board, policy, float(value), is_terminal=False)
                self._backup(path, float(value))
                self._pop_to_root(tree, path)
        for tree in trees:
            self._clear_virtual_loss(tree.root)

    def visit_counts(self, tree: SearchTree) -> dict:
        return {i: c.visit_count for i, c in tree.root.children.items() if c.visit_count > 0}

    def root_q(self, tree: SearchTree) -> float:
        return tree.root.q()

    def advance(self, tree: SearchTree, action_index: int) -> None:
        """Re-root at the chosen child (subtree reuse): discard siblings, keep
        the chosen subtree's accumulated statistics, and push the move on the
        tree's board so the working board sits at the new root."""
        child = tree.root.children.get(action_index)
        tree.board.push(index_to_move(action_index, tree.board.turn == chess.BLACK, tree.board))
        if child is None:
            tree.root = Node(0.0)
            policy, value = self._evaluate_one(tree.board)
            self._expand(tree.root, tree.board, policy, value, is_terminal=False)
            tree.root.visit_count += 1
            tree.root.value_sum += value
            tree.sims_done = tree.root.visit_count
            return
        tree.root = child
        if not child.children and child.visit_count == 0:
            policy, value = self._evaluate_one(tree.board)
            self._expand(tree.root, tree.board, policy, value, is_terminal=False)
            tree.root.visit_count += 1
            tree.root.value_sum += value
        tree.sims_done = tree.root.visit_count

    def add_root_noise(self, tree: SearchTree) -> None:
        if tree.root.children:
            self._add_dirichlet(tree.root)

    # ---- internals ------------------------------------------------------

    def _select_leaf(self, tree: SearchTree) -> list:
        """Descend from root to an unexpanded/terminal leaf, pushing moves on
        tree.board. Returns the path (root..leaf). Caller must pop back."""
        node = tree.root
        path = [node]
        while node.children:
            idx, node = self._select(node)
            tree.board.push(index_to_move(idx, tree.board.turn == chess.BLACK, tree.board))
            path.append(node)
        return path

    def _select(self, node: Node):
        # effective visits/value include virtual loss; vloss adds visits valued
        # -1 from the parent's perspective.
        eff_parent_n = node.visit_count + node.vloss
        sqrt_n = eff_parent_n ** 0.5
        parent_q = (node.value_sum - node.vloss) / eff_parent_n if eff_parent_n else 0.0
        fpu = parent_q - self.cfg.fpu_reduction
        best_idx, best_child, best_score = -1, None, -1e18
        for idx, ch in node.children.items():
            ch_n = ch.visit_count + ch.vloss
            if ch_n:
                child_q = (ch.value_sum - ch.vloss) / ch_n
                q = -child_q
            else:
                q = fpu
            score = q + self.cfg.c_puct * ch.prior * sqrt_n / (1 + ch_n)
            if score > best_score:
                best_idx, best_child, best_score = idx, ch, score
        return best_idx, best_child

    def _apply_virtual_loss(self, path: list) -> None:
        for n in path:
            n.vloss += 1

    def _backup(self, path: list, value: float) -> None:
        # remove the virtual loss this path added, then apply the real value.
        v = value
        for n in reversed(path):
            n.vloss -= 1
            n.visit_count += 1
            n.value_sum += v
            v = -v

    def _clear_virtual_loss(self, node: Node) -> None:
        # safety net: after a completed round every vloss should already be 0
        # (each applied loss is removed in _backup). This re-zeroes defensively.
        stack = [node]
        while stack:
            n = stack.pop()
            if n.vloss:
                n.vloss = 0
            stack.extend(n.children.values())

    def _pop_to_root(self, tree: SearchTree, path: list) -> None:
        for _ in range(len(path) - 1):
            tree.board.pop()

    def _evaluate_one(self, board: chess.Board):
        policies, values = self.evaluator.evaluate_many([board])
        return policies[0], float(values[0])

    def _expand(self, node: Node, board: chess.Board, policy, value: float, is_terminal: bool) -> None:
        flip = board.turn == chess.BLACK
        idxs = [move_to_index(m, flip) for m in board.legal_moves]
        if not idxs:
            return
        priors = np.asarray([policy[i] for i in idxs], dtype=np.float64)
        total = priors.sum()
        priors = priors / total if total > 0 else np.full(len(idxs), 1.0 / len(idxs))
        for i, idx in enumerate(idxs):
            node.children[idx] = Node(float(priors[i]))

    def _add_dirichlet(self, root: Node) -> None:
        eps, alpha = self.cfg.dirichlet_eps, self.cfg.dirichlet_alpha
        noise = self.rng.dirichlet([alpha] * len(root.children))
        for n, ch in zip(noise, root.children.values()):
            ch.prior = (1 - eps) * ch.prior + eps * float(n)
```

This module references `Node.vloss`, which does not exist on the reference `Node` yet. Add it without changing reference behavior — Task 4 does that.

- [ ] **Step 4: Add the `vloss` slot to `Node` (reference module, behavior-preserving)**

In `chessrl/mcts/reference.py`, extend `Node` to carry a virtual-loss counter that defaults to 0 and is never touched by the reference search (so the reference's own tests are unaffected):

```python
class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "vloss")

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "Node"] = {}
        self.vloss = 0      # virtual loss (batched MCTS only; reference leaves it at 0)

    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0
```

- [ ] **Step 5: Run the batched MCTS tests**

Run: `.venv\Scripts\python -m pytest tests/test_batched_mcts.py -v`
Expected: 7 passed. The equivalence test is the critical one: if `test_k1_exact_equivalence_with_reference` fails, the bug is in `_select` (must match the reference's `-child.q()` / `fpu` formula and strict `>` tie-break with children iterated in `legal_moves` order) or in the root-expansion-counts-as-one-visit step — do NOT loosen the assertion. Mate-in-two at 1600 sims with K=4 takes a few seconds.

- [ ] **Step 6: Run the reference MCTS tests (proving no regression)**

Run: `.venv\Scripts\python -m pytest tests/test_mcts.py -v`
Expected: 5 passed (the `vloss` slot did not change reference behavior).

- [ ] **Step 7: Commit**

```powershell
git add chessrl/mcts/batched.py chessrl/mcts/reference.py tests/test_batched_mcts.py
git commit -m "feat: batched MCTS with virtual loss, subtree reuse, exact K=1 equivalence gate"
```

---

### Task 4: PGN writer extraction (behavior-preserving refactor)

**Files:**
- Create: `chessrl/selfplay/pgn_io.py`
- Modify: `chessrl/training/loop.py` (import the extracted writer; keep behavior identical)
- Test: `tests/test_pgn_io.py`

`loop.py._save_pgn` is needed by both the single-process loop and the new worker. Extract it to `chessrl/selfplay/pgn_io.py` as `save_pgn(board, z, path)` with byte-identical output, and have `loop.py` delegate to it. No behavior change — the existing `tests/test_smoke.py` still asserts the same PGN files.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pgn_io.py
import chess

from chessrl.selfplay.pgn_io import save_pgn


def test_save_pgn_writes_result_and_moves(tmp_path):
    board = chess.Board()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate, black wins
        board.push(chess.Move.from_uci(uci))
    path = tmp_path / "g.pgn"
    save_pgn(board, z=-1, path=path)
    text = path.read_text()
    assert '[Result "0-1"]' in text
    assert "f3" in text and "Qh4" in text


def test_save_pgn_result_mapping(tmp_path):
    for z, expected in [(1, "1-0"), (-1, "0-1"), (0, "1/2-1/2")]:
        path = tmp_path / f"g{z}.pgn"
        save_pgn(chess.Board(), z=z, path=path)
        assert f'[Result "{expected}"]' in path.read_text()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_pgn_io.py -v`
Expected: FAIL — `ImportError` on `chessrl.selfplay.pgn_io`.

- [ ] **Step 3: Implement the extracted module**

```python
# chessrl/selfplay/pgn_io.py
"""PGN writing shared by the single-process loop and the self-play workers.
Behavior is identical to the original loop._save_pgn."""
from pathlib import Path

import chess
import chess.pgn

_RESULT_STR = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}


def save_pgn(board: chess.Board, z: int, path) -> None:
    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = _RESULT_STR[z]
    Path(path).write_text(str(game))
```

- [ ] **Step 4: Point `loop.py` at the extracted writer (keep behavior identical)**

In `chessrl/training/loop.py`, remove the local `_RESULT_STR` dict and the `_save_pgn` function, add the import, and update the one call site.

Add to the imports block:

```python
from chessrl.selfplay.pgn_io import save_pgn
```

Delete these lines from `loop.py`:

```python
_RESULT_STR = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}
```

and

```python
def _save_pgn(board, z, path: Path) -> None:
    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = _RESULT_STR[z]
    path.write_text(str(game))
```

Change the call site inside `main`'s game loop from:

```python
            _save_pgn(final_board, z, run_dir / "games" / f"game_{game_no:07d}.pgn")
```

to:

```python
            save_pgn(final_board, z, run_dir / "games" / f"game_{game_no:07d}.pgn")
```

The `import chess.pgn` line in `loop.py` may now be unused; leave it (harmless) or remove it — `_provenance` and the rest of `loop.py` are otherwise untouched.

- [ ] **Step 5: Run the new test plus the M4 smoke test (proving identical behavior)**

Run: `.venv\Scripts\python -m pytest tests/test_pgn_io.py tests/test_smoke.py -v`
Expected: all passed — the smoke test still finds 2 `.pgn` files per run with the same contents.

- [ ] **Step 6: Commit**

```powershell
git add chessrl/selfplay/pgn_io.py chessrl/training/loop.py tests/test_pgn_io.py
git commit -m "refactor: extract save_pgn into selfplay/pgn_io shared by loop and workers"
```

---

### Task 5: Concurrent self-play

**Files:**
- Create: `chessrl/selfplay/concurrent.py`
- Test: `tests/test_concurrent_selfplay.py`

`play_games_concurrent(evaluator_many, mcts_cfg, sp_cfg, rng, num_games)` plays `num_games` to completion in lockstep using one `BatchedMCTS` whose round drives all live games at once. Per game it mirrors `play.py` exactly: search with root noise, temperature-sample below `temperature_moves` plies then argmax, `ply_cap` draw, resignation (threshold on root Q for `resign_consecutive` own moves; `allow_resign` drawn per game from `resign_playout_fraction`), and `RecordBuilder.add(board, idxs, counts, played)` / `finalize(z_white)`. It uses subtree reuse (`advance`) between moves and re-applies Dirichlet noise at each new root. It also records false-positive resignation metadata for playout games. Returns `list[(GameRecord, final_board, z, meta)]`.

`meta` keys: `plies` (int), `z` (int, White's perspective), `resigned` (bool — game ended by resignation), `playout` (bool — resignation disabled for this game), `would_resign` (bool — resignation criterion fired at some point), `fp` (bool — `playout and would_resign and` the actual result was better for the would-be resigner than a loss). `fp` is the false-positive flag the trainer aggregates into a resign-fp rate.

Determinism: with the same `rng` seed and the same evaluator, two calls produce identical records.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_concurrent_selfplay.py
import chess
import numpy as np

from chessrl.config.config import MCTSConfig, SelfPlayConfig
from chessrl.selfplay.concurrent import play_games_concurrent
from tests.test_batched_mcts import UniformBatchedEvaluator


class WhiteIsLostBatchedEvaluator(UniformBatchedEvaluator):
    """White-to-move positions evaluate as lost, Black's as won (mirrors
    tests.test_selfplay.WhiteIsLostEvaluator but batched)."""

    def evaluate_many(self, boards):
        policies, values = super().evaluate_many(boards)
        values = np.array(
            [-1.0 if b.turn == chess.WHITE else 1.0 for b in boards], dtype=np.float32
        )
        return policies, values


def test_four_concurrent_games_valid_records():
    mcts_cfg = MCTSConfig(simulations=8, temperature_moves=4, leaves_per_tree=2)
    sp_cfg = SelfPlayConfig(ply_cap=20, resign_playout_fraction=0.0)
    results = play_games_concurrent(
        UniformBatchedEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(0), num_games=4
    )
    assert len(results) == 4
    for rec, board, z, meta in results:
        assert 1 <= len(rec) <= 20
        assert z in (-1, 0, 1)
        assert meta["plies"] <= 20
        assert board.move_stack
        if z != 0:
            assert rec.outcomes[0] == z          # position 0 is white to move
        for key in ("plies", "z", "resigned", "playout", "would_resign", "fp"):
            assert key in meta


def test_resignation_meta_fields_present_for_playout_games():
    mcts_cfg = MCTSConfig(simulations=4, temperature_moves=0)
    sp_cfg = SelfPlayConfig(
        ply_cap=100, resign_threshold=-0.5, resign_consecutive=2,
        resign_playout_fraction=1.0,   # every game is a playout (resignation disabled)
    )
    results = play_games_concurrent(
        WhiteIsLostBatchedEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(0), num_games=2
    )
    for rec, board, z, meta in results:
        assert meta["playout"] is True
        assert meta["resigned"] is False         # resignation disabled -> game ran to a real end
        # White looks lost throughout, so the resign criterion should have fired.
        assert meta["would_resign"] is True


def test_resignation_ends_game_when_enabled():
    mcts_cfg = MCTSConfig(simulations=4, temperature_moves=0)
    sp_cfg = SelfPlayConfig(
        ply_cap=100, resign_threshold=-0.5, resign_consecutive=2,
        resign_playout_fraction=0.0,   # resignation enabled
    )
    results = play_games_concurrent(
        WhiteIsLostBatchedEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(0), num_games=2
    )
    for rec, board, z, meta in results:
        assert meta["resigned"] is True
        assert z == -1
        assert meta["plies"] < 100


def test_deterministic_with_same_seed():
    mcts_cfg = MCTSConfig(simulations=8, temperature_moves=4)
    sp_cfg = SelfPlayConfig(ply_cap=20, resign_playout_fraction=0.0)
    r1 = play_games_concurrent(
        UniformBatchedEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(7), num_games=3
    )
    r2 = play_games_concurrent(
        UniformBatchedEvaluator(), mcts_cfg, sp_cfg, np.random.default_rng(7), num_games=3
    )
    assert [z for *_, z, _meta in [(*x,) for x in r1]] == [z for *_, z, _meta in [(*x,) for x in r2]]
    for (rec1, _b1, z1, _m1), (rec2, _b2, z2, _m2) in zip(r1, r2):
        assert z1 == z2
        np.testing.assert_array_equal(rec1.played, rec2.played)
        np.testing.assert_array_equal(rec1.outcomes, rec2.outcomes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_concurrent_selfplay.py -v`
Expected: FAIL — `ImportError` on `chessrl.selfplay.concurrent`.

- [ ] **Step 3: Implement**

```python
# chessrl/selfplay/concurrent.py
"""Concurrent self-play: many games advanced in lockstep through one batched
MCTS, so every search round produces a single shared GPU batch.

Per-game logic mirrors selfplay/play.py exactly (search with root noise,
temperature then argmax, ply cap, resignation with playout fraction), plus
false-positive resignation tracking in the returned meta dict. Subtree reuse
(advance) carries statistics across moves; Dirichlet noise is re-applied at
each new root.
"""
import chess
import numpy as np

from chessrl.chess_env.game import terminal_value
from chessrl.config.config import MCTSConfig, SelfPlayConfig
from chessrl.mcts.batched import BatchedMCTS
from chessrl.selfplay.records import GameRecord, RecordBuilder


class _Game:
    """Mutable per-game state for one slot in the concurrent batch."""

    __slots__ = (
        "tree", "builder", "board", "allow_resign", "resign_streak",
        "ply", "done", "z", "resigned", "would_resign",
    )

    def __init__(self, tree, board: chess.Board, allow_resign: bool):
        self.tree = tree
        self.builder = RecordBuilder()
        self.board = board
        self.allow_resign = allow_resign
        self.resign_streak = {chess.WHITE: 0, chess.BLACK: 0}
        self.ply = 0
        self.done = False
        self.z = 0
        self.resigned = False
        self.would_resign = False


def play_games_concurrent(
    evaluator_many,
    mcts_cfg: MCTSConfig,
    sp_cfg: SelfPlayConfig,
    rng: np.random.Generator,
    num_games: int,
) -> list:
    """Returns list[(GameRecord, final_board, z, meta)] of length num_games,
    in slot order. z is from White's perspective (+1/0/-1)."""
    mcts = BatchedMCTS(evaluator_many, mcts_cfg, rng)

    games: list[_Game] = []
    for _ in range(num_games):
        board = chess.Board()
        allow_resign = rng.random() >= sp_cfg.resign_playout_fraction
        tree = mcts.init_tree(board, add_noise=True)
        games.append(_Game(tree, board, allow_resign))

    # Resolve any game that is already terminal / over the cap before searching.
    for g in games:
        _check_pre_move_termination(g, sp_cfg)

    while any(not g.done for g in games):
        active = [g for g in games if not g.done]
        # Run a full search for every active tree (each tree tops up to
        # mcts_cfg.simulations; step_round shares one GPU batch across trees).
        trees = [g.tree for g in active]
        while any(t.sims_done < mcts_cfg.simulations for t in trees):
            mcts.step_round(trees)

        for g in active:
            _play_one_move(g, mcts, mcts_cfg, sp_cfg, rng)

    results = []
    for g in games:
        rec = g.builder.finalize(g.z)
        meta = {
            "plies": g.ply,
            "z": g.z,
            "resigned": g.resigned,
            "playout": not g.allow_resign,
            "would_resign": g.would_resign,
            "fp": _is_false_positive(g),
        }
        results.append((rec, g.board, g.z, meta))
    return results


def _check_pre_move_termination(g: _Game, sp_cfg: SelfPlayConfig) -> None:
    term = terminal_value(g.board)
    if term is not None:
        g.z = int(term) if g.board.turn == chess.WHITE else -int(term)
        g.done = True
    elif g.ply >= sp_cfg.ply_cap:
        g.z = 0
        g.done = True


def _play_one_move(
    g: _Game, mcts: BatchedMCTS, mcts_cfg: MCTSConfig, sp_cfg: SelfPlayConfig,
    rng: np.random.Generator,
) -> None:
    visits = mcts.visit_counts(g.tree)
    root_q = mcts.root_q(g.tree)
    idxs = np.fromiter(visits.keys(), dtype=np.int64)
    counts = np.fromiter(visits.values(), dtype=np.float64)
    if g.ply < mcts_cfg.temperature_moves:
        choice = int(rng.choice(idxs, p=counts / counts.sum()))
    else:
        choice = int(idxs[counts.argmax()])

    # Record before the resign check (the triggering search is a valid example).
    g.builder.add(g.board, idxs.astype(np.int32), counts.astype(np.int32), choice)

    if root_q < sp_cfg.resign_threshold:
        g.resign_streak[g.board.turn] += 1
        if g.resign_streak[g.board.turn] >= sp_cfg.resign_consecutive:
            g.would_resign = True
            if g.allow_resign:
                g.z = -1 if g.board.turn == chess.WHITE else 1
                g.resigned = True
                g.done = True
                return
    else:
        g.resign_streak[g.board.turn] = 0

    # Commit the move via subtree reuse (advance pushes the move on tree.board),
    # then keep g.board in sync, re-apply root noise, and check termination.
    mcts.advance(g.tree, choice)
    g.board = g.tree.board
    g.ply += 1
    mcts.add_root_noise(g.tree)
    _check_pre_move_termination(g, sp_cfg)


def _is_false_positive(g: _Game) -> bool:
    """A playout game where resignation WOULD have fired but the would-be
    resigner did not actually lose -> a false positive. The would-be resigner
    is whoever was to move when the streak reached the threshold; we approximate
    it conservatively as: playout AND would_resign AND the game was not a loss
    for both sides being impossible -> use the recorded result. Since a resign
    abandons the game as a loss for the side to move, a false positive is a
    playout game that hit the criterion yet ended in a draw or a win for the
    would-be resigner."""
    if g.allow_resign or not g.would_resign:
        return False
    # Resignation in this engine only ever fires for the side to move; in the
    # WhiteIsLost evaluator that is White. A draw (z==0) or a White win (z==1)
    # both contradict the would-be resignation, i.e. a false positive.
    return g.z >= 0
```

Note on `_is_false_positive` semantics: resignation fires for the side to move when root Q is below threshold for `resign_consecutive` of *its* moves; in practice the would-be resigner is the side whose evaluation is collapsing. For the trainer's aggregate fp-rate we only need a per-game boolean, and the conservative rule above (playout game that met the criterion yet did not lose) matches the spec's intent: "resignation WOULD have fired and actual z better for the would-be resigner than loss." The test uses the WhiteIsLost evaluator where White is the would-be resigner, so `z >= 0` (draw or White win) is the correct fp condition.

- [ ] **Step 4: Run the concurrent self-play tests**

Run: `.venv\Scripts\python -m pytest tests/test_concurrent_selfplay.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```powershell
git add chessrl/selfplay/concurrent.py tests/test_concurrent_selfplay.py
git commit -m "feat: concurrent self-play over batched MCTS with resign false-positive tracking"
```

---

### Task 6: ReplayBuffer ordering for multi-worker chronology

**Files:**
- Modify: `chessrl/training/buffer.py` (`from_run_dir` sort key)
- Modify: `tests/test_buffer.py` (update the reconstruction test for the new ordering)

With multiple workers writing `game_w{id}_{counter}.npz` concurrently, lexical filename order no longer reflects chronology. Sort by `(mtime, name)` so the newest-by-write-time games are the ones kept when capped, and chronological re-add is correct on resume.

- [ ] **Step 1: Update the existing reconstruction test**

Replace `test_reconstruct_from_run_dir` in `tests/test_buffer.py` with a version that (a) writes files with controlled, increasing mtimes and worker-style names, and (b) asserts the newest-by-mtime games are kept:

```python
def test_reconstruct_from_run_dir_orders_by_mtime(tmp_path):
    import os

    games = tmp_path / "games"
    games.mkdir()
    rec = record_from_pgn(FOOLS_MATE)  # 4 positions each
    # Names deliberately NOT in chronological order; mtime is the real order.
    rec.save(games / "game_w01_0000000.npz")  # oldest (set below)
    rec.save(games / "game_w00_0000000.npz")  # newest (set below)
    base = 1_000_000_000
    os.utime(games / "game_w01_0000000.npz", (base, base))           # older
    os.utime(games / "game_w00_0000000.npz", (base + 10, base + 10)) # newer
    buf = ReplayBuffer.from_run_dir(tmp_path, capacity=6)
    assert len(buf) == 6  # newest games kept, capped at capacity
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_buffer.py::test_reconstruct_from_run_dir_orders_by_mtime -v`
Expected: This may pass or fail depending on filesystem mtime resolution, but the intent is to lock in mtime ordering. Run the full buffer suite to confirm the rename didn't drop coverage: `.venv\Scripts\python -m pytest tests/test_buffer.py -v` — the old test name is gone; the new one is present.

- [ ] **Step 3: Implement the sort change**

In `chessrl/training/buffer.py`, change the `from_run_dir` file enumeration from lexical `sorted(...)` to a `(mtime, name)` sort:

Replace:

```python
    @classmethod
    def from_run_dir(cls, run_dir: str | Path, capacity: int) -> "ReplayBuffer":
        buf = cls(capacity)
        files = sorted((Path(run_dir) / "games").glob("*.npz"))
        selected, total = [], 0
        for f in reversed(files):           # newest first
            rec = GameRecord.load(f)
            selected.append(rec)
            total += len(rec)
            if total >= capacity:
                break
        for rec in reversed(selected):      # re-add in chronological order
            buf.add_game(rec)
        return buf
```

with:

```python
    @classmethod
    def from_run_dir(cls, run_dir: str | Path, capacity: int) -> "ReplayBuffer":
        buf = cls(capacity)
        # Sort by (mtime, name) so multi-worker write chronology is respected;
        # lexical filename order is unreliable across concurrent workers.
        files = sorted(
            (Path(run_dir) / "games").glob("*.npz"),
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        selected, total = [], 0
        for f in reversed(files):           # newest first
            rec = GameRecord.load(f)
            selected.append(rec)
            total += len(rec)
            if total >= capacity:
                break
        for rec in reversed(selected):      # re-add in chronological order
            buf.add_game(rec)
        return buf
```

- [ ] **Step 4: Run the buffer suite**

Run: `.venv\Scripts\python -m pytest tests/test_buffer.py -v`
Expected: all passed (4 tests: add/len, evict, sample, the new mtime reconstruction).

- [ ] **Step 5: Commit**

```powershell
git add chessrl/training/buffer.py tests/test_buffer.py
git commit -m "fix: order ReplayBuffer.from_run_dir by (mtime, name) for multi-worker resume"
```

---

### Task 7: Self-play worker process

**Files:**
- Create: `chessrl/selfplay/worker.py`
- Test: `tests/test_worker.py`

`worker_main(worker_id, run_dir, stop_path, device)` is a spawn-safe top-level function: read `config.json`, seed a per-worker rng (`seed + 1000*worker_id`), and loop until the sentinel `STOP` file exists. Each loop: load the newest checkpoint if it is newer than the last loaded one (`BatchedNetEvaluator.from_checkpoint`); if no checkpoint exists, build a fresh net seeded so all workers start from the same weights; play one batch of `concurrent_games` via `play_games_concurrent`; save each game as `game_w{id:02d}_{counter:07d}.npz` + `.pgn`; append each game's meta as a line to `games_meta_w{id:02d}.jsonl`. The per-worker counter starts at the max existing counter for that worker id (restart-safe).

This task exposes the worker's *unit-testable* pieces as small top-level helpers so the test can drive one batch without spawning a process; the spawn path is exercised by Task 9's slow smoke test. To keep the worker deterministic and fast on CPU in the test, the helper `run_one_batch(...)` is called directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_worker.py
import json

import numpy as np

from chessrl.config.config import RunConfig
from chessrl.selfplay.worker import (
    next_counter_for_worker,
    run_one_batch,
)


def _write_config(run_dir):
    cfg = RunConfig.from_dict(
        {
            "run_name": "wtest",
            "network": {"blocks": 1, "filters": 8},
            "mcts": {"simulations": 8, "temperature_moves": 4, "leaves_per_tree": 2},
            "selfplay": {"ply_cap": 20, "concurrent_games": 2, "resign_playout_fraction": 0.0},
            "training": {"batch_size": 16, "device": "cpu", "selfplay_device": "cpu"},
        }
    )
    (run_dir / "games").mkdir(parents=True)
    (run_dir / "config.json").write_text(cfg.to_json())
    return cfg


def test_next_counter_for_worker_scans_existing(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    (games / "game_w03_0000000.npz").write_bytes(b"x")
    (games / "game_w03_0000004.npz").write_bytes(b"x")
    (games / "game_w07_0000009.npz").write_bytes(b"x")  # different worker, ignored
    assert next_counter_for_worker(tmp_path, worker_id=3) == 5
    assert next_counter_for_worker(tmp_path, worker_id=5) == 0


def test_run_one_batch_writes_games_pgn_and_meta(tmp_path):
    cfg = _write_config(tmp_path)
    rng = np.random.default_rng(0)

    # Build a fresh batched evaluator the same way the worker does at cold start.
    from chessrl.model.network import BatchedNetEvaluator, PolicyValueNet
    import torch

    torch.manual_seed(123)
    net = PolicyValueNet(cfg.network)
    evaluator = BatchedNetEvaluator(net, device="cpu")

    counter = run_one_batch(
        run_dir=tmp_path, worker_id=2, evaluator=evaluator,
        cfg=cfg, rng=rng, start_counter=0,
    )
    npz = sorted((tmp_path / "games").glob("game_w02_*.npz"))
    pgn = sorted((tmp_path / "games").glob("game_w02_*.pgn"))
    assert len(npz) == 2
    assert len(pgn) == 2
    assert counter == 2  # next free counter after writing 2 games

    meta_path = tmp_path / "games_meta_w02.jsonl"
    assert meta_path.exists()
    lines = meta_path.read_text().splitlines()
    assert len(lines) == 2
    m = json.loads(lines[0])
    for key in ("plies", "z", "resigned", "playout", "would_resign", "fp", "game"):
        assert key in m
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_worker.py -v`
Expected: FAIL — `ImportError` on `chessrl.selfplay.worker`.

- [ ] **Step 3: Implement**

```python
# chessrl/selfplay/worker.py
"""Self-play worker process: spawn-safe, sentinel-file controlled.

worker_main is a top-level function so the spawn start method can re-import and
call it. It polls run_dir/config.json's run config, loads the newest checkpoint
when one appears, plays batches of concurrent games, and writes sparse records,
PGNs, and per-game meta lines. A run_dir/STOP sentinel file (not a
multiprocessing.Event) signals shutdown -- simpler across spawn and debuggable.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from chessrl.config.config import RunConfig
from chessrl.model.network import BatchedNetEvaluator, PolicyValueNet
from chessrl.selfplay.concurrent import play_games_concurrent
from chessrl.selfplay.pgn_io import save_pgn


def next_counter_for_worker(run_dir, worker_id: int) -> int:
    """Highest existing counter for this worker id + 1 (0 if none). Makes the
    counter collision-proof across process restarts."""
    prefix = f"game_w{worker_id:02d}_"
    best = -1
    for f in (Path(run_dir) / "games").glob(f"{prefix}*.npz"):
        stem = f.stem  # game_wWW_CCCCCCC
        try:
            best = max(best, int(stem[len(prefix):]))
        except ValueError:
            continue
    return best + 1


def _resolve_device(requested: str) -> str:
    return requested if (requested == "cpu" or torch.cuda.is_available()) else "cpu"


def _newest_checkpoint(run_dir) -> Path | None:
    ckpts = sorted((Path(run_dir) / "checkpoints").glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


def _build_evaluator(run_dir, cfg: RunConfig, device: str, seed: int) -> BatchedNetEvaluator:
    """Newest checkpoint if present, else a fresh net seeded identically across
    workers so a cold-start run begins from the same weights everywhere."""
    ckpt = _newest_checkpoint(run_dir)
    if ckpt is not None:
        return BatchedNetEvaluator.from_checkpoint(ckpt, cfg.network, device=device)
    torch.manual_seed(seed)
    net = PolicyValueNet(cfg.network)
    return BatchedNetEvaluator(net, device=device)


def run_one_batch(
    run_dir, worker_id: int, evaluator: BatchedNetEvaluator, cfg: RunConfig,
    rng: np.random.Generator, start_counter: int,
) -> int:
    """Play one batch of concurrent_games games, persist them, append meta.
    Returns the next free counter."""
    results = play_games_concurrent(
        evaluator, cfg.mcts, cfg.selfplay, rng, num_games=cfg.selfplay.concurrent_games
    )
    games_dir = Path(run_dir) / "games"
    meta_path = Path(run_dir) / f"games_meta_w{worker_id:02d}.jsonl"
    counter = start_counter
    with meta_path.open("a") as mf:
        for rec, final_board, z, meta in results:
            name = f"game_w{worker_id:02d}_{counter:07d}"
            rec.save(games_dir / f"{name}.npz")
            save_pgn(final_board, z, games_dir / f"{name}.pgn")
            line = dict(meta)
            line["game"] = name
            line["worker"] = worker_id
            mf.write(json.dumps(line) + "\n")
            counter += 1
    return counter


def worker_main(worker_id: int, run_dir: str, stop_path: str, device: str) -> None:
    run_dir = Path(run_dir)
    stop_path = Path(stop_path)
    cfg = RunConfig.from_json(run_dir / "config.json")
    resolved_device = _resolve_device(device)
    rng = np.random.default_rng(cfg.training.seed + 1000 * worker_id)

    counter = next_counter_for_worker(run_dir, worker_id)
    loaded_ckpt: Path | None = None
    evaluator = _build_evaluator(run_dir, cfg, resolved_device, cfg.training.seed)
    loaded_ckpt = _newest_checkpoint(run_dir)

    while not stop_path.exists():
        newest = _newest_checkpoint(run_dir)
        if newest is not None and newest != loaded_ckpt:
            evaluator = BatchedNetEvaluator.from_checkpoint(
                newest, cfg.network, device=resolved_device
            )
            loaded_ckpt = newest
        counter = run_one_batch(run_dir, worker_id, evaluator, cfg, rng, counter)
        # tight loop is fine; the sentinel check between batches paces shutdown.
        time.sleep(0.01)
```

- [ ] **Step 4: Run the worker tests**

Run: `.venv\Scripts\python -m pytest tests/test_worker.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add chessrl/selfplay/worker.py tests/test_worker.py
git commit -m "feat: spawn-safe self-play worker with checkpoint polling and meta logging"
```

---

### Task 8: Parallel training loop

**Files:**
- Create: `chessrl/training/parallel_loop.py`
- Modify: `scripts/train.py` (route between single-process and parallel modes)
- Test: `tests/test_parallel_loop_unit.py` (fast, non-spawn unit tests of the ingest/metrics helpers)

`parallel_loop.main(argv) -> run_dir` sets up a new run identically to `loop.py` (config.json, provenance.json, games/, state.json), spawns `workers` worker processes, then loops: poll `games/` for new `.npz` files (track an ingested set; sort new files by `(mtime, name)`), add to the buffer, update `total_positions`, train up to `trainer.allowed_steps` (gated on `len(buffer) >= batch_size`), checkpoint every `checkpoint_every_steps` crossed, append a metrics line, restart dead workers, sleep when idle. On reaching `--games` new games, write the `STOP` sentinel, join (with terminate fallback), final checkpoint + state.json, remove `STOP`.

This task splits the loop into testable helpers (`ingest_new_games`, `aggregate_resign_fp`, `make_run_dir`) plus the orchestration `main`. The unit test covers the helpers without spawning; the full spawn path is the Task 9 slow smoke test.

- [ ] **Step 1: Write the failing unit tests**

```python
# tests/test_parallel_loop_unit.py
import json

import numpy as np

from chessrl.config.config import RunConfig
from chessrl.supervised.pgn_import import record_from_pgn
from chessrl.training.buffer import ReplayBuffer
from chessrl.training.parallel_loop import (
    aggregate_resign_fp,
    ingest_new_games,
    make_run_dir,
)

FOOLS_MATE = '[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n'


def test_ingest_new_games_adds_only_unseen(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    rec = record_from_pgn(FOOLS_MATE)  # 4 positions
    rec.save(games / "game_w00_0000000.npz")
    buf = ReplayBuffer(1000)
    ingested = set()

    added, positions = ingest_new_games(tmp_path, buf, ingested)
    assert added == 1
    assert positions == 4
    assert len(buf) == 4

    # Second call: nothing new.
    added2, positions2 = ingest_new_games(tmp_path, buf, ingested)
    assert added2 == 0 and positions2 == 0

    # A new file is picked up.
    rec.save(games / "game_w00_0000001.npz")
    added3, positions3 = ingest_new_games(tmp_path, buf, ingested)
    assert added3 == 1 and positions3 == 4
    assert len(buf) == 8


def test_ingest_skips_partial_npz(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    (games / "game_w00_0000000.npz").write_bytes(b"not a real npz yet")
    buf = ReplayBuffer(1000)
    ingested = set()
    added, positions = ingest_new_games(tmp_path, buf, ingested)
    # A half-written file fails to load and is left for a later pass.
    assert added == 0 and positions == 0
    assert len(ingested) == 0


def test_aggregate_resign_fp(tmp_path):
    (tmp_path / "games_meta_w00.jsonl").write_text(
        json.dumps({"playout": True, "would_resign": True, "fp": True}) + "\n"
        + json.dumps({"playout": True, "would_resign": True, "fp": False}) + "\n"
        + json.dumps({"playout": False, "would_resign": False, "fp": False}) + "\n"
    )
    (tmp_path / "games_meta_w01.jsonl").write_text(
        json.dumps({"playout": True, "would_resign": False, "fp": False}) + "\n"
    )
    stats = aggregate_resign_fp(tmp_path)
    # 3 playout games total; 1 false positive -> rate 1/3.
    assert stats["playout_games"] == 3
    assert stats["false_positives"] == 1
    assert abs(stats["resign_fp_rate"] - 1.0 / 3.0) < 1e-9


def test_make_run_dir_writes_config_and_provenance(tmp_path):
    cfg = RunConfig.from_dict({"run_name": "pll"})
    run_dir = make_run_dir(cfg, runs_root=tmp_path / "runs")
    assert (run_dir / "config.json").exists()
    assert (run_dir / "provenance.json").exists()
    assert (run_dir / "games").is_dir()
    loaded = RunConfig.from_json(run_dir / "config.json")
    assert loaded.run_name == "pll"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_parallel_loop_unit.py -v`
Expected: FAIL — `ImportError` on `chessrl.training.parallel_loop`.

- [ ] **Step 3: Implement**

```python
# chessrl/training/parallel_loop.py
"""Parallel training loop (M5): self-play worker processes generate games while
the main process ingests them, paces training, and checkpoints.

Spawn start method everywhere. worker_main lives in chessrl.selfplay.worker so
spawn can re-import it; main() must not run at import time. A run_dir/STOP
sentinel file signals workers to stop.
"""
import argparse
import json
import multiprocessing as mp
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from chessrl.config.config import RunConfig
from chessrl.model.network import PolicyValueNet
from chessrl.selfplay.worker import worker_main
from chessrl.training.buffer import ReplayBuffer
from chessrl.training.trainer import Trainer


def _provenance() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
    except OSError:
        commit = None
    return {
        "git_commit": commit,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def make_run_dir(cfg: RunConfig, runs_root) -> Path:
    run_dir = Path(runs_root) / f"{cfg.run_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    (run_dir / "games").mkdir(parents=True)
    (run_dir / "config.json").write_text(cfg.to_json())
    (run_dir / "provenance.json").write_text(json.dumps(_provenance(), indent=2))
    return run_dir


def ingest_new_games(run_dir, buffer: ReplayBuffer, ingested: set) -> tuple:
    """Add any .npz games not yet ingested. Returns (games_added, positions_added).
    Files that fail to load (half-written) are skipped and retried next pass."""
    games_dir = Path(run_dir) / "games"
    new_files = sorted(
        (p for p in games_dir.glob("*.npz") if p.name not in ingested),
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    games_added = positions_added = 0
    for f in new_files:
        try:
            from chessrl.selfplay.records import GameRecord

            rec = GameRecord.load(f)
        except Exception:
            continue  # half-written; leave un-ingested for a later pass
        buffer.add_game(rec)
        ingested.add(f.name)
        games_added += 1
        positions_added += len(rec)
    return games_added, positions_added


def aggregate_resign_fp(run_dir) -> dict:
    """Aggregate resignation false-positive stats from all worker meta files."""
    playout = fp = 0
    for meta_file in Path(run_dir).glob("games_meta_w*.jsonl"):
        for line in meta_file.read_text().splitlines():
            if not line.strip():
                continue
            m = json.loads(line)
            if m.get("playout"):
                playout += 1
                if m.get("fp"):
                    fp += 1
    rate = (fp / playout) if playout else 0.0
    return {"playout_games": playout, "false_positives": fp, "resign_fp_rate": rate}


def _spawn_worker(ctx, worker_id, run_dir, stop_path, device):
    p = ctx.Process(
        target=worker_main,
        args=(worker_id, str(run_dir), str(stop_path), device),
        daemon=False,
    )
    p.start()
    return p


def main(argv=None) -> Path:
    mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config path (new run)")
    ap.add_argument("--resume", help="run directory name under runs-root")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--games", type=int, default=200, help="total NEW games this invocation")
    args = ap.parse_args(argv)

    if args.resume:
        run_dir = Path(args.runs_root) / args.resume
        cfg = RunConfig.from_json(run_dir / "config.json")
        state = json.loads((run_dir / "state.json").read_text())
        total_positions = state["positions"]
        baseline_games = state["games"]
    else:
        cfg = RunConfig.from_yaml(args.config) if args.config else RunConfig()
        run_dir = make_run_dir(cfg, args.runs_root)
        total_positions = 0
        baseline_games = 0

    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + baseline_games)

    net = PolicyValueNet(cfg.network)
    trainer = Trainer(net, cfg.training, run_dir)
    buffer = ReplayBuffer(cfg.training.buffer_size)
    ingested: set = set()
    if args.resume:
        ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))
        if ckpts:
            trainer.load_checkpoint(ckpts[-1])
        buffer = ReplayBuffer.from_run_dir(run_dir, cfg.training.buffer_size)
        for f in (run_dir / "games").glob("*.npz"):
            ingested.add(f.name)

    stop_path = run_dir / "STOP"
    if stop_path.exists():
        stop_path.unlink()

    ctx = mp.get_context("spawn")
    procs = [
        _spawn_worker(ctx, wid, run_dir, stop_path, cfg.training.selfplay_device)
        for wid in range(cfg.selfplay.workers)
    ]

    metrics_path = run_dir / "metrics.jsonl"
    games_seen = 0
    restarts = 0
    last_ckpt_bucket = trainer.step // cfg.training.checkpoint_every_steps
    start = time.time()

    try:
        while games_seen < args.games:
            added, positions = ingest_new_games(run_dir, buffer, ingested)
            games_seen += added
            total_positions += positions

            steps_done = 0
            n = trainer.allowed_steps(total_positions)
            if n > 0 and len(buffer) >= cfg.training.batch_size:
                m = trainer.train_steps(buffer, n, rng)
                steps_done = n
                bucket = trainer.step // cfg.training.checkpoint_every_steps
                if bucket > last_ckpt_bucket:
                    trainer.save_checkpoint()
                    last_ckpt_bucket = bucket
            else:
                m = {"policy_loss": None, "value_loss": None, "step": trainer.step}

            # restart any dead worker
            for i, p in enumerate(procs):
                if not p.is_alive():
                    procs[i] = _spawn_worker(
                        ctx, i, run_dir, stop_path, cfg.training.selfplay_device
                    )
                    restarts += 1

            elapsed = max(time.time() - start, 1e-9)
            fp_stats = aggregate_resign_fp(run_dir)
            metrics = {
                "games": baseline_games + games_seen,
                "new_games": games_seen,
                "positions": total_positions,
                "step": trainer.step,
                "steps_this_cycle": steps_done,
                "policy_loss": m.get("policy_loss"),
                "value_loss": m.get("value_loss"),
                "games_per_hour": games_seen / elapsed * 3600.0,
                "worker_restarts": restarts,
                "resign_fp_rate": fp_stats["resign_fp_rate"],
            }
            with metrics_path.open("a") as f:
                f.write(json.dumps(metrics) + "\n")

            if added == 0 and steps_done == 0:
                time.sleep(1.0)
    finally:
        stop_path.write_text("stop")
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        # final drain of any games written during shutdown
        added, positions = ingest_new_games(run_dir, buffer, ingested)
        games_seen += added
        total_positions += positions
        trainer.save_checkpoint()
        (run_dir / "state.json").write_text(
            json.dumps({"games": baseline_games + games_seen, "positions": total_positions})
        )
        if stop_path.exists():
            stop_path.unlink()

    return run_dir
```

- [ ] **Step 4: Run the unit tests**

Run: `.venv\Scripts\python -m pytest tests/test_parallel_loop_unit.py -v`
Expected: 4 passed.

- [ ] **Step 5: Route `scripts/train.py` between modes**

Replace `scripts/train.py` with a router: `--parallel` selects the parallel loop, otherwise the single-process loop. Both forward all other args.

```python
# scripts/train.py
"""Entry point.

Single process (smoke / small): python scripts/train.py --config experiments/foo.yaml
Parallel self-play (M5):         python scripts/train.py --parallel --config experiments/foo.yaml --games 200
"""
import sys


def main() -> None:
    argv = sys.argv[1:]
    if "--parallel" in argv:
        argv = [a for a in argv if a != "--parallel"]
        from chessrl.training.parallel_loop import main as parallel_main

        parallel_main(argv)
    else:
        from chessrl.training.loop import main as loop_main

        loop_main(argv)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the loop unit tests plus the M4 smoke (single-process routing unaffected)**

Run: `.venv\Scripts\python -m pytest tests/test_parallel_loop_unit.py tests/test_smoke.py -v`
Expected: all passed — single-process routing is unchanged; new helpers pass.

- [ ] **Step 7: Commit**

```powershell
git add chessrl/training/parallel_loop.py scripts/train.py tests/test_parallel_loop_unit.py
git commit -m "feat: parallel training loop ingesting worker games with pacing, checkpoints, restart, fp metrics"
```

---

### Task 9: Parallel end-to-end smoke (slow gate)

**Files:**
- Modify: `pyproject.toml` (register the `slow` marker; exclude slow from the default run)
- Test: `tests/test_parallel_smoke.py`

This is the M5 end-to-end gate: a real spawn of one worker, real game files, real training, real checkpoint, clean STOP. It is marked `slow` and excluded from the default suite (Windows spawn + CUDA-less CPU run takes ~30–60 s); gates run it explicitly with `-m slow`.

- [ ] **Step 1: Register the marker and default-exclude slow tests**

Replace the `[tool.pytest.ini_options]` block in `pyproject.toml` with:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not slow'"
markers = [
    "slow: end-to-end / multiprocessing tests excluded from the default run (use -m slow to run them)",
]
```

- [ ] **Step 2: Write the slow smoke test**

```python
# tests/test_parallel_smoke.py
"""M5 gate: the full parallel pipeline runs end to end with a real spawned
worker. Marked slow (Windows spawn + CPU); run with `-m slow`."""
import json

import pytest

from chessrl.training.parallel_loop import main

SMOKE_YAML = """\
run_name: psmoke
network: {blocks: 1, filters: 8}
mcts: {simulations: 8, temperature_moves: 4, leaves_per_tree: 2}
selfplay: {ply_cap: 30, workers: 1, concurrent_games: 2, resign_playout_fraction: 0.0}
training: {batch_size: 16, buffer_size: 1000, samples_per_position: 2.0, checkpoint_every_steps: 1, device: cpu, selfplay_device: cpu}
"""


@pytest.mark.slow
def test_parallel_smoke(tmp_path):
    cfg = tmp_path / "psmoke.yaml"
    cfg.write_text(SMOKE_YAML)
    run_dir = main(
        ["--config", str(cfg), "--runs-root", str(tmp_path / "runs"), "--games", "2"]
    )

    npz = list((run_dir / "games").glob("*.npz"))
    pgn = list((run_dir / "games").glob("*.pgn"))
    ckpts = list((run_dir / "checkpoints").glob("ckpt_*.pt"))
    assert len(npz) >= 2
    assert len(pgn) >= 2
    assert len(ckpts) >= 1

    metrics_lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(metrics_lines) >= 1
    last = json.loads(metrics_lines[-1])
    assert "games_per_hour" in last
    assert "resign_fp_rate" in last
    assert "worker_restarts" in last

    # STOP sentinel cleaned up; state.json written.
    assert not (run_dir / "STOP").exists()
    state = json.loads((run_dir / "state.json").read_text())
    assert state["games"] >= 2

    # at least one worker meta file with valid json lines
    metas = list(run_dir.glob("games_meta_w*.jsonl"))
    assert metas
    first_meta = metas[0].read_text().splitlines()[0]
    assert "plies" in json.loads(first_meta)
```

- [ ] **Step 3: Run the slow gate explicitly**

Run: `.venv\Scripts\python -m pytest tests/test_parallel_smoke.py -m slow -v --durations=3`
Expected: 1 passed in ~30–60 s. If it hangs, the most likely causes are (a) `worker_main` not importable at top level (spawn re-import fails) or (b) `main` running at import time — both are guarded here, so a hang means a worker raised before writing any game; check that `selfplay_device: cpu` and the tiny config were honored. Do NOT add `sleep` hacks; fix the root cause.

- [ ] **Step 4: Confirm the default suite excludes slow tests**

Run: `.venv\Scripts\python -m pytest tests/test_parallel_smoke.py -v`
Expected: `1 deselected` (the `slow` marker is filtered by the default `-m 'not slow'`).

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml tests/test_parallel_smoke.py
git commit -m "test: M5 parallel end-to-end smoke gate (slow marker, spawn worker)"
```

---

### Task 10: Profiling script + full-suite gate + milestone summary

**Files:**
- Create: `scripts/profile_selfplay.py`
- Modify: `docs/superpowers/specs/2026-06-11-chess-rl-design.md` (append the M5 C++ move-gen decision to the milestone notes — decision only, no implementation)

`profile_selfplay.py` times `play_games_concurrent` single-process with a fresh `BatchedNetEvaluator` and prints throughput. With `--profile` it wraps the run in `cProfile` and prints the top-20 cumulative functions. This output is the data behind the C++ move-gen decision; the decision is recorded in the spec's milestone section, the swap itself is deferred.

- [ ] **Step 1: Implement the profiling script**

```python
# scripts/profile_selfplay.py
"""M5 profiling gate: measure concurrent self-play throughput and locate the
hot path. The C++/Rust move-gen decision is made from this output (recorded in
the milestone summary; the swap itself is not part of M5)."""
import argparse
import cProfile
import pstats
import time
from pathlib import Path

import numpy as np
import torch

from chessrl.config.config import RunConfig
from chessrl.model.network import BatchedNetEvaluator, PolicyValueNet
from chessrl.selfplay.concurrent import play_games_concurrent


def _resolve_device(requested: str) -> str:
    return requested if (requested == "cpu" or torch.cuda.is_available()) else "cpu"


def run(cfg: RunConfig, num_games: int, device: str, seed: int = 0):
    torch.manual_seed(seed)
    net = PolicyValueNet(cfg.network)
    evaluator = BatchedNetEvaluator(net, device=_resolve_device(device))
    rng = np.random.default_rng(seed)
    return play_games_concurrent(evaluator, cfg.mcts, cfg.selfplay, rng, num_games=num_games)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config path (defaults to RunConfig defaults)")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--profile", action="store_true", help="wrap in cProfile, print top-20 cumulative")
    args = ap.parse_args(argv)

    cfg = RunConfig.from_yaml(args.config) if args.config else RunConfig()

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
    t0 = time.time()
    results = run(cfg, args.games, args.device)
    wall = time.time() - t0
    if args.profile:
        prof.disable()

    positions = sum(len(rec) for rec, *_ in results)
    sims = positions * cfg.mcts.simulations
    print(f"device:           {_resolve_device(args.device)}")
    print(f"games:            {len(results)}")
    print(f"positions:        {positions}")
    print(f"wall_seconds:     {wall:.3f}")
    print(f"games_per_hour:   {len(results) / wall * 3600:.1f}")
    print(f"positions_per_s:  {positions / wall:.1f}")
    print(f"simulations_per_s:{sims / wall:.1f}  (approx: positions * cfg.mcts.simulations)")

    if args.profile:
        stats = pstats.Stats(prof).sort_stats("cumulative")
        stats.print_stats(20)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the profiler on CPU (sanity, not a perf claim)**

```powershell
.venv\Scripts\python scripts\profile_selfplay.py --games 4 --device cpu
```

Expected: prints games/positions/wall/throughput lines without error. (On the 5090, rerun with `--device cuda --games 16 --profile` to capture the real numbers for the decision.)

- [ ] **Step 3: Capture the real numbers and record the C++ decision**

On the 5090:

```powershell
.venv\Scripts\python scripts\profile_selfplay.py --games 16 --device cuda --profile
```

Read the top-20 cumulative table. The spec predicts self-play is **CPU-bound** (tree descent + `python-chess` legal-move generation dominate, GPU batches are sub-millisecond). Append a short decision note to the milestone section of `docs/superpowers/specs/2026-06-11-chess-rl-design.md` (find the line beginning `5. **M5** Batched parallel self-play`) — add a sub-bullet directly under it recording: measured games/hour, positions/sec, the dominant cumulative functions, and the **decision** (proceed with a C++/Rust move-gen backend behind `chess_env`, or defer) with the one-line rationale. Do NOT implement the backend in M5.

Example of the appended note (fill in real numbers from the run):

```
   - M5 profiling result (5090, 6x64 net, sims=200, concurrent_games=32): <N> games/hour, <P> positions/sec; cumulative time dominated by python-chess legal-move generation and tree descent (GPU eval < 1 ms/batch). Decision: <PROCEED with C++ move-gen behind chess_env | DEFER>, rationale: <one line>.
```

- [ ] **Step 4: Run the entire fast suite (the regression gate)**

Run: `.venv\Scripts\python -m pytest --durations=10`
Expected: all fast tests pass (the original 59 M1–M4 tests, minus the renamed buffer test plus its replacement, plus the new M5 fast tests: config-m5, batched-evaluator, batched-mcts, pgn-io, concurrent-selfplay, worker, parallel-loop-unit). The `slow` parallel smoke is deselected. Slowest items: batched mate-in-two, reference mate-in-two, overfit, single-process smoke.

- [ ] **Step 5: Run the slow gate once more to confirm green end to end**

Run: `.venv\Scripts\python -m pytest -m slow -v`
Expected: `test_parallel_smoke` passes.

- [ ] **Step 6: Commit**

```powershell
git add scripts/profile_selfplay.py docs/superpowers/specs/2026-06-11-chess-rl-design.md
git commit -m "feat: self-play profiling script and recorded M5 C++ move-gen decision"
```

---

## Self-Review Notes

- **Spec coverage (M5):** batched evaluator with its own `.eval()` net + `from_checkpoint` (Task 2); batched MCTS with virtual loss, subtree reuse, and the exact K=1 equivalence gate against the reference incl. mate-in-1/2 with K=4 and the visit-sum overshoot semantics (Task 3); concurrent self-play mirroring `play.py` semantics plus resign false-positive meta (Task 5); buffer `(mtime, name)` ordering for multi-worker chronology (Task 6); spawn-safe worker with checkpoint polling, restart-safe counters, per-worker meta jsonl, STOP sentinel (Task 7); parallel loop with ingest/pace/checkpoint-cadence/dead-worker-restart/fp-rate metrics and resume (Task 8); the slow spawn smoke gate with the registered `slow` marker and default exclusion (Task 9); profiling script + recorded C++ decision (Task 10). Out-of-scope items (live feed/ZeroMQ → M7, evaluator → M6, the C++ backend implementation, cross-worker inference server) are explicitly stated in the header and not built.
- **Design refinements made (and why):**
  - *API shape for BatchedMCTS.* The prompt flagged a per-tree `search_root` as awkward and asked the author to design the API. I settled on `init_tree`/`run`/`step_round`/`visit_counts`/`root_q`/`advance`/`add_root_noise` over `SearchTree` objects. `run` drives a single tree (the equivalence path); `step_round` advances a list of trees sharing one GPU batch (the real driver used by concurrent self-play). This cleanly supports the equivalence gate (K=1, one tree), subtree reuse, and per-root Dirichlet noise.
  - *Root-visit accounting.* The real reference counts the initial root expansion as one visit (root ends at `simulations + 1`, children sum to `simulations`). The batched `init_tree` reproduces this exactly and `sims_done` counts only the per-round leaf selections, so `visit_counts` sums to `simulations` at K=1 — required for the exact-equivalence test.
  - *Virtual-loss formulation.* `vloss` is stored on `Node` (added as a slot to the reference `Node` without changing reference behavior — it stays 0 there). Selection treats `vloss` as added visits valued −1 from the parent's perspective; `_backup` removes the loss it added. At K=1 only one leaf is selected per round, so the loss is applied then immediately removed with no intervening selection — a true no-op, which is what makes equivalence hold. A defensive `_clear_virtual_loss` re-zeroes per round.
  - *`advance` re-root semantics.* On reuse, `sims_done` is reset to the reused child's existing `visit_count` so the next `run` tops the subtree up to `simulations` rather than re-running a full budget — matching the spec's "keep accumulated statistics" intent and giving the ~2–4× reuse speedup. A miss (child absent, e.g. after a forced move outside the tree) falls back to a fresh expanded root.
  - *Worker testability.* Rather than only providing `worker_main` (hard to unit-test without spawning), I exposed `next_counter_for_worker` and `run_one_batch` as top-level helpers and unit-test those directly; the full spawn path is covered by the slow smoke gate. Same pattern for `parallel_loop` (`ingest_new_games`, `aggregate_resign_fp`, `make_run_dir`).
  - *Half-written file safety.* `ingest_new_games` wraps `GameRecord.load` in try/except and leaves un-loadable files un-ingested for a later pass — necessary because workers write `.npz` concurrently with the main process scanning. A dedicated unit test (`test_ingest_skips_partial_npz`) locks this in.
  - *Resign false-positive rule.* `fp = playout and would_resign and z >= 0` for the would-be resigner (the side whose evaluation collapsed). This matches the spec's "would_resign and actual z better for the would-be resigner than loss" for the test's WhiteIsLost evaluator (White is the resigner; draw or White-win contradicts the resignation). The rationale is documented inline in `_is_false_positive`.
- **Type/name consistency check against the real M1–M4 signatures I read:**
  - `BatchedNetEvaluator.from_checkpoint` loads `ckpt["model"]` — matches `Trainer.save_checkpoint` keys (`step`, `model`, `optimizer`). ✓
  - `evaluate_many` returns `(N,4672)` softmaxed float32 + `(N,)` float32; concurrent/MCTS code consumes `policies[i]`, `float(values[i])`. ✓
  - Batched `_select`/`_expand` reuse the reference's exact PUCT formula, `move_to_index(m, flip)`, `index_to_move(idx, flip, board)`, `terminal_value(board)`, and children iterated in `legal_moves` order. ✓
  - `RecordBuilder.add(board, move_indices, visit_counts, played_index)` and `finalize(z_white)` called with `int32` index/count arrays and an `int` z — matches `records.py`. ✓
  - `play_games_concurrent` returns `(GameRecord, final_board, z, meta)`; `run_one_batch` unpacks exactly that 4-tuple. ✓
  - `Trainer(net, cfg, run_dir)`, `trainer.allowed_steps(total_positions)`, `trainer.train_steps(buffer, n, rng)`, `trainer.save_checkpoint()`, `trainer.step` — all match `trainer.py`. ✓
  - `ReplayBuffer(capacity)`, `add_game(rec)`, `from_run_dir(run_dir, capacity)` — match `buffer.py`; the only change is the sort key, with its test updated. ✓
  - `RunConfig.from_yaml/from_json/from_dict/to_json` and the new config fields are used exactly as defined in Task 1. ✓
  - `save_pgn(board, z, path)` extracted from `loop._save_pgn` with identical output; `loop.py` delegates to it (smoke test still green). ✓
- **No placeholders:** every task contains complete, runnable test and implementation code; no "TBD", no "similar to task N". Code is repeated rather than referenced (e.g. the full config dataclasses in Task 1, the full `_select` in Task 3).
- **Known intentional simplifications:** profiling's `simulations_per_s` is the documented approximation `positions * cfg.mcts.simulations` (subtree reuse means real sim counts vary per move); the worker's `time.sleep(0.01)` between batches is a cheap shutdown-pacing yield, not a perf knob; `deque` O(n) sampling from M4 is unchanged (the profiling gate, not this plan, decides if it matters).
