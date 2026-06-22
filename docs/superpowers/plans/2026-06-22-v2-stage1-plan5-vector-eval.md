# v2 Stage 1 — Plan 5 (core): Vector-Net Eval

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the existing milestone/ladder Elo eval work for the v2 **vector** dual-head net, so we can Elo-curve v2 vs vanilla. A `VectorGoalMCTSPlayer` (plays to win = terminal pursuit conditioned on `win_vector`, means-end MCTS at α=0) + an `emergent` branch in `_default_agent_factory`. After this, `scripts/eval_milestone.py --run v2-stage1 --target-games N` works unchanged.

**Architecture:** Eval measures *playing strength*: the agent plays to win. For v2 that is the means-end search in **terminal pursuit** — conditioned on the net's reserved `win_vector`, with `meansend_alpha=0` so the leaf value is purely the tanh terminal-reward head (no goal shaping). Reuse `BatchedMCTS(meansend=True)` driving a single tree (it supports single-tree `run()`); no new search code. `eval_milestone.py`/`evaluate_checkpoint` are unchanged except the factory gains the emergent branch.

**Tech Stack:** Python, python-chess, pytest. Files: `chessrl/evaluation/players.py`, `chessrl/evaluation/daemon.py`.

## Global Constraints

- Eval plays to WIN: `goal_vec = net.win_vector`, `meansend_alpha = 0.0` (pure terminal-reward leaf), noise/temperature OFF.
- `VectorGoalMCTSPlayer` interface matches `NetMCTSPlayer`/`GoalNetMCTSPlayer`: `.name`, `.play(board) -> chess.Move`, sets `_last_thoughts`/`_last_root_q`.
- Factory: `goal_mode == "emergent"` → `VectorGoalMCTSPlayer`; `goal_mode != "none"` (v1) → `GoalNetMCTSPlayer` UNCHANGED; vanilla → `NetMCTSPlayer` UNCHANGED.
- Single-tree means-end search via `BatchedMCTS(meansend=True)` + `init_tree_for_meansend` + `run` + `visit_counts`. `deadline` for terminal pursuit = a fixed horizon (use `cfg.max_plies` if available else 60); at α=0 it only feeds the FiLM scalar.
- Windows venv tests, unpiped/foreground. Stage only named files; never `git add -A`.

---

### Task 1: `VectorGoalMCTSPlayer`

**Files:**
- Modify: `chessrl/evaluation/players.py` (add the class, after `GoalNetMCTSPlayer`)
- Test: `tests/test_vector_eval_player.py` (new)

**Interfaces:** `VectorGoalMCTSPlayer(name, checkpoint_path, network_cfg, simulations, device="cpu", seed=0)`; `.play(board) -> chess.Move`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_vector_eval_player.py
import chess
from chessrl.config.config import NetworkConfig
from chessrl.model.network import PolicyValueNet
from chessrl.training.trainer import Trainer
from chessrl.config.config import TrainingConfig


def _save_vector_ckpt(tmp_path):
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=16, goal_cond="vector"), goal_conditioned=True)
    tr = Trainer(net, TrainingConfig(device="cpu"), run_dir=str(tmp_path))
    return tr.save_checkpoint(), NetworkConfig(blocks=2, filters=16, goal_cond="vector")


def test_vector_player_plays_legal_move(tmp_path):
    from chessrl.evaluation.players import VectorGoalMCTSPlayer
    ckpt, ncfg = _save_vector_ckpt(tmp_path)
    p = VectorGoalMCTSPlayer("v2@0", ckpt, ncfg, simulations=8, device="cpu")
    board = chess.Board()
    mv = p.play(board)
    assert mv in board.legal_moves
    assert hasattr(p, "_last_root_q") and isinstance(p._last_thoughts, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_vector_eval_player.py -v`
Expected: FAIL — `VectorGoalMCTSPlayer` missing.

- [ ] **Step 3: Implement** (add to `chessrl/evaluation/players.py`):

```python
class VectorGoalMCTSPlayer:
    """The agent for the v2 EMERGENT (vector dual-head) arm, evaluated by playing
    to WIN: terminal pursuit conditioned on the net's reserved ``win_vector`` with
    the means-end MCTS at alpha=0 (leaf value = the tanh terminal-reward head, no
    goal shaping). Interface matches NetMCTSPlayer."""

    def __init__(self, name, checkpoint_path, network_cfg, simulations, device="cpu", seed=0):
        from chessrl.config.config import MCTSConfig
        from chessrl.mcts.batched import BatchedMCTS
        from chessrl.model.network import VectorGoalNetEvaluator

        self.name = name
        self._eval = VectorGoalNetEvaluator.from_checkpoint(
            checkpoint_path, network_cfg, device=device
        )
        self._win_vec = self._eval.net.win_vector.detach().cpu().numpy()
        self._cfg = MCTSConfig(simulations=simulations, meansend_alpha=0.0)
        self._mcts = BatchedMCTS(
            self._eval, self._cfg, rng=np.random.default_rng(seed), meansend=True
        )
        self._deadline = 60   # terminal-pursuit horizon; at alpha=0 only feeds the FiLM scalar

    def play(self, board: chess.Board) -> chess.Move:
        from chessrl.chess_env.moves import index_to_move

        tree = self._mcts.init_tree_for_meansend(
            board, self._win_vec, self._deadline, add_noise=False
        )
        self._mcts.run(tree)
        visits = self._mcts.visit_counts(tree)
        best_idx = max(visits, key=visits.get)
        flip = board.turn == chess.BLACK
        total = float(sum(visits.values())) or 1.0
        top = sorted(visits.items(), key=lambda kv: kv[1], reverse=True)[:5]
        self._last_thoughts = [
            [index_to_move(idx, flip, board).uci(), c / total] for idx, c in top
        ]
        self._last_root_q = float(self._mcts.root_q(tree))
        return index_to_move(best_idx, flip, board)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_vector_eval_player.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add chessrl/evaluation/players.py tests/test_vector_eval_player.py
git commit -m "feat(v2): VectorGoalMCTSPlayer (eval plays to win via means-end alpha=0)"
```

---

### Task 2: Emergent branch in the agent factory

**Files:**
- Modify: `chessrl/evaluation/daemon.py` (`_default_agent_factory`)
- Test: `tests/test_vector_eval_player.py` (extend)

**Interfaces:** `_default_agent_factory` returns a `VectorGoalMCTSPlayer` when `run_cfg.goal.goal_mode == "emergent"`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_vector_eval_player.py
def test_factory_routes_emergent_to_vector_player(tmp_path):
    import dataclasses
    from chessrl.config.config import RunConfig, GoalConfig, NetworkConfig, EvalConfig
    from chessrl.evaluation.daemon import _default_agent_factory
    from chessrl.evaluation.players import VectorGoalMCTSPlayer
    ckpt, ncfg = _save_vector_ckpt(tmp_path)
    run_cfg = RunConfig(
        network=NetworkConfig(blocks=2, filters=16, goal_cond="vector"),
        goal=GoalConfig(goal_mode="emergent"),
    )
    cfg = EvalConfig(agent_simulations=8)
    agent = _default_agent_factory("v2@0", ckpt, run_cfg, cfg)
    assert isinstance(agent, VectorGoalMCTSPlayer)
```

(If `RunConfig`/`EvalConfig` construction differs, adjust to the real dataclass fields — read `chessrl/config/config.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_vector_eval_player.py::test_factory_routes_emergent_to_vector_player -v`
Expected: FAIL — factory returns a `GoalNetMCTSPlayer` (the `!= "none"` branch).

- [ ] **Step 3: Implement** — in `chessrl/evaluation/daemon.py`, `_default_agent_factory`, add the emergent branch BEFORE the `goal_mode != "none"` branch:

```python
def _default_agent_factory(agent_name, ckpt_path, run_cfg, cfg):
    if run_cfg.goal.goal_mode == "emergent":
        from chessrl.evaluation.players import VectorGoalMCTSPlayer
        return VectorGoalMCTSPlayer(
            agent_name, ckpt_path, run_cfg.network, cfg.agent_simulations, device="cpu",
        )
    if run_cfg.goal.goal_mode != "none":
        from chessrl.evaluation.players import GoalNetMCTSPlayer
        return GoalNetMCTSPlayer(
            agent_name, ckpt_path, run_cfg.network, cfg.agent_simulations, device="cpu",
        )
    return NetMCTSPlayer(
        agent_name, ckpt_path, run_cfg.network, cfg.agent_simulations, device="cpu",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_vector_eval_player.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add chessrl/evaluation/daemon.py tests/test_vector_eval_player.py
git commit -m "feat(v2): eval factory routes emergent arm to VectorGoalMCTSPlayer"
```

---

## Plan 5 (core) deliverable

`scripts/eval_milestone.py --run v2-stage1 --target-games N --sims 200 --games-per-rung 20` produces a v2 Elo point on the shared ladder, comparable to vanilla — so we can curve v2 vs vanilla (754) and the v1 collapse (230–286).

## Out of scope (Plan 5 follow-ons)

- α-sweep harness (Elo at α ∈ {1.0, 0.5, 0.0} at a fixed budget) — the means↔ends mechanistic curve.
- Live UI cluster-id + win-value display on the means-end games.
- These are independent and built after the first run produces data.
