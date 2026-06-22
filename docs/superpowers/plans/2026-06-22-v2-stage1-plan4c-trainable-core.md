# v2 Stage 1 — Plan 4c: Trainable Core (training-loop assembly)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Assemble the pieces into a **trainable** v2 emergent run: a dual-head `train_steps_vector`, a `parallel_loop` `emergent` branch (frozen-encoder snapshot + GoalSpace refresh + `VectorGoalReplayBuffer` + buffer-rebuild-on-refit), the worker's emergent self-play dispatch, and `experiments/v2-stage1.yaml`. Uses 4b's uniform cluster-goal assigner (win-value curriculum is Plan 4d).

**Architecture:** Emergent mode (`goal_mode=="emergent"`, `network.goal_cond=="vector"`). The **trainer** owns the GoalSpace: it snapshots the live net to a frozen encoder, observes window-deltas from ingested games into the GoalSpace reservoir, re-fits centroids every `refresh_every` games (re-snapshotting the frozen encoder, then **rebuilding the `VectorGoalReplayBuffer`** against the new goal space — the I1 fix), and persists both the GoalSpace and the frozen-encoder checkpoint for workers. The **worker** loads the GoalSpace centroids (for cluster assignment) + the live net (`VectorGoalNetEvaluator`) and plays via `play_meansend_games_concurrent`; with no GoalSpace yet it assigns terminal-only (cold start). The dual-head loss = masked-MSE on `v_win` (terminal-reward outcome) + weighted-BCE on `v_goal` (achievement) + masked-CE policy.

**Tech Stack:** Python, PyTorch, NumPy, pytest. Files: `chessrl/training/trainer.py`, `chessrl/training/parallel_loop.py`, `chessrl/selfplay/worker.py`, `experiments/v2-stage1.yaml`.

## Global Constraints

- Emergent net: `PolicyValueNet(cfg.network, goal_conditioned=True)` with `cfg.network.goal_cond=="vector"` (dual head). `win_vector = net.win_vector` (detached numpy) is the terminal-pursuit goal vector.
- Frozen encoder: a `VectorGoalNetEvaluator` built from a snapshot checkpoint `run_dir/frozen_encoder.pt`; used by the trainer for GoalSpace delta embedding + HER. Re-snapshotted at each refresh.
- GoalSpace persisted to `run_dir/goalspace/` (Plan 2 save/load); workers reload on mtime change. Worker assignment needs only `centroids`/`ready`/`n_clusters`/`centroid()` — NOT the embedder.
- Buffer-rebuild-on-refit (I1): when GoalSpace re-fits, rebuild `VectorGoalReplayBuffer` from the run dir against the NEW goal space so HER labels are epoch-consistent.
- Dual-head loss: `loss = masked_MSE(v_win, v_win_target; mask=v_win_mask) + weighted_BCE(v_goal, v_goal_target; w=v_goal_weight) + masked_CE(policy; mask=p_mask)`. BCE computed in fp32 outside autocast (PyTorch forbids `binary_cross_entropy` under autocast — mirror `train_steps_goal`). Deadlines passed RAW (the vector net scales internally).
- v1 (`none`/`always_win`/`random`/`lp`) and vanilla paths UNCHANGED.
- Windows venv tests, unpiped/foreground. Stage only named files; never `git add -A`.

---

### Task 1: `Trainer.train_steps_vector` (dual-head loss)

**Files:**
- Modify: `chessrl/training/trainer.py`
- Test: `tests/test_train_steps_vector.py` (new)

**Interfaces:** `Trainer.train_steps_vector(buffer, n, rng) -> {"policy_loss", "value_loss", "step"}`. Consumes the `VectorGoalReplayBuffer.sample` 9-tuple `(x, goal_vec, deadline, p, p_mask, v_win, v_win_mask, v_goal, v_goal_weight)`. Forward: `logits, v_win_pred, v_goal_pred = net(x, deadline, goal_vec)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_steps_vector.py
import numpy as np
import chess
from chessrl.config.config import NetworkConfig, TrainingConfig
from chessrl.model.network import PolicyValueNet
from chessrl.training.trainer import Trainer
from chessrl.training.vector_buffer import VectorGoalReplayBuffer
from chessrl.selfplay.records import RecordBuilder
from chessrl.chess_env.moves import move_to_index
from tests.test_cluster_her import FakeEmbedder, FakeGoalSpace


def _cluster_game(n=4):
    b = RecordBuilder(); board = chess.Board()
    for _ in range(n):
        move = list(board.legal_moves)[0]
        idx = move_to_index(move, board.turn == chess.BLACK)
        b.add(board, [idx, idx + 1], [3, 1], played_index=idx, protagonist=board.turn,
              cluster_active=1, cluster_assigned=1,
              active_vec=np.array([1, 0, 0, 0], np.float32), explore=False)
        board.push(move)
    return b.finalize(z_white=1)


def test_train_steps_vector_runs_and_steps():
    cfg = TrainingConfig(batch_size=8, device="cpu")
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=4, goal_cond="vector"), goal_conditioned=True)
    tr = Trainer(net, cfg, run_dir=".")
    buf = VectorGoalReplayBuffer(1000, FakeEmbedder(), FakeGoalSpace())
    buf.add_game(_cluster_game(), rng=np.random.default_rng(0))
    s0 = tr.step
    m = tr.train_steps_vector(buf, 3, np.random.default_rng(1))
    assert tr.step == s0 + 3
    assert m["policy_loss"] is not None and m["value_loss"] is not None
    assert np.isfinite(m["value_loss"])
```

(Note: `FakeEmbedder`/`FakeGoalSpace` give `d=4`; the net's `filters=4` so `goal_vec` width matches `d`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train_steps_vector.py -v`
Expected: FAIL — `train_steps_vector` missing.

- [ ] **Step 3: Implement** (add to `chessrl/training/trainer.py`, mirroring `train_steps_goal`)

```python
    def train_steps_vector(self, buffer, n: int, rng: np.random.Generator) -> dict:
        """Dual-head SGD step for the v2 vector net (Plan 4c). Loss =
        masked-MSE on the tanh terminal-reward head (v_win, masked to active
        samples) + weighted-BCE on the sigmoid goal head (v_goal) + masked-CE
        policy. BCE is computed in fp32 outside autocast (autocast forbids
        binary_cross_entropy). Deadlines are passed RAW; the vector net scales
        them internally."""
        import torch
        import torch.nn.functional as F

        self.net.train()
        lp_sum = lv_sum = 0.0
        for _ in range(n):
            x, gv, deadline, p, p_mask, v_win, v_win_mask, v_goal, v_goal_w = \
                buffer.sample(self.cfg.batch_size, rng)
            xt = torch.from_numpy(x).to(self.device)
            gvt = torch.from_numpy(gv).to(self.device)
            dt = torch.from_numpy(deadline).to(self.device)
            pt = torch.from_numpy(p).to(self.device)
            pmask = torch.from_numpy(p_mask).to(self.device)
            vwin = torch.from_numpy(v_win).to(self.device)
            vwmask = torch.from_numpy(v_win_mask).to(self.device)
            vgoal = torch.from_numpy(v_goal).to(self.device)
            vgw = torch.from_numpy(v_goal_w).to(self.device)
            with torch.autocast(self.device, enabled=self.device == "cuda"):
                logits, v_win_pred, v_goal_pred = self.net(xt, deadline=dt, goal_vec=gvt)
                ce = -(pt * F.log_softmax(logits, dim=1)).sum(dim=1)
                loss_p = (ce * pmask).sum() / pmask.sum().clamp_min(1.0)
                # masked MSE on the win head (active samples only)
                se = (v_win_pred.squeeze(1) - vwin) ** 2
                loss_win = (se * vwmask).sum() / vwmask.sum().clamp_min(1.0)
            with torch.autocast(self.device, enabled=False):
                vg = v_goal_pred.float().squeeze(1).clamp(1e-6, 1.0 - 1e-6)
                bce = F.binary_cross_entropy(vg, vgoal.float(), reduction="none")
                loss_goal = (bce * vgw.float()).sum() / vgw.float().sum().clamp_min(1e-6)
                loss = loss_p.float() + loss_win.float() + loss_goal
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            self.step += 1
            lp_sum += loss_p.detach().item()
            lv_sum += (loss_win + loss_goal).detach().item()
        n = max(n, 1)
        return {"policy_loss": lp_sum / n, "value_loss": lv_sum / n, "step": self.step}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train_steps_vector.py -v`
Expected: PASS. Regression: `.venv\Scripts\python.exe -m pytest tests/test_trainer.py -v` (if present) — PASS.

- [ ] **Step 5: Commit**

```bash
git add chessrl/training/trainer.py tests/test_train_steps_vector.py
git commit -m "feat(v2): Trainer.train_steps_vector (dual-head loss)"
```

---

### Task 2: `experiments/v2-stage1.yaml` + config emergent wiring

**Files:**
- Create: `experiments/v2-stage1.yaml`
- Test: `tests/test_v2_config.py` (new)

**Interfaces:** A run config that loads via `RunConfig.from_yaml` with `goal.goal_mode=="emergent"`, `network.goal_cond=="vector"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_v2_config.py
from pathlib import Path
from chessrl.config.config import RunConfig


def test_v2_yaml_loads_emergent_vector():
    cfg = RunConfig.from_yaml(Path("experiments/v2-stage1.yaml"))
    assert cfg.goal.goal_mode == "emergent"
    assert cfg.network.goal_cond == "vector"
    assert cfg.goal.cluster_k > 0
    assert cfg.goal.goal_window > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_v2_config.py -v`
Expected: FAIL — file missing.

- [ ] **Step 3: Implement** — create `experiments/v2-stage1.yaml`. Mirror an existing experiment yaml (e.g. `experiments/gp-vanilla.yaml`) for the run/network/training/selfplay/mcts blocks (same net size + selfplay/training cadence as the v1 arms for a fair curve), and set:

```yaml
run_name: v2-stage1
network:
  goal_cond: vector
goal:
  goal_mode: emergent
  cluster_k: 48
  refresh_every: 2000
  reservoir_size: 20000
  min_reservoir: 5000
  goal_window: 8
  win_floor: 0.2
  deadline_max: 60
mcts:
  meansend_alpha: 0.25
```

Copy the other blocks (network.blocks/filters, training.*, selfplay.*, mcts.simulations/etc.) verbatim from `experiments/gp-vanilla.yaml` so the only differences are the goal/conditioning fields above. (Read `gp-vanilla.yaml` first to get the exact block values.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_v2_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/v2-stage1.yaml tests/test_v2_config.py
git commit -m "feat(v2): experiments/v2-stage1.yaml (emergent + vector)"
```

---

### Task 3: Worker emergent self-play dispatch

**Files:**
- Modify: `chessrl/selfplay/worker.py`
- Test: `tests/test_worker_emergent.py` (new)

**Interfaces:**
- `_build_evaluator` returns a `VectorGoalNetEvaluator` for emergent mode (from newest ckpt, else fresh seeded net).
- `_load_goalspace(run_dir, cfg, device) -> GoalSpace | None`: load `run_dir/goalspace/` with a frozen embedder from `run_dir/frozen_encoder.pt` if both exist; else `None`.
- `run_one_batch` emergent branch calls `play_meansend_games_concurrent(evaluator, cfg.mcts, cfg.selfplay, cfg.goal, goalspace, win_vector, rng, num_games=cfg.selfplay.concurrent_games, publisher=..., game_id_prefix=...)` with `win_vector = evaluator.net.win_vector.detach().cpu().numpy()`.
- The checkpoint-reload and goalspace-reload (on mtime change) mirror the lp repertoire pattern.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worker_emergent.py
import numpy as np
from chessrl.config.config import RunConfig
from chessrl.selfplay import worker as W


def test_build_evaluator_emergent_returns_vector(tmp_path):
    from pathlib import Path
    cfg = RunConfig.from_yaml(Path("experiments/v2-stage1.yaml"))
    ev = W._build_evaluator(tmp_path, cfg, "cpu", seed=0)
    from chessrl.model.network import VectorGoalNetEvaluator
    assert isinstance(ev, VectorGoalNetEvaluator)


def test_load_goalspace_none_when_absent(tmp_path):
    from pathlib import Path
    cfg = RunConfig.from_yaml(Path("experiments/v2-stage1.yaml"))
    assert W._load_goalspace(tmp_path, cfg, "cpu") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_worker_emergent.py -v`
Expected: FAIL — emergent path / `_load_goalspace` missing.

- [ ] **Step 3: Implement** — in `chessrl/selfplay/worker.py`:

Add imports: `from chessrl.model.network import VectorGoalNetEvaluator`; `from chessrl.goals.goalspace import GoalSpace`; `from chessrl.selfplay.concurrent import play_meansend_games_concurrent`.

In `_build_evaluator`, before the existing `goal_mode` branch, add:

```python
    if cfg.goal.goal_mode == "emergent":
        if ckpt is not None:
            return VectorGoalNetEvaluator.from_checkpoint(ckpt, cfg.network, device=device)
        torch.manual_seed(seed)
        net = PolicyValueNet(cfg.network, goal_conditioned=True)
        return VectorGoalNetEvaluator(net, device=device)
```

Add the goalspace loader:

```python
GOALSPACE_DIR = "goalspace"
FROZEN_ENCODER = "frozen_encoder.pt"


def _load_goalspace(run_dir, cfg: RunConfig, device: str):
    """Load the persisted GoalSpace (centroids for cluster assignment) with a
    frozen embedder from frozen_encoder.pt. Returns None until the trainer has
    written both (cold start -> the assigner plays terminal-only)."""
    gdir = Path(run_dir) / GOALSPACE_DIR
    fenc = Path(run_dir) / FROZEN_ENCODER
    if not gdir.exists() or not fenc.exists():
        return None
    try:
        embedder = VectorGoalNetEvaluator.from_checkpoint(fenc, cfg.network, device=device)
        return GoalSpace.load(gdir, cfg.goal, embedder, np.random.default_rng(0))
    except Exception:
        return None
```

In `run_one_batch`, add an emergent branch BEFORE the existing `goal_mode != "none"` branch:

```python
    if cfg.goal.goal_mode == "emergent":
        win_vector = evaluator.net.win_vector.detach().cpu().numpy()
        results = play_meansend_games_concurrent(
            evaluator, cfg.mcts, cfg.selfplay, cfg.goal, goalspace, win_vector, rng,
            num_games=cfg.selfplay.concurrent_games,
            publisher=publisher, game_id_prefix=f"w{worker_id:02d}_b{batch_index}_",
        )
    elif cfg.goal.goal_mode != "none":
        ...  # existing v1 goal branch unchanged
```

`run_one_batch` gains a `goalspace=None` parameter (threaded from `worker_main`). In `worker_main`: build `goalspace = _load_goalspace(...)` after the evaluator; reload it on mtime change of `run_dir/goalspace/meta.json` (mirror the repertoire-reload block); pass it to `run_one_batch`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_worker_emergent.py -v`
Expected: PASS. Regression: `.venv\Scripts\python.exe -m pytest tests/test_selfplay_goals.py -v`.

- [ ] **Step 5: Commit**

```bash
git add chessrl/selfplay/worker.py tests/test_worker_emergent.py
git commit -m "feat(v2): worker emergent self-play dispatch + goalspace load"
```

---

### Task 4: `parallel_loop` emergent branch (frozen encoder + GoalSpace + vector buffer)

**Files:**
- Modify: `chessrl/training/parallel_loop.py`
- Test: `tests/test_parallel_loop_emergent.py` (new — a fast smoke test of the helpers, NOT a full spawn run)

**Interfaces:** new helpers + an `emergent` branch in `main`:
- `snapshot_frozen_encoder(net, run_dir, network_cfg, device) -> VectorGoalNetEvaluator`: save `net.state_dict()` to `run_dir/frozen_encoder.pt` (atomic) and return a `VectorGoalNetEvaluator` from it.
- `observe_game_deltas(goalspace, rec, embedder, max_samples, rng)`: for an ingested cluster/any record, sample up to `max_samples` plies, compute `embedder`-window deltas `e(s_{i+window}) - e(s_i)` and `goalspace.observe_delta(...)`.
- In `main`, when `cfg.goal.goal_mode=="emergent"`: build the frozen encoder, `GoalSpace` (load on resume), `VectorGoalReplayBuffer`; in the ingest loop observe deltas + `maybe_refresh` (re-snapshot encoder, rebuild buffer on refit), persist GoalSpace + frozen encoder; train with `train_steps_vector`.

- [ ] **Step 1: Write the failing test** (smoke-test the two new pure-ish helpers; the full loop is integration-tested by a short real run)

```python
# tests/test_parallel_loop_emergent.py
import numpy as np
import chess
from chessrl.config.config import NetworkConfig
from chessrl.model.network import PolicyValueNet, VectorGoalNetEvaluator
from chessrl.training.parallel_loop import snapshot_frozen_encoder, observe_game_deltas
from chessrl.goals.goalspace import GoalSpace
from chessrl.config.config import GoalConfig
from chessrl.selfplay.records import RecordBuilder
from chessrl.chess_env.moves import move_to_index


def test_snapshot_frozen_encoder(tmp_path):
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=8, goal_cond="vector"), goal_conditioned=True)
    ev = snapshot_frozen_encoder(net, tmp_path, NetworkConfig(blocks=2, filters=8, goal_cond="vector"), "cpu")
    assert isinstance(ev, VectorGoalNetEvaluator)
    assert (tmp_path / "frozen_encoder.pt").exists()
    e = ev.embed_boards([chess.Board()])
    assert e.shape == (1, 8)


def test_observe_game_deltas_fills_reservoir(tmp_path):
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=8, goal_cond="vector"), goal_conditioned=True)
    ev = snapshot_frozen_encoder(net, tmp_path, NetworkConfig(blocks=2, filters=8, goal_cond="vector"), "cpu")
    gs = GoalSpace(GoalConfig(goal_mode="emergent", goal_window=2, min_reservoir=3, cluster_k=2), ev, np.random.default_rng(0))
    b = RecordBuilder(); board = chess.Board()
    for _ in range(6):
        mv = list(board.legal_moves)[0]; idx = move_to_index(mv, board.turn == chess.BLACK)
        b.add(board, [idx], [1], idx); board.push(mv)
    rec = b.finalize(0)
    observe_game_deltas(gs, rec, ev, max_samples=4, rng=np.random.default_rng(0))
    assert len(gs.reservoir) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_parallel_loop_emergent.py -v`
Expected: FAIL — helpers missing.

- [ ] **Step 3: Implement** — in `chessrl/training/parallel_loop.py`:

Add imports: `import torch`, `from chessrl.model.network import VectorGoalNetEvaluator`, `from chessrl.goals.goalspace import GoalSpace`, `from chessrl.training.vector_buffer import VectorGoalReplayBuffer`, `from chessrl.training.her import reconstruct_states`.

Add the helpers:

```python
GOALSPACE_DIR = "goalspace"
FROZEN_ENCODER = "frozen_encoder.pt"


def snapshot_frozen_encoder(net, run_dir, network_cfg, device) -> "VectorGoalNetEvaluator":
    import os
    path = Path(run_dir) / FROZEN_ENCODER
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"model": net.state_dict()}, tmp)
    os.replace(tmp, path)
    return VectorGoalNetEvaluator.from_checkpoint(path, network_cfg, device=device)


def observe_game_deltas(goalspace, rec, embedder, max_samples: int, rng) -> None:
    """Sample up to max_samples plies of a game, compute frozen-encoder window
    deltas e(s_{i+w}) - e(s_i), and add them to the GoalSpace reservoir."""
    states = reconstruct_states(rec)
    T = len(states) - 1
    w = goalspace.cfg.goal_window
    starts = [i for i in range(T) if i + w <= T]
    if not starts:
        return
    if len(starts) > max_samples:
        starts = [int(s) for s in rng.choice(starts, size=max_samples, replace=False)]
    emb = embedder.embed_boards([states[i] for i in starts] + [states[i + w] for i in starts])
    half = len(starts)
    for k in range(half):
        goalspace.observe_delta(emb[half + k] - emb[k])
```

In `main`, after the existing buffer/goal-mode setup, add an emergent branch that:
1. builds `frozen_encoder = snapshot_frozen_encoder(net, run_dir, cfg.network, trainer.device)` (or loads GoalSpace + rebuilds encoder on resume);
2. `goalspace = GoalSpace.load(run_dir/GOALSPACE_DIR, cfg.goal, frozen_encoder, rng)` if resuming and it exists, else `GoalSpace(cfg.goal, frozen_encoder, rng)`;
3. `buffer = VectorGoalReplayBuffer(cfg.training.buffer_size, frozen_encoder, goalspace, deadline_max=cfg.goal.deadline_max)` (rebuild from run dir on resume via `VectorGoalReplayBuffer.from_run_dir`);
4. persists GoalSpace (`goalspace.save(run_dir/GOALSPACE_DIR)`) so workers can load.

In the ingest loop (emergent): for each newly ingested record call `observe_game_deltas(goalspace, rec, frozen_encoder, max_samples=8, rng)`; then `if goalspace.maybe_refresh(baseline_games+games_seen, embedder=snapshot_frozen_encoder(trainer.net, run_dir, cfg.network, trainer.device)):` rebuild the buffer (`buffer = VectorGoalReplayBuffer.from_run_dir(run_dir, cfg.training.buffer_size, frozen_encoder, goalspace, deadline_max=...)`) and `goalspace.save(...)`. (Use the refreshed `frozen_encoder` returned by the snapshot.) Then train with `trainer.train_steps_vector(buffer, n, rng)`.

Wire the branch so `goal_mode` (the boolean) is True for emergent (it is, since `goal_mode != "none"`), but the buffer/trainer calls use the vector path. Keep the existing `recent_records`/repertoire logic gated to the v1 goal modes only (emergent does not use the repertoire).

Because the full loop spawns workers, the unit test only covers the two helpers (Step 1). A real short run is the integration check (run instructions below).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_parallel_loop_emergent.py -v`
Expected: PASS. Then a SHORT real smoke run (integration):
`.venv\Scripts\python.exe scripts\train.py --parallel --config experiments\v2-stage1.yaml --games 8`
Expected: completes without error; `runs/v2-stage1-*/` has games with cluster columns, a `frozen_encoder.pt`, and a `goalspace/` dir; metrics.jsonl written.

- [ ] **Step 5: Commit**

```bash
git add chessrl/training/parallel_loop.py tests/test_parallel_loop_emergent.py
git commit -m "feat(v2): parallel_loop emergent branch (frozen encoder + GoalSpace + vector buffer)"
```

---

## Plan 4c deliverable

`python scripts/train.py --parallel --config experiments/v2-stage1.yaml --games N` trains a v2 emergent run end-to-end: means-end self-play with discovered cluster goals, dual-head training, periodic frozen-encoder re-snapshot + GoalSpace re-fit + buffer rebuild. **v2 is trainable.** Uniform cluster-goal assignment (win-value curriculum = Plan 4d).

## Out of scope (4d, 5)

- 4d: `winvalue.py` (interventional `P(win|do g)`) + curriculum `γ·win_value` + ε-explore in `assign_cluster_goal`.
- 5: milestone Elo eval curve backfill, α-sweep, live UI cluster-id + win-value display.
