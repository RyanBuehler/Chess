# v2 Stage 1 — Plan 1: Network Vector Goal-Conditioning (FiLM)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `vector` goal-conditioning mode to `PolicyValueNet` — a FiLM pathway driven by a goal *vector* (cluster centroid ⊕ deadline) — plus a trunk `embed()` primitive and a batched vector evaluator, all coexisting with v1's existing `planes` mode.

**Architecture:** v1 conditions the goal-net on board *planes* + a deadline scalar through one sigmoid value head. Embedding-space cluster goals (v2) cannot be planes, so we add a second conditioning mode: the goal vector `(centroid ∈ R^d ⊕ scaled deadline)` feeds a small MLP that produces per-channel FiLM (feature-wise affine) parameters modulating the trunk output before the policy and value heads. The single sigmoid head is unchanged; `V(s,win)` is the head under a reserved learned `win_vector`. The legacy `planes` path is untouched so vanilla and v1 goal arms stay runnable.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest. Net lives in `chessrl/model/network.py`; config in `chessrl/config/config.py`.

## Global Constraints

- Values are side-to-move perspective (+1 = side to move wins); the goal-conditioned value head is a **sigmoid in [0,1]** = P(achieve goal). Verbatim from project conventions.
- Action index = `from_square * 73 + move_type`; policy logits are `(B, 4672)`. Verbatim.
- `d` (goal-vector / embedding dim) = `NetworkConfig.filters` (global-average-pooled trunk features). One value, used everywhere.
- Backward compatibility is mandatory: `goal_cond="planes"` (the default) must reproduce v1 behavior byte-for-byte; vanilla (`goal_conditioned=False`) is untouched.
- Run tests on Windows with the venv, unpiped and foreground: `.venv\Scripts\python.exe -m pytest <path> -v`.
- Do NOT `git add -A`; stage only the named files. Commit only the files each task lists.

---

### Task 1: `NetworkConfig.goal_cond` field

**Files:**
- Modify: `chessrl/config/config.py` (NetworkConfig dataclass, ~lines 12–20)
- Test: `tests/test_network_vector_conditioning.py` (new)

**Interfaces:**
- Produces: `NetworkConfig.goal_cond: str` ∈ `{"planes","vector"}`, default `"planes"`; raises `ValueError` on other values in `__post_init__`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_network_vector_conditioning.py
import pytest
from chessrl.config.config import NetworkConfig


def test_goal_cond_defaults_to_planes():
    assert NetworkConfig().goal_cond == "planes"


def test_goal_cond_accepts_vector():
    assert NetworkConfig(goal_cond="vector").goal_cond == "vector"


def test_goal_cond_rejects_unknown():
    with pytest.raises(ValueError):
        NetworkConfig(goal_cond="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_network_vector_conditioning.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'goal_cond'` (and the default test fails on missing attribute).

- [ ] **Step 3: Add the field + validation**

In `chessrl/config/config.py`, inside the `NetworkConfig` dataclass add the field (keep existing fields):

```python
    goal_cond: str = "planes"   # planes (v1) | vector (v2 FiLM centroid conditioning)
```

Add (or extend) its `__post_init__`:

```python
    def __post_init__(self):
        if self.goal_cond not in ("planes", "vector"):
            raise ValueError(f"bad goal_cond {self.goal_cond}")
```

If `NetworkConfig` already has a `__post_init__`, append the check to it rather than adding a second one.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_network_vector_conditioning.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add chessrl/config/config.py tests/test_network_vector_conditioning.py
git commit -m "feat(v2): NetworkConfig.goal_cond mode (planes|vector)"
```

---

### Task 2: `PolicyValueNet` vector mode — FiLM conditioning, `embed()`, `win_vector`

**Files:**
- Modify: `chessrl/model/network.py` (PolicyValueNet, ~lines 43–97)
- Test: `tests/test_network_vector_conditioning.py` (extend)

**Interfaces:**
- Consumes: `NetworkConfig.goal_cond` (Task 1); `NUM_PLANES` from `chessrl.chess_env.encoding`.
- Produces:
  - `PolicyValueNet(cfg, goal_conditioned=True)` with `cfg.goal_cond=="vector"` builds the FiLM path.
  - `forward(x, deadline=None, goal_vec=None)`: in vector mode requires `goal_vec` shape `(B, d)` and `deadline` shape `(B,)` or `(B,1)`; returns `(logits (B,4672), value (B,1) sigmoid)`.
  - `embed(x) -> Tensor (B, d)`: global-average-pooled trunk features (`d == cfg.filters`).
  - `win_vector -> nn.Parameter (d,)`: reserved learned WIN goal vector.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_network_vector_conditioning.py`:

```python
import torch
from chessrl.config.config import NetworkConfig
from chessrl.chess_env.encoding import NUM_PLANES
from chessrl.model.network import PolicyValueNet

_CFG = NetworkConfig(blocks=2, filters=16, goal_cond="vector")


def _board_batch(n=4):
    return torch.zeros(n, NUM_PLANES, 8, 8)


def test_vector_forward_shapes_and_range():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    d = _CFG.filters
    x = _board_batch(4)
    gv = torch.randn(4, d)
    dl = torch.zeros(4, 1)
    logits, value = net(x, deadline=dl, goal_vec=gv)
    assert logits.shape == (4, 4672)
    assert value.shape == (4, 1)
    assert float(value.min()) >= 0.0 and float(value.max()) <= 1.0  # sigmoid


def test_vector_forward_requires_goal_vec():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    with pytest.raises(ValueError):
        net(_board_batch(2), deadline=torch.zeros(2, 1), goal_vec=None)


def test_film_actually_conditions():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    d = _CFG.filters
    x = _board_batch(1)
    dl = torch.zeros(1, 1)
    # distinct goal vectors should produce distinct values (after a forward that
    # exercises the FiLM MLP with non-zero params).
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.1)
        va = net(x, deadline=dl, goal_vec=torch.full((1, d), -2.0))[1]
        vb = net(x, deadline=dl, goal_vec=torch.full((1, d), 2.0))[1]
    assert abs(float(va) - float(vb)) > 1e-5


def test_embed_shape_and_determinism():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    x = _board_batch(3)
    e1 = net.embed(x)
    e2 = net.embed(x)
    assert e1.shape == (3, _CFG.filters)
    assert torch.allclose(e1, e2)


def test_win_vector_present():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    assert net.win_vector.shape == (_CFG.filters,)


def test_planes_mode_unchanged():
    # Default goal_cond="planes" still builds and runs the legacy interface.
    cfg = NetworkConfig(blocks=2, filters=16)  # goal_cond defaults to "planes"
    from chessrl.goals.encoding import GOAL_PLANES
    net = PolicyValueNet(cfg, goal_conditioned=True).eval()
    x = torch.zeros(2, NUM_PLANES + GOAL_PLANES, 8, 8)
    logits, value = net(x, deadline=torch.zeros(2, 1))
    assert logits.shape == (2, 4672) and value.shape == (2, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_network_vector_conditioning.py -v`
Expected: FAIL — vector-mode tests raise (the constructor builds the planes path and `forward` has no `goal_vec`; `embed`/`win_vector` are missing). `test_planes_mode_unchanged` should already PASS.

- [ ] **Step 3: Implement the vector path**

In `chessrl/model/network.py`, add a FiLM module above `PolicyValueNet`:

```python
class FiLM(nn.Module):
    """Maps a goal-conditioning vector to per-channel (gamma, beta) affine params.

    Final layer initialized to zero so the initial modulation is the identity
    (gamma=0, beta=0 -> h*(1+0)+0 == h), which keeps early training stable."""

    def __init__(self, in_dim: int, channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, cond):
        gamma, beta = self.net(cond).chunk(2, dim=1)
        g = gamma.unsqueeze(-1).unsqueeze(-1)
        b = beta.unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + g) + b
```

In `PolicyValueNet.__init__`, after `self.tower = ...` and the `policy_conv` line, branch on the mode. Replace the existing `if goal_conditioned:` / `else:` block with:

```python
        self.goal_cond = cfg.goal_cond if goal_conditioned else "none"
        d = ch  # embedding dim == filters (GAP of trunk)
        if goal_conditioned and cfg.goal_cond == "vector":
            # FiLM conditioning from (centroid d ⊕ deadline scalar).
            self.win_vector = nn.Parameter(torch.zeros(d))
            self.film = FiLM(in_dim=d + 1, channels=ch)
            self.value_head = nn.Sequential(
                nn.Conv2d(ch, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8 * 64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        elif goal_conditioned:  # "planes" (legacy v1) — UNCHANGED
            self.value_body = nn.Sequential(
                nn.Conv2d(ch, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.value_fc = nn.Sequential(
                nn.Linear(8 * 64 + 1, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
        else:  # vanilla — UNCHANGED
            self.value_head = nn.Sequential(
                nn.Conv2d(ch, 8, 1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(8 * 64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh(),
            )
```

Note: the input-plane count must stay `NUM_PLANES` for vector mode (no goal planes). Change the `in_planes` line:

```python
        in_planes = NUM_PLANES
        if goal_conditioned and cfg.goal_cond == "planes":
            in_planes = NUM_PLANES + GOAL_PLANES
```

Add `embed()` and rewrite `forward()`:

```python
    def embed(self, x):
        """Global-average-pooled trunk features (B, filters): the frozen-encoder
        embedding e(s) that Plan 2 clusters into goals."""
        h = self.tower(self.stem(x))
        return h.mean(dim=(2, 3))

    def forward(self, x, deadline=None, goal_vec=None):
        h = self.tower(self.stem(x))
        if self.goal_cond == "vector":
            if goal_vec is None:
                raise ValueError("vector goal-conditioned net requires goal_vec")
            if deadline is None:
                deadline = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            if deadline.dim() == 1:
                deadline = deadline.unsqueeze(1)
            cond = torch.cat([goal_vec.to(h.dtype), deadline.to(h.dtype)], dim=1)
            h = self.film(h, cond)
            logits = self.policy_conv(h).permute(0, 2, 3, 1).flatten(1)
            return logits, self.value_head(h)
        logits = self.policy_conv(h).permute(0, 2, 3, 1).flatten(1)
        if self.goal_cond == "planes":
            if deadline is None:
                raise ValueError("goal-conditioned net requires a deadline scalar")
            feat = self.value_body(h)
            if deadline.dim() == 1:
                deadline = deadline.unsqueeze(1)
            value = self.value_fc(torch.cat([feat, deadline.to(feat.dtype)], dim=1))
            return logits, value
        return logits, self.value_head(h)
```

(Keep `self.goal_conditioned = goal_conditioned` as before; `self.goal_cond` is the new mode discriminator.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_network_vector_conditioning.py -v`
Expected: PASS (all, including `test_planes_mode_unchanged`).

- [ ] **Step 5: Run the existing network/goal regression tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_goal_network.py -v`
Expected: PASS (planes path unchanged).

- [ ] **Step 6: Commit**

```bash
git add chessrl/model/network.py tests/test_network_vector_conditioning.py
git commit -m "feat(v2): FiLM vector goal-conditioning + embed() + win_vector"
```

---

### Task 3: `VectorGoalNetEvaluator` — batched vector inference + embeddings

**Files:**
- Modify: `chessrl/model/network.py` (add class near `BatchedGoalNetEvaluator`, ~line 275+)
- Test: `tests/test_network_vector_conditioning.py` (extend)

**Interfaces:**
- Consumes: `PolicyValueNet` vector mode (Task 2); `_scale_deadlines` (existing in `network.py`); `to_model_input`, `encode_board` (existing imports).
- Produces:
  - `VectorGoalNetEvaluator(net, device="cpu")` — asserts `net.goal_cond == "vector"`.
  - `evaluate_planes(planes_batch (N,NUM_PLANES,8,8), goal_vecs (N,d), deadlines (N,)) -> (policies (N,4672), values (N,))`.
  - `embed_boards(boards: list[chess.Board]) -> np.ndarray (N,d)` — the frozen-encoder embedding for Plan 2.
  - `win_value(planes_batch (N,NUM_PLANES,8,8), deadlines (N,)) -> values (N,)` — convenience: the head under `win_vector`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_network_vector_conditioning.py`:

```python
import numpy as np
import chess
from chessrl.chess_env.encoding import encode_board, to_model_input
from chessrl.model.network import VectorGoalNetEvaluator


def _planes(n=3):
    b = chess.Board()
    return np.stack([to_model_input(encode_board(b)) for _ in range(n)]).astype(np.float32)


def test_evaluator_requires_vector_net():
    bad = PolicyValueNet(NetworkConfig(blocks=2, filters=16))  # planes default
    with pytest.raises(AssertionError):
        VectorGoalNetEvaluator(bad)


def test_evaluate_planes_shapes():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    n, d = 3, _CFG.filters
    pol, val = ev.evaluate_planes(_planes(n), np.zeros((n, d), np.float32), np.zeros(n, np.float32))
    assert pol.shape == (n, 4672) and val.shape == (n,)
    assert val.min() >= 0.0 and val.max() <= 1.0


def test_embed_boards_shape():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    e = ev.embed_boards([chess.Board(), chess.Board()])
    assert e.shape == (2, _CFG.filters)


def test_win_value_uses_win_vector():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    n = 2
    wv = ev.win_value(_planes(n), np.zeros(n, np.float32))
    # equals evaluate_planes under the net's win_vector broadcast
    win_vecs = np.broadcast_to(net.win_vector.detach().numpy(), (n, _CFG.filters)).copy()
    _, val = ev.evaluate_planes(_planes(n), win_vecs, np.zeros(n, np.float32))
    assert np.allclose(wv, val, atol=1e-5)


def test_empty_batch():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    pol, val = ev.evaluate_planes(
        np.zeros((0, 21, 8, 8), np.float32), np.zeros((0, _CFG.filters), np.float32), np.zeros(0, np.float32)
    )
    assert pol.shape[0] == 0 and val.shape[0] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_network_vector_conditioning.py -v`
Expected: FAIL — `ImportError: cannot import name 'VectorGoalNetEvaluator'`.

- [ ] **Step 3: Implement the evaluator**

In `chessrl/model/network.py`, add after `BatchedGoalNetEvaluator`:

```python
class VectorGoalNetEvaluator:
    """Batched evaluator for the vector (FiLM) goal-conditioned net (v2).

    Conditions on a goal *vector* (cluster centroid) plus the deadline scalar,
    instead of goal planes. Also exposes ``embed_boards`` (the frozen-encoder
    embedding e(s) Plan 2 clusters) and ``win_value`` (the head under the
    reserved learned ``win_vector``)."""

    def __init__(self, net: "PolicyValueNet", device: str = "cpu"):
        assert getattr(net, "goal_cond", "none") == "vector", \
            "VectorGoalNetEvaluator requires a vector goal-conditioned net"
        self.device = device
        self.net = net.to(device)
        self.net.eval()

    @classmethod
    def from_checkpoint(cls, path, network_cfg: NetworkConfig, device: str = "cpu") -> "VectorGoalNetEvaluator":
        net = PolicyValueNet(network_cfg, goal_conditioned=True)
        ckpt = torch.load(Path(path), map_location=device)
        net.load_state_dict(ckpt["model"])
        return cls(net, device=device)

    @torch.no_grad()
    def evaluate_planes(self, planes_batch: np.ndarray, goal_vecs: np.ndarray, deadlines: np.ndarray):
        n = planes_batch.shape[0]
        d = self.net.win_vector.shape[0]
        if n == 0:
            return (np.zeros((0, self.net.policy_conv.out_channels * 64), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32))
        x = torch.from_numpy(planes_batch).to(self.device)
        gv = torch.from_numpy(np.asarray(goal_vecs, dtype=np.float32).reshape(n, d)).to(self.device)
        dl = torch.tensor(_scale_deadlines(deadlines).reshape(-1, 1), dtype=torch.float32, device=self.device)
        logits, value = self.net(x, deadline=dl, goal_vec=gv)
        policies = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        values = value.squeeze(1).cpu().numpy().astype(np.float32)
        return policies, values

    @torch.no_grad()
    def embed_boards(self, boards: list) -> np.ndarray:
        if not boards:
            return np.zeros((0, self.net.win_vector.shape[0]), dtype=np.float32)
        x = torch.from_numpy(np.stack([to_model_input(encode_board(b)) for b in boards])).to(self.device)
        return self.net.embed(x).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def win_value(self, planes_batch: np.ndarray, deadlines: np.ndarray) -> np.ndarray:
        n = planes_batch.shape[0]
        d = self.net.win_vector.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=np.float32)
        win_vecs = self.net.win_vector.detach().cpu().numpy().reshape(1, d).repeat(n, axis=0)
        _, values = self.evaluate_planes(planes_batch, win_vecs, deadlines)
        return values
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_network_vector_conditioning.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add chessrl/model/network.py tests/test_network_vector_conditioning.py
git commit -m "feat(v2): VectorGoalNetEvaluator (batched vector inference + embeddings)"
```

---

## Plan 1 deliverable

A `PolicyValueNet(cfg, goal_conditioned=True)` with `cfg.goal_cond="vector"` that conditions on a goal vector via FiLM, exposes `embed()` and a learned `win_vector`, and a `VectorGoalNetEvaluator` for batched inference + frozen-encoder embeddings — all green, with v1 `planes` mode and vanilla unchanged. This is the substrate Plan 2 (frozen-encoder goal-space + clustering + centroid achievement) builds on.

## Out of scope (later plans)

- Plan 2: `GoalSpace` (frozen-encoder snapshot, reservoir, online k-means, centroid achievement, persistence).
- Plan 3: HER/buffer rewrite onto cluster goals (`her.py`, `buffer.py`).
- Plan 4: means-end objective (win-value target + PBS shaping + α) , interventional win-value, curriculum term, ε-explore assignment, `experiments/v2-stage1.yaml`.
- Plan 5: eval (curve backfill, α-sweep) + live UI (cluster-id + win-value).
