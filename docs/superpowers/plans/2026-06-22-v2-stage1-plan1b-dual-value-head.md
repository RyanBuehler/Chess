# v2 Stage 1 — Plan 1b: Dual Value Head (tanh win + sigmoid goal)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Revise the Plan 1 vector goal-conditioned net from a single sigmoid value head to a **dual head**: a tanh/MSE game-outcome head `V(s,win)` off the *unconditioned* trunk (identical to vanilla), and a sigmoid achievement head `V_goal(s,g)` off the *FiLM-conditioned* features. Driven by the v1 ablation (single sigmoid achievement value cost ~500 Elo even for pure win-seeking).

**Architecture:** In vector mode, `forward(x, deadline, goal_vec)` returns a **3-tuple** `(logits, v_win, v_goal)`: policy + `v_goal` come from FiLM-conditioned features; `v_win` (tanh) comes from the unconditioned trunk and is goal-agnostic. `win_vector` is retained as the learned goal code used to *condition the policy/goal-head while pursuing winning* (it no longer drives the win value). Planes mode and vanilla are unchanged.

**Tech Stack:** Python, PyTorch, pytest. File: `chessrl/model/network.py`.

## Global Constraints

- `V(s,win)` = tanh in [-1,1] off the **unconditioned** trunk (goal-agnostic). `V_goal(s,g)` = sigmoid in [0,1] off the **FiLM-conditioned** features. Verbatim.
- Vector-mode `forward(x, deadline=None, goal_vec=None) -> (logits (B,4672), v_win (B,1), v_goal (B,1))`. `goal_vec` required in vector mode (raise `ValueError` if None).
- `d` = `NetworkConfig.filters`. Planes mode and vanilla (`goal_conditioned=False`) behavior UNCHANGED (still return 2-tuple `(logits, value)`).
- `win_vector` stays a learned `nn.Parameter (d,)` — conditions policy/goal-head during win-pursuit; does NOT feed the tanh win head.
- Run tests on Windows venv, unpiped/foreground: `.venv\Scripts\python.exe -m pytest <path> -v`.
- Stage only the named files; never `git add -A`.

---

### Task 1: Dual-head vector net + evaluator

**Files:**
- Modify: `chessrl/model/network.py` (PolicyValueNet vector branch + `forward`; `VectorGoalNetEvaluator`)
- Modify: `tests/test_network_vector_conditioning.py` (update vector-mode tests to the dual-head 3-tuple)

**Interfaces:**
- `PolicyValueNet(cfg, goal_conditioned=True)` with `cfg.goal_cond=="vector"`: builds `win_head` (tanh, unconditioned), `goal_head` (sigmoid, conditioned), `film`, `win_vector`.
- `forward(x, deadline=None, goal_vec=None)`: vector mode returns `(logits, v_win, v_goal)`; planes/vanilla unchanged (2-tuple).
- `VectorGoalNetEvaluator.evaluate_planes(planes, goal_vecs, deadlines) -> (policies (N,4672), v_win (N,), v_goal (N,))`; `win_value(planes, deadlines) -> (N,)` (the tanh head; goal-agnostic); `embed_boards` unchanged.

- [ ] **Step 1: Update the failing tests**

In `tests/test_network_vector_conditioning.py`, REPLACE the vector-mode forward/conditioning tests (the ones that unpacked `logits, value = net(...)` and asserted a single sigmoid value, and the win_value equivalence test) with these. Leave Task-1/planes/vanilla tests intact.

```python
def test_vector_forward_dual_head_shapes_and_ranges():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    d = _CFG.filters
    x = _board_batch(4); gv = torch.randn(4, d); dl = torch.zeros(4, 1)
    logits, v_win, v_goal = net(x, deadline=dl, goal_vec=gv)
    assert logits.shape == (4, 4672)
    assert v_win.shape == (4, 1) and v_goal.shape == (4, 1)
    assert float(v_win.min()) >= -1.0 and float(v_win.max()) <= 1.0   # tanh
    assert float(v_goal.min()) >= 0.0 and float(v_goal.max()) <= 1.0  # sigmoid


def test_vector_requires_goal_vec():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    with pytest.raises(ValueError):
        net(_board_batch(2), deadline=torch.zeros(2, 1), goal_vec=None)


def test_win_head_is_goal_agnostic():
    net = PolicyValueNet(_CFG, goal_conditioned=True).eval()
    d = _CFG.filters; x = _board_batch(1); dl = torch.zeros(1, 1)
    with torch.no_grad():
        for p in net.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, std=0.1)
        _, vw_a, vg_a = net(x, deadline=dl, goal_vec=torch.full((1, d), -2.0))
        _, vw_b, vg_b = net(x, deadline=dl, goal_vec=torch.full((1, d), 2.0))
    assert abs(float(vw_a) - float(vw_b)) < 1e-6      # win value invariant to goal
    assert abs(float(vg_a) - float(vg_b)) > 1e-5      # goal value varies with goal
```

Also update `test_planes_mode_unchanged` only if it referenced vector mode (it does not — leave it). Update the `VectorGoalNetEvaluator` tests:

```python
def test_evaluate_planes_dual_head_shapes():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    n, d = 3, _CFG.filters
    pol, vw, vg = ev.evaluate_planes(_planes(n), np.zeros((n, d), np.float32), np.zeros(n, np.float32))
    assert pol.shape == (n, 4672) and vw.shape == (n,) and vg.shape == (n,)
    assert vw.min() >= -1.0 and vw.max() <= 1.0
    assert vg.min() >= 0.0 and vg.max() <= 1.0


def test_win_value_is_goal_agnostic():
    net = PolicyValueNet(_CFG, goal_conditioned=True)
    ev = VectorGoalNetEvaluator(net)
    n, d = 2, _CFG.filters
    wv = ev.win_value(_planes(n), np.zeros(n, np.float32))
    # equals v_win from evaluate_planes under ANY goal vectors (goal-agnostic)
    _, vw, _ = ev.evaluate_planes(_planes(n), np.full((n, d), 3.0, np.float32), np.zeros(n, np.float32))
    assert np.allclose(wv, vw, atol=1e-5)
```

Delete the old `test_win_value_uses_win_vector` (win value no longer routes through the win vector).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_network_vector_conditioning.py -v`
Expected: FAIL — vector forward still returns a 2-tuple; `v_win`/dual-head don't exist.

- [ ] **Step 3: Implement the dual head**

In `chessrl/model/network.py`, in `PolicyValueNet.__init__`, REPLACE the `if goal_conditioned and cfg.goal_cond == "vector":` block with:

```python
        if goal_conditioned and cfg.goal_cond == "vector":
            self.win_vector = nn.Parameter(torch.zeros(d))   # conditions policy/goal-head when pursuing win
            self.film = FiLM(in_dim=d + 1, channels=ch)
            # WIN value: tanh game-outcome off the UNCONDITIONED trunk (like vanilla).
            self.win_head = nn.Sequential(
                nn.Conv2d(ch, 8, 1), nn.BatchNorm2d(8), nn.ReLU(), nn.Flatten(),
                nn.Linear(8 * 64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh(),
            )
            # GOAL achievement: sigmoid off the FiLM-CONDITIONED features.
            self.goal_head = nn.Sequential(
                nn.Conv2d(ch, 8, 1), nn.BatchNorm2d(8), nn.ReLU(), nn.Flatten(),
                nn.Linear(8 * 64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid(),
            )
```

REPLACE the vector branch of `forward` with:

```python
        if self.goal_cond == "vector":
            if goal_vec is None:
                raise ValueError("vector goal-conditioned net requires goal_vec")
            if deadline is None:
                deadline = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            if deadline.dim() == 1:
                deadline = deadline.unsqueeze(1)
            v_win = self.win_head(h)                              # unconditioned, goal-agnostic
            cond = torch.cat([goal_vec.to(h.dtype), deadline.to(h.dtype)], dim=1)
            h_cond = self.film(h, cond)
            logits = self.policy_conv(h_cond).permute(0, 2, 3, 1).flatten(1)
            v_goal = self.goal_head(h_cond)
            return logits, v_win, v_goal
```

(`h = self.tower(self.stem(x))` is computed once at the top of `forward`, before this branch — keep it. The planes and vanilla branches that follow are UNCHANGED and still return 2-tuples.)

In `VectorGoalNetEvaluator.evaluate_planes`, update the forward unpack + returns:

```python
    @torch.no_grad()
    def evaluate_planes(self, planes_batch, goal_vecs, deadlines):
        n = planes_batch.shape[0]
        d = self.net.win_vector.shape[0]
        if n == 0:
            z = np.zeros((0, self.net.policy_conv.out_channels * 64), dtype=np.float32)
            return z, np.zeros((0,), np.float32), np.zeros((0,), np.float32)
        x = torch.from_numpy(planes_batch).to(self.device)
        gv = torch.from_numpy(np.asarray(goal_vecs, np.float32).reshape(n, d)).to(self.device)
        dl = torch.tensor(_scale_deadlines(deadlines).reshape(-1, 1), dtype=torch.float32, device=self.device)
        logits, v_win, v_goal = self.net(x, deadline=dl, goal_vec=gv)
        policies = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        return policies, v_win.squeeze(1).cpu().numpy().astype(np.float32), v_goal.squeeze(1).cpu().numpy().astype(np.float32)
```

Update `win_value` (goal-agnostic now — pass the win_vector broadcast as a valid goal_vec; v_win ignores it):

```python
    @torch.no_grad()
    def win_value(self, planes_batch, deadlines):
        n = planes_batch.shape[0]
        d = self.net.win_vector.shape[0]
        if n == 0:
            return np.zeros((0,), np.float32)
        win_vecs = self.net.win_vector.detach().cpu().numpy().reshape(1, d).repeat(n, axis=0)
        _, v_win, _ = self.evaluate_planes(planes_batch, win_vecs, deadlines)
        return v_win
```

`embed_boards` is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_network_vector_conditioning.py -v`
Expected: PASS (all). Then regression: `.venv\Scripts\python.exe -m pytest tests/test_goal_network.py -v` — PASS (planes path unchanged).

- [ ] **Step 5: Commit**

```bash
git add chessrl/model/network.py tests/test_network_vector_conditioning.py
git commit -m "feat(v2): dual value head (tanh win + sigmoid goal) per ablation finding"
```

---

## Plan 1b deliverable

The vector net now has a vanilla-strength tanh win head (the RL target) plus a sigmoid goal-achievement head (shaping only), addressing the ablation's ~500 Elo machinery tax. Downstream Plans 3–4 consume `(logits, v_win, v_goal)` and use `v_win` as the outcome target, `v_goal` as the shaping potential.
