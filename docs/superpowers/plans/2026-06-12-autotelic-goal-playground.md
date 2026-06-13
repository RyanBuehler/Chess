# Autotelic Goal Playground — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-agent autotelic learner (goal-conditioned AlphaZero) and run the
four-arm transfer experiment, per `docs/superpowers/specs/2026-06-12-autotelic-goal-playground.md`.

**Architecture:** Extend the existing AlphaZero stack with value-agnostic, deadline-bounded
goals expressed in the board's own feature vocabulary. A protagonist-frame **minimax** (not
negamax) goal-conditioned value head (sigmoid/BCE) is trained densely via HER; a learning-progress
curriculum selects goals. Four segregated arms (vanilla / always-win / random-goal / lp-goal) run
round-robin to 30k games each; Elo is measured post-hoc.

**Tech Stack:** Python, PyTorch, python-chess, NumPy, ZeroMQ feed, existing `chessrl/*` modules.

**Normative references:** the spec (above); base system `docs/superpowers/specs/2026-06-11-chess-rl-design.md`.

**Critical correctness gate:** the value-head/backup redesign (Stage 2) MUST reproduce the current
pipeline when `g = win` (regression test, Task 2.4). Do not proceed past Stage 2 until it passes.

---

## File structure

**New modules (`chessrl/goals/`):**
- `chessrl/goals/features.py` — rule-level state-feature extraction from a `chess.Board`.
- `chessrl/goals/templates.py` — goal template dataclass + the delta vocabulary + individuation.
- `chessrl/goals/verifier.py` — achieved-by-deadline? over a game record (exact).
- `chessrl/goals/encoding.py` — goal → conditioning planes + deadline scalar.
- `chessrl/goals/repertoire.py` — minting, child-spawning, per-template statistics.
- `chessrl/goals/curriculum.py` — Beta-Bernoulli LP estimator + sampling distribution.

**Modified:**
- `chessrl/config/config.py` — `GoalConfig`; wire into `RunConfig`/provenance.
- `chessrl/model/network.py` — conditioning planes, deadline FC side-input, sigmoid value head.
- `chessrl/mcts/reference.py` — protagonist-frame minimax + goal terminals.
- `chessrl/mcts/batched.py` — same, threaded through the copy-free leaf-parking path (riskiest).
- `chessrl/selfplay/play.py`, `concurrent.py`, `worker.py` — goal assignment, pure pursuit,
  switch-to-win, win-floor, record the assigned goal.
- `chessrl/training/{buffer,trainer,loop,parallel_loop}.py` — HER sample generation, BCE value
  loss, hung-worker watchdog.
- `chessrl/training/provenance.py` — record goal config.

**New scripts / experiments:**
- `scripts/round_robin.py` — the rotation orchestrator.
- `scripts/analyze_transfer.py` — post-hoc games-to-θ with bootstrapped CIs.
- `experiments/gp-vanilla.yaml`, `gp-always-win.yaml`, `gp-random-goal.yaml`, `gp-lp-goal.yaml`.

**Tests:** `tests/test_goal_features.py`, `test_goal_verifier.py`, `test_goal_encoding.py`,
`test_goal_network.py`, `test_mcts_goal_regression.py`, `test_value_deadline_monotonic.py`,
`test_her_samples.py`, `test_repertoire.py`, `test_curriculum.py`, `test_round_robin.py`.

---

## Stage 0 — Config and scaffolding

### Task 0.1: GoalConfig dataclass

**Files:** Modify `chessrl/config/config.py`; Test `tests/test_config.py` (extend).

- [ ] **Step 1: Write the failing test**
```python
def test_goal_config_defaults_and_modes():
    from chessrl.config.config import GoalConfig
    g = GoalConfig()
    assert g.goal_mode == "none"                 # vanilla default
    assert g.win_floor == 0.2
    assert g.lp_window == 200
    assert g.novelty_beta > 0
    assert GoalConfig(goal_mode="lp").goal_mode == "lp"
    import pytest
    with pytest.raises(ValueError):
        GoalConfig(goal_mode="bogus")            # validated in __post_init__
```

- [ ] **Step 2: Run it, verify it fails** — `pytest tests/test_config.py::test_goal_config_defaults_and_modes -v` → ImportError/AttributeError.

- [ ] **Step 3: Implement**
```python
@dataclass(frozen=True)
class GoalConfig:
    goal_mode: str = "none"          # none | always_win | random | lp
    win_floor: float = 0.2           # min fraction of games assigned g=win
    lp_window: int = 200             # attempts in the LP window
    novelty_beta: float = 1.0        # weight of the novelty bonus
    min_attempts_for_lp: int = 20    # gate LP on attempt count
    deadline_max: int = 60           # cap on goal deadline horizon (plies)
    def __post_init__(self):
        if self.goal_mode not in ("none", "always_win", "random", "lp"):
            raise ValueError(f"bad goal_mode {self.goal_mode}")
```
Add `goal: GoalConfig = field(default_factory=GoalConfig)` to `RunConfig`; parse from YAML/JSON.

- [ ] **Step 4: Run tests** — green.
- [ ] **Step 5: Commit** — `feat(config): add GoalConfig`.

### Task 0.2: Provenance records goal config

**Files:** Modify `chessrl/training/provenance.py`; Test `tests/test_provenance.py`.

- [ ] **Step 1:** Test asserts `build_provenance(cfg)["goal"]` includes `goal_mode`, `win_floor`, `lp_window`.
- [ ] **Step 2:** Run, fail.
- [ ] **Step 3:** Add a `goal` block to the provenance dict.
- [ ] **Step 4:** Green. **Step 5:** Commit `feat(provenance): record goal config`.

---

## Stage 1 — Verifier and goal representation (cheap, exact core)

### Task 1.1: Rule-level state features

**Files:** Create `chessrl/goals/features.py`; Test `tests/test_goal_features.py`.

The feature basis (value-agnostic): per-(piece-type, color) counts (12), side-to-move-in-check,
castling rights (4), and a result flag. Plus occupancy is read from the board directly for spatial
goals.

- [ ] **Step 1: Write the failing test**
```python
import chess
from chessrl.goals.features import board_features

def test_features_startpos():
    f = board_features(chess.Board())
    assert f.counts[(chess.PAWN, chess.WHITE)] == 8
    assert f.counts[(chess.QUEEN, chess.BLACK)] == 1
    assert f.in_check is False
    assert f.result is None

def test_features_after_capture():
    b = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    b.push(chess.Move.from_uci("e4d5"))           # exd5 captures the queen
    f = board_features(b)
    assert f.counts[(chess.QUEEN, chess.BLACK)] == 0
```

- [ ] **Step 2:** Run, fail (ImportError).
- [ ] **Step 3:** Implement `board_features(board) -> BoardFeatures` (a frozen dataclass with
  `counts: dict`, `in_check: bool`, `castling: tuple`, `result: str|None`).
- [ ] **Step 4:** Green. **Step 5:** Commit `feat(goals): rule-level state features`.

### Task 1.2: Goal templates and the delta vocabulary

**Files:** Create `chessrl/goals/templates.py`; Test `tests/test_goal_templates.py`.

A `GoalTemplate` is `(kind, params, deadline)` where `kind ∈ {count_delta, reach_square, check,
castle, promote, win}`, at the piece-type abstraction. Provide a canonical `key()` for repertoire
identity and an `is_win()` helper.

- [ ] **Step 1: Test**
```python
from chessrl.goals.templates import GoalTemplate, WIN_GOAL
def test_template_key_individuates_by_piece_type():
    g1 = GoalTemplate.capture(chess.KNIGHT, deadline=15)
    g2 = GoalTemplate.capture(chess.QUEEN, deadline=15)
    assert g1.key() != g2.key()                   # type individuated
    assert GoalTemplate.capture(chess.KNIGHT, 20).key() == g1.key()  # deadline not in identity
    assert WIN_GOAL.is_win()
```

- [ ] **Step 2:** Fail. **Step 3:** Implement. Note: deadline is NOT part of `key()` (a template's
  identity is the delta kind; deadline is a difficulty knob / child refinement). **Step 4:** Green.
  **Step 5:** Commit `feat(goals): goal templates + vocabulary`.

### Task 1.3: The verifier

**Files:** Create `chessrl/goals/verifier.py`; Test `tests/test_goal_verifier.py`.

Given a sequence of board states (a game record) and a `GoalTemplate` with a deadline measured from
a start ply, return `(achieved: bool, achieved_ply: int|None)` for a given protagonist color.

- [ ] **Step 1: Test (covers capture, promotion-as-positional, deadline boundary, protagonist)**
```python
from chessrl.goals.verifier import achieved_by_deadline
def test_capture_within_deadline():
    states = replay("1.e4 d5 2.exd5")            # helper builds board list
    g = GoalTemplate.capture(chess.PAWN, deadline=3)
    ok, ply = achieved_by_deadline(states, g, protagonist=chess.WHITE, start_ply=0)
    assert ok and ply == 4                        # exd5 at ply index 4 (0-based half-moves)
def test_missed_deadline_is_failure():
    g = GoalTemplate.capture(chess.QUEEN, deadline=2)
    ok, _ = achieved_by_deadline(replay("1.e4 d5 2.exd5"), g, chess.WHITE, 0)
    assert ok is False                            # capture happened after the deadline window
def test_promotion_reaches_rank8():
    g = GoalTemplate.reach_rank(chess.PAWN, rank=7, deadline=99)  # rank index 7 == 8th rank
    ok, _ = achieved_by_deadline(replay_promo(), g, chess.WHITE, 0)
    assert ok
```

- [ ] **Step 2:** Fail. **Step 3:** Implement by diffing `board_features`/occupancy across the
  state list within the deadline window, protagonist-relative. **Step 4:** Green.
  **Step 5:** Commit `feat(goals): exact verifier`.

### Task 1.4: Goal → conditioning planes

**Files:** Create `chessrl/goals/encoding.py`; Test `tests/test_goal_encoding.py`.

Produce the extra input planes for a goal (spatial mask + piece-type channel + scalar target
channels) and the deadline scalar (fed at the head, not as a plane). Plane count is fixed; the
goal is a sparse target. Mirror like the board encoding when Black is protagonist.

- [ ] **Step 1: Test**
```python
from chessrl.goals.encoding import encode_goal, GOAL_PLANES
def test_goal_planes_shape_and_compositional():
    p, deadline = encode_goal(GoalTemplate.capture(chess.KNIGHT, 15), remaining=10, protagonist=chess.WHITE)
    assert p.shape == (GOAL_PLANES, 8, 8)
    assert deadline == 10
    # a different piece type changes only the type channel, not the plane count (compositional)
    p2, _ = encode_goal(GoalTemplate.capture(chess.QUEEN, 15), remaining=10, protagonist=chess.WHITE)
    assert p2.shape == p.shape
def test_win_goal_planes_are_neutral():
    p, _ = encode_goal(WIN_GOAL, remaining=200, protagonist=chess.WHITE)
    assert p.shape == (GOAL_PLANES, 8, 8)         # win-goal: a reserved channel, no spatial mask
```

- [ ] **Step 2:** Fail. **Step 3:** Implement. **Step 4:** Green. **Step 5:** Commit
  `feat(goals): conditioning-plane encoding`.

---

## Stage 2 — Value-head/backup redesign + regression gate (CORRECTNESS GATE)

### Task 2.1: Network — conditioning planes, deadline side-input, sigmoid value head

**Files:** Modify `chessrl/model/network.py`; Test `tests/test_goal_network.py`.

Input becomes `21 + GOAL_PLANES` channels. The value head takes the conv features **plus** the
deadline scalar concatenated at its first Linear, and ends in **sigmoid** (achievement probability
in [0,1]). Policy head unchanged in shape but now conditioned via the input planes.

- [ ] **Step 1: Test**
```python
def test_value_head_is_sigmoid_and_takes_deadline():
    net = PolicyValueNet(blocks=2, filters=16)
    planes = torch.zeros(1, 21 + GOAL_PLANES, 8, 8)
    deadline = torch.tensor([[0.5]])
    pol, val = net(planes, deadline)
    assert 0.0 <= val.item() <= 1.0               # sigmoid range
    assert pol.shape == (1, 4672)
```

- [ ] **Step 2:** Fail. **Step 3:** Implement: widen stem `Conv2d(21+GOAL_PLANES, ch, 3)`; value
  head `... Flatten -> cat(deadline) -> Linear -> ReLU -> Linear -> Sigmoid`. Keep a constructor
  flag so a net with `GOAL_PLANES` zeroed + deadline=1 is well-defined for the always-win/vanilla
  bridge. **Step 4:** Green. **Step 5:** Commit `feat(net): goal-conditioned sigmoid value head`.

### Task 2.2: Reference MCTS — protagonist-frame minimax + goal terminals

**Files:** Modify `chessrl/mcts/reference.py`; Test `tests/test_mcts_goal_terminals.py`.

Tag each search with a `protagonist` and a `goal`. Value is `P(protagonist achieves goal)` in
[0,1]. Backup: protagonist-to-move node maximizes child value; opponent-to-move node **minimizes**
it (NO sign flip). Terminals: achieved → 1, deadline expired → 0, real game-over → evaluate goal
(win: win=1/draw=0.5/loss=0).

- [ ] **Step 1: Test** — a position one ply from a forced knight capture, goal=capture-knight,
  asserts the search's chosen move is the capture and root value → ~1; a position where the
  deadline expires next ply asserts root value → ~0.
- [ ] **Step 2:** Fail. **Step 3:** Implement the min/max selection by protagonist-vs-opponent
  node, replacing the `value = -value` negamax in the goal-conditioned path. **Step 4:** Green.
  **Step 5:** Commit `feat(mcts): protagonist-frame minimax + goal terminals (reference)`.

### Task 2.3: Deadline monotonicity test (training gate)

**Files:** Test `tests/test_value_deadline_monotonic.py` (+ small helper in network or eval).

- [ ] **Step 1: Test** — for a fixed (state, goal) and a *trained-tiny* or hand-constructed net
  wrapper, sweeping `remaining` from high→0 must produce non-increasing achievement probability,
  and `remaining==0 & not achieved` → ~0. (For the unit test, assert the *wiring* — that lower
  `remaining` cannot increase V given monotone-by-construction post-processing, OR mark this as the
  calibration check run during training and assert the harness exists.)
- [ ] **Step 2-4:** Implement the monotonicity calibration hook + test. **Step 5:** Commit
  `test(net): deadline monotonicity gate`.

### Task 2.4: REGRESSION GATE — g=win reduces to the current pipeline

**Files:** Test `tests/test_mcts_goal_regression.py`.

This is the hard gate. With `goal=WIN_GOAL` and no sub-goals, the goal-conditioned minimax search
must produce the **same visit distribution** as the existing negamax `reference.py` on fixed
positions, under the affine map `v_negamax = 2*p - 1` (draw=0.5).

- [ ] **Step 1: Test**
```python
def test_win_goal_matches_negamax_visit_distribution():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    visits_ref = legacy_negamax_search(board, sims=64, seed=0)        # pre-redesign reference
    visits_goal = goal_search(board, goal=WIN_GOAL, protagonist=board.turn, sims=64, seed=0)
    assert_visit_distributions_equal(visits_ref, visits_goal, tol=0)  # exact, same seed
```
Capture `legacy_negamax_search` from the current `reference.py` behavior (snapshot a frozen copy or
a git-tagged reference) before modifying it.

- [ ] **Step 2:** Run → must FAIL until the redesign is correct. **Step 3:** Fix the
  implementation until exact equality holds. **Step 4:** PASS. **Do not proceed to Stage 3 until
  this is green.** **Step 5:** Commit `test(mcts): g=win regression gate vs negamax`.

---

## Stage 3 — always-win + random-goal arms (apparatus end-to-end)

### Task 3.1: Batched MCTS goal-conditioning (riskiest change)

**Files:** Modify `chessrl/mcts/batched.py`; Test `tests/test_batched_goal_equivalence.py`.

Thread protagonist/goal/minimax + goal terminals through the copy-free leaf-parking path and the
goal-conditioned `evaluate_planes` (now `(N, 21+GOAL_PLANES, 8, 8)` + deadline vector). Preserve the
existing K=1 exact-equivalence gate against `reference.py` — now goal-conditioned.

- [ ] **Step 1: Test** — extend the existing batched-vs-reference equivalence test to the
  goal-conditioned path (K=1, fixed seed, a sub-goal AND the win-goal).
- [ ] **Step 2:** Fail. **Step 3:** Implement (park-time goal encoding alongside board encoding).
  **Step 4:** Green. **Step 5:** Commit `feat(mcts): goal-conditioned batched MCTS`.

### Task 3.2: Self-play goal assignment, pure pursuit, switch-to-win, win-floor

**Files:** Modify `chessrl/selfplay/{play,concurrent,worker}.py`; Test `tests/test_selfplay_goals.py`.

Each side is assigned a goal at game start (from the curriculum hook; for random-goal = uniform over
the repertoire with the win-floor; for always-win = WIN_GOAL). Search the assigned goal until
resolved (achieved/expired), then switch the active goal to WIN_GOAL. Record per-move: assigned
goal, active goal, visit counts. Enforce the win-floor at assignment time. Record fraction of plies
under g=win.

- [ ] **Step 1: Test** — a stubbed curriculum returns a 1-ply sub-goal; assert the game record shows
  the active goal switching to win after resolution, and that ≥`win_floor` of assignments are win
  over many games.
- [ ] **Step 2-4:** Implement. **Step 5:** Commit `feat(selfplay): goal assignment + switch-to-win + floor`.

### Task 3.3: HER sample generation + BCE value loss

**Files:** Modify `chessrl/training/{buffer,trainer}.py`; Test `tests/test_her_samples.py`.

At train-sample generation: policy targets only for the assigned goal (visit counts). Value targets
via HER — for the assigned goal and "future"-sampled achieved deltas, label achieved-by-deadline
(exact, via verifier), plus sampled negatives (incl. opponent-prevented deltas). Prefer
search-laundered targets where a goal-terminal was reached within search; weight raw HER positives
down, negatives up. Value loss = **BCE** (sigmoid). Samples generated from stored games (source of
truth), not persisted.

- [ ] **Step 1: Test** — given a stored game + a goal achieved at ply k, assert positive value
  samples for states before k (within deadline) and negative for a never-achieved goal; assert the
  trainer uses BCE on the value head and CE only on assigned-goal policy targets.
- [ ] **Step 2-4:** Implement. **Step 5:** Commit `feat(training): HER value samples + BCE loss`.

### Task 3.4: Wishful-thinking thermometer

**Files:** Modify `chessrl/training/loop.py` (metrics); Test `tests/test_her_samples.py` (extend).

- [ ] **Step 1:** Test asserts metrics emit, per goal, self-play achievement rate and (when eval
  data present) the self-play-vs-Stockfish achievement gap.
- [ ] **Step 2-4:** Implement metric emission. **Step 5:** Commit `feat(metrics): wishful-thinking thermometer`.

### Task 3.5: Experiment YAMLs (vanilla, always-win, random-goal) + end-to-end smoke

**Files:** Create `experiments/gp-vanilla.yaml`, `gp-always-win.yaml`, `gp-random-goal.yaml`;
Test `tests/test_goal_smoke.py` (slow).

All identical except `goal.goal_mode` (none / always_win / random); 6×64, sims 200, distinct
`feed_port` ranges; from scratch.

- [ ] **Step 1:** A slow smoke test runs ~few games per arm and asserts: runs start, checkpoints
  write, the regression gate still holds in-pipeline for always-win, no worker crashes.
- [ ] **Step 2-4:** Author YAMLs; fix integration. **Step 5:** Commit `feat(exp): vanilla/always-win/random-goal arms`.

---

## Stage 4 — LP curriculum (lp-goal arm)

### Task 4.1: Repertoire (minting + child-spawning + stats)

**Files:** Create `chessrl/goals/repertoire.py`; Test `tests/test_repertoire.py`.

- [ ] **Step 1: Test** — first-seen delta mints a template; re-seen does not; a template whose
  windowed success rate plateaus high spawns tighter-deadline children; stats track
  attempts/successes per template with a sliding window.
- [ ] **Step 2-4:** Implement (append-only identities; persisted alongside run state for resume).
  **Step 5:** Commit `feat(goals): repertoire minting + refinement`.

### Task 4.2: Beta-Bernoulli LP estimator + sampling

**Files:** Create `chessrl/goals/curriculum.py`; Test `tests/test_curriculum.py`.

- [ ] **Step 1: Test**
```python
def test_lp_gated_on_attempts_and_samples_frontier():
    cur = Curriculum(window=200, novelty_beta=1.0, min_attempts=20, win_floor=0.2)
    # untried template -> novelty-driven, not LP-driven
    # a template improving over the window -> high LP weight
    # a flat-at-zero (impossible) and flat-at-high (mastered) -> ~0 LP
    # win-goal sampled at >= win_floor regardless of LP
    ...
```

- [ ] **Step 2-4:** Implement Beta-Bernoulli posterior, windowed absolute-LP, attempt-count gating,
  `w(g) ∝ LP + β·novelty` with the win-floor applied on top. **Step 5:** Commit
  `feat(goals): LP curriculum`.

### Task 4.3: Wire curriculum into self-play + lp-goal YAML

**Files:** Modify `chessrl/selfplay/*` (use `Curriculum` when `goal_mode=="lp"`); Create
`experiments/gp-lp-goal.yaml`; Test `tests/test_goal_smoke.py` (extend to lp arm).

- [ ] **Steps:** Implement the mode switch (none/always_win/random/lp select the assignment source);
  author YAML; smoke-test. **Commit** `feat(exp): lp-goal arm + curriculum wiring`.

---

## Stage 5 — Execution, observability, evaluation

### Task 5.1: Round-robin orchestrator

**Files:** Create `scripts/round_robin.py`; Test `tests/test_round_robin.py`.

Rotate over the four arms, advancing each by 1,000 games per round via `train.py --resume
--games <current+1000>`, until each reaches 30,000. Distinct feed-port ranges per arm. Resilient to
restart (reads each arm's `state.json`).

- [ ] **Step 1: Test** — with a fake trainer command, assert the orchestrator computes the correct
  per-arm next-target sequence (1k increments), skips arms already at budget, and round-robins.
- [ ] **Step 2-4:** Implement. **Step 5:** Commit `feat(scripts): round-robin orchestrator`.

### Task 5.2: Hung-worker watchdog

**Files:** Modify `chessrl/training/parallel_loop.py`; Test `tests/test_parallel_loop.py` (extend).

Restart workers that are dead **or hung** (no new game in a heartbeat window via last-game mtime),
not just `not is_alive()`.

- [ ] **Step 1:** Test simulates a worker that stops producing games; assert it is restarted within
  the heartbeat window. **Step 2-4:** Implement. **Step 5:** Commit `fix(training): restart hung workers`.

### Task 5.3: Live observability (repertoire / LP / win-ply / thermometer)

**Files:** Modify feed/metrics emission (`chessrl/selfplay/feed.py`, `training/loop.py`) and the
web layer (`web/dashboard.js` or a new panel). **UI changes require the Playwright slow gate**
(`tests/test_ui_browser.py`) per project rule — vendored libs only, zero console errors.

- [ ] **Step 1:** Extend the browser gate: assert the new diagnostics panel renders for a goal run
  (repertoire size, per-goal LP, win-ply fraction) with no console errors.
- [ ] **Step 2-4:** Emit metrics; render panel (vendored, dark theme). **Step 5:** Commit
  `feat(ui): goal diagnostics panel`.

### Task 5.4: Post-hoc Elo sweep at training sims

**Files:** Modify `scripts/evaluate.py` (or add a sweep mode); Test `tests/test_eval_sweep.py`.

Sweep saved checkpoints, eval at **sims=200** (not 50), with a higher `games_per_rung` for tighter
noise. Goal arms conditioned on `g=win`. Emit Elo-vs-games per arm for `/compare.html`.

- [ ] **Step 1:** Test asserts the sweep enumerates checkpoints, evaluates at the configured sims,
  and writes per-arm Elo-vs-games. **Step 2-4:** Implement. **Step 5:** Commit
  `feat(eval): post-hoc checkpoint sweep at training sims`.

### Task 5.5: Transfer analysis (pre-registered metric)

**Files:** Create `scripts/analyze_transfer.py`; Test `tests/test_analyze_transfer.py`.

Compute games-to-Elo-θ per arm (θ read off vanilla), with seed-bootstrapped CIs, and apply the
pre-registered confirm/refute/inconclusive decision (spec §2).

- [ ] **Step 1:** Test on synthetic curves: confirm-case (≥15% fewer games, CI excludes 0),
  refute-case, inconclusive-case all classified correctly. **Step 2-4:** Implement.
  **Step 5:** Commit `feat(analysis): pre-registered transfer test`.

---

## Execution notes

- **Phase 1 (exploratory, 1 seed/arm):** after Stage 5, run all four arms round-robin to 30k. This
  validates the apparatus and reveals curve shape only — NOT a confirmation/refutation (spec §2).
- **Phase 2 (confirmatory, 3 seeds/arm):** only if Phase 1 warrants the ~2-week investment.
- The base runs (`baseline-*`, `arch-10x128-*`) remain paused; the GPU is free for the round-robin.

## Self-review checklist (run before execution)

- [ ] Spec coverage: every spec section (§2 criteria, §6 lifecycle, §7 floor, §8 redesign, §11 HER,
  §12 curriculum, §13 arms, §14 round-robin, §15 eval) maps to a task above.
- [ ] The regression gate (Task 2.4) is a hard blocker before Stage 3.
- [ ] No arm config differs except `goal.goal_mode` (+ feed ports/seed).
- [ ] UI task carries the Playwright gate.
- [ ] Type consistency: `GoalTemplate.key()`, `encode_goal`, `achieved_by_deadline`,
  `Curriculum` signatures match across tasks.
