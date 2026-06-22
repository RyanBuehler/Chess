# v2 Stage 1 — Emergent Means-End Goals: Design Spec

**Date:** 2026-06-19
**Status:** Approved (brainstorm), pending implementation plan
**Roadmap context:** `docs/notes/research-explorations.md` → "v2 ROADMAP (2026-06-19)". This spec covers
**Stage 1** only. Stages 0 (machinery viability, in flight), 2 (goal→goal graph), 3 (hierarchical
controller), 4 (continuous latent goals) are out of scope.

---

## Goal

The first goal system that makes **discovered goals serve winning**, fixing v1's speedrunner
pathology (goals pursued as ends, win-gate off, ~−400 Elo). Success is measured by **learning**
and **rate of progression**, not by beating the baseline.

One sentence: *goals are emergent state-deltas discovered from play (clusters in the value net's
own embedding space), each trainable to pursue, valued causally by how much they raise P(win), and
subordinate to winning via potential-based shaping.*

## Non-Goals (scope fence)

- **No goal→goal graph** (Stage 2). Win-value is per-goal vs the win outcome only.
- **No manager/controller, no multi-goal plans** (Stage 3). One goal per side per game, as in v1.
- **No continuous latent goal selection** (Stage 4). Goals are discretized clusters.
- **No trunk redesign.** Reuse the existing ResNet trunk. v2 DOES add a goal-conditioning pathway
  and uses a **DUAL value head** (see "Goal conditioning" — REVISED 2026-06-22): a tanh/MSE
  game-outcome head off the unconditioned trunk for `V(s,win)` (identical to vanilla's head), plus a
  sigmoid achievement head off the FiLM-conditioned features for `V_goal(s,g)`. This reverses the
  earlier single-sigmoid-head plan: the v1 ablation showed the sigmoid achievement value costs ~500
  Elo even for pure win-seeking, so the win objective must be the strong tanh/MSE target. Network
  shape changes → new `provenance.json`.

What changes vs v1: **goal discovery** (no hardcoded vocabulary), **causal win-valuation**, and the
**means-end objective**. Everything else (one goal per side, switch-to-win on resolve, HER buffer,
sequential round-robin harness) stays.

---

## Architecture

### Goal representation — learned-embedding clusters

A **goal** is a target cluster in the value net's learned state-delta space.

- **Encoder:** the value net **trunk** (penultimate shared features before the policy/value heads),
  giving `e(s) ∈ R^d`. No new parameters; goals live in the representation the net already learns.
- **Delta:** over a goal window starting at ply `start`, the realized delta is
  `Δ(s) = e(s) − e(s_start)`.
- **Discretization:** maintain a reservoir of recent `Δ` vectors and fit **online k-means**
  (default `K = 48` clusters). Centroids `{c_1…c_K}` are the discovered goal codes. Re-fit every
  `refresh_every` games (default 2000). Cluster membership is by nearest centroid.
- **Encoder versioning — FROZEN SNAPSHOT (decision, 2026-06-19).** The embedding `e(·)` used to
  *define and cluster* goals comes from a **frozen snapshot** of the trunk, NOT the live training
  net. Goals are defined and clustered in that frozen embedding for an *epoch*; at each
  `refresh_every` the snapshot is re-taken from the current net and clusters are re-fit. This makes
  the goal space **stationary within an epoch** — HER targets are well-defined, achievement is
  reproducible, and non-stationarity becomes a discrete, controlled re-fit rather than per-step
  drift (the live-encoder alternative, where goals drift every gradient step, is deferred to Stage
  4). The frozen encoder snapshot is persisted to the run dir and reloaded on resume.
- **Achievement test (replaces the predicate verifiers):** goal `g = c_k` is *achieved at ply t*
  iff `argmin_j ‖Δ(s_t) − c_j‖ == k` AND `‖Δ(s_t) − c_k‖ ≤ τ` (default `τ` = median intra-cluster
  radius at last refit), within the goal window/deadline. This is a **new verification path that
  replaces** v1's board-predicate verifiers (`chessrl/goals/verifier.py`, `features.py`,
  `_goal_achieved`) for discovered goals — a cluster centroid has no board-predicate meaning. The
  **WIN goal keeps a real game-outcome check** (it is not a cluster). Resolution (achieved OR
  deadline) switches the active goal to WIN, preserving v1's "every game ends win-directed."

This **replaces `_CANDIDATE_KINDS`** entirely. Clusters can be labelled post-hoc (inspect member
positions) for the UI/research read, but the agent never sees a hand-named category.

### Goal conditioning (how `g` reaches the net)

v1 fed the goal to the net as **board planes** (`GOAL_PLANES` = spatial mask + piece-type + kind
one-hots) plus a **deadline scalar** at the value FC, routed through `encode_goal`,
`GoalNetEvaluator`, and `BatchedGoalNetEvaluator`. Cluster goals live in *embedding* space and have
no board-plane rendering, so Stage 1 **replaces that entire interface** with a **vector pathway**:
the goal's centroid `c_k ∈ R^d` (concatenated with the scaled deadline scalar) is projected by a
small MLP and injected into the trunk output via **FiLM** (feature-wise affine modulation) before
the policy and value heads. This touches every evaluator that currently builds goal planes — they
switch from `(board_planes ⊕ goal_planes, deadline)` to `(board_planes, goal_vector)`.

**DESIGN REVISION (2026-06-22), driven by the v1 ablation:** the ablation showed the v1
single-sigmoid-achievement value head costs **~500 Elo even when the goal is always WIN**
(`vanilla 754` vs `always-win 230`; see decision log). The sigmoid achievement-probability value is
the prime suspect. So v2 uses a **DUAL value head**, not one:

- **WIN value head — tanh/MSE game-outcome**, read from the **UNCONDITIONED trunk** (winning is
  goal-agnostic). This is exactly vanilla's value head: `V(s,win) = tanh_head(trunk(s))` in [-1,1].
  It is the RL training target, so v2's win objective is represented identically to the strong
  vanilla baseline — *not* as a sigmoid achievement probability.
- **GOAL achievement head — sigmoid/BCE**, read from the **FiLM-CONDITIONED features**:
  `V_goal(s,g) = sigmoid_head(FiLM(trunk(s), c_g))` in [0,1] = `P(achieve g)`. Used ONLY for
  sub-goal shaping, never as the win objective.

Both heads come from one forward pass (no double-eval). The reserved WIN goal vector `c_win` is no
longer needed for the win *value* (the tanh head is goal-agnostic); the policy and goal head still
take a goal vector `c_g`.

**Coexistence (not in-place removal).** The vector pathway is added as a **new conditioning mode**
(`goal_cond="vector"`) alongside v1's existing `planes` mode, rather than ripping `GOAL_PLANES` out.
Rationale: vanilla and the v1 goal arms must stay runnable so their Elo-vs-games curves remain the
comparison baselines (success is curve-vs-curve). v2 selects `vector`; v1 keeps `planes`. The
legacy planes path is removed only once v1 arms are retired (out of scope here). The v2 net shape
differs from v1 → record in `provenance.json`.

### Causal per-goal win-value (de-confounded)

Observational goal→win lift is confounded (winning side achieves more of everything; see notes A).
Stage 1 estimates win-value **interventionally**:

- **ε-explore branch** in assignment: with probability `ε` (default 0.15), ignore the curriculum
  and assign a **uniformly random** cluster — this is the `do(assign g)` intervention. Otherwise
  sample from the curriculum.
- **Estimator** (`winvalue.py`): per cluster, a Beta posterior over `P(win | do(assign g))` built
  **only from ε-explore games** (the de-confounded sample), phase-stratified (opening/midgame/
  endgame). `win_value(g) = E[P(win | do g)] − base_winrate`.
- Updated each time an ε-explore game resolves; refreshed alongside cluster re-fit (centroids move,
  so stale clusters age out with their stats).

### Means-end objective

**FRAMING (2026-06-22, Ryan): "winning" is NOT hardcoded — it is the environment's TERMINAL /
EXTRINSIC reward.** The agent grounds on `r_extrinsic` (a configurable signal; in *this* chess
environment `r_extrinsic` = game outcome z ∈ {-1,0,1}), and self-invented cluster goals are the
**intrinsic** layer on top — the standard intrinsic-motivation formulation `r = r_extrinsic +
β·r_intrinsic`. Nothing in the net or search says "win": the tanh head predicts *expected terminal
reward*, and in chess that target happens to be the outcome. The code symbols (`v_win`, `win_head`)
are the chess instance of this general concept, not a chess-specific objective. The grounding signal
is necessary — v1 (no grounding, pure autotelic) collapsed to ~230 Elo vs 754 — but it is provided
by the environment, not wired into the architecture. **Emergent terminal goal** (the agent
*discovering* which goal is terminal by how chains of goals lead to `r_extrinsic`) is the Stage 2→3
hierarchy payoff, built on this grounded Stage 1; it is explicitly out of scope here.

While a side is assigned goal `g` (with the DUAL head above: the tanh head = expected terminal
reward `V(s,r_ext)` (a.k.a. `V(s,win)` in chess), `V_goal(s,g)` the sigmoid achievement head — both
from one forward):

- **RL value target = expected terminal reward** (tanh outcome head; chess: game outcome, like
  vanilla). Default `α = 0`: maximizing terminal reward is the objective, always. Resignation gate
  **on**. This is the change that removes the ~500 Elo machinery tax — the objective is now the
  strong tanh/MSE terminal-reward target, not a sigmoid achievement probability.
- **Potential-based shaping** adds reward `F = γ_shape · (Φ(s′; g) − Φ(s; g))` with potential
  `Φ(s; g) = V_goal(s, g)` (the sigmoid goal head). By the potential-based-shaping theorem this
  cannot make the agent sacrifice the game for the goal — it only accelerates credit toward goal
  progress. Default `γ_shape = 0.25` (× the per-step discount, standard PBS form).
- **α-blend knob (for sweeps):** the training value target is
  `(1−α)·V(s,win) + α·V_goal(s,g)`. `α = 1` reproduces v1 speedrunning; `α = 0` is pure
  means-end. The α=1→0 sweep is the core mechanistic experiment. (Note `V_goal` is [0,1] and
  `V(s,win)` is [-1,1]; the blend maps `V_goal` to [-1,1] via `2·V_goal−1` so the targets share a
  scale.)

### Curriculum

`w(g) = LP(g) + β·novelty(g) + γ·max(0, win_value(g))` with the existing win-floor
(≥ some fraction of assignments forced to WIN). `γ` default 1.0; win-value is a lift (can be
negative → clamped to 0 so harmful goals are simply not up-weighted, never down-weighted below
novelty/LP).

---

## Components (files)

| File | Change | Responsibility |
|------|--------|----------------|
| `chessrl/goals/discovery.py` | **new** | `GoalSpace`: trunk-embed states, form deltas, online k-means, assign/achieve, refresh. |
| `chessrl/goals/winvalue.py` | **new** | Interventional per-cluster win-value (Beta, phase-stratified, ε-only accounting). |
| `chessrl/goals/assignment.py` | modify | ε-Bernoulli explore branch + curriculum sample. |
| `chessrl/goals/curriculum.py` | modify | add `γ·win_value(g)` term. |
| `chessrl/goals/repertoire.py` | retire/bypass | `_CANDIDATE_KINDS` no longer the goal source. |
| `chessrl/goals/verifier.py`, `features.py`, `encoding.py` | retire/bypass for clusters | predicate verifiers + `GOAL_PLANES` replaced by centroid-distance achievement + vector conditioning (WIN keeps a real outcome check). |
| `chessrl/model/network.py` | modify | FiLM goal-conditioning from `(centroid ⊕ deadline)` vector (coexists with v1 planes mode); **DUAL value head**: tanh outcome head off unconditioned trunk (`V(s,win)`) + sigmoid achievement head off conditioned features (`V_goal(s,g)`); policy off conditioned features. |
| self-play objective (`chessrl/selfplay/…`) | modify | RL target = tanh `V(s,win)` + PBS shaping `Φ=V_goal(s,g)` (sigmoid head) + α knob; gate on. |
| `experiments/v2-stage1.yaml` | **new** | `goal_mode: emergent`, ε/K/α/γ_shape config. |
| eval/UI | small add | render cluster-id + win-value in the live aux; backfill vanilla curve. |

## Data flow (per game)

1. Assign `g` — curriculum (prob 1−ε) or uniform-random (prob ε, flagged as explore).
2. Play under win-value target + goal shaping; gate on.
3. Each ply: achievement test; on achieve-or-deadline, switch active goal to WIN.
4. Log `(g, explore?, achieved?, outcome, phase, start_ply)`.
5. Periodically (`refresh_every` games): re-fit clusters from the reservoir; recompute `τ`; update
   win-values from accumulated explore games.

## Error handling / degeneracy guards

- **Cold start:** until the reservoir has ≥ `min_reservoir` deltas (default 5000), `GoalSpace`
  falls back to WIN-only assignment (no clusters yet). Logged.
- **Degenerate clusters** (one giant cluster / empty clusters): refit drops empty clusters and
  splits the largest; if K collapses below `K_min` (default 8), widen the reservoir window. Logged
  as a health metric.
- **Sparse win-value:** a cluster with < `min_explore` (default 30) explore games reports
  `win_value = 0` (no up-weight) until enough interventional data accrues.
- **Resume:** `GoalSpace` centroids + reservoir + win-value posteriors persist to the run dir and
  reload on resume (alongside the existing HER buffer rebuild).

---

## Success criteria

**Framing (Ryan, 2026-06-19): we are testing whether it LEARNS and how FAST — not whether it beats
vanilla.**

1. **Primary — is it learning?** v2-stage1's Elo-vs-**games** curve shows genuine upward
   progression end-to-end. A flat/degenerate curve is the failure mode.
2. **Bar — competitiveness, not dominance.** *Closely tracking* vanilla's Elo-vs-games curve counts
   as success. Matching/beating is bonus; beating is **not** required. Avoiding v1's −400 collapse
   is the real bar.
3. **Comparison is curve-vs-curve (rate of progression).** Plot both arms' Elo-vs-games on
   `/compare.html`; compare slopes and position-at-matched-games, not a single endpoint. Backfill
   vanilla's curve from its existing checkpoints for a fair comparison.
4. **Mechanistic:** the α=1→0 sweep shows Elo improving as goals subordinate to winning.
5. **Sanity:** discovered clusters are inspectable (post-hoc labels make chess sense); win-values
   are non-degenerate (spread, not all ~0).

## Evaluation

- Milestone Elo eval (existing `scripts/eval_milestone.py`) at several game counts (e.g.
  2.5k/5k/7.5k/10k) for v2-stage1, to draw a curve rather than a point.
- Backfill vanilla at the same milestones from its existing `_checkpoints`.
- α-sweep: a small set of short runs at α ∈ {1.0, 0.5, 0.0} (shared discovery/curriculum), Elo at a
  fixed budget, to chart the means↔ends axis.

## Testing

- `GoalSpace`: delta encoding shape/determinism; nearest-centroid assignment; achievement test
  boundary (`τ`); refresh drops empties / splits largest; cold-start fallback.
- `winvalue`: Beta update math; **explore-only** accounting (curriculum games excluded);
  phase stratification; sparse-cluster `win_value = 0`.
- `assignment`: ε-Bernoulli rate over many draws; explore picks uniform over live clusters.
- shaping: PBS form; net shaping = 0 along a constant-`Φ` trajectory; α=0 target equals `V(win)`,
  α=1 equals `V_goal`.
- conditioning (DUAL head): the **tanh win head is goal-agnostic** — `V(s,win)` is invariant to the
  goal vector (FiLM does not feed it); the **sigmoid goal head varies** with the goal vector (FiLM
  conditions — `V_goal(s,c_a) ≠ V_goal(s,c_b)` for distinct centroids); forward accepts
  `(board_planes, goal_vector)` and returns `(policy, v_win tanh, v_goal sigmoid)`; vector mode uses
  no `GOAL_PLANES` channels.
- Integration: a short smoke run produces clusters, explore games, non-trivial win-values, and a
  rising metric, without crashing on resume.
- UI: live aux renders cluster-id + win-value (Playwright slow gate, per project rule).

---

## Open knobs (defaults set, tunable in plan)

`K=48`, `refresh_every=2000`, `ε=0.15`, `α=0` (operating), `γ_shape=0.25`, `γ=1.0`,
`min_reservoir=5000`, `min_explore=30`, `K_min=8`. Goal window/deadline inherits v1's settings.
