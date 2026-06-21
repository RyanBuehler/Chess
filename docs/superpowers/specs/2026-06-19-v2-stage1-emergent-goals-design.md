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
- **No trunk redesign / no new heads.** Reuse the existing trunk + `V(s,win)` head + goal head
  `V_goal(s,g)`. One small addition is unavoidable (see "Goal conditioning" below): a
  goal-conditioning pathway, because embedding-space cluster goals cannot be fed as board planes
  the way v1 fed goal-kinds. This changes the network shape → new `provenance.json`.

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
- **Achievement test:** goal `g = c_k` is *achieved at ply t* iff
  `argmin_j ‖Δ(s_t) − c_j‖ == k` AND `‖Δ(s_t) − c_k‖ ≤ τ` (default `τ` = median intra-cluster
  radius at last refit), within the goal window/deadline. Resolution (achieved OR deadline) switches
  the active goal to WIN, preserving v1's "every game ends win-directed."

This **replaces `_CANDIDATE_KINDS`** entirely. Clusters can be labelled post-hoc (inspect member
positions) for the UI/research read, but the agent never sees a hand-named category.

### Goal conditioning (how `g` reaches the net)

v1 fed the goal to the net as **board planes** (target squares etc. for hand-named kinds). Cluster
goals live in *embedding* space and have no board-plane rendering, so Stage 1 conditions via a
**vector pathway** instead: the goal's centroid `c_k ∈ R^d` is projected by a small MLP and
injected into the trunk output via **FiLM** (feature-wise affine modulation) before the policy and
`V_goal` heads. This is the one new sub-module. `V(s,win)` reads the *unconditioned* trunk output
(winning is goal-agnostic); policy and `V_goal` read the conditioned features. The old goal-plane
input channels are removed. Network shape changes → record in `provenance.json`.

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

While a side is assigned goal `g`:

- **RL value target = `V(s, win)`** (protagonist win-value). Default `α = 0`: winning is the
  objective, always. Resignation gate **on**.
- **Potential-based shaping** adds reward `F = γ_shape · (Φ(s′; g) − Φ(s; g))` with potential
  `Φ(s; g) = V_goal(s, g)` (the goal head). By the potential-based-shaping theorem this cannot make
  the agent sacrifice the game for the goal — it only accelerates credit toward goal progress.
  Default `γ_shape = 0.25` (× the per-step discount, standard PBS form).
- **α-blend knob (for sweeps):** the training value target is
  `(1−α)·V(s,win) + α·V_goal(s,g)`. `α = 1` reproduces v1 speedrunning; `α = 0` is pure
  means-end. The α=1→0 sweep is the core mechanistic experiment.

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
| network module (`chessrl/nn/…`) | modify | FiLM goal-conditioning pathway (centroid→MLP→affine on trunk output); drop goal-plane inputs. |
| self-play objective (`chessrl/selfplay/…`) | modify | win-value target + PBS shaping + α knob; gate on. |
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
- conditioning: `V(s,win)` output is invariant to the goal input; policy and `V_goal` outputs vary
  with the goal centroid (FiLM actually conditions).
- Integration: a short smoke run produces clusters, explore games, non-trivial win-values, and a
  rising metric, without crashing on resume.
- UI: live aux renders cluster-id + win-value (Playwright slow gate, per project rule).

---

## Open knobs (defaults set, tunable in plan)

`K=48`, `refresh_every=2000`, `ε=0.15`, `α=0` (operating), `γ_shape=0.25`, `γ=1.0`,
`min_reservoir=5000`, `min_explore=30`, `K_min=8`. Goal window/deadline inherits v1's settings.
