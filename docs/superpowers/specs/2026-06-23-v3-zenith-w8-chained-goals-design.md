# v3-zenith-w8 — Chained Means-End Goals (Design, rev. 2)

**Date:** 2026-06-23 (rev. 2 after adversarial review)
**Codename:** `v3-zenith-w8` (lineage: `v2-halcyon` = flat Stage-1 goals; this = greedy chaining; `v4` = learned goal→goal graph, see `2026-06-23-v4-learned-goal-graph-design.md`)

## Goal

Evolve v2-halcyon's **flat** means-end self-play — one ~8-ply subgoal, then win-pursuit for the
rest of the game (~9% of plies goal-directed) — into **chained** means-end: each side always holds
an active subgoal, re-selected on achievement-or-expiry, with goal-influence smoothly fading to pure
win-pursuit as the position becomes decisive. Goal scale stays `goal_window=8` to isolate **goal
scale** as a variable (vs a future `v3-zenith-w16`). Train to 10k; compare Elo head-to-head with
v2-halcyon (853@5k, 924@10k) and vanilla@10k.

## Honest scope note (from review)

v3 is **not** a single-variable ablation of "chaining." It bundles three changes vs v2: (a)
reassign-on-achieve/expire, (b) state-dependent next-goal selection, (c) the α-schedule win-apex.
`goal_window=8` cleanly isolates *goal scale* (vs w16 later); v3-vs-v2 is a **design** comparison
(chaining package vs flat), so a win/loss can't be pinned to any single one of (a)/(b)/(c).

## Why

v2-halcyon proved the substrate works (goal head trains every ply via HER; Elo climbs 853→924, no v1
collapse) but goals shape *behavior* only ~9% of plies. Chaining + the α-schedule make the whole
contested phase of the game goal-directed, so the goal knowledge the net already learns drives the
moves — while decisive positions still get pure-strength play.

## Components (all in means-end self-play; net / MCTS core / HER schema / eval unchanged)

### 1. State-dependent next-goal selection — `select_next_goal(...)`
1. One **batched** eval of the current state conditioned on all K cluster centroids → `v_goal(s, g)`
   for every cluster, plus goal-agnostic `v_win(s)`.
2. Score: **`score(g) = curriculum_weight(g) · v_goal(s, g)`**, where `curriculum_weight(g) =
   β·novelty(g) + γ·max(0, win_value(g))` — the SAME weight `ClusterCurriculum` already uses to
   sample (never degenerate: the novelty term floors it > 0 for unattempted clusters). *[Fixes C1:
   raw `win_value` is centered/sparse/often-exactly-0 → product was near-random.]*
3. Choose a subgoal by **softmax-sample** over `score(g)` with temperature `goal_select_temp`. With
   prob `epsilon` (existing 0.15) inject a uniform-random cluster (interventional, unchanged).
4. There is **no discrete "win" candidate** — win-pursuit emerges via the α-schedule (component 3),
   so the agent always holds a real subgoal but its influence vanishes when decisive.

### 2. Reassignment lifecycle — `maybe_reassign_goal(...)` (replaces `maybe_switch_cluster_to_terminal`)
Reassign (call `select_next_goal`) when the active subgoal is **achieved** OR **expires**:
- **Achieved (live):** each ply, test `goalspace.achieved(e(s_now) − e(s_goal_start), active_cluster)`
  (cache `e(s_goal_start)` at assignment). *Known limitation (I2):* τ is calibrated on
  `goal_window`-spaced deltas, so sub-window gaps under-detect → achievement fires mostly near the
  8-ply boundary early in training, self-correcting as the net learns to reach goals faster. Accepted.
- **Expired:** `goal_window` (8) plies elapsed since the goal's start.
Game-start assignment also uses `select_next_goal`. The per-ply `active_cluster`/`active_vec` record
the chain. *Games never span a goalspace refit (a game lives within one `run_one_batch`, and workers
reload the goalspace only between batches), so the cached `active_vec`/`e(s_goal_start)` stay
consistent within a game [resolves I4].*

### 3. α-schedule win-apex (replaces the hard threshold)
The means-end leaf is `(1−α)·v_win + α·(2·v_goal−1)`. Make α decay with **decisiveness**:
`α(s) = α_max · (1 − clip(|v_win(s)| / win_ramp, 0, 1))`, plus a safety override `α = 0` when within
`endgame_margin` plies of `ply_cap`. So α = `α_max` (0.25) in unclear positions (where goals can
help), → 0 when the position is clearly won *or* lost (pure-strength play) and near game end. "Win
as apex" emerges smoothly; no magic threshold, no discontinuity. *[Fixes C2.]*
Resignation: gate the existing root-Q resign rule on `α(s) ≤ α_resign_gate` (i.e. allow resigning
only when effectively in win-pursuit), so the agent never resigns mid-goal-exploration.

### 4. Win-value crediting under chaining *(fixes I3)*
`update_winvalue_from_record` must credit **every distinct cluster pursued during an ε-explore
segment** with the game outcome, not just the first subgoal (`break`). Requires per-segment `explore`
tracking in records (currently effectively per-game) [also resolves M3].

### 5. HER achievement look-ahead cap *(fixes C3)*
In `cluster_goal_samples`, cap `_achieved_cluster`'s look-ahead at the active goal's true remaining
pursuit window (`goal_window − (i − goal_start_i)`), not `deadline_max=60`, so a *later, different*
pursuit drifting into a cluster's τ-ball can't false-positive-label an earlier ply. (Also tightens v2.)

### 6. Batched reassignment evals *(fixes I1)*
In the lockstep driver, gather all sides reassigning in the same ply-round into **one** `(Σsides)·K`
batched `evaluate_planes` call rather than per-side K-row evals. Budget this cost explicitly against
the 200-sim search; if throughput suffers, cap reassignment frequency.

## Config — `experiments/v3-zenith-w8.yaml`

Copy of `experiments/v2-stage1.yaml`; identical to v2-halcyon except the new chaining block, for
clean goal-scale isolation (6×64, 200 sims, `goal_window=8`, `min_reservoir=1500`,
`delta_samples_per_game=16`, `epsilon=0.15`, refresh_every 2000, ply_cap 512, 12 workers, distinct
`feed_port`). New `GoalConfig` fields (defaults reproduce v2-halcyon when off):
- `goal_chaining: bool = False` → True for v3. Gates the ENTIRE entry path (incl. game-start
  selection) so `False` is byte-for-byte v2 incl. RNG draw order [resolves the no-op concern].
- `goal_select_temp: float = 0.5`
- `win_ramp: float = 0.6`   (|v_win| at which α→0)
- `alpha_resign_gate: float = 0.05`
- `endgame_margin: int = 20`
(`meansend_alpha` stays the α_max, 0.25.)

## Testing

- **Unit:** `select_next_goal` (curriculum-weight·v_goal scoring, softmax, ε path, no zero-collapse);
  α-schedule (α_max at v_win=0, →0 at |v_win|≥win_ramp, =0 near ply_cap); `maybe_reassign_goal`
  (achieved→reassign, expire→reassign); win-value crediting over a multi-segment chain (I3);
  `_achieved_cluster` look-ahead cap (C3).
- **Integration:** a `goal_chaining=True` game (goalspace ready) yields a **chain** — multiple
  distinct `active_cluster` across plies — and goal-directed-ply fraction ≫ v2's ~9% (assert >50%,
  guarded on `goalspace.ready` [M2]).
- **Regression:** `goal_chaining=False` reproduces v2 exactly; existing emergent self-play / HER /
  loop / resume tests pass unchanged.

## Comparison protocol

Fresh `v3-zenith-w8` run (machinery + config identical to v2-halcyon except the chaining block),
trained to 10k. Eval at 5k + 10k (200 sims, 40 g/rung, α=0). Plot vs v2-halcyon (853→924) and
vanilla (753→@10k). Success = v3 ≥ v2; hoped-for = steeper climb / higher 10k. If promising, run
`v3-zenith-w16` next (goal_window 16) as a clean goal-scale variable.

## Out of scope (→ v4 / Stage 2)

Replace the greedy `curriculum_weight·v_goal` heuristic with a **learned** goal→goal transition value
(history-conditioned, with credit assignment) — see the v4 design doc. v3's greedy is its
baseline/bootstrap.

## Adversarial review resolution map

C1 (degenerate score) → §1.2 curriculum weight. C2 (uncalibrated threshold) → §3 α-schedule.
C3 (HER false-positive achievement) → §5 look-ahead cap. I1 (eval batching) → §6. I2 (sub-window τ)
→ §2 accepted+documented. I3 (first-subgoal-only credit) → §4. I4 (mid-chain refit) → §2 (games
don't span refits). M1 (discontinuity) → removed by §3. M2 (not-ready) → §test guard. M3 (explore
crediting) → §4. M4 (resume ordering) → covered by the resume-clobber fix already in `main`.
