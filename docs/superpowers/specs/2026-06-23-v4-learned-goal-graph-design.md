# v4-apex — Learned Goal→Goal Transition Graph (Design)

**Date:** 2026-06-23
**Codename:** `v4-apex` (lineage: `v2-halcyon` = flat Stage-1 goals; `v3-zenith` = greedy chaining;
this = Stage-2 **learned** goal→goal graph with WIN as sink)
**Status:** banked design. **Not being built now** — v4 begins after v3-zenith trains to 10k and we
have a v3-vs-v2 Elo read. This doc records the design so it is ready to spec→plan→implement then.

## Goal

Replace v3-zenith's *greedy* next-subgoal heuristic — `score(g) = win_value(g) · v_goal(s, g)` with
a hardcoded win-apex at `v_win(s) ≥ threshold` — with a **learned goal→goal transition value**:

> `T(s, g_hist, g′)` ≈ the value of pursuing candidate subgoal `g′` next, given current state `s`
> and the **history of recently-achieved goals** `g_hist`, scored by how much `g′` raises
> `P(eventually win)`.

This is the notes' **Stage 2** ("learn `P(achieve g′ | recently achieved g)` → a directed graph
among discovered goals with WIN as the sink node"). The agent should *learn which subgoals enable
which others*, composing a plan whose sink is "win" — instead of being told, by a fixed formula, that
the most win-correlated immediately-achievable cluster is always best. v3's greedy is exactly the
quantity v4 is allowed to bootstrap from and must beat.

**One sentence:** v4 learns the *edges* of the means-end graph (which goal leads usefully to which),
where v3 only ever scored *nodes* in isolation.

## Why

v3 chains, but each link is chosen myopically: `win_value(g)` is a **global, state-free** causal
average (`WinValueEstimator`, `P(win|do g) − base`), and `v_goal(s, g)` only asks "can I reach `g`
from here?". Neither asks the stepping-stone question — *given I just achieved `g`, which `g′` now
unlocks the rest of the plan?* A goal that is only valuable **as a setup for another** (develop →
*then* attack) is invisible to v3: develop-from-the-opening has middling `win_value` on its own.
v4's transition value is conditioned on `g_hist`, so it can learn "develop is worth little alone but
high when it precedes king-attack," which is precisely Ryan's "agent learns which goals improve the
chances of accomplishing other goals, recursively, until a plan to win."

The research payoff is independent of Elo: the learned edges form an **inspectable directed graph**
over discovered clusters (§5), the first artifact in the project that can be checked against human
chess structure.

## Scope of change

**Only the next-goal selection learns; everything downstream is untouched.** Unchanged: the dual-head
net's means-end MCTS (α-blend leaf toward `active_vec`), the records schema, HER
(`cluster_goal_samples` still trains the goal head per-ply on whatever goal was active), the training
loop, the discrete `GoalSpace` cluster vocabulary, the eval harness (`VectorGoalMCTSPlayer` at α=0 —
selection affects only self-play/training, never playing-strength eval), and the `WinValueEstimator` /
`ClusterCurriculum` (still used for the interventional exploration prior and the cold-start
bootstrap). The single seam that changes is `select_next_goal` in `selfplay/concurrent.py`.

## Components

### 1. The learned transition model `T` (new)

**Decision needed — new head vs separate small model. Recommendation: a separate small MLP scorer,
not a new net head.**

- **A — new head on `PolicyValueNet`.** Add a third head conditioned on `(trunk-embed e(s),
  g_hist-embed, g′ centroid)`. Pro: shares the trunk, one checkpoint. Con: the trunk is frozen-ish
  within an epoch for `GoalSpace` clustering, the heads already serve two masters (`v_win` tanh,
  `v_goal` sigmoid), and a transition target that *moves as the goal vocabulary refits* (§7) would
  inject non-stationarity straight into the shared trunk. Risky for a component we want to ablate.
- **B (recommended) — a standalone `GoalTransitionNet`**: a small MLP that consumes **precomputed**
  embeddings and centroids and never backprops into the policy/value trunk. Inputs (all available at
  selection time, no new encodes beyond what selection already does):
  - `e(s)` — the frozen-encoder board embedding (`VectorGoalNetEvaluator.embed_boards`, dim = filters).
  - `g_hist` — a fixed-width summary of the **recently-achieved** goal chain (see §2), e.g. the
    concatenation/attention-pool of the last `H` achieved centroids (+ a learned "WIN-pending" token).
  - `g′` — the candidate cluster centroid (the same centroids v3 already scores).
  - optional scalars: `v_win(s)`, `win_value(g′)`, plies-into-game / deadline.
  - **Output:** a scalar `T(s, g_hist, g′)` ∈ [0,1] = predicted `P(eventually win | pursue g′ next)`.
    A second optional head `R(s, g_hist, g′)` ∈ [0,1] = `P(reach g′ within goal_window)` gives the
    explicit stepping-stone `P(achieve g′ | from g)` the notes name; the controller uses
    `T` (value) for selection and `R` (reachability) to prune unreachable candidates.

  Pro: cleanly ablatable (delete the file → fall back to v3 greedy), tiny (fits/refits cheaply when
  the vocabulary changes), no contamination of the strength-bearing trunk. Con: a second artifact to
  checkpoint and version alongside the net + goalspace + winvalue. Net call: **B** — isolation is the
  whole project methodology, and `T` is exactly the thing we must be able to turn off.

`T` is **keyed by cluster id** (discrete vocabulary), but consumes **centroids** (vectors), so it
degrades gracefully across a refit: ids change meaning every 2000 games but centroids stay in the
same embedding metric, so a vector-keyed `T` can be *warm-started* rather than reset (§7).

### 2. Conditioning on goal HISTORY (not just current goal)

The distinguishing feature vs v3. `g_hist` is the sequence of clusters this side has **achieved**
(not merely pursued) so far this game, in order — the live chain v3 already detects (its component-1
achievement test) and records per ply. Two encodings, pick by cost:

- **Recommendation: last-`H` window pooled (H≈3–4).** Concatenate the last `H` achieved centroids
  (zero-pad early game), pass through a tiny set-or-sequence encoder (mean-pool or 1-layer GRU/attention)
  → fixed `g_hist` vector. Cheap, Markov-ish, robust to refits.
- **Full-sequence transformer over the chain.** The notes' "deltas-as-tokens transformer over the
  corpus" (Item C). More faithful to long plans but heavier and harder to refit. **Defer to a v4.1
  variant** if the windowed version shows history matters; do not front-load it.

Either way `T` is **history-conditioned**, so the same `(s, g′)` can score differently depending on
what was already achieved — the mechanism that lets the graph encode *ordering* (center→development→
king-safety→material→win) rather than a static node ranking.

### 3. Training signal & credit assignment

**Data source — already recorded.** v3-style means-end self-play writes per-ply `active_cluster` /
`active_vec`, and the live achievement detector marks where each subgoal was reached. From any
finished game we extract, per side, the **achieved-goal chain** `g_0 → g_1 → … → g_k → [WIN|loss|draw]`
with the ply index and the board `s_i` at each transition, plus the terminal outcome `z ∈ {win, draw,
loss}` from the side-to-move perspective. No new self-play data is needed — v4 is trained on the same
corpus v3 already produces.

**Targets.**
- `T` head: for each transition "at `s_i`, with history `g_{<i}`, the side chose `g_i`," the target is
  the **eventual game outcome** `z` (win=1, draw=0.5, loss=0) — a Monte-Carlo return to the game's end.
  This is the literal `P(eventually win | pursued g_i next from here)`. Optionally bootstrap with the
  net's `v_win` at the end of the chain for variance reduction (TD(λ) over the *goal* sequence, not
  the ply sequence — the chain is short, so MC is a fine v4.0 default).
- `R` head: target = 1 if `g_i` was achieved within `goal_window` plies, else 0 (directly from the
  achievement detector).

**Credit assignment — "which subgoals carried the parent" (Ryan's explicit ask).** A raw MC target
credits *every* subgoal in a winning chain equally, which is too noisy: a win-chain contains both the
decisive setup and incidental filler. Three mechanisms, layered:

1. **Interventional de-confounding (reuse the v3 ε-explore path).** Transitions taken under ε-explore
   (uniform-random `g′`, already flagged `explore=True` on the side and recorded) are the *unconfounded*
   samples — the choice of `g′` was independent of the state, so their outcome correlation is causal,
   not "the policy only picks `g′` when already winning." Weight ε-explore transitions higher (or train
   `T` only on them, mirroring how `WinValueEstimator` uses only do-games). This is the single most
   important guard against the greedy-bootstrap feedback loop (§7).
2. **Advantage-style baseline.** Credit a transition by `z − v_win(s_i)` (or `z − base_winrate`), so a
   subgoal taken when *already winning* gets little credit and one that *flipped a non-winning position
   to a win* gets a lot. This is the potential-based `ΔV(s, win)` idea from the notes' Stage-1 shaping,
   lifted to the goal level: the subgoal that produced the biggest jump in win-value "carried" the parent.
3. **Sequence credit (deltas-as-tokens, optional / v4.1).** Train the full-sequence encoder (§2) with an
   attention readout and inspect attention weights / integrated-gradients over the chain to attribute the
   eventual win to specific earlier subgoals — the literal "which subgoals contributed most" readout.
   This is a *research instrument* more than a training necessity; the windowed `T` + advantage baseline
   already gives a usable controller. Defer the transformer-attribution to v4.1.

**Recommendation for v4.0:** MC outcome target, trained with the advantage baseline (#2), with
ε-explore transitions up-weighted (#1). That is concrete, cheap, and de-confounded; the attention
attribution (#3) is the v4.1 enhancement.

### 4. Integration — the `select_next_goal` seam

`select_next_goal(board, goalspace, winvalue, evaluator, win_vector, rng)` (the v3 function) is the
only thing replaced. v4 swaps its body for a **learned policy/value over next-goal**:

```
v4 select_next_goal(board, goalspace, winvalue, evaluator, win_vector, rng, transition_model, g_hist):
  if not goalspace.ready or transition_model is None or transition_model.cold(g_hist):
      return v3_greedy_select_next_goal(...)          # COLD-START / FALLBACK = the v3 baseline
  e = evaluator.embed_boards([board])[0]
  cand = all K cluster centroids  +  WIN (win_vector as a sentinel candidate)
  R = transition_model.reach(e, g_hist, cand)         # prune unreachable (R < r_min), keep WIN
  T = transition_model.value(e, g_hist, cand)         # predicted P(eventually win | pursue cand)
  # WIN is just another candidate scored by T — the apex is LEARNED, not a fixed v_win threshold.
  with prob epsilon: return uniform-random cluster (explore=True)     # unchanged interventional path
  else: softmax-sample over T / goal_select_temp                      # exploration, as v3
```

Key points:
- **WIN becomes a learned sink, not a hardcoded threshold.** v3 forces terminal pursuit when
  `v_win ≥ win_switch_threshold`. v4 makes WIN one candidate among the clusters, scored by the same
  `T`; the agent *learns* when committing to direct win-pursuit beats one more setup move. (Safety
  net: keep a high `v_win` hard-override, e.g. 0.6, so a clearly-won position can't be talked out of
  finishing — a guardrail, not the decision rule.)
- **v3 greedy is the cold-start and the fallback.** Until `T` has enough samples for the current
  vocabulary (per-cluster attempt counts, mirroring `WinValueEstimator.attempts`), and on any history
  it hasn't seen, `select_next_goal` returns the v3 greedy choice. v4 thus *strictly contains* v3 and
  degrades to it — no cold-start cliff.
- Everything else in concurrent self-play (lifecycle `maybe_reassign_goal`, achievement detection,
  records, HER, resign logic) is identical to v3.
- New `GoalConfig` flags (defaults preserve v3 behavior when off): `learned_transition: bool = False`,
  `transition_min_samples: int` (cold-start gate), `transition_reach_floor: float` (R-prune), reuse
  v3's `goal_select_temp` and `epsilon`.

### 5. The emergent goal→goal graph as a research artifact

Independent of any Elo gain, v4 yields a **directed weighted graph** `G = (clusters ∪ {WIN}, edges)`
where edge `g → g′` carries `T(·, g_hist=g, g′)` (and reachability `R`). Building & inspecting it is a
first-class deliverable:

- **Construction:** for each cluster `g`, evaluate `T` with `g_hist = [g]` over all `g′` (and WIN);
  threshold/top-k the edges. Average over a sample of representative states `s` (or condition on a
  canonical opening/midgame/endgame `s` to get phase-specific graphs).
- **Inspection (the payoff):** does the graph recover sensible chess structure — a rough topological
  flow **center-control → development → king-safety → material → WIN**, with WIN as the dominant sink?
  Label clusters via the existing `cluster_labels` (the LIVE-view label machinery in
  `_meansend_aux`) so edges read in human terms. Render to the `/compare.html` UI (a new graph panel)
  or export GraphML for offline viewing.
- **Diagnostics:** sink-reachability (can every useful cluster path to WIN?), cycles (A↔B oscillation
  = a credit-assignment or refit pathology), orphan clusters (never a useful predecessor → candidates
  for the vocabulary to drop). These double as health checks for the learned model.

This is the concrete answer to "agent learns which goals improve chances of other goals" — you can
*look at* the learned dependency structure, not just its downstream Elo.

### 6. Experiment / eval plan

Fresh run `v4-apex`, **same machinery and config as v3-zenith-w8** except `learned_transition=True`
(and v3's `goal_chaining=True`), trained to 10k. Compare on the **shared Elo ladder** (200 sims,
40 games/rung, α=0 terminal pursuit, recorded vs the Stockfish anchors), three-way:

- **v2-halcyon** (flat, 853@5k → 924@10k) — the substrate baseline.
- **v3-zenith** (greedy chaining) — the immediate predecessor / bootstrap.
- **v4-apex** (learned transition) — this run.

**"v4 works" =** (a) **v4 ≥ v3** on the Elo curve at 5k and 10k (the learned graph at least doesn't
hurt, ideally a steeper climb from better-ordered plans), **and** (b) the emergent graph (§5) is
**legible** — recovers a recognizable center→development→king-safety→material→WIN flow with WIN as
sink. (b) can succeed as a research result even if (a) only ties.

**Isolation:** `learned_transition=False` must reproduce v3 byte-for-byte (regression gate, same as
v3's `goal_chaining=False` reproduces v2). Optionally an **ablation arm** `v4-apex-nohist` (`H=0`, i.e.
`g_hist` dropped, `T(s, g′)` only) isolates the *history* contribution from the *learned-vs-greedy*
contribution — the cleanest way to attribute any gain to the stepping-stone idea specifically.

### 7. Risks & open questions

- **Non-stationarity from vocabulary refits (largest risk).** `GoalSpace.maybe_refresh` re-fits
  k-means every `refresh_every` (2000) games; cluster ids are reassigned and centroids move, so
  id-keyed transition stats are invalidated. **Mitigations:** (i) key `T` on **centroids**, not ids
  (§1) so it generalizes across the metric and can be *warm-started* post-refit instead of reset;
  (ii) on refit, **re-anchor** old→new clusters by nearest-centroid and carry forward `T` mass (a
  soft transfer, like re-labeling); (iii) decay-weight transition samples so stale (pre-refit) data
  fades. **Decision needed:** reset-on-refit (simple, safe, slow to re-warm) vs centroid-transfer
  (faster, risk of carrying wrong structure). **Recommendation:** centroid-keyed `T` + nearest-centroid
  transfer + sample decay; reset only if transfer proves unstable.
- **Credit-assignment noise / bootstrap feedback loop.** If `T` is trained on *on-policy* transitions
  the policy itself selected, it confirms its own choices (the classic value-iteration-on-its-own-data
  trap). Mitigated by training primarily on ε-explore (interventional) transitions (§3 #1) and the
  advantage baseline (#2). Open question: how much ε is enough to keep `T` honest without diluting
  play — start at v3's 0.15, monitor edge stability across refits.
- **Bootstrapping from greedy.** Cold-start returns v3 greedy, so early `T` data is greedy-selected
  (confounded). The ε-explore stream is the unconfounded antidote; verify `T` diverges *usefully* from
  greedy (different, better-ordered edges) rather than just memorizing greedy's choices. If `T` only
  ever reproduces greedy, v4 has added cost for no signal — a clean negative result worth knowing.
- **Discrete clusters vs continuous latent.** v4 stays discrete (readability, inspectable graph). If
  the graph is legible but Elo stalls, the bottleneck may be the discrete vocabulary itself → that is
  the Stage-4 continuous-latent question, explicitly out of scope here.
- **Short chains.** With `goal_window=8` and ~120-ply games, a chain is ~10–15 achieved goals; MC
  targets over that are reasonably low-variance, so TD bootstrapping is optional. If we move to bigger
  windows (v3-zenith-w16), revisit.

### 8. Out of scope (→ Stage 3 / Stage 4)

- **Stage 3 hierarchical manager/worker controller.** v4 builds the *graph* (the learned edges) and a
  greedy-over-`T` selector; it does **not** add a manager that plans multi-step subgoal *sequences*
  with lookahead, nor the full feudal/options two-timescale controller. v4's selector is still
  one-step (pick the best next goal); planning over the graph is Stage 3.
- **Stage 4 continuous latent goals.** The discovered discrete cluster vocabulary stays. Replacing it
  with a learned continuous latent (Director-style) is deferred to last, per the notes' decision log.
- **Full deltas-as-tokens sequence-attribution transformer** (§3 #3) beyond the optional v4.1
  instrument — the v4.0 controller uses windowed history + advantage credit only.

v4 is the **learned-transition step only**: turn v3's greedy node-scoring into a learned, history-
conditioned edge-scoring graph with WIN as sink, and make that graph inspectable.
