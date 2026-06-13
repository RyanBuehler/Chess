# Autotelic Goal Playground — Design Spec (v2)

**Date:** 2026-06-12
**Status:** Approved — finalized after two independent adversarial reviews (RL-correctness + experiment-design lenses); ready for implementation planning
**Type:** Experiment / research subsystem

> **Changelog from v1:** value-head/backup redesigned (the negamax convention was incompatible
> with non-zero-sum goal achievement — a correctness bug, not tuning); added the always-win
> control arm; added a win-goal training floor; budget raised to 30k games with a round-robin
> execution model; **win-lift cut from this experiment** (deferred to the league); pre-registered
> success/refutation/inconclusive criteria; eval moved to training-level sims; staged build order.

## 1. Goal and hypothesis

A single-agent autotelic learner: an AlphaZero-style agent that invents its own deadline-bounded
practice goals, pursues whichever sit at the frontier of its competence, and accumulates a growing
repertoire — and a test of whether that self-directed practice transfers to playing strength.

**Primary hypothesis (transfer):** an agent trained with hindsight-relabeled, learning-progress-
curriculum goal practice reaches a target playing strength in *fewer self-play games* than vanilla
self-play, because hindsight relabeling extracts many exact learning signals per game and the
curriculum keeps training at the competence frontier.

**Why it matters:** this is the **premise test for the autotelic adversarial league** (the
project's North Star). Transfer here justifies the league; failure here means the league is built
on sand. A *clean* negative is valuable; an *uninterpretable* result is not — hence the rigor below.

## 2. Pre-registered evaluation criteria (fixes the "unfalsifiable" finding)

Committed before launch, so no outcome can be narrated either way after the fact:

- **Primary metric:** games-to-Elo-θ, where θ is a fixed threshold *read off the vanilla arm*
  (the Elo all arms demonstrably reach — chosen after vanilla's curve is known, not assumed; given
  prior runs, expected around 600–700, well below the noise-dominated ceiling).
- **Confirmation:** an LP-goal (or random-goal) arm reaches θ in ≥15% fewer games than vanilla,
  with seed-bootstrapped confidence intervals excluding zero.
- **Refutation:** CIs overlap, or vanilla reaches θ first.
- **Inconclusive (a distinct, named outcome — NOT a "publishable negative"):** arms do not separate
  beyond noise, or fail to reach θ within budget. This means *failed measurement → more budget/seeds
  required*, and may not be reported as evidence against the hypothesis.
- **Seeds:** Phase 1 = **1 seed/arm, pre-committed as exploratory only** (validates apparatus +
  regression gate + reveals curve shape + go/no-go on Phase 2). Phase 2 = **3 seeds/arm**, the only
  phase from which a confirmation/refutation claim may be drawn.

## 3. Scope

**In:** single-agent self-play; value-agnostic emergent goal space; HER; LP curriculum; goal-
conditioned MCTS; four segregated arms; 30k-game budget per arm; round-robin execution; post-hoc Elo.

**Explicitly OUT (deferred to the league), with reasons:**
- **Win-lift / interventional goal valuation.** Cut from this experiment. Its causal estimate is
  confounded because the curriculum assigns goals by LP (not randomly), "matched positions" is
  near-impossible in chess, and it cannot be estimated at a 30k-game budget where wins are sparse.
  It belongs to the league design, where a randomized-assignment held-out fraction can support it.
- Multi-agent league, exploiters/opponent modeling, learned (non-rule-level) goal predicates,
  any pre-decided piece values or hand-shaped rewards.

## 4. Core principle: value-agnostic goals

No piece values, no hand-shaped rewards, no pre-selected "important" goal categories. The only
injected structure is the **rule-level feature basis** — distinguishing a knight from a queen is
*reading the rules*, not valuing them. With win-lift deferred (§3), goal *worth* in this experiment
is expressed only implicitly: a goal matters insofar as practicing it lets the agent reach the
apex win-goal in fewer games (the transfer metric). Nothing asserts a goal is good a priori.

## 5. Goal representation

A **goal** is a deadline-bounded target delta expressed in the board's own feature vocabulary
(goal vocabulary = state vocabulary → completeness by construction).

- **Spatial component:** a goal-mask plane over the 8×8 board (rank 8 for promotion, a destination
  square, etc.) plus the relevant piece-type channel. Conv-native; covers positional/movement/
  promotion goals.
- **Non-spatial component:** scalar/categorical broadcast planes for count deltas (per rule-level
  piece type, both colors), check, castling, result.
- **Deadline:** encoded as a **value-head FC side-input** (concatenated at the value/policy head),
  NOT only as a broadcast plane. (Review finding: a single spatially-constant plane is poorly
  legible to a conv tower; a scalar fed at the FC is.) A monotonicity unit test gates training —
  see §9.

**Compositional, never one-hot-per-goal:** a goal is a sparse target over fixed feature
dimensions, so a newly minted goal is a new combination (new mask / new scalar target), needing no
architecture change. This is what lets the repertoire grow online and lets the network share
learning across goals.

**Individuation:** piece-type level, with timing/location as emergent refinements (§6). Coarser
than raw feature vectors (avoids the noisy-TV explosion), finer than "material changed" (preserves
"take the queen" ≻ "take a pawn").

## 6. Goal lifecycle and the repertoire

- **Minting rule (concrete):** maintain a set of *goal templates* keyed by `(feature-dimension,
  direction)` at the piece-type abstraction. The first time a game's verifier records a delta whose
  template is not in the repertoire, add that template (deadline initialized from the move it
  occurred). The repertoire is a list of templates + per-template statistics (success-count,
  attempt-count, windowed success rate). Append-only in identity.
- **Refinement (child-spawning):** when a template's windowed success rate plateaus high, spawn its
  tighter children by tightening the deadline (T→T−k) and, later, by adding a location mask. Same
  novelty/first-seen trigger applies to children.
- **Discovery is play-distribution-dependent (acknowledged risk):** early weak self-play mints the
  deltas of *bad* chess (hung pieces). The repertoire is therefore non-stationary and biased toward
  blunder-deltas early. Mitigation: periodic re-mining; **track repertoire composition over training
  as a pre-registered diagnostic** (does it shift from blunder-deltas toward structural deltas?).

## 7. In-game objective

**Winning is the apex goal, not a separate objective.** "Win (by move N)" is the ultimate delta in
the same machinery.

- The agent always optimizes its currently-assigned goal g: MCTS searches toward g, value head is
  `V(s,g) = P(protagonist achieves g by its deadline)` (§9), policy from visit counts.
- **One goal per side per game** (both sides conditioned; doubles data).
- **Pure pursuit (start pure):** while chasing g, optimize only g. A stay-alive blend is the
  fallback knob if pure pursuit learns reckless habits that fail to transfer.
- **On goal resolution** (achieved or deadline expired) → switch to the apex win-goal for the rest
  of the game. Sequential, not blended; every game ends with real win-directed play and a result.
- **Win-goal training floor (fixes the "starved policy" finding):** regardless of LP, **≥20% of
  assigned goals are always g=win**, from game one. This keeps `π(·|win)` and `V(·|win)` — the only
  quantities eval depends on — continuously trained on full games from the opening, preventing the
  permanent dip that a fully-starved win-goal would cause. Also instrument **fraction of plies played
  under g=win per arm** as a control variable (it must be comparable across arms or the per-game
  comparison is confounded).

## 8. Value head and backup — the correctness redesign

v1 reused the existing **negamax** value (zero-sum, side-to-move frame, tanh∈[−1,1]). That is wrong
for sub-goals: "White captures a knight" is not the negation of anything — both sides' sub-goals can
be simultaneously true, and the protagonist does not flip with the side to move. Reusing negamax
makes the search *minimize its own goal* on half the plies. Redesign:

- **Protagonist-relative value:** each search is tagged with a protagonist P. `V(s,g) =
  P(protagonist achieves g by deadline | s) ∈ [0,1]`. **Sigmoid head, BCE loss** (not tanh/MSE) — a
  [0,1] head represents achievement probability honestly; the old tanh wasted half its range and
  collided "draw" with "0% achieve."
- **Backup is minimax in protagonist frame, NOT negamax:** at protagonist-to-move nodes the search
  maximizes V; at opponent-to-move nodes the opponent acts to **minimize** V (an explicit min in the
  *same* protagonist frame — no sign-flip into a shared accumulator).
- **Terminals (exact, protagonist frame):** goal achieved → 1; deadline expired → 0; real game-over
  → evaluate g (for g=win: win=1, draw=0.5, loss=0).
- **Win-goal collapses to the current pipeline:** P(protagonist wins) with the opponent minimizing
  = minimax = the existing negamax up to the affine map `v₍₋₁,₁₎ = 2p − 1`, with draws at 0.5.
- **Regression gate (hard automated test, must pass before any arm launches):** with g=win and no
  sub-goals, the goal-conditioned search must reproduce `reference.py`'s visit distribution on fixed
  positions (within the affine map). If it doesn't, §8 is wrong. The §v1 "additive" claim is
  retracted — this changes the backup algebra and the value semantics, and draw handling must be
  verified explicitly.
- **Deadline monotonicity test (gates training):** hold (s,g) fixed, sweep moves-remaining; assert
  V is monotone non-increasing and → 0 at T=0 with g unachieved. Calibrate V against held-out
  achievement frequency *binned by remaining moves*.

## 9. Network changes

Extend `chessrl/model/network.py`: goal-conditioning planes (§5) appended to the 21 board planes;
deadline as an FC side-input; **both heads condition on g**; value head becomes **sigmoid (BCE)**
with the protagonist-frame achievement target (§8).

## 10. MCTS terminals and search

Per §8: protagonist-frame value, minimax backup, exact goal terminals (achieved 1 / expired 0 /
game-over evaluate-g). The actual game still plays to a true chess result (or ply cap) regardless of
goal terminals, since full games are needed for the transfer metric and the win-floor games.

## 11. HER and the replay buffer

- **Policy head — trained only on the assigned goal** (valid search visit counts exist only there).
  The win-floor (§7) guarantees the eval-relevant `π(·|win)` is continuously trained.
- **Value head — HER-relabeled densely**, with two corrections from review:
  - **Prefer search-laundered targets over raw hindsight.** Raw "achieved delta" labels are
    co-authored by a weak opponent (our own lesson #2: the opponent's fingerprints are on
    everything), so V trained on them is *optimistic* — it believes goals are achievable against
    cooperative play, which hurts vs real opponents. Where a goal-terminal was reached *within
    search*, use that (adversarially-laundered) estimate as the target; use raw HER labels as
    weighted-down auxiliary signal, with negatives weighted up.
  - **Wishful-thinking thermometer:** track, per goal, the gap between self-play achievement rate
    and held-out (vs-Stockfish) achievement rate. A large gap flags optimism — a pre-registered
    diagnostic.
- **Negatives required:** sample deltas that did not happen (incl. positions where the opponent
  *prevented* the delta) → target 0, or V collapses to "everything achievable."
- **"Future" relabeling:** per state, sample a few goals from deltas achieved later in the same game.
- **Storage / source of truth:** save raw games (moves + assigned goal + per-move visit counts);
  generate HER samples at train time via the verifier; relabeled samples never persisted.

## 12. Curriculum (LP) — concretely specified

- **LP estimator:** a Beta-Bernoulli posterior over each template's success probability, with an
  absolute-learning-progress signal computed over a **window of W=200 attempts** (config field), and
  **assignment confidence gated on attempt-count** (untried/barely-tried templates use the novelty
  bonus, not a noisy LP estimate). This replaces v1's hand-wavy "windowed statistics" — finite-
  differencing a noisy Bernoulli rate (the review's finding) is avoided by using posterior
  uncertainty explicitly.
- **Sampling distribution:** `w(g) ∝ LP(g) + β·novelty(g)`, with the win-floor (§7) applied on top.
  Weights β, W, and the floor are config fields recorded in provenance.
- **Emergent easy→win arc:** win starts hard (LP≈0, but protected by the floor so never fully
  starved); as sub-goal competence accumulates, win's LP rises and the curriculum shifts toward it.
  This self-driven climb is the transfer hypothesis, and is directly watchable.

## 13. Arms and segregation (four arms)

| Arm | Mechanism | Isolates (vs the arm above) |
|-----|-----------|------------------------------|
| **vanilla** | win/loss only, current pipeline | control |
| **always-win** | full goal apparatus (planes, BCE value, HER), only ever g=win | the *apparatus* (planes, value-retarget, HER) — NOT goal diversity |
| **random-goal** | apparatus + diverse goals sampled uniformly (+ win-floor) | goal *diversity* |
| **lp-goal** | apparatus + LP curriculum (+ win-floor) | the *learning-progress* selector |

The always-win arm (new, from review) is the capacity/apparatus control: vanilla→always-win moves
the planes/value-target/HER without goal diversity, so a later vanilla→random gap is attributable to
diversity rather than to the apparatus. **Param-count note:** goal arms have a wider stem (extra
input planes), so "identical net size" is impossible; report the param delta and pad/verify it is
negligible relative to the 6×64 tower.

**Segregation:** each arm = its own `experiments/*.yaml` + run dir + `provenance.json` recording
`goal_mode`, win-floor, LP config, seed, net shape, sims. From scratch, all arms. Sims held at 200.
Net 6×64 (iteration speed).

## 14. Execution — round-robin (replaces concurrent)

Run arms in **rotation, 1,000 games per arm per round, ~30 rounds to 30k games each.** Each arm gets
the **whole GPU during its slice**, so there is no shared-GPU contention and no samples-per-position
drift between arms — a clean per-game comparison — while all curves climb in lockstep for live
side-by-side watching on `/compare.html`.

- Implemented by a `scripts/round_robin.py` orchestrator that resumes each run for +1000 games in
  turn (reusing `--resume`, which reconstructs the buffer from records). Resume overhead (buffer
  rebuild + worker spawn) is a few minutes per slice — negligible over the run.
- **Distinct feed-port ranges per arm** (e.g. 5555/5565/5575/5585; workers bind feed_port+w).
- **Hung-worker watchdog:** the supervisor restarts dead *and* hung workers (current code restarts
  only `not is_alive()`; a hung worker silently halves an arm's throughput). Per-arm
  last-game-mtime heartbeat.

## 15. Evaluation (post-hoc)

- Checkpoint regularly during training; after the run, sweep saved checkpoints for Elo and plot
  **Elo vs games** for all arms.
- **Eval at training-level sims (200), not 50.** (Review finding: the transfer mechanism routes
  through MCTS using `V(s,win)`; at 50 sims, value-driven search cannot express a learned value over
  a weak policy prior, biasing goal arms downward. Report Elo-vs-sims as a controlled axis.)
- **Raise `games_per_rung`** for the sweep (decoupled from training, so cheap) to shrink eval noise.
- Goal arms evaluated by conditioning on g=win.

## 16. Live observability

Repertoire size and composition over time; per-goal LP and success-rate windows with CIs; fraction
of plies under g=win per arm; the wishful-thinking thermometer; policy/value losses; throughput.

## 17. Stopping condition

**30,000 self-play games per arm** (per phase). Elo is post-hoc (§15).

## 18. Risks and open questions

- **Budget/noise:** 30k is past baseline's ~5k plateau; vanilla-as-control fixes the threshold θ.
  The 1-seed Phase-1 is explicitly exploratory (§2).
- **Compute reality:** 4 arms × 30k = 120k games (Phase 1, ~days); ×3 seeds = 360k (Phase 2, ~2wk).
- **Wishful-thinking / co-authored labels** (§11) — the main RL risk; thermometer + search-laundered
  targets are the mitigations.
- **Plasticity** over a long shifting curriculum — contested frontier; replay buffer + gradual shift
  are two of three claimed protective conditions; this run doubles as a live test.
- **Goal-hacking** — pure pursuit + the win metric; stay-alive blend is the fallback.
- **Repertoire seeded by blunder-deltas** (§6) — re-mine; composition-shift is a diagnostic.
- **Pure-pursuit transfer may genuinely fail** — a clean negative (Phase 2 only), not inconclusive.

## 19. Build order (staged)

1. **Verifier** — standalone, unit-tested: given a game record + a goal template, did it achieve by
   deadline? (The one component that is cheap and exact.)
2. **Value-head/backup redesign + regression gate** (§8) — protagonist-frame, sigmoid/BCE, minimax;
   prove g=win reduces to `reference.py` before anything else.
3. **always-win + random-goal arms** — conditioning planes, HER, uniform sampling, win-floor. Prove
   the apparatus runs and the regression gate holds end-to-end.
4. **LP curriculum** — add the estimator + sampler → lp-goal arm.
5. **Round-robin orchestrator + watchdog + post-hoc eval sweep.**

## 20. Relationship to the codebase

Extends: `chessrl/model/network.py` (planes, FC deadline, sigmoid value), `chessrl/mcts/reference.py`
+ **`chessrl/mcts/batched.py`** (protagonist-frame minimax + goal terminals + goal-conditioned
`evaluate_planes` — **the riskiest change: batched.py's copy-free leaf-parking and `(N,21,8,8)`
evaluator is the hottest, most-optimized path, and goal-conditioning it is a non-additive rewrite,
not an extension**), `chessrl/selfplay/*` (assigned-goal play, goal switch, win-floor),
`chessrl/training/*` (HER sample generation, BCE value loss, buffer). New modules: goal space +
verifier + repertoire + curriculum. `reference.py` (post-redesign) remains the equivalence baseline
via the §8 regression gate.
