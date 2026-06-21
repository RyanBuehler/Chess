# Research Explorations — Strategy, Exploration, and Beyond AlphaZero

A curated working document (pruned 2026-06-12 from the original session log). Structure:
recurring lessons → reference notes → the current design direction → the experiment queue →
idea graveyard. Update only on Ryan's explicit request.

---

## Recurring lessons (the tests every idea must pass)

1. **The label question (the final boss).** Every architecture eventually faces: *what exact
   number is the training target, and why is it trustworthy?* AlphaZero's dominance comes from
   an unusually clean answer — search output is always better than network output, so the
   network forever imitates a stronger version of itself.
2. **The opponent's fingerprints are on everything.** In adversarial domains, any learned
   quantity — Q-values, plans, achieved deltas — was co-authored by the opponent. Conditioning
   on co-authored outcomes ⇒ wishful thinking (Dichotomy of Control, Yang et al. 2022).
   Search is wishful-thinking-removal machinery: minimax models the opponent fighting back.
3. **Never train a network to predict what you can compute.** Legality, repetition, material,
   goal achievement — mask, feed, or verify in code; don't spend gradient on them.
4. **Representation sharing is where sample efficiency lives.** One tower, many heads.
   Specialize experts by *input structure* (domains, regimes, position character), never by
   *output structure* (actions, pieces); keep a co-adaptation phase so router and experts
   negotiate via shared gradients (the MoE lesson; also why per-piece/per-action nets fail —
   independently trained nets share no calibration scale).
5. **No free lunch: "unbiased" and "sample-efficient" are in tension.** All efficiency is
   purchased with inductive bias. The defensible goal is *domain-general* bias (novelty,
   diversity, compression, adversarial pressure), not chess-specific bias.
6. **Novelty alone is the noisy-TV problem.** Exploration bonuses must be anchored to
   competence. Learning progress (dCompetence/dt) is the principled anchor — a noisy TV yields
   zero LP, so an LP agent walks away.
7. **In adversarial games, the opponent population is the constraint system.** A strategy (or
   goal) is "real" exactly insofar as something else must adapt to beat it (AlphaStar league
   insight). Ecology is also the anti-Goodhart mechanism: stale statistics get exploited and
   thereby corrected.
8. **Reward can move from feedback to context.** Hindsight relabeling flips outcomes from
   training targets into conditioning inputs (Decision Transformer / Upside-Down RL / HER).
   The win signal must survive *somewhere* as grounding, but as data annotation, not trickle.

---

## Reference notes

### Mutual-information skill discovery (VIC, DIAYN, VALOR, DADS…)
The formalization of "constrained entropic strategy discovery." Maximize I(z; s) = H(z) −
H(z|s): keep the skill distribution diverse (entropy) while a discriminator must infer z from
visited states (constraint → coherent, recognizable behavior). VIC (Gregor 2016, origin);
DIAYN (Eysenbach 2018, the popular one — reward = log q(z|s) − log p(z)); VALOR (Achiam 2018,
trajectory-level discrimination — right shape for a chess "strategy footprint"); DADS/CIC/LSD
(skills useful for planning). Limits: collapses to trivially-distinguishable-but-useless
diversity without competence anchoring; never demonstrated in adversarial combinatorial
domains.

### The autotelic literature map
- Self-invented goals → **autotelic agents** (Colas, Karch, Sigaud & Oudeyer survey,
  [arXiv:2012.09830](https://arxiv.org/abs/2012.09830)); lineage: IMGEP (Forestier & Oudeyer),
  PowerPlay (Schmidhuber 2011 — perpetually invent the simplest unsolved problem).
- Greedy-until-plateau → **learning progress goal selection** (Oudeyer & Kaplan 2007); modern:
  ALP-GMM, AMIGo, Goal GAN. Formalizes flow / zone of proximal development / level of
  aspiration.
- Backward-from-goal → **reverse curriculum generation** (Florensa 2017), Backplay —
  backward-schedule the curriculum, don't backward-chain the tree.
- Fall fast, get back up → **Go-Explore** (Ecoffet et al.): "first return, then explore."
- Direction over perfection → **open-endedness** (Stanley & Lehman, *Why Greatness Cannot Be
  Planned*; POET).
- Achievement-space framing → **Crafter** (Hafner 2021, score = achievements unlocked),
  **Voyager** (2023, self-proposed milestones + skill library).
- Rewards-as-context → **Decision Transformer** (Chen 2021), Upside-Down RL, RvS family.
  Known weakness: wishful thinking in stochastic/adversarial settings.
- Reusable experts → **Branch-Train-MiX** (Meta 2024: train domain experts independently,
  merge into MoE, finetune router), DEMix layers (hot-swappable experts), successor features +
  GPI (Barreto — provably sound policy composition), RIMs (dynamic modular competition).

### Frontier status (web-verified 2026-06-12)
1. **Goal-space discovery from scratch — OPEN.** Active work
   ([ProQ](https://arxiv.org/abs/2506.18847),
   [reachability abstraction](https://arxiv.org/pdf/2309.07168)); subgoal/option discovery
   still cited as open.
2. **Autotelic × adversarial — OPEN; gap narrowing.** Near misses:
   [autotelic multi-agent](https://arxiv.org/abs/2211.06082) is cooperative only;
   [Foundation Model Self-Play](https://arxiv.org/pdf/2507.06466) is adversarial +
   quality-diversity but not autotelic, no board games. Self-invented goals inside an
   adversarial league in a deep game: undone.
3. **Plasticity at scale — CONTESTED.**
   [Replay may fix it](https://arxiv.org/pdf/2503.20018); transformers largely resistant;
   [may not occur under gradual shift](https://arxiv.org/html/2602.09234v1);
   mechanism: [Hessian spectral collapse](https://openreview.net/forum?id=l3ZwWmZ5Ht).
   Our pipeline already has replay buffer + gradual shift — two of three claimed protections.
4. **Open-ended skill accumulation in deep games — OPEN.** Nothing PowerPlay-shaped at
   chess depth ([position paper](https://arxiv.org/pdf/2406.04268)).
- **Chess exploits:** [adversarial endgame *positions* exist](https://openreview.net/pdf/841c14e7b0db8a5007d76f62a113d1cf306f41f4.pdf)
  (Stockfish vulnerabilities are depth/config-specific). A full adversarial *policy* from move
  one (the KataGo-style result, Wang et al. 2022 — superhuman agent beaten >90% by a generally
  weak exploiter; exploit human-learnable) is undone for chess. Unclaimed ground at any scale.

---

## Current direction: the autotelic adversarial league

Ryan's goal: train toward specific/arbitrary goals, fail fast, no decaying schedules, a
continuously malleable always-learning system. Human model: fantasy → plan → achieve → horizon
shifts → new fantasy. Adversarial, not in a vacuum. Strategy vocabulary discovered from
scratch (human-concept seeding rejected on principle). Expertise to leverage: game dev,
procedural content generation, world building — PCG is the missing discipline in the autotelic
literature (the field hand-designs goal spaces; PCG designs goal *generators*).

**The one-paragraph vision:** an ecosystem of chess agents, each inventing its own practice
goals, pursuing whichever sit at the frontier of its competence (LP), abandoning them when
mastered — while the league keeps changing what's achievable, because goal difficulty in an
adversarial game is relative to the opponent. External metric: Elo-per-game vs vanilla
self-play at matched compute. Internal study: what does self-invented practice look like in a
deep game?

**The novel object (the publishable-shaped sentence):** an LP curriculum over a 2D
**(opponent × goal)** space where the opponent axis co-evolves. Nobody has studied this.

**Components** (three of five already built):
1. *Goal-conditioned agents* — existing PolicyValueNet + goal embedding as input planes; the
   "population" = one net + frozen snapshots + exploiters (AlphaStar-style, single-GPU
   affordable).
2. *Goal space = state deltas* (Session-3 upgrade over the predicate grammar): turn deltas and
   game deltas over rule-level primitives — (nPawns −1, turn ≤5) = "take a pawn by turn 5."
   Hindsight relabeling free and exact; deltas compose (turn deltas sum to game deltas);
   subsumes material/spatial/temporal predicates. Limits: endpoints-only (no path constraints
   or maintained invariants); some concepts (king safety) aren't deltas of raw features.
   Open design question: accept that ceiling for v1, or add a slot for learned predicates
   (second research frontier stacked on the first).

   **v1 status (2026-06-13):** goals are minted from a *hand-enumerated* rule-level vocabulary
   (capture-per-piece-type, check, castle, promote, reach-rank) — value-agnostic (worth is
   learned, not assigned), and ~equivalent to "emergence" because the rule-level basis is small
   and discrete. It is NOT yet pure dynamic discovery over the raw delta space. Kept for v1 by
   choice; pure-emergent-over-raw-deltas was scoped to v2.

   **v2 INTENTION — DO NOT LOSE SIGHT OF THIS (Ryan, 2026-06-13):** the deeper goal is to let
   agents form goals from **ANY state delta** (the full delta-state space), and to **implicitly
   learn which emergent deltas actually led to future goals/wins** — i.e. a learned, in-the-loop
   *evaluation of subgoals by their contribution to downstream goals/outcomes* (credit
   assignment over the delta space, not a hand-picked vocabulary). The point is bigger than
   chess: both the goal *vocabulary* AND their *values* should emerge from "which state-changes
   lead to good downstream outcomes," which orients the agent toward **exploring/structuring the
   delta-state space itself** and makes the method **domain-general** (transfers beyond chess).
   This is the real payoff of "from scratch, unbiased" — the rule-level v1 vocabulary is a
   tractable stand-in. Connects to: deferred learned-predicates, win-lift (deferred to the
   league), HER (relabel achieved deltas), and successor-features / open-ended skill-discovery
   (which deltas are useful stepping stones). Experiment in v2; keep v1 running for now.
3. *Verifier* — goal achievement is computable from game records (lesson 3). No learned
   discriminator needed — a usual autotelic failure surface simply absent.
4. *LP curriculum engine* — per-goal-region success rates and derivatives, outside the net
   (ALP-GMM-style). "Greedy until plateau" as bookkeeping.
5. *League manager* — matchmaking over the population; extends existing arena/match code.

**Three knowledge stores (division of labor):** the net learns HOW (π(a|s,g), V(s,g)); the
curriculum learns WHAT'S LEARNABLE (LP statistics); the goal-value model learns WHAT'S WORTH
WANTING (mined statistics). The net never judges productivity — taste stays in inspectable
statistics, not weights.

**Goal escalation = achievement-system design:** parameterized goal families with smooth
difficulty knobs (LP needs gradients of achievability); a subsumption lattice derived
syntactically (stronger goal implies weaker → tech-tree edges for free) plus mined temporal
precedence for empirical edges; generator proposes lattice-parents of mastered nodes.
"Impressiveness" = rarity × win-lift × current difficulty.

**The RCT insight:** the system *assigns* goals, so goal→win statistics are interventional,
not observational — comparing win rates when g was assigned vs not estimates the causal effect
of pursuing g. Vanilla self-play can never give this. Raw observational conditionals are
confounded (P(win | promoted) is huge because promotion is a symptom of winning); assignment
breaks the confound. League re-mining keeps values honest against exploitation (lesson 7).

**Anti-goal-hacking:** pursuing strategically empty goals loses games; the extrinsic channel
prunes them. The mixing weight goal-achievement vs winning is THE sensitive hyperparameter.

**Temporal attention's one real use:** chess is fully observable, so game history adds nothing
for vanilla play — except **opponent recognition** in a league (the position can't tell you who
you're playing; their move history can). The exploiter's sense of smell. Incompatible with
MCTS leaf evaluation (per-branch context explodes) — fits policy-only or root-level use.

**Failure modes:** goal-hacking (mixing weight tuning); LP noise (windowed stats, patience);
compute dilution (goal-conditioning multiplies the task distribution — Elo-per-game may dip
before transfer kicks in; run long enough to see the crossover or report its absence); needs a
warm start from a competent checkpoint; plasticity over a long shifting curriculum (doubles as
a live test of the contested frontier).

**Scale:** small nets (6x64-ish — iteration speed over capacity), warm-started; league = main
agent + 3–5 snapshots + exploiters; goal grammar v1 = 4–6 delta families with difficulty
parameters; baseline throughput (~1,400 games/hr) ⇒ a matched-compute comparison is weeks, not
months. Deliverables regardless of outcome: goal-conditioning surgery, delta/predicate engine,
league manager, repertoire dashboard (repertoire curve next to the Elo curve).

---

## Experiment queue (ranked by interest, 2026-06-12)

Roadmap shape: **2 → 3 → merge into 1**; 4 and 5 are league expansions; 6–9 standalone.

1. **Autotelic × adversarial league** — the destination (design above).
2. **Exploiter vs frozen champion** — freeze arch-10x128's best checkpoint, train a fresh
   small net purely against it. Unclaimed ground (KataGo-style adversarial *policy* undone for
   chess); nearly free with existing pipeline; founding population for the league. Best
   interest-per-effort.
3. **Goal playground** (single-agent autotelic stepping stone) — goal-conditioned net + HER +
   LP curriculum; measures whether self-directed practice transfers to faster Elo gain.
4. **z-conditioned league** (MI skill discovery, VALOR-style) — strategy-from-scratch in its
   purest form; riskiest; payoff = visibly distinct, watchable play styles.
5. **League distillation** (BTX-style: frozen exploiters as MoE experts + learned router +
   brief co-finetune) — the ecosystem compressed into one agent; needs a league first.
6. **Offline delta-conditioned transformer** (Decision Transformer on our existing corpus,
   condition on "desired delta = win", no search) — costs zero new self-play; hands-on
   transformer training; measures how much chess strength lives in the search the DT family
   amputates.
7. **MoE tower vs dense at matched FLOPs** — capacity-vs-search science; partially answered
   free by the 10x128-vs-6x64 comparison.
8. **Novelty-bonus exploration** (count-based bonus over position hashes replacing Dirichlet
   noise) — cheap baseline-improver; tunes AlphaZero rather than challenging it.
9. **Q-head on shared tower** — salvage of the per-piece idea; legitimate, narrow.
10. **Auxiliary concept heads** (KataGo-style) — would likely help Elo-per-game most, benched
    on principle (chess-biased). Kept for the day pragmatism outvotes purity.

---

## Idea graveyard (and what each contributed)

- **Per-piece networks** (one net per piece, argmax across) — died of: cross-network
  calibration, 16x params with no shared learning, unstable piece identity, greedy 1-ply play,
  the label question. Contributed: the Q-head idea; lesson 4.
- **Per-action experts** (200 nets, one per action, router on top) — died of: action-identity
  is the wrong specialization axis (output-side); the frozen-experts trap (a router smart
  enough to patch 200 miscalibrated experts is itself a full policy net — experts become
  decoration). Contributed: the BTX/league-distillation idea; sharpened lesson 4.
- **WFC backward planning** (chain from checkmate, entropy-weighted plan choice) — died of:
  category error (CSP vs minimax — nothing fights back in WFC; plans must survive every reply
  → AND/OR trees); "checkmate" isn't a node (retrograde analysis = tablebases, caps at 7
  pieces). Contributed: novelty-decaying exploration → intrinsic motivation literature;
  reverse *curriculum* as the tractable home for backward-from-goal.
- **Raw entropic strategy discovery** ("describe every action chain, find the best strategy")
  — died of: the set of all chains is the 10^120 tree; a full strategy is a *bigger* object
  than a move; strategy space only shrinks after abstraction, which is the unsolved part;
  no-free-lunch (lesson 5). Contributed: the MI-skill-discovery reference note; the league
  direction.
- **Pure observational goal statistics** ("80% of games where X…") — died of: confounding
  (symptoms vs causes of winning; Goodhart). Contributed: the RCT insight — goal *assignment*
  makes the statistics interventional.
- **Temporal-transformer-as-the-agent** (deltas as tokens, condition on desired outcome) —
  died of: co-authored deltas / wishful thinking (lesson 2); MCTS incompatibility (giving up
  the label factory). Contributed: delta goal space (adopted into the league design); opponent
  recognition; the offline DT experiment (queue #6).

---

## Training log context

- **baseline-20260612-001337** (6x64, 200 sims): complete. 5,056 games, 741k positions, Elo
  327 → peak 814 → ~684 (noisy, games_per_rung=4); throughput self-accelerated 635→1,403
  games/hr; resign FP rate 0.6%.
- **arch-10x128-20260612-100655** (10 blocks × 128 filters, 3.02M params, sims held at 200):
  running. As of 2026-06-12 18:20 — 9,696 games (19% of 50k target), step ~10,000, latest Elo
  661 (noisy: 742/482/661 on last three evals). Roughly tracking baseline's curve at the same
  game count so far; the bet is on a higher plateau, not a faster start. Watch on /compare.html.

---

## v2 — goal→win valuation & hierarchy: investigation plan (2026-06-14)

**Motivating concern (Ryan):** v1 trains *flat, learnability-selected goal speedruns*. During a
sub-goal a side maximizes `P(achieve delta by deadline)` with the resignation gate OFF and zero
regard for winning; goals are ends, not means, and there is no hierarchy. Risk: a strong
delta-*speedrunner* that is a weak *player*. The intended system treats goals as **subgoals →
bigger goals → winning**, where a goal's value flows down from `P(win)` and we learn *which
deltas matter, and when*. v1 stays the premise test (does undirected practice transfer to Elo,
measured post-hoc?); the items below are the directed/hierarchical successor.

### A — observational goal→win analysis (DONE 2026-06-14, `scripts/analyze_goal_win.py`)
800 lp-goal games, 1600 side-games, base win-rate 0.473.
- **L1 (assigned sub-goal achieved vs missed → win):** lifts are *small* — capture +0.05,
  check +0.04, reach_rank +0.01, castle ~0 (n_ach=2), **promote +0.12** (lone standout); win
  +1.00 is the tautology sanity-check (achieving "win" ⟺ winning, confirms data wiring). The
  confound *inflates* these, so the causal contribution of achieving most sub-goals is plausibly
  ~0. **The worry, in data: hitting the self-set sub-goal barely changes win-prob.**
- **L2 (delta achieved in phase → win-rate vs base):** structure IS present and phase-dependent
  — top: promote/reach-7 midgame +0.15 (collinear: reach-7 ⇒ promote), capture-Q midgame +0.13,
  check endgame +0.12 vs check opening ~0; bottom: capture-P endgame −0.13, castle midgame −0.16.
  Some deltas in some phases track winning strongly (queen capture / promotion midgame, checks
  late); early pawn captures/checks are neutral.
- **CONFOUND (hard):** the winning side has initiative and achieves more of everything → lift ≠
  causation. Descriptive only (matches graveyard "pure observational goal statistics").
- **Takeaway:** flat speedrun goals are mostly win-neutral (worry confirmed), BUT real,
  phase-structured goal→win signal exists for a directed curriculum / credit model to exploit.

### B — interventional goal valuation (de-confound A)
- Randomize a fraction ε of assignments (bypass the LP curriculum → uniform goal) so
  achieving-vs-not is no longer selected by position strength; then `P(win | do(assign g))` and
  `P(win | do(assign g) ∧ achieved)` are *causal* per-goal win-lifts. Stratify by phase;
  bootstrap CIs.
- Build: an "explore" assignment branch (Bernoulli ε) in `GoalAssigner`/`Curriculum.sample`;
  assigned+achieved+outcome are already logged. Cost: ε of self-play goes off-curriculum (small
  exploration tax). This is the graveyard "RCT insight — goal assignment makes the statistics
  interventional," made concrete.

### C — credit model over the delta trajectory ("which deltas mattered, when")
- Tokenize each game as an ordered sequence of (delta, ply-bucket, side) events; train a small
  transformer to predict outcome z; read attention/attribution → a *learned* delta-importance-
  by-phase map capturing order/interactions (vs A's marginal correlations). "Deltas as tokens"
  (queue #6 / graveyard temporal-transformer).
- Offline first: runs on the existing corpus, no training change, no search. Still observational
  (predicts, not intervenes) → pair with B for causality; mind co-authored-delta / wishful
  thinking (lesson 2).

### D — win-value-weighted curriculum + reward shaping
- With a per-goal win-value from B/C, extend the sampler to **`w(g) = LP + β·novelty +
  γ·win_value(g)`** so practice biases toward goals that *matter*, not just learnable ones.
- And/or potential-based reward shaping: credit a delta by the `ΔV(s, win)` it produced (reuse
  the existing win value head) so achieving a delta that *worsens* the position isn't rewarded.
  First real bend of v1 toward means-end without a full hierarchy.

### Destination — hierarchical / means-end goals
- B→C→D are stepping stones; the structural fix is a hierarchy (feudal/options): a higher level
  proposes a subgoal, a lower level achieves it, and the subgoal's value bootstraps from its
  parent down to `P(win)`. That makes goals genuinely "subgoals to bigger goals to winning."

### Sequencing
A (done) → C offline (free, existing corpus) → B (data-collection change, de-confounds) → D
(fold valuation into curriculum) → hierarchy (v2 redesign). Keep the v1 sequential run finishing
as the falsification baseline.

---

## v2 ROADMAP (2026-06-19): staged means-end / emergent-goal plan

**Approved direction (Ryan).** This supersedes the A/B/C/D *sequencing* above by folding those
items into a staged build; A's findings and B/C/D's mechanisms all survive, re-slotted as stages.

**The vision, one sentence:** goals are *emergent state-deltas* discovered from play, each
trainable to pursue and valued by how much it raises `P(achieving other goals → ultimately
winning)`, composed into plans where the agent learns *which subgoals carried the parent* — with
as little hand-specification as possible. Ryan's list, verbatim: goals = state deltas; new goals
discovered through play; goals trainable; goals composable of subgoals (model learns which
subgoals contributed most); ideally nothing hard-coded; agent learns which goals improve chances
of accomplishing other goals, recursively, until one or more *plans* to achieve the end goal —
winning.

**Lineage to the literature** (this is open-ended hierarchical RL, unsolved in adversarial
self-play domains): Feudal Networks (manager picks goals in a learned space, worker achieves
them); **Director** (Hafner 2022 — goals as codes in a world-model *latent*, discovered not
hardcoded); HER (learn from achieved deltas as goals). Each piece exists individually; nobody has
them working *together* under self-play. The plan's job is to de-risk, not assume.

**Core principle — EARN YOUR EMERGENCE.** Everything in the vision pulls toward maximal emergence
(continuous latent goals + emergent hierarchy + emergent credit, all at once) — which is exactly
the configuration that fails *silently* (degenerate goals that look fine but teach nothing). So
emergence increases ONE stage at a time, and **every stage is a working system falsifiable
against Elo** before the next builds on it. The vision is the destination; the only debate is
order.

**Falsification baseline (v1 result, 2026-06-14):** lp-goal@5k Elo ≈ **286** vs vanilla@5k ≈
**~700** (666–754). Flat, learnability-selected goal *speedruns* cost ~400 Elo — they do NOT
transfer to strength. v2 must beat vanilla, not just beat v1.

### Stage 0 — machinery viability (IN FLIGHT)
always-win@5k vs vanilla@5k. Gate: is the goal-conditioning apparatus itself (goal planes,
`V(s,g)` head, HER buffer) Elo-neutral? Every stage below reuses it; if it leaks ~300 Elo, fix
the net before anything else. (random-goal arm dropped — v2 replaces flat goal-sampling anyway.)

### Stage 1 — means-end credit on discovered-but-discrete goals  ← FIRST SPEC
Mine goals from real play: cluster observed `s_t → s_{t+k}` deltas into a goal set that refreshes
over time (emergent, *not* a hardcoded vocabulary), but keep them **discrete so they're
readable**. Add a **causal per-goal win-value** via randomized assignment
(`P(win | do(assign g))`, de-confounded — this is item B) that drives both the curriculum
(`w(g) = LP + β·novelty + γ·win_value(g)` — item D) and **potential-based reward shaping**
(credit a delta by the `ΔV(s, win)` it produced, reusing the win value head — item D). This is
the first version that could *beat vanilla*, and it tests THE question: does means-end goal
practice transfer to strength? Discreteness keeps every downstream stage building on something
you can see.

### Stage 2 — the goal→goal stepping-stone graph
Learn `P(achieve g′ | recently achieved g)` causally → a directed graph among discovered goals
with **win as the sink node** (this is "learn which goals improve chances of other goals"). No
controller yet — build and *inspect* the emergent graph; research payoff is whether it recovers
sensible chess structure (center → development → king-safety → material → win). Item C's credit
model (deltas-as-tokens transformer over the corpus) is the offline tool that powers this.

### Stage 3 — hierarchical controller (composition + plans)
The manager/worker (feudal/options) that selects subgoal *sequences* toward winning, with
explicit **credit assignment over which subgoals carried the parent** (Ryan's exact ask). Built
here because it needs Stages 1–2 and is the most iteration-hungry piece. This is the "hierarchy"
destination from the 2026-06-14 plan.

### Stage 4 — fully continuous emergent goals (most ambitious)
Replace the discrete discovered set with a learned *continuous latent* goal space (Director
style). Purest "nothing hard-coded." Deferred to last on purpose: front-loading it is where most
emergent-HRL projects die; earn it only if Stages 1–3 show discreteness is the bottleneck.

### Decision log
- **Discrete-discovered-first** chosen over latent-continuous-first (2026-06-19): debuggable,
  ships, beats-or-loses-to-vanilla cleanly; can't easily tell whether a bad latent-first result
  is the goal space or the controller. Climb to continuous (Stage 4) only when earned.
- Each stage = its own spec → plan → implement cycle. Stage 1 is the next spec.
