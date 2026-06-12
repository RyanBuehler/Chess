# Chess RL System — Design

**Date:** 2026-06-11
**Status:** Approved (brainstorming complete; revised after adversarial review)
**Goal:** A research/education platform for training AlphaZero-style policy+value networks to play chess via self-play, with an Elo ladder as the primary metric and a web UI for watching, playing, and running experiments.

## Context & Constraints

- **Purpose:** learning modern RL and NN theory through hands-on experimentation. Fast experiment turnaround matters more than maximum playing strength.
- **Hardware:** RTX 5090 laptop (Windows 11) now; NVIDIA DGX Spark (aarch64 Linux) later for longer offline runs. Design must make that migration trivial.
- **User posture:** strong engineer with some ML/RL experience. Libraries for commodity parts; the RL system itself is hand-built so it can be instrumented, decomposed, and modified.
- **Approach chosen:** build-your-own AlphaZero (vs adopting LightZero/OpenSpiel). An external framework baseline may be added later as an experiment, not as architecture.

### Environment contract (portability)

- PyTorch pinned to a build with CUDA 12.8+ (Blackwell/sm_120 requires it); torch + CUDA versions recorded in every run config.
- `multiprocessing.set_start_method("spawn")` set explicitly everywhere — Windows default, and prevents the silent fork-with-CUDA breakage when moving to Linux.
- `pathlib` for all paths; no OS-specific IPC (see live feed below); no Windows-only APIs. The codebase must run unmodified on the Spark.

### Expected performance regime (set expectations now, not at M5)

With python-chess as the rules engine, **self-play is CPU-bound, not GPU-bound**: a 6×64 net evaluates a 32-leaf batch in well under a millisecond on the 5090, while tree descent and legal-move generation in pure Python cost far more. Plan for: low simulation counts during training (100–400, config), throughput in the hundreds of games/hour (not thousands), and these mitigations from the start: push/pop instead of `Board.copy()` in tree descent, cached legal-move lists per node, incremental position encoding. A C++/Rust move-generation backend is the **expected outcome of M5 profiling**, as a contained swap behind the `chess_env` interface — not a remote contingency.

## Success Criteria

1. The **Elo-over-training-time curve** is the primary metric. Every run produces one automatically via the evaluator daemon (see Evaluation).
2. A full training run can be launched, resumed, monitored live, and compared against other runs without code changes (config-only).
3. The user can at any time: watch live self-play, play any checkpoint, or pit any two players against each other with a settable move delay.
4. The smoke pipeline — **single-process path only** (no worker spawn, no server, no evaluator) — completes in under ~2 minutes and is the default regression check.

## Architecture

Two independent halves connected only by the run directory format and one live feed:

```
C:\Chess\
├── chessrl/              # Python package — the training system
│   ├── chess_env/        #   rules wrapper (python-chess), position & move encoding
│   ├── model/            #   policy-value network (configurable ResNet tower)
│   ├── mcts/             #   PUCT tree search (reference + batched implementations)
│   ├── selfplay/         #   parallel self-play workers, live-feed publisher
│   ├── training/         #   replay buffer, train loop, checkpointing
│   ├── evaluation/       #   Elo ladder: anchors, round-robins, rating fits
│   ├── supervised/       #   optional PGN (Lichess) pretraining
│   └── config/           #   experiment configs (dataclasses + YAML overrides)
├── server/               # FastAPI app — REST + websockets, GameRoom model
├── web/                  # static frontend — chessground board, charts (no build step)
├── scripts/              # train.py, evaluate.py, serve.py, play_match.py
├── tests/                # unit / integration / behavioral
└── runs/                 # gitignored artifacts:
    ├── <run-id>/         #   config snapshot, checkpoints, games (PGN + training
    │                     #   records), metrics (JSONL + SQLite, WAL mode)
    ├── ladder.sqlite     #   global match-results store — evaluator is the ONLY writer
    └── ladder_inbox/     #   JSON result files from other processes (server arena
                          #   games); evaluator ingests and deletes
```

### Process model

- **Trainer** (one process): owns the SGD loop, the replay buffer, and supervision of self-play worker processes (restarting dead workers — note: each restart on Windows spawn pays ~10–20 s of interpreter + CUDA init, so workers are designed not to crash routinely). Knows nothing about the server. Writes all observable state to `runs/<run-id>/`.
- **Self-play workers** (N processes, config-capped with a documented VRAM budget — each PyTorch CUDA context costs ~0.5–0.8 GB before the model): generate games, publish in-progress moves to the live feed.
- **Live feed:** ZeroMQ PUB/SUB over `tcp://127.0.0.1`, one topic per game, **bounded HWM with drop-on-full**. Publishing never blocks a worker; a stalled or absent subscriber loses frames, not training time. Works identically on Windows and Linux (CPython has no `AF_UNIX` on Windows).
- **Server** (independent process): reads run directories (read-only contract — the server can never corrupt training state), subscribes to the live feed, loads its own checkpoint copies for human-vs-AI and arena play. **Server-side inference runs on CPU by default** (a small net at interactive simulation counts is fine on CPU; GPU is opt-in) so playing against the agent doesn't contend with training on the single laptop GPU. Arena results are submitted as JSON files to `runs/ladder_inbox/`, not written to any database.
- **Evaluator** (daemon): polls `runs/<run-id>/checkpoints/` and evaluates every Nth checkpoint per a config-driven policy — this is what makes the Elo curve automatic. Also ingests `ladder_inbox/`. Sole writer of `ladder.sqlite`. On a single-GPU machine it necessarily time-slices against training; the schedule (eval frequency, games per rung) is config and the default is conservative.

This mirrors AlphaZero's actor/learner/evaluator separation and makes the DGX Spark migration trivial: trainer on the Spark, server anywhere, browser on the laptop.

## Training Core

### Position encoding
`8×8×~22` planes: 12 piece-placement planes (6 types × 2 colors) + side-to-move, castling rights, en-passant, fifty-move counter (normalized to [0,1] by /100), and **2 repetition-count planes** (position seen once / twice before, à la AlphaZero — without them the value head is structurally blind to threefold-repetition draws, exactly the regime self-play chess converges to). **No 8-position history stack initially**; adding history is an early architecture experiment, but repetition planes are base infrastructure, not part of that experiment.

**Value convention (normative, unit-tested):** the value target is the game outcome **from the perspective of the side to move** in the encoded position: +1 = side to move went on to win. The classic sign bug gets one explicit test.

### Move encoding
AlphaZero's `8×8×73 = 4672` action space (56 queen-style moves, 8 knight moves, 9 underpromotions per from-square). Illegal moves masked before softmax. Fixed infrastructure, not an experiment axis.

### Network
Configurable ResNet: conv tower → policy head + value head. **Policy head is conv 1×1 → 73 planes reshaped to 4672 logits** (AZ-style, ~5k params), *not* flatten→FC (which would add ~19M params and dominate a small tower). Value head: conv 1×1 → FC → scalar, tanh. Start **6 blocks × 64 filters** for fast pipeline debugging; scale via config (10×128+ later). All architectural knobs are config fields.

### MCTS
PUCT per the AlphaZero paper. Search hyperparameters are enumerated config fields with chess-standard defaults, not hardcoded: Dirichlet α = 0.3, noise weight ε = 0.25 (root only, **self-play only — noise and temperature are per-context flags, always off in evaluation and UI play**), c_puct ≈ 1.5, FPU reduction (explicit policy, it matters a lot at low sim counts), temperature 1.0 for the first 30 plies then →0 (schedule is config).

Two implementations, both kept permanently:
1. **Reference MCTS** — simple, sequential, verified correct (must solve mate-in-2 by search alone). The diffing baseline.
2. **Batched MCTS** — per worker: ~32 concurrent game trees; each batching round selects up to **K leaves per tree** (virtual loss applied within each tree to diversify selection), giving GPU batches of up to 32×K. K is config; K=1 is the safe default, K>1 trades visit-distribution fidelity for batch size. **Subtree reuse between moves is in scope for M5** (~2–4× speedup).

### Self-play game termination
- **Ply cap** (default 512, config): games hitting it are adjudicated as draws (optionally by material, config).
- **Resignation:** resign when root value < threshold (default −0.95) for N consecutive own moves. **10% of resignable games are played out anyway** to measure the false-positive rate, which is a tracked metric with an alert threshold (~5%) — a miscalibrated resign threshold is a documented training-collapse cause (minigo).

### Self-play workers
Output per move: `(position encoding, sparse MCTS visit distribution, outcome)` training records, plus PGN for human viewing. **Records are stored sparse**: legal-move indices + visit counts (~300 B/position vs ~19 KB dense) and int8 planes, expanded to dense tensors at batch time.

### Trainer
- Uniform sampling from a sliding replay window (default ~500k positions, config). With sparse storage this is well under 1 GB RAM.
- Loss = policy cross-entropy (vs visit distribution) + value MSE + weight decay. Mixed precision by default.
- **Pacing (normative):** the trainer targets a configured **samples-per-position ratio** (default ~2 over a position's buffer lifetime; lc0-style "train every N games"). When training gets ahead of generation it waits. The generation:consumption ratio is a first-class dashboard metric — an unthrottled trainer lapping a slow buffer is the canonical hobby-AZ collapse mode.
- Checkpoint every K steps; checkpoints are the unit of evaluation, rating, and UI play.
- **Resume** (`train.py --resume <run-id>`): model + optimizer state from the last checkpoint; the replay buffer is **reconstructed from the most recent training records in the run dir** — no buffer serialization format, resume is nearly free.

## Evaluation & Elo Ladder

### Roster
| Tier | Opponents | Purpose |
|---|---|---|
| Floor | random mover, greedy material-grabber, 1-ply minimax | signal in the first hours |
| Mid rungs | depth-2/3 minimax (with eval noise); Stockfish at tiny fixed node counts (`nodes` = 1, 10, 100, …) | **fills the ~1000–1350 dead zone** between the floor and the weakest calibrated anchor; orderable even if not absolutely calibrated |
| Anchors | Stockfish `UCI_LimitStrength`, `UCI_Elo` ∈ {1320, 1500, 1700, 2000, 2300, 2700} | absolute calibrated Elo (1320 is Stockfish's floor) |
| Peers | the run's own previous checkpoints | smooth relative progress between rungs |

**Stockfish conditions are pinned in config and recorded per match:** engine version (UCI_Elo calibration changes across releases — an unpinned anchor silently moves), `Threads=1`, `Ponder=off`, fixed `movetime` or `nodes`. The binary path + version is checked at evaluator startup (provisioning: user-supplied binary, path in config, health-checked).

### Rating method
All match results (opponent, color, result, full conditions) append to `runs/ladder.sqlite` (WAL mode, `busy_timeout`; evaluator is the only writer). Ratings are refit jointly over the whole results graph with a maximum-likelihood model with an explicit **draw parameter** (Davidson/Rao-Kupper — chess vs anchors is draw-heavy and result=0.5 hacks bias the fit) plus a **weak prior / pseudo-game regularizer** so checkpoints at 100% or 0% against everything they've played get finite ratings with honest error bars (an unregularized MLE diverges there). Anchors pinned at known ratings. Per-run Elo curves are derived by filtering the global store to that run's checkpoints.

### Match protocol
- Openings drawn from a **suite of ~50 short book positions** (deterministic players at fixed openings produce exactly 2 distinct games per pairing — a small fixed set silently collapses the effective sample size). No evaluation-time temperature: it would change the thing being measured.
- **Exactly equal game counts per color** per pairing (normative — otherwise the fit absorbs color advantage into ratings).
- Agent plays at a configured simulation count; "Elo vs simulations-per-move" is a supported first-class experiment.
- Evaluation games saved as PGN, browsable in the UI.

## Web Server & UI

**Server:** FastAPI. REST for the catalog (runs, checkpoints, ratings, stored games); websockets for live content. Core abstraction: **GameRoom** — every active board (mirrored self-play game, human game, arena match) is a room broadcasting moves + metadata (eval, visit counts) to subscribers.

**Frontend:** static HTML/JS served by FastAPI, no build toolchain. Board: **chessground** (Lichess's library). Small chart library for curves.

**Views:**
1. **Dashboard** — run list, live loss/Elo curves, throughput (games/hr, positions/sec, generation:consumption ratio, resign false-positive rate).
2. **Live training** — a **sampled, capped subset** of in-progress games (default 12 boards, rotating; the server subscribes per-game-topic, never to the full firehose — full concurrency can be 256 games). Click to focus, with net eval and top policy moves overlaid. Purely observational.
3. **Play** — choose checkpoint, color, simulations; drag-and-drop play with eval bar and optional agent "thoughts" (top candidates with visit %). Interpretability tool: distinguishes policy misses from value misjudgments.
4. **Arena** — any two players (checkpoints, Stockfish levels, baselines); live-adjustable move-delay slider, pause / step / jump-to-end. Results are submitted to `ladder_inbox/` for the evaluator to ingest.

## Experimentation Workflow

- A run is fully described by one config (dataclasses + YAML overrides): network size, MCTS sims + search hyperparameters (α, ε, c_puct, FPU, temperature schedule), buffer window, pacing ratio, resignation policy, LR, evaluator schedule, Stockfish conditions, etc.
- Launch (`train.py --config …`) snapshots the resolved config + git commit hash into the run dir: every curve is traceable to exactly what produced it.
- Metrics: JSONL + SQLite (WAL) in the run dir (read by the dashboard) mirrored to TensorBoard. No cloud dependency; W&B optional later.
- Comparing experiments = overlaying run curves on the dashboard.
- **Disk retention:** training records and PGNs are append-only but checkpoints follow a retention policy (keep every Nth + all ladder-rated ones, config); a long run on a laptop SSD must not require manual cleanup.

## Error Handling

- Self-play workers are crash-isolated: a dead worker loses only in-flight games and is restarted by the trainer (with restart-rate alerting, since Windows spawn restarts are expensive).
- Training is fully resumable from the run dir (checkpoint + optimizer state + buffer reconstruction from records).
- The server treats run dirs as **read-only**; arena results go through `ladder_inbox/`. Single-writer rule for every SQLite file; WAL + busy_timeout everywhere a reader coexists with the writer.
- Stockfish/UCI subprocesses run under timeouts with auto-restart (UCI engines occasionally hang; the evaluator must not).

## Testing

- **Determinism policy:** all tests and the overfit/smoke gates run with fixed seeds (torch, Python, NumPy); seeds are config fields.
- **Unit:** encoding round-trips (move → index → move across tricky positions incl. promotions/en-passant/castling), value-perspective sign test, repetition-plane test, Elo fit vs hand-computed cases (including the all-wins regularization behavior), replay buffer semantics, sparse→dense target expansion.
- **Integration:** *smoke pipeline* — tiny net, few games, few steps, **single process** (no spawn, no server, no evaluator), end-to-end in <~2 min; the default regression check. *Overfit test* — seeded, loss → ~0 on 10 fixed games proves learning plumbing.
- **Behavioral:** mate-in-1/mate-in-2 puzzle suites scored per checkpoint, tracked as a metric (search-solved vs policy-solved reported separately).

## Milestones

1. **M1** Rules wrapper + position/move encodings, fully tested
2. **M2** Network + overfit-on-tiny-PGN sanity check (seeds the optional SL module)
3. **M3** Reference (unbatched) MCTS — must solve mate-in-2 by search alone
4. **M4** End-to-end self-play → buffer → train → checkpoint loop, single process, toy scale (this path *is* the smoke pipeline)
5. **M5** Batched parallel self-play + subtree reuse; profile on the 5090 against the expected CPU-bound regime; decide on the C++ move-gen swap with data in hand
6. **M6** Elo evaluator daemon + Stockfish ladder + rating fit
7. **M7** Web server + four UI views + live feed
8. **M8** First real training run; experiment program begins

From M4 onward the system always works end-to-end; later milestones make it faster, measurable, and watchable.

### Post-M8 experiment backlog (initial)
- History planes on/off; network depth/width sweeps
- Simulations-per-move vs Elo (training-time and play-time)
- Supervised warm-start (Lichess PGNs) vs tabula rasa
- Training *against* ladder opponents vs pure self-play (status-quo comparison)
- External framework baseline (LightZero) as ladder opponent / curve benchmark

## Out of Scope (for now)

- MuZero/Gumbel variants, transformer policies (future experiments, enabled by the config-driven model module)
- Distributed multi-node training
- Public deployment / auth (server is LAN-only, trusted network)
