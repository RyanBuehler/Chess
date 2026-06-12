# Chess RL System — Design

**Date:** 2026-06-11
**Status:** Approved (brainstorming complete)
**Goal:** A research/education platform for training AlphaZero-style policy+value networks to play chess via self-play, with an Elo ladder as the primary metric and a web UI for watching, playing, and running experiments.

## Context & Constraints

- **Purpose:** learning modern RL and NN theory through hands-on experimentation. Fast experiment turnaround matters more than maximum playing strength.
- **Hardware:** RTX 5090 laptop now; NVIDIA DGX Spark later for longer offline runs. Design must make that migration trivial.
- **User posture:** strong engineer with some ML/RL experience. Libraries for commodity parts; the RL system itself is hand-built so it can be instrumented, decomposed, and modified.
- **Approach chosen:** build-your-own AlphaZero (vs adopting LightZero/OpenSpiel). An external framework baseline may be added later as an experiment, not as architecture.

## Success Criteria

1. The **Elo-over-training-time curve** is the primary metric. Every run produces one automatically.
2. A full training run can be launched, resumed, monitored live, and compared against other runs without code changes (config-only).
3. The user can at any time: watch live self-play, play any checkpoint, or pit any two players against each other with a settable move delay.
4. The smoke pipeline (tiny config, end-to-end) completes in under ~2 minutes and is the default regression check.

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
└── runs/                 # per-run artifacts (gitignored): config snapshot,
                          #   checkpoints, games (PGN + training records),
                          #   metrics (JSONL/SQLite), results store
```

### Process model

- **Trainer** (headless): orchestrates self-play workers and the SGD loop. Knows nothing about the server. Writes all observable state to `runs/<run-id>/`. Self-play workers additionally publish in-progress moves to a lightweight local pub/sub socket (the live feed). Training is never blocked or slowed by observers.
- **Server** (independent process): reads run directories (read-only contract), subscribes to the live feed, and loads its own copies of checkpoints to host human-vs-AI and arena games — so interactive play never competes with training for GPU mid-run and can use higher simulation counts than training.
- **Evaluator**: separate scheduled/on-demand process so it never steals training compute unbidden.

This mirrors AlphaZero's actor/learner/evaluator separation and makes the DGX Spark migration trivial: trainer on the Spark, server anywhere, browser on the laptop.

## Training Core

### Position encoding
`8×8×~20` planes: 12 piece-placement planes (6 types × 2 colors) + side-to-move, castling rights, en-passant, fifty-move counter. **No 8-position history stack initially** — smaller and faster; adding history is an early architecture experiment.

### Move encoding
AlphaZero's `8×8×73 = 4672` action space (56 queen-style moves, 8 knight moves, 9 underpromotions per from-square). Illegal moves masked before softmax. Fixed infrastructure, not an experiment axis.

### Network
Configurable ResNet: conv tower → policy head (4672 logits) + value head (scalar, tanh). Start **6 blocks × 64 filters** for fast pipeline debugging; scale via config (10×128+ later). All architectural knobs are config fields.

### MCTS
PUCT per the AlphaZero paper: prior-weighted UCB selection, Dirichlet noise at root during self-play, temperature sampling for first ~30 plies then greedy.

Two implementations, both kept permanently:
1. **Reference MCTS** — simple, sequential, verified correct (must solve mate-in-2 by search alone). The diffing baseline.
2. **Batched MCTS** — each worker runs ~32 concurrent games; unevaluated leaves are parked into a batch using **virtual loss**, evaluated in single GPU calls. This is the performance-critical component; naive per-position inference wastes the GPU.

### Self-play workers
N worker processes (config), each owning its concurrent games with its own CUDA context (simple per-worker model on the 5090; a shared inference-server process is a possible later optimization on the Spark). Output per move: `(position tensor, MCTS visit distribution, final outcome)` training records, plus PGN for human viewing.

### Trainer
Uniform sampling from a sliding replay window (default: most recent ~500k positions, config). Loss = policy cross-entropy (vs visit distribution) + value MSE + weight decay. Mixed precision by default. Checkpoint every K steps; checkpoints are the unit of evaluation, rating, and UI play. Fully resumable: `train.py --resume <run-id>`.

## Evaluation & Elo Ladder

### Roster
| Tier | Opponents | Purpose |
|---|---|---|
| Floor | random mover, greedy material-grabber, 1-ply minimax | signal in the first hours |
| Anchors | Stockfish `UCI_LimitStrength` at `UCI_Elo` ≈ 1350/1500/1700/2000/2300/2700 | absolute calibrated Elo |
| Peers | the run's own previous checkpoints | smooth relative progress between anchors |

### Rating method
All match results (opponent, color, result, conditions) append to a SQLite results store. Ratings are refit jointly over the whole results graph with a maximum-likelihood logistic (Elo) model, anchors pinned at known ratings — BayesElo-style. Joint refits keep checkpoint ratings mutually consistent. Pinned external anchors guard against self-play Elo inflation.

### Match protocol
Each pairing plays a small game set from a few fixed shallow openings, both colors (prevents deterministic duplicate games). Agent plays at a configured simulation count; "Elo vs simulations-per-move" is a supported first-class experiment. Evaluation games saved as PGN, browsable in the UI.

## Web Server & UI

**Server:** FastAPI. REST for the catalog (runs, checkpoints, ratings, stored games); websockets for live content. Core abstraction: **GameRoom** — every active board (mirrored self-play game, human game, arena match) is a room broadcasting moves + metadata (eval, visit counts) to subscribers.

**Frontend:** static HTML/JS served by FastAPI, no build toolchain. Board: **chessground** (Lichess's library). Small chart library for curves.

**Views:**
1. **Dashboard** — run list, live loss/Elo curves, throughput (games/hr, positions/sec).
2. **Live training** — thumbnail grid of in-progress self-play games from the feed; click to focus, with net eval and top policy moves overlaid. Purely observational.
3. **Play** — choose checkpoint, color, simulations; drag-and-drop play with eval bar and optional agent "thoughts" (top candidates with visit %). Interpretability tool: distinguishes policy misses from value misjudgments.
4. **Arena** — any two players (checkpoints, Stockfish levels, baselines); live-adjustable move-delay slider, pause / step / jump-to-end. Results may feed the ratings store.

## Experimentation Workflow

- A run is fully described by one config (dataclasses + YAML overrides): network size, MCTS sims, buffer window, LR, ladder schedule, etc.
- Launch (`train.py --config …`) snapshots the resolved config + git commit hash into the run dir: every curve is traceable to exactly what produced it.
- Metrics: JSONL/SQLite in the run dir (read by the dashboard) mirrored to TensorBoard. No cloud dependency; W&B optional later.
- Comparing experiments = overlaying run curves on the dashboard.

## Error Handling

- Self-play workers are crash-isolated: a dead worker loses only in-flight games and is restarted by the trainer.
- Training state is fully resumable from run dir (buffer + checkpoints + optimizer state).
- The server treats run dirs as **read-only**; it cannot corrupt training state.
- Stockfish/UCI subprocesses run under timeouts with auto-restart (UCI engines occasionally hang; the evaluator must not).

## Testing

- **Unit:** encoding round-trips (move → index → move across tricky positions incl. promotions/en-passant/castling), Elo fit vs hand-computed cases, replay buffer semantics.
- **Integration:** *smoke pipeline* — tiny net, few games, few steps, end-to-end in <~2 min; the default regression check. *Overfit test* — loss → ~0 on 10 fixed games proves learning plumbing.
- **Behavioral:** mate-in-1/mate-in-2 puzzle suites scored per checkpoint, tracked as a metric (search-solved vs policy-solved reported separately).

## Milestones

1. **M1** Rules wrapper + position/move encodings, fully tested
2. **M2** Network + overfit-on-tiny-PGN sanity check (seeds the optional SL module)
3. **M3** Reference (unbatched) MCTS — must solve mate-in-2 by search alone
4. **M4** End-to-end self-play → buffer → train → checkpoint loop, single process, toy scale
5. **M5** Batched parallel self-play; profile on the 5090, set and hit a games/hour target
6. **M6** Elo evaluator + Stockfish ladder
7. **M7** Web server + four UI views
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
- C++ move generation (only if profiling shows python-chess dominates over NN inference)
