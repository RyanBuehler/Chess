# Chess RL — project guide

AlphaZero-style chess RL research/education project. Goal: learning modern RL/NN theory through
experimentation. Elo curve vs Stockfish anchors is the primary success metric.

## Key documents

- **Research exploration notes (living doc): `docs/notes/research-explorations.md`** — design
  brainstorms, challenges, lessons, and the experiment ladder. Append new sessions there.
- Normative spec: `docs/superpowers/specs/2026-06-11-chess-rl-design.md`
- Implementation plans: `docs/superpowers/plans/`

## Operating the system

- Train: `python scripts/train.py --parallel --config experiments/<name>.yaml --games N`
- Evaluate (daemon): `python scripts/evaluate.py --runs-root runs --config <yaml>` (stop via `runs/EVAL_STOP`)
- Serve UI: `python scripts/serve.py --runs-root runs --feed-ports 5555,5556,5557,5558 --stockfish tools/stockfish/stockfish.exe` → http://127.0.0.1:8000
- Stop a run: create a `STOP` file in its run directory.

## Conventions

- Values are from the side-to-move perspective (+1 = side to move wins).
- Action index = from_square * 73 + move_type (mirrored when Black to move).
- Each architecture experiment = one YAML in `experiments/` + one run dir; provenance.json
  records the network shape. Compare runs at `/compare.html`.
- Arena results are recorded but excluded from the Elo rating fit (source="arena").
- Web UI must be fully local — vendored libs in `web/vendor/`, never CDN. UI changes require
  the Playwright browser gate (`tests/test_ui_browser.py`, slow tests).
