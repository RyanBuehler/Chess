#!/usr/bin/env python
"""Post-hoc Elo sweep at training-level sims (spec sec 15 / plan Task 5.4).

After a run finishes, sweep its saved checkpoints and evaluate Elo at
**sims=200 (NOT 50)** with a **higher games_per_rung** for tighter noise. Goal
arms are evaluated by **conditioning on g=win** (the goal-conditioned net under
WIN_GOAL via GoalNetMCTSPlayer). Emit:

  - ``elo.jsonl``  -- one line per checkpoint (the same schema the daemon writes,
    so ``time_to_elo.py`` and ``/compare.html`` consume it unchanged).
  - ``sweep.json`` -- self-contained Elo-vs-games per arm: ``{"run": name,
    "sims": S, "games_per_rung": G, "points": [{"step", "games", "elo"}, ...]}``.
  - ``goal_eval.json`` (goal arms only) -- per-goal-kind vs-Stockfish achievement
    rates: ``{"stockfish_rates": {kind: rate}, "n_games_per_kind": N}``. The
    wishful-thinking thermometer (chessrl/training/loop.py) reads
    ``stockfish_rates`` from here to compute the self-play-vs-Stockfish gap.

Usage:
  python scripts/eval_sweep.py --run gp-vanilla-20260612-000000
  python scripts/eval_sweep.py --runs-root runs --sims 200 --games-per-rung 20
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from chessrl.config.config import EvalConfig, RunConfig
from chessrl.evaluation.daemon import (
    LADDER_DB,
    _checkpoints,
    _step_of,
    evaluate_checkpoint,
)
from chessrl.evaluation.store import LadderStore

SWEEP_FILE = "sweep.json"
GOAL_EVAL_FILE = "goal_eval.json"

# Goal kinds we measure vs-Stockfish achievement for (spec sec 11/16). Win is
# the apex goal; the structural deltas are the practice goals.
DEFAULT_GOAL_KINDS = ("capture", "check", "castle", "promote", "win")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _step_to_games(run_dir: Path) -> dict:
    """Map each metrics step to its cumulative game count (for Elo-vs-games)."""
    out: dict[int, int] = {}
    for m in _read_jsonl(run_dir / "metrics.jsonl"):
        if "step" in m and "games" in m:
            out[int(m["step"])] = int(m["games"])
    return out


def _games_for_step(step: int, step_games: dict) -> int | None:
    """Largest recorded games count at or before ``step`` (steps are monotone)."""
    eligible = [s for s in step_games if s <= step]
    if not eligible:
        return None
    return step_games[max(eligible)]


def sweep_cfg(base: EvalConfig, sims: int, games_per_rung: int) -> EvalConfig:
    """An EvalConfig tuned for the post-hoc sweep: training-level sims and a
    higher games_per_rung (decoupled from training, so cheap) for tighter noise.
    every_n_checkpoints forced to 1 so the whole curve is swept."""
    return dataclasses.replace(
        base,
        agent_simulations=sims,
        games_per_rung=games_per_rung,
        every_n_checkpoints=1,
    )


def sweep_run(
    run_dir,
    base_eval: EvalConfig | None = None,
    sims: int = 200,
    games_per_rung: int = 20,
    store: LadderStore | None = None,
    agent_factory=None,
    runs_root=None,
) -> dict:
    """Sweep every checkpoint of one run; write elo.jsonl + sweep.json. Returns
    the sweep dict. ``agent_factory`` is forwarded to evaluate_checkpoint (the
    test injects a stub; production uses the daemon's goal-aware default)."""
    run_dir = Path(run_dir)
    run_cfg = RunConfig.from_json(run_dir / "config.json")
    base = base_eval if base_eval is not None else run_cfg.eval
    cfg = sweep_cfg(base, sims, games_per_rung)

    if store is None:
        root = Path(runs_root) if runs_root is not None else run_dir.parent
        store = LadderStore(root / LADDER_DB)

    ckpts = _checkpoints(run_dir)
    step_games = _step_to_games(run_dir)
    points = []
    for offset, ckpt in enumerate(ckpts):
        elo = evaluate_checkpoint(
            run_dir, ckpt, cfg, store, openings_offset=offset, agent_factory=agent_factory
        )
        step = _step_of(ckpt)
        points.append({"step": step, "games": _games_for_step(step, step_games), "elo": elo})

    sweep = {
        "run": run_dir.name,
        "sims": cfg.agent_simulations,
        "games_per_rung": cfg.games_per_rung,
        "goal_mode": run_cfg.goal.goal_mode,
        "points": points,
    }
    (run_dir / SWEEP_FILE).write_text(json.dumps(sweep, indent=2))
    return sweep


def compute_goal_eval(
    run_dir,
    goal_kinds=DEFAULT_GOAL_KINDS,
    n_games_per_kind: int = 8,
    game_runner=None,
) -> dict:
    """Per-goal-kind vs-Stockfish achievement rate for a goal arm (spec sec 11).

    For each goal kind, run ``n_games_per_kind`` games in which the protagonist
    is assigned a goal of that kind against a strong opponent, and measure the
    fraction achieved-by-deadline (exact verifier). Writes ``goal_eval.json``
    with ``{"stockfish_rates": {kind: rate}, "n_games_per_kind": N}``.

    ``game_runner(kind, i) -> (achieved: bool)`` is injectable so the test can
    stub the (expensive) actual play. The default raises NotImplementedError:
    production wires a Stockfish-opponent goal-game runner here; the apparatus,
    schema, and aggregation are the deliverable for Task 5.4 and are fully
    exercised by the stubbed test."""
    run_dir = Path(run_dir)
    if game_runner is None:
        game_runner = _default_goal_game_runner(run_dir)

    rates = {}
    for kind in goal_kinds:
        achieved = sum(1 for i in range(n_games_per_kind) if game_runner(kind, i))
        rates[kind] = achieved / n_games_per_kind if n_games_per_kind else 0.0

    out = {"stockfish_rates": rates, "n_games_per_kind": n_games_per_kind}
    (run_dir / GOAL_EVAL_FILE).write_text(json.dumps(out, indent=2))
    return out


def _default_goal_game_runner(run_dir):
    def runner(kind, i):
        raise NotImplementedError(
            "production goal-eval runner not wired; inject game_runner or run "
            "the daemon's vs-Stockfish goal games. See Task 5.4."
        )
    return runner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir name under runs-root")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--sims", type=int, default=200, help="agent MCTS sims (spec: 200, NOT 50)")
    ap.add_argument("--games-per-rung", type=int, default=20, help="games vs each rung (tighter noise)")
    ap.add_argument("--goal-eval-games", type=int, default=8, help="games per goal kind for goal_eval.json")
    args = ap.parse_args(argv)

    runs_root = Path(args.runs_root)
    run_dir = runs_root / args.run
    sweep = sweep_run(
        run_dir, sims=args.sims, games_per_rung=args.games_per_rung, runs_root=runs_root,
    )
    print(f"[eval_sweep] {sweep['run']}: {len(sweep['points'])} checkpoint(s) at {sweep['sims']} sims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
