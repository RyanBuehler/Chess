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

import chess

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

    try:
        rates = {}
        for kind in goal_kinds:
            achieved = sum(1 for i in range(n_games_per_kind) if game_runner(kind, i))
            rates[kind] = achieved / n_games_per_kind if n_games_per_kind else 0.0
    finally:
        # The production runner holds a Stockfish subprocess; release it.
        getattr(game_runner, "close", lambda: None)()

    out = {"stockfish_rates": rates, "n_games_per_kind": n_games_per_kind}
    (run_dir / GOAL_EVAL_FILE).write_text(json.dumps(out, indent=2))
    return out


def _goal_for_kind(kind: str, deadline_max: int):
    """Concrete ``GoalTemplate`` for a goal kind, drawn from the SAME enumeration
    the self-play assigner uses (chessrl.goals.assignment._default_subgoals), so
    the vs-Stockfish achievement rate is comparable to the self-play rate the
    thermometer measures. ``win`` is the apex WIN_GOAL."""
    from chessrl.goals.assignment import _default_subgoals
    from chessrl.goals.templates import WIN_GOAL, GoalTemplate

    if kind == "win":
        return WIN_GOAL
    # First subgoal of the requested kind in the assigner's enumeration (e.g.
    # capture -> capture a pawn), so deadlines match self-play assignments.
    for tmpl in _default_subgoals(deadline_max):
        if tmpl.kind == kind:
            return tmpl
    # Fallback for kinds not in the default enumeration: construct directly with
    # a modest deadline so the runner still covers the kind.
    d = max(1, min(deadline_max, 20))
    ctor = getattr(GoalTemplate, kind, None)
    if ctor is None:
        raise ValueError(f"no goal template for kind {kind!r}")
    return ctor(deadline=d)


def play_goal_eval_game(
    agent_player,
    opponent_player,
    goal,
    *,
    agent_color: bool,
    max_plies: int,
    rng=None,
) -> bool:
    """Play ONE goal-conditioned game and return whether the agent achieved its
    goal by the deadline (the SAME exact verifier the self-play thermometer uses).

    The agent (``agent_player``) plays ``agent_color`` and PURSUES ``goal`` as the
    protagonist on every one of its plies (goal-conditioned search, identical
    machinery to self-play's pure-pursuit path). ``opponent_player`` plays the
    other side with its own ``.play(board)`` (Stockfish in production, a cheap
    stub in tests). The game runs to a real chess result or ``max_plies`` (ply
    cap mirrors the eval match so it can't hang). Achievement is measured by
    ``achieved_by_deadline`` over the accumulated board states, protagonist ==
    ``agent_color``, ``start_ply == 0`` — exactly as ``goal_achievement_rates``.

    ``agent_player`` exposes ``play_goal(board, goal, protagonist) -> move``;
    ``opponent_player`` exposes ``play(board) -> move``.
    """
    from chessrl.chess_env.game import terminal_value
    from chessrl.goals.verifier import achieved_by_deadline

    board = chess.Board()
    states = [board.copy()]
    ply = 0
    while True:
        if terminal_value(board) is not None:
            break
        if ply >= max_plies:
            break
        if board.turn == agent_color:
            move = agent_player.play_goal(board, goal, agent_color)
        else:
            move = opponent_player.play(board)
        board.push(move)
        ply += 1
        states.append(board.copy())
        # Early exit once achieved (cheap; the verifier is also run at the end
        # but stopping early bounds work and is equivalent for a pass/fail rate).
        ok, _ = achieved_by_deadline(states, goal, agent_color, 0)
        if ok:
            return True
    ok, _ = achieved_by_deadline(states, goal, agent_color, 0)
    return ok


class _GoalSearchAgent:
    """Wraps a goal-conditioned net so it can pursue an arbitrary goal kind as
    protagonist (not just g=win like GoalNetMCTSPlayer). Same GoalReferenceMCTS
    machinery as self-play: ``play_goal(board, goal, protagonist) -> move``."""

    def __init__(self, checkpoint_path, network_cfg, simulations: int,
                 device: str = "cpu", seed: int = 0):
        import numpy as np

        from chessrl.config.config import MCTSConfig
        from chessrl.mcts.reference import GoalReferenceMCTS
        from chessrl.model.network import GoalNetEvaluator

        self._eval = GoalNetEvaluator.from_checkpoint(
            checkpoint_path, network_cfg, device=device
        )
        self._mcts = GoalReferenceMCTS(
            self._eval, MCTSConfig(simulations=simulations),
            rng=np.random.default_rng(seed),
        )

    def play_goal(self, board, goal, protagonist):
        from chessrl.chess_env.moves import index_to_move

        visits, _ = self._mcts.search(board, goal, protagonist, add_noise=False)
        best_idx = max(visits, key=visits.get)
        return index_to_move(best_idx, board.turn == chess.BLACK, board)


def _default_goal_game_runner(run_dir):
    """Production vs-Stockfish goal-game runner (spec sec 11, Task 5.4).

    Builds the goal-net agent from the run's latest checkpoint and a fixed-rung
    Stockfish opponent from the run's eval config, then plays goal-conditioned
    games per kind: the agent (alternating colors across games for balance)
    pursues a goal of the requested kind while Stockfish plays the other side,
    and the exact verifier decides achievement. Requires a provisioned Stockfish
    binary (eval.stockfish_path, or the default tools/stockfish); raises if
    absent so the caller can skip rather than silently produce floor rates."""
    from chessrl.evaluation.players import StockfishPlayer, default_stockfish_path

    run_dir = Path(run_dir)
    run_cfg = RunConfig.from_json(run_dir / "config.json")
    eval_cfg = run_cfg.eval

    ckpts = _checkpoints(run_dir)
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints under {run_dir} to evaluate")
    latest = ckpts[-1]

    sf_path = eval_cfg.stockfish_path or default_stockfish_path()
    if not sf_path:
        raise FileNotFoundError(
            "vs-Stockfish goal-eval needs a Stockfish binary (eval.stockfish_path "
            "or tools/stockfish); none found"
        )
    sf_path = str(Path(sf_path).resolve())
    # Fixed rung from the eval config (the strongest calibrated anchor) so the
    # opponent strength is reproducible and matches the eval ladder's top anchor.
    opponent = StockfishPlayer(
        sf_path, elo=1700, movetime_ms=eval_cfg.stockfish_movetime_ms, name="sf_elo1700"
    )

    def runner(kind, i):
        agent = _GoalSearchAgent(
            latest, run_cfg.network, eval_cfg.agent_simulations, device="cpu", seed=i,
        )
        goal = _goal_for_kind(kind, run_cfg.goal.deadline_max)
        # Alternate the agent's color across the kind's games for color balance.
        agent_color = chess.WHITE if (i % 2 == 0) else chess.BLACK
        return play_goal_eval_game(
            agent, opponent, goal,
            agent_color=agent_color, max_plies=eval_cfg.max_plies,
        )

    runner.close = opponent.close  # compute_goal_eval closes the engine when done
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
