"""Evaluator daemon: the single writer that makes the Elo curve automatic.

evaluate_checkpoint plays a checkpoint against the ladder, records results into
ladder.sqlite, refits ratings over the WHOLE store (anchors pinned), and appends
the checkpoint's Elo to run_dir/elo.jsonl. watch() polls runs for new every-Nth
checkpoints and ingests ladder_inbox, stopping when runs_root/EVAL_STOP appears.
Synchronous and single-process by design (no threads); the GPU is time-sliced
against training, so the agent player runs on CPU by default.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from chessrl.config.config import EvalConfig, RunConfig
from chessrl.evaluation.match import play_pairing
from chessrl.evaluation.players import (
    GreedyMaterialPlayer,
    MinimaxPlayer,
    NetMCTSPlayer,
    RandomPlayer,
    StockfishPlayer,
)
from chessrl.evaluation.ratings import fit_ratings
from chessrl.evaluation.store import LadderStore

STOP_FILE = "EVAL_STOP"
LADDER_DB = "ladder.sqlite"
INBOX_DIR = "ladder_inbox"

# Default Stockfish anchor rungs (pinned) and dead-zone node rungs (unpinned).
_ANCHOR_ELOS = (1320, 1500, 1700)
_NODE_RUNGS = (1, 100)


def _run_id(run_dir: Path) -> str:
    return Path(run_dir).name


def _checkpoints(run_dir) -> list:
    return sorted((Path(run_dir) / "checkpoints").glob("ckpt_*.pt"))


def _step_of(ckpt) -> int:
    return int(Path(ckpt).stem.split("_")[1])


def eligible_checkpoints(run_dir, cfg: EvalConfig, store: LadderStore) -> list:
    """Every Nth checkpoint (by index) not already evaluated."""
    ckpts = _checkpoints(run_dir)
    selected = ckpts[:: cfg.every_n_checkpoints]
    return [c for c in selected if not store.is_evaluated(str(c))]


def _build_floor_players(seed: int) -> list:
    return [
        (RandomPlayer(seed=seed), "floor", None),
        (GreedyMaterialPlayer(seed=seed), "floor", None),
        (MinimaxPlayer(depth=2, seed=seed), "floor", None),
    ]


def _build_stockfish_players(cfg: EvalConfig) -> list:
    """(player, kind, anchor_elo) tuples for Stockfish rungs; empty if disabled."""
    if not cfg.stockfish_path:
        return []
    # Resolve to absolute path so subprocess_exec works on Windows (which requires
    # absolute paths or PATH-resolvable commands, not relative paths).
    sf_path = str(Path(cfg.stockfish_path).resolve())
    out = []
    for nodes in _NODE_RUNGS:
        out.append((StockfishPlayer(sf_path, nodes=nodes,
                                    name=f"sf_nodes{nodes}"), "rung", None))
    for elo in _ANCHOR_ELOS:
        out.append((StockfishPlayer(sf_path, elo=elo,
                                    movetime_ms=cfg.stockfish_movetime_ms,
                                    name=f"sf_elo{elo}"), "anchor", float(elo)))
    return out


def evaluate_checkpoint(run_dir, ckpt_path, cfg: EvalConfig, store: LadderStore,
                        openings_offset: int) -> float:
    """Play ckpt vs the ladder, record results, refit ratings, append elo.jsonl.
    Returns the checkpoint's fitted Elo."""
    run_dir = Path(run_dir)
    run_cfg = RunConfig.from_json(run_dir / "config.json")
    step = _step_of(ckpt_path)
    agent_name = f"{_run_id(run_dir)}@{step}"
    agent = NetMCTSPlayer(
        agent_name, ckpt_path, run_cfg.network, cfg.agent_simulations, device="cpu",
    )
    store.upsert_player(agent_name, kind="agent", anchor_elo=None)

    rungs = _build_floor_players(seed=step) + _build_stockfish_players(cfg)
    eval_games_dir = run_dir / "eval_games"
    eval_games_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, (opp, kind, anchor_elo) in enumerate(rungs):
            store.upsert_player(opp.name, kind=kind, anchor_elo=anchor_elo)
            conditions = opp.conditions() if hasattr(opp, "conditions") else {}
            results = play_pairing(
                agent, opp, games=cfg.games_per_rung,
                openings_start=openings_offset + i * (cfg.games_per_rung // 2),
                max_plies=cfg.max_plies,
            )
            for r in results:
                store.record_result(r.white_name, r.black_name, r.z, r.opening_idx, conditions)
                fname = f"{step:08d}_{opp.name}_{r.opening_idx:02d}_{r.white_name.replace('@','_at_')}.pgn"
                (eval_games_dir / fname).write_text(r.pgn)
    finally:
        for opp, _kind, _elo in rungs:
            getattr(opp, "close", lambda: None)()

    ratings, nu = fit_ratings(store.result_triples(), anchors=store.anchors())
    elo = float(ratings.get(agent_name, float("nan")))
    entry = {"ts": time.time(), "step": step, "ckpt": str(ckpt_path), "elo": elo, "nu": nu}
    with (run_dir / "elo.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    store.mark_evaluated(str(ckpt_path))
    return elo


def run_once(runs_root, cfg: EvalConfig, run_filter: str | None) -> int:
    """Evaluate every eligible un-evaluated checkpoint of each run once; return
    the number of checkpoints evaluated."""
    runs_root = Path(runs_root)
    store = LadderStore(runs_root / LADDER_DB)
    store.ingest_inbox(runs_root / INBOX_DIR)
    count = 0
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir() and (p / "config.json").exists()):
        if run_filter is not None and run_dir.name != run_filter:
            continue
        for offset, ckpt in enumerate(eligible_checkpoints(run_dir, cfg, store)):
            evaluate_checkpoint(run_dir, ckpt, cfg, store, openings_offset=offset)
            count += 1
    return count


def watch(runs_root, cfg: EvalConfig, run_filter: str | None = None) -> None:
    """Poll for new eligible checkpoints + inbox until runs_root/EVAL_STOP exists."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    stop = runs_root / STOP_FILE
    while not stop.exists():
        run_once(runs_root, cfg, run_filter)
        if stop.exists():
            break
        time.sleep(cfg.poll_seconds)


def _load_eval_cfg(config_path: str | None) -> EvalConfig:
    if not config_path:
        return EvalConfig()
    return RunConfig.from_yaml(config_path).eval


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Elo evaluator daemon")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--run", default=None, help="restrict to one run id")
    ap.add_argument("--once", action="store_true", help="evaluate eligible checkpoints then exit")
    ap.add_argument("--config", default=None, help="YAML with an eval: section (overrides EvalConfig)")
    args = ap.parse_args(argv)

    cfg = _load_eval_cfg(args.config)
    if args.once:
        n = run_once(args.runs_root, cfg, args.run)
        print(f"evaluated {n} checkpoint(s)")
        return n
    watch(args.runs_root, cfg, args.run)
    return 0
