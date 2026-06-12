"""M5 profiling gate: measure concurrent self-play throughput and locate the
hot path. The C++/Rust move-gen decision is made from this output (recorded in
the milestone summary; the swap itself is not part of M5)."""
import argparse
import cProfile
import pstats
import time
from pathlib import Path

import numpy as np
import torch

from chessrl.config.config import RunConfig
from chessrl.model.network import BatchedNetEvaluator, PolicyValueNet
from chessrl.selfplay.concurrent import play_games_concurrent


def _resolve_device(requested: str) -> str:
    return requested if (requested == "cpu" or torch.cuda.is_available()) else "cpu"


def run(cfg: RunConfig, num_games: int, device: str, seed: int = 0):
    torch.manual_seed(seed)
    net = PolicyValueNet(cfg.network)
    evaluator = BatchedNetEvaluator(net, device=_resolve_device(device))
    rng = np.random.default_rng(seed)
    return play_games_concurrent(evaluator, cfg.mcts, cfg.selfplay, rng, num_games=num_games)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="YAML config path (defaults to RunConfig defaults)")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--profile", action="store_true", help="wrap in cProfile, print top-20 cumulative")
    args = ap.parse_args(argv)

    cfg = RunConfig.from_yaml(args.config) if args.config else RunConfig()

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
    t0 = time.time()
    results = run(cfg, args.games, args.device)
    wall = time.time() - t0
    if args.profile:
        prof.disable()

    positions = sum(len(rec) for rec, *_ in results)
    sims = positions * cfg.mcts.simulations
    print(f"device:           {_resolve_device(args.device)}")
    print(f"games:            {len(results)}")
    print(f"positions:        {positions}")
    print(f"wall_seconds:     {wall:.3f}")
    print(f"games_per_hour:   {len(results) / wall * 3600:.1f}")
    print(f"positions_per_s:  {positions / wall:.1f}")
    print(f"simulations_per_s:{sims / wall:.1f}  (approx: positions * cfg.mcts.simulations)")

    if args.profile:
        stats = pstats.Stats(prof).sort_stats("cumulative")
        stats.print_stats(20)


if __name__ == "__main__":
    main()
