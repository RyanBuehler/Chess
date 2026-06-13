#!/usr/bin/env python
"""Time-to-Elo: the standard cross-experiment yardstick.

For each run, report the *first sustained* arrival at an Elo threshold (default
1000, then 2000), measured two ways:

  - games-to-threshold   (sample efficiency; the fair axis across architectures)
  - hours-to-threshold   (wall-clock; the practical axis)

"Sustained" means the median of a sliding window of evals clears the threshold,
so a single noisy spike (evals bounce hard at low games_per_rung) does not count
as having reached it.

Usage:
  python scripts/time_to_elo.py                       # all runs under runs/
  python scripts/time_to_elo.py runs/arch-10x128-*    # specific run dirs
  python scripts/time_to_elo.py --thresholds 1000 1500 2000 --window 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from bisect import bisect_right
from statistics import median


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _step_to_games(metrics: list[dict]) -> tuple[list[int], list[int]]:
    """Sorted (steps, games) so we can map any eval step to a game count."""
    pairs = sorted((m["step"], m["games"]) for m in metrics if "step" in m and "games" in m)
    steps = [s for s, _ in pairs]
    games = [g for _, g in pairs]
    return steps, games


def _games_at_step(step: int, steps: list[int], games: list[int]) -> int | None:
    if not steps:
        return None
    i = bisect_right(steps, step) - 1
    if i < 0:
        i = 0
    return games[i]


def crossing(evals: list[dict], threshold: float, window: int):
    """First eval index where the trailing-window median Elo >= threshold.

    Returns the eval dict at the crossing, or None if never sustained.
    """
    elos = [e["elo"] for e in evals]
    for i in range(len(evals)):
        lo = max(0, i - window + 1)
        win = elos[lo : i + 1]
        if len(win) == window and median(win) >= threshold:
            return evals[i]
    return None


def analyze(run_dir: str, thresholds: list[float], window: int) -> dict:
    evals = sorted(_read_jsonl(os.path.join(run_dir, "elo.jsonl")), key=lambda e: e["step"])
    metrics = _read_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    steps, games = _step_to_games(metrics)

    name = os.path.basename(run_dir.rstrip("/\\"))
    peak = max((e["elo"] for e in evals), default=None)
    t0 = evals[0]["ts"] if evals else None

    out = {"run": name, "peak_elo": peak, "evals": len(evals), "thresholds": {}}
    for th in thresholds:
        ev = crossing(evals, th, window)
        if ev is None:
            out["thresholds"][th] = None
            continue
        g = _games_at_step(ev["step"], steps, games)
        hours = (ev["ts"] - t0) / 3600.0 if t0 is not None else None
        out["thresholds"][th] = {"games": g, "hours": hours, "step": ev["step"], "elo": ev["elo"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run dirs or globs (default: runs/*)")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1000.0, 2000.0])
    ap.add_argument("--window", type=int, default=3, help="evals in the median window (default 3)")
    ap.add_argument("--runs-root", default="runs")
    args = ap.parse_args()

    patterns = args.runs or [os.path.join(args.runs_root, "*")]
    run_dirs = []
    for pat in patterns:
        run_dirs.extend(sorted(d for d in glob.glob(pat) if os.path.isdir(d)))
    run_dirs = [d for d in run_dirs if os.path.exists(os.path.join(d, "elo.jsonl"))]

    if not run_dirs:
        print("No runs with elo.jsonl found.")
        return

    ths = args.thresholds
    head = f"{'run':<34} {'peak':>6}"
    for th in ths:
        head += f" | {int(th)}: {'games':>8} {'hrs':>6}"
    print(head)
    print("-" * len(head))

    for d in run_dirs:
        r = analyze(d, ths, args.window)
        peak = f"{r['peak_elo']:.0f}" if r["peak_elo"] is not None else "  -"
        row = f"{r['run']:<34} {peak:>6}"
        for th in ths:
            hit = r["thresholds"][th]
            if hit is None:
                row += f" | {'':>5}{'not yet':>11} {'':>6}"
            else:
                g = f"{hit['games']:,}" if hit["games"] is not None else "?"
                h = f"{hit['hours']:.1f}" if hit["hours"] is not None else "?"
                row += f" | {int(th):>5}: {g:>8} {h:>6}"
        print(row)


if __name__ == "__main__":
    main()
