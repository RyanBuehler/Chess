#!/usr/bin/env python
"""Pre-registered transfer analysis (spec sec 2 / plan Task 5.5).

The primary metric is **games-to-Elo-theta**, where theta is a fixed threshold
*read off the vanilla arm* (the Elo all arms demonstrably reach; overridable via
``--theta``). For each arm we compute games-to-theta per seed, then bootstrap
over seeds to get a confidence interval on the *fractional reduction* vs vanilla.
The pre-registered decision is then applied verbatim:

  - **CONFIRM:** the arm reaches theta in >= 15% fewer games than vanilla, with
    the seed-bootstrapped CI on the reduction excluding 0.
  - **REFUTE:** the CI overlaps 0, or vanilla reaches theta first (negative
    reduction point estimate).
  - **INCONCLUSIVE:** arms do not separate beyond noise, or an arm fails to reach
    theta within budget -- a distinct, named *failed-measurement* outcome (NOT a
    negative): more budget / seeds required (spec sec 2). A single seed per arm
    is inherently inconclusive (the bootstrap cannot separate signal from noise).

Input curves come from each arm's sweep.json (Task 5.4) or elo.jsonl+metrics.

Usage:
  python scripts/analyze_transfer.py --runs-root runs \
      --vanilla gp-vanilla --arms gp-always-win gp-random-goal gp-lp-goal
  python scripts/analyze_transfer.py --theta 650 ...      # override theta
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

import numpy as np

CONFIRM = "confirm"
REFUTE = "refute"
INCONCLUSIVE = "inconclusive"

MIN_REDUCTION = 0.15   # >= 15% fewer games (spec sec 2)


def games_to_theta(curve, theta: float, window: int = 3) -> float | None:
    """First *sustained* arrival at ``theta`` for one (games, elo) curve.

    "Sustained" = the trailing-window median Elo clears theta (so a single noisy
    spike does not count), mirroring time_to_elo.py. Returns the game count at
    the crossing, or None if theta is never sustained within the curve (the arm
    did not reach theta within budget)."""
    pts = sorted(curve, key=lambda p: p[0])
    elos = [e for _, e in pts]
    games = [g for g, _ in pts]
    for i in range(len(pts)):
        lo = max(0, i - window + 1)
        win = elos[lo : i + 1]
        if len(win) == window and median(win) >= theta:
            return float(games[i])
    return None


def vanilla_theta(vanilla_curves, percentile: float = 90.0) -> float:
    """Theta read off the vanilla arm: a high percentile of the Elo it
    demonstrably reaches across its seed curves (spec sec 2 -- 'the Elo all arms
    demonstrably reach'). Conservative: the peak per seed, then the min across
    seeds so every vanilla seed actually attains it."""
    peaks = [max(e for _, e in c) for c in vanilla_curves if c]
    if not peaks:
        return float("nan")
    # Use the min seed-peak so theta is reachable by every vanilla seed; this is
    # the threshold "all arms demonstrably reach".
    return float(min(peaks))


def _arm_games(curves, theta, window) -> list[float | None]:
    return [games_to_theta(c, theta, window) for c in curves]


def bootstrap_reduction(
    vanilla_g: list[float | None],
    arm_g: list[float | None],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Seed-bootstrap the fractional reduction (vanilla - arm) / vanilla.

    Resamples seeds with replacement *independently* per arm (seeds are not
    paired across arms in this design). A seed that never reached theta
    contributes None and is dropped from that resample's mean; if a resample has
    no valid seeds for either arm it is skipped. Returns point estimate + CI +
    bookkeeping. With a single seed the CI collapses to a point (width 0) and
    cannot exclude 0 -> downstream INCONCLUSIVE."""
    rng = np.random.default_rng(seed)
    v = [x for x in vanilla_g]
    a = [x for x in arm_g]

    def mean_or_none(vals):
        ok = [x for x in vals if x is not None]
        return float(np.mean(ok)) if ok else None

    def reduction(vmean, amean):
        if vmean is None or amean is None or vmean <= 0:
            return None
        return (vmean - amean) / vmean

    point = reduction(mean_or_none(v), mean_or_none(a))

    samples = []
    nv, na = len(v), len(a)
    for _ in range(n_boot):
        vb = [v[i] for i in rng.integers(0, nv, nv)] if nv else []
        ab = [a[i] for i in rng.integers(0, na, na)] if na else []
        r = reduction(mean_or_none(vb), mean_or_none(ab))
        if r is not None:
            samples.append(r)

    if not samples:
        return {"point": point, "ci_low": None, "ci_high": None, "n_boot": 0,
                "vanilla_reach": sum(x is not None for x in v),
                "arm_reach": sum(x is not None for x in a)}

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return {"point": point, "ci_low": lo, "ci_high": hi, "n_boot": len(samples),
            "vanilla_reach": sum(x is not None for x in v),
            "arm_reach": sum(x is not None for x in a)}


def classify(stats: dict, vanilla_g: list, arm_g: list) -> str:
    """Apply the pre-registered decision (spec sec 2) to one arm's bootstrap."""
    v_reached = [x for x in vanilla_g if x is not None]
    a_reached = [x for x in arm_g if x is not None]

    # Failed measurement: an arm (or vanilla) never reaches theta within budget,
    # or the bootstrap produced no usable samples -> INCONCLUSIVE, not negative.
    if not v_reached or not a_reached or stats["ci_low"] is None:
        return INCONCLUSIVE

    # A single reaching seed per arm gives a degenerate (zero-width) bootstrap CI
    # that cannot separate signal from noise. Spec sec 2: Phase 1 (1 seed/arm) is
    # explicitly exploratory-only; a confirmation/refutation needs >= 2 seeds.
    if len(v_reached) < 2 or len(a_reached) < 2:
        return INCONCLUSIVE

    point = stats["point"]
    lo, hi = stats["ci_low"], stats["ci_high"]

    # Vanilla first (arm slower): negative reduction -> REFUTE.
    if point is not None and point < 0:
        return REFUTE

    # CONFIRM: >= 15% fewer games AND the CI on the reduction excludes 0.
    if point is not None and point >= MIN_REDUCTION and lo > 0.0:
        return CONFIRM

    # CI overlaps 0 (cannot exclude no-effect). With a real multi-seed spread
    # that overlaps zero this is REFUTE per spec; a degenerate single-seed (zero-
    # width) CI cannot separate signal from noise -> INCONCLUSIVE.
    if lo <= 0.0 <= hi:
        if lo == hi:                       # degenerate (e.g. 1 seed) -> failed measurement
            return INCONCLUSIVE
        return REFUTE

    # Reduction positive but below the 15% bar, CI excludes 0: real but too small
    # to confirm the pre-registered effect size -> does not separate -> REFUTE.
    return REFUTE


def analyze(
    arm_curves: dict,
    vanilla_name: str,
    theta: float | None = None,
    window: int = 3,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Full transfer analysis. ``arm_curves`` maps arm-name -> list-of-seed-
    curves, each curve a list of (games, elo). Returns per-arm decisions."""
    vanilla_curves = arm_curves[vanilla_name]
    th = float(theta) if theta is not None else vanilla_theta(vanilla_curves)
    vanilla_g = _arm_games(vanilla_curves, th, window)

    out = {"theta": th, "vanilla": vanilla_name, "vanilla_games_to_theta": vanilla_g, "arms": {}}
    for arm, curves in arm_curves.items():
        if arm == vanilla_name:
            continue
        arm_g = _arm_games(curves, th, window)
        stats = bootstrap_reduction(vanilla_g, arm_g, n_boot=n_boot, ci=ci, seed=seed)
        decision = classify(stats, vanilla_g, arm_g)
        out["arms"][arm] = {
            "games_to_theta": arm_g,
            "reduction": stats["point"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "decision": decision,
            "vanilla_reach": stats["vanilla_reach"],
            "arm_reach": stats["arm_reach"],
        }
    return out


# --- curve loading from the run dirs -------------------------------------

def load_curve(run_dir: Path) -> list:
    """Load one run's (games, elo) curve, preferring sweep.json (Task 5.4) and
    falling back to elo.jsonl + metrics.jsonl (the daemon's live output)."""
    sweep = run_dir / "sweep.json"
    if sweep.exists():
        data = json.loads(sweep.read_text())
        return [(p["games"], p["elo"]) for p in data["points"]
                if p.get("games") is not None and p.get("elo") is not None]
    # Fallback: join elo.jsonl (step,elo) to metrics.jsonl (step,games).
    step_games = {}
    mfile = run_dir / "metrics.jsonl"
    if mfile.exists():
        for line in mfile.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                if "step" in m and "games" in m:
                    step_games[int(m["step"])] = int(m["games"])
    curve = []
    efile = run_dir / "elo.jsonl"
    if efile.exists():
        for line in efile.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                step = int(e["step"])
                eligible = [s for s in step_games if s <= step]
                if eligible:
                    curve.append((step_games[max(eligible)], float(e["elo"])))
    return curve


def collect_arm_curves(runs_root: Path, arm_specs: dict) -> dict:
    """arm_specs maps arm-name -> list of run-dir names (the seeds). Returns
    arm-name -> list of curves."""
    out = {}
    for arm, run_names in arm_specs.items():
        out[arm] = [load_curve(runs_root / rn) for rn in run_names]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--vanilla", default="gp-vanilla", help="vanilla arm name (also a run-dir glob base)")
    ap.add_argument("--arms", nargs="+", default=["gp-always-win", "gp-random-goal", "gp-lp-goal"])
    ap.add_argument("--theta", type=float, default=None, help="override theta (default: read off vanilla)")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import glob as _glob
    runs_root = Path(args.runs_root)

    def seeds_for(prefix):
        # All run dirs whose name starts with the arm prefix are its seeds.
        return sorted(Path(d).name for d in _glob.glob(str(runs_root / f"{prefix}*"))
                      if Path(d).is_dir())

    specs = {args.vanilla: seeds_for(args.vanilla)}
    for a in args.arms:
        specs[a] = seeds_for(a)
    arm_curves = collect_arm_curves(runs_root, specs)

    result = analyze(arm_curves, args.vanilla, theta=args.theta,
                     window=args.window, n_boot=args.n_boot, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
