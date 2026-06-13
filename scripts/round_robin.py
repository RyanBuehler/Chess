#!/usr/bin/env python
"""Round-robin orchestrator (spec sec 14 / plan Task 5.1).

Rotate over the four goal-playground arms, advancing each by a fixed slice of
games per round (default 1,000) via the existing trainer's ``--resume`` path,
until every arm reaches the budget (default 30,000 games). Each arm gets the
WHOLE GPU during its slice (no shared-GPU contention, no samples-per-position
drift), while all curves climb in lockstep for live side-by-side watching on
``/compare.html``.

Progress is re-derived from each arm's ``run_dir/state.json`` (``{"games": N}``)
on every pass, so the orchestrator is restart-resilient: kill it and relaunch
and it picks up exactly where each arm's persisted game count left off. Arms
already at budget are skipped; when every arm is at budget the orchestrator
exits.

Each arm runs from its own ``experiments/*.yaml`` (distinct feed-port ranges)
and writes to a run dir whose name is the arm's ``run_name``. The first time an
arm is touched it must already have a resumable run dir (created by launching
the YAML once); the orchestrator only resumes -- it does not create runs (so a
mistyped arm name fails loudly rather than silently spawning a fresh run).

Usage:
  python scripts/round_robin.py                          # defaults: 4 arms -> 30k, 1k/round
  python scripts/round_robin.py --budget 30000 --round-size 1000
  python scripts/round_robin.py --arms gp-vanilla gp-lp-goal --runs-root runs
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_ARMS = ("gp-vanilla", "gp-always-win", "gp-random-goal", "gp-lp-goal")
DEFAULT_BUDGET = 30_000
DEFAULT_ROUND_SIZE = 1_000


def read_games(run_dir: Path) -> int:
    """Current persisted game count for an arm, 0 if no state.json yet.

    The trainer flushes state.json every cycle (and in its finally block), so
    this is the authoritative restart-resilient progress signal."""
    state = Path(run_dir) / "state.json"
    if not state.exists():
        return 0
    try:
        return int(json.loads(state.read_text()).get("games", 0))
    except (json.JSONDecodeError, ValueError, OSError):
        return 0


def next_target(current: int, budget: int, round_size: int) -> int | None:
    """The cumulative game target for an arm's next slice, or None if at budget.

    Advances by round_size but never past budget, so the final slice is short
    when current is not a multiple of round_size."""
    if current >= budget:
        return None
    return min(current + round_size, budget)


def build_resume_cmd(
    arm: str, target: int, runs_root: str, python_exe: str
) -> list[str]:
    """The trainer command that resumes ``arm`` until it has ``target`` total
    games. ``--games`` is the trainer's TOTAL-new-games-this-invocation budget;
    the slice size is target-current, computed by the caller."""
    return [
        python_exe, "-u", "scripts/train.py", "--parallel",
        "--resume", arm, "--runs-root", runs_root,
        "--games", str(target),
    ]


def plan_round(
    arms, runs_root: Path, budget: int, round_size: int
) -> list[tuple[str, int, int]]:
    """One rotation's worth of work: for each arm not yet at budget, the
    (arm, current_games, next_target) it should advance to this round.

    Pure (no side effects) so the schedule is unit-testable: re-derives each
    arm's current games from state.json, skips arms already at budget."""
    plan = []
    for arm in arms:
        current = read_games(runs_root / arm)
        tgt = next_target(current, budget, round_size)
        if tgt is None:
            continue
        plan.append((arm, current, tgt))
    return plan


def advance_arm(
    arm: str, target: int, runs_root: Path, python_exe: str, runner=None
) -> int:
    """Resume one arm for a single slice up to ``target`` total games. Returns
    the subprocess return code. ``runner`` is injectable for tests (default:
    subprocess.run from the repo root)."""
    cmd = build_resume_cmd(arm, target, str(runs_root), python_exe)
    if runner is None:
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
        return proc.returncode
    return runner(cmd)


def round_robin(
    arms=DEFAULT_ARMS,
    runs_root="runs",
    budget=DEFAULT_BUDGET,
    round_size=DEFAULT_ROUND_SIZE,
    python_exe=None,
    runner=None,
    max_rounds: int | None = None,
) -> list[tuple[str, int]]:
    """Drive arms in rotation until every arm reaches budget.

    Returns the ordered list of (arm, target) slices actually launched -- the
    schedule, for inspection/testing. Progress is re-read from state.json each
    pass, so a stub runner that bumps state.json reproduces the real loop's
    advancement exactly.

    ``max_rounds`` caps the number of rotations (None = until all at budget);
    a safety bound so a runner that fails to advance an arm cannot spin forever.
    """
    runs_root = Path(runs_root)
    python_exe = python_exe or sys.executable
    launched: list[tuple[str, int]] = []
    rounds = 0
    while True:
        plan = plan_round(arms, runs_root, budget, round_size)
        if not plan:
            break  # every arm at budget
        if max_rounds is not None and rounds >= max_rounds:
            break
        for arm, _current, target in plan:
            rc = advance_arm(arm, target, runs_root, python_exe, runner=runner)
            launched.append((arm, target))
            if rc != 0:
                # A failed slice: report and stop rather than hammering a broken
                # arm. The next launch re-derives progress from state.json.
                print(f"[round_robin] arm {arm} exited rc={rc}; stopping.", file=sys.stderr)
                return launched
        rounds += 1
    return launched


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS),
                    help="run-dir names (arm run_names) to rotate over")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="per-arm total games target (default 30000)")
    ap.add_argument("--round-size", type=int, default=DEFAULT_ROUND_SIZE,
                    help="games to advance each arm per round (default 1000)")
    ap.add_argument("--max-rounds", type=int, default=None,
                    help="cap rotations (default: until all arms at budget)")
    args = ap.parse_args(argv)

    launched = round_robin(
        arms=args.arms,
        runs_root=args.runs_root,
        budget=args.budget,
        round_size=args.round_size,
        max_rounds=args.max_rounds,
    )
    print(f"[round_robin] launched {len(launched)} slice(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
