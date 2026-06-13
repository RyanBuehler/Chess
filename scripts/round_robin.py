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

Each arm runs from its own ``experiments/<arm>.yaml`` (distinct feed-port
ranges) and writes to a run dir named EXACTLY after the arm (no timestamp). The
orchestrator is create-on-first-touch: an arm whose ``runs/<arm>`` lacks
``config.json`` is launched FRESH (``--config experiments/<arm>.yaml
--run-dir-name <arm>``) for its first slice; thereafter it is resumed
(``--resume <arm>``). So ``python scripts/round_robin.py`` works out of the box.

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


def arm_yaml(arm: str) -> str:
    """The experiment YAML for an arm, by convention ``experiments/<arm>.yaml``."""
    return f"experiments/{arm}.yaml"


def has_run(runs_root: Path, arm: str) -> bool:
    """True once an arm's run dir has been initialized (``config.json`` written
    by a fresh launch). Used to decide create-on-first-touch vs resume."""
    return (Path(runs_root) / arm / "config.json").exists()


def build_slice_cmd(
    arm: str, current: int, target: int, runs_root: str, python_exe: str
) -> list[str]:
    """The trainer command that advances ``arm`` by one slice. ``--games`` is the
    trainer's NEW-games-THIS-invocation budget, so it must be the slice size
    ``target - current`` (NOT the cumulative target). ``next_target`` still
    returns the cumulative target for progress logic; only the value passed to
    ``--games`` is the slice.

    First touch (no ``config.json`` under ``runs_root/arm``): launch FRESH from
    ``experiments/<arm>.yaml`` with ``--run-dir-name <arm>`` so the run dir is
    created under the bare arm name (no timestamp). Otherwise ``--resume``."""
    slice_games = target - current
    base = [python_exe, "-u", "scripts/train.py", "--parallel",
            "--runs-root", runs_root, "--games", str(slice_games)]
    if has_run(runs_root, arm):
        return base + ["--resume", arm]
    return base + ["--config", arm_yaml(arm), "--run-dir-name", arm]


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
    arm: str, current: int, target: int, runs_root: Path, python_exe: str, runner=None
) -> int:
    """Advance one arm by a single slice from ``current`` up to ``target`` total
    games (slice size = target-current). Creates the run on first touch, else
    resumes. Returns the subprocess return code. ``runner`` is injectable for
    tests (default: subprocess.run from the repo root)."""
    cmd = build_slice_cmd(arm, current, target, str(runs_root), python_exe)
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
        for arm, current, target in plan:
            rc = advance_arm(arm, current, target, runs_root, python_exe, runner=runner)
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
