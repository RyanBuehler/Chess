"""Round-robin orchestrator tests (plan Task 5.1).

A FAKE trainer (a stub runner that just bumps each arm's state.json) stands in
for scripts/train.py, so the schedule logic is exercised deterministically and
fast: assert the per-arm next-target sequence (1k increments), that arms already
at budget are skipped, and that it rotates across arms.
"""
import json
from pathlib import Path

from scripts.round_robin import (
    next_target,
    plan_round,
    read_games,
    round_robin,
)


def _write_state(runs_root: Path, arm: str, games: int) -> None:
    d = runs_root / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps({"games": games, "positions": games * 40}))


def _arm_of(cmd) -> str:
    """The arm name from a slice command: ``--resume <arm>`` on a resume, or
    ``--run-dir-name <arm>`` on a fresh first-touch launch."""
    if "--resume" in cmd:
        return cmd[cmd.index("--resume") + 1]
    return cmd[cmd.index("--run-dir-name") + 1]


def _stub_runner(runs_root: Path):
    """A fake trainer: ``--games`` is the SLICE size (new games this invocation,
    NOT a cumulative target), so it ADDS that to the arm's state.json -- exactly
    as the real trainer advances ``baseline_games + games_seen``. On first touch
    the arm is launched fresh (``--config``/``--run-dir-name``); we also drop a
    ``config.json`` so the next round sees it as resumable."""
    def runner(cmd):
        arm = _arm_of(cmd)
        slice_games = int(cmd[cmd.index("--games") + 1])
        d = runs_root / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}")
        _write_state(runs_root, arm, read_games(d) + slice_games)
        return 0
    return runner


def test_read_games_missing_is_zero(tmp_path):
    assert read_games(tmp_path / "nope") == 0
    _write_state(tmp_path, "a", 1234)
    assert read_games(tmp_path / "a") == 1234


def test_next_target_increments_and_caps_at_budget():
    assert next_target(0, budget=30000, round_size=1000) == 1000
    assert next_target(1000, budget=30000, round_size=1000) == 2000
    # final short slice: 29500 -> 30000, not 30500
    assert next_target(29500, budget=30000, round_size=1000) == 30000
    # at/over budget -> None (skip)
    assert next_target(30000, budget=30000, round_size=1000) is None
    assert next_target(31000, budget=30000, round_size=1000) is None


def test_plan_round_skips_arms_at_budget(tmp_path):
    arms = ("gp-vanilla", "gp-always-win", "gp-random-goal", "gp-lp-goal")
    _write_state(tmp_path, "gp-vanilla", 30000)     # done -> skipped
    _write_state(tmp_path, "gp-always-win", 0)
    _write_state(tmp_path, "gp-random-goal", 2000)
    # gp-lp-goal: no state.json -> treated as 0
    plan = plan_round(arms, tmp_path, budget=30000, round_size=1000)
    assert plan == [
        ("gp-always-win", 0, 1000),
        ("gp-random-goal", 2000, 3000),
        ("gp-lp-goal", 0, 1000),
    ]


def test_round_robin_rotates_and_advances_1k(tmp_path):
    arms = ("gp-vanilla", "gp-always-win")
    runner = _stub_runner(tmp_path)
    launched = round_robin(
        arms=arms, runs_root=tmp_path, budget=3000, round_size=1000, runner=runner,
    )
    # Two arms, 3 rounds each -> rotation interleaves arms every round.
    assert launched == [
        ("gp-vanilla", 1000), ("gp-always-win", 1000),
        ("gp-vanilla", 2000), ("gp-always-win", 2000),
        ("gp-vanilla", 3000), ("gp-always-win", 3000),
    ]
    # Both arms land exactly on budget.
    assert read_games(tmp_path / "gp-vanilla") == 3000
    assert read_games(tmp_path / "gp-always-win") == 3000


def test_round_robin_skips_already_complete_arm(tmp_path):
    arms = ("gp-vanilla", "gp-lp-goal")
    _write_state(tmp_path, "gp-vanilla", 3000)   # already at budget
    runner = _stub_runner(tmp_path)
    launched = round_robin(
        arms=arms, runs_root=tmp_path, budget=3000, round_size=1000, runner=runner,
    )
    # Only gp-lp-goal advances; gp-vanilla never launched.
    assert [a for a, _ in launched] == ["gp-lp-goal", "gp-lp-goal", "gp-lp-goal"]
    assert launched[-1] == ("gp-lp-goal", 3000)


def test_round_robin_resumes_from_persisted_progress(tmp_path):
    # Simulate a restart: gp-vanilla is mid-flight at 1500, gp-lp-goal at 1000.
    arms = ("gp-vanilla", "gp-lp-goal")
    _write_state(tmp_path, "gp-vanilla", 1500)
    _write_state(tmp_path, "gp-lp-goal", 1000)
    runner = _stub_runner(tmp_path)
    launched = round_robin(
        arms=arms, runs_root=tmp_path, budget=3000, round_size=1000, runner=runner,
    )
    # gp-vanilla: 1500 -> 2500 -> 3000 (short final). gp-lp-goal: 1000->2000->3000.
    assert launched == [
        ("gp-vanilla", 2500), ("gp-lp-goal", 2000),
        ("gp-vanilla", 3000), ("gp-lp-goal", 3000),
    ]


def test_round_robin_stops_on_failed_slice(tmp_path):
    arms = ("gp-vanilla",)

    def failing(cmd):
        return 1  # never advances state.json

    launched = round_robin(
        arms=arms, runs_root=tmp_path, budget=3000, round_size=1000,
        runner=failing, max_rounds=10,
    )
    # One launch, non-zero rc -> stop immediately (no infinite loop).
    assert launched == [("gp-vanilla", 1000)]
