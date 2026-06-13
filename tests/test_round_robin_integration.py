"""Non-stub round-robin integration test (launch-path bug guard).

Drives the REAL trainer (scripts/train.py via the default subprocess runner)
through TWO slices of round_size on ONE vanilla arm, with a tiny net so it runs
on CPU in seconds. This is the test that the old state.json-bumping stub masked:
it exercises ``parallel_loop.main``'s actual ``--games`` (slice) and
``--run-dir-name`` (create-on-first-touch) handling.

It proves the two launch-path fixes:
  1. ``--games`` is the SLICE size (target-current), so each slice adds ~round_size
     NEW games -- NOT the cumulative target (which over-runs exponentially:
     1000, then 2000, then 3000...). After slice 1 games ~= round_size; after
     slice 2 games ~= 2*round_size.
  2. The run dir is created under the BARE arm name (no timestamp) on first
     touch, then resumed -- so the orchestrator can both create and resume arms
     by their bare names, and relaunch re-derives progress from state.json.

Marked slow: it spawns a real worker process (Windows spawn + CPU).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import round_robin

REPO_ROOT = Path(round_robin.__file__).resolve().parent.parent

# Tiny everything: 1 block / 8 filters, a couple sims, 1 worker, small batches of
# concurrent games, checkpoint immediately. Vanilla (goal_mode none) so there is
# no goal/curriculum overhead. concurrent_games=2 so a slice of round_size=4
# completes in ~2 batches.
TINY_YAML = """\
run_name: rr-int
network: {blocks: 1, filters: 8}
mcts: {simulations: 4, temperature_moves: 4, leaves_per_tree: 2}
selfplay: {ply_cap: 24, workers: 1, concurrent_games: 2, resign_playout_fraction: 0.0, feed_port: 0}
training: {batch_size: 16, buffer_size: 1000, samples_per_position: 2.0, checkpoint_every_steps: 1, device: cpu, selfplay_device: cpu}
goal: {goal_mode: none}
"""

ROUND_SIZE = 4
# Budget far above any single-slice overshoot, so BOTH slices always launch
# regardless of how many in-flight games the worker drains at stop. The launch-
# path bug is proven by the captured --games value (slice vs cumulative target)
# and the create-then-resume flags, which are independent of the exact overshoot.
BUDGET = 10_000
ARM = "rr-int"


@pytest.mark.slow
def test_round_robin_real_trainer_two_slices(tmp_path, monkeypatch):
    # Point the arm->yaml mapping at a tmp config so we don't touch experiments/.
    cfg_path = tmp_path / f"{ARM}.yaml"
    cfg_path.write_text(TINY_YAML)
    monkeypatch.setattr(round_robin, "arm_yaml", lambda arm: str(cfg_path))

    runs_root = tmp_path / "runs"
    run_dir = runs_root / ARM

    # A runner that RECORDS the launched command (so we can assert the load-bearing
    # --games slice value and create-vs-resume flags) and then invokes the REAL
    # trainer (scripts/train.py) from the repo root -- not a state-bumping stub.
    seen_cmds: list[list[str]] = []

    def recording_runner(cmd):
        seen_cmds.append(list(cmd))
        return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode

    def games_of(cmd):
        return int(cmd[cmd.index("--games") + 1])

    # --- Slice 1: first touch must CREATE runs/<arm> fresh (no timestamp). ---
    launched1 = round_robin.round_robin(
        arms=(ARM,),
        runs_root=str(runs_root),
        budget=BUDGET,
        round_size=ROUND_SIZE,
        python_exe=sys.executable,
        runner=recording_runner,
        max_rounds=1,
    )
    assert launched1 == [(ARM, ROUND_SIZE)]  # cumulative target recorded (0+4)

    # (1) Created under the BARE arm name via --config/--run-dir-name, no timestamp.
    assert "--config" in seen_cmds[0] and "--run-dir-name" in seen_cmds[0]
    assert seen_cmds[0][seen_cmds[0].index("--run-dir-name") + 1] == ARM
    assert run_dir.is_dir(), "run dir not created under bare arm name"
    assert (run_dir / "config.json").exists()
    siblings = [p.name for p in runs_root.iterdir() if p.is_dir()]
    assert siblings == [ARM], f"unexpected (timestamped?) run dirs: {siblings}"

    # BUG 1 proof: --games passed to the trainer is the SLICE size (target-current
    # = 4-0), NOT the cumulative target. With the bug it would also be 4 here, so
    # the discriminator is slice 2 below.
    assert games_of(seen_cmds[0]) == ROUND_SIZE

    games_after_1 = json.loads((run_dir / "state.json").read_text())["games"]
    # Trainer overshoots a slice by up to ~one worker batch (it checks the budget
    # only between batches and drains in-flight games on stop), so assert "made
    # progress", not an exact count.
    assert games_after_1 >= ROUND_SIZE, f"slice 1 games {games_after_1} < {ROUND_SIZE}"

    # --- Slice 2: relaunch re-derives progress from state.json, then RESUMES. ---
    launched2 = round_robin.round_robin(
        arms=(ARM,),
        runs_root=str(runs_root),
        budget=BUDGET,
        round_size=ROUND_SIZE,
        python_exe=sys.executable,
        runner=recording_runner,
        max_rounds=1,
    )
    # (3) Relaunch re-derived current from state.json => the next CUMULATIVE target
    # is games_after_1 + round_size (progress logic still uses the cumulative
    # target -- this is what proves progress is re-derived from state.json, not
    # restarted from 0).
    assert launched2 == [(ARM, games_after_1 + ROUND_SIZE)]

    # (2) Second touch RESUMES the bare arm name (not a fresh create).
    assert "--resume" in seen_cmds[1] and seen_cmds[1][seen_cmds[1].index("--resume") + 1] == ARM
    assert "--config" not in seen_cmds[1]

    # BUG 1 proof (the discriminator): the trainer's NEW-games budget for slice 2
    # is the SLICE size = target-current = round_size (4) -- NOT the cumulative
    # target (games_after_1 + round_size). With the bug, --games would be the
    # cumulative target, re-running the whole accumulated budget every slice
    # (1000, then ~2000, ...). Here we assert it is exactly the slice.
    slice2_games_arg = games_of(seen_cmds[1])
    assert slice2_games_arg == ROUND_SIZE, (
        f"slice 2 --games {slice2_games_arg} != round_size {ROUND_SIZE}: the trainer "
        f"was told to run a CUMULATIVE budget ({games_after_1 + ROUND_SIZE}), not a "
        f"slice (BUG 1 not fixed)"
    )

    games_after_2 = json.loads((run_dir / "state.json").read_text())["games"]
    # Each slice ADDS new games on top of the persisted baseline (resume keeps the
    # prior count): total strictly grew, never reset or restarted to 0.
    assert games_after_2 > games_after_1, "slice 2 did not add games on top of slice 1"


@pytest.mark.slow
def test_run_dir_name_errors_if_exists(tmp_path):
    """--run-dir-name must refuse to clobber an existing dir (so a stray fresh
    launch never silently resumes on top of an arm)."""
    from chessrl.training.parallel_loop import main

    cfg_path = tmp_path / "rr-int.yaml"
    cfg_path.write_text(TINY_YAML)
    runs_root = tmp_path / "runs"
    (runs_root / ARM).mkdir(parents=True)  # pre-existing

    with pytest.raises(SystemExit):
        main(["--config", str(cfg_path), "--runs-root", str(runs_root),
              "--run-dir-name", ARM, "--games", "1"])
