"""Post-hoc Elo sweep tests (plan Task 5.4).

A tiny stubbed evaluator (monkeypatched evaluate_checkpoint) and a stubbed
goal-game runner keep this fast: assert the sweep enumerates checkpoints,
evaluates at the configured sims (200, not 50) with the configured
games_per_rung, writes per-arm Elo-vs-games, and writes goal_eval.json in the
schema the wishful-thinking thermometer reads.
"""
import json
from pathlib import Path

import pytest

import scripts.eval_sweep as es
from chessrl.config.config import RunConfig


def _make_run(tmp_path, name, goal_mode="none", steps=(500, 1000, 1500)):
    run_dir = tmp_path / name
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "games").mkdir()
    cfg = RunConfig.from_dict({"run_name": name, "goal": {"goal_mode": goal_mode}})
    (run_dir / "config.json").write_text(cfg.to_json())
    # Fake checkpoints (content irrelevant; the evaluator is stubbed).
    for s in steps:
        (run_dir / "checkpoints" / f"ckpt_{s:08d}.pt").write_bytes(b"stub")
    # metrics.jsonl maps step -> games for the Elo-vs-games axis.
    with (run_dir / "metrics.jsonl").open("w") as f:
        for i, s in enumerate(steps):
            f.write(json.dumps({"step": s, "games": (i + 1) * 1000}) + "\n")
    return run_dir


def test_sweep_enumerates_checkpoints_at_configured_sims(tmp_path, monkeypatch):
    run_dir = _make_run(tmp_path, "gp-vanilla-x")
    seen = []

    def fake_eval(rd, ckpt, cfg, store, openings_offset, agent_factory=None):
        seen.append((es._step_of(ckpt), cfg.agent_simulations, cfg.games_per_rung,
                     cfg.every_n_checkpoints))
        return 100.0 + es._step_of(ckpt) / 10.0  # deterministic fake elo

    monkeypatch.setattr(es, "evaluate_checkpoint", fake_eval)

    sweep = es.sweep_run(run_dir, sims=200, games_per_rung=20, store=object())

    # All three checkpoints evaluated, in step order, at sims=200 / gpr=20 / every=1.
    assert [s for s, _, _, _ in seen] == [500, 1000, 1500]
    assert all(sims == 200 for _, sims, _, _ in seen)
    assert all(gpr == 20 for _, _, gpr, _ in seen)
    assert all(every == 1 for _, _, _, every in seen)

    # sweep.json written with per-checkpoint Elo-vs-games.
    written = json.loads((run_dir / "sweep.json").read_text())
    assert written["run"] == "gp-vanilla-x"
    assert written["sims"] == 200 and written["games_per_rung"] == 20
    assert [p["games"] for p in written["points"]] == [1000, 2000, 3000]
    assert [p["step"] for p in written["points"]] == [500, 1000, 1500]
    assert all(p["elo"] is not None for p in written["points"])


def test_sweep_elo_jsonl_compatible_points(tmp_path, monkeypatch):
    # The points must carry (step, games, elo) so time_to_elo / compare.html
    # can plot Elo vs games per arm.
    run_dir = _make_run(tmp_path, "gp-lp-goal-x", goal_mode="lp", steps=(500,))
    monkeypatch.setattr(es, "evaluate_checkpoint",
                        lambda *a, **k: 642.0)
    sweep = es.sweep_run(run_dir, sims=200, games_per_rung=20, store=object())
    assert sweep["goal_mode"] == "lp"
    pt = sweep["points"][0]
    assert pt == {"step": 500, "games": 1000, "elo": 642.0}


def test_compute_goal_eval_writes_stockfish_rates(tmp_path):
    run_dir = _make_run(tmp_path, "gp-random-goal-x", goal_mode="random", steps=(500,))

    # Stub the (expensive) per-game play: capture achieved 3/4, check 0/4, etc.
    def runner(kind, i):
        table = {"capture": 3, "check": 0, "castle": 4, "promote": 1, "win": 2}
        return i < table[kind]

    out = es.compute_goal_eval(
        run_dir, goal_kinds=("capture", "check", "castle", "promote", "win"),
        n_games_per_kind=4, game_runner=runner,
    )
    rates = out["stockfish_rates"]
    assert rates["capture"] == 0.75
    assert rates["check"] == 0.0
    assert rates["castle"] == 1.0
    assert rates["promote"] == 0.25
    assert rates["win"] == 0.5
    assert out["n_games_per_kind"] == 4

    # Written to goal_eval.json in the schema the thermometer reads.
    on_disk = json.loads((run_dir / "goal_eval.json").read_text())
    assert on_disk["stockfish_rates"] == rates

    # The thermometer's loader actually parses it.
    from chessrl.training.parallel_loop import load_stockfish_achievement_rates
    loaded = load_stockfish_achievement_rates(run_dir)
    assert loaded == rates


def test_default_goal_runner_is_not_wired(tmp_path):
    run_dir = _make_run(tmp_path, "gp-always-win-x", goal_mode="always_win", steps=(500,))
    with pytest.raises(NotImplementedError):
        es.compute_goal_eval(run_dir, goal_kinds=("win",), n_games_per_kind=1)
