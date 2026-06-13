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


def test_default_goal_runner_requires_checkpoint_and_stockfish(tmp_path, monkeypatch):
    # The production runner is now wired (no longer NotImplementedError). With no
    # Stockfish binary provisioned it raises FileNotFoundError so the caller can
    # skip rather than silently emit floor rates.
    run_dir = _make_run(tmp_path, "gp-always-win-x", goal_mode="always_win", steps=(500,))
    monkeypatch.setattr(
        "chessrl.evaluation.players.default_stockfish_path", lambda: None
    )
    with pytest.raises(FileNotFoundError):
        es.compute_goal_eval(run_dir, goal_kinds=("win",), n_games_per_kind=1)


# ---- Real runner with injected stub agent + opponent (no Stockfish, fast) ----

def _tiny_goal_agent(tmp_path):
    """A real _GoalSearchAgent over a tiny goal-conditioned net (no Stockfish)."""
    import torch

    from chessrl.config.config import NetworkConfig
    from chessrl.model.network import PolicyValueNet

    cfg = NetworkConfig(blocks=1, filters=8)
    net = PolicyValueNet(cfg, goal_conditioned=True)
    ckpt = tmp_path / "goal_ckpt.pt"
    torch.save({"model": net.state_dict()}, ckpt)
    return es._GoalSearchAgent(ckpt, cfg, simulations=4, device="cpu", seed=0)


def test_play_goal_eval_game_returns_bool_and_respects_ply_cap(tmp_path):
    from chessrl.evaluation.players import RandomPlayer
    from chessrl.goals.templates import GoalTemplate

    agent = _tiny_goal_agent(tmp_path)
    opponent = RandomPlayer(seed=1)
    # A check-goal with a short deadline; outcome is a clean True/False either way.
    goal = GoalTemplate.check(deadline=8)
    achieved = es.play_goal_eval_game(
        agent, opponent, goal, agent_color=__import__("chess").WHITE, max_plies=12
    )
    assert isinstance(achieved, bool)


def test_compute_goal_eval_with_injected_real_game_runner(tmp_path):
    """End-to-end: a real goal-net agent vs an injected stub opponent (Random)
    plays the games, produces per-kind rates in [0,1], and writes goal_eval.json
    in the schema the thermometer consumes to a non-None gap."""
    import chess

    from chessrl.evaluation.players import GreedyMaterialPlayer
    from chessrl.training.loop import (
        goal_achievement_rates,
        wishful_thinking_thermometer,
    )
    from chessrl.training.parallel_loop import load_stockfish_achievement_rates

    run_dir = _make_run(tmp_path, "gp-random-goal-y", goal_mode="random", steps=(500,))
    agent = _tiny_goal_agent(tmp_path)
    opponent = GreedyMaterialPlayer(seed=2)
    kinds = ("capture", "check", "castle", "promote", "win")

    def runner(kind, i):
        goal = es._goal_for_kind(kind, deadline_max=20)
        agent_color = chess.WHITE if i % 2 == 0 else chess.BLACK
        return es.play_goal_eval_game(
            agent, opponent, goal, agent_color=agent_color, max_plies=24
        )

    out = es.compute_goal_eval(
        run_dir, goal_kinds=kinds, n_games_per_kind=2, game_runner=runner
    )
    rates = out["stockfish_rates"]
    assert set(rates) == set(kinds)
    assert all(0.0 <= r <= 1.0 for r in rates.values())
    assert out["n_games_per_kind"] == 2

    # The thermometer's loader parses it, and the kind keys align with the
    # self-play rate keys, so the gap is populated (non-None) for shared kinds.
    loaded = load_stockfish_achievement_rates(run_dir)
    assert loaded == rates
    # Synthesize self-play rates over the SAME kinds and confirm a real gap.
    sp_rates = {k: 1.0 for k in kinds}
    thermo = wishful_thinking_thermometer(sp_rates, loaded)
    assert all(thermo[k]["gap"] is not None for k in kinds)
    assert all(thermo[k]["gap"] == 1.0 - rates[k] for k in kinds)


def test_goal_for_kind_matches_self_play_assigner_kinds():
    # Each measured kind maps to a concrete template OF THAT KIND, so the
    # vs-Stockfish rate keys line up with goal_achievement_rates' keys.
    for kind in ("capture", "check", "castle", "promote", "win"):
        tmpl = es._goal_for_kind(kind, deadline_max=20)
        assert tmpl.kind == kind


@pytest.mark.skipif(
    __import__("chessrl.evaluation.players", fromlist=["default_stockfish_path"]).default_stockfish_path() is None,
    reason="stockfish binary not provisioned",
)
def test_default_runner_smoke_vs_real_stockfish(tmp_path):
    """One real game vs Stockfish per the production default runner (gated)."""
    import torch

    from chessrl.config.config import NetworkConfig
    from chessrl.model.network import PolicyValueNet

    # Build a run with a real tiny goal-conditioned checkpoint and a fast eval cfg.
    cfg = RunConfig.from_dict({
        "run_name": "gp-sf-smoke",
        "goal": {"goal_mode": "random"},
        "network": {"blocks": 1, "filters": 8},
        "eval": {"agent_simulations": 4, "max_plies": 12, "stockfish_movetime_ms": 20},
    })
    run_dir = tmp_path / "gp-sf-smoke"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "games").mkdir()
    (run_dir / "config.json").write_text(cfg.to_json())
    net = PolicyValueNet(cfg.network, goal_conditioned=True)
    torch.save({"model": net.state_dict()}, run_dir / "checkpoints" / "ckpt_00000500.pt")

    out = es.compute_goal_eval(run_dir, goal_kinds=("check",), n_games_per_kind=1)
    assert 0.0 <= out["stockfish_rates"]["check"] <= 1.0
