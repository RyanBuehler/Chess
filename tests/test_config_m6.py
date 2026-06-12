from chessrl.config.config import EvalConfig, RunConfig


def test_m6_eval_defaults():
    cfg = RunConfig()
    assert cfg.eval.every_n_checkpoints == 5
    assert cfg.eval.games_per_rung == 4
    assert cfg.eval.agent_simulations == 200
    assert cfg.eval.max_plies == 256
    assert cfg.eval.stockfish_path == ""
    assert cfg.eval.stockfish_movetime_ms == 100
    assert cfg.eval.poll_seconds == 10.0


def test_m6_eval_fields_overridable():
    e = EvalConfig(every_n_checkpoints=1, games_per_rung=2, agent_simulations=8)
    assert e.every_n_checkpoints == 1
    assert e.games_per_rung == 2
    assert e.agent_simulations == 8


def test_m6_games_per_rung_default_is_even():
    # games_per_rung must be even so both colors are played equally.
    assert RunConfig().eval.games_per_rung % 2 == 0


def test_m6_yaml_partial_override(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text(
        "eval:\n"
        "  every_n_checkpoints: 2\n"
        "  games_per_rung: 6\n"
        "  agent_simulations: 50\n"
        "  stockfish_path: tools/stockfish/stockfish.exe\n"
        "  poll_seconds: 1.0\n"
    )
    cfg = RunConfig.from_yaml(p)
    assert cfg.eval.every_n_checkpoints == 2
    assert cfg.eval.games_per_rung == 6
    assert cfg.eval.agent_simulations == 50
    assert cfg.eval.stockfish_path == "tools/stockfish/stockfish.exe"
    assert cfg.eval.poll_seconds == 1.0
    assert cfg.eval.max_plies == 256            # untouched default survives
    assert cfg.mcts.simulations == 200          # other sections untouched


def test_m6_eval_in_json_round_trip(tmp_path):
    cfg = RunConfig()
    p = tmp_path / "config.json"
    p.write_text(cfg.to_json())
    cfg2 = RunConfig.from_json(p)
    assert cfg2 == cfg
    assert cfg2.eval.games_per_rung == 4
