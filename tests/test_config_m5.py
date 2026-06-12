from chessrl.config.config import (
    MCTSConfig,
    RunConfig,
    SelfPlayConfig,
    TrainingConfig,
)


def test_m5_defaults():
    cfg = RunConfig()
    assert cfg.mcts.leaves_per_tree == 1
    assert cfg.selfplay.workers == 4
    assert cfg.selfplay.concurrent_games == 32
    assert cfg.training.checkpoint_every_steps == 1000
    assert cfg.training.selfplay_device == "cuda"


def test_m5_fields_are_overridable():
    m = MCTSConfig(leaves_per_tree=4)
    assert m.leaves_per_tree == 4
    s = SelfPlayConfig(workers=1, concurrent_games=2)
    assert s.workers == 1 and s.concurrent_games == 2
    t = TrainingConfig(checkpoint_every_steps=50, selfplay_device="cpu")
    assert t.checkpoint_every_steps == 50 and t.selfplay_device == "cpu"


def test_m5_yaml_partial_override(tmp_path):
    p = tmp_path / "exp.yaml"
    p.write_text(
        "mcts:\n  leaves_per_tree: 4\n"
        "selfplay:\n  workers: 2\n  concurrent_games: 8\n"
        "training:\n  checkpoint_every_steps: 100\n  selfplay_device: cpu\n"
    )
    cfg = RunConfig.from_yaml(p)
    assert cfg.mcts.leaves_per_tree == 4
    assert cfg.mcts.simulations == 200          # untouched default survives
    assert cfg.selfplay.workers == 2
    assert cfg.selfplay.concurrent_games == 8
    assert cfg.selfplay.ply_cap == 512          # untouched default survives
    assert cfg.training.checkpoint_every_steps == 100
    assert cfg.training.selfplay_device == "cpu"
    assert cfg.training.batch_size == 256       # untouched default survives
