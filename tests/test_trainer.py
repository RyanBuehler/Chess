import numpy as np
import torch

from chessrl.config.config import NetworkConfig, TrainingConfig
from chessrl.model.network import PolicyValueNet
from chessrl.supervised.pgn_import import record_from_pgn
from chessrl.training.buffer import ReplayBuffer
from chessrl.training.trainer import Trainer

FOOLS_MATE = '[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n'


def _setup(tmp_path):
    torch.manual_seed(0)
    net = PolicyValueNet(NetworkConfig(blocks=1, filters=16))
    cfg = TrainingConfig(batch_size=4, device="cpu", samples_per_position=2.0)
    trainer = Trainer(net, cfg, tmp_path)
    buf = ReplayBuffer(100)
    buf.add_game(record_from_pgn(FOOLS_MATE))
    return trainer, buf


def test_train_steps_returns_finite_losses(tmp_path):
    trainer, buf = _setup(tmp_path)
    m = trainer.train_steps(buf, 3, np.random.default_rng(0))
    assert trainer.step == 3
    assert np.isfinite(m["policy_loss"]) and np.isfinite(m["value_loss"])


def test_pacing_budget(tmp_path):
    trainer, buf = _setup(tmp_path)
    # 10 positions * 2.0 spp / batch 4 = 5 allowed steps total
    assert trainer.allowed_steps(total_positions=10) == 5
    trainer.train_steps(buf, 3, np.random.default_rng(0))
    assert trainer.allowed_steps(total_positions=10) == 2


def test_checkpoint_save_load(tmp_path):
    trainer, buf = _setup(tmp_path)
    trainer.train_steps(buf, 2, np.random.default_rng(0))
    trainer.save_checkpoint()
    path = tmp_path / "checkpoints" / "ckpt_00000002.pt"
    assert path.exists()

    net2 = PolicyValueNet(NetworkConfig(blocks=1, filters=16))
    trainer2 = Trainer(net2, TrainingConfig(batch_size=4, device="cpu"), tmp_path)
    trainer2.load_checkpoint(path)
    assert trainer2.step == 2
    for a, b in zip(trainer.net.parameters(), trainer2.net.parameters()):
        assert torch.equal(a, b)
