"""M2 gate: the network can memorize 3 miniature games (proves the
encode -> record -> buffer -> loss -> optimize plumbing end to end)."""
import numpy as np
import torch

from chessrl.config.config import NetworkConfig, TrainingConfig
from chessrl.model.network import PolicyValueNet
from chessrl.supervised.pgn_import import record_from_pgn
from chessrl.training.buffer import ReplayBuffer
from chessrl.training.trainer import Trainer

PGNS = [
    '[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1\n',
    '[Result "1-0"]\n\n1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n',
    '[Result "1-0"]\n\n1. e4 e5 2. Nf3 d6 3. Bc4 Bg4 4. Nc3 g6 '
    "5. Nxe5 Bxd1 6. Bxf7+ Ke7 7. Nd5# 1-0\n",
]


def test_overfit_three_games(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=32))
    trainer = Trainer(net, TrainingConfig(batch_size=16, learning_rate=1e-3,
                                          weight_decay=0.0, device="cpu"), tmp_path)
    buf = ReplayBuffer(1000)
    for pgn in PGNS:
        buf.add_game(record_from_pgn(pgn))

    last = {"policy_loss": float("inf"), "value_loss": float("inf")}
    for _ in range(40):                       # up to 800 steps
        last = trainer.train_steps(buf, 20, rng)
        if last["policy_loss"] + last["value_loss"] < 0.4:
            break
    total = last["policy_loss"] + last["value_loss"]
    assert total < 0.4, f"failed to memorize 3 games: final loss {total:.3f}"
