import chess
import torch

from chessrl.config.config import NetworkConfig
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.model.network import NetEvaluator, PolicyValueNet


def test_forward_shapes():
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=16))
    x = torch.zeros(4, 21, 8, 8)
    logits, value = net(x)
    assert logits.shape == (4, NUM_ACTIONS)
    assert value.shape == (4, 1)
    assert value.abs().max() <= 1.0


def test_policy_head_is_conv_not_fc():
    small = sum(p.numel() for p in PolicyValueNet(NetworkConfig(blocks=1, filters=16)).parameters())
    assert small < 100_000  # a flatten->FC policy head alone would be ~19M params


def test_net_evaluator_returns_distribution():
    net = PolicyValueNet(NetworkConfig(blocks=1, filters=16))
    ev = NetEvaluator(net, device="cpu")
    policy, value = ev.evaluate(chess.Board())
    assert policy.shape == (NUM_ACTIONS,)
    assert abs(policy.sum() - 1.0) < 1e-4
    assert -1.0 <= value <= 1.0
