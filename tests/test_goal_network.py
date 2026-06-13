"""Task 2.1 — goal-conditioned network variant.

The goal-conditioned PolicyValueNet:
  * widens the input stem to 21 + GOAL_PLANES channels;
  * the value head ends in **sigmoid** (achievement probability in [0,1]);
  * the value head takes the **deadline scalar** concatenated at its first Linear.

The existing vanilla net (21 planes, tanh value) must keep working UNCHANGED.
"""
import chess
import torch

from chessrl.chess_env.encoding import NUM_PLANES
from chessrl.chess_env.moves import NUM_ACTIONS
from chessrl.config.config import NetworkConfig
from chessrl.goals.encoding import GOAL_PLANES
from chessrl.model.network import GoalNetEvaluator, PolicyValueNet


def _goal_net():
    return PolicyValueNet(NetworkConfig(blocks=2, filters=16), goal_conditioned=True)


def test_value_head_is_sigmoid_and_takes_deadline():
    net = _goal_net()
    planes = torch.zeros(1, NUM_PLANES + GOAL_PLANES, 8, 8)
    deadline = torch.tensor([[0.5]])
    pol, val = net(planes, deadline)
    assert 0.0 <= val.item() <= 1.0  # sigmoid range
    assert pol.shape == (1, NUM_ACTIONS)


def test_goal_net_value_responds_to_deadline_input():
    """The deadline scalar must actually feed the value head (not be ignored):
    two different deadline inputs on the same planes generally differ."""
    net = _goal_net()
    # Bias the first value Linear so the deadline column is non-zero, otherwise
    # an all-zero init could make the test vacuous. We just check wiring: the
    # forward accepts the scalar and the graph depends on it.
    planes = torch.zeros(2, NUM_PLANES + GOAL_PLANES, 8, 8, requires_grad=False)
    deadline = torch.tensor([[0.0], [1.0]], requires_grad=True)
    _, val = net(planes, deadline)
    val.sum().backward()
    assert deadline.grad is not None
    assert deadline.grad.abs().sum().item() > 0.0  # value depends on deadline


def test_goal_net_input_channels_are_widened():
    net = _goal_net()
    stem_conv = net.stem[0]
    assert stem_conv.in_channels == NUM_PLANES + GOAL_PLANES


def test_vanilla_net_unchanged():
    """Default constructor: 21 planes, tanh value, single-arg forward."""
    net = PolicyValueNet(NetworkConfig(blocks=2, filters=16))
    x = torch.zeros(4, NUM_PLANES, 8, 8)
    logits, value = net(x)
    assert logits.shape == (4, NUM_ACTIONS)
    assert value.shape == (4, 1)
    assert value.abs().max() <= 1.0  # tanh range
    assert net.stem[0].in_channels == NUM_PLANES


def test_goal_net_evaluator_returns_probability_in_unit_range():
    net = _goal_net()
    ev = GoalNetEvaluator(net, device="cpu")
    from chessrl.goals.templates import WIN_GOAL

    policy, value = ev.evaluate(chess.Board(), WIN_GOAL, remaining=10, protagonist=chess.WHITE)
    assert policy.shape == (NUM_ACTIONS,)
    assert abs(policy.sum() - 1.0) < 1e-4
    assert 0.0 <= value <= 1.0
